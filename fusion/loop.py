"""Backend- and domain-agnostic 8-step ARES + Fusion + Decision Probe investigation loop.

Orchestrates (ARES Plan §5.2):
1. Seed: Domain verifiers, optional ExecutionBackend, NoteStore, TaskGraph, version-0 board
2. Propose (Parallel, Bounded): `gemini-4-hydrogen` swarm workers via `ARESRunner.solve`
3. Verify (Deterministic, Cheapest-First): guarded magnet queries + TRUSTED verifiers + slices
4. Fuse: multimodal `gemini-4-carbon` synthesis + deterministic canonical verifier grounding
   validated through `validate_and_merge_patch` (Invariants I1–I7)
5. Decide & 6. Probe: `gemini-decision-1` closed-set evaluation, single-field `"unknown"`
   ablations, reliance shares, and counterfactual VoI scoring
7. Acquire: targeted follow-up task generation over highest-VoI unknown/uncertified fields
8. Terminate & Final Audit: certified success vs. honest uncertified / budget stop, plus
   deterministic NoteStore rebuild drift and zero-violation board audit
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ares.engine import ARESRunner, RunReport, SolveEvent, TaskGraph, TaskSpec
from ares.store import NoteStore
from contracts import (
    ArtifactRef,
    DecisionQuestion,
    Domain,
    ExecutionBackend,
    HypothesisSpec,
    InMemoryVerifierRegistry,
    InvestigationBoard,
    Note,
)
from fusion.patch import (
    BoardPatch,
    FieldPatch,
    NoteExtractorFn,
    compute_board_drift,
    compute_weakest_status,
    create_initial_board,
    deterministic_patch_from_notes,
    rebuild_board_from_notes,
    validate_and_merge_patch,
    value_is_grounded_in_notes,
)
from models.decision_probe import DecisionProbeClient
from models.gemini_provider import GeminiModelProvider, ModelTelemetry

TerminationReason = Literal[
    "certified_success",
    "honest_uncertified",
    "max_rounds",
    "timeout",
]

_ABSTAIN_ANSWERS: frozenset[str] = frozenset(
    {"unknown", "none_verified", "insufficient_evidence", "abstain", "none", ""}
)


class LoopConfig(BaseModel):
    """Configuration for the 8-step ARES investigation loop."""

    max_rounds: int = Field(default=3, ge=1)
    max_concurrent_workers: int = Field(default=4, ge=1)
    wall_clock_budget_s: float = Field(default=180.0, gt=0.0)
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_uncertified_share: float = Field(default=0.15, ge=0.0, le=1.0)
    max_unknown_voi: float = Field(default=0.20, ge=0.0, le=1.0)
    include_images_in_fusion: bool = True
    artifact_dir: str | None = None


class RoundTelemetry(BaseModel):
    """Per-round execution, fusion, and counterfactual probe telemetry."""

    round_index: int
    board_version: int
    proposed_hypotheses: int
    executed_tasks: int
    skipped_duplicates: int
    rejected_guard: int
    new_note_ids: list[str] = Field(default_factory=list)
    certified_notes: int = 0
    uncertified_notes: int = 0
    artifact_paths: list[str] = Field(default_factory=list)
    applied_fields: list[str] = Field(default_factory=list)
    rejected_patch_fields: dict[str, str] = Field(default_factory=dict)
    decision_answer: str = "unknown"
    decisions_summary: dict[str, str] = Field(default_factory=dict)
    sufficiently_verified: str = "no"
    decision_confidence: float = 0.0
    certified_influence_share: float = 0.0
    uncertified_influence_share: float = 0.0
    unknown_voi_max: float = 0.0
    field_influences: dict[str, float] = Field(default_factory=dict)
    field_voi: dict[str, float] = Field(default_factory=dict)
    top_voi_fields: list[str] = Field(default_factory=list)
    board_drift: float = 0.0
    duration_ms: float = 0.0


class LoopRunResult(BaseModel):
    """Complete end-of-run investigation board, audit metrics, and telemetry."""

    board: InvestigationBoard
    rebuilt_board: InvestigationBoard
    final_board_drift: float
    rounds: list[RoundTelemetry] = Field(default_factory=list)
    notes: list[Note] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    termination_reason: TerminationReason
    blocking_fields: list[str] = Field(default_factory=list)
    invariant_violations_caught: int = 0
    ungrounded_fields_on_board: int = 0
    illegal_cert_upgrades_on_board: int = 0
    queries_executed: int = 0
    verifiers_executed: int = 0
    skipped_duplicates: int = 0
    rejected_guard: int = 0
    candidates_pruned_by_verifiers: int = 0
    model_telemetry: ModelTelemetry = Field(default_factory=ModelTelemetry)
    probe_api_calls: int = 0
    wall_clock_ms: float = 0.0


def _collect_verifier_png_artifacts(note_store: NoteStore) -> list[ArtifactRef]:
    seen: set[str] = set()
    out: list[ArtifactRef] = []
    for note in note_store.snapshot():
        for art in note.artifacts:
            if art.artifact_id in seen or not art.produced_by_verifier:
                continue
            is_png = art.kind in ("plot_png", "image") or art.mime_type == "image/png"
            if is_png and Path(art.path).is_file():
                seen.add(art.artifact_id)
                out.append(art)
    return out


def _collect_all_artifacts(note_store: NoteStore) -> list[ArtifactRef]:
    seen: set[str] = set()
    out: list[ArtifactRef] = []
    for note in note_store.snapshot():
        for art in note.artifacts:
            if art.artifact_id not in seen:
                seen.add(art.artifact_id)
                out.append(art)
    return out


def _rank_voi_fields(
    board: InvestigationBoard,
    protected_fields: Collection[str],
    limit: int = 4,
) -> list[str]:
    protected_set = frozenset(protected_fields)
    candidates = [
        (fname, fld.voi)
        for fname, fld in board.fields.items()
        if fld.status in ("unknown", "uncertified")
    ]
    candidates.sort(
        key=lambda item: (
            -item[1],
            0 if item[0] in protected_set else 1,
            item[0],
        )
    )
    return [fname for fname, _ in candidates[:limit]]


def _compute_blocking_fields(
    board: InvestigationBoard,
    termination_reason: TerminationReason,
    pivotal_fields: Sequence[str],
) -> list[str]:
    if termination_reason == "certified_success":
        return []
    blocking: list[str] = []
    for req in pivotal_fields:
        fld = board.fields.get(req)
        if fld is None or fld.status != "certified" or fld.value in ("unknown", "none"):
            blocking.append(req)
    extra = [
        (fname, fld.voi)
        for fname, fld in board.fields.items()
        if fname not in blocking
        and (fld.status != "certified" or fld.value in ("unknown", "none"))
    ]
    extra.sort(key=lambda item: (-item[1], item[0]))
    blocking.extend(fname for fname, _ in extra)
    return blocking


def _compute_candidates_pruned(
    board: InvestigationBoard,
    primary_options: Sequence[str],
    required_signal_fields: Sequence[str],
) -> int:
    candidate_pool = [opt for opt in primary_options if opt not in _ABSTAIN_ANSWERS]
    if not candidate_pool:
        return 0
    candidate_set = set(candidate_pool)

    surviving_sets: list[set[str]] = []
    target_fields = list(required_signal_fields) if required_signal_fields else list(board.fields.keys())
    for fname in target_fields:
        fld = board.fields.get(fname)
        if fld is None or fld.status == "unknown":
            continue
        if isinstance(fld.value, str):
            matched = {fld.value} & candidate_set
            surviving_sets.append(matched)
        elif isinstance(fld.value, list):
            matched = {str(v) for v in fld.value} & candidate_set
            surviving_sets.append(matched)

    if not surviving_sets:
        return 0
    intersection = set.intersection(*surviving_sets)
    return max(0, len(candidate_pool) - len(intersection))


def _audit_final_board_invariants(
    board: InvestigationBoard,
    note_store: NoteStore,
) -> tuple[int, int]:
    """Deterministically verify I1 groundedness and I2 certification monotonicity on board."""
    ungrounded = 0
    illegal_upgrades = 0

    for fname, fld in board.fields.items():
        is_untouched_unknown = (
            fld.status == "unknown"
            and fld.value == "unknown"
            and len(fld.sources) == 0
        )
        if is_untouched_unknown:
            continue

        if not fld.sources or any(not note_store.exists(nid) for nid in fld.sources):
            ungrounded += 1
            if fld.status == "certified":
                illegal_upgrades += 1
            continue

        cited_notes = [note_store.get(nid) for nid in fld.sources]
        expected_status = compute_weakest_status(
            fld.sources, note_store, field_name=fname
        )
        if fld.status == "certified" and expected_status != "certified":
            illegal_upgrades += 1

        if fld.status == "certified" and not value_is_grounded_in_notes(
            fld.value, cited_notes
        ):
            ungrounded += 1

    return ungrounded, illegal_upgrades


def _merge_canonical_and_carbon_patches(
    board: InvestigationBoard,
    new_notes: list[Note],
    carbon_patch: dict[str, Any] | BoardPatch,
    note_store: NoteStore,
    *,
    protected_fields: Collection[str],
    note_extractor: NoteExtractorFn | None = None,
    max_extra_fields: int = 4,
) -> BoardPatch:
    """Combine deterministic verifier-extracted patches with Carbon's extra field patches."""
    verifier_notes = [n for n in new_notes if n.verifier_name is not None]
    det_verifier_patch = deterministic_patch_from_notes(
        verifier_notes, note_extractor=note_extractor
    )

    canonical_keys = set(det_verifier_patch.updates.keys()) | set(protected_fields)
    merged_updates: dict[str, Any] = {}
    protected_certified_values: list[Any] = []
    protected_certified_sources: set[str] = set()

    for cname, fp in det_verifier_patch.updates.items():
        merged_updates[cname] = fp
        if all(
            note_store.exists(s) and note_store.get(s).certification == "CERTIFIED"
            for s in fp.sources
        ):
            protected_certified_values.append(fp.value)
            protected_certified_sources.update(fp.sources)

    for cname in canonical_keys:
        if (
            cname not in merged_updates
            and cname in board.fields
            and board.fields[cname].status == "certified"
        ):
            protected_certified_values.append(board.fields[cname].value)
            protected_certified_sources.update(board.fields[cname].sources)

    raw_carbon_updates: dict[str, Any] = {}
    rationale = ""
    if isinstance(carbon_patch, BoardPatch):
        raw_carbon_updates = dict(carbon_patch.updates)
        rationale = carbon_patch.rationale
    elif isinstance(carbon_patch, dict):
        if isinstance(carbon_patch.get("updates"), dict):
            raw_carbon_updates = dict(carbon_patch["updates"])
        elif isinstance(carbon_patch.get("fields"), dict):
            raw_carbon_updates = dict(carbon_patch["fields"])
        rationale = str(carbon_patch.get("rationale", ""))

    added_extra = 0
    for fname, raw_fp in raw_carbon_updates.items():
        if fname in merged_updates:
            continue
        if fname not in canonical_keys and added_extra >= max_extra_fields:
            break

        fp_sources: list[str] = []
        fp_val: Any = None
        if isinstance(raw_fp, FieldPatch):
            fp_sources = list(raw_fp.sources)
            fp_val = raw_fp.value
        elif isinstance(raw_fp, dict):
            raw_srcs = raw_fp.get("sources")
            if isinstance(raw_srcs, list):
                fp_sources = [str(s) for s in raw_srcs]
            fp_val = raw_fp.get("value")

        has_uncertified_src = any(
            note_store.exists(s) and note_store.get(s).certification != "CERTIFIED"
            for s in fp_sources
        )
        if has_uncertified_src:
            collides_value = any(fp_val == pval for pval in protected_certified_values)
            shares_protected_src = bool(set(fp_sources) & protected_certified_sources)
            if collides_value or shares_protected_src:
                continue

        merged_updates[fname] = raw_fp
        if fname not in canonical_keys:
            added_extra += 1

    return BoardPatch(
        updates=merged_updates,
        rationale=rationale or det_verifier_patch.rationale,
    )


def _build_execution_tasks_from_specs(
    *,
    round_index: int,
    board_version: int,
    specs: list[HypothesisSpec],
    unexecuted_verifiers: list[str],
    artifact_dir: str,
    has_query_backend: bool,
) -> list[TaskSpec]:
    """Build ARES `query` + deterministic `verify` TaskSpecs for a round."""
    exec_tasks: list[TaskSpec] = []
    scheduled_verifiers: set[str] = set()

    for v_idx, vname in enumerate(unexecuted_verifiers):
        scheduled_verifiers.add(vname)
        exec_tasks.append(
            TaskSpec(
                task_id=f"r{round_index}_verify_{v_idx}_{vname}",
                task_type="verify",
                requested_under=board_version,
                priority=0,
                verifier_name=vname,
                verifier_params={"artifact_dir": artifact_dir},
            )
        )

    for q_idx, hspec in enumerate(specs):
        if hspec.verifier_name and hspec.verifier_name not in scheduled_verifiers:
            scheduled_verifiers.add(hspec.verifier_name)
            exec_tasks.append(
                TaskSpec(
                    task_id=f"r{round_index}_verify_bound_{q_idx}_{hspec.verifier_name}",
                    task_type="verify",
                    requested_under=board_version,
                    priority=0,
                    verifier_name=hspec.verifier_name,
                    verifier_params={**hspec.params, "artifact_dir": artifact_dir},
                )
            )
        if has_query_backend and hspec.query and hspec.query.strip():
            query_only_spec = hspec.model_copy(
                update={
                    "verifier_name": None,
                    "requested_under": board_version,
                }
            )
            exec_tasks.append(
                TaskSpec(
                    task_id=f"r{round_index}_query_{q_idx}",
                    task_type="query",
                    requested_under=board_version,
                    priority=1,
                    hypothesis_spec=query_only_spec,
                )
            )

    return exec_tasks


def _telemetry_from_ares_report(report: RunReport) -> ModelTelemetry:
    if report.usage.model_calls == 0:
        return ModelTelemetry()
    prompt_t = report.usage.total_prompt_tokens
    comp_t = report.usage.total_completion_tokens
    return ModelTelemetry(
        prompt_tokens=prompt_t,
        completion_tokens=comp_t,
        thought_tokens=0,
        total_tokens=prompt_t + comp_t,
        model_calls=report.usage.model_calls,
        latency_ms=report.usage.wall_clock_ms,
    )


def _find_verification_gate_answer(
    board: InvestigationBoard,
    questions: Sequence[DecisionQuestion],
) -> str:
    for q in questions:
        if q.question_type == "yes_no" and (
            q.question_id == "sufficiently_verified" or "verified" in q.question.lower()
        ):
            dec = board.decisions.get(q.question_id)
            return dec.answer if dec is not None else "no"
    return "yes"


def run_investigation_loop(
    domain: Domain,
    workspace_dir: str,
    *,
    backend: ExecutionBackend | None = None,
    provider: GeminiModelProvider | None = None,
    probe_client: DecisionProbeClient | None = None,
    config: LoopConfig | None = None,
    on_round: Callable[[RoundTelemetry, InvestigationBoard], None] | None = None,
    on_ares_event: Callable[[SolveEvent], None] | None = None,
) -> LoopRunResult:
    """Execute the complete 8-step ARES + Fusion + Decision Probe investigation loop."""
    t_loop_start = time.perf_counter()
    cfg = config or LoopConfig()
    eff_provider = provider or GeminiModelProvider()
    eff_probe = probe_client or DecisionProbeClient()

    resolved_workspace = str(Path(workspace_dir).resolve())
    eff_artifact_dir = (
        str(Path(cfg.artifact_dir).resolve())
        if cfg.artifact_dir
        else str(Path(resolved_workspace) / "_artifacts")
    )
    Path(eff_artifact_dir).mkdir(parents=True, exist_ok=True)

    # 1. Seed
    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    available_verifiers = registry.list_verifiers()
    trusted_verifiers = [
        v for v in available_verifiers if registry.metadata(v).get("trusted", True)
    ]

    env_desc = domain.environment_description()
    magnet_rules = domain.magnet_rules()
    decision_questions = domain.decision_questions()
    primary_options = decision_questions[0].options if decision_questions else []
    pivotal_fields = list(domain.pivotal_fields())
    required_signals = tuple(domain.required_signal_fields())
    protected_fields = frozenset(pivotal_fields) | frozenset(required_signals)
    note_extractor = domain.extract_field_patches_from_note

    note_store = NoteStore()
    eff_backend = backend if backend is not None else domain.create_backend(resolved_workspace)
    propose_proc = eff_provider.make_propose_processor(
        environment_description=env_desc,
        magnet_rules=magnet_rules,
        pivotal_fields=pivotal_fields,
        available_verifiers=available_verifiers,
    )
    graph = TaskGraph()
    runner = ARESRunner(
        backend=eff_backend,
        registry=registry,
        note_store=note_store,
        processors={"propose": propose_proc},
    )
    board = create_initial_board(decision_questions, seed_fields=pivotal_fields)

    rounds: list[RoundTelemetry] = []
    total_model_telemetry = ModelTelemetry()
    total_probe_calls = 0
    total_queries_executed = 0
    total_verifiers_executed = 0
    total_skipped_duplicates = 0
    total_rejected_guard = 0
    invariant_violations_caught = 0
    termination_reason: TerminationReason = "max_rounds"

    for round_idx in range(1, cfg.max_rounds + 1):
        if (time.perf_counter() - t_loop_start) >= cfg.wall_clock_budget_s:
            termination_reason = "timeout"
            break

        t_round_start = time.perf_counter()

        # 2. Propose / Acquire + 3. Verify
        executed_verifier_names = {
            n.verifier_name for n in note_store.snapshot() if n.verifier_name
        }
        unexecuted_verifiers = [
            v for v in available_verifiers if v not in executed_verifier_names
        ]

        if round_idx == 1:
            families = domain.hypothesis_families()[: cfg.max_concurrent_workers]
            propose_tasks = [
                TaskSpec(
                    task_id=f"r1_propose_{idx}_{fam}",
                    task_type="propose",
                    requested_under=board.board_version,
                    context={
                        "hypothesis_family": fam,
                        "max_queries": 2,
                        "thinking_level": "low",
                        "environment_description": env_desc,
                        "magnet_rules": magnet_rules,
                        "pivotal_fields": pivotal_fields,
                        "available_verifiers": available_verifiers,
                        "board": board,
                    },
                )
                for idx, fam in enumerate(families)
            ]
            prop_report = runner.solve(
                graph=graph,
                tasks=propose_tasks,
                max_concurrent=cfg.max_concurrent_workers,
                on_event=on_ares_event,
            )
            total_model_telemetry = total_model_telemetry.plus(
                _telemetry_from_ares_report(prop_report)
            )

            proposed_specs: list[HypothesisSpec] = []
            for ptask in propose_tasks:
                node_out = graph.nodes[ptask.task_id].output
                if isinstance(node_out, list):
                    for item in node_out:
                        if isinstance(item, HypothesisSpec):
                            proposed_specs.append(item)
        else:
            high_voi_targets = _rank_voi_fields(board, protected_fields, limit=4)
            proposed_specs, followup_tel = eff_provider.propose_followup_tasks(
                board=board,
                high_voi_fields=high_voi_targets,
                environment_description=env_desc,
                magnet_rules=magnet_rules,
                available_verifiers=unexecuted_verifiers or available_verifiers,
                max_queries=3,
                thinking_level="low",
            )
            total_model_telemetry = total_model_telemetry.plus(followup_tel)

        exec_tasks = _build_execution_tasks_from_specs(
            round_index=round_idx,
            board_version=board.board_version,
            specs=proposed_specs,
            unexecuted_verifiers=unexecuted_verifiers,
            artifact_dir=eff_artifact_dir,
            has_query_backend=eff_backend is not None,
        )

        if not exec_tasks:
            termination_reason = "honest_uncertified"
            break

        exec_report = runner.solve(
            graph=graph,
            tasks=exec_tasks,
            max_concurrent=cfg.max_concurrent_workers,
            on_event=on_ares_event,
        )
        total_queries_executed += exec_report.usage.queries_executed
        total_verifiers_executed += exec_report.usage.verifiers_executed
        total_skipped_duplicates += exec_report.usage.skipped_duplicate_tasks
        total_rejected_guard += exec_report.usage.rejected_guard_tasks

        new_notes = exec_report.new_notes
        if round_idx >= 2 and len(new_notes) == 0:
            termination_reason = "honest_uncertified"
            break

        # 4. Fuse
        carbon_patch, carbon_tel = eff_provider.propose_board_patch(
            board=board,
            new_notes=new_notes,
            environment_description=env_desc,
            pivotal_fields=pivotal_fields,
            include_images=cfg.include_images_in_fusion,
            thinking_level="low",
        )
        total_model_telemetry = total_model_telemetry.plus(carbon_tel)

        merged_patch = _merge_canonical_and_carbon_patches(
            board=board,
            new_notes=new_notes,
            carbon_patch=carbon_patch,
            note_store=note_store,
            protected_fields=protected_fields,
            note_extractor=note_extractor,
        )
        board, patch_report = validate_and_merge_patch(
            board,
            merged_patch,
            note_store,
            protected_fields=protected_fields,
            seed_fields=pivotal_fields,
            note_extractor=note_extractor,
            strict=False,
            advance_round=True,
            verify_artifact_files=True,
            compute_drift=True,
        )
        invariant_violations_caught += len(patch_report.rejected_fields)

        # 5. Decide & 6. Probe
        png_artifacts = _collect_verifier_png_artifacts(note_store)
        probe_res = eff_probe.probe_board(
            board,
            decision_questions,
            required_signal_fields=required_signals,
            artifact_refs=png_artifacts,
        )
        board = probe_res.updated_board
        total_probe_calls += probe_res.api_calls
        total_model_telemetry = total_model_telemetry.plus(probe_res.telemetry)

        top_voi_fields = _rank_voi_fields(board, protected_fields, limit=4)
        suff_ver = _find_verification_gate_answer(board, decision_questions)
        decisions_summary = {qid: dec.answer for qid, dec in board.decisions.items()}

        round_artifacts = [art.path for n in new_notes for art in n.artifacts]
        round_tel = RoundTelemetry(
            round_index=round_idx,
            board_version=board.board_version,
            proposed_hypotheses=len(proposed_specs),
            executed_tasks=exec_report.usage.completed_tasks,
            skipped_duplicates=exec_report.usage.skipped_duplicate_tasks,
            rejected_guard=exec_report.usage.rejected_guard_tasks,
            new_note_ids=[n.note_id for n in new_notes],
            certified_notes=exec_report.usage.certified_notes,
            uncertified_notes=exec_report.usage.uncertified_notes,
            artifact_paths=round_artifacts,
            applied_fields=list(patch_report.applied_fields),
            rejected_patch_fields=dict(patch_report.rejected_fields),
            decision_answer=board.decision.answer,
            decisions_summary=decisions_summary,
            sufficiently_verified=suff_ver,
            decision_confidence=board.decision.confidence,
            certified_influence_share=board.decision.certified_influence_share,
            uncertified_influence_share=board.decision.uncertified_influence_share,
            unknown_voi_max=board.decision.unknown_voi_max,
            field_influences={k: round(v, 6) for k, v in probe_res.field_influences.items()},
            field_voi={k: round(v, 6) for k, v in probe_res.field_voi.items()},
            top_voi_fields=top_voi_fields,
            board_drift=patch_report.board_drift,
            duration_ms=round((time.perf_counter() - t_round_start) * 1000.0, 3),
        )
        rounds.append(round_tel)
        if on_round is not None:
            on_round(round_tel, board)

        # 8. Terminate Check
        is_certified_success = (
            board.decision.confidence >= cfg.confidence_threshold
            and board.decision.uncertified_influence_share <= cfg.max_uncertified_share
            and board.decision.unknown_voi_max < cfg.max_unknown_voi
            and board.decision.answer not in _ABSTAIN_ANSWERS
            and suff_ver == "yes"
        )
        if is_certified_success:
            termination_reason = "certified_success"
            break

        if (time.perf_counter() - t_loop_start) >= cfg.wall_clock_budget_s:
            termination_reason = "timeout"
            break

        executed_verifier_names = {
            n.verifier_name for n in note_store.snapshot() if n.verifier_name
        }
        all_trusted_ran = bool(trusted_verifiers) and all(
            v in executed_verifier_names for v in trusted_verifiers
        )
        missing_certified_signal = bool(pivotal_fields) and any(
            board.fields.get(req) is None
            or board.fields[req].status != "certified"
            or board.fields[req].value in ("unknown", "none")
            for req in pivotal_fields
        )
        if round_idx >= 2 and all_trusted_ran and missing_certified_signal:
            termination_reason = "honest_uncertified"
            break

    # Final Audit (Invariant I1, I2, I3)
    rebuilt_board = rebuild_board_from_notes(
        note_store,
        decision_questions,
        seed_fields=pivotal_fields,
        protected_fields=protected_fields,
        note_extractor=note_extractor,
        verify_artifact_files=True,
    )
    final_drift = compute_board_drift(board, rebuilt_board)
    ungrounded_cnt, illegal_upgrades_cnt = _audit_final_board_invariants(
        board, note_store
    )
    blocking = _compute_blocking_fields(board, termination_reason, pivotal_fields)
    pruned_cnt = _compute_candidates_pruned(board, primary_options, required_signals)

    return LoopRunResult(
        board=board,
        rebuilt_board=rebuilt_board,
        final_board_drift=final_drift,
        rounds=rounds,
        notes=note_store.snapshot(),
        artifacts=_collect_all_artifacts(note_store),
        termination_reason=termination_reason,
        blocking_fields=blocking,
        invariant_violations_caught=invariant_violations_caught,
        ungrounded_fields_on_board=ungrounded_cnt,
        illegal_cert_upgrades_on_board=illegal_upgrades_cnt,
        queries_executed=total_queries_executed,
        verifiers_executed=total_verifiers_executed,
        skipped_duplicates=total_skipped_duplicates,
        rejected_guard=total_rejected_guard,
        candidates_pruned_by_verifiers=pruned_cnt,
        model_telemetry=total_model_telemetry,
        probe_api_calls=total_probe_calls,
        wall_clock_ms=round((time.perf_counter() - t_loop_start) * 1000.0, 3),
    )


__all__ = [
    "LoopConfig",
    "LoopRunResult",
    "RoundTelemetry",
    "TerminationReason",
    "run_investigation_loop",
]
