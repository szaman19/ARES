"""Backend- and domain-agnostic ARES swarm orchestration & context-management loop.

Orchestrates:
1. Seed: Domain verifiers, optional ExecutionBackend, NoteStore, TaskGraph, version-0 board
   (or resume from a saved JSON checkpoint)
2. Propose (Parallel, Bounded): `ModelProvider` workers via `ARESRunner.solve`
3. Verify (Deterministic, Cheapest-First): guarded magnet queries + TRUSTED verifiers + slices
4. Context Synthesis (Fuse): lossless `ModelProvider` synthesis (+ optional note batching)
   validated through `validate_and_merge_patch` (Invariants I1–I7, bounded active board)
5. Evaluate: pluggable `BoardEvaluator` (defaults to `DeterministicBoardEvaluator`)
6. Acquire: targeted (optionally parallel) follow-up task generation over highest-uncertainty
   unresolved fields with attempted-query memory
7. Terminate & Final Audit: certified success, honest uncertified / stagnation, context overflow,
   or budget stop, plus deterministic NoteStore rebuild drift and zero-violation board audit
"""

from __future__ import annotations

import inspect
import json
import os
import time
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ares.context import (
    BoardPatch,
    DeterministicBoardEvaluator,
    FieldPatch,
    NoteExtractorFn,
    archive_stale_fields,
    compute_board_drift,
    compute_weakest_status,
    create_initial_board,
    deterministic_patch_from_notes,
    rebuild_board_from_notes,
    validate_and_merge_patch,
    value_is_grounded_in_notes,
)
from ares.contracts import (
    ArtifactRef,
    BoardEvaluator,
    Domain,
    ExecutionBackend,
    HypothesisSpec,
    InMemoryVerifierRegistry,
    InvestigationBoard,
    Note,
    SearchQuestion,
)
from ares.engine import (
    ARESRunner,
    ProcessorResult,
    RunReport,
    SolveEvent,
    TaskGraph,
    TaskSpec,
)
from ares.models.base import (
    ContextLengthExceededError,
    ModelProvider,
    ModelTelemetry,
)
from ares.models.gemini import GeminiModelProvider
from ares.store import NoteStore

TerminationReason = Literal[
    "certified_success",
    "honest_uncertified",
    "max_rounds",
    "timeout",
    "context_overflow",
]

_ABSTAIN_ANSWERS: frozenset[str] = frozenset(
    {"unknown", "none_verified", "insufficient_evidence", "abstain", "none", ""}
)


class LoopConfig(BaseModel):
    """Configuration for the ARES swarm orchestration and context-management loop."""

    max_rounds: int = Field(default=3, ge=1)
    max_concurrent_workers: int = Field(default=4, ge=1)
    wall_clock_budget_s: float = Field(default=180.0, gt=0.0)
    min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    max_active_fields: int = Field(default=12, ge=1)
    include_images_in_fusion: bool = True
    artifact_dir: str | None = None
    max_notes_per_patch: int | None = Field(default=None, ge=1)
    parallel_followup: bool = False
    stop_when_all_verifiers_exhausted: bool = True
    max_stagnant_rounds: int | None = Field(default=None, ge=1)
    checkpoint_path: str | None = None
    resume_from_checkpoint: str | None = None


class RoundTelemetry(BaseModel):
    """Per-round execution, context synthesis, and evaluation telemetry."""

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
    conclusion_answer: str = "unknown"
    conclusions_summary: dict[str, str] = Field(default_factory=dict)
    sufficiently_verified: str = "no"
    conclusion_confidence: float = 0.0
    certified_field_ratio: float = 0.0
    archived_fields: list[str] = Field(default_factory=list)
    field_priorities: dict[str, float] = Field(default_factory=dict)
    field_uncertainties: dict[str, float] = Field(default_factory=dict)
    top_followup_fields: list[str] = Field(default_factory=list)
    board_drift: float = 0.0
    peak_prompt_tokens: int = 0
    patch_batches: int = 1
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
    peak_prompt_tokens: int = 0
    evaluator_calls: int = 0
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


def _collect_attempted_queries(note_store: NoteStore) -> list[str]:
    seen: set[str] = set()
    attempted: list[str] = []
    for note in note_store.snapshot():
        q = (note.query or "").strip()
        if q and q not in seen:
            seen.add(q)
            attempted.append(q)
    return attempted


def _supports_kwarg(fn: Any, kwarg: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    if kwarg in sig.parameters:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def _invoke_propose_followup(
    provider: ModelProvider,
    *,
    board: InvestigationBoard,
    target_gap_fields: list[str],
    environment_description: str,
    magnet_rules: str,
    available_verifiers: Sequence[str],
    max_queries: int,
    thinking_level: str,
    attempted_queries: Sequence[str],
) -> tuple[list[HypothesisSpec], ModelTelemetry]:
    kwargs: dict[str, Any] = {
        "board": board,
        "target_gap_fields": target_gap_fields,
        "environment_description": environment_description,
        "magnet_rules": magnet_rules,
        "available_verifiers": available_verifiers,
        "max_queries": max_queries,
        "thinking_level": thinking_level,
    }
    if _supports_kwarg(provider.propose_followup_tasks, "attempted_queries"):
        kwargs["attempted_queries"] = list(attempted_queries)
    return provider.propose_followup_tasks(**kwargs)


def _raise_if_context_overflow(report: RunReport) -> None:
    for node in report.graph.nodes.values():
        if node.status != "failed":
            continue
        err = node.error or ""
        if (
            node.error_type == "ContextLengthExceededError"
            or "ContextLengthExceededError" in err
            or "Context window exceeded" in err
        ):
            raise ContextLengthExceededError(err or "Context window exceeded in worker task.")


def _board_field_fingerprint(board: InvestigationBoard) -> dict[str, tuple[str, str]]:
    return {
        fname: (fld.status, json.dumps(fld.value, sort_keys=True, default=str))
        for fname, fld in board.fields.items()
    }


def _dedup_list_preserve_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _save_checkpoint(
    checkpoint_path: str,
    *,
    last_round_index: int,
    board: InvestigationBoard,
    note_store: NoteStore,
    rounds: list[RoundTelemetry],
    model_telemetry: ModelTelemetry,
    evaluator_calls: int,
    queries_executed: int,
    verifiers_executed: int,
    skipped_duplicates: int,
    rejected_guard: int,
    invariant_violations_caught: int,
    stagnant_rounds: int,
) -> None:
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "last_round_index": last_round_index,
        "board": board.model_dump(mode="json"),
        "notes": note_store.to_dicts(),
        "rounds": [r.model_dump(mode="json") for r in rounds],
        "model_telemetry": model_telemetry.model_dump(mode="json"),
        "evaluator_calls": evaluator_calls,
        "queries_executed": queries_executed,
        "verifiers_executed": verifiers_executed,
        "skipped_duplicates": skipped_duplicates,
        "rejected_guard": rejected_guard,
        "invariant_violations_caught": invariant_violations_caught,
        "stagnant_rounds": stagnant_rounds,
    }
    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, target)


def _load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    raw = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid checkpoint payload in {checkpoint_path}.")
    return raw


def _rank_followup_fields(
    board: InvestigationBoard,
    protected_fields: Collection[str],
    limit: int = 4,
) -> list[str]:
    protected_set = frozenset(protected_fields)
    candidates = [
        (fname, fld.uncertainty)
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
        (fname, fld.uncertainty)
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
    """Combine deterministic verifier-extracted patches with provider's extra field patches."""
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
    peak_p = report.usage.peak_prompt_tokens if report.usage.peak_prompt_tokens > 0 else prompt_t
    return ModelTelemetry(
        prompt_tokens=prompt_t,
        completion_tokens=comp_t,
        thought_tokens=0,
        total_tokens=prompt_t + comp_t,
        peak_prompt_tokens=peak_p,
        model_calls=report.usage.model_calls,
        latency_ms=report.usage.wall_clock_ms,
    )


def _telemetry_from_eval_dict(tel: dict[str, Any]) -> ModelTelemetry:
    if not tel:
        return ModelTelemetry()
    prompt_t = int(tel.get("prompt_tokens", 0))
    peak_p = int(tel.get("peak_prompt_tokens", prompt_t))
    return ModelTelemetry(
        prompt_tokens=prompt_t,
        completion_tokens=int(tel.get("completion_tokens", 0)),
        thought_tokens=int(tel.get("thought_tokens", 0)),
        total_tokens=int(tel.get("total_tokens", 0)),
        peak_prompt_tokens=peak_p,
        model_calls=int(tel.get("model_calls", 0)),
        latency_ms=float(tel.get("latency_ms", 0.0)),
    )


def _find_verification_gate_answer(
    board: InvestigationBoard,
    questions: Sequence[SearchQuestion],
) -> str:
    for q in questions:
        if q.question_type == "yes_no" and (
            q.question_id == "sufficiently_verified" or "verified" in q.question.lower()
        ):
            conc = board.conclusions.get(q.question_id)
            return conc.answer if conc is not None else "no"
    return "yes"


def run_investigation_loop(
    domain: Domain,
    workspace_dir: str,
    *,
    backend: ExecutionBackend | None = None,
    provider: ModelProvider | None = None,
    evaluator: BoardEvaluator | None = None,
    config: LoopConfig | None = None,
    on_round: Callable[[RoundTelemetry, InvestigationBoard], None] | None = None,
    on_ares_event: Callable[[SolveEvent], None] | None = None,
) -> LoopRunResult:
    """Execute the complete ARES swarm orchestration and automatic context-management loop."""
    t_loop_start = time.perf_counter()
    cfg = config or LoopConfig()
    eff_provider: ModelProvider = provider if provider is not None else GeminiModelProvider()
    eff_evaluator: BoardEvaluator = evaluator or DeterministicBoardEvaluator(
        answer_resolver=getattr(domain, "resolve_answers", None)
    )

    resolved_workspace = str(Path(workspace_dir).resolve())
    eff_artifact_dir = (
        str(Path(cfg.artifact_dir).resolve())
        if cfg.artifact_dir
        else str(Path(resolved_workspace) / "_artifacts")
    )
    Path(eff_artifact_dir).mkdir(parents=True, exist_ok=True)

    # 1. Seed (or resume from JSON checkpoint)
    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    available_verifiers = registry.list_verifiers()
    trusted_verifiers = [
        v for v in available_verifiers if registry.metadata(v).get("trusted", True)
    ]

    env_desc = domain.environment_description()
    magnet_rules = domain.magnet_rules()
    search_questions = domain.search_questions()
    primary_options = search_questions[0].options if search_questions else []
    pivotal_fields = list(domain.pivotal_fields())
    required_signals = tuple(domain.required_signal_fields())
    protected_fields = frozenset(pivotal_fields) | frozenset(required_signals)
    note_extractor = domain.extract_field_patches_from_note

    note_store = NoteStore()
    board = create_initial_board(search_questions, seed_fields=pivotal_fields)
    rounds: list[RoundTelemetry] = []
    total_model_telemetry = ModelTelemetry()
    total_evaluator_calls = 0
    total_queries_executed = 0
    total_verifiers_executed = 0
    total_skipped_duplicates = 0
    total_rejected_guard = 0
    invariant_violations_caught = 0
    stagnant_rounds = 0
    start_round_idx = 1

    if cfg.resume_from_checkpoint and Path(cfg.resume_from_checkpoint).is_file():
        ckpt = _load_checkpoint(cfg.resume_from_checkpoint)
        board = InvestigationBoard.model_validate(ckpt["board"])
        note_store = NoteStore.from_dicts(ckpt.get("notes", []))
        rounds = [RoundTelemetry.model_validate(r) for r in ckpt.get("rounds", [])]
        total_model_telemetry = ModelTelemetry.model_validate(ckpt.get("model_telemetry", {}))
        total_evaluator_calls = int(ckpt.get("evaluator_calls", 0))
        total_queries_executed = int(ckpt.get("queries_executed", 0))
        total_verifiers_executed = int(ckpt.get("verifiers_executed", 0))
        total_skipped_duplicates = int(ckpt.get("skipped_duplicates", 0))
        total_rejected_guard = int(ckpt.get("rejected_guard", 0))
        invariant_violations_caught = int(ckpt.get("invariant_violations_caught", 0))
        stagnant_rounds = int(ckpt.get("stagnant_rounds", 0))
        start_round_idx = int(ckpt.get("last_round_index", len(rounds))) + 1

    eff_backend = backend if backend is not None else domain.create_backend(resolved_workspace)
    propose_proc = eff_provider.make_propose_processor(
        environment_description=env_desc,
        magnet_rules=magnet_rules,
        pivotal_fields=pivotal_fields,
        available_verifiers=available_verifiers,
    )

    def _followup_proc(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
        del dep_outputs
        ctx = spec.context or {}
        gap_field = str(ctx.get("gap_field", ""))
        raw_board = ctx.get("board", board)
        board_obj = (
            raw_board
            if isinstance(raw_board, InvestigationBoard)
            else InvestigationBoard.model_validate(raw_board)
        )
        verifiers_arg = list(ctx.get("available_verifiers", available_verifiers))
        attempted_arg = list(ctx.get("attempted_queries", ()))
        max_q = int(ctx.get("max_queries", 2))
        t_level = str(ctx.get("thinking_level", "low"))
        specs, tel = _invoke_propose_followup(
            eff_provider,
            board=board_obj,
            target_gap_fields=[gap_field] if gap_field else [],
            environment_description=env_desc,
            magnet_rules=magnet_rules,
            available_verifiers=verifiers_arg,
            max_queries=max_q,
            thinking_level=t_level,
            attempted_queries=attempted_arg,
        )
        return ProcessorResult(
            output=specs,
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            peak_prompt_tokens=tel.peak_prompt_tokens,
            model_calls=tel.model_calls,
        )

    graph = TaskGraph()
    runner = ARESRunner(
        backend=eff_backend,
        registry=registry,
        note_store=note_store,
        processors={"propose": propose_proc, "followup": _followup_proc},
    )

    termination_reason: TerminationReason = "max_rounds"

    for round_idx in range(start_round_idx, cfg.max_rounds + 1):
        if (time.perf_counter() - t_loop_start) >= cfg.wall_clock_budget_s:
            termination_reason = "timeout"
            break

        t_round_start = time.perf_counter()
        round_model_telemetry = ModelTelemetry()
        pre_round_fingerprint = _board_field_fingerprint(board)
        unmerged_round_notes: list[Note] = []

        try:
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
                _raise_if_context_overflow(prop_report)
                prop_tel = _telemetry_from_ares_report(prop_report)
                round_model_telemetry = round_model_telemetry.plus(prop_tel)
                total_model_telemetry = total_model_telemetry.plus(prop_tel)

                proposed_specs: list[HypothesisSpec] = []
                for ptask in propose_tasks:
                    node_out = graph.nodes[ptask.task_id].output
                    if isinstance(node_out, list):
                        for item in node_out:
                            if isinstance(item, HypothesisSpec):
                                proposed_specs.append(item)
            else:
                target_gaps = _rank_followup_fields(board, protected_fields, limit=4)
                eff_followup_verifiers = unexecuted_verifiers or available_verifiers
                attempted_queries = _collect_attempted_queries(note_store)

                if cfg.parallel_followup and len(target_gaps) > 1:
                    followup_tasks = [
                        TaskSpec(
                            task_id=f"r{round_idx}_followup_{idx}_{gap}",
                            task_type="custom",
                            processor_name="followup",
                            requested_under=board.board_version,
                            context={
                                "gap_field": gap,
                                "board": board,
                                "available_verifiers": eff_followup_verifiers,
                                "attempted_queries": attempted_queries,
                                "max_queries": 2,
                                "thinking_level": "low",
                            },
                        )
                        for idx, gap in enumerate(target_gaps[: cfg.max_concurrent_workers])
                    ]
                    followup_report = runner.solve(
                        graph=graph,
                        tasks=followup_tasks,
                        max_concurrent=cfg.max_concurrent_workers,
                        on_event=on_ares_event,
                    )
                    _raise_if_context_overflow(followup_report)
                    followup_tel = _telemetry_from_ares_report(followup_report)
                    round_model_telemetry = round_model_telemetry.plus(followup_tel)
                    total_model_telemetry = total_model_telemetry.plus(followup_tel)

                    proposed_specs = []
                    for ftask in followup_tasks:
                        node_out = graph.nodes[ftask.task_id].output
                        if isinstance(node_out, list):
                            for item in node_out:
                                if isinstance(item, HypothesisSpec):
                                    proposed_specs.append(item)
                else:
                    proposed_specs, followup_tel = _invoke_propose_followup(
                        eff_provider,
                        board=board,
                        target_gap_fields=target_gaps,
                        environment_description=env_desc,
                        magnet_rules=magnet_rules,
                        available_verifiers=eff_followup_verifiers,
                        max_queries=3,
                        thinking_level="low",
                        attempted_queries=attempted_queries,
                    )
                    round_model_telemetry = round_model_telemetry.plus(followup_tel)
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
            unmerged_round_notes = list(new_notes)
            if round_idx >= 2 and len(new_notes) == 0:
                termination_reason = "honest_uncertified"
                break

            # 4. Lossless Context Synthesis (with optional count-based note batching) & Invariants
            if cfg.max_notes_per_patch is not None and len(new_notes) > cfg.max_notes_per_patch:
                bsize = cfg.max_notes_per_patch
                note_batches = [
                    new_notes[i : i + bsize] for i in range(0, len(new_notes), bsize)
                ]
            else:
                note_batches = [new_notes]

            multi_batch = len(note_batches) > 1
            round_applied_fields: list[str] = []
            round_rejected_fields: dict[str, str] = {}
            round_archived_fields: list[str] = []
            round_board_drift = 0.0

            for batch_notes in note_batches:
                carbon_patch, carbon_tel = eff_provider.propose_board_patch(
                    board=board,
                    new_notes=batch_notes,
                    environment_description=env_desc,
                    pivotal_fields=pivotal_fields,
                    include_images=cfg.include_images_in_fusion,
                    thinking_level="low",
                )
                round_model_telemetry = round_model_telemetry.plus(carbon_tel)
                total_model_telemetry = total_model_telemetry.plus(carbon_tel)

                merged_patch = _merge_canonical_and_carbon_patches(
                    board=board,
                    new_notes=batch_notes,
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
                    advance_round=(not multi_batch),
                    max_active_fields=cfg.max_active_fields,
                    verify_artifact_files=True,
                    compute_drift=(not multi_batch),
                )
                round_applied_fields.extend(patch_report.applied_fields)
                round_rejected_fields.update(patch_report.rejected_fields)
                round_archived_fields.extend(patch_report.archived_fields)
                invariant_violations_caught += len(patch_report.rejected_fields)
                unmerged_round_notes = unmerged_round_notes[len(batch_notes) :]
                if not multi_batch:
                    round_board_drift = patch_report.board_drift

            round_applied_fields = _dedup_list_preserve_order(round_applied_fields)
            round_archived_fields = _dedup_list_preserve_order(round_archived_fields)

            if multi_batch:
                board, end_archived = archive_stale_fields(
                    board,
                    updated_fields=round_applied_fields,
                    protected_fields=protected_fields,
                    advance_round=True,
                    max_active_fields=cfg.max_active_fields,
                )
                round_archived_fields = _dedup_list_preserve_order(
                    [*round_archived_fields, *end_archived]
                )
                rebuilt_mid = rebuild_board_from_notes(
                    note_store,
                    seed_fields=pivotal_fields,
                    protected_fields=protected_fields,
                    note_extractor=note_extractor,
                    verify_artifact_files=True,
                )
                round_board_drift = compute_board_drift(board, rebuilt_mid)

            # 5. Evaluate Board State
            png_artifacts = _collect_verifier_png_artifacts(note_store)
            eval_res = eff_evaluator.evaluate(
                board,
                search_questions,
                required_signal_fields=required_signals,
                artifact_refs=png_artifacts,
            )
            board = eval_res.updated_board
            total_evaluator_calls += eval_res.evaluator_calls
            if eval_res.telemetry:
                eval_tel = _telemetry_from_eval_dict(eval_res.telemetry)
                round_model_telemetry = round_model_telemetry.plus(eval_tel)
                total_model_telemetry = total_model_telemetry.plus(eval_tel)
            unmerged_round_notes = []
        except ContextLengthExceededError:
            if unmerged_round_notes:
                det_recovery = deterministic_patch_from_notes(
                    unmerged_round_notes, note_extractor=note_extractor
                )
                if det_recovery.updates:
                    board, recovery_report = validate_and_merge_patch(
                        board,
                        det_recovery,
                        note_store,
                        protected_fields=protected_fields,
                        seed_fields=pivotal_fields,
                        note_extractor=note_extractor,
                        strict=False,
                        advance_round=False,
                        max_active_fields=cfg.max_active_fields,
                        verify_artifact_files=True,
                        compute_drift=False,
                    )
                    invariant_violations_caught += len(recovery_report.rejected_fields)
            termination_reason = "context_overflow"
            break

        post_round_fingerprint = _board_field_fingerprint(board)
        if post_round_fingerprint == pre_round_fingerprint:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0

        top_followup = _rank_followup_fields(board, protected_fields, limit=4)
        suff_ver = _find_verification_gate_answer(board, search_questions)
        conclusions_summary = {qid: conc.answer for qid, conc in board.conclusions.items()}

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
            applied_fields=round_applied_fields,
            rejected_patch_fields=round_rejected_fields,
            conclusion_answer=board.conclusion.answer,
            conclusions_summary=conclusions_summary,
            sufficiently_verified=suff_ver,
            conclusion_confidence=board.conclusion.confidence,
            certified_field_ratio=board.conclusion.certified_field_ratio,
            archived_fields=round_archived_fields,
            field_priorities={k: round(v, 6) for k, v in eval_res.field_priorities.items()},
            field_uncertainties={
                k: round(v, 6) for k, v in eval_res.field_uncertainties.items()
            },
            top_followup_fields=top_followup,
            board_drift=round_board_drift,
            peak_prompt_tokens=round_model_telemetry.peak_prompt_tokens,
            patch_batches=len(note_batches),
            duration_ms=round((time.perf_counter() - t_round_start) * 1000.0, 3),
        )
        rounds.append(round_tel)
        if on_round is not None:
            on_round(round_tel, board)

        if cfg.checkpoint_path:
            _save_checkpoint(
                cfg.checkpoint_path,
                last_round_index=round_idx,
                board=board,
                note_store=note_store,
                rounds=rounds,
                model_telemetry=total_model_telemetry,
                evaluator_calls=total_evaluator_calls,
                queries_executed=total_queries_executed,
                verifiers_executed=total_verifiers_executed,
                skipped_duplicates=total_skipped_duplicates,
                rejected_guard=total_rejected_guard,
                invariant_violations_caught=invariant_violations_caught,
                stagnant_rounds=stagnant_rounds,
            )

        # 7. Terminate Check
        is_certified_success = (
            eval_res.verified
            and board.conclusion.confidence >= cfg.min_confidence
            and board.conclusion.answer not in _ABSTAIN_ANSWERS
            and suff_ver == "yes"
        )
        if is_certified_success:
            termination_reason = "certified_success"
            break

        if (time.perf_counter() - t_loop_start) >= cfg.wall_clock_budget_s:
            termination_reason = "timeout"
            break

        if cfg.max_stagnant_rounds is not None and stagnant_rounds >= cfg.max_stagnant_rounds:
            termination_reason = "honest_uncertified"
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
        if (
            cfg.stop_when_all_verifiers_exhausted
            and round_idx >= 2
            and all_trusted_ran
            and missing_certified_signal
        ):
            termination_reason = "honest_uncertified"
            break

    # Final Audit (Invariant I1, I2, I3)
    rebuilt_board = rebuild_board_from_notes(
        note_store,
        search_questions,
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
        peak_prompt_tokens=total_model_telemetry.peak_prompt_tokens,
        evaluator_calls=total_evaluator_calls,
        wall_clock_ms=round((time.perf_counter() - t_loop_start) * 1000.0, 3),
    )


run_search_loop = run_investigation_loop


__all__ = [
    "LoopConfig",
    "LoopRunResult",
    "RoundTelemetry",
    "TerminationReason",
    "run_investigation_loop",
    "run_search_loop",
]
