"""Unit & integration tests for `ExoplanetTriageDomain` (Real NASA TESS Vetting)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from ares.engine import ARESRunner, TaskGraph, TaskSpec
from ares.store import NoteStore
from contracts import (
    DecisionQuestion,
    Domain,
    GuardValidationError,
    HypothesisSpec,
    InMemoryVerifierRegistry,
    InvestigationBoard,
)
from domain.exoplanet_artifacts import render_exoplanet_diagnostic_card
from domain.exoplanet_triage import (
    EXOPLANET_PIVOTAL_FIELDS,
    EXOPLANET_REQUIRED_SIGNALS,
    ExoplanetTriageDomain,
)
from fusion.loop import LoopConfig, run_investigation_loop
from fusion.patch import (
    create_initial_board,
    deterministic_patch_from_notes,
    rebuild_board_from_notes,
    validate_and_merge_patch,
)
from models.decision_probe import (
    BoardEvaluation,
    ModelTelemetry,
    ProbeResult,
    QuestionEvaluation,
)


def test_exoplanet_domain_implements_protocol() -> None:
    domain = ExoplanetTriageDomain()
    assert isinstance(domain, Domain)
    assert domain.pivotal_fields() == EXOPLANET_PIVOTAL_FIELDS
    assert domain.required_signal_fields() == EXOPLANET_REQUIRED_SIGNALS
    assert len(domain.needle_menu()) == 9


def test_zero_ground_truth_leakage_and_guard_enforcement(tmp_path: Path) -> None:
    domain = ExoplanetTriageDomain()
    ws = tmp_path / "sector"
    domain.generate(str(ws), size_gb=0.02, seed=42)
    needle = domain.needle_menu()[3]  # look_alike_decoy case
    domain.plant(str(ws), needle)

    forbidden_cols = {
        "tfopwg_disp",
        "disposition",
        "ground_truth",
        "is_planet",
        "category",
        "toi",
        "label",
    }
    for tbl in ("lightcurves", "stellar_catalog", "pixel_stamps"):
        for pq_file in (ws / tbl).glob("*.parquet"):
            schema_names = {n.lower() for n in pq.read_schema(pq_file).names}
            assert schema_names.isdisjoint(forbidden_cols), (
                f"Forbidden label column leaked in {pq_file}: {schema_names & forbidden_cols}"
            )

    backend = domain.create_backend(str(ws))
    with pytest.raises(GuardValidationError):
        backend.validate_query("DROP TABLE lightcurves")
    with pytest.raises(GuardValidationError):
        backend.validate_query("SELECT * FROM read_parquet('/etc/passwd')")


def test_dedup_and_all_nine_cases_deterministic_verifiers(tmp_path: Path) -> None:
    domain = ExoplanetTriageDomain()
    base_dir = tmp_path / "base"
    domain.generate(str(base_dir), size_gb=0.02, seed=42)

    # Check content-hash deduplication across multi-round ARESRunner.solve(graph=...)
    dedup_ws = tmp_path / "dedup_ws"
    domain.generate(str(dedup_ws), size_gb=0.01, seed=7)
    domain.plant(str(dedup_ws), domain.needle_menu()[0])

    backend = domain.create_backend(str(dedup_ws))
    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    store = NoteStore()
    graph = TaskGraph()
    runner = ARESRunner(backend=backend, registry=registry, note_store=store)

    q_spec = HypothesisSpec(
        query="SELECT tic_id, COUNT(*) AS n FROM lightcurves WHERE quality_flag = 0 GROUP BY tic_id LIMIT 10",
        hypothesis_family="periodic_transit_photometry_scan",
    )
    rep1 = runner.solve(
        graph=graph,
        tasks=[TaskSpec(task_id="t1", task_type="query", hypothesis_spec=q_spec)],
    )
    rep2 = runner.solve(
        graph=graph,
        tasks=[TaskSpec(task_id="t2", task_type="query", hypothesis_spec=q_spec)],
    )
    assert rep1.usage.queries_executed == 1
    assert rep2.usage.skipped_duplicate_tasks == 1

    # Verify all 9 benchmark cases (3 planet, 3 impostor decoy, 3 under-determined)
    import shutil

    for needle in domain.needle_menu():
        case_ws = tmp_path / f"case_{needle.case_id}"
        shutil.copytree(base_dir, case_ws)
        truth = domain.plant(str(case_ws), needle)

        c_backend = domain.create_backend(str(case_ws))
        c_store = NoteStore()
        art_dir = tmp_path / "arts" / needle.case_id
        notes = []
        for idx, vname in enumerate(registry.list_verifiers()):
            _vres, note = c_backend.execute_verifier(
                registry,
                vname,
                params={"artifact_dir": str(art_dir)},
                requested_under=0,
                hypothesis_family=vname,
                content_hash=f"{needle.case_id}_{idx}_{vname}",
                note_store=c_store,
            )
            notes.append(note)

        board = create_initial_board(
            domain.decision_questions(),
            seed_fields=domain.pivotal_fields(),
        )
        patch = deterministic_patch_from_notes(
            notes, note_extractor=domain.extract_field_patches_from_note
        )
        board, report = validate_and_merge_patch(
            board,
            patch,
            c_store,
            protected_fields=domain.pivotal_fields(),
            seed_fields=domain.pivotal_fields(),
            note_extractor=domain.extract_field_patches_from_note,
        )
        assert report.board_drift == 0.0

        conf_field = board.fields["confirmed_exoplanet_target"]
        if needle.category in ("multi_clue_join", "look_alike_decoy"):
            assert conf_field.status == "certified", f"Failed cert on {needle.case_id}"
            assert conf_field.value == truth.primary_target
            if truth.decoy_target:
                assert conf_field.value != truth.decoy_target
                # Loud decoy MUST have triggered periodic_transit_snr!
                snr_val = board.fields["periodic_transit_snr"].value
                snr_list = [snr_val] if isinstance(snr_val, str) else list(snr_val)
                assert truth.decoy_target in snr_list
        else:
            assert conf_field.status == "uncertified"
            assert conf_field.value == "none"
            assert board.fields["ccd_centroid_lock"].status == "uncertified"


def test_deterministic_png_sha256_and_offline_fallback(tmp_path: Path) -> None:
    # Exercise offline fallback generator explicitly
    fallback_domain = ExoplanetTriageDomain(staged_dir=tmp_path / "nonexistent_staged")
    assert fallback_domain.is_using_real_nasa_data is False

    ws = tmp_path / "fallback_ws"
    fallback_domain.generate(str(ws), size_gb=0.01, seed=99)
    needle = fallback_domain.needle_menu()[3]  # decoy case
    fallback_domain.plant(str(ws), needle)

    import duckdb

    con = duckdb.connect(":memory:")
    art1 = render_exoplanet_diagnostic_card(
        con, str(ws), needle.target_entity, str(tmp_path / "out1")
    )
    art2 = render_exoplanet_diagnostic_card(
        con, str(ws), needle.target_entity, str(tmp_path / "out2")
    )
    art_masked = render_exoplanet_diagnostic_card(
        con, str(ws), needle.target_entity, str(tmp_path / "out_mask"), mask_ccd_panel=True
    )
    con.close()

    assert art1.produced_by_verifier is True
    assert art1.sha256 == art2.sha256
    assert art1.sha256 != art_masked.sha256
    assert Path(art1.path).stat().st_size > 15_000


class _StubGeminiProvider:
    def make_propose_processor(self, **kwargs: Any):
        from ares.engine import ProcessorResult

        def _proc(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
            h = HypothesisSpec(
                query="SELECT tic_id, tmag, crowding_metric FROM stellar_catalog WHERE candidate_period_days IS NOT NULL LIMIT 10",
                hypothesis_family=str(spec.context.get("hypothesis_family", "scan")),
                expected_signal="List candidate TCEs",
                target_fields=["periodic_transit_snr"],
            )
            return ProcessorResult(output=[h])

        return _proc

    def propose_followup_tasks(self, **kwargs: Any):
        return [], ModelTelemetry()

    def propose_board_patch(self, **kwargs: Any):
        return {"updates": {}, "rationale": "Deterministic canonical verifier merge"}, ModelTelemetry()


class _StubDecisionProbe:
    def probe_board(
        self,
        board: InvestigationBoard,
        questions: list[DecisionQuestion],
        *,
        required_signal_fields: tuple[str, ...] = (),
        artifact_refs: list[Any] | None = None,
        **kwargs: Any,
    ) -> ProbeResult:
        # Mirror exact intersection semantics over required_signal_fields
        sets: list[set[str]] = []
        all_cert = True
        for req in required_signal_fields:
            fld = board.fields.get(req)
            if fld is None or fld.status != "certified" or fld.value in ("unknown", "none"):
                all_cert = False
                break
            vals = [fld.value] if isinstance(fld.value, str) else list(fld.value)
            sets.append(set(vals))

        common = sorted(set.intersection(*sets)) if (all_cert and sets) else []
        if len(common) == 1:
            ans_tic = common[0]
            ans_act = "trigger_flagship_slew"
            ans_ver = "yes"
            conf = 0.94
            influences = {k: (0.25 if k in required_signal_fields else 0.0) for k in board.fields}
            cert_share = 1.0
            voi_map = {k: 0.0 for k in board.fields}
            voi_max = 0.0
        else:
            ans_tic = "none_verified"
            ans_act = "request_spectroscopy"
            ans_ver = "no"
            conf = 0.88
            influences = {k: 0.0 for k in board.fields}
            cert_share = 0.0
            voi_map = {
                k: (0.82 if board.fields[k].status != "certified" and k in required_signal_fields else 0.0)
                for k in board.fields
            }
            voi_max = max(voi_map.values(), default=0.0)

        eval_answers = {
            "confirmed_planet_tic": QuestionEvaluation(
                question_id="confirmed_planet_tic", answer=ans_tic, confidence=conf
            ),
            "recommended_followup": QuestionEvaluation(
                question_id="recommended_followup", answer=ans_act, confidence=conf
            ),
            "sufficiently_verified": QuestionEvaluation(
                question_id="sufficiently_verified", answer=ans_ver, confidence=conf
            ),
        }
        updated = board.model_copy(deep=True)
        for k, fobj in updated.fields.items():
            fobj.influence = influences.get(k, 0.0)
            fobj.voi = voi_map.get(k, 0.0)
        for q in questions:
            qe = eval_answers[q.question_id]
            updated.decisions[q.question_id].answer = qe.answer
            updated.decisions[q.question_id].confidence = qe.confidence
            updated.decisions[q.question_id].certified_influence_share = cert_share
            updated.decisions[q.question_id].uncertified_influence_share = 0.0
            updated.decisions[q.question_id].unknown_voi_max = voi_max
        updated.decision = updated.decisions["confirmed_planet_tic"].model_copy(deep=True)

        return ProbeResult(
            base_evaluation=BoardEvaluation(answers=eval_answers),
            field_influences=influences,
            raw_swings=influences,
            certified_influence_share=cert_share,
            uncertified_influence_share=0.0,
            field_voi=voi_map,
            unknown_voi_max=voi_max,
            updated_board=updated,
            api_calls=6,
        )


def test_end_to_end_loop_invariants_on_real_nasa_decoy_and_ud(tmp_path: Path) -> None:
    domain = ExoplanetTriageDomain()
    for needle_idx, expected_term in ((3, "certified_success"), (6, "honest_uncertified")):
        ws = tmp_path / f"loop_{needle_idx}"
        domain.generate(str(ws), size_gb=0.02, seed=42)
        needle = domain.needle_menu()[needle_idx]
        truth = domain.plant(str(ws), needle)

        res = run_investigation_loop(
            domain,
            str(ws),
            provider=_StubGeminiProvider(),  # type: ignore[arg-type]
            probe_client=_StubDecisionProbe(),  # type: ignore[arg-type]
            config=LoopConfig(max_rounds=2, artifact_dir=str(tmp_path / f"art_{needle_idx}")),
        )
        assert res.termination_reason == expected_term
        assert res.final_board_drift == 0.0
        assert res.ungrounded_fields_on_board == 0
        assert res.illegal_cert_upgrades_on_board == 0
        assert len(res.artifacts) >= 1

        sc = domain.score(res.board.model_dump(), truth)
        assert sc["overall_pass"] is True
