"""Tests for `examples/incident_rca`: dataset generation, isolation, DuckDB verifiers, scoring, and headless charts."""

from __future__ import annotations

import ast
import hashlib
import shutil
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from ares.contracts import (
    ArtifactRef,
    BoardConclusion,
    BoardField,
    Domain,
    InMemoryVerifierRegistry,
    InvestigationBoard,
    Note,
)
from examples.incident_rca.artifacts import render_candidate_comparison_plot
from examples.incident_rca.charts import (
    render_benchmark_summary_chart,
    render_board_priority_chart,
    render_calibration_plot,
    render_influence_vs_voi_plot,
)
from examples.incident_rca.domain import SERVICES, MicroBenchDomain


@pytest.fixture(scope="module")
def domain() -> MicroBenchDomain:
    return MicroBenchDomain(num_background_shards=4)


@pytest.fixture(scope="module")
def base_haystack_dir(tmp_path_factory: pytest.TempPathFactory, domain: MicroBenchDomain) -> Path:
    out_dir = tmp_path_factory.mktemp("haystack_base")
    domain.generate(str(out_dir), size_gb=0.02, seed=42)
    return out_dir


def _copy_haystack(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_no_pyplot_import(py_path: Path) -> None:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "pyplot" not in alias.name, f"Forbidden pyplot import in {py_path}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "pyplot" not in mod, f"Forbidden pyplot import in {py_path}"
            for alias in node.names:
                assert alias.name != "pyplot", f"Forbidden pyplot import in {py_path}"


def test_no_global_pyplot_imports_across_examples() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    _assert_no_pyplot_import(repo_root / "examples" / "incident_rca" / "artifacts.py")
    _assert_no_pyplot_import(repo_root / "examples" / "incident_rca" / "charts.py")
    _assert_no_pyplot_import(repo_root / "examples" / "exoplanet_triage" / "artifacts.py")
    _assert_no_pyplot_import(repo_root / "examples" / "nyc_cameras" / "domain.py")


def test_domain_protocol_conformance(domain: MicroBenchDomain) -> None:
    assert isinstance(domain, Domain)
    assert domain.name == "incident_logs_micro_bench"
    assert len(domain.hypothesis_families()) >= 3
    questions = domain.search_questions()
    q_ids = {q.question_id for q in questions}
    assert {"root_cause", "recommended_action", "sufficiently_verified"} <= q_ids
    assert "service_logs" in domain.environment_description()
    assert "metrics" in domain.environment_description()
    assert "traces" in domain.environment_description()
    assert "SELECT" in domain.magnet_rules()
    assert domain.pivotal_fields() == [
        "logs_retry_spike",
        "metrics_zscore_breach",
        "traces_error_cluster",
        "three_way_root_cause",
    ]
    assert domain.required_signal_fields() == (
        "logs_retry_spike",
        "metrics_zscore_breach",
        "traces_error_cluster",
    )


def test_needle_menu_structure(domain: MicroBenchDomain) -> None:
    menu = domain.needle_menu()
    assert len(menu) == 9
    categories = [n.category for n in menu]
    assert categories.count("multi_clue_join") == 3
    assert categories.count("look_alike_decoy") == 3
    assert categories.count("under_determined") == 3
    assert len({n.case_id for n in menu}) == 9

    for n in menu:
        if n.category == "look_alike_decoy":
            assert n.decoy_entity is not None
            assert n.decoy_entity != n.target_entity
        if n.category == "under_determined":
            assert n.missing_signal in ("logs", "metrics", "traces")


def test_deterministic_generation_and_row_scale(tmp_path: Path, domain: MicroBenchDomain) -> None:
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    domain.generate(str(dir_a), size_gb=0.02, seed=123)
    domain.generate(str(dir_b), size_gb=0.02, seed=123)

    total_rows = 0
    for table in ("service_logs", "metrics", "traces"):
        files_a = sorted((dir_a / table).glob("*.parquet"))
        files_b = sorted((dir_b / table).glob("*.parquet"))
        assert [f.name for f in files_a] == [f.name for f in files_b]
        assert len(files_a) == 4
        for fa, fb in zip(files_a, files_b):
            assert _sha256_file(fa) == _sha256_file(fb)
            total_rows += pq.read_metadata(fa).num_rows

    assert 100_000 <= total_rows <= 250_000


def test_background_haystack_isolation_and_scan_speed(
    base_haystack_dir: Path,
    domain: MicroBenchDomain,
) -> None:
    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    assert set(registry.list_verifiers()) >= {
        "verify_upstream_retry_spike",
        "verify_metric_zscore_breach",
        "verify_trace_error_cluster",
        "verify_three_way_root_cause",
    }

    con = duckdb.connect(":memory:")
    t0 = time.perf_counter()
    for vname in registry.list_verifiers():
        res = registry.get(vname)(con, str(base_haystack_dir))
        assert res.passed is False, f"Background haystack falsely triggered {vname}: {res}"
        assert res.certification == "UNCERTIFIED"
        assert res.row_count == 0
    elapsed = time.perf_counter() - t0
    con.close()

    assert elapsed < 2.0, f"DuckDB scan too slow: {elapsed:.2f}s"


def test_no_ground_truth_leakage_in_schema_or_filenames(
    tmp_path: Path,
    base_haystack_dir: Path,
    domain: MicroBenchDomain,
) -> None:
    case_dir = tmp_path / "leak_check"
    _copy_haystack(base_haystack_dir, case_dir)
    needle = domain.needle_menu()[3]
    truth = domain.plant(str(case_dir), needle)

    forbidden_tokens = ("needle", "decoy", "truth", "label", needle.case_id)
    for table in ("service_logs", "metrics", "traces"):
        for f in (case_dir / table).glob("*.parquet"):
            fname_lower = f.name.lower()
            for tok in forbidden_tokens:
                assert tok.lower() not in fname_lower, f"Filename leaked token {tok}: {f.name}"
            schema = pq.read_schema(f)
            for col in schema.names:
                for tok in forbidden_tokens:
                    assert tok.lower() not in col.lower(), f"Column leaked token {tok}: {col}"

    con = duckdb.connect(":memory:")
    leaked_logs = con.execute(
        f"""
        SELECT COUNT(*) FROM read_parquet('{case_dir}/service_logs/*.parquet')
        WHERE LOWER(trace_id) LIKE '%needle%'
           OR LOWER(trace_id) LIKE '%decoy%'
           OR LOWER(message) LIKE '%needle%'
           OR LOWER(message) LIKE '%decoy%'
        """
    ).fetchone()[0]
    con.close()
    assert leaked_logs == 0
    assert len(truth.planted_artifacts) > 0


def test_plant_appends_shards_without_modifying_background(
    tmp_path: Path,
    base_haystack_dir: Path,
    domain: MicroBenchDomain,
) -> None:
    case_dir = tmp_path / "append_check"
    _copy_haystack(base_haystack_dir, case_dir)

    before_hashes = {
        str(p.relative_to(case_dir)): _sha256_file(p)
        for p in case_dir.rglob("*.parquet")
    }

    needle = domain.needle_menu()[0]
    truth = domain.plant(str(case_dir), needle)

    after_files = {str(p.relative_to(case_dir)) for p in case_dir.rglob("*.parquet")}
    for rel_path, old_hash in before_hashes.items():
        assert rel_path in after_files
        assert _sha256_file(case_dir / rel_path) == old_hash

    new_files = after_files - set(before_hashes.keys())
    assert len(new_files) == 3
    assert len(truth.planted_artifacts) == 3


@pytest.mark.parametrize("case_idx", range(9))
def test_all_nine_micro_bench_cases_ground_truth_isolation(
    case_idx: int,
    tmp_path: Path,
    base_haystack_dir: Path,
    domain: MicroBenchDomain,
) -> None:
    needle = domain.needle_menu()[case_idx]
    case_dir = tmp_path / needle.case_id
    _copy_haystack(base_haystack_dir, case_dir)
    truth = domain.plant(str(case_dir), needle)

    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    con = duckdb.connect(":memory:")

    res_logs = registry.get("verify_upstream_retry_spike")(con, str(case_dir))
    res_metrics = registry.get("verify_metric_zscore_breach")(con, str(case_dir))
    res_traces = registry.get("verify_trace_error_cluster")(con, str(case_dir))
    res_3way = registry.get("verify_three_way_root_cause")(con, str(case_dir))
    con.close()

    if needle.category == "multi_clue_join":
        assert truth.is_sufficiently_verified is True
        assert truth.expected_status == "CERTIFIED"
        assert res_logs.passed is True
        assert res_metrics.passed is True
        assert res_traces.passed is True
        assert res_3way.passed is True
        assert res_3way.certification == "CERTIFIED"
        assert res_3way.evidence["verified_root_causes"] == [truth.primary_target]
        assert res_3way.evidence["candidates"][0]["shard_id"] == truth.metadata["shard_id"]

    elif needle.category == "look_alike_decoy":
        assert truth.is_sufficiently_verified is True
        assert truth.expected_status == "CERTIFIED"
        assert truth.decoy_target is not None

        decoy_signals = 0
        for res, count_key in (
            (res_logs, "distinct_traces"),
            (res_metrics, "breach_buckets"),
            (res_traces, "distinct_traces"),
        ):
            by_svc = {c["service"]: c[count_key] for c in res.evidence["candidates"]}
            if truth.decoy_target in by_svc:
                decoy_signals += 1
                if count_key == "distinct_traces":
                    assert by_svc[truth.decoy_target] > by_svc[truth.primary_target]
        assert decoy_signals == 2

        assert res_3way.passed is True
        assert res_3way.certification == "CERTIFIED"
        assert res_3way.evidence["verified_root_causes"] == [truth.primary_target]
        assert truth.decoy_target not in res_3way.evidence["verified_root_causes"]

    elif needle.category == "under_determined":
        assert truth.is_sufficiently_verified is False
        assert truth.expected_status == "UNCERTIFIED"
        assert truth.primary_target is None

        signal_map = {
            "logs": res_logs.passed,
            "metrics": res_metrics.passed,
            "traces": res_traces.passed,
        }
        assert sum(signal_map.values()) == 2
        assert signal_map[needle.missing_signal] is False

        assert res_3way.passed is False
        assert res_3way.certification == "UNCERTIFIED"
        assert res_3way.evidence["verified_root_causes"] == []


def test_score_certified_decoy_and_under_determined(
    tmp_path: Path,
    base_haystack_dir: Path,
    domain: MicroBenchDomain,
) -> None:
    decoy_needle = domain.needle_menu()[3]
    case_dir = tmp_path / "score_decoy"
    _copy_haystack(base_haystack_dir, case_dir)
    truth_decoy = domain.plant(str(case_dir), decoy_needle)
    exp_action = truth_decoy.expected_decisions["recommended_action"]

    good_board = {
        "board_version": 5,
        "conclusion": {
            "question_id": "root_cause",
            "question": "Which component is the verified root cause?",
            "answer": truth_decoy.primary_target,
            "confidence": 1.0,
            "verified": True,
            "certified_field_ratio": 0.92,
        },
        "conclusions": {
            "root_cause": {"answer": truth_decoy.primary_target},
            "recommended_action": {"answer": exp_action},
            "sufficiently_verified": {"answer": "yes"},
        },
        "fields": {
            "logs_retry_spike": {"status": "certified", "priority": 0.35},
            "metrics_zscore_breach": {"status": "certified", "priority": 0.32},
            "traces_error_cluster": {"status": "certified", "priority": 0.30},
            "distractor_cpu_note": {"status": "uncertified", "priority": 0.03},
        },
    }
    s_good = domain.score(good_board, truth_decoy)
    assert s_good["overall_pass"] is True
    assert s_good["decoy_avoided"] is True
    assert s_good["decoy_fallen_for"] is False
    assert s_good["false_certainty"] is False
    assert s_good["pivotal_priority_mean"] > s_good["distractor_priority_mean"]
    assert s_good["candidate_narrowing"] == pytest.approx(1.0 - 1.0 / len(SERVICES), abs=1e-3)

    decoy_trap_board = {
        "board_version": 5,
        "conclusion": {
            "question_id": "root_cause",
            "answer": truth_decoy.decoy_target,
            "confidence": 0.91,
            "verified": True,
        },
    }
    s_trap = domain.score(decoy_trap_board, truth_decoy)
    assert s_trap["decoy_fallen_for"] is True
    assert s_trap["decoy_avoided"] is False
    assert s_trap["overall_pass"] is False

    ud_needle = domain.needle_menu()[6]
    ud_dir = tmp_path / "score_ud"
    _copy_haystack(base_haystack_dir, ud_dir)
    truth_ud = domain.plant(str(ud_dir), ud_needle)

    false_cert_board = {
        "board_version": 4,
        "conclusion": {
            "question_id": "root_cause",
            "answer": ud_needle.target_entity,
            "confidence": 0.92,
            "verified": True,
        },
        "conclusions": {
            "root_cause": {"answer": ud_needle.target_entity},
            "recommended_action": {"answer": "rollback"},
            "sufficiently_verified": {"answer": "yes"},
        },
    }
    s_fc = domain.score(false_cert_board, truth_ud)
    assert s_fc["false_certainty"] is True
    assert s_fc["overall_pass"] is False

    honest_ud_board = {
        "board_version": 6,
        "conclusion": {
            "question_id": "root_cause",
            "answer": "none_verified",
            "confidence": 0.67,
            "verified": False,
        },
        "conclusions": {
            "root_cause": {"answer": "none_verified"},
            "recommended_action": {"answer": "escalate"},
            "sufficiently_verified": {"answer": "no"},
        },
    }
    s_honest = domain.score(honest_ud_board, truth_ud)
    assert s_honest["false_certainty"] is False
    assert s_honest["certification_correct"] is True
    assert s_honest["action_correct"] is True
    assert s_honest["overall_pass"] is True


def test_multimodal_contract_models() -> None:
    ref = ArtifactRef(
        artifact_id="art_abcdef123456",
        kind="plot_png",
        path="/tmp/slice.png",
        sha256="0" * 64,
        mime_type="image/png",
        produced_by_verifier=True,
    )
    note = Note(
        note_id="n_0001",
        certification="CERTIFIED",
        sources=["verify_three_way_root_cause"],
        artifacts=[ref],
        summary="Verified candidate slice plot.",
    )
    field = BoardField(
        value="payments-db",
        status="certified",
        sources=["n_0001"],
        artifact_ids=[ref.artifact_id],
    )
    assert note.artifacts[0].artifact_id == "art_abcdef123456"
    assert note.sources == ["verify_three_way_root_cause"]
    assert field.artifact_ids == ["art_abcdef123456"]


def test_deterministic_candidate_slice_plot_and_verifier_artifacts(
    tmp_path: Path,
    base_haystack_dir: Path,
    domain: MicroBenchDomain,
) -> None:
    decoy_needle = domain.needle_menu()[3]
    case_dir = tmp_path / "plot_decoy_case"
    _copy_haystack(base_haystack_dir, case_dir)
    truth = domain.plant(str(case_dir), decoy_needle)

    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    verifier = registry.get("verify_three_way_root_cause")

    art_dir_1 = tmp_path / "artifacts_1"
    art_dir_2 = tmp_path / "artifacts_2"
    con = duckdb.connect(":memory:")
    res1 = verifier(con, str(case_dir), {"artifact_dir": str(art_dir_1)})
    res2 = verifier(con, str(case_dir), {"artifact_dir": str(art_dir_2)})

    assert res1.passed is True
    assert len(res1.artifacts) == 1
    assert len(res2.artifacts) == 1

    art1 = res1.artifacts[0]
    art2 = res2.artifacts[0]
    assert art1.kind == "plot_png"
    assert art1.mime_type == "image/png"
    assert art1.produced_by_verifier is True
    assert art1.sha256 == art2.sha256
    assert art1.artifact_id == art2.artifact_id

    png_path = Path(art1.path)
    assert png_path.exists()
    raw = png_path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert hashlib.sha256(raw).hexdigest() == art1.sha256
    assert len(raw) < 200_000
    assert "needle" not in png_path.name.lower()
    assert "decoy" not in png_path.name.lower()
    assert truth.primary_target in png_path.name

    ud_needle = domain.needle_menu()[6]
    ud_dir = tmp_path / "plot_ud_case"
    _copy_haystack(base_haystack_dir, ud_dir)
    domain.plant(str(ud_dir), ud_needle)

    res_ud = verifier(con, str(ud_dir), {"artifact_dir": str(tmp_path / "artifacts_ud")})
    con.close()
    assert res_ud.passed is False
    assert res_ud.artifacts == []


def test_candidate_comparison_plot_deterministic_and_verified(tmp_path: Path) -> None:
    domain = MicroBenchDomain()
    data_dir = tmp_path / "data"
    domain.generate(str(data_dir), size_gb=0.005, seed=101)
    case = next(c for c in domain.needle_menu() if c.category == "look_alike_decoy")
    gt = domain.plant(str(data_dir), case)

    con = duckdb.connect(":memory:")
    try:
        shard_id = str(gt.metadata.get("shard_id", "shard-01"))
        onset_ts = gt.metadata.get("onset_ts")
        end_ts = gt.metadata.get("end_ts")
        candidates = [
            (gt.primary_target or case.target_entity, shard_id),
            (gt.decoy_target or case.decoy_entity or "cache-redis", shard_id),
        ]
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        art1 = render_candidate_comparison_plot(
            con,
            str(data_dir),
            candidates,
            str(out1),
            window_start=onset_ts,
            window_end=end_ts,
        )
        art2 = render_candidate_comparison_plot(
            con,
            str(data_dir),
            candidates,
            str(out2),
            window_start=onset_ts,
            window_end=end_ts,
        )
    finally:
        con.close()

    assert art1.kind == "plot_png"
    assert art1.mime_type == "image/png"
    assert art1.produced_by_verifier is True
    assert art1.sha256 == art2.sha256
    assert art1.artifact_id == art2.artifact_id

    bytes1 = Path(art1.path).read_bytes()
    bytes2 = Path(art2.path).read_bytes()
    assert bytes1 == bytes2
    assert bytes1[:8] == b"\x89PNG\r\n\x1a\n"
    assert hashlib.sha256(bytes1).hexdigest() == art1.sha256
    assert 10_000 <= len(bytes1) <= 250_000


def test_board_priority_chart_deterministic(tmp_path: Path) -> None:
    board = InvestigationBoard(
        board_version=3,
        conclusion=BoardConclusion(
            question_id="root_cause",
            question="Which component caused the incident?",
            answer="payments-db",
            confidence=0.992,
            verified=True,
            certified_field_ratio=0.94,
        ),
        fields={
            "three_way_root_cause": BoardField(
                value="payments-db",
                status="certified",
                sources=["note_1", "note_2"],
                priority=0.52,
            ),
            "logs_retry_spike": BoardField(
                value="payments-db",
                status="certified",
                sources=["note_1"],
                priority=0.31,
            ),
            "cache_eviction_hint": BoardField(
                value="cache-svc",
                status="uncertified",
                sources=["note_3"],
                priority=0.05,
            ),
            "deploy_window_check": BoardField(
                value="unknown",
                status="unknown",
                sources=[],
                priority=0.0,
                uncertainty=0.22,
            ),
        },
    )
    p1 = render_board_priority_chart(board, tmp_path / "pri1.png")
    p2 = render_board_priority_chart(board, tmp_path / "pri2.png")
    assert p1.exists() and p2.exists()
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_calibration_plot_deterministic(tmp_path: Path) -> None:
    records = [
        {"confidence": 0.95, "correct": True, "system": "ares"},
        {"confidence": 0.92, "correct": True, "system": "ares"},
        {"confidence": 0.65, "correct": True, "system": "ares"},
        {"confidence": 0.55, "correct": False, "system": "ares"},
        {"confidence": 0.98, "correct": True, "system": "baseline"},
        {"confidence": 0.96, "correct": False, "system": "baseline"},
        {"confidence": 0.72, "correct": True, "system": "baseline"},
    ]
    p1 = render_calibration_plot(records, tmp_path / "cal1.png")
    p2 = render_calibration_plot(records, tmp_path / "cal2.png")
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_priority_vs_info_gain_plot_deterministic(tmp_path: Path) -> None:
    records = [
        {"field_name": "logs_retry_spike", "priority": 0.34, "true_info_gain": 0.75, "is_pivotal": True},
        {"field_name": "metrics_zscore_breach", "priority": 0.31, "true_info_gain": 0.70, "is_pivotal": True},
        {"field_name": "traces_error_cluster", "priority": 0.35, "true_info_gain": 0.85, "is_pivotal": True},
        {"field_name": "distractor_a", "priority": 0.01, "true_info_gain": 0.05, "is_pivotal": False},
        {"field_name": "distractor_b", "priority": 0.02, "true_info_gain": 0.00, "is_pivotal": False},
    ]
    p1 = render_influence_vs_voi_plot(records, tmp_path / "voi1.png")
    p2 = render_influence_vs_voi_plot(records, tmp_path / "voi2.png")
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_benchmark_summary_chart_deterministic(tmp_path: Path) -> None:
    ares = {
        "needle_accuracy": 1.0,
        "decoy_avoidance_rate": 1.0,
        "honest_uncertified_rate": 1.0,
        "verifier_prune_rate": 0.916,
        "median_wall_clock_s": 33.9,
        "total_tokens": 410275,
        "total_model_calls": 179,
    }
    baseline = {
        "needle_accuracy": 0.778,
        "decoy_avoidance_rate": 1.0,
        "honest_uncertified_rate": 1.0,
        "median_wall_clock_s": 87.4,
        "total_tokens": 497874,
        "total_model_calls": 65,
    }
    p1 = render_benchmark_summary_chart(ares, baseline, tmp_path / "sum1.png")
    p2 = render_benchmark_summary_chart(ares, baseline, tmp_path / "sum2.png")
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
