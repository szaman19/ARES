"""Tests for deterministic chart & visual artifact suite (`domain/artifacts.py` + `viz/charts.py`)."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import duckdb

from contracts import BoardDecision, BoardField, InvestigationBoard
from domain.artifacts import render_candidate_comparison_plot
from domain.micro_bench import MicroBenchDomain
from viz.charts import (
    render_benchmark_summary_chart,
    render_board_influence_chart,
    render_calibration_plot,
    render_influence_vs_voi_plot,
)


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


def test_no_global_pyplot_imports() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    _assert_no_pyplot_import(repo_root / "domain" / "artifacts.py")
    _assert_no_pyplot_import(repo_root / "viz" / "charts.py")


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


def test_board_influence_chart_deterministic(tmp_path: Path) -> None:
    board = InvestigationBoard(
        board_version=3,
        decision=BoardDecision(
            question_id="root_cause",
            question="Which component caused the incident?",
            answer="payments-db",
            confidence=0.992,
            certified_influence_share=0.94,
            uncertified_influence_share=0.06,
            unknown_voi_max=0.22,
        ),
        fields={
            "three_way_root_cause": BoardField(
                value="payments-db",
                status="certified",
                sources=["note_1", "note_2"],
                influence=0.52,
            ),
            "logs_retry_spike": BoardField(
                value="payments-db",
                status="certified",
                sources=["note_1"],
                influence=0.31,
            ),
            "cache_eviction_hint": BoardField(
                value="cache-svc",
                status="uncertified",
                sources=["note_3"],
                influence=0.05,
            ),
            "deploy_window_check": BoardField(
                value="unknown",
                status="unknown",
                sources=[],
                influence=0.0,
                voi=0.22,
            ),
        },
    )
    p1 = render_board_influence_chart(board, tmp_path / "inf1.png")
    p2 = render_board_influence_chart(board, tmp_path / "inf2.png")
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


def test_influence_vs_voi_plot_deterministic(tmp_path: Path) -> None:
    records = [
        {"field_name": "logs_retry_spike", "influence": 0.34, "true_info_gain": 0.75, "is_pivotal": True},
        {"field_name": "metrics_zscore_breach", "influence": 0.31, "true_info_gain": 0.70, "is_pivotal": True},
        {"field_name": "traces_error_cluster", "influence": 0.35, "true_info_gain": 0.85, "is_pivotal": True},
        {"field_name": "distractor_a", "influence": 0.01, "true_info_gain": 0.05, "is_pivotal": False},
        {"field_name": "distractor_b", "influence": 0.02, "true_info_gain": 0.00, "is_pivotal": False},
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
