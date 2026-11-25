"""Offline tests for the DuckDB coarse-to-fine 2-level Spawn example."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from ares import ArtifactStore, TaskSpec
from examples.coarse_to_fine_search.machine import (
    TOP_K,
    build_machine,
    create_anomaly_dataset,
)


def _run_case(tmp_path: Path, case_type: str):
    db_path = tmp_path / f"{case_type}.duckdb"
    run_dir = tmp_path / f"run_{case_type}"
    create_anomaly_dataset(db_path, case_type=case_type)  # type: ignore[arg-type]
    machine = build_machine(db_path)
    result = machine.run(
        TaskSpec(state="coarse_scan", input={"db_path": str(db_path)}),
        run_dir=run_dir,
        max_concurrent=4,
    )
    return result


def test_true_needle_verifies_payments_db_and_rejects_loud_decoy(tmp_path: Path) -> None:
    result = _run_case(tmp_path, "true_needle")

    assert result.status == "done"
    out = result.output
    assert out["status"] == "verified"
    assert out["target"] == "payments-db-02"
    assert "cache-redis-03" in out["rejected_decoys"]
    assert out["signals"] == {
        "signal_error_spike": True,
        "signal_latency_drift": True,
        "signal_lock_trace": True,
    }

    # Verify Round 1 probed all 6 buckets and Round 2 pruned to K=3 survivors.
    probe_tasks = [t for t in result.tasks.values() if t.state == "probe_bucket"]
    verify_tasks = [t for t in result.tasks.values() if t.state == "verify_candidate"]
    assert len(probe_tasks) == 6
    assert len(verify_tasks) == TOP_K
    assert out["total_scanned"] > TOP_K

    # Verify receipt exists in ArtifactStore and matches winner.
    assert result.run_dir is not None
    store = ArtifactStore(Path(result.run_dir) / "artifacts")
    receipt = store.get_json(out["receipt_ref"])
    assert receipt["candidate_id"] == "payments-db-02"
    assert receipt["passed"] is True
    assert receipt["signals"]["signal_lock_trace"] is True

    # Confirm the loud 2-of-3 decoy passed 2 checks but failed the 3rd structural check.
    decoy_report = next(
        c for c in out["checked_candidates"] if c["candidate_id"] == "cache-redis-03"
    )
    assert decoy_report["passed"] is False
    assert decoy_report["signals"]["signal_error_spike"] is True
    assert decoy_report["signals"]["signal_latency_drift"] is True
    assert decoy_report["signals"]["signal_lock_trace"] is False


def test_decoy_only_yields_unverified_without_false_certification(tmp_path: Path) -> None:
    result = _run_case(tmp_path, "decoy_only")

    assert result.status == "done"
    out = result.output
    assert out["status"] == "unverified"
    assert out["target"] is None
    assert "cache-redis-03" in out["rejected_decoys"]
    assert all(not c["passed"] for c in out["checked_candidates"])


def test_under_determined_missing_third_signal_is_unverified(tmp_path: Path) -> None:
    result = _run_case(tmp_path, "under_determined")

    assert result.status == "done"
    out = result.output
    assert out["status"] == "unverified"
    assert out["target"] is None
    assert "payments-db-02" in out["rejected_decoys"]
    assert "cache-redis-03" in out["rejected_decoys"]

    needle_report = next(
        c for c in out["checked_candidates"] if c["candidate_id"] == "payments-db-02"
    )
    assert needle_report["passed"] is False
    assert needle_report["signals"]["signal_error_spike"] is True
    assert needle_report["signals"]["signal_latency_drift"] is True
    assert needle_report["signals"]["signal_lock_trace"] is None


def test_core_ares_contains_zero_duckdb_imports() -> None:
    ares_dir = Path(__file__).resolve().parent.parent / "ares"
    for py_file in sorted(ares_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("duckdb"), f"{py_file} imports duckdb"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("duckdb"), f"{py_file} imports duckdb"

    proc = subprocess.run(
        [sys.executable, "-c", "import sys, ares; assert 'duckdb' not in sys.modules"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
