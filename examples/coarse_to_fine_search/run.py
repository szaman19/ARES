"""CLI runner demonstrating 2-level coarse-to-fine DuckDB search across 3 scenarios."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ares import ArtifactStore, TaskSpec
from examples.coarse_to_fine_search.machine import (
    CaseType,
    build_machine,
    create_anomaly_dataset,
)

SCENARIOS: tuple[CaseType, ...] = ("true_needle", "decoy_only", "under_determined")


def run_demo() -> None:
    """Execute all three coarse-to-fine search scenarios and print summaries."""
    with tempfile.TemporaryDirectory(prefix="ares_coarse_to_fine_") as tmp:
        root_dir = Path(tmp)
        for case_type in SCENARIOS:
            db_path = root_dir / f"{case_type}.duckdb"
            run_dir = root_dir / f"run_{case_type}"
            create_anomaly_dataset(db_path, case_type=case_type)

            machine = build_machine(db_path)
            result = machine.run(
                TaskSpec(state="coarse_scan", input={"db_path": str(db_path)}),
                run_dir=run_dir,
                max_concurrent=4,
            )
            out = result.output
            probe_tasks = [t for t in result.tasks.values() if t.state == "probe_bucket"]
            verify_tasks = [t for t in result.tasks.values() if t.state == "verify_candidate"]

            print(f"=== Scenario: {case_type} ===")
            print(
                f"Round 1 (coarse): probed {len(probe_tasks)} buckets / "
                f"{out['total_scanned']} entities -> pruned to top {out['round1_survivors']}"
            )
            print(f"Round 2 (fine): ran {len(verify_tasks)} verifiers:")
            for cand in out["checked_candidates"]:
                sigs = cand["signals"]
                print(
                    f"  - {cand['candidate_id']} ({cand['bucket']}): passed={cand['passed']} "
                    f"signals={sigs}"
                )
            print(
                f"Outcome: status={out['status']} target={out['target']} "
                f"rejected_decoys={out['rejected_decoys']}"
            )
            if out["status"] == "verified":
                store = ArtifactStore(Path(result.run_dir) / "artifacts")
                receipt = store.get_json(out["receipt_ref"])
                print(
                    f"Receipt verified in ArtifactStore ({out['receipt_ref']}): "
                    f"target={receipt['candidate_id']}"
                )
            print()


if __name__ == "__main__":
    run_demo()
