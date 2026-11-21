"""Standalone 9-case Real NASA TESS benchmark runner (`ExoplanetTriageDomain` vs. Carbon SQL Baseline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.run_micro_bench import run_stage1_micro_benchmark
from domain.exoplanet_triage import ExoplanetTriageDomain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 9-Case Real NASA TESS ARES vs. Carbon Baseline Benchmark."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "benchmark" / "results" / "exoplanet_tess_stage1.json",
    )
    parser.add_argument("--size-gb", type=float, default=0.02)
    parser.add_argument("--budget-s", type=float, default=180.0)
    parser.add_argument("--baseline-max-queries", type=int, default=10)
    parser.add_argument(
        "--baseline-thinking", type=str, default="low", choices=["low", "medium", "high"]
    )
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    domain = ExoplanetTriageDomain()
    art_root = REPO_ROOT / "benchmark" / "results" / "exoplanet_artifacts"
    run_stage1_micro_benchmark(
        domain=domain,
        out_json_path=args.out,
        artifacts_root=art_root,
        size_gb=args.size_gb,
        wall_clock_budget_s=args.budget_s,
        baseline_max_queries=args.baseline_max_queries,
        baseline_thinking_level=args.baseline_thinking,
        parallel_runners=not args.sequential,
    )


if __name__ == "__main__":
    main()
