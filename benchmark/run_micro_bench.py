"""Stage 1 9-Case Micro-Benchmark Runner & Go/No-Go Gate Evaluator (ARES Plan §7).

Generates a base `MicroBenchDomain` haystack once, isolates each of the 9 cases
(`3x multi_clue_join`, `3x look_alike_decoy`, `3x under_determined`), plants the
needle shard, and evaluates:
  a) ARES (`run_investigation_loop`)
  b) Honest Single-Agent SQL Baseline (`GeminiModelProvider.run_carbon_sql_baseline`)
under identical DuckDB read-only views, schema descriptions, decision questions,
and wall-clock budgets.

Writes structured per-case + aggregate comparison results to:
  `benchmark/results/micro_bench_stage1.json`
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ares.duckdb_runner import DuckDBTrustedRunner
from contracts import Domain, GroundTruth, NeedleSpec
from domain.micro_bench import MicroBenchDomain
from fusion.loop import LoopConfig, LoopRunResult, run_investigation_loop
from models.decision_probe import DecisionProbeClient
from models.gemini_provider import GeminiModelProvider

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = REPO_ROOT / "benchmark" / "results" / "micro_bench_stage1.json"
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "benchmark" / "results" / "artifacts"


def build_baseline_board_for_scoring(baseline_res: dict[str, Any]) -> dict[str, Any]:
    """Construct a fair board dict from `run_carbon_sql_baseline` output for `domain.score()`.

    When the baseline claims `sufficiently_verified == 'yes'` and selects a concrete
    `root_cause` component (not `'none_verified'` or `'unknown'`), sets
    `uncertified_influence_share = 0.0` and `unknown_voi_max = 0.0` so `domain.score()`:
    - fairly credits `board_certified = True` when confident & right on verified needles, and
    - fairly flags `false_certainty = True` when claiming `'yes'` on under-determined cases.
    """
    rc = str(baseline_res.get("root_cause", "none_verified"))
    action = str(baseline_res.get("recommended_action", "escalate"))
    suff = str(baseline_res.get("sufficiently_verified", "no")).strip().lower()
    suff_norm = "yes" if suff in ("yes", "true") else "no"
    conf = max(0.0, min(1.0, float(baseline_res.get("confidence", 0.0))))

    claims_verified = suff_norm == "yes" and rc not in ("none_verified", "unknown", "")
    uncert_share = 0.0 if claims_verified else 1.0
    unk_voi = 0.0 if claims_verified else 1.0

    return {
        "board_version": 1,
        "decision": {
            "question_id": "root_cause",
            "question": "Which component is the verified root cause?",
            "answer": rc,
            "confidence": conf,
            "certified_influence_share": 1.0 - uncert_share,
            "uncertified_influence_share": uncert_share,
            "unknown_voi_max": unk_voi,
        },
        "decisions": {
            "root_cause": {
                "question_id": "root_cause",
                "answer": rc,
                "confidence": conf,
            },
            "recommended_action": {
                "question_id": "recommended_action",
                "answer": action,
                "confidence": conf,
            },
            "sufficiently_verified": {
                "question_id": "sufficiently_verified",
                "answer": suff_norm,
                "confidence": conf,
            },
        },
        "fields": {},
    }


def _run_single_baseline(
    domain: Domain,
    case_dir: Path,
    *,
    wall_clock_budget_s: float,
    max_queries: int = 10,
    thinking_level: str = "low",
) -> tuple[dict[str, Any], float]:
    provider = GeminiModelProvider()
    db_runner = DuckDBTrustedRunner(str(case_dir))

    def _sql_executor(sql: str) -> dict[str, Any]:
        qres, _note = db_runner.execute_readonly_query(sql)
        return {
            "columns": qres.columns,
            "rows": qres.rows,
            "row_count": qres.row_count,
            "truncated": qres.truncated,
            "duration_ms": qres.duration_ms,
        }

    t0 = time.perf_counter()
    res = provider.run_carbon_sql_baseline(
        schema_description=domain.schema_description(),
        decision_questions=domain.decision_questions(),
        sql_executor=_sql_executor,
        max_queries=max_queries,
        timeout_seconds=wall_clock_budget_s,
        thinking_level=thinking_level,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return res, elapsed_ms


def _run_single_ares(
    domain: Domain,
    case_dir: Path,
    artifact_dir: Path,
    *,
    wall_clock_budget_s: float,
    max_rounds: int = 3,
) -> LoopRunResult:
    provider = GeminiModelProvider()
    probe = DecisionProbeClient()
    cfg = LoopConfig(
        max_rounds=max_rounds,
        max_concurrent_workers=4,
        wall_clock_budget_s=wall_clock_budget_s,
        artifact_dir=str(artifact_dir),
    )
    return run_investigation_loop(
        domain,
        str(case_dir),
        provider=provider,
        probe_client=probe,
        config=cfg,
    )


def _evaluate_one_case(
    domain: Domain,
    base_haystack_dir: Path,
    work_root: Path,
    artifacts_root: Path,
    needle: NeedleSpec,
    *,
    wall_clock_budget_s: float,
    baseline_max_queries: int,
    baseline_thinking_level: str,
    parallel_runners: bool,
) -> dict[str, Any]:
    case_dir = work_root / needle.case_id
    shutil.copytree(base_haystack_dir, case_dir)
    truth: GroundTruth = domain.plant(str(case_dir), needle)

    case_art_dir = artifacts_root / needle.case_id
    case_art_dir.mkdir(parents=True, exist_ok=True)

    if parallel_runners:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_ares = pool.submit(
                _run_single_ares,
                domain,
                case_dir,
                case_art_dir,
                wall_clock_budget_s=wall_clock_budget_s,
            )
            fut_base = pool.submit(
                _run_single_baseline,
                domain,
                case_dir,
                wall_clock_budget_s=wall_clock_budget_s,
                max_queries=baseline_max_queries,
                thinking_level=baseline_thinking_level,
            )
            ares_res = fut_ares.result()
            base_raw, base_wall_ms = fut_base.result()
    else:
        ares_res = _run_single_ares(
            domain,
            case_dir,
            case_art_dir,
            wall_clock_budget_s=wall_clock_budget_s,
        )
        base_raw, base_wall_ms = _run_single_baseline(
            domain,
            case_dir,
            wall_clock_budget_s=wall_clock_budget_s,
            max_queries=baseline_max_queries,
            thinking_level=baseline_thinking_level,
        )

    ares_score = domain.score(ares_res.board.model_dump(), truth)
    baseline_board = build_baseline_board_for_scoring(base_raw)
    baseline_score = domain.score(baseline_board, truth)

    piv_mean = ares_score.get("pivotal_influence_mean")
    dist_mean = ares_score.get("distractor_influence_mean")
    pivotal_beats_distractor = (
        (piv_mean is not None and dist_mean is not None and piv_mean > dist_mean)
        or (piv_mean is not None and dist_mean is None and piv_mean > 0.0)
    )

    base_tel = base_raw["telemetry"]
    return {
        "case_id": needle.case_id,
        "category": needle.category,
        "needle": needle.model_dump(),
        "ground_truth": truth.model_dump(),
        "ares": {
            "score": ares_score,
            "termination_reason": ares_res.termination_reason,
            "rounds_executed": len(ares_res.rounds),
            "blocking_fields": ares_res.blocking_fields,
            "final_board_drift": ares_res.final_board_drift,
            "invariant_violations_caught": ares_res.invariant_violations_caught,
            "ungrounded_fields_on_board": ares_res.ungrounded_fields_on_board,
            "illegal_cert_upgrades_on_board": ares_res.illegal_cert_upgrades_on_board,
            "pivotal_beats_distractor": pivotal_beats_distractor,
            "sql_queries_executed": ares_res.sql_queries_executed,
            "verifiers_executed": ares_res.verifiers_executed,
            "skipped_duplicates": ares_res.skipped_duplicates,
            "rejected_guard": ares_res.rejected_guard,
            "candidates_pruned_by_verifiers": ares_res.candidates_pruned_by_verifiers,
            "probe_api_calls": ares_res.probe_api_calls,
            "model_telemetry": ares_res.model_telemetry.model_dump(),
            "wall_clock_ms": ares_res.wall_clock_ms,
            "artifact_paths": [a.path for a in ares_res.artifacts],
            "board": ares_res.board.model_dump(),
            "rounds": [r.model_dump() for r in ares_res.rounds],
        },
        "baseline": {
            "score": baseline_score,
            "root_cause": base_raw["root_cause"],
            "recommended_action": base_raw["recommended_action"],
            "sufficiently_verified": base_raw["sufficiently_verified"],
            "confidence": base_raw["confidence"],
            "rationale": base_raw["rationale"],
            "query_count": base_raw["query_count"],
            "queries_executed": base_raw["queries_executed"],
            "model_telemetry": base_tel.model_dump(),
            "wall_clock_ms": base_wall_ms,
            "board": baseline_board,
        },
    }


def _summarize_results(
    case_records: list[dict[str, Any]],
    total_wall_s: float,
    domain_name: str,
) -> dict[str, Any]:
    n = len(case_records)
    decoy_cases = [c for c in case_records if c["category"] == "look_alike_decoy"]
    under_cases = [c for c in case_records if c["category"] == "under_determined"]

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    ares_passes = sum(1 for c in case_records if c["ares"]["score"]["overall_pass"])
    base_passes = sum(1 for c in case_records if c["baseline"]["score"]["overall_pass"])

    ares_rc_ok = sum(1 for c in case_records if c["ares"]["score"]["root_cause_correct"])
    base_rc_ok = sum(1 for c in case_records if c["baseline"]["score"]["root_cause_correct"])

    ares_act_ok = sum(1 for c in case_records if c["ares"]["score"]["action_correct"])
    base_act_ok = sum(1 for c in case_records if c["baseline"]["score"]["action_correct"])

    ares_cert_ok = sum(1 for c in case_records if c["ares"]["score"]["certification_correct"])
    base_cert_ok = sum(1 for c in case_records if c["baseline"]["score"]["certification_correct"])

    ares_decoy_ok = sum(
        1 for c in decoy_cases if c["ares"]["score"]["decoy_avoided"] is True
    )
    base_decoy_ok = sum(
        1 for c in decoy_cases if c["baseline"]["score"]["decoy_avoided"] is True
    )

    ares_false_cert = sum(
        1 for c in case_records if c["ares"]["score"]["false_certainty"]
    )
    base_false_cert = sum(
        1 for c in case_records if c["baseline"]["score"]["false_certainty"]
    )

    total_ungrounded = sum(c["ares"]["ungrounded_fields_on_board"] for c in case_records)
    total_illegal_upgrades = sum(
        c["ares"]["illegal_cert_upgrades_on_board"] for c in case_records
    )
    total_caught = sum(c["ares"]["invariant_violations_caught"] for c in case_records)
    max_drift = max((c["ares"]["final_board_drift"] for c in case_records), default=0.0)

    # Pivotal > distractor check across verifiable cases (and any case with non-zero influence)
    verifiable_cases = [c for c in case_records if c["category"] != "under_determined"]
    piv_gt_dist_verifiable = sum(
        1 for c in verifiable_cases if c["ares"]["pivotal_beats_distractor"]
    )
    piv_means = [
        float(c["ares"]["score"]["pivotal_influence_mean"])
        for c in verifiable_cases
        if c["ares"]["score"]["pivotal_influence_mean"] is not None
    ]
    dist_means = [
        float(c["ares"]["score"]["distractor_influence_mean"])
        for c in verifiable_cases
        if c["ares"]["score"]["distractor_influence_mean"] is not None
    ]

    zero_invariant_violations = total_ungrounded == 0 and total_illegal_upgrades == 0
    pivotal_dominates = (
        len(verifiable_cases) > 0
        and piv_gt_dist_verifiable == len(verifiable_cases)
    )
    gate_passed = (
        zero_invariant_violations
        and max_drift == 0.0
        and pivotal_dominates
        and ares_false_cert == 0
        and ares_decoy_ok == len(decoy_cases)
        and ares_passes >= max(7, base_passes)
    )

    return {
        "benchmark": "stage1_micro_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "domain": domain_name,
        "num_cases": n,
        "total_wall_clock_s": round(total_wall_s, 3),
        "gate_1_go_no_go": {
            "gate_passed": gate_passed,
            "zero_invariant_violations": zero_invariant_violations,
            "total_ungrounded_fields_on_board": total_ungrounded,
            "total_illegal_cert_upgrades_on_board": total_illegal_upgrades,
            "total_invariant_violations_caught_by_guard": total_caught,
            "max_final_board_drift": max_drift,
            "verifiable_pivotal_beats_distractor_count": piv_gt_dist_verifiable,
            "verifiable_cases_count": len(verifiable_cases),
            "mean_pivotal_influence_verifiable": _mean(piv_means),
            "mean_distractor_influence_verifiable": _mean(dist_means),
            "ares_overall_pass_count": ares_passes,
            "baseline_overall_pass_count": base_passes,
            "ares_decoy_avoided_count": ares_decoy_ok,
            "baseline_decoy_avoided_count": base_decoy_ok,
            "decoy_cases_count": len(decoy_cases),
            "ares_false_certainty_count": ares_false_cert,
            "baseline_false_certainty_count": base_false_cert,
            "under_determined_cases_count": len(under_cases),
        },
        "summary": {
            "ares": {
                "overall_pass_count": ares_passes,
                "overall_pass_rate": round(ares_passes / max(1, n), 4),
                "root_cause_accuracy": round(ares_rc_ok / max(1, n), 4),
                "action_accuracy": round(ares_act_ok / max(1, n), 4),
                "certification_accuracy": round(ares_cert_ok / max(1, n), 4),
                "decoy_avoidance_rate": round(ares_decoy_ok / max(1, len(decoy_cases)), 4),
                "false_certainty_count": ares_false_cert,
                "mean_wall_clock_ms": _mean([c["ares"]["wall_clock_ms"] for c in case_records]),
                "total_tokens": sum(
                    c["ares"]["model_telemetry"]["total_tokens"] for c in case_records
                ),
                "total_model_calls": sum(
                    c["ares"]["model_telemetry"]["model_calls"] for c in case_records
                ),
                "total_probe_api_calls": sum(
                    c["ares"]["probe_api_calls"] for c in case_records
                ),
                "total_sql_queries_executed": sum(
                    c["ares"]["sql_queries_executed"] for c in case_records
                ),
                "total_verifiers_executed": sum(
                    c["ares"]["verifiers_executed"] for c in case_records
                ),
                "mean_candidates_pruned": _mean(
                    [float(c["ares"]["candidates_pruned_by_verifiers"]) for c in case_records]
                ),
                "mean_final_board_drift": _mean(
                    [c["ares"]["final_board_drift"] for c in case_records]
                ),
            },
            "baseline": {
                "overall_pass_count": base_passes,
                "overall_pass_rate": round(base_passes / max(1, n), 4),
                "root_cause_accuracy": round(base_rc_ok / max(1, n), 4),
                "action_accuracy": round(base_act_ok / max(1, n), 4),
                "certification_accuracy": round(base_cert_ok / max(1, n), 4),
                "decoy_avoidance_rate": round(base_decoy_ok / max(1, len(decoy_cases)), 4),
                "false_certainty_count": base_false_cert,
                "mean_wall_clock_ms": _mean(
                    [c["baseline"]["wall_clock_ms"] for c in case_records]
                ),
                "total_tokens": sum(
                    c["baseline"]["model_telemetry"]["total_tokens"] for c in case_records
                ),
                "total_model_calls": sum(
                    c["baseline"]["model_telemetry"]["model_calls"] for c in case_records
                ),
                "total_sql_queries_executed": sum(
                    c["baseline"]["query_count"] for c in case_records
                ),
            },
        },
        "cases": case_records,
    }


def _print_scoreboard(report: dict[str, Any]) -> None:
    print("\n" + "=" * 112)
    print("STAGE 1 MICRO-BENCHMARK SCOREBOARD (9 CASES: ARES vs. SINGLE-AGENT CARBON SQL BASELINE)")
    print("=" * 112)
    header = (
        f"{'Case ID':<28} | {'Category':<16} | {'ARES RC (Act)':<26} | "
        f"{'ARES Pass':<9} | {'Base RC (Act)':<26} | {'Base Pass':<9}"
    )
    print(header)
    print("-" * 112)
    for c in report["cases"]:
        a_sc = c["ares"]["score"]
        b_sc = c["baseline"]["score"]
        a_str = f"{a_sc['predicted_root_cause']} ({a_sc['predicted_action']})"
        b_str = f"{b_sc['predicted_root_cause']} ({b_sc['predicted_action']})"
        a_pass = "PASS" if a_sc["overall_pass"] else "FAIL"
        b_pass = "PASS" if b_sc["overall_pass"] else "FAIL"
        print(
            f"{c['case_id']:<28} | {c['category']:<16} | {a_str:<26} | "
            f"{a_pass:<9} | {b_str:<26} | {b_pass:<9}"
        )
    print("-" * 112)
    g = report["gate_1_go_no_go"]
    sa = report["summary"]["ares"]
    sb = report["summary"]["baseline"]
    print(
        f"ARES Overall Pass:     {sa['overall_pass_count']}/{report['num_cases']} "
        f"({sa['overall_pass_rate']*100:.1f}%) | Action Acc: {sa['action_accuracy']*100:.1f}% | "
        f"Decoy Avoided: {g['ares_decoy_avoided_count']}/{g['decoy_cases_count']} | "
        f"False Certainty: {sa['false_certainty_count']}"
    )
    print(
        f"Baseline Overall Pass: {sb['overall_pass_count']}/{report['num_cases']} "
        f"({sb['overall_pass_rate']*100:.1f}%) | Action Acc: {sb['action_accuracy']*100:.1f}% | "
        f"Decoy Avoided: {g['baseline_decoy_avoided_count']}/{g['decoy_cases_count']} | "
        f"False Certainty: {sb['false_certainty_count']}"
    )
    print(
        f"Invariants Audit:      ungrounded={g['total_ungrounded_fields_on_board']}, "
        f"illegal_cert_upgrades={g['total_illegal_cert_upgrades_on_board']}, "
        f"max_drift={g['max_final_board_drift']:.4f}, "
        f"pivotal>distractor={g['verifiable_pivotal_beats_distractor_count']}/{g['verifiable_cases_count']} "
        f"(mean {g['mean_pivotal_influence_verifiable']:.3f} vs {g['mean_distractor_influence_verifiable']:.3f})"
    )
    print(
        f"Total Wall Clock:      {report['total_wall_clock_s']:.1f}s | "
        f"GATE 1 VERDICT: {'GO (PASSED)' if g['gate_passed'] else 'REVIEW'}"
    )
    print("=" * 112 + "\n")


def run_stage1_micro_benchmark(
    *,
    out_json_path: Path = DEFAULT_RESULTS_PATH,
    artifacts_root: Path = DEFAULT_ARTIFACTS_ROOT,
    size_gb: float = 0.02,
    seed: int = 42,
    wall_clock_budget_s: float = 180.0,
    baseline_max_queries: int = 10,
    baseline_thinking_level: str = "low",
    parallel_runners: bool = True,
) -> dict[str, Any]:
    """Execute the full 9-case Stage 1 Micro-Benchmark (ARES vs. Baseline)."""
    load_dotenv()
    t_total_start = time.perf_counter()

    domain = MicroBenchDomain()
    needles = domain.needle_menu()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    work_parent = REPO_ROOT / "data"
    work_parent.mkdir(parents=True, exist_ok=True)

    case_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stage1_bench_", dir=work_parent) as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        base_dir = tmp_root / "base_haystack"
        cases_work_dir = tmp_root / "cases"
        cases_work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Stage 1] Generating base haystack (~{size_gb} GB, seed={seed})...", flush=True)
        domain.generate(str(base_dir), size_gb=size_gb, seed=seed)

        for idx, needle in enumerate(needles, start=1):
            print(
                f"[Stage 1] ({idx}/{len(needles)}) Running {needle.case_id} [{needle.category}]...",
                flush=True,
            )
            t_case = time.perf_counter()
            record = _evaluate_one_case(
                domain=domain,
                base_haystack_dir=base_dir,
                work_root=cases_work_dir,
                artifacts_root=artifacts_root,
                needle=needle,
                wall_clock_budget_s=wall_clock_budget_s,
                baseline_max_queries=baseline_max_queries,
                baseline_thinking_level=baseline_thinking_level,
                parallel_runners=parallel_runners,
            )
            case_records.append(record)
            a_sc = record["ares"]["score"]
            b_sc = record["baseline"]["score"]
            print(
                f"          -> Done in {time.perf_counter() - t_case:.1f}s | "
                f"ARES: rc={a_sc['predicted_root_cause']} act={a_sc['predicted_action']} "
                f"pass={a_sc['overall_pass']} ({record['ares']['termination_reason']}) | "
                f"Baseline: rc={b_sc['predicted_root_cause']} act={b_sc['predicted_action']} "
                f"pass={b_sc['overall_pass']}",
                flush=True,
            )

    total_wall_s = time.perf_counter() - t_total_start
    report = _summarize_results(case_records, total_wall_s, domain.name)
    out_json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    _print_scoreboard(report)
    print(f"[Stage 1] Saved detailed JSON results to: {out_json_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 9-Case ARES vs. Carbon SQL Micro-Benchmark."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Output JSON path (default: benchmark/results/micro_bench_stage1.json).",
    )
    parser.add_argument(
        "--size-gb",
        type=float,
        default=0.02,
        help="Background haystack scale in GB (default: 0.02 -> ~186K rows).",
    )
    parser.add_argument(
        "--budget-s",
        type=float,
        default=180.0,
        help="Per-case wall-clock budget in seconds for both ARES and Baseline.",
    )
    parser.add_argument(
        "--baseline-max-queries",
        type=int,
        default=10,
        help="Max SQL queries per case for the Carbon baseline (default: 10).",
    )
    parser.add_argument(
        "--baseline-thinking",
        type=str,
        default="low",
        choices=["low", "medium", "high"],
        help="Thinking level for Carbon SQL baseline (default: low).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run ARES and Baseline sequentially per case instead of concurrently.",
    )
    args = parser.parse_args()

    run_stage1_micro_benchmark(
        out_json_path=args.out,
        size_gb=args.size_gb,
        wall_clock_budget_s=args.budget_s,
        baseline_max_queries=args.baseline_max_queries,
        baseline_thinking_level=args.baseline_thinking,
        parallel_runners=not args.sequential,
    )


if __name__ == "__main__":
    main()
