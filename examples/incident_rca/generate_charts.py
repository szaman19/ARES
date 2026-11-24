"""Generate all 5 deterministic Incident RCA benchmark & demo charts from saved results."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

import duckdb

from ares.contracts import InvestigationBoard
from examples.incident_rca.artifacts import render_candidate_comparison_plot
from examples.incident_rca.charts import (
    render_benchmark_summary_chart,
    render_board_priority_chart,
    render_calibration_plot,
    render_influence_vs_voi_plot,
)
from examples.incident_rca.domain import MicroBenchDomain, _decoy_shard_for_case

EXAMPLE_DIR = Path(__file__).resolve().parent
RESULTS_JSON = EXAMPLE_DIR / "results" / "micro_bench_stage1.json"
CHARTS_DIR = EXAMPLE_DIR / "results" / "charts"


def _coerce_board(raw_board: dict[str, Any]) -> InvestigationBoard:
    if "conclusion" in raw_board:
        return InvestigationBoard.model_validate(raw_board)
    dec = raw_board.get("decision", {})
    decs = raw_board.get("decisions", {})
    fields = {}
    for fname, fdata in raw_board.get("fields", {}).items():
        fields[fname] = {
            **fdata,
            "priority": fdata.get("priority", fdata.get("influence", 0.0)),
            "uncertainty": fdata.get("uncertainty", fdata.get("voi", 0.0)),
        }
    return InvestigationBoard.model_validate(
        {
            "board_version": raw_board.get("board_version", 0),
            "conclusion": {
                "question_id": dec.get("question_id", "root_cause"),
                "question": dec.get("question", "Which component is the verified root cause?"),
                "answer": dec.get("answer", "unknown"),
                "confidence": dec.get("confidence", 0.0),
                "verified": dec.get("answer", "unknown") not in ("unknown", "none_verified"),
            },
            "conclusions": {
                qid: {
                    "question_id": qid,
                    "question": d.get("question", qid),
                    "answer": d.get("answer", "unknown"),
                    "confidence": d.get("confidence", 0.0),
                }
                for qid, d in decs.items()
            },
            "fields": fields,
            "archived_fields": raw_board.get("archived_fields", []),
        }
    )


def main() -> None:
    report = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Multimodal Candidate Comparison Plot (3/3 needle vs 2/3 decoy)
    decoy_case = next(c for c in cases if c.get("category") == "look_alike_decoy")
    gt = decoy_case["ground_truth"]
    planted_files = gt.get("planted_files", [])
    decoy_shard = _decoy_shard_for_case(decoy_case["case_id"])
    existing_data_dir = Path(planted_files[0]).parent.parent if planted_files else None

    if existing_data_dir is not None and existing_data_dir.exists():
        con = duckdb.connect(":memory:")
        try:
            comp_ref = render_candidate_comparison_plot(
                con,
                str(existing_data_dir),
                [
                    (gt["root_cause_component"], gt["shard_id"]),
                    (gt["decoy_component"], decoy_shard),
                ],
                str(CHARTS_DIR),
                window_start=gt["onset_ts"],
                window_end=gt["end_ts"],
            )
            print(f"[1/5] Comparison plot: {comp_ref.path} (sha256={comp_ref.sha256[:12]})")
        finally:
            con.close()
    else:
        with tempfile.TemporaryDirectory(prefix="ares_chart_decoy_") as tmpdir:
            dom = MicroBenchDomain()
            dom.generate(tmpdir, size_gb=0.005, seed=42)
            spec = next(n for n in dom.needle_menu() if n.case_id == decoy_case["case_id"])
            regen_gt = dom.plant(tmpdir, spec)
            con = duckdb.connect(":memory:")
            try:
                comp_ref = render_candidate_comparison_plot(
                    con,
                    tmpdir,
                    [
                        (regen_gt.primary_target or spec.target_entity, regen_gt.metadata["shard_id"]),
                        (regen_gt.decoy_target or spec.decoy_entity or "cache-redis", decoy_shard),
                    ],
                    str(CHARTS_DIR),
                    window_start=regen_gt.metadata["onset_ts"],
                    window_end=regen_gt.metadata["end_ts"],
                )
                print(f"[1/5] Comparison plot: {comp_ref.path} (sha256={comp_ref.sha256[:12]})")
            finally:
                con.close()

    # 2. Board Priority Chart (for decoy_01_payments_vs_cache)
    board = _coerce_board(decoy_case["ares"]["board"])
    p_board = render_board_priority_chart(board, CHARTS_DIR / "board_influence_decoy_01.png")
    print(f"[2/5] Board priority chart: {p_board}")

    # 3. Calibration Plot (all 9 cases, ARES + Baseline)
    cal_records: list[dict[str, object]] = []
    for c in cases:
        a_score = c["ares"]["score"]
        b_score = c["baseline"]["score"]
        cal_records.append(
            {
                "confidence": float(a_score.get("confidence", 0.0)),
                "correct": bool(a_score.get("overall_pass", False)),
                "system": "ares",
            }
        )
        cal_records.append(
            {
                "confidence": float(b_score.get("confidence", 0.0)),
                "correct": bool(
                    b_score.get("root_cause_correct", False)
                    and b_score.get("action_correct", False)
                ),
                "system": "baseline",
            }
        )
    p_cal = render_calibration_plot(cal_records, CHARTS_DIR / "calibration_stage1.png")
    print(f"[3/5] Calibration plot: {p_cal}")

    # 4. Priority vs. Ground-Truth Info Gain Plot
    inf_records: list[dict[str, object]] = []
    for c in cases:
        pivotal_set = set(c["ground_truth"].get("pivotal_clues", c["ground_truth"].get("pivotal_fields", []))) | {"three_way_root_cause"}
        cand_narrowing = float(c["ares"]["score"].get("candidate_narrowing", 0.0))
        b_fields = c["ares"]["board"].get("fields", {})
        for fname, fdata in b_fields.items():
            is_piv = fname in pivotal_set
            pri_val = float(fdata.get("priority", fdata.get("influence", 0.0)))
            true_gain = cand_narrowing if (is_piv and fdata.get("status") == "certified") else 0.0
            inf_records.append(
                {
                    "field_name": fname,
                    "priority": pri_val,
                    "true_info_gain": true_gain,
                    "is_pivotal": is_piv,
                }
            )
    p_voi = render_influence_vs_voi_plot(inf_records, CHARTS_DIR / "influence_vs_voi_stage1.png")
    print(f"[4/5] Priority vs. info gain plot: {p_voi}")

    # 5. Benchmark Summary Comparison Chart
    summary = report["summary"]
    ares_summary = dict(summary["ares"])
    base_summary = dict(summary["baseline"])
    ares_times = [float(c["ares"]["wall_clock_ms"]) / 1000.0 for c in cases]
    base_times = [float(c["baseline"]["wall_clock_ms"]) / 1000.0 for c in cases]
    ares_summary["median_wall_clock_s"] = median(ares_times)
    base_summary["median_wall_clock_s"] = median(base_times)
    p_sum = render_benchmark_summary_chart(
        ares_summary,
        base_summary,
        CHARTS_DIR / "benchmark_summary_stage1.png",
    )
    print(f"[5/5] Benchmark summary chart: {p_sum}")


if __name__ == "__main__":
    main()
