"""CLI runner for the live NYC DOT street-camera ARES search example."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.nyc_cameras.domain import run_live_nyc_search


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 2-round ARES parallel street-camera search on NYC DOT cameras."
    )
    parser.add_argument(
        "--area",
        default="Manhattan",
        help="NYC borough filter (Manhattan, Brooklyn, Queens, Bronx, Staten Island, or all).",
    )
    parser.add_argument(
        "--query",
        default="open curb space and light traffic (no heavy congestion or double-parked blockage)",
        help="Plain-English live search target.",
    )
    parser.add_argument(
        "--keyword",
        default=None,
        help="Optional street/avenue substring filter (e.g. 'Broadway', '5 Ave').",
    )
    parser.add_argument("--round1-k", type=int, default=6, help="Parallel cameras in Round 1.")
    parser.add_argument("--round2-k", type=int, default=3, help="Nearest neighbors in Round 2.")
    parser.add_argument(
        "--out-dir",
        default="examples/nyc_cameras/runs/nyc_live",
        help="Output directory for live JPEGs, JSON result, and contact sheet PNG.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default=None,
        help="Optional local snapshot directory (e.g. examples/nyc_cameras/fixtures/nyc_snapshot).",
    )
    args = parser.parse_args()

    res = run_live_nyc_search(
        query=args.query,
        area=args.area,
        keyword=args.keyword,
        round1_k=args.round1_k,
        round2_k=args.round2_k,
        out_dir=args.out_dir,
        snapshot_dir=args.snapshot_dir,
    )

    out_json = Path(args.out_dir) / "nyc_live_result.json"
    out_json.write_text(
        json.dumps(res.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"\n=== ARES Live NYC Camera Search ({res.area}) ===\n"
        f"Query:   {res.query}\n"
        f"Pruned:  {res.total_online_cameras_in_area} online cameras -> "
        f"only {res.cameras_inspected_count} live JPEGs fetched in {res.wall_clock_s:.1f}s\n"
        f"Verdict: {res.board.conclusion.answer} "
        f"(confidence={res.board.conclusion.confidence:.1%}, "
        f"certified_ratio={res.board.conclusion.certified_field_ratio:.1%})\n"
    )
    print("Camera Findings & Priority:")
    for item in res.inspected:
        pri = res.evaluation.field_priorities.get(item.field_name, 0.0)
        dist = f" ({int(item.distance_m_from_anchor)}m from anchor)" if item.distance_m_from_anchor else ""
        print(
            f"  [R{item.round_num}] {item.camera.name:30s}{dist:20s} | "
            f"{item.observation.verdict:14s} score={item.observation.match_score:.2f} | "
            f"priority={pri:.3f} | {item.observation.summary}"
        )
    print(f"\nContact sheet PNG: {res.contact_sheet_path}")
    print(f"JSON result:       {out_json}")


if __name__ == "__main__":
    main()
