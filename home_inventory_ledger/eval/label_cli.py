"""Fast ~2-minute ground-truth labeler CLI for unseen stranger test clips (PLAN_v2 §3.5).

Usage:
    python -m eval.label_cli eval/labels/t01.json --clip-id t01 --contributor alex
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def build_template_label(clip_id: str, contributor: str) -> dict:
    return {
        "clip_id": clip_id,
        "contributor": contributor,
        "labeled_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "consent_confirmed": True,
        "conditions": {
            "lighting": "indoor_ambient",
            "motion": "handheld_pan",
            "audio": "venue_background",
        },
        "items": [
            {
                "category": "laptop",
                "brand": "Apple",
                "model": "A2992",
                "qty": 1,
                "first_seen_s": 4.0,
                "price_usd": 1999.0,
            }
        ],
        "planted_contradiction": None,
        "followup_clips": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or inspect a pre-run ground-truth label file.")
    parser.add_argument("output", type=Path, help="Path to eval/labels/<clip_id>.json")
    parser.add_argument("--clip-id", required=True, help="Clip identifier (e.g. t01)")
    parser.add_argument("--contributor", default="volunteer", help="First name or handle only")
    parser.add_argument("--interactive", action="store_true", help="Prompt item-by-item in terminal")
    args = parser.parse_args()

    doc = build_template_label(args.clip_id, args.contributor)
    if args.interactive:
        items = []
        print("Enter ground-truth items (blank category to finish):")
        while True:
            cat = input("  Category (e.g. laptop, monitor, headphones): ").strip()
            if not cat:
                break
            brand = input("  Brand (blank if unknown): ").strip() or None
            model = input("  Visible label model # (blank if not legible): ").strip() or None
            qty_s = input("  Quantity [1]: ").strip()
            qty = int(qty_s) if qty_s else 1
            ts_s = input("  First seen seconds [0]: ").strip()
            ts = float(ts_s) if ts_s else 0.0
            pr_s = input("  True replacement price USD (optional): ").strip()
            pr = float(pr_s) if pr_s else None
            items.append(
                {
                    "category": cat,
                    "brand": brand,
                    "model": model,
                    "qty": qty,
                    "first_seen_s": ts,
                    "price_usd": pr,
                }
            )
        doc["items"] = items
        contra_idx = input("Planted contradiction item index (blank if none): ").strip()
        if contra_idx:
            said = input("  What wrong brand/model did narrator say?: ").strip()
            doc["planted_contradiction"] = {"item_index": int(contra_idx), "said": said}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2))
    print(f"Saved ground-truth label file -> {args.output}")
    print("IMPORTANT: Commit this file to git BEFORE running any model on the clip.")


if __name__ == "__main__":
    main()
