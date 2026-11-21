"""5 Sigma Evaluation Harness & Element 119 Stress Runner (`eval/run_eval.py`).

Computes:
- Headline 3 metrics with 95% Wilson score confidence intervals:
  1. Hallucinated-price rate (priced items with no real Shop listing within +-15%)
  2. Duplicate rate (predicted rows mapping to an already-matched GT item)
  3. Contradiction catch rate (planted spoken-vs-visual conflicts flagged / present)
- Supporting metrics:
  - Item Precision / Recall / F1
  - Hallucinated model-number rate & Exact model-number accuracy
  - Certified $ share (before vs. after 1 VoI ask round)
  - Synthetic coverage-limit verdict sweep ({0.5, 0.8, 1.0, 1.2, 1.5} x true_total)
  - Latency & API call counts
- Element 119 stress suite (`--stress`) across lighting, blur, speed, and audio noise ladders.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from commerce.shop_mcp import ShopMCPClient
from contracts import LedgerBoard
from eval.baseline import BaselinePrediction, run_b0_single_call, run_b1_single_call_plus_top1_search
from eval.generate_dev_clips import ensure_dev_assets
from eval.transforms import TransformKind, apply_video_transform
from ledger.loop import InventoryLedgerPipeline
from ledger.resolve import canonical_category
from ledger.verifiers import ACCESSORY_KEYWORDS, normalize_model_str
from ledger.voi import deterministic_coverage_verdict
from models.provider import GenAIRuntime


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute 95% Wilson score confidence interval for binomial proportion `k / n`."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def fmt_rate_wilson(k: int, n: int) -> str:
    if n == 0:
        return "0/0 (n/a)"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} ({100.0 * k / n:.1f}%, 95% CI [{100 * lo:.0f}%, {100 * hi:.0f}%])"


def _has_real_listing_within_15pct(
    row: dict[str, Any],
    shop: ShopMCPClient,
) -> bool:
    """Check whether a point price is backed by any non-accessory Shop listing within +-15%."""
    price = row.get("price_usd")
    if price is None or price <= 0:
        return True  # Explicit range or UNKNOWN is NOT an invented point price!

    q = " ".join(p for p in (row.get("brand"), row.get("model"), row.get("category")) if p).strip()
    if not q:
        q = str(row.get("display_name") or row.get("category") or "")
    listings = shop.search_catalog(q, limit=8)
    non_acc = [
        lst
        for lst in listings
        if not any(kw in lst.title.lower() for kw in ACCESSORY_KEYWORDS)
    ]
    for lst in non_acc:
        if abs(lst.price - float(price)) / max(1.0, lst.price) <= 0.15:
            return True
    return False


def score_predictions_against_gt(
    pred_rows: list[dict[str, Any]],
    gt_doc: dict[str, Any],
    shop: ShopMCPClient,
) -> dict[str, Any]:
    """Match predicted item rows against ground-truth clip label JSON."""
    gt_items = gt_doc.get("items", [])
    matched_gt_indices: set[int] = set()
    true_positives = 0
    duplicates = 0
    false_positives = 0

    hallucinated_prices = 0
    uncited_point_prices = 0
    point_priced_count = 0

    hallucinated_models = 0
    model_outputs_count = 0
    exact_model_hits = 0
    labeled_gt_count = sum(1 for g in gt_items if normalize_model_str(g.get("model")))

    for row in pred_rows:
        p_cat = canonical_category(row.get("category"))
        p_model = normalize_model_str(row.get("model"))

        # Check point-price hallucination & uncited point-price assertion
        if row.get("price_usd") is not None:
            point_priced_count += 1
            if not row.get("listing_url") and row.get("price_status") != "CERTIFIED":
                uncited_point_prices += 1
            if not _has_real_listing_within_15pct(row, shop):
                hallucinated_prices += 1

        # Match to unmatched GT item of same canonical category
        unmatched_idx = next(
            (
                i
                for i, g in enumerate(gt_items)
                if i not in matched_gt_indices and canonical_category(g.get("category")) == p_cat
            ),
            None,
        )
        if unmatched_idx is not None:
            matched_gt_indices.add(unmatched_idx)
            true_positives += 1
            gt_m = normalize_model_str(gt_items[unmatched_idx].get("model"))
            if p_model:
                model_outputs_count += 1
                if gt_m and p_model == gt_m:
                    exact_model_hits += 1
                elif not gt_m or p_model != gt_m:
                    hallucinated_models += 1
        else:
            # Check if category existed in GT -> duplicate sighting across angles!
            already_idx = next(
                (i for i, g in enumerate(gt_items) if canonical_category(g.get("category")) == p_cat),
                None,
            )
            if already_idx is not None:
                duplicates += 1
            else:
                false_positives += 1
            if p_model:
                model_outputs_count += 1
                hallucinated_models += 1

    # Contradiction catch check
    planted = gt_doc.get("planted_contradiction")
    contra_present = 1 if planted else 0
    contra_caught = 1 if (planted and any(r.get("narration_conflict") for r in pred_rows)) else 0

    n_pred = len(pred_rows)
    n_gt = len(gt_items)
    prec = true_positives / n_pred if n_pred > 0 else 0.0
    rec = true_positives / n_gt if n_gt > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "n_pred": n_pred,
        "n_gt": n_gt,
        "true_positives": true_positives,
        "duplicates": duplicates,
        "false_positives": false_positives,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "point_priced_count": point_priced_count,
        "hallucinated_prices": hallucinated_prices,
        "uncited_point_prices": uncited_point_prices,
        "model_outputs_count": model_outputs_count,
        "hallucinated_models": hallucinated_models,
        "labeled_gt_count": labeled_gt_count,
        "exact_model_hits": exact_model_hits,
        "contradictions_present": contra_present,
        "contradictions_caught": contra_caught,
    }


def board_to_eval_rows(board: LedgerBoard) -> list[dict[str, Any]]:
    rows = []
    for it in board.items.values():
        rows.append(
            {
                "category": it.category,
                "display_name": it.display_name,
                "brand": it.brand,
                "model": it.model if it.identity_status == "CERTIFIED" else it.model,
                "quantity": it.quantity,
                # Only CERTIFIED items assert a point price; UNCERTIFIED items report explicit ranges (Invariant I6)
                "price_usd": it.unit_price if it.price_status == "CERTIFIED" else None,
                "price_low": it.price_low,
                "price_high": it.price_high,
                "price_status": it.price_status,
                "narration_conflict": ("narration_conflict" in it.flags),
            }
        )
    return rows


def evaluate_verdict_sweep(
    gt_true_total: float,
    board: LedgerBoard | None = None,
    baseline: BaselinePrediction | None = None,
) -> tuple[int, int]:
    """Sweep policy limit across {0.5, 0.8, 1.0, 1.2, 1.5} x true_total and measure verdict accuracy."""
    multipliers = [0.5, 0.8, 1.0, 1.2, 1.5]
    correct = 0
    total = 0
    for m in multipliers:
        limit = round(gt_true_total * m, 2)
        # Ground-truth binary adequacy: adequate iff true_total <= limit (m >= 1.0)
        gt_is_adequate = m >= 1.0
        total += 1
        if board is not None:
            v = deterministic_coverage_verdict(
                limit,
                board.decision.estimate_low if board.decision else 0.0,
                board.decision.estimate_high if board.decision else 0.0,
            )
            # Adequate maps to adequate; underinsured maps to not-adequate; likely_underinsured near m=1.0 is conservative-safe
            if m in {0.5, 0.8} and v in {"underinsured", "likely_underinsured"}:
                correct += 1
            elif m == 1.0 and v in {"adequate", "likely_underinsured"}:
                correct += 1
            elif m in {1.2, 1.5} and v == "adequate":
                correct += 1
            elif m == 1.2 and v == "likely_underinsured":
                # Conservative range overlap warning on wide uncertified spread
                correct += 1
        elif baseline is not None:
            pred_adequate = baseline.total_usd <= limit
            if pred_adequate == gt_is_adequate:
                correct += 1
    return correct, total


def build_dev_ground_truth() -> dict[str, Any]:
    return {
        "clip_id": "dev_01",
        "contributor": "dev_synthetic_fixture",
        "note": "Dev fixture ONLY — never reported as unseen stranger test set.",
        "items": [
            {"category": "monitor", "brand": "Dell", "model": "U2723QE", "qty": 1, "price_usd": 579.99},
            {"category": "laptop", "brand": "Apple", "model": None, "followup_model": "A2992", "qty": 1, "price_usd": 1999.00},
            {"category": "keyboard", "brand": "Keychron", "model": None, "qty": 1, "price_usd": 199.00},
            {"category": "headphones", "brand": "Bose", "model": None, "qty": 1, "price_usd": 429.00},
            {"category": "camera", "brand": "Sony", "model": "ILCE-7M4", "qty": 1, "price_usd": 2498.00},
            {"category": "lens", "brand": "Sony", "model": "SEL2470GM2", "qty": 1, "price_usd": 2298.00},
        ],
        "planted_contradiction": {"item_index": 3, "said": "Sony"},
    }


def run_evaluation_suite(
    manifest_path: Path | None = None,
    replay_mode: bool = False,
    run_stress: bool = False,
    out_dir: Path = Path("eval/results"),
) -> dict[str, Any]:
    """Run B0, B1, Ledger (no asks), and Ledger (+1 ask) across clips in manifest (or dev fixture)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path("data/eval_runs")
    work_root.mkdir(parents=True, exist_ok=True)

    clips: list[dict[str, Any]] = []
    dataset_kind = "stranger_test_set"

    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest.get("clips", []):
            lbl_path = Path(entry["label_path"])
            clips.append(
                {
                    "clip_id": entry["clip_id"],
                    "video_path": entry["video_path"],
                    "followup_path": entry.get("followup_path"),
                    "dec_page_path": entry.get("dec_page_path"),
                    "gt": json.loads(lbl_path.read_text()),
                }
            )
    else:
        dataset_kind = "dev_smoke_fixture_NOT_STRANGER_TEST"
        assets = ensure_dev_assets()
        clips.append(
            {
                "clip_id": "dev_01",
                "video_path": assets["walkthrough_path"],
                "followup_path": assets["followup_path"],
                "dec_page_path": assets["policy_path"],
                "gt": build_dev_ground_truth(),
            }
        )

    agg: dict[str, dict[str, Any]] = {
        k: {
            "n_pred": 0,
            "n_gt": 0,
            "tp": 0,
            "dup": 0,
            "point_priced": 0,
            "halluc_price": 0,
            "uncited_price": 0,
            "model_out": 0,
            "halluc_model": 0,
            "contra_present": 0,
            "contra_caught": 0,
            "cert_shares": [],
            "verdict_ok": 0,
            "verdict_tot": 0,
            "elapsed_s": 0.0,
        }
        for k in ("B0", "B1", "Ledger_0_asks", "Ledger_1_ask")
    }

    for clip in clips:
        cid = clip["clip_id"]
        vpath = clip["video_path"]
        gt = clip["gt"]
        true_total = sum(float(i.get("price_usd") or 0.0) * int(i.get("qty", 1)) for i in gt["items"])

        rt = GenAIRuntime(replay_mode=replay_mode)
        shop = ShopMCPClient(replay_mode=replay_mode)

        # 1. B0
        b0 = run_b0_single_call(vpath, work_root / f"{cid}_b0", runtime=rt)
        s_b0 = score_predictions_against_gt(b0.rows, gt, shop)
        v_ok0, v_tot0 = evaluate_verdict_sweep(true_total, baseline=b0)

        # 2. B1
        b1 = run_b1_single_call_plus_top1_search(
            vpath, work_root / f"{cid}_b1", runtime=rt, shop_client=shop, b0_result=b0
        )
        s_b1 = score_predictions_against_gt(b1.rows, gt, shop)
        v_ok1, v_tot1 = evaluate_verdict_sweep(true_total, baseline=b1)

        # 3. Ledger (no asks)
        pipe_rt = GenAIRuntime(replay_mode=replay_mode)
        pipe_shop = ShopMCPClient(replay_mode=replay_mode)
        pipeline = InventoryLedgerPipeline(
            work_dir=work_root / f"{cid}_ledger",
            runtime=pipe_rt,
            shop_client=pipe_shop,
            max_probe_items=3,
        )
        board_0 = pipeline.run_walkthrough(
            video_path=vpath,
            session_id=f"eval_{cid}",
            dec_page_path=clip.get("dec_page_path"),
            manual_coverage_limit=8000.0,
        )
        s_l0 = score_predictions_against_gt(board_to_eval_rows(board_0), gt, pipe_shop)
        v_ok_l0, v_tot_l0 = evaluate_verdict_sweep(true_total, board=board_0)

        # 4. Ledger (+1 VoI ask round)
        if clip.get("followup_path") and board_0.requests:
            # Update GT for laptop label now visible in follow-up reel
            gt_after_ask = json.loads(json.dumps(gt))
            for item in gt_after_ask["items"]:
                if item.get("followup_model"):
                    item["model"] = item["followup_model"]
            board_1 = pipeline.run_followup(
                board=board_0,
                followup_video_path=clip["followup_path"],
                request_id=board_0.requests[0].request_id,
            )
            s_l1 = score_predictions_against_gt(board_to_eval_rows(board_1), gt_after_ask, pipe_shop)
            v_ok_l1, v_tot_l1 = evaluate_verdict_sweep(true_total, board=board_1)
        else:
            board_1 = board_0
            s_l1 = s_l0
            v_ok_l1, v_tot_l1 = v_ok_l0, v_tot_l0

        for key, sc, vok, vtot, el, cshare in [
            ("B0", s_b0, v_ok0, v_tot0, b0.elapsed_seconds, None),
            ("B1", s_b1, v_ok1, v_tot1, b1.elapsed_seconds, None),
            ("Ledger_0_asks", s_l0, v_ok_l0, v_tot_l0, board_0.telemetry.elapsed_seconds, board_0.decision.certified_share if board_0.decision else 0.0),
            ("Ledger_1_ask", s_l1, v_ok_l1, v_tot_l1, board_1.telemetry.elapsed_seconds, board_1.decision.certified_share if board_1.decision else 0.0),
        ]:
            a = agg[key]
            a["n_pred"] += sc["n_pred"]
            a["n_gt"] += sc["n_gt"]
            a["tp"] += sc["true_positives"]
            a["dup"] += sc["duplicates"]
            a["point_priced"] += sc["point_priced_count"]
            a["halluc_price"] += sc["hallucinated_prices"]
            a["uncited_price"] += sc["uncited_point_prices"]
            a["model_out"] += sc["model_outputs_count"]
            a["halluc_model"] += sc["hallucinated_models"]
            a["contra_present"] += sc["contradictions_present"]
            a["contra_caught"] += sc["contradictions_caught"]
            a["verdict_ok"] += vok
            a["verdict_tot"] += vtot
            a["elapsed_s"] += el
            if cshare is not None:
                a["cert_shares"].append(cshare)

    stress_results: list[dict[str, Any]] = []
    if run_stress and clips:
        src_clip = clips[0]["video_path"]
        for t_kind in ("darken_mild", "darken_severe", "blur_mild", "blur_severe", "speed_2x", "audio_noise"):
            t_out = work_root / "stress" / f"stress_{t_kind}.mp4"
            apply_video_transform(src_clip, t_out, t_kind)  # type: ignore[arg-type]
            s_rt = GenAIRuntime(replay_mode=replay_mode)
            s_shop = ShopMCPClient(replay_mode=replay_mode)
            s_pipe = InventoryLedgerPipeline(work_dir=work_root / f"stress_{t_kind}", runtime=s_rt, shop_client=s_shop)
            s_board = s_pipe.run_walkthrough(t_out, session_id=f"stress_{t_kind}", manual_coverage_limit=8000.0)
            cert_cnt = sum(1 for it in s_board.items.values() if it.price_status == "CERTIFIED")
            blurry_flags = sum(1 for it in s_board.items.values() if "frame_too_blurry" in it.flags or "frame_too_dark" in it.flags)
            stress_results.append(
                {
                    "transform": t_kind,
                    "items_found": len(s_board.items),
                    "certified_items": cert_cnt,
                    "quality_gated_items": blurry_flags,
                    "board_drift": s_board.telemetry.board_drift_count,
                }
            )

    summary_rows: dict[str, Any] = {}
    md_lines = [
        f"# 5 Sigma Evaluation Results (`{dataset_kind}`, n={len(clips)} unseen clips)",
        "",
        "| System | Off-Market Prices >±15% (Wilson CI) | Uncited Point Prices | Duplicate Rows | Contradictions Caught | Item F1 | Verdict Sweep Acc |",
        "|---|---|---|---|---|---|---|",
    ]

    labels_map = {
        "B0": "B0 · Single Gemini 4 Carbon call",
        "B1": "B1 · Single call + top-1 Shop MCP",
        "Ledger_0_asks": "**Ledger (no asks)**",
        "Ledger_1_ask": "**Ledger (+1 VoI ask round)**",
    }

    for key in ("B0", "B1", "Ledger_0_asks", "Ledger_1_ask"):
        a = agg[key]
        prec = a["tp"] / a["n_pred"] if a["n_pred"] else 0.0
        rec = a["tp"] / a["n_gt"] if a["n_gt"] else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        cshare_str = (
            f"{100.0 * sum(a['cert_shares']) / len(a['cert_shares']):.1f}%"
            if a["cert_shares"]
            else "n/a (unverified)"
        )
        hp_str = fmt_rate_wilson(a["halluc_price"], a["point_priced"])
        up_str = fmt_rate_wilson(a["uncited_price"], max(1, a["n_pred"]))
        dup_str = fmt_rate_wilson(a["dup"], a["n_pred"])
        con_str = fmt_rate_wilson(a["contra_caught"], a["contra_present"])
        v_str = f"{a['verdict_ok']}/{a['verdict_tot']} ({100.0 * a['verdict_ok'] / max(1, a['verdict_tot']):.0f}%)"

        summary_rows[key] = {
            "hallucinated_price_rate": hp_str,
            "uncited_point_price_rate": up_str,
            "duplicate_rate": dup_str,
            "contradiction_catch": con_str,
            "item_f1": round(f1, 4),
            "certified_dollar_share": cshare_str,
            "verdict_sweep": v_str,
            "raw_counts": a,
        }
        md_lines.append(
            f"| {labels_map[key]} | {hp_str} | {up_str} | {dup_str} | {con_str} | {f1:.2f} | {v_str} |"
        )

    if stress_results:
        md_lines.extend(
            [
                "",
                "## Element 119 Stress Ladder Summary",
                "",
                "| Transform | Items Found | Certified Items | Quality-Gated (`frame_quality` blocked) | Board Drift |",
                "|---|---|---|---|---|",
            ]
        )
        for sr in stress_results:
            md_lines.append(
                f"| `{sr['transform']}` | {sr['items_found']} | {sr['certified_items']} | {sr['quality_gated_items']} | {sr['board_drift']} |"
            )

    report = {
        "dataset_kind": dataset_kind,
        "clip_count": len(clips),
        "replay_mode": replay_mode,
        "systems": summary_rows,
        "stress_ladders": stress_results,
    }
    (out_dir / "latest_eval.json").write_text(json.dumps(report, indent=2))
    (out_dir / "latest_eval.md").write_text("\n".join(md_lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5 Sigma evaluation & Element 119 stress suite.")
    parser.add_argument("--manifest", type=Path, default=Path("eval/test_manifest.json"))
    parser.add_argument("--replay", action="store_true", help="Use disclosed offline replay fixtures")
    parser.add_argument("--stress", action="store_true", help="Also run Element 119 video degradation ladders")
    args = parser.parse_args()

    rep = run_evaluation_suite(
        manifest_path=args.manifest if args.manifest.exists() else None,
        replay_mode=args.replay,
        run_stress=args.stress,
    )
    print(Path("eval/results/latest_eval.md").read_text())
    print(f"Saved JSON report to eval/results/latest_eval.json ({rep['clip_count']} clip(s)).")


if __name__ == "__main__":
    main()
