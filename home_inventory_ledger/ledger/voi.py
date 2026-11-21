"""Counterfactual influence (Total Variation distance) and Value-of-Information capture requests (PLAN §12)."""

from __future__ import annotations

from contracts import CaptureRequest, LedgerBoard, LedgerItem, VerdictLabel


ELECTRONICS_CATEGORIES = {
    "laptop",
    "television",
    "monitor",
    "camera",
    "lens",
    "headphones",
    "phone",
    "keyboard",
    "vacuum",
    "tablet",
    "console",
    "speaker",
    "router",
    "printer",
}

APPRAISAL_CATEGORIES = {"antique", "art", "painting", "jewelry", "watch", "sculpture", "rug"}

CATEGORY_FALLBACK_SPREAD_USD: dict[str, float] = {
    "laptop": 1400.0,
    "television": 1200.0,
    "camera": 1500.0,
    "lens": 1100.0,
    "sofa": 1200.0,
    "guitar": 900.0,
    "phone": 650.0,
    "tablet": 550.0,
    "monitor": 450.0,
    "headphones": 240.0,
    "vacuum": 350.0,
    "chair": 350.0,
    "keyboard": 140.0,
}


def deterministic_coverage_verdict(
    coverage_limit: float | None,
    estimate_low: float,
    estimate_high: float,
    has_items: bool = True,
) -> VerdictLabel:
    """Deterministic sanity-check rule (PLAN §12.1)."""
    if coverage_limit is None or not has_items:
        return "insufficient_evidence"
    if estimate_high <= coverage_limit:
        return "adequate"
    if estimate_low > coverage_limit:
        return "underinsured"
    return "likely_underinsured"


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """Compute Total Variation distance `0.5 * sum_k |P(k) - Q(k)|` in [0, 1]."""
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0
    diff_sum = sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys)
    return min(1.0, max(0.0, 0.5 * diff_sum))


def rank_capture_requests(
    board: LedgerBoard,
    influences: dict[str, float],
    max_requests: int = 3,
) -> tuple[dict[str, LedgerItem], list[CaptureRequest]]:
    """Compute per-item `voi_usd` and generate concrete 'show me X' capture requests."""
    updated_items: dict[str, LedgerItem] = {}
    candidates: list[CaptureRequest] = []

    raw_sum = sum(influences.values())
    norm_influences = {
        k: (v / raw_sum if raw_sum > 1e-6 else v) for k, v in influences.items()
    }

    limit = board.policy.coverage_c_limit or 10_000.0

    for iid, item in board.items.items():
        inf = float(influences.get(iid, 0.0))
        norm_inf = float(norm_influences.get(iid, 0.0))

        if item.price_status == "CERTIFIED":
            updated_items[iid] = item.model_copy(
                update={"influence": round(norm_inf, 4), "voi_usd": 0.0}
            )
            continue

        if item.price_status == "UNCERTIFIED":
            lo = float(item.price_low or 0.0)
            hi = float(item.price_high or lo)
            width = max(0.0, (hi - lo) * item.quantity)
        else:
            # UNKNOWN item: use category-realistic spread proxy rather than over-weighting small peripherals
            cat_spread = CATEGORY_FALLBACK_SPREAD_USD.get(item.category, min(600.0, 0.08 * limit))
            width = cat_spread * item.quantity

        # Effective swing weight combines Decision-1 counterfactual TV swing + proximity to coverage threshold
        eff_weight = max(inf, 0.25 if width >= 250 else 0.10)
        voi_usd = round(0.5 * width * (1.0 + eff_weight), 2)

        updated_items[iid] = item.model_copy(
            update={"influence": round(norm_inf, 4), "voi_usd": voi_usd}
        )

        if item.category in APPRAISAL_CATEGORIES:
            continue

        # Map evidence gap to concrete user capture instruction
        if "frame_too_blurry" in item.flags or "frame_too_dark" in item.flags:
            gap_kind = "blurry_frame"
            answer_mode = "video"
            instruction = (
                f"Hold steady for 3 seconds with better light on the {item.display_name} "
                f"so its model label is legible."
            )
        elif item.category in {"sofa", "couch", "chair", "table", "desk", "bed"}:
            gap_kind = "material_ambiguity"
            answer_mode = "voice"
            instruction = (
                f"What brand and material (e.g., solid wood, leather, fabric) is the {item.display_name}?"
            )
        elif item.quantity > 1 and "quantity_uncertain" in item.flags:
            gap_kind = "quantity_ambiguity"
            answer_mode = "video"
            instruction = (
                f"Pan slowly across the {item.room} so all {item.display_name} units are visible in one sweep."
            )
        else:
            gap_kind = "missing_label"
            answer_mode = "video"
            where = "underneath" if item.category in {"laptop", "keyboard"} else "on the back/side of"
            instruction = f"Show me the model label {where} the {item.display_name}"

        candidates.append(
            CaptureRequest(
                request_id=f"ask_{iid}_v{board.board_version}",
                item_id=iid,
                item_name=item.display_name,
                room=item.room,
                gap_kind=gap_kind,  # type: ignore[arg-type]
                instruction=instruction,
                voi_usd=voi_usd,
                uncertainty_width_usd=round(width, 2),
                influence=round(norm_inf, 4),
                answer_mode=answer_mode,  # type: ignore[arg-type]
            )
        )

    # Also add missing room / unopened closet bonus requests
    for idx, space in enumerate(board.rooms_missing[:2]):
        space_voi = round(0.12 * limit, 2)
        candidates.append(
            CaptureRequest(
                request_id=f"ask_room_{idx}_v{board.board_version}",
                item_id=None,
                item_name=space,
                room=space,
                gap_kind="missing_room",
                instruction=f"Open and briefly film inside {space} (no items captured there yet).",
                voi_usd=space_voi,
                uncertainty_width_usd=round(space_voi * 2, 2),
                influence=0.15,
                answer_mode="video",
            )
        )

    candidates.sort(key=lambda r: (r.voi_usd, r.uncertainty_width_usd), reverse=True)
    return updated_items, candidates[:max_requests]
