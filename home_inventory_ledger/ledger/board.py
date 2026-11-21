"""Versioned, cited, invariant-checked LedgerBoard (PLAN §11 Invariants I1-I7).

Invariants enforced on every `apply_patch`:
- I1 Provenance: Every non-UNKNOWN item field cites >=1 existing `note_id`; every CERTIFIED price
  cites a `Listing` in `listings_index` AND a passing `catalog_exact_match` CERTIFIED note.
- I2 Weakest link: Item status cannot exceed the certification of its supporting notes for that
  dimension. Models cannot upgrade `UNCERTIFIED` -> `CERTIFIED` without a verifier note.
- I3 Notes are the truth: Notes are immutable and append-only. `rebuild_board(notes, ...)`
  reconstructs the board deterministically with zero drift (`compute_board_drift == 0`).
- I4 Context vs verification: Enforced structurally in `verifiers.py` (never imports `LedgerBoard`).
- I5 Staleness: Every note and patch records `requested_under`; stale arrivals are logged.
- I6 No invented prices: `unit_price` is strictly None unless `price_status == "CERTIFIED"`.
  UNCERTIFIED items carry `(price_low, price_high)` ranges only.
- I7 Bounded probe board: `build_probe_board_view` caps the Decision-1 input to top-M items
  and rolls minor items into `misc_total`.
"""

from __future__ import annotations

from typing import Any

from contracts import (
    BoardPatch,
    Cert,
    ItemPatch,
    LedgerBoard,
    LedgerItem,
    Listing,
    Note,
    PolicyInfo,
)


class InvariantViolationError(ValueError):
    """Raised when a proposed `BoardPatch` violates any of Invariants I1-I7."""


CERT_RANK: dict[Cert, int] = {"UNKNOWN": 0, "UNCERTIFIED": 1, "CERTIFIED": 2}


def append_notes_and_listings(
    board: LedgerBoard,
    new_notes: list[Note],
    new_listings: list[Listing] | None = None,
) -> LedgerBoard:
    """Append immutable notes and register candidate listings on the board."""
    existing_note_ids = {n.note_id for n in board.notes}
    updated_notes = list(board.notes)
    stale_inc = 0

    for note in new_notes:
        if note.note_id in existing_note_ids:
            raise InvariantViolationError(f"I3 violation: duplicate/mutated note_id '{note.note_id}'")
        if note.requested_under < max(0, board.board_version - 1):
            stale_inc += 1
        updated_notes.append(note)
        existing_note_ids.add(note.note_id)

    updated_listings = dict(board.listings_index)
    for lst in new_listings or []:
        updated_listings[lst.listing_id] = lst

    telemetry = board.telemetry.model_copy(
        update={"stale_notes_logged": board.telemetry.stale_notes_logged + stale_inc}
    )
    return board.model_copy(
        update={
            "notes": updated_notes,
            "listings_index": updated_listings,
            "telemetry": telemetry,
        }
    )


def _validate_item_patch(
    patch: ItemPatch,
    existing: LedgerItem | None,
    notes_by_id: dict[str, Note],
    listings_index: dict[str, Listing],
) -> None:
    all_cited_ids = list(
        dict.fromkeys((existing.sources if existing else []) + list(patch.cite_note_ids))
    )

    # I1: Every cited note must exist and belong to this item
    for nid in all_cited_ids:
        if nid not in notes_by_id:
            raise InvariantViolationError(f"I1 violation: cited note '{nid}' does not exist")
        if notes_by_id[nid].item_id != patch.item_id:
            raise InvariantViolationError(
                f"I1 violation: note '{nid}' belongs to '{notes_by_id[nid].item_id}', not '{patch.item_id}'"
            )

    eff_id_status = patch.identity_status or (existing.identity_status if existing else "UNKNOWN")
    eff_pr_status = patch.price_status or (existing.price_status if existing else "UNKNOWN")
    eff_unit_price = patch.unit_price if patch.unit_price is not None else (existing.unit_price if existing else None)
    eff_low = patch.price_low if patch.price_low is not None else (existing.price_low if existing else None)
    eff_high = patch.price_high if patch.price_high is not None else (existing.price_high if existing else None)

    # If transitioning price_status from UNCERTIFIED -> CERTIFIED, we clear ranges below;
    # if price_status != CERTIFIED, I6 forbids unit_price!
    if eff_pr_status != "CERTIFIED" and patch.unit_price is not None:
        raise InvariantViolationError(
            f"I6 violation on '{patch.item_id}': unit_price ({patch.unit_price}) cannot be set when price_status={eff_pr_status}"
        )

    if eff_id_status != "UNKNOWN" or eff_pr_status != "UNKNOWN":
        if not all_cited_ids:
            raise InvariantViolationError(
                f"I1 violation on '{patch.item_id}': non-UNKNOWN status requires >=1 cited note"
            )

    cited_notes = [notes_by_id[nid] for nid in all_cited_ids]

    # I2: Identity status cannot exceed certification of cited identity notes
    if eff_id_status == "CERTIFIED":
        id_cert_notes = [
            n for n in cited_notes if n.kind == "identity" and n.certification == "CERTIFIED" and n.verifier
        ]
        if not id_cert_notes:
            raise InvariantViolationError(
                f"I2 violation on '{patch.item_id}': CERTIFIED identity requires a CERTIFIED verifier note"
            )

    # I1 + I2 + I6: CERTIFIED price requires a passing `catalog_exact_match` CERTIFIED note + valid Listing
    if eff_pr_status == "CERTIFIED":
        pr_cert_notes = [
            n
            for n in cited_notes
            if n.kind == "price"
            and n.certification == "CERTIFIED"
            and n.verifier == "catalog_exact_match"
            and len(n.listing_ids) >= 1
        ]
        if not pr_cert_notes:
            raise InvariantViolationError(
                f"I1/I2 violation on '{patch.item_id}': CERTIFIED price requires a CERTIFIED 'catalog_exact_match' note citing a listing"
            )
        latest_pr_note = pr_cert_notes[-1]
        lid = latest_pr_note.listing_ids[0]
        if lid not in listings_index:
            raise InvariantViolationError(
                f"I1 violation on '{patch.item_id}': CERTIFIED note cites unknown listing '{lid}'"
            )
        listing = listings_index[lid]
        if eff_unit_price is None or abs(eff_unit_price - listing.price) > 0.01:
            raise InvariantViolationError(
                f"I6 violation on '{patch.item_id}': CERTIFIED unit_price ({eff_unit_price}) must equal matched listing price ({listing.price})"
            )

    if eff_pr_status == "UNCERTIFIED":
        pr_notes = [n for n in cited_notes if n.kind == "price"]
        if not pr_notes:
            raise InvariantViolationError(
                f"I1 violation on '{patch.item_id}': UNCERTIFIED price requires a price note"
            )
        if eff_low is None or eff_high is None or eff_low > eff_high or eff_low < 0:
            raise InvariantViolationError(
                f"I6 violation on '{patch.item_id}': UNCERTIFIED price requires valid (price_low <= price_high)"
            )


def apply_patch(board: LedgerBoard, patch: BoardPatch) -> LedgerBoard:
    """Validate `patch` against Invariants I1-I7 and return a new versioned `LedgerBoard`."""
    notes_by_id = {n.note_id: n for n in board.notes}
    stale_inc = 1 if patch.requested_under < max(0, board.board_version - 1) else 0

    new_items = {k: v.model_copy(deep=True) for k, v in board.items.items()}

    try:
        for ip in patch.item_patches:
            existing = new_items.get(ip.item_id)
            _validate_item_patch(ip, existing, notes_by_id, board.listings_index)

            merged_sources = list(
                dict.fromkeys((existing.sources if existing else []) + list(ip.cite_note_ids))
            )
            merged_flags = list(
                dict.fromkeys((existing.flags if existing else []) + list(ip.add_flags))
            )

            eff_pr_status = ip.price_status or (existing.price_status if existing else "UNKNOWN")
            if eff_pr_status == "CERTIFIED":
                unit_price = ip.unit_price if ip.unit_price is not None else (existing.unit_price if existing else None)
                price_low = None
                price_high = None
            elif eff_pr_status == "UNCERTIFIED":
                unit_price = None
                price_low = ip.price_low if ip.price_low is not None else (existing.price_low if existing else None)
                price_high = ip.price_high if ip.price_high is not None else (existing.price_high if existing else None)
            else:
                unit_price = None
                price_low = None
                price_high = None

            if existing is None:
                new_items[ip.item_id] = LedgerItem(
                    item_id=ip.item_id,
                    room=ip.room or "general",
                    category=ip.category or "unknown",
                    display_name=ip.display_name or ip.category or ip.item_id,
                    brand=ip.brand,
                    model=ip.model,
                    description=ip.description or "",
                    quantity=ip.quantity or 1,
                    identity_status=ip.identity_status or "UNKNOWN",
                    price_status=eff_pr_status,
                    unit_price=unit_price,
                    price_low=price_low,
                    price_high=price_high,
                    matched_listing=ip.matched_listing,
                    candidate_listings=list(ip.candidate_listings or []),
                    best_frame_id=ip.best_frame_id,
                    best_frame_path=ip.best_frame_path,
                    first_seen_ms=ip.first_seen_ms or 0,
                    spoken_claim_text=ip.spoken_claim_text,
                    conflict_detail=ip.conflict_detail,
                    sources=merged_sources,
                    flags=merged_flags,
                )
            else:
                new_items[ip.item_id] = existing.model_copy(
                    update={
                        "room": ip.room if ip.room is not None else existing.room,
                        "category": ip.category if ip.category is not None else existing.category,
                        "display_name": ip.display_name if ip.display_name is not None else existing.display_name,
                        "brand": ip.brand if ip.brand is not None else existing.brand,
                        "model": ip.model if ip.model is not None else existing.model,
                        "description": ip.description if ip.description is not None else existing.description,
                        "quantity": ip.quantity if ip.quantity is not None else existing.quantity,
                        "identity_status": ip.identity_status if ip.identity_status is not None else existing.identity_status,
                        "price_status": eff_pr_status,
                        "unit_price": unit_price,
                        "price_low": price_low,
                        "price_high": price_high,
                        "matched_listing": ip.matched_listing if ip.matched_listing is not None else existing.matched_listing,
                        "candidate_listings": list(ip.candidate_listings) if ip.candidate_listings is not None else existing.candidate_listings,
                        "best_frame_id": ip.best_frame_id if ip.best_frame_id is not None else existing.best_frame_id,
                        "best_frame_path": ip.best_frame_path if ip.best_frame_path is not None else existing.best_frame_path,
                        "first_seen_ms": ip.first_seen_ms if ip.first_seen_ms is not None else existing.first_seen_ms,
                        "spoken_claim_text": ip.spoken_claim_text if ip.spoken_claim_text is not None else existing.spoken_claim_text,
                        "conflict_detail": ip.conflict_detail if ip.conflict_detail is not None else existing.conflict_detail,
                        "sources": merged_sources,
                        "flags": merged_flags,
                    }
                )
    except InvariantViolationError:
        board.telemetry.rejected_patches += 1
        raise

    rooms_seen = patch.rooms_seen if patch.rooms_seen is not None else board.rooms_seen
    rooms_missing = patch.rooms_missing if patch.rooms_missing is not None else board.rooms_missing
    telemetry = board.telemetry.model_copy(
        update={"stale_notes_logged": board.telemetry.stale_notes_logged + stale_inc}
    )
    return board.model_copy(
        update={
            "board_version": board.board_version + 1,
            "items": new_items,
            "rooms_seen": rooms_seen,
            "rooms_missing": rooms_missing,
            "telemetry": telemetry,
        }
    )


def merge_items_weakest_link(item_a: LedgerItem, item_b: LedgerItem) -> LedgerItem:
    """Merge two `LedgerItem`s under Invariant I2 (union of sources, weakest certification)."""
    inv_rank: dict[int, Cert] = {0: "UNKNOWN", 1: "UNCERTIFIED", 2: "CERTIFIED"}
    id_cert = inv_rank[min(CERT_RANK[item_a.identity_status], CERT_RANK[item_b.identity_status])]
    pr_cert = inv_rank[min(CERT_RANK[item_a.price_status], CERT_RANK[item_b.price_status])]

    if pr_cert == "CERTIFIED":
        unit_price = item_a.unit_price if item_a.unit_price is not None else item_b.unit_price
        low, high = None, None
    elif pr_cert == "UNCERTIFIED":
        unit_price = None
        lows = [x for x in (item_a.price_low, item_b.price_low, item_a.unit_price, item_b.unit_price) if x is not None]
        highs = [x for x in (item_a.price_high, item_b.price_high, item_a.unit_price, item_b.unit_price) if x is not None]
        low = min(lows) if lows else 0.0
        high = max(highs) if highs else low
    else:
        unit_price, low, high = None, None, None

    return item_a.model_copy(
        update={
            "identity_status": id_cert,
            "price_status": pr_cert,
            "unit_price": unit_price,
            "price_low": low,
            "price_high": high,
            "sources": list(dict.fromkeys(item_a.sources + item_b.sources)),
            "flags": list(dict.fromkeys(item_a.flags + item_b.flags)),
        }
    )


def rebuild_board(
    notes: list[Note],
    listings_index: dict[str, Listing],
    policy: PolicyInfo | None = None,
    session_id: str = "rebuilt",
) -> LedgerBoard:
    """Deterministically reconstruct `LedgerBoard` strictly from the append-only `notes` log (Invariant I3)."""
    board = LedgerBoard(
        session_id=session_id,
        board_version=0,
        policy=policy or PolicyInfo(),
        notes=list(notes),
        listings_index=dict(listings_index),
    )

    items: dict[str, LedgerItem] = {}
    rooms_seen: list[str] = []
    rooms_missing: list[str] = []

    for note in notes:
        if note.kind == "coverage":
            for r in note.value.get("rooms_seen", []):
                if r not in rooms_seen:
                    rooms_seen.append(r)
            rooms_missing = list(note.value.get("rooms_missing", rooms_missing))
            continue

        iid = note.item_id
        cur = items.get(iid) or LedgerItem(
            item_id=iid,
            room=str(note.value.get("room", "general")),
            category=str(note.value.get("category", "unknown")),
            display_name=str(note.value.get("display_name") or note.value.get("category") or iid),
        )
        sources = list(dict.fromkeys(cur.sources + [note.note_id]))
        flags = list(cur.flags)
        for fl in note.value.get("flags", []):
            if fl not in flags:
                flags.append(fl)

        updates: dict[str, Any] = {"sources": sources, "flags": flags}
        for field in ("room", "category", "display_name", "brand", "model", "description", "best_frame_id", "best_frame_path", "first_seen_ms", "spoken_claim_text"):
            if note.value.get(field) is not None:
                updates[field] = note.value[field]

        if note.kind == "identity":
            updates["identity_status"] = note.certification
        elif note.kind == "quantity":
            if note.value.get("quantity") is not None:
                updates["quantity"] = int(note.value["quantity"])
        elif note.kind == "contradiction":
            if "narration_conflict" not in flags:
                flags.append("narration_conflict")
            updates["conflict_detail"] = note.summary
        elif note.kind == "price":
            if (
                note.certification == "CERTIFIED"
                and note.verifier == "catalog_exact_match"
                and note.listing_ids
                and note.listing_ids[0] in listings_index
            ):
                lst = listings_index[note.listing_ids[0]]
                updates["price_status"] = "CERTIFIED"
                updates["unit_price"] = lst.price
                updates["price_low"] = None
                updates["price_high"] = None
                updates["matched_listing"] = lst
            elif "price_low" in note.value and "price_high" in note.value:
                # Only apply UNCERTIFIED range if item isn't already CERTIFIED, or if explicitly overriding
                updates["price_status"] = "UNCERTIFIED"
                updates["unit_price"] = None
                updates["price_low"] = float(note.value["price_low"])
                updates["price_high"] = float(note.value["price_high"])
            cands = [listings_index[lid] for lid in note.listing_ids if lid in listings_index]
            if cands:
                updates["candidate_listings"] = cands

        items[iid] = cur.model_copy(update=updates)

    return board.model_copy(
        update={
            "items": items,
            "rooms_seen": rooms_seen,
            "rooms_missing": rooms_missing,
        }
    )


def compute_board_drift(patched: LedgerBoard, rebuilt: LedgerBoard) -> int:
    """Count item-level discrepancies between incrementally patched board and rebuilt-from-notes board (must be 0)."""
    drift = 0
    all_ids = set(patched.items.keys()) | set(rebuilt.items.keys())
    for iid in all_ids:
        a = patched.items.get(iid)
        b = rebuilt.items.get(iid)
        if a is None or b is None:
            drift += 1
            continue
        for attr in ("identity_status", "price_status", "unit_price", "price_low", "price_high", "quantity"):
            if getattr(a, attr) != getattr(b, attr):
                drift += 1
                break
    return drift


def compute_board_totals(board: LedgerBoard) -> dict[str, float]:
    """Compute split totals: never blend CERTIFIED and UNCERTIFIED without explicit range bounds."""
    certified_total = 0.0
    uncert_low = 0.0
    uncert_high = 0.0
    uncert_mid = 0.0
    unknown_count = 0

    for item in board.items.values():
        q = item.quantity
        if item.price_status == "CERTIFIED" and item.unit_price is not None:
            certified_total += item.unit_price * q
        elif item.price_status == "UNCERTIFIED":
            lo = float(item.price_low or 0.0)
            hi = float(item.price_high or lo)
            uncert_low += lo * q
            uncert_high += hi * q
            uncert_mid += 0.5 * (lo + hi) * q
        else:
            unknown_count += 1

    point_est = certified_total + uncert_mid
    cert_share = (certified_total / point_est) if point_est > 0 else 0.0
    return {
        "certified_total": round(certified_total, 2),
        "estimate_low": round(certified_total + uncert_low, 2),
        "estimate_high": round(certified_total + uncert_high, 2),
        "point_estimate": round(point_est, 2),
        "certified_share": round(cert_share, 4),
        "unknown_count": float(unknown_count),
    }


def build_probe_board_view(
    board: LedgerBoard,
    max_items: int = 10,
    overrides: dict[str, str | float] | None = None,
) -> str:
    """Serialize a bounded board view (Invariant I7) for Gemini Decision-1.

    - Caps item rows at `max_items` (sorted by uncertainty width / value), rolling remaining items
      into a single `misc_total` line.
    - When `overrides` specifies `{item_id: "unknown"}` or `{item_id: float_price}`, produces
      a single counterfactual board variant for Decision-1 probing.
    """
    overrides = overrides or {}

    def _item_sort_key(it: LedgerItem) -> float:
        if it.price_status == "CERTIFIED" and it.unit_price is not None:
            return it.unit_price * it.quantity
        lo = float(it.price_low or 0.0)
        hi = float(it.price_high or 0.0)
        return max((hi - lo) * it.quantity * 1.5, hi * it.quantity)

    sorted_items = sorted(board.items.values(), key=_item_sort_key, reverse=True)
    pivotal = sorted_items[:max_items]
    remainder = sorted_items[max_items:]

    lines: list[str] = [
        "=== HOME INVENTORY LEDGER BOARD ===",
        f"Policy Coverage C Limit: {f'${board.policy.coverage_c_limit:,.2f}' if board.policy.coverage_c_limit else 'unknown'}",
        f"Policy Deductible: {f'${board.policy.deductible:,.2f}' if board.policy.deductible is not None else 'unknown'}",
        f"Rooms Seen: {', '.join(board.rooms_seen) if board.rooms_seen else 'none'}",
        f"Rooms / Spaces Missing: {', '.join(board.rooms_missing) if board.rooms_missing else 'none'}",
        "",
        "Pivotal Items:",
    ]

    cert_sum = 0.0
    lo_sum = 0.0
    hi_sum = 0.0
    ablated_count = 0

    for it in pivotal:
        q = it.quantity
        ov = overrides.get(it.item_id)
        flags_str = f" [FLAGS: {', '.join(it.flags)}]" if it.flags else ""
        if ov == "unknown":
            val_str = "unknown"
            status_str = "UNKNOWN (ablated)"
            ablated_count += 1
        elif isinstance(ov, (int, float)):
            p = float(ov)
            val_str = f"${p * q:,.2f} (counterfactual unit=${p:,.2f} x{q})"
            status_str = "COUNTERFACTUAL_POINT"
            lo_sum += p * q
            hi_sum += p * q
        elif it.price_status == "CERTIFIED" and it.unit_price is not None:
            tot = it.unit_price * q
            val_str = f"${tot:,.2f} (unit=${it.unit_price:,.2f} x{q})"
            status_str = "CERTIFIED"
            cert_sum += tot
            lo_sum += tot
            hi_sum += tot
        elif it.price_status == "UNCERTIFIED":
            lo = float(it.price_low or 0.0) * q
            hi = float(it.price_high or lo) * q
            val_str = f"${lo:,.2f} - ${hi:,.2f} (qty={q})"
            status_str = "UNCERTIFIED_RANGE"
            lo_sum += lo
            hi_sum += hi
        else:
            val_str = "unknown"
            status_str = "UNKNOWN"
            ablated_count += 1

        lines.append(
            f"- [{it.item_id}] {it.display_name} ({it.category}, room={it.room}) | status={status_str} | replacement={val_str}{flags_str}"
        )

    if remainder:
        misc_lo = 0.0
        misc_hi = 0.0
        for it in remainder:
            q = it.quantity
            if it.price_status == "CERTIFIED" and it.unit_price is not None:
                misc_lo += it.unit_price * q
                misc_hi += it.unit_price * q
                cert_sum += it.unit_price * q
            elif it.price_status == "UNCERTIFIED":
                misc_lo += float(it.price_low or 0.0) * q
                misc_hi += float(it.price_high or 0.0) * q
        lo_sum += misc_lo
        hi_sum += misc_hi
        lines.append(
            f"- [misc_total] {len(remainder)} lower-value items rolled up | replacement=${misc_lo:,.2f} - ${misc_hi:,.2f}"
        )

    lines.extend(
        [
            "",
            "Board Summary Totals:",
            f"- Certified Floor: ${cert_sum:,.2f}",
            f"- Estimated Total Range (known items): ${lo_sum:,.2f} to ${hi_sum:,.2f}",
            f"- Unknown / Unpriced Pivotal Items: {ablated_count}",
        ]
    )
    return "\n".join(lines)
