"""100% invariant test suite for `ledger/board.py` (Invariants I1-I7)."""

from __future__ import annotations

import pytest

from commerce.cache import utc_now_iso
from contracts import BoardPatch, ItemPatch, LedgerBoard, Listing, Note, PolicyInfo
from ledger.board import (
    InvariantViolationError,
    append_notes_and_listings,
    apply_patch,
    build_probe_board_view,
    compute_board_drift,
    merge_items_weakest_link,
    rebuild_board,
)


def _sample_listing(lid: str = "lst_mbp", price: float = 1999.0) -> Listing:
    return Listing(
        listing_id=lid,
        title="Apple MacBook Pro 14-inch (Model A2992)",
        url="https://shop.app/p/mbp",
        price=price,
        sku="A2992",
        retrieved_at=utc_now_iso(),
    )


def test_i1_and_i6_forbid_uncited_or_invented_certified_prices() -> None:
    board = LedgerBoard(session_id="s1", policy=PolicyInfo(coverage_c_limit=10000.0))

    # Reject non-UNKNOWN item without cited notes (I1)
    with pytest.raises(InvariantViolationError, match="I1"):
        apply_patch(
            board,
            BoardPatch(
                requested_under=0,
                item_patches=[
                    ItemPatch(
                        item_id="it1",
                        category="laptop",
                        price_status="UNCERTIFIED",
                        price_low=900.0,
                        price_high=1800.0,
                    )
                ],
            ),
        )

    # Append UNCERTIFIED range note, then try to sneak in a unit_price (I6 violation!)
    n_uncert = Note(
        note_id="n1",
        item_id="it1",
        certification="UNCERTIFIED",
        kind="price",
        value={"price_low": 900.0, "price_high": 1800.0, "category": "laptop"},
        summary="Comparable MacBook range",
        requested_under=0,
    )
    board = append_notes_and_listings(board, [n_uncert])

    with pytest.raises(InvariantViolationError, match="I6"):
        apply_patch(
            board,
            BoardPatch(
                requested_under=0,
                item_patches=[
                    ItemPatch(
                        item_id="it1",
                        category="laptop",
                        price_status="UNCERTIFIED",
                        unit_price=1350.0,
                        price_low=900.0,
                        price_high=1800.0,
                        cite_note_ids=["n1"],
                    )
                ],
            ),
        )


def test_i2_forbids_certified_upgrade_without_catalog_exact_match_verifier() -> None:
    lst = _sample_listing()
    n_llm_guess = Note(
        note_id="n_guess",
        item_id="it1",
        certification="UNCERTIFIED",
        kind="price",
        verifier=None,
        listing_ids=[lst.listing_id],
        value={"price_low": 1500.0, "price_high": 2200.0},
        summary="Model guessed it might be A2992",
        requested_under=0,
    )
    board = append_notes_and_listings(LedgerBoard(), [n_llm_guess], [lst])

    with pytest.raises(InvariantViolationError, match="I1/I2"):
        apply_patch(
            board,
            BoardPatch(
                requested_under=0,
                item_patches=[
                    ItemPatch(
                        item_id="it1",
                        category="laptop",
                        price_status="CERTIFIED",
                        unit_price=1999.0,
                        cite_note_ids=["n_guess"],
                    )
                ],
            ),
        )


def test_i3_deterministic_rebuild_zero_drift_and_certified_flip() -> None:
    lst = _sample_listing()
    notes = [
        Note(
            note_id="n_id_0",
            item_id="it_laptop",
            certification="UNCERTIFIED",
            kind="identity",
            value={"room": "desk", "category": "laptop", "display_name": "Apple Laptop"},
            summary="Apple laptop seen on desk, label hidden",
            requested_under=0,
        ),
        Note(
            note_id="n_pr_0",
            item_id="it_laptop",
            certification="UNCERTIFIED",
            kind="price",
            value={"price_low": 999.0, "price_high": 2499.0},
            summary="25th-75th percentile of Apple laptops",
            requested_under=0,
        ),
    ]
    board = append_notes_and_listings(
        LedgerBoard(policy=PolicyInfo(coverage_c_limit=8000.0)), notes, [lst]
    )
    board = apply_patch(
        board,
        BoardPatch(
            requested_under=0,
            item_patches=[
                ItemPatch(
                    item_id="it_laptop",
                    room="desk",
                    category="laptop",
                    display_name="Apple Laptop",
                    identity_status="UNCERTIFIED",
                    price_status="UNCERTIFIED",
                    price_low=999.0,
                    price_high=2499.0,
                    cite_note_ids=["n_id_0", "n_pr_0"],
                )
            ],
        ),
    )

    rebuilt_1 = rebuild_board(board.notes, board.listings_index, board.policy)
    assert compute_board_drift(board, rebuilt_1) == 0

    # Follow-up clip certifies model A2992 + exact match!
    followup_notes = [
        Note(
            note_id="n_id_1",
            item_id="it_laptop",
            certification="CERTIFIED",
            kind="identity",
            verifier="model_number_format",
            value={"model": "A2992", "display_name": "Apple Laptop (A2992)"},
            summary="Bottom chassis engraving reads Model A2992",
            requested_under=board.board_version,
        ),
        Note(
            note_id="n_pr_1",
            item_id="it_laptop",
            certification="CERTIFIED",
            kind="price",
            verifier="catalog_exact_match",
            listing_ids=[lst.listing_id],
            value={"unit_price": lst.price},
            summary="Exact match on SKU/title A2992",
            requested_under=board.board_version,
        ),
    ]
    board = append_notes_and_listings(board, followup_notes)
    board = apply_patch(
        board,
        BoardPatch(
            requested_under=board.board_version,
            item_patches=[
                ItemPatch(
                    item_id="it_laptop",
                    display_name="Apple Laptop (A2992)",
                    model="A2992",
                    identity_status="CERTIFIED",
                    price_status="CERTIFIED",
                    unit_price=1999.0,
                    matched_listing=lst,
                    cite_note_ids=["n_id_1", "n_pr_1"],
                )
            ],
        ),
    )

    assert board.items["it_laptop"].price_status == "CERTIFIED"
    assert board.items["it_laptop"].unit_price == 1999.0
    assert board.items["it_laptop"].price_low is None
    rebuilt_2 = rebuild_board(board.notes, board.listings_index, board.policy)
    assert compute_board_drift(board, rebuilt_2) == 0


def test_i2_weakest_link_merge_and_i7_bounded_probe_view() -> None:
    lst = _sample_listing()
    notes = []
    patches = []
    for i in range(14):
        nid = f"n_{i}"
        iid = f"item_{i:02d}"
        notes.append(
            Note(
                note_id=nid,
                item_id=iid,
                certification="UNCERTIFIED",
                kind="price",
                value={"price_low": 100.0 + i * 50, "price_high": 200.0 + i * 100, "category": "gadget"},
                summary="range",
            )
        )
        patches.append(
            ItemPatch(
                item_id=iid,
                category="gadget",
                display_name=f"Gadget {i}",
                price_status="UNCERTIFIED",
                price_low=100.0 + i * 50,
                price_high=200.0 + i * 100,
                cite_note_ids=[nid],
            )
        )

    board = append_notes_and_listings(LedgerBoard(policy=PolicyInfo(coverage_c_limit=5000.0)), notes, [lst])
    board = apply_patch(board, BoardPatch(requested_under=0, item_patches=patches))

    # I7: capping max_items=10 rolls 4 items into misc_total
    view = build_probe_board_view(board, max_items=10, overrides={"item_13": "unknown"})
    assert "[misc_total] 4 lower-value items rolled up" in view
    assert "UNKNOWN (ablated)" in view

    # I2 merge weakest link check
    cert_item = board.items["item_00"].model_copy(
        update={"identity_status": "CERTIFIED", "price_status": "CERTIFIED", "unit_price": 500.0}
    )
    uncert_item = board.items["item_01"]
    merged = merge_items_weakest_link(cert_item, uncert_item)
    assert merged.price_status == "UNCERTIFIED"
    assert merged.unit_price is None
