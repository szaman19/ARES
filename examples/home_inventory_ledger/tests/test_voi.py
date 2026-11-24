"""Unit tests for counterfactual Total Variation influence and VoI capture request ranking."""

from __future__ import annotations

from contracts import LedgerBoard, LedgerItem, PolicyInfo
from ledger.voi import (
    deterministic_coverage_verdict,
    rank_capture_requests,
    total_variation_distance,
)


def test_total_variation_and_deterministic_verdict() -> None:
    p = {"adequate": 0.1, "likely_underinsured": 0.7, "underinsured": 0.2}
    q = {"adequate": 0.6, "likely_underinsured": 0.3, "underinsured": 0.1}
    tv = total_variation_distance(p, q)
    assert abs(tv - 0.5) < 1e-6

    assert deterministic_coverage_verdict(10000.0, 4000.0, 8500.0) == "adequate"
    assert deterministic_coverage_verdict(8000.0, 9200.0, 14000.0) == "underinsured"
    assert deterministic_coverage_verdict(8000.0, 6200.0, 11400.0) == "likely_underinsured"
    assert deterministic_coverage_verdict(None, 5000.0, 7000.0) == "insufficient_evidence"


def test_voi_ranks_widest_high_influence_uncertified_item_first() -> None:
    board = LedgerBoard(
        policy=PolicyInfo(coverage_c_limit=8000.0),
        items={
            "it_cam": LedgerItem(
                item_id="it_cam",
                room="desk",
                category="camera",
                display_name="Sony Camera (ILCE-7M4)",
                price_status="CERTIFIED",
                unit_price=2498.0,
            ),
            "it_laptop": LedgerItem(
                item_id="it_laptop",
                room="desk",
                category="laptop",
                display_name="Apple MacBook Pro",
                price_status="UNCERTIFIED",
                price_low=900.0,
                price_high=2600.0,
            ),
            "it_kb": LedgerItem(
                item_id="it_kb",
                room="desk",
                category="keyboard",
                display_name="Mechanical Keyboard",
                price_status="UNCERTIFIED",
                price_low=80.0,
                price_high=200.0,
            ),
        },
    )

    influences = {"it_cam": 0.40, "it_laptop": 0.52, "it_kb": 0.05}
    updated_items, asks = rank_capture_requests(board, influences, max_requests=3)

    # Certified item has 0 VoI even when high influence
    assert updated_items["it_cam"].voi_usd == 0.0
    # Laptop ($1,700 spread + 0.52 influence) must rank #1 over keyboard ($120 spread)
    assert len(asks) == 2
    assert asks[0].item_id == "it_laptop"
    assert asks[0].voi_usd > asks[1].voi_usd
    assert "label" in asks[0].instruction.lower()
