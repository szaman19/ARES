"""Exhaustive unit tests for all 9 pure-Python deterministic verifiers (PLAN §10)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts import Frame, Listing
from ledger.verifiers import (
    catalog_exact_match,
    frame_quality,
    listing_freshness,
    model_number_format,
    normalize_model_str,
    policy_limit_parse,
    price_sanity,
    quantity_consistency,
    room_coverage,
    total_arithmetic,
)


def _now_iso(offset_seconds: int = 0) -> str:
    dt = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return dt.replace(microsecond=0).isoformat()


def test_model_number_format_across_brands() -> None:
    cases_valid = [
        ("A2992", "Apple"),
        ("Model A2338", "Apple"),
        ("U2723QE", "Dell"),
        ("ILCE-7M4", "Sony"),
        ("SEL2470GM2", "Sony"),
        ("WH-1000XM5", "Sony"),
        ("QN65Q80C", "Samsung"),
        ("OLED65C3PUA", "LG"),
        ("880066-0100", "Bose"),
        ("SV22", "Dyson"),
        ("KSM150PSER", "KitchenAid"),
        ("BES870XL", "Breville"),
        ("Q1P-M1", "Keychron"),
        ("SM7B", "Shure"),
        ("HAC-001", "Nintendo"),
    ]
    for model_txt, brand in cases_valid:
        ok, details = model_number_format(model_txt, brand)
        assert ok, f"Expected {model_txt} ({brand}) to pass, got {details}"

    cases_invalid = [
        ("", "Apple"),
        ("unknown", "Apple"),
        ("2023", "Apple"),
        ("14 inch silver laptop with sticker", "Apple"),
        ("??", "Sony"),
    ]
    for model_txt, brand in cases_invalid:
        ok, _ = model_number_format(model_txt, brand)
        assert not ok, f"Expected {model_txt} to be rejected"


def test_catalog_exact_match_passes_and_rejects() -> None:
    listing_mbp = Listing(
        listing_id="p1",
        title="Apple MacBook Pro 14-inch M3 Pro (Model A2992) Space Black",
        url="https://shop.app/p/mbp",
        price=1999.0,
        sku="A2992-M3",
        retrieved_at=_now_iso(),
    )
    ok, details = catalog_exact_match("Model A2992", "Apple", listing_mbp)
    assert ok
    assert details["normalized_model"] == "A2992"

    # Different model number -> reject
    ok_wrong, _ = catalog_exact_match("A2338", "Apple", listing_mbp)
    assert not ok_wrong

    # Hyphenated Sony camera model matches non-hyphenated SKU/title
    listing_sony = Listing(
        listing_id="p2",
        title="Sony Alpha 7 IV Body ILCE7M4",
        url="https://shop.app/p/sony",
        price=2498.0,
        retrieved_at=_now_iso(),
    )
    ok_sony, _ = catalog_exact_match("ILCE-7M4", "Sony", listing_sony)
    assert ok_sony
    assert normalize_model_str("ILCE-7M4") == "ILCE7M4"


def test_price_sanity_blocks_accessory_trap_and_outliers() -> None:
    trap_case = Listing(
        listing_id="trap1",
        title="Hard Shell Case for MacBook Pro 14 A2992 Clear Cover",
        url="https://shop.app/p/case",
        price=24.99,
        retrieved_at=_now_iso(),
    )
    real_peers = [
        Listing(listing_id="r1", title="MacBook Pro 14 M3 A2992", url="https://shop.app/1", price=1999.0, retrieved_at=_now_iso()),
        Listing(listing_id="r2", title="MacBook Pro 14 M3 Pro", url="https://shop.app/2", price=1899.0, retrieved_at=_now_iso()),
        Listing(listing_id="r3", title="MacBook Pro 14 Refurb", url="https://shop.app/3", price=1699.0, retrieved_at=_now_iso()),
    ]

    ok_trap, trap_info = price_sanity(trap_case, real_peers + [trap_case], item_category="laptop")
    assert not ok_trap
    assert trap_info["reason"] == "accessory_or_parts_keyword"

    # Extreme cheap non-keyword outlier ($150 vs ~$1900 median) also blocked
    cheap_outlier = Listing(
        listing_id="out1",
        title="Broken Laptop Logic Board A2992",
        url="https://shop.app/out",
        price=150.0,
        retrieved_at=_now_iso(),
    )
    ok_out, out_info = price_sanity(cheap_outlier, real_peers, item_category="laptop")
    assert not ok_out
    assert out_info["reason"] == "outside_peer_median_band"

    ok_real, _ = price_sanity(real_peers[0], real_peers, item_category="laptop")
    assert ok_real


def test_listing_freshness_enforces_10_min_ttl() -> None:
    fresh = Listing(listing_id="f1", title="Monitor", url="https://shop.app/m", price=500.0, retrieved_at=_now_iso(-120))
    stale = Listing(listing_id="f2", title="Monitor", url="https://shop.app/m", price=500.0, retrieved_at=_now_iso(-720))
    assert listing_freshness(fresh)[0] is True
    assert listing_freshness(stale)[0] is False


def test_quantity_consistency_and_frame_quality_and_policy_and_arithmetic() -> None:
    assert quantity_consistency(1, [1, 1, 1])[0] is True
    assert quantity_consistency(3, [1, 1, 1])[0] is False
    assert quantity_consistency(4, [2, 4, 3])[0] is True

    sharp_frame = Frame(
        frame_id="v@1000", video_id="v", ts_ms=1000, path="/tmp/f.jpg",
        phash="0011223344556677", blur_score=110.0, brightness=120.0, usable=True
    )
    blurry_dark = Frame(
        frame_id="v@2000", video_id="v", ts_ms=2000, path="/tmp/f2.jpg",
        phash="0011223344556677", blur_score=9.0, brightness=14.0, usable=False
    )
    assert frame_quality(sharp_frame)[0] is True
    ok_fq, fq_det = frame_quality(blurry_dark)
    assert ok_fq is False
    assert "frame_too_blurry" in fq_det["reasons"] and "frame_too_dark" in fq_det["reasons"]

    assert policy_limit_parse(15000.0, 500.0)[0] is True
    assert policy_limit_parse(200.0, 500.0)[0] is False
    assert policy_limit_parse(15000.0, 20000.0)[0] is False

    rc_ok, rc_det = room_coverage(["desk", "bedroom"], ["desk", "bedroom", "closet"])
    assert rc_ok is False
    assert "closet" in rc_det["rooms_missing"]

    rows = [
        {"quantity": 1, "price_status": "CERTIFIED", "unit_price": 1999.0},
        {"quantity": 2, "price_status": "UNCERTIFIED", "price_low": 200.0, "price_high": 400.0},
    ]
    assert total_arithmetic(rows, 1999.0, 2399.0, 2799.0)[0] is True
    assert total_arithmetic(rows, 2500.0, 2399.0, 2799.0)[0] is False
