"""Pure-Python deterministic verifiers (PLAN §10).

CRITICAL INVARIANT (I4):
No LLM calls occur here, and no verifier ever accepts `LedgerBoard` as an input.
Every verifier is a pure deterministic function returning `(passed: bool, details: dict)`.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

from commerce.cache import PRICE_TTL_SECONDS, is_listing_fresh
from contracts import Frame, Listing


# Brand-specific model number regexes for >20 common electronics & home brands
BRAND_MODEL_PATTERNS: dict[str, re.Pattern[str]] = {
    "apple": re.compile(r"^(?:A\d{4}|M[A-Z0-9]{4,5}(?:LL/A)?|MAC\d{2},\d+)$"),
    "samsung": re.compile(r"^(?:QN\d{2}[A-Z0-9]{3,8}|UN\d{2}[A-Z0-9]{3,8}|SM-[A-Z]\d{3,4}[A-Z0-9]*|LS\d{2}[A-Z0-9]{4,8}|RF\d{2}[A-Z0-9]{4,8})$"),
    "sony": re.compile(r"^(?:ILCE\d?[A-Z0-9]{2,6}|SEL\d{2,6}[A-Z0-9]{1,5}|WH\d{4}XM\d|WF\d{4}XM\d|XR\d{2}[A-Z0-9]{3,6}|KD\d{2}[A-Z0-9]{3,6}|CFI-\d{4}[A-Z0-9]*)$"),
    "lg": re.compile(r"^(?:OLED\d{2}[A-Z0-9]{2,6}|(?:\d{2})?[A-Z]{2}\d{2,4}[A-Z0-9]{1,5}|WM\d{4}[A-Z0-9]{2,4})$"),
    "dell": re.compile(r"^(?:U\d{4}[A-Z]{1,3}|S\d{4}[A-Z]{1,3}|P\d{4}[A-Z]{1,3}|AW\d{4}[A-Z]{1,3}|XPS\d{4}|210-[A-Z0-9]{4})$"),
    "bose": re.compile(r"^(?:\d{6}-\d{4}|QC\d{2}|QCULTRA\d*|7\d{5}-\d{4})$"),
    "dyson": re.compile(r"^(?:SV\d{2}|V\d{1,2}[A-Z]*|HD\d{2}|TP\d{2}|HP\d{2}|\d{6}-\d{2})$"),
    "canon": re.compile(r"^(?:EOSR\d{1,2}[A-Z]*|RF\d{2,4}[A-Z0-9-]*|EF\d{2,4}[A-Z0-9-]*|\d{4}C\d{3})$"),
    "nikon": re.compile(r"^(?:Z[56789]\d*|ZFC|ZF|NIKKORZ\d{2,4}[A-Z0-9]*)$"),
    "fujifilm": re.compile(r"^(?:X-?T\d{1,2}|X100(?:V|VI|F)|X-?S\d{2}|X-?H\d[A-Z]*|XF\d{2,3}[A-Z0-9]*)$"),
    "lenovo": re.compile(r"^(?:20[A-Z0-9]{2,8}|21[A-Z0-9]{2,8}|82[A-Z0-9]{2,8}|83[A-Z0-9]{2,8})$"),
    "hp": re.compile(r"^(?:\d{2}-[A-Z]{2}\d{4}[A-Z]{2}|[A-Z0-9]{5,7}#ABA)$"),
    "asus": re.compile(r"^(?:ROG[A-Z0-9]{3,10}|PG\d{2}[A-Z0-9]{2,6}|PA\d{3,4}[A-Z0-9]{1,5}|GA\d{3}[A-Z0-9]{2,6}|UX\d{3,4}[A-Z0-9]*)$"),
    "logitech": re.compile(r"^(?:910-\d{6}|920-\d{6}|MX[A-Z0-9]{2,8}|YR\d{4}|MR\d{4})$"),
    "keychron": re.compile(r"^(?:[QKVC]\d{1,2}[A-Z]?-[A-Z]\d|K\d{1,2}PRO|Q\d{1,2}PRO|Q\d{1,2}MAX)$"),
    "nintendo": re.compile(r"^(?:HAC-\d{3}|HEG-\d{3}|HDH-\d{3})$"),
    "valve": re.compile(r"^(?:V\d{4}|1010|1030)$"),
    "microsoft": re.compile(r"^(?:QEZ-\d{5}|RRT-\d{5}|MODEL1[789]\d{2})$"),
    "sennheiser": re.compile(r"^(?:HD\d{3}[A-Z]*|M4AEBT|IE\d{3}|50\d{4})$"),
    "shure": re.compile(r"^(?:SM7B|SM7DB|MV7[X+]?|SM58|SE\d{3})$"),
    "kitchenaid": re.compile(r"^(?:KSM\d{2,4}[A-Z0-9]*|KV\d{2}[A-Z0-9]*|KFE\d{2}[A-Z0-9]*)$"),
    "breville": re.compile(r"^(?:BES\d{3}[A-Z]{0,3}|BOV\d{3}[A-Z]{0,3}|BCG\d{3}[A-Z]{0,3})$"),
}

# Generic alphanumeric industrial model-number fallback (must include both letters/digits or hyphenated digits, >=4 chars)
GENERIC_MODEL_RE = re.compile(r"^(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{3,22}$")

ACCESSORY_KEYWORDS = (
    "case for",
    "cover for",
    "sleeve for",
    "remote for",
    "replacement remote",
    "replacement filter",
    "replacement earpad",
    "earpads for",
    "ear pads for",
    "screen protector",
    "tempered glass",
    "power cord for",
    "charger for",
    "adapter for",
    "skin for",
    "decal for",
    "mount for",
    "wall mount for",
    "stand for",
    "parts only",
    "for parts",
    "empty box",
    "box only",
    "manual only",
)

ACCESSORY_ITEM_CATEGORIES = {
    "case",
    "cover",
    "cable",
    "charger",
    "adapter",
    "mount",
    "stand",
    "remote",
    "accessory",
}


def normalize_model_str(s: str | None) -> str:
    """Normalize a model string by stripping whitespace/dashes/underscores and uppercasing."""
    if not s:
        return ""
    cleaned = s.strip().upper()
    cleaned = re.sub(r"^MODEL\s*[:#-]?\s*", "", cleaned)
    cleaned = re.sub(r"[\s_-]+", "", cleaned)
    return cleaned


def model_number_format(
    model_text: str | None,
    brand: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Verify `model_text` looks like a genuine hardware model identifier, not a vague phrase."""
    raw = (model_text or "").strip()
    if not raw or raw.lower() in {"unknown", "none", "n/a", "null", "?"}:
        return False, {"reason": "empty_or_unknown_model"}
    # Reject multi-word descriptive sentences posing as model numbers
    if len(raw.split()) > 3:
        return False, {"reason": "descriptive_phrase_not_model", "raw": raw}

    norm_keep_hyphen = re.sub(r"\s+", "", raw.upper())
    norm_compact = normalize_model_str(raw)
    if len(norm_compact) < 4:
        return False, {"reason": "too_short", "normalized": norm_compact}

    # Reject standalone 4-digit calendar years (1980..2039) before any brand/generic regex
    if re.match(r"^(?:19[89]\d|20[0-3]\d)$", norm_compact):
        return False, {"reason": "year_not_model_number", "normalized": norm_compact}

    brand_key = (brand or "").strip().lower()
    if brand_key in BRAND_MODEL_PATTERNS:
        pat = BRAND_MODEL_PATTERNS[brand_key]
        if pat.match(norm_keep_hyphen) or pat.match(norm_compact):
            return True, {
                "normalized": norm_compact,
                "brand_matched": brand_key,
                "pattern_kind": "brand_specific",
            }

    # Check if any known brand regex matches even when brand differed slightly
    for bname, pat in BRAND_MODEL_PATTERNS.items():
        if pat.match(norm_keep_hyphen) or pat.match(norm_compact):
            return True, {
                "normalized": norm_compact,
                "brand_matched": bname,
                "pattern_kind": "cross_brand_pattern",
            }

    # Generic strict hardware identifier check: must have digits and not be a plain year (1990..2030)
    if re.match(r"^(?:19|20)\d{2}$", norm_compact):
        return False, {"reason": "year_not_model_number", "normalized": norm_compact}

    if GENERIC_MODEL_RE.match(norm_keep_hyphen) and any(c.isalpha() for c in norm_compact):
        return True, {
            "normalized": norm_compact,
            "brand_matched": None,
            "pattern_kind": "generic_alphanumeric",
        }

    return False, {"reason": "failed_format_regex", "normalized": norm_compact}


def catalog_exact_match(
    model_text: str | None,
    brand: str | None,
    listing: Listing,
) -> tuple[bool, dict[str, Any]]:
    """Certify identity + price ONLY when a format-valid model string matches the listing title/SKU/specs."""
    fmt_ok, fmt_details = model_number_format(model_text, brand)
    if not fmt_ok:
        return False, {"reason": "model_format_invalid", **fmt_details}

    norm_model = fmt_details["normalized"]

    # Check token-boundary and normalized containment across title, sku, and spec fields
    candidate_fields: dict[str, str] = {"title": listing.title}
    if listing.sku:
        candidate_fields["sku"] = listing.sku
    for k in ("model", "mpn", "sku", "part_number"):
        if listing.specs.get(k):
            candidate_fields[f"specs.{k}"] = str(listing.specs[k])

    matched_in: list[str] = []
    for field_name, text_val in candidate_fields.items():
        # Split on non-alphanumeric boundaries AND check hyphen-joined tokens
        tokens = re.split(r"[^A-Za-z0-9/-]+", text_val.upper())
        for i, tok in enumerate(tokens):
            t_norm = normalize_model_str(tok)
            if t_norm == norm_model:
                matched_in.append(field_name)
                break
            # Also check adjacent 2-token join (e.g. "ILCE" + "7M4" or "880066" + "0100")
            if i + 1 < len(tokens):
                joined = normalize_model_str(tok + tokens[i + 1])
                if joined == norm_model:
                    matched_in.append(field_name)
                    break

    if not matched_in:
        return False, {
            "reason": "model_not_in_listing",
            "normalized_model": norm_model,
            "listing_id": listing.listing_id,
        }

    # If brand is known, verify listing does not explicitly belong to a conflicting brand
    if brand:
        b_low = brand.strip().lower()
        hay_low = f"{listing.title} {listing.merchant or ''} {listing.specs.get('brand', '')}".lower()
        if b_low not in hay_low and fmt_details.get("pattern_kind") == "generic_alphanumeric":
            return False, {
                "reason": "brand_missing_on_generic_model",
                "expected_brand": brand,
                "listing_id": listing.listing_id,
            }

    return True, {
        "normalized_model": norm_model,
        "matched_in": matched_in,
        "listing_id": listing.listing_id,
        "price": listing.price,
    }


def price_sanity(
    listing: Listing,
    peer_listings: list[Listing],
    item_category: str = "",
) -> tuple[bool, dict[str, Any]]:
    """Block accessory traps ('case for MacBook $24') and extreme price outliers outside [0.3x, 3.0x] median."""
    if listing.price <= 0:
        return False, {"reason": "non_positive_price", "price": listing.price}

    cat_norm = item_category.strip().lower()
    title_low = listing.title.lower()

    if cat_norm not in ACCESSORY_ITEM_CATEGORIES:
        for kw in ACCESSORY_KEYWORDS:
            if kw in title_low:
                return False, {
                    "reason": "accessory_or_parts_keyword",
                    "keyword": kw,
                    "title": listing.title,
                }

    # Filter peer listings against accessory traps (and prefer same-category peers if >=3 exist)
    clean_peers = [
        p
        for p in peer_listings
        if p.price > 0
        and (
            cat_norm in ACCESSORY_ITEM_CATEGORIES
            or not any(kw in p.title.lower() for kw in ACCESSORY_KEYWORDS)
        )
    ]
    same_cat_peers = [
        p for p in clean_peers if cat_norm and (cat_norm in p.title.lower() or cat_norm == str(p.specs.get("category", "")).lower())
    ]
    ref_peers = same_cat_peers if len(same_cat_peers) >= 2 else clean_peers
    clean_peer_prices = [p.price for p in ref_peers]

    if len(clean_peer_prices) >= 3:
        med = float(statistics.median(clean_peer_prices))
        low_bound = 0.30 * med
        high_bound = 3.00 * med
        if not (low_bound <= listing.price <= high_bound):
            return False, {
                "reason": "outside_peer_median_band",
                "price": listing.price,
                "peer_median": round(med, 2),
                "allowed_band": [round(low_bound, 2), round(high_bound, 2)],
            }
        return True, {"price": listing.price, "peer_median": round(med, 2)}

    return True, {"price": listing.price, "peer_count": len(clean_peer_prices)}


def listing_freshness(
    listing: Listing,
    max_age_seconds: int = PRICE_TTL_SECONDS,
) -> tuple[bool, dict[str, Any]]:
    """Verify `now - retrieved_at <= 10 min` per shop.dev price caching rules."""
    ok = is_listing_fresh(listing.retrieved_at, max_age_seconds=max_age_seconds)
    return ok, {"retrieved_at": listing.retrieved_at, "max_age_seconds": max_age_seconds}


def quantity_consistency(
    proposed_quantity: int,
    simultaneous_counts_per_frame: list[int],
) -> tuple[bool, dict[str, Any]]:
    """Ensure item quantity does not exceed max simultaneous sightings in a single frame (blocks multi-angle summing)."""
    if proposed_quantity < 1:
        return False, {"reason": "quantity_below_one", "proposed": proposed_quantity}
    max_sim = max(simultaneous_counts_per_frame) if simultaneous_counts_per_frame else 1
    if proposed_quantity > max_sim:
        return False, {
            "reason": "exceeds_max_simultaneous_in_single_frame",
            "proposed": proposed_quantity,
            "max_simultaneous": max_sim,
        }
    return True, {"quantity": proposed_quantity, "max_simultaneous": max_sim}


def room_coverage(
    rooms_seen: list[str],
    expected_rooms: list[str] | None = None,
    unopened_spaces: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Check whether expected rooms and closets/drawers were shown."""
    seen_norm = {r.strip().lower() for r in rooms_seen if r.strip()}
    expected_norm = {r.strip().lower() for r in (expected_rooms or []) if r.strip()}
    missing_rooms = sorted(expected_norm - seen_norm)
    unopened = list(unopened_spaces or [])
    passed = len(missing_rooms) == 0 and len(unopened) == 0
    return passed, {
        "rooms_seen": sorted(seen_norm),
        "rooms_missing": missing_rooms + unopened,
    }


def frame_quality(
    frame: Frame | None,
    min_blur_score: float = 12.0,
    min_brightness: float = 28.0,
) -> tuple[bool, dict[str, Any]]:
    """Gate certification on visual legibility of the best proof frame."""
    if frame is None:
        return False, {"reason": "missing_frame"}
    reasons: list[str] = []
    if frame.blur_score < min_blur_score:
        reasons.append("frame_too_blurry")
    if frame.brightness < min_brightness:
        reasons.append("frame_too_dark")
    return len(reasons) == 0, {
        "frame_id": frame.frame_id,
        "blur_score": frame.blur_score,
        "brightness": frame.brightness,
        "reasons": reasons,
    }


def policy_limit_parse(
    coverage_c_limit: float | None,
    deductible: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Validate extracted Coverage C limit is numeric, plausible ($1,000-$5,000,000), and > deductible."""
    if coverage_c_limit is None:
        return False, {"reason": "missing_coverage_limit"}
    if not (1_000.0 <= coverage_c_limit <= 5_000_000.0):
        return False, {
            "reason": "limit_outside_plausible_range",
            "coverage_c_limit": coverage_c_limit,
        }
    if deductible is not None and (deductible < 0 or deductible >= coverage_c_limit):
        return False, {
            "reason": "invalid_deductible_vs_limit",
            "coverage_c_limit": coverage_c_limit,
            "deductible": deductible,
        }
    return True, {"coverage_c_limit": coverage_c_limit, "deductible": deductible}


def total_arithmetic(
    rows: list[dict[str, Any]],
    reported_certified_total: float,
    reported_low: float,
    reported_high: float,
    tol: float = 0.05,
) -> tuple[bool, dict[str, Any]]:
    """Deterministically recompute certified total and estimate range from item rows."""
    cert_sum = 0.0
    uncert_low_sum = 0.0
    uncert_high_sum = 0.0

    for r in rows:
        qty = int(r.get("quantity", 1))
        status = r.get("price_status", "UNKNOWN")
        if status == "CERTIFIED" and r.get("unit_price") is not None:
            cert_sum += float(r["unit_price"]) * qty
        elif status == "UNCERTIFIED":
            lo = float(r.get("price_low") or 0.0)
            hi = float(r.get("price_high") or lo)
            uncert_low_sum += lo * qty
            uncert_high_sum += hi * qty

    exp_cert = round(cert_sum, 2)
    exp_low = round(cert_sum + uncert_low_sum, 2)
    exp_high = round(cert_sum + uncert_high_sum, 2)

    ok = (
        abs(exp_cert - reported_certified_total) <= tol
        and abs(exp_low - reported_low) <= tol
        and abs(exp_high - reported_high) <= tol
    )
    return ok, {
        "expected_certified_total": exp_cert,
        "expected_estimate_low": exp_low,
        "expected_estimate_high": exp_high,
        "reported_certified_total": reported_certified_total,
        "reported_estimate_low": reported_low,
        "reported_estimate_high": reported_high,
    }
