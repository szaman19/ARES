"""Observation deduplication, cross-angle tracking, and spoken narration alignment (PLAN §9.4).

Handles two classic single-call multimodal failure modes:
1. Double-counting physical items across camera pans and adjacent room segments.
2. Catching spoken narration vs. visual/label contradictions (e.g., user says "Sony headphones"
   while the visible logo/label reads "Bose").
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable

from contracts import Frame, ItemObservation, SpokenClaim
from ingest.frames import hamming_distance_hex
from ledger.verifiers import normalize_model_str


CATEGORY_SYNONYMS: dict[str, str] = {
    "tv": "television",
    "smart_tv": "television",
    "flatscreen": "television",
    "television": "television",
    "macbook": "laptop",
    "notebook": "laptop",
    "computer_laptop": "laptop",
    "laptop": "laptop",
    "display": "monitor",
    "screen": "monitor",
    "computer_monitor": "monitor",
    "monitor": "monitor",
    "headphone": "headphones",
    "headset": "headphones",
    "earphones": "headphones",
    "over_ear_headphones": "headphones",
    "headphones": "headphones",
    "mechanical_keyboard": "keyboard",
    "wireless_keyboard": "keyboard",
    "keyboard": "keyboard",
    "mirrorless_camera": "camera",
    "dslr": "camera",
    "digital_camera": "camera",
    "camera_body": "camera",
    "camera": "camera",
    "camera_lens": "lens",
    "zoom_lens": "lens",
    "lens": "lens",
    "smartphone": "phone",
    "iphone": "phone",
    "cellphone": "phone",
    "mobile_phone": "phone",
    "phone": "phone",
    "ipad": "tablet",
    "android_tablet": "tablet",
    "tablet": "tablet",
    "acoustic_guitar": "guitar",
    "electric_guitar": "guitar",
    "guitar": "guitar",
    "couch": "sofa",
    "sectional": "sofa",
    "sofa": "sofa",
    "vacuum_cleaner": "vacuum",
    "cordless_vacuum": "vacuum",
    "vacuum": "vacuum",
    "office_chair": "chair",
    "desk_chair": "chair",
    "dining_chair": "chair",
    "wicker_chair": "chair",
    "armchair": "chair",
    "accent_chair": "chair",
    "chair": "chair",
    "writing_desk": "desk",
    "wooden_desk": "desk",
    "office_desk": "desk",
    "desk": "desk",
    "four_poster_bed": "bed",
    "bed_frame": "bed",
    "bed": "bed",
    "chest_of_drawers": "dresser",
    "cabinet_dresser": "dresser",
    "dresser": "dresser",
    "area_rug": "rug",
    "carpet": "rug",
    "patterned_rug": "rug",
    "rug": "rug",
    "bookcase": "bookshelf",
    "shelving_unit": "bookshelf",
    "shelf": "bookshelf",
    "bookshelf": "bookshelf",
    "gas_range": "stove",
    "electric_range": "stove",
    "range": "stove",
    "oven": "stove",
    "cooktop": "stove",
    "stove": "stove",
    "kitchen_mixer": "stand_mixer",
    "mixer": "stand_mixer",
    "stand_mixer": "stand_mixer",
    "laptop_stand": "stand",
    "aluminum_stand": "stand",
    "clamshell_stand": "stand",
    "stand": "stand",
    "table_lamp": "lamp",
    "floor_lamp": "lamp",
    "lamp": "lamp",
}


def canonical_category(cat: str | None) -> str:
    if not cat:
        return "item"
    cleaned = re.sub(r"[^a-z0-9]+", "_", cat.strip().lower()).strip("_")
    return CATEGORY_SYNONYMS.get(cleaned, cleaned)


def canonical_brand(brand: str | None) -> str:
    if not brand:
        return ""
    b = brand.strip().lower()
    if b in {"unknown", "none", "n/a", "unbranded", "generic", "?"}:
        return ""
    return b


@dataclass
class ResolvedItemTrack:
    item_id: str
    room: str
    category: str
    observations: list[ItemObservation] = field(default_factory=list)
    display_name: str = ""
    brand: str | None = None
    model_text: str | None = None
    description: str = ""
    quantity: int = 1
    simultaneous_counts: list[int] = field(default_factory=list)
    best_frame: Frame | None = None
    best_obs: ItemObservation | None = None
    catalog_queries: list[str] = field(default_factory=list)
    matched_claims: list[SpokenClaim] = field(default_factory=list)
    narration_conflict: str | None = None
    flags: list[str] = field(default_factory=list)


def _incompatible_identity(a: ItemObservation, b: ItemObservation) -> bool:
    """Deterministic gate: never merge two sightings with explicitly conflicting brands or models."""
    ba, bb = canonical_brand(a.brand), canonical_brand(b.brand)
    if ba and bb and ba != bb:
        return True
    ma, mb = normalize_model_str(a.model_text), normalize_model_str(b.model_text)
    if ma and mb and ma != mb:
        return True
    return False


def _bbox_iou(
    box_a: tuple[float, float, float, float] | None,
    box_b: tuple[float, float, float, float] | None,
) -> float:
    if not box_a or not box_b:
        return 0.0
    ax0, ay0, aw, ah = box_a
    bx0, by0, bw, bh = box_b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 1e-6 else 0.0


def _looks_like_mirror_duplicate(a: ItemObservation, b: ItemObservation) -> bool:
    """Detect mirror-reflection phantom duplicates seen in the exact same frame symmetrically."""
    if a.frame_id != b.frame_id or not a.bbox or not b.bbox:
        return False
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    # Same vertical band and similar size, mirrored horizontally across frame
    same_y = abs(ay - by) < 0.12
    same_size = abs(aw - bw) < 0.12 and abs(ah - bh) < 0.12
    mirrored_x = abs((ax + aw / 2.0) + (bx + bw / 2.0) - 1.0) < 0.22
    return same_y and same_size and mirrored_x


def resolve_observations(
    observations: list[ItemObservation],
    frames_by_id: dict[str, Frame],
    segment_rooms: dict[str, str],
    claims: list[SpokenClaim],
    ambiguous_merge_fn: Callable[[ItemObservation, ItemObservation], bool] | None = None,
) -> list[ResolvedItemTrack]:
    """Deduplicate `ItemObservation`s across camera angles/segments and align spoken claims."""
    sorted_obs = sorted(observations, key=lambda o: (o.ts_ms, o.obs_id))
    tracks: list[ResolvedItemTrack] = []

    for obs in sorted_obs:
        cat = canonical_category(obs.category)
        room = segment_rooms.get(obs.segment_id, "desk_area")
        fr = frames_by_id.get(obs.frame_id)

        merged_into: ResolvedItemTrack | None = None
        for tr in tracks:
            if tr.category != cat:
                continue
            ref_obs = tr.observations[-1]
            if _incompatible_identity(ref_obs, obs):
                continue

            # Same-frame check: two separate instances in the SAME frame stay distinct UNLESS mirror phantom
            if ref_obs.frame_id == obs.frame_id:
                if _looks_like_mirror_duplicate(ref_obs, obs):
                    if "mirror_reflection_filtered" not in tr.flags:
                        tr.flags.append("mirror_reflection_filtered")
                    merged_into = tr
                    break
                if _bbox_iou(ref_obs.bbox, obs.bbox) > 0.45:
                    merged_into = tr
                    break
                continue

            dt_ms = abs(obs.ts_ms - ref_obs.ts_ms)
            same_room = tr.room == room

            # Exact matching model text or same brand + close time in same room -> deterministic merge
            ma, mb = normalize_model_str(tr.model_text), normalize_model_str(obs.model_text)
            if ma and mb and ma == mb and same_room:
                merged_into = tr
                break

            ba, bb = canonical_brand(tr.brand), canonical_brand(obs.brand)
            if same_room and dt_ms <= 25_000:
                if (ba and bb and ba == bb) or (not ba or not bb):
                    # Check frame similarity or smooth pan continuity
                    ref_fr = frames_by_id.get(ref_obs.frame_id)
                    if ref_fr and fr and hamming_distance_hex(ref_fr.phash, fr.phash) <= 18:
                        merged_into = tr
                        break
                    # Same category in a single continuous desk/room pan within 20s
                    if dt_ms <= 20_000 and obs.segment_id == ref_obs.segment_id:
                        merged_into = tr
                        break
                    if ambiguous_merge_fn is not None and ambiguous_merge_fn(ref_obs, obs):
                        merged_into = tr
                        break

        if merged_into is None:
            idx = len(tracks) + 1
            tr = ResolvedItemTrack(
                item_id=f"item_{idx:02d}_{cat}",
                room=room,
                category=cat,
                observations=[obs],
                brand=obs.brand if canonical_brand(obs.brand) else None,
                model_text=obs.model_text if normalize_model_str(obs.model_text) else None,
                description=obs.description,
                quantity=max(1, obs.quantity),
                simultaneous_counts=[max(1, obs.quantity)],
                best_frame=fr,
                best_obs=obs,
                catalog_queries=list(obs.catalog_queries),
            )
            tracks.append(tr)
        else:
            merged_into.observations.append(obs)
            merged_into.simultaneous_counts.append(max(1, obs.quantity))
            # Quantity = max simultaneous count seen in any single frame (never sum across frames!)
            merged_into.quantity = max(merged_into.simultaneous_counts)

            if not merged_into.brand and canonical_brand(obs.brand):
                merged_into.brand = obs.brand
            if not merged_into.model_text and normalize_model_str(obs.model_text):
                merged_into.model_text = obs.model_text
            if len(obs.description) > len(merged_into.description):
                merged_into.description = obs.description

            # Keep sharpest usable frame, or frame that actually has legible model_text
            if fr is not None:
                cur_fr = merged_into.best_frame
                obs_has_label = bool(normalize_model_str(obs.model_text))
                cur_has_label = bool(
                    merged_into.best_obs and normalize_model_str(merged_into.best_obs.model_text)
                )
                if cur_fr is None or (obs_has_label and not cur_has_label) or (
                    obs_has_label == cur_has_label and fr.blur_score > cur_fr.blur_score
                ):
                    merged_into.best_frame = fr
                    merged_into.best_obs = obs

            for q in obs.catalog_queries:
                if q not in merged_into.catalog_queries:
                    merged_into.catalog_queries.append(q)

    # Align timestamped SpokenClaims to nearest compatible item track (window +-6s, fallback category match)
    for claim in claims:
        claim_cat = canonical_category(claim.category_hint) if claim.category_hint else None
        best_tr: ResolvedItemTrack | None = None
        best_dt = 999_999

        for tr in tracks:
            min_dt = min(abs(claim.ts_ms - o.ts_ms) for o in tr.observations)
            cat_compatible = (claim_cat == tr.category) or (
                claim_cat is None and tr.category in claim.text.lower()
            )
            if cat_compatible and min_dt <= 8_000 and min_dt < best_dt:
                best_dt = min_dt
                best_tr = tr

        if best_tr is None and claim_cat:
            # Fallback if narrator spoke slightly earlier/later in a short clip about the sole item of that category
            cat_tracks = [t for t in tracks if t.category == claim_cat]
            if len(cat_tracks) == 1:
                best_tr = cat_tracks[0]

        if best_tr is None:
            continue

        best_tr.matched_claims.append(claim)
        spoken_b = canonical_brand(claim.brand)
        visual_b = canonical_brand(best_tr.brand)
        spoken_m = normalize_model_str(claim.model)
        visual_m = normalize_model_str(best_tr.model_text)

        # Contradiction detection: narrator claims Brand/Model X, but visual logo/label reads Brand/Model Y!
        if spoken_b and visual_b and spoken_b != visual_b:
            conflict = (
                f'Narration at {claim.ts_ms / 1000:.1f}s claimed "{claim.brand}" ("{claim.text}"), '
                f'but visual/logo evidence shows "{best_tr.brand}". Kept visual identity.'
            )
            best_tr.narration_conflict = conflict
            if "narration_conflict" not in best_tr.flags:
                best_tr.flags.append("narration_conflict")
        elif spoken_m and visual_m and spoken_m != visual_m:
            conflict = (
                f'Narration at {claim.ts_ms / 1000:.1f}s claimed model "{claim.model}", '
                f'but legible device label reads "{best_tr.model_text}". Kept label value.'
            )
            best_tr.narration_conflict = conflict
            if "narration_conflict" not in best_tr.flags:
                best_tr.flags.append("narration_conflict")
        else:
            # Corroborating or supplemental spoken context (kept UNCERTIFIED unless label verified)
            if not best_tr.brand and claim.brand:
                best_tr.brand = claim.brand
                if "brand_from_narration_only" not in best_tr.flags:
                    best_tr.flags.append("brand_from_narration_only")

    # Build human-readable display names and fallback search magnets
    for tr in tracks:
        parts = [p for p in (tr.brand, tr.category.replace("_", " ").title()) if p]
        if tr.model_text:
            parts.append(f"({tr.model_text})")
        tr.display_name = " ".join(parts) if parts else tr.category.title()

        # Ensure deterministic search query ladder: exact `{brand} {model}` -> Hydrogen magnets -> generic fallback
        exact_queries: list[str] = []
        if tr.brand and tr.model_text:
            exact_queries.append(f"{tr.brand} {tr.model_text}")
        fallback_queries: list[str] = []
        if tr.brand:
            fallback_queries.append(f"{tr.brand} {tr.category}")
        fallback_queries.append(tr.category)

        merged_q: list[str] = []
        for q in exact_queries + tr.catalog_queries + fallback_queries:
            q_clean = " ".join(q.split())[:120]
            if q_clean and q_clean not in merged_q:
                merged_q.append(q_clean)
        tr.catalog_queries = merged_q[:4]

    return tracks
