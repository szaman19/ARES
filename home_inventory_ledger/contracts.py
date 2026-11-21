"""Core Pydantic v2 data contracts for Home Inventory Ledger.

Every stage of the pipeline—ingest, Gemini 4 Carbon understanding, parallel
Gemini 4 Hydrogen extraction, Shop MCP pricing, deterministic verification,
invariant-checked board patching, and Gemini Decision-1 counterfactual probing—
communicates exclusively through these immutable or versioned types.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

Cert = Literal["CERTIFIED", "UNCERTIFIED", "UNKNOWN"]
EvidenceKind = Literal["visual", "spoken", "label_read", "catalog_match", "user_answer"]
VerdictLabel = Literal[
    "adequate",
    "likely_underinsured",
    "underinsured",
    "insufficient_evidence",
    "unknown",
]


class Frame(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_id: str  # f"{video_id}@{ts_ms}"
    video_id: str
    ts_ms: int
    path: str
    phash: str  # 64-bit perceptual dHash/aHash hex string (16 hex chars)
    blur_score: float  # edge/Laplacian variance; low = blurry
    brightness: float  # mean luminance 0..255
    usable: bool = True


class PolicyInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    coverage_c_limit: float | None = None
    deductible: float | None = None
    policy_number_masked: str | None = None
    carrier: str | None = None
    confidence: float = 0.0
    source_path: str | None = None


class RoomSegment(BaseModel):
    model_config = ConfigDict(frozen=True)

    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    room_label: str  # e.g., "office_desk", "living_room", "bedroom"
    summary: str = ""
    closets_or_drawers_unopened: list[str] = Field(default_factory=list)


class SpokenClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    ts_ms: int
    text: str
    category_hint: str | None = None
    brand: str | None = None
    model: str | None = None
    purchase_year: int | None = None
    material: str | None = None
    stated_price: float | None = None


class ItemObservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    obs_id: str
    segment_id: str
    frame_id: str
    ts_ms: int = 0
    bbox: tuple[float, float, float, float] | None = None  # normalized (x, y, w, h) in 0..1
    category: str
    description: str
    brand: str | None = None
    model_text: str | None = None  # ONLY when characters are visibly legible on device/label
    quantity: int = 1
    condition: Literal["new", "good", "worn", "damaged", "unknown"] = "unknown"
    evidence: list[EvidenceKind] = Field(default_factory=lambda: ["visual"])
    catalog_queries: list[str] = Field(default_factory=list)
    produced_by: str = "gemini-4-hydrogen"


class Listing(BaseModel):
    model_config = ConfigDict(frozen=True)

    listing_id: str
    title: str
    url: str
    price: float
    currency: str = "USD"
    merchant: str | None = None
    sku: str | None = None
    variant_id: str | None = None
    available: bool = True
    specs: dict[str, Any] = Field(default_factory=dict)
    image_url: str | None = None
    retrieved_at: str  # ISO-8601 UTC timestamp


class Note(BaseModel):
    """Immutable, append-only evidence record. Single source of truth for the board."""

    model_config = ConfigDict(frozen=True)

    note_id: str
    item_id: str
    certification: Literal["CERTIFIED", "UNCERTIFIED"]
    kind: Literal["identity", "price", "quantity", "contradiction", "coverage"]
    verifier: str | None = None
    frame_ids: list[str] = Field(default_factory=list)
    listing_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    value: dict[str, Any] = Field(default_factory=dict)
    summary: str
    requested_under: int = 0  # board_version at dispatch


class LedgerItem(BaseModel):
    item_id: str
    room: str
    category: str
    display_name: str
    brand: str | None = None
    model: str | None = None
    description: str = ""
    quantity: int = 1
    identity_status: Cert = "UNKNOWN"
    price_status: Cert = "UNKNOWN"
    unit_price: float | None = None  # Invariant I6: ONLY set when price_status == "CERTIFIED"
    price_low: float | None = None  # Set for UNCERTIFIED range
    price_high: float | None = None
    matched_listing: Listing | None = None
    candidate_listings: list[Listing] = Field(default_factory=list)
    best_frame_id: str | None = None
    best_frame_path: str | None = None
    first_seen_ms: int = 0
    spoken_claim_text: str | None = None
    conflict_detail: str | None = None
    sources: list[str] = Field(default_factory=list)  # cited note_ids (Invariant I1)
    flags: list[str] = Field(default_factory=list)
    influence: float = 0.0
    voi_usd: float = 0.0


class ItemPatch(BaseModel):
    """Proposed mutation to a single LedgerItem, validated against Invariants I1-I7."""

    item_id: str
    room: str | None = None
    category: str | None = None
    display_name: str | None = None
    brand: str | None = None
    model: str | None = None
    description: str | None = None
    quantity: int | None = None
    identity_status: Cert | None = None
    price_status: Cert | None = None
    unit_price: float | None = None
    price_low: float | None = None
    price_high: float | None = None
    matched_listing: Listing | None = None
    candidate_listings: list[Listing] | None = None
    best_frame_id: str | None = None
    best_frame_path: str | None = None
    first_seen_ms: int | None = None
    spoken_claim_text: str | None = None
    conflict_detail: str | None = None
    cite_note_ids: list[str] = Field(default_factory=list)
    add_flags: list[str] = Field(default_factory=list)


class BoardPatch(BaseModel):
    requested_under: int
    item_patches: list[ItemPatch] = Field(default_factory=list)
    rooms_seen: list[str] | None = None
    rooms_missing: list[str] | None = None


class CoverageDecision(BaseModel):
    coverage_c_limit: float | None
    deductible: float | None = None
    certified_total: float
    estimate_low: float
    estimate_high: float
    point_estimate: float
    verdict: VerdictLabel
    deterministic_verdict: VerdictLabel
    verdict_disagreement: bool = False
    confidence: float
    probabilities: dict[str, float] = Field(default_factory=dict)
    inventory_complete: bool = True
    hinges_on: list[str] = Field(default_factory=list)  # top item_ids by counterfactual influence
    certified_share: float = 0.0  # share of point estimate backed by CERTIFIED prices


class CaptureRequest(BaseModel):
    request_id: str
    item_id: str | None = None
    item_name: str | None = None
    room: str | None = None
    gap_kind: Literal[
        "missing_label",
        "blurry_frame",
        "material_ambiguity",
        "quantity_ambiguity",
        "missing_room",
    ] = "missing_label"
    instruction: str
    voi_usd: float
    uncertainty_width_usd: float = 0.0
    influence: float = 0.0
    answer_mode: Literal["video", "voice", "text"] = "video"
    resolved: bool = False


class PipelineTelemetry(BaseModel):
    carbon_calls: int = 0
    hydrogen_calls: int = 0
    decision_calls: int = 0
    shop_mcp_calls: int = 0
    frames_extracted: int = 0
    frames_kept_after_dedup: int = 0
    board_drift_count: int = 0
    rejected_patches: int = 0
    stale_notes_logged: int = 0
    elapsed_seconds: float = 0.0
    mode: Literal["live", "replay", "hybrid"] = "live"


class LedgerBoard(BaseModel):
    session_id: str = "default"
    board_version: int = 0
    policy: PolicyInfo = Field(default_factory=PolicyInfo)
    items: dict[str, LedgerItem] = Field(default_factory=dict)
    notes: list[Note] = Field(default_factory=list)
    listings_index: dict[str, Listing] = Field(default_factory=dict)
    rooms_seen: list[str] = Field(default_factory=list)
    rooms_missing: list[str] = Field(default_factory=list)
    decision: CoverageDecision | None = None
    requests: list[CaptureRequest] = Field(default_factory=list)
    telemetry: PipelineTelemetry = Field(default_factory=PipelineTelemetry)
