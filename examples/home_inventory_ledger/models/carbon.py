"""Gemini 4 Carbon (`gemini-4-carbon`) — Global video/audio/document understander & merge reviewer.

Clear single model job (Hackathon Rule 1):
1. Watches the full walkthrough (video + raw audio narration + photographed insurance declarations
   page) in one long-context multimodal pass to extract ordered room segments, timestamped spoken
   owner claims, and Coverage C policy limits.
2. Arbitrates ambiguous cross-angle deduplication pairs when deterministic gates are inconclusive.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field

from contracts import Frame, ItemObservation, PolicyInfo, RoomSegment, SpokenClaim
from models.provider import GenAIRuntime, audio_part, image_part


class _SegmentItem(BaseModel):
    start_ms: int
    end_ms: int
    room_label: str
    summary: str = ""
    closets_or_drawers_unopened: list[str] = Field(default_factory=list)


class _ClaimItem(BaseModel):
    ts_ms: int
    text: str
    category_hint: str | None = None
    brand: str | None = None
    model: str | None = None
    purchase_year: int | None = None
    material: str | None = None
    stated_price: float | None = None


class _PolicyExtract(BaseModel):
    coverage_c_limit: float | None = None
    deductible: float | None = None
    carrier: str | None = None
    confidence: float = 0.85


class _UnderstandSchema(BaseModel):
    segments: list[_SegmentItem] = Field(default_factory=list)
    spoken_claims: list[_ClaimItem] = Field(default_factory=list)
    policy: _PolicyExtract = Field(default_factory=_PolicyExtract)


class _MergeDecisionSchema(BaseModel):
    same_physical_item: bool
    reason: str = ""


CARBON_SYSTEM_INSTRUCTION = """You are the Global Multimodal Understander for Home Inventory Ledger.
Analyze the home/desk walkthrough video (and audio narration) plus optional insurance declarations page image.
Rules:
1. Segment the video into contiguous, non-overlapping room or area segments in chronological order (`start_ms` < `end_ms`).
2. Transcribe every spoken claim about any item in `spoken_claims` with accurate `ts_ms` timestamps, extracting what the speaker *claimed* (`brand`, `model`, `purchase_year`, `material`, `category_hint`), even if the speaker misnamed the brand compared to what is visible!
3. Note any visible closed closets, cabinets, or drawers that were never opened in `closets_or_drawers_unopened`.
4. If an insurance declarations page image is attached, read Personal Property / Coverage C limit (`coverage_c_limit`) and deductible (`deductible`) in USD. Never invent policy numbers or limits if no document is provided."""


class CarbonClient:
    def __init__(self, runtime: GenAIRuntime) -> None:
        self.rt = runtime

    def extract_policy_only(self, dec_page_path: str | Path) -> PolicyInfo:
        """Read Coverage C limit and deductible from a photographed declarations page."""
        p = Path(dec_page_path)
        if self.rt.replay_mode:
            self.rt.record_call("carbon")
            return PolicyInfo(
                coverage_c_limit=7500.0,
                deductible=500.0,
                carrier="Sample Homeowners / Renters Policy",
                confidence=0.98,
                source_path=str(p),
            )

        parsed = self.rt.structured_interaction(
            role="carbon",
            model=self.rt.carbon_model,
            input_parts=[
                image_part(p),
                (
                    "Extract the Personal Property (Coverage C / Contents) replacement limit in USD "
                    "and the Section I deductible from this insurance declarations page. "
                    "If no limit is visible, return null."
                ),
            ],
            schema_cls=_PolicyExtract,
            system_instruction="Read insurance declarations pages accurately. Never guess missing numbers.",
        )
        return PolicyInfo(
            coverage_c_limit=parsed.coverage_c_limit,
            deductible=parsed.deductible,
            carrier=parsed.carrier,
            confidence=parsed.confidence,
            source_path=str(p),
        )

    def understand(
        self,
        video_path: str | Path,
        frames: list[Frame],
        audio_wav_path: str | Path | None = None,
        dec_page_path: str | Path | None = None,
        video_id: str = "clip",
    ) -> tuple[list[RoomSegment], list[SpokenClaim], PolicyInfo]:
        """Run full-clip multimodal understanding on `gemini-4-carbon`."""
        max_ts = max((f.ts_ms for f in frames), default=10_000) + 500

        if self.rt.replay_mode:
            self.rt.record_call("carbon")
            mid_ts = max(2000, max_ts // 2)
            segments = [
                RoomSegment(
                    segment_id=f"{video_id}_seg_01",
                    video_id=video_id,
                    start_ms=0,
                    end_ms=mid_ts,
                    room_label="office_desk",
                    summary="Left and center desk setup with monitor, laptop, and mechanical keyboard",
                ),
                RoomSegment(
                    segment_id=f"{video_id}_seg_02",
                    video_id=video_id,
                    start_ms=mid_ts,
                    end_ms=max_ts,
                    room_label="office_desk",
                    summary="Right side of desk with over-ear headphones and mirrorless camera + lens",
                ),
            ]
            claims = [
                SpokenClaim(
                    claim_id=f"{video_id}_cl_01",
                    ts_ms=min(2500, max_ts - 500),
                    text="MacBook Pro from last year",
                    category_hint="laptop",
                    brand="Apple",
                    purchase_year=2025,
                ),
                SpokenClaim(
                    claim_id=f"{video_id}_cl_02",
                    ts_ms=min(6000, max_ts - 300),
                    text="Next to the keyboard are my Sony headphones",
                    category_hint="headphones",
                    brand="Sony",
                ),
                SpokenClaim(
                    claim_id=f"{video_id}_cl_03",
                    ts_ms=min(8200, max_ts - 100),
                    text="Sony Alpha a7 IV camera with 24-70 lens",
                    category_hint="camera",
                    brand="Sony",
                    model="ILCE-7M4",
                ),
            ]
            pol = PolicyInfo(
                coverage_c_limit=7500.0 if dec_page_path else None,
                deductible=500.0 if dec_page_path else None,
                carrier="Sample Policy (Replay)" if dec_page_path else None,
                confidence=0.97 if dec_page_path else 0.0,
                source_path=str(dec_page_path) if dec_page_path else None,
            )
            return segments, claims, pol

        parts: list[dict | str] = []
        vid_part = self.rt.upload_video_part(video_path)
        if vid_part is not None:
            parts.append(vid_part)
        else:
            # Fallback: keyframe strip + extracted WAV audio
            stride = max(1, len(frames) // 12)
            for fr in frames[::stride][:12]:
                parts.append(f"[Frame @{fr.ts_ms}ms]")
                parts.append(image_part(fr.path))
            if audio_wav_path and Path(audio_wav_path).exists():
                parts.append(audio_part(audio_wav_path))

        if dec_page_path and Path(dec_page_path).exists():
            parts.append("[Insurance Declarations Page Image]")
            parts.append(image_part(dec_page_path))

        parts.append(
            f"Video duration is ~{max_ts} ms. Segment into 1-4 chronological segments "
            "(for a single desk/room clip >6s, split into 2 temporal pan segments of ~5-15s so "
            "parallel item extractors can inspect each half closely). Extract all timestamped "
            "spoken claims from narration audio, and parse Coverage C policy limit if attached."
        )

        out = self.rt.structured_interaction(
            role="carbon",
            model=self.rt.carbon_model,
            input_parts=parts,
            schema_cls=_UnderstandSchema,
            system_instruction=CARBON_SYSTEM_INSTRUCTION,
        )

        # Normalize & validate non-overlapping ordered segments
        raw_segs = sorted(out.segments, key=lambda s: s.start_ms)
        segments: list[RoomSegment] = []
        cursor = 0
        for idx, s in enumerate(raw_segs):
            st = max(cursor, s.start_ms)
            en = max(st + 500, min(max_ts + 2000, s.end_ms))
            segments.append(
                RoomSegment(
                    segment_id=f"{video_id}_seg_{idx + 1:02d}",
                    video_id=video_id,
                    start_ms=st,
                    end_ms=en,
                    room_label=s.room_label or "desk_area",
                    summary=s.summary,
                    closets_or_drawers_unopened=s.closets_or_drawers_unopened,
                )
            )
            cursor = en

        if not segments:
            segments = [
                RoomSegment(
                    segment_id=f"{video_id}_seg_01",
                    video_id=video_id,
                    start_ms=0,
                    end_ms=max_ts,
                    room_label="desk_area",
                )
            ]

        claims = [
            SpokenClaim(
                claim_id=f"{video_id}_cl_{i + 1:02d}",
                ts_ms=max(0, c.ts_ms),
                text=c.text,
                category_hint=c.category_hint,
                brand=c.brand,
                model=c.model,
                purchase_year=c.purchase_year,
                material=c.material,
                stated_price=c.stated_price,
            )
            for i, c in enumerate(out.spoken_claims)
        ]

        policy = PolicyInfo(
            coverage_c_limit=out.policy.coverage_c_limit,
            deductible=out.policy.deductible,
            carrier=out.policy.carrier,
            confidence=out.policy.confidence if out.policy.coverage_c_limit else 0.0,
            source_path=str(dec_page_path) if dec_page_path else None,
        )
        return segments, claims, policy

    def review_ambiguous_merge(
        self,
        crop_a_path: str | Path,
        crop_b_path: str | Path,
        obs_a: ItemObservation,
        obs_b: ItemObservation,
    ) -> bool:
        """Ask Carbon whether two ambiguous crops depict the exact same physical item."""
        if self.rt.replay_mode:
            return obs_a.category == obs_b.category
        res = self.rt.structured_interaction(
            role="carbon",
            model=self.rt.carbon_model,
            input_parts=[
                f"Sighting A (@{obs_a.ts_ms}ms): {obs_a.description}",
                image_part(crop_a_path),
                f"Sighting B (@{obs_b.ts_ms}ms): {obs_b.description}",
                image_part(crop_b_path),
                "Are Sighting A and Sighting B the exact same physical object filmed from two angles?",
            ],
            schema_cls=_MergeDecisionSchema,
        )
        return res.same_physical_item
