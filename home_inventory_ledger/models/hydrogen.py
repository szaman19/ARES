"""Gemini 4 Hydrogen (`gemini-4-hydrogen`) — Parallel per-segment item & label extractors.

Clear single model job (Hackathon Rule 1):
Runs concurrently across video segments (`ThreadPoolExecutor`) to inspect high-res keyframes,
read visibly legible hardware model labels (strictly never guessing), count simultaneous items
per frame, and author ranked Shopify catalog search magnets.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from pydantic import BaseModel, Field

from contracts import CaptureRequest, Frame, ItemObservation, LedgerItem, RoomSegment, SpokenClaim
from models.provider import GenAIRuntime, image_part


class _RawObs(BaseModel):
    frame_id: str
    bbox: list[float] | None = None  # [x, y, w, h] normalized 0..1
    category: str
    description: str
    brand: str | None = None
    model_text: str | None = None
    quantity: int = 1
    condition: Literal["new", "good", "worn", "damaged", "unknown"] = "good"
    has_legible_label: bool = False
    catalog_queries: list[str] = Field(default_factory=list)


class _SegmentExtractionSchema(BaseModel):
    observations: list[_RawObs] = Field(default_factory=list)


HYDROGEN_SYSTEM_INSTRUCTION = """You are a fast, rigorous Visual Hardware & Home Item Extractor powered by Gemini 4 Hydrogen.
Inspect the provided keyframes from one video segment and return distinct physical items worth >= $25.
STRICT RULES:
1. NEVER invent, infer, or guess a model number! Populate `model_text` ONLY if literal model characters (e.g. 'U2723QE', 'ILCE-7M4', 'Model A2992') are directly legible in the pixels of the frame. Otherwise set `model_text` to null.
2. Read visible brand logos directly from the frame (`brand`), NOT from what the speaker said if the logo shows a different brand! (For example, if narration says 'Sony' but the earcup reads 'BOSE', set `brand` to 'Bose'.)
3. `quantity` is the number of identical instances visible SIMULTANEOUSLY in the SAME single frame. Never add counts across multiple frames.
4. Ignore background walls, floors, cables, disposable cups, or digital items shown on a TV/monitor screen.
5. For each item, write 2-3 plain-text Shopify product search queries (`catalog_queries`, <=100 chars, no URLs): exact `{brand} {model_text}` if legible, then `{brand} {size/spec} {category}`, then generic `{category}`."""


def _clamp_bbox(raw: list[float] | None) -> tuple[float, float, float, float] | None:
    if not raw or len(raw) != 4:
        return None
    x, y, w, h = (max(0.0, min(1.0, float(v))) for v in raw)
    if w <= 0.01 or h <= 0.01:
        return None
    return (x, y, min(w, 1.0 - x), min(h, 1.0 - y))


class HydrogenClient:
    def __init__(self, runtime: GenAIRuntime) -> None:
        self.rt = runtime

    def extract_segment(
        self,
        segment: RoomSegment,
        segment_frames: list[Frame],
        segment_claims: list[SpokenClaim],
    ) -> list[ItemObservation]:
        """Extract `ItemObservation`s and search magnets from a single `RoomSegment`."""
        usable_frames = [f for f in segment_frames if f.usable] or segment_frames
        if not usable_frames:
            return []

        if self.rt.replay_mode:
            self.rt.record_call("hydrogen")
            return self._replay_segment_observations(segment, usable_frames)

        selected = usable_frames[:14]
        frame_lookup = {f.frame_id: f for f in selected}
        parts: list[dict | str] = [
            f"Segment {segment.segment_id} ({segment.room_label}, {segment.start_ms}..{segment.end_ms}ms)."
        ]
        if segment_claims:
            claims_txt = "; ".join(f"@{c.ts_ms}ms: '{c.text}'" for c in segment_claims)
            parts.append(
                f"Spoken narration in this window (treat as unverified context; trust visual logos/labels over speech): {claims_txt}"
            )

        for fr in selected:
            parts.append(f"FRAME_ID={fr.frame_id} (ts_ms={fr.ts_ms})")
            parts.append(image_part(fr.path))

        parts.append(
            "List each physical item seen in these frames using exact `FRAME_ID` references."
        )

        parsed = self.rt.structured_interaction(
            role="hydrogen",
            model=self.rt.hydrogen_model,
            input_parts=parts,
            schema_cls=_SegmentExtractionSchema,
            system_instruction=HYDROGEN_SYSTEM_INSTRUCTION,
        )

        results: list[ItemObservation] = []
        default_frame = selected[len(selected) // 2]
        for idx, raw in enumerate(parsed.observations):
            fr = frame_lookup.get(raw.frame_id, default_frame)
            ev: list[Literal["visual", "spoken", "label_read", "catalog_match", "user_answer"]] = ["visual"]
            model_txt = raw.model_text.strip() if (raw.model_text and raw.has_legible_label) else raw.model_text
            if model_txt:
                ev.append("label_read")
            results.append(
                ItemObservation(
                    obs_id=f"{segment.segment_id}_obs_{idx + 1:02d}",
                    segment_id=segment.segment_id,
                    frame_id=fr.frame_id,
                    ts_ms=fr.ts_ms,
                    bbox=_clamp_bbox(raw.bbox),
                    category=raw.category,
                    description=raw.description,
                    brand=raw.brand,
                    model_text=model_txt,
                    quantity=max(1, raw.quantity),
                    condition=raw.condition,
                    evidence=ev,
                    catalog_queries=[q[:120] for q in raw.catalog_queries if q.strip()][:3],
                    produced_by=self.rt.hydrogen_model,
                )
            )
        return results

    def extract_segments_parallel(
        self,
        segments: list[RoomSegment],
        all_frames: list[Frame],
        all_claims: list[SpokenClaim],
        max_concurrent: int = 4,
    ) -> list[ItemObservation]:
        """Fan out `extract_segment` concurrently across all `RoomSegment`s."""
        if not segments or not all_frames:
            return []

        tasks: list[tuple[RoomSegment, list[Frame], list[SpokenClaim]]] = []
        for i, seg in enumerate(segments):
            s_frames = [
                f for f in all_frames if seg.start_ms <= f.ts_ms <= seg.end_ms
            ]
            if not s_frames and i == len(segments) - 1:
                s_frames = all_frames[-min(4, len(all_frames)) :]
            elif not s_frames:
                s_frames = all_frames[: min(4, len(all_frames))]

            s_claims = [
                c for c in all_claims if (seg.start_ms - 2000) <= c.ts_ms <= (seg.end_ms + 2000)
            ]
            tasks.append((seg, s_frames, s_claims))

        collected: list[ItemObservation] = []
        workers = max(1, min(max_concurrent, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fut_map = {
                pool.submit(self.extract_segment, seg, s_fr, s_cl): seg.segment_id
                for seg, s_fr, s_cl in tasks
            }
            for fut in as_completed(fut_map):
                collected.extend(fut.result())

        collected.sort(key=lambda o: (o.ts_ms, o.obs_id))
        return collected

    def extract_followup_clip(
        self,
        request: CaptureRequest,
        frames: list[Frame],
        target_item: LedgerItem | None,
    ) -> list[ItemObservation]:
        """Inspect a short follow-up close-up clip (e.g., bottom laptop label) using Hydrogen."""
        if not frames:
            return []

        sharp_frames = sorted(frames, key=lambda f: f.blur_score, reverse=True)[:6]
        best_fr = sharp_frames[0]

        if self.rt.replay_mode:
            self.rt.record_call("hydrogen")
            cat = target_item.category if target_item else "laptop"
            brand = target_item.brand if (target_item and target_item.brand) else "Apple"
            model_code = "A2992" if cat == "laptop" else "880066-0100"
            return [
                ItemObservation(
                    obs_id=f"followup_{request.request_id}_obs_01",
                    segment_id="followup",
                    frame_id=best_fr.frame_id,
                    ts_ms=best_fr.ts_ms,
                    bbox=(0.15, 0.18, 0.70, 0.62),
                    category=cat,
                    description=f"Close-up chassis label reading Model {model_code}",
                    brand=brand,
                    model_text=model_code,
                    quantity=target_item.quantity if target_item else 1,
                    condition="good",
                    evidence=["visual", "label_read"],
                    catalog_queries=[
                        f"{brand} {model_code}",
                        f"{brand} MacBook Pro 14 {model_code}" if cat == "laptop" else f"{brand} {cat} {model_code}",
                    ],
                    produced_by=self.rt.hydrogen_model,
                )
            ]

        parts: list[dict | str] = [
            f"Follow-up capture for request: '{request.instruction}'. "
            f"Target item: {target_item.display_name if target_item else request.item_name or 'unknown'}. "
            "Read any visibly legible hardware model number / SKU / regulatory model identifier accurately."
        ]
        frame_lookup = {f.frame_id: f for f in sharp_frames}
        for fr in sharp_frames:
            parts.append(f"FRAME_ID={fr.frame_id} (blur_score={fr.blur_score})")
            parts.append(image_part(fr.path))

        parsed = self.rt.structured_interaction(
            role="hydrogen",
            model=self.rt.hydrogen_model,
            input_parts=parts,
            schema_cls=_SegmentExtractionSchema,
            system_instruction=HYDROGEN_SYSTEM_INSTRUCTION,
        )

        obs_list: list[ItemObservation] = []
        for idx, raw in enumerate(parsed.observations):
            fr = frame_lookup.get(raw.frame_id, best_fr)
            ev: list[Literal["visual", "spoken", "label_read", "catalog_match", "user_answer"]] = ["visual"]
            if raw.model_text:
                ev.append("label_read")
            obs_list.append(
                ItemObservation(
                    obs_id=f"followup_{request.request_id}_{idx + 1:02d}",
                    segment_id="followup",
                    frame_id=fr.frame_id,
                    ts_ms=fr.ts_ms,
                    bbox=_clamp_bbox(raw.bbox),
                    category=raw.category or (target_item.category if target_item else "item"),
                    description=raw.description,
                    brand=raw.brand or (target_item.brand if target_item else None),
                    model_text=raw.model_text,
                    quantity=max(1, raw.quantity),
                    condition=raw.condition,
                    evidence=ev,
                    catalog_queries=[q[:120] for q in raw.catalog_queries if q.strip()][:3],
                    produced_by=self.rt.hydrogen_model,
                )
            )
        return obs_list

    def _replay_segment_observations(
        self,
        segment: RoomSegment,
        frames: list[Frame],
    ) -> list[ItemObservation]:
        f0 = frames[0]
        f_last = frames[-1]
        if segment.segment_id.endswith("seg_01"):
            return [
                ItemObservation(
                    obs_id=f"{segment.segment_id}_obs_01",
                    segment_id=segment.segment_id,
                    frame_id=f0.frame_id,
                    ts_ms=f0.ts_ms,
                    bbox=(0.12, 0.08, 0.52, 0.65),
                    category="monitor",
                    description="Dell 27-inch 4K USB-C Hub Monitor on desk stand",
                    brand="Dell",
                    model_text="U2723QE" if f0.blur_score >= 12.0 and f0.brightness >= 28.0 else None,
                    quantity=1,
                    condition="good",
                    evidence=["visual", "label_read"],
                    catalog_queries=["Dell U2723QE", "Dell UltraSharp 27 4K Monitor"],
                ),
                ItemObservation(
                    obs_id=f"{segment.segment_id}_obs_02",
                    segment_id=segment.segment_id,
                    frame_id=f_last.frame_id,
                    ts_ms=f_last.ts_ms,
                    bbox=(0.35, 0.58, 0.42, 0.36),
                    category="laptop",
                    description="Apple MacBook Pro silver/space-black aluminum laptop, bottom model label not visible",
                    brand="Apple",
                    model_text=None,
                    quantity=1,
                    condition="good",
                    evidence=["visual"],
                    catalog_queries=["Apple MacBook Pro 14", "Apple MacBook Pro 16", "Apple MacBook Air"],
                ),
                ItemObservation(
                    obs_id=f"{segment.segment_id}_obs_03",
                    segment_id=segment.segment_id,
                    frame_id=f_last.frame_id,
                    ts_ms=f_last.ts_ms,
                    bbox=(0.68, 0.68, 0.28, 0.18),
                    category="keyboard",
                    description="Compact 75% aluminum wireless mechanical keyboard",
                    brand=None,
                    model_text=None,
                    quantity=1,
                    condition="good",
                    evidence=["visual"],
                    catalog_queries=["Keychron Q1 Pro mechanical keyboard", "75% aluminum mechanical keyboard"],
                ),
            ]
        return [
            # Sighting of the same keyboard in seg_02 tests cross-segment deduplication!
            ItemObservation(
                obs_id=f"{segment.segment_id}_obs_01",
                segment_id=segment.segment_id,
                frame_id=f0.frame_id,
                ts_ms=f0.ts_ms,
                bbox=(0.05, 0.66, 0.25, 0.18),
                category="keyboard",
                description="Compact 75% mechanical keyboard seen from right angle",
                brand=None,
                model_text=None,
                quantity=1,
                condition="good",
                evidence=["visual"],
                catalog_queries=["75% mechanical keyboard"],
            ),
            ItemObservation(
                obs_id=f"{segment.segment_id}_obs_02",
                segment_id=segment.segment_id,
                frame_id=f0.frame_id,
                ts_ms=f0.ts_ms,
                bbox=(0.32, 0.44, 0.24, 0.34),
                category="headphones",
                description="Bose over-ear wireless noise-cancelling headphones (BOSE logo visible on earcup)",
                brand="Bose",
                model_text=None,
                quantity=1,
                condition="good",
                evidence=["visual"],
                catalog_queries=["Bose QuietComfort Ultra Headphones", "Bose QuietComfort Wireless Headphones"],
            ),
            ItemObservation(
                obs_id=f"{segment.segment_id}_obs_03",
                segment_id=segment.segment_id,
                frame_id=f_last.frame_id,
                ts_ms=f_last.ts_ms,
                bbox=(0.56, 0.54, 0.32, 0.34),
                category="camera",
                description="Sony Alpha a7 IV full-frame mirrorless camera body (ILCE-7M4)",
                brand="Sony",
                model_text="ILCE-7M4" if f_last.blur_score >= 12.0 and f_last.brightness >= 28.0 else None,
                quantity=1,
                condition="good",
                evidence=["visual", "label_read"],
                catalog_queries=["Sony ILCE-7M4", "Sony Alpha a7 IV body"],
            ),
            ItemObservation(
                obs_id=f"{segment.segment_id}_obs_04",
                segment_id=segment.segment_id,
                frame_id=f_last.frame_id,
                ts_ms=f_last.ts_ms,
                bbox=(0.62, 0.58, 0.20, 0.28),
                category="lens",
                description="Sony FE 24-70mm f/2.8 GM II zoom lens mounted on camera",
                brand="Sony",
                model_text="SEL2470GM2" if f_last.blur_score >= 12.0 and f_last.brightness >= 28.0 else None,
                quantity=1,
                condition="good",
                evidence=["visual", "label_read"],
                catalog_queries=["Sony SEL2470GM2", "Sony FE 24-70mm f/2.8 GM II"],
            ),
        ]
