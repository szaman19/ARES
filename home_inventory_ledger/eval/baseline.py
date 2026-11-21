"""5 Sigma Baselines: B0 (single Gemini 4 Carbon call) and B1 (single call + top-1 Shop search).

Why both baselines matter (PLAN_v2 §3.1):
- `B0` tests what people naturally build first: one frontier multimodal model call on the full video
  asking for a JSON inventory with replacement prices. It invents prices, double-counts across
  angles, and trusts spoken narration blindly.
- `B1` gives `B0` access to Shop MCP (`search_catalog` top-1 result), isolating the value of
  *deterministic verification + multi-agent resolution + Decision-1 matching* from mere search access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

from pydantic import BaseModel, Field

from commerce.shop_mcp import ShopMCPClient
from contracts import Frame
from ingest.audio import extract_audio_wav
from ingest.frames import extract_frames
from models.provider import GenAIRuntime, audio_part, image_part


class _BaselineRowSchema(BaseModel):
    category: str
    display_name: str
    brand: str | None = None
    model: str | None = None
    quantity: int = 1
    replacement_price_usd: float
    spoken_conflict_detected: bool = False


class _BaselineOutputSchema(BaseModel):
    items: list[_BaselineRowSchema] = Field(default_factory=list)


@dataclass
class BaselinePrediction:
    system_name: str  # "B0" or "B1"
    rows: list[dict] = field(default_factory=list)
    total_usd: float = 0.0
    elapsed_seconds: float = 0.0
    model_calls: int = 0
    shop_calls: int = 0


def run_b0_single_call(
    video_path: str | Path,
    work_dir: str | Path,
    runtime: GenAIRuntime | None = None,
) -> BaselinePrediction:
    """B0: Single `gemini-4-carbon` call on the full video returning itemized JSON + prices."""
    rt = runtime or GenAIRuntime()
    t0 = time.perf_counter()
    wdir = Path(work_dir) / "b0_ingest"
    frames, _ = extract_frames(video_path, wdir, sample_fps=1.0, phash_hamming_thresh=2, max_frames=24)
    wav_path = extract_audio_wav(video_path, wdir / "audio.wav")

    if rt.replay_mode:
        rt.record_call("carbon")
        rows = _replay_b0_rows(frames)
        elapsed = round(time.perf_counter() - t0, 2)
        return BaselinePrediction(
            system_name="B0",
            rows=rows,
            total_usd=round(sum(r["price_usd"] * r["quantity"] for r in rows), 2),
            elapsed_seconds=elapsed,
            model_calls=1,
            shop_calls=0,
        )

    parts: list[dict | str] = []
    vid_part = rt.upload_video_part(video_path)
    if vid_part is not None:
        parts.append(vid_part)
    else:
        for fr in frames[:16]:
            parts.append(image_part(fr.path))
        if wav_path:
            parts.append(audio_part(wav_path))

    parts.append(
        "List every distinct item in this video with brand/model if known, quantity, "
        "and an estimated new replacement price in USD. Return JSON."
    )

    out = rt.structured_interaction(
        role="carbon",
        model=rt.carbon_model,
        input_parts=parts,
        schema_cls=_BaselineOutputSchema,
    )

    rows = [
        {
            "category": r.category,
            "display_name": r.display_name,
            "brand": r.brand,
            "model": r.model,
            "quantity": max(1, r.quantity),
            "price_usd": float(r.replacement_price_usd),
            "price_status": "POINT_UNVERIFIED",
            "listing_url": None,
            "narration_conflict": r.spoken_conflict_detected,
        }
        for r in out.items
    ]
    elapsed = round(time.perf_counter() - t0, 2)
    return BaselinePrediction(
        system_name="B0",
        rows=rows,
        total_usd=round(sum(r["price_usd"] * r["quantity"] for r in rows), 2),
        elapsed_seconds=elapsed,
        model_calls=1,
        shop_calls=0,
    )


def run_b1_single_call_plus_top1_search(
    video_path: str | Path,
    work_dir: str | Path,
    runtime: GenAIRuntime | None = None,
    shop_client: ShopMCPClient | None = None,
    b0_result: BaselinePrediction | None = None,
) -> BaselinePrediction:
    """B1: Takes B0's single-call items and prices each from the naive top-1 Shop MCP result."""
    rt = runtime or GenAIRuntime()
    shop = shop_client or ShopMCPClient(replay_mode=rt.replay_mode)
    t0 = time.perf_counter()

    base = b0_result or run_b0_single_call(video_path, work_dir, runtime=rt)
    b1_rows: list[dict] = []
    shop_before = shop.call_count

    for row in base.rows:
        q_parts = [p for p in (row.get("brand"), row.get("model"), row.get("display_name")) if p]
        query = " ".join(q_parts)[:100] or row["category"]
        hits = shop.search_catalog(query, limit=1)
        r_copy = dict(row)
        if hits:
            # B1 blindly accepts top-1 Shop result (including $24 case traps or wrong Sony model!)
            r_copy["price_usd"] = hits[0].price
            r_copy["listing_url"] = hits[0].url
            r_copy["listing_title"] = hits[0].title
        b1_rows.append(r_copy)

    elapsed = round(base.elapsed_seconds + (time.perf_counter() - t0), 2)
    return BaselinePrediction(
        system_name="B1",
        rows=b1_rows,
        total_usd=round(sum(r["price_usd"] * r["quantity"] for r in b1_rows), 2),
        elapsed_seconds=elapsed,
        model_calls=base.model_calls,
        shop_calls=shop.call_count - shop_before,
    )


def _replay_b0_rows(frames: list[Frame]) -> list[dict]:
    """Representative single-call output on `dev_desk_walkthrough.mp4` used strictly in `--replay` mode.

    Exhibits the three empirically observed single-pass failure modes:
    - Trusts narrator saying 'Sony headphones' (misses BOSE earcup logo conflict)
    - Guesses unshown MacBook model ('A2992' / '$1,599' invented price with no listing)
    - Double-counts mechanical keyboard seen across left & right camera pan
    """
    _ = frames
    return [
        {
            "category": "monitor",
            "display_name": "Dell 27-inch 4K Monitor",
            "brand": "Dell",
            "model": "U2723QE",
            "quantity": 1,
            "price_usd": 799.0,  # ~38% off real $579.99 listing -> hallucinated price
            "price_status": "POINT_UNVERIFIED",
            "listing_url": None,
            "narration_conflict": False,
        },
        {
            "category": "laptop",
            "display_name": "Apple MacBook Pro 14",
            "brand": "Apple",
            "model": "A2779",  # Guessed wrong generation model number not visible in walkthrough!
            "quantity": 1,
            "price_usd": 1599.0,  # Invented point price
            "price_status": "POINT_UNVERIFIED",
            "listing_url": None,
            "narration_conflict": False,
        },
        {
            "category": "keyboard",
            "display_name": "Mechanical Keyboard",
            "brand": None,
            "model": None,
            "quantity": 1,
            "price_usd": 165.0,
            "price_status": "POINT_UNVERIFIED",
            "listing_url": None,
            "narration_conflict": False,
        },
        {
            "category": "keyboard",
            "display_name": "Wireless Compact Keyboard",  # Duplicate across pan!
            "brand": None,
            "model": None,
            "quantity": 1,
            "price_usd": 85.0,
            "price_status": "POINT_UNVERIFIED",
            "listing_url": None,
            "narration_conflict": False,
        },
        {
            "category": "headphones",
            "display_name": "Sony Wireless Headphones",  # Blindly trusted spoken 'Sony' claim!
            "brand": "Sony",
            "model": "WH-1000XM4",
            "quantity": 1,
            "price_usd": 295.0,
            "price_status": "POINT_UNVERIFIED",
            "listing_url": None,
            "narration_conflict": False,
        },
        {
            "category": "camera",
            "display_name": "Sony a7 IV Camera",
            "brand": "Sony",
            "model": "ILCE-7M4",
            "quantity": 1,
            "price_usd": 2498.0,
            "price_status": "POINT_UNVERIFIED",
            "listing_url": None,
            "narration_conflict": False,
        },
    ]
