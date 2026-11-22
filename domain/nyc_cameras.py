"""Live NYC DOT Street-Camera ARES Search Demo (`domain/nyc_cameras.py`).

Uses ARES's parallel `ARESRunner` + immutable `NoteStore` + `fusion/patch.py` guardrails +
`gemini-4-hydrogen` vision + `gemini-decision-1` counterfactual probing over live NYC DOT
intersection cameras (`https://webcams.nyctmc.org/api/cameras`).

Workflow (only ~9 live JPEG fetches instead of scanning all ~964 cameras):
  1. Fetch 1 JSON catalog of ~964 online NYC cameras (GPS + borough + intersection name).
  2. Round 1 (Coarse Parallel Sweep): farthest-point spatial sample of `k=6` cameras
     fetched & inspected in parallel via `ARESRunner(max_concurrent=6)`.
  3. Round 2 (Spatial Zoom-In): pick the highest-signal Round-1 intersection, fetch its
     `k=3` geographically nearest neighboring cameras in parallel to confirm the area.
  4. Fusion Guardrails (`fusion/patch.py`): validate & merge grounded board fields citing
     immutable `Note` IDs + SHA-256-verified live JPEG `ArtifactRef`s.
  5. Counterfactual Probe (`gemini-decision-1`): evaluate the best intersection + measure
     the exact confidence drop when each camera's evidence is hidden (`"unknown"`).
"""

from __future__ import annotations

import base64
from collections import OrderedDict
import concurrent.futures
import hashlib
import io
import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
import matplotlib
import matplotlib.image as mpimg
from dotenv import load_dotenv
from google import genai
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from ares.engine import ARESRunner, ProcessorResult, TaskGraph, TaskSpec
from ares.store import NoteStore
from contracts import (
    ArtifactRef,
    BoardDecision,
    DecisionQuestion,
    InvestigationBoard,
    Note,
)
from fusion.patch import validate_and_merge_patch
from models.api_fallback import (
    call_fallback_vision_json,
    get_fallback_vision_providers,
    get_gemini_api_keys,
    has_any_vision_api_key,
)
from models.decision_probe import DecisionProbeClient, ProbeResult
from models.gemini_provider import ModelTelemetry, _strip_json_fences, extract_telemetry

matplotlib.use("Agg")

DEFAULT_VISION_MODEL = os.getenv("ARES_VISION_MODEL", "gemini-4-hydrogen")

NYC_CAMERAS_API = "https://webcams.nyctmc.org/api/cameras"
CATALOG_CACHE_TTL_S = float(os.environ.get("NYC_CATALOG_CACHE_TTL_S", "1800"))  # 30 min
FRAME_CACHE_TTL_S = float(os.environ.get("NYC_FRAME_CACHE_TTL_S", "180"))  # 3 min
FRAME_CACHE_MAX_ITEMS = 1024
ORIGIN_MAX_CONCURRENT = int(os.environ.get("NYC_ORIGIN_MAX_CONCURRENT", "3"))
ORIGIN_MIN_INTERVAL_S = float(os.environ.get("NYC_ORIGIN_MIN_INTERVAL_S", "0.18"))

_CATALOG_LOCK = threading.Lock()
_CATALOG_CACHE: tuple[float, list[dict[str, Any]]] | None = None

_FRAME_CACHE_LOCK = threading.Lock()
_FRAME_CACHE: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
_CAM_LOCKS_GUARD = threading.Lock()
_CAM_LOCKS: dict[str, threading.Lock] = {}

_ORIGIN_SEMAPHORE = threading.BoundedSemaphore(value=max(1, ORIGIN_MAX_CONCURRENT))
_ORIGIN_PACE_LOCK = threading.Lock()
_ORIGIN_LAST_FETCH_TS = 0.0


def _pace_origin_request() -> None:
    """Enforce a polite minimum interval between outbound requests to webcams.nyctmc.org."""
    global _ORIGIN_LAST_FETCH_TS
    if ORIGIN_MIN_INTERVAL_S <= 0:
        return
    with _ORIGIN_PACE_LOCK:
        now = time.monotonic()
        wait_s = (_ORIGIN_LAST_FETCH_TS + ORIGIN_MIN_INTERVAL_S) - now
        if wait_s > 0:
            time.sleep(wait_s)
        _ORIGIN_LAST_FETCH_TS = time.monotonic()


def _get_camera_lock(cam_id: str) -> threading.Lock:
    with _CAM_LOCKS_GUARD:
        lock = _CAM_LOCKS.get(cam_id)
        if lock is None:
            lock = threading.Lock()
            _CAM_LOCKS[cam_id] = lock
        return lock


def _get_raw_camera_catalog(timeout_s: float = 12.0, *, bypass_cache: bool = False) -> list[dict[str, Any]]:
    global _CATALOG_CACHE
    now = time.monotonic()
    if not bypass_cache and _CATALOG_CACHE is not None:
        fetched_at, payload = _CATALOG_CACHE
        if now - fetched_at < CATALOG_CACHE_TTL_S:
            return payload

    with _CATALOG_LOCK:
        now = time.monotonic()
        if not bypass_cache and _CATALOG_CACHE is not None:
            fetched_at, payload = _CATALOG_CACHE
            if now - fetched_at < CATALOG_CACHE_TTL_S:
                return payload
        resp = httpx.get(
            NYC_CAMERAS_API,
            timeout=timeout_s,
            headers={"User-Agent": "ARES-Hackathon-Demo/1.0"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        raw_list = resp.json()
        _CATALOG_CACHE = (time.monotonic(), raw_list)
        return raw_list


def _get_camera_jpeg_bytes(
    camera: CameraInfo,
    *,
    timeout_s: float = 10.0,
    bypass_cache: bool = False,
) -> bytes:
    now = time.monotonic()
    if not bypass_cache:
        with _FRAME_CACHE_LOCK:
            cached = _FRAME_CACHE.get(camera.id)
            if cached is not None and (now - cached[0]) < FRAME_CACHE_TTL_S:
                _FRAME_CACHE.move_to_end(camera.id)
                return cached[1]

    cam_lock = _get_camera_lock(camera.id)
    with cam_lock:
        now = time.monotonic()
        if not bypass_cache:
            with _FRAME_CACHE_LOCK:
                cached = _FRAME_CACHE.get(camera.id)
                if cached is not None and (now - cached[0]) < FRAME_CACHE_TTL_S:
                    _FRAME_CACHE.move_to_end(camera.id)
                    return cached[1]

        with _ORIGIN_SEMAPHORE:
            _pace_origin_request()
            resp = httpx.get(
                camera.image_url,
                timeout=timeout_s,
                headers={"User-Agent": "ARES-Hackathon-Demo/1.0"},
                follow_redirects=True,
            )
        resp.raise_for_status()
        img_bytes = resp.content
        if len(img_bytes) < 512:
            raise ValueError(f"Camera {camera.name} returned empty/tiny image ({len(img_bytes)} bytes).")

        with _FRAME_CACHE_LOCK:
            _FRAME_CACHE[camera.id] = (time.monotonic(), img_bytes)
            _FRAME_CACHE.move_to_end(camera.id)
            while len(_FRAME_CACHE) > FRAME_CACHE_MAX_ITEMS:
                _FRAME_CACHE.popitem(last=False)
        return img_bytes


def clear_nyc_camera_caches() -> None:
    """Clear in-memory catalog and JPEG caches (useful for tests)."""
    global _CATALOG_CACHE
    with _CATALOG_LOCK:
        _CATALOG_CACHE = None
    with _FRAME_CACHE_LOCK:
        _FRAME_CACHE.clear()
COLOR_CERTIFIED = "#1b9e77"
COLOR_UNCERTIFIED = "#d95f02"
COLOR_NEUTRAL = "#666666"

# Substrings indicating highways, bridges, tunnels, or ramps (excluded when street_only=True)
_HIGHWAY_BRIDGE_TOKENS: tuple[str, ...] = (
    "pkwy",
    "parkway",
    "expwy",
    "expressway",
    "expy",
    "bridge",
    "brg",
    "tunnel",
    "tunl",
    "hwy",
    "highway",
    "turnpike",
    "tpke",
    "bqe",
    "fdr",
    "hhp",
    "hrp",
    "bpu",
    "wbb",
    "qbb",
    "gwb",
    "vnb",
    "rfk",
    "lie",
    "sie",
    "gcp",
    "belt",
    "deegan",
    "cross bronx",
    "bruckner",
    "sheridan",
    "van wyck",
    "whitestone",
    "throgs neck",
    "verrazzano",
    "triboro",
    "holland",
    "lincoln",
    "midtown tun",
    "battery",
    "west side",
    "west st",
    "south st",
    "12 ave",
    "207 st @ 9 ave",
    "henry hudson",
    "harlem river dr",
    "fdr dr",
    "ramp",
    "toll",
    "i-95",
    "i-278",
    "i-495",
    "i-678",
    "i-87",
    "north bound",
    "south bound",
    "east bound",
    "west bound",
    "nb ",
    "sb ",
    "eb ",
    "wb ",
)


def is_street_intersection_camera(name: str) -> bool:
    """Return True if camera name looks like a local street intersection (not highway/bridge/tunnel)."""
    lowered = f" {name.lower()} "
    if "@" not in lowered:
        return False
    return not any(tok in lowered for tok in _HIGHWAY_BRIDGE_TOKENS)


class GeminiSoloBaseline(BaseModel):
    """What Gemini guesses on its own without ARES live camera search or verification."""

    predicted_intersection: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    spot_box: list[int] | None = Field(
        default=None,
        description="Optional [ymin, xmin, ymax, xmax] bounding box in 0..1000 normalized coordinates circling where Gemini Solo thinks the spot is.",
    )
    has_live_evidence: bool = False
    wall_clock_s: float = 0.0
    telemetry: ModelTelemetry = Field(default_factory=ModelTelemetry)


class CameraInfo(BaseModel):
    """Metadata for a single live NYC DOT street camera."""

    id: str
    name: str
    area: str
    latitude: float
    longitude: float
    image_url: str


class CameraObservation(BaseModel):
    """Structured vision assessment returned by `gemini-4-hydrogen` for one live camera."""

    verdict: Literal["strong_match", "partial_match", "no_match", "unclear_camera"] = "unclear_camera"
    match_score: float = Field(default=0.0, ge=0.0, le=1.0)
    curb_visibility: Literal["clear", "partial", "blocked_or_blurry"] = "blocked_or_blurry"
    is_legal_parking_street: bool = False
    open_curb_spot_visible: bool = False
    spot_box: list[int] | None = Field(
        default=None,
        description="Optional [ymin, xmin, ymax, xmax] bounding box in 0..1000 normalized coordinates circling the open curb parking space.",
    )
    disqualifier: Literal[
        "none",
        "red_bus_lane",
        "green_bike_lane",
        "bridge_or_highway",
        "guardrail_no_curb",
        "crosswalk_or_no_standing",
        "obstructed_camera",
    ] = "none"
    summary: str = ""
    key_detail: str = ""


class InspectedCamera(BaseModel):
    """Combined live camera metadata, verified JPEG artifact, and vision note."""

    round_num: int
    camera: CameraInfo
    artifact: ArtifactRef
    note_id: str
    observation: CameraObservation
    field_name: str
    distance_m_from_anchor: float | None = None


class NYCDemoResult(BaseModel):
    """End-to-end result of a live NYC street-camera ARES search."""

    query: str
    area: str
    street_only: bool = True
    timestamp_utc: str
    total_online_cameras_in_area: int
    cameras_inspected_count: int
    anchor_camera: CameraInfo | None = None
    inspected: list[InspectedCamera] = Field(default_factory=list)
    board: InvestigationBoard
    question: DecisionQuestion | None = None
    probe_result: ProbeResult
    solo_baseline: GeminiSoloBaseline | None = None
    contact_sheet_path: str
    wall_clock_s: float
    telemetry: ModelTelemetry = Field(default_factory=ModelTelemetry)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two WGS84 coordinates."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def fetch_camera_catalog(
    *,
    area: str | None = "Manhattan",
    keyword: str | None = None,
    street_only: bool = True,
    timeout_s: float = 12.0,
    bypass_cache: bool = False,
) -> list[CameraInfo]:
    """Fetch online NYC DOT cameras filtered by borough (`area`), street_only, and optional keyword."""
    raw_list = _get_raw_camera_catalog(timeout_s=timeout_s, bypass_cache=bypass_cache)

    area_norm = area.strip().lower() if area and area.strip().lower() != "all" else None
    kw_norm = keyword.strip().lower() if keyword and keyword.strip() else None

    cameras: list[CameraInfo] = []
    for item in raw_list:
        if str(item.get("isOnline", "")).lower() != "true":
            continue
        cam_area = str(item.get("area") or "").strip()
        cam_name = str(item.get("name") or "").strip()
        if area_norm and cam_area.lower() != area_norm:
            continue
        if street_only and not is_street_intersection_camera(cam_name):
            continue
        if kw_norm and kw_norm not in cam_name.lower():
            continue
        lat = item.get("latitude")
        lon = item.get("longitude")
        if lat is None or lon is None:
            continue
        cid = str(item.get("id") or "").strip()
        if not cid:
            continue
        img_url = str(item.get("imageUrl") or f"{NYC_CAMERAS_API}/{cid}/image")
        cameras.append(
            CameraInfo(
                id=cid,
                name=cam_name or cid[:8],
                area=cam_area or "NYC",
                latitude=float(lat),
                longitude=float(lon),
                image_url=img_url,
            )
        )

    cameras.sort(key=lambda c: (c.area, c.name, c.id))
    return cameras


def select_spread_cameras(cameras: list[CameraInfo], k: int = 6) -> list[CameraInfo]:
    """Deterministically pick `k` spatially spread-out cameras via farthest-point sampling."""
    if len(cameras) <= k:
        return list(cameras)
    # Start near the geographic centroid of the candidate pool
    mean_lat = sum(c.latitude for c in cameras) / len(cameras)
    mean_lon = sum(c.longitude for c in cameras) / len(cameras)
    first = min(
        cameras,
        key=lambda c: haversine_m(c.latitude, c.longitude, mean_lat, mean_lon),
    )
    selected = [first]
    selected_ids = {first.id}

    while len(selected) < k:
        best_cam = None
        best_min_dist = -1.0
        for cam in cameras:
            if cam.id in selected_ids:
                continue
            min_d = min(
                haversine_m(cam.latitude, cam.longitude, s.latitude, s.longitude)
                for s in selected
            )
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_cam = cam
        if best_cam is None:
            break
        selected.append(best_cam)
        selected_ids.add(best_cam.id)
    return selected


def select_nearest_cameras(
    anchor: CameraInfo,
    cameras: list[CameraInfo],
    *,
    k: int = 3,
    exclude_ids: set[str] | None = None,
) -> list[tuple[CameraInfo, float]]:
    """Pick the `k` geographically nearest online cameras to `anchor`."""
    skip = set(exclude_ids or set()) | {anchor.id}
    ranked = [
        (cam, haversine_m(anchor.latitude, anchor.longitude, cam.latitude, cam.longitude))
        for cam in cameras
        if cam.id not in skip
    ]
    ranked.sort(key=lambda pair: (pair[1], pair[0].id))
    return ranked[:k]


def fetch_camera_jpeg(
    camera: CameraInfo,
    out_dir: Path,
    *,
    timeout_s: float = 10.0,
    bypass_cache: bool = False,
) -> ArtifactRef:
    """Download one live JPEG frame (with TTL + singleflight cache), verify bytes, compute SHA-256, and return ArtifactRef."""
    img_bytes = _get_camera_jpeg_bytes(
        camera,
        timeout_s=timeout_s,
        bypass_cache=bypass_cache,
    )

    sha256 = hashlib.sha256(img_bytes).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in camera.name)[:32].strip("_")
    out_path = out_dir / f"cam_{safe_name}_{sha256[:10]}.jpg"
    out_path.write_bytes(img_bytes)

    return ArtifactRef(
        artifact_id=f"art_{sha256[:12]}",
        kind="image",
        path=str(out_path),
        sha256=sha256,
        mime_type="image/jpeg",
        produced_by_verifier=True,
    )


def _inspect_camera_with_hydrogen(
    client: Any,
    camera: CameraInfo,
    artifact: ArtifactRef,
    query: str,
    *,
    model: str = DEFAULT_VISION_MODEL,
) -> tuple[CameraObservation, ModelTelemetry]:
    """Inspect one live street camera JPEG with Gemini (multi-key fallback) or fallback vision APIs."""
    b64_img = base64.b64encode(Path(artifact.path).read_bytes()).decode("ascii")
    prompt = (
        f"Live NYC DOT traffic camera at: {camera.name} ({camera.area}).\n"
        f"User search target: {query}\n\n"
        "Inspect this live street camera photo STRICTLY and honestly:\n"
        "1. If the camera is dark, blurry, frozen, obstructed by poles/bars, or angled away, "
        "set verdict='unclear_camera', disqualifier='obstructed_camera', is_legal_parking_street=false.\n"
        "2. Check for NON-PARKING disqualifiers if the query asks about parking/curb spots:\n"
        "   - Red painted BUS ONLY / TRUCK ONLY lane -> disqualifier='red_bus_lane', is_legal_parking_street=false.\n"
        "   - Protected green bike lane along the curb -> disqualifier='green_bike_lane', is_legal_parking_street=false.\n"
        "   - Bridge approach, overpass, highway, divided roadway, or tunnel -> disqualifier='bridge_or_highway', is_legal_parking_street=false.\n"
        "   - Guardrail or concrete barrier with no sidewalk parking curb -> disqualifier='guardrail_no_curb', is_legal_parking_street=false.\n"
        "3. Set open_curb_spot_visible=true ONLY if you literally see an empty car-length space along a normal city parking curb (typically beside other legally parked cars).\n"
        "4. If open_curb_spot_visible=true (or partial_match with visible curb space), return spot_box=[ymin, xmin, ymax, xmax] "
        "in 0..1000 normalized image coordinates tightly circling that exact open curb parking spot; otherwise set spot_box=null.\n"
        "5. Keep summary to 1 concise factual sentence describing what you literally see."
    )
    system_instruction = (
        "You are a strict, skeptical street-camera verifier. Never count red bus lanes, "
        "green bike lanes, bridge approaches, highways, or guardrails as street parking. "
        "Output strict JSON matching the schema."
    )
    schema = CameraObservation.model_json_schema()

    clients = client if isinstance(client, list) else ([client] if client is not None else [])
    errors: list[str] = []
    obs: CameraObservation | None = None
    telemetry = ModelTelemetry()

    for idx, gem_client in enumerate(clients):
        try:
            t0 = time.perf_counter()
            interaction = gem_client.interactions.create(
                model=model,
                store=False,
                input=[
                    {"type": "text", "text": prompt},
                    {"type": "image", "data": b64_img, "mime_type": "image/jpeg"},
                ],
                system_instruction=system_instruction,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                generation_config={"thinking_level": "low"},
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            telemetry = extract_telemetry(interaction, latency_ms)
            raw = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
            obs = CameraObservation.model_validate_json(raw)
            break
        except Exception as exc:
            errors.append(f"gemini[{idx}]: {exc}")

    if obs is None:
        for provider, prov_key, prov_model in get_fallback_vision_providers():
            try:
                raw_text, p_toks, c_toks, latency_ms = call_fallback_vision_json(
                    provider,
                    prov_key,
                    prov_model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    b64_img=b64_img,
                    schema=schema,
                )
                raw = _strip_json_fences(raw_text)
                obs = CameraObservation.model_validate_json(raw)
                telemetry = ModelTelemetry(
                    prompt_tokens=p_toks,
                    completion_tokens=c_toks,
                    model_calls=1,
                    latency_ms=latency_ms,
                )
                break
            except Exception as exc:
                errors.append(f"{provider}({prov_model}): {exc}")

    if obs is None:
        raise RuntimeError(
            "All configured vision API keys/providers failed: " + " | ".join(errors or ["no keys configured"])
        )

    # Deterministic ARES verifier enforcement: disqualify non-parking road types in Python
    wants_parking = any(
        w in query.lower() for w in ("park", "curb", "spot", "street")
    )
    if obs.disqualifier == "obstructed_camera":
        obs.verdict = "unclear_camera"
        obs.match_score = min(obs.match_score, 0.15)
        obs.is_legal_parking_street = False
        obs.open_curb_spot_visible = False
        obs.spot_box = None
    elif wants_parking and (not obs.is_legal_parking_street or obs.disqualifier != "none"):
        obs.verdict = "no_match"
        obs.match_score = min(obs.match_score, 0.15)
        obs.open_curb_spot_visible = False
        obs.spot_box = None
    elif wants_parking and not obs.open_curb_spot_visible and obs.verdict == "strong_match":
        obs.verdict = "partial_match"
        obs.match_score = min(obs.match_score, 0.55)

    # Validate spot_box coordinates if present
    if obs.spot_box is not None:
        if len(obs.spot_box) != 4:
            obs.spot_box = None
        else:
            ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in obs.spot_box]
            if ymax <= ymin or xmax <= xmin:
                obs.spot_box = None
            else:
                obs.spot_box = [ymin, xmin, ymax, xmax]

    return obs, telemetry


def _safe_field_name(round_num: int, idx: int, cam_name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in cam_name)
    while "__" in slug:
        slug = slug.replace("__", "_")
    slug = slug.strip("_")[:28]
    return f"r{round_num}_{idx}_{slug}"


def render_nyc_contact_sheet(
    inspected: list[InspectedCamera],
    field_influences: dict[str, float],
    out_path: Path,
    *,
    query: str,
    decision_answer: str,
    decision_confidence: float,
) -> Path:
    """Render a deterministic headless Matplotlib Agg contact sheet of live NYC camera frames."""
    n = len(inspected)
    cols = 3
    rows = max(1, math.ceil(n / cols))
    fig = Figure(figsize=(9.0, max(5.5, 2.4 * rows + 1.0)), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(rows, cols, squeeze=False)

    short_query = query if len(query) <= 68 else query[:65] + "..."
    fig.suptitle(
        f"ARES Live NYC Camera Search — {short_query}",
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        f"Verdict: {decision_answer}  |  gemini-decision-1 Confidence: {decision_confidence:.1%}",
        ha="center",
        fontsize=9,
        color="#222222",
    )

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        if idx >= n:
            ax.axis("off")
            continue
        item = inspected[idx]
        try:
            img = mpimg.imread(item.artifact.path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image load error", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        verdict = item.observation.verdict
        if verdict == "strong_match":
            border_color = COLOR_CERTIFIED
        elif verdict == "partial_match":
            border_color = COLOR_UNCERTIFIED
        else:
            border_color = COLOR_NEUTRAL

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.2)

        inf = field_influences.get(item.field_name, 0.0)
        dist_txt = (
            f" ({int(item.distance_m_from_anchor)}m)"
            if item.distance_m_from_anchor is not None
            else ""
        )
        ax.set_title(
            f"R{item.round_num}: {item.camera.name[:28]}{dist_txt}\n"
            f"{verdict} ({item.observation.match_score:.2f}) | drop={inf:.2f}",
            fontsize=7.5,
            fontweight="bold",
            color=border_color,
            pad=4,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, metadata={"Software": "ARES"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf.getvalue())
    return out_path


def run_gemini_solo_baseline(
    *,
    query: str,
    area: str,
    round1_items: list[tuple[CameraInfo, ArtifactRef]] | None = None,
    model: str = "gemini-4-carbon",
) -> GeminiSoloBaseline:
    """Run naive Gemini solo on the EXACT SAME Round-1 live JPEGs (no ARES verifiers, no Round-2 zoom, no Decision-1)."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required.")
    client = genai.Client(api_key=api_key)

    class _SoloSchema(BaseModel):
        predicted_intersection: str
        confidence: float = Field(ge=0.0, le=1.0)
        reasoning: str
        spot_box: list[int] | None = Field(
            default=None,
            description="Optional [ymin, xmin, ymax, xmax] bounding box in 0..1000 normalized image coordinates tightly circling the open curb spot in your chosen image (or null if none_verified).",
        )

    cam_items = list(round1_items or [])
    if cam_items:
        cam_list_str = "\n".join(
            f"{i + 1}. {cam.name} ({cam.area})"
            for i, (cam, _) in enumerate(cam_items)
        )
        prompt_text = (
            f"You are given {len(cam_items)} live NYC traffic camera photos from {area}:\n"
            f"{cam_list_str}\n\n"
            f"User request: '{query}'\n\n"
            "Apply the exact same strict NYC street-parking rules:\n"
            "1. Reject cameras that are dark, blurry, frozen, obstructed, or angled away.\n"
            "2. Never count red painted BUS ONLY / TRUCK ONLY lanes, protected green bike lanes along the curb, "
            "bridge approaches, overpasses, highways, divided roadways, tunnels, or guardrails with no sidewalk curb.\n"
            "3. Only pick an intersection if you literally see an empty car-length space along a normal city parking curb "
            "(typically beside other legally parked cars); otherwise return 'none_verified'.\n\n"
            "Look at the attached images in order and return the single best intersection name "
            "(or 'none_verified'), your confidence (0.0 to 1.0), a 1-2 sentence explanation, "
            "and spot_box=[ymin, xmin, ymax, xmax] in 0..1000 normalized image coordinates "
            "tightly circling the exact spot in your chosen image (or null if none_verified)."
        )
        multimodal_input: Any = [{"type": "text", "text": prompt_text}]
        for _, art in cam_items:
            b64 = base64.b64encode(Path(art.path).read_bytes()).decode("ascii")
            multimodal_input.append(
                {"type": "image", "data": b64, "mime_type": art.mime_type or "image/jpeg"}
            )
    else:
        multimodal_input = (
            f"Right now in {area}, NYC, which specific street intersection best matches: '{query}'?\n"
            "Answer with an intersection name, confidence (0.0 to 1.0), and a 1-2 sentence explanation."
        )

    t0 = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        store=False,
        input=multimodal_input,
        system_instruction=(
            "You are a strict, skeptical street-camera verifier. Never count red bus lanes, "
            "green bike lanes, bridge approaches, highways, or guardrails as street parking. "
            "Respond in strict JSON matching the schema."
        ),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": _SoloSchema.model_json_schema(),
        },
        generation_config={"thinking_level": "low"},
    )
    elapsed_s = round(time.perf_counter() - t0, 2)
    tel = extract_telemetry(interaction, elapsed_s * 1000.0)
    raw = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
    parsed = _SoloSchema.model_validate_json(raw)
    valid_box: list[int] | None = None
    if parsed.spot_box is not None and len(parsed.spot_box) == 4:
        ymin, xmin, ymax, xmax = [max(0, min(1000, int(v))) for v in parsed.spot_box]
        if ymax > ymin and xmax > xmin:
            valid_box = [ymin, xmin, ymax, xmax]
    return GeminiSoloBaseline(
        predicted_intersection=parsed.predicted_intersection,
        confidence=parsed.confidence,
        reasoning=parsed.reasoning,
        spot_box=valid_box,
        has_live_evidence=bool(cam_items),
        wall_clock_s=elapsed_s,
        telemetry=tel,
    )


def load_snapshot_catalog(
    snapshot_dir: str | Path,
    *,
    area: str | None = "Manhattan",
    keyword: str | None = None,
) -> tuple[list[CameraInfo], dict[str, Path]]:
    """Load saved NYC street-intersection cameras and local JPEG paths from a captured snapshot."""
    sdir = Path(snapshot_dir)
    manifest_path = sdir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    area_norm = area.strip().lower() if area and area.strip().lower() != "all" else None
    kw_norm = keyword.strip().lower() if keyword and keyword.strip() else None

    cameras: list[CameraInfo] = []
    frame_map: dict[str, Path] = {}
    for entry in manifest.get("cameras", []):
        cam = CameraInfo.model_validate(entry["camera"])
        if area_norm and cam.area.lower() != area_norm:
            continue
        if kw_norm and kw_norm not in cam.name.lower():
            continue
        fpath = sdir / "frames" / str(entry["frame_file"])
        if not fpath.exists():
            continue
        cameras.append(cam)
        frame_map[cam.id] = fpath

    cameras.sort(key=lambda c: (c.area, c.name, c.id))
    return cameras, frame_map


def _artifact_from_local_jpeg(fpath: Path) -> ArtifactRef:
    img_bytes = fpath.read_bytes()
    sha256 = hashlib.sha256(img_bytes).hexdigest()
    return ArtifactRef(
        artifact_id=f"art_{sha256[:12]}",
        kind="image",
        path=str(fpath),
        sha256=sha256,
        mime_type="image/jpeg",
        produced_by_verifier=True,
    )


def run_live_nyc_search(
    *,
    query: str = "clear curb space and light traffic (no heavy congestion or blocked lanes)",
    area: str = "Manhattan",
    keyword: str | None = None,
    street_only: bool = True,
    include_solo_baseline: bool = False,
    round1_k: int = 12,
    round2_k: int = 6,
    center_lat: float | None = None,
    center_lon: float | None = None,
    radius_m: float | None = None,
    out_dir: str | Path = "benchmark/results/nyc_live",
    snapshot_dir: str | Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> NYCDemoResult:
    """Execute the 2-round parallel ARES search + `gemini-decision-1` counterfactual probe live (or from saved snapshot)."""
    load_dotenv()
    if not has_any_vision_api_key():
        raise RuntimeError(
            "No vision API key configured. Set GEMINI_API_KEY (or GEMINI_API_KEYS / OPENAI_API_KEY / ANTHROPIC_API_KEY / OPENROUTER_API_KEY)."
        )

    t_start = time.perf_counter()

    def _emit(step: str, message: str, **extra: Any) -> None:
        if on_event is not None:
            on_event(
                {
                    "step": step,
                    "message": message,
                    "elapsed_s": round(time.perf_counter() - t_start, 2),
                    **extra,
                }
            )

    dest = Path(out_dir)
    img_dir = dest / "frames"
    img_dir.mkdir(parents=True, exist_ok=True)

    use_geo = center_lat is not None and center_lon is not None
    eff_area = "all" if use_geo else area

    snapshot_frames: dict[str, Path] = {}
    if snapshot_dir is not None:
        _emit("catalog", f"Loading saved NYC street snapshot from {snapshot_dir} (area={eff_area})...")
        cameras, snapshot_frames = load_snapshot_catalog(
            snapshot_dir, area=eff_area, keyword=keyword
        )
        if not cameras:
            raise ValueError(
                f"No saved snapshot cameras found in {snapshot_dir} for area={eff_area!r}, keyword={keyword!r}."
            )
        _emit("catalog_done", f"Loaded {len(cameras)} saved street cameras.")
    else:
        _emit("catalog", f"GET {NYC_CAMERAS_API} (area={eff_area}, street_only={street_only})...")
        cameras = fetch_camera_catalog(area=eff_area, keyword=keyword, street_only=street_only)
        if not cameras:
            raise ValueError(
                f"No online NYC cameras found for area={eff_area!r}, keyword={keyword!r}, street_only={street_only}."
            )
        if use_geo and center_lat is not None and center_lon is not None:
            ranked_geo = sorted(
                cameras,
                key=lambda c: haversine_m(center_lat, center_lon, c.latitude, c.longitude),
            )
            eff_radius = float(radius_m) if radius_m is not None else 350.0
            in_radius = [
                c
                for c in ranked_geo
                if haversine_m(center_lat, center_lon, c.latitude, c.longitude) <= eff_radius
            ]
            if len(in_radius) >= max(3, min(round1_k, 6)):
                cameras = in_radius
                _emit(
                    "catalog_done",
                    f"Found {len(cameras)} live street cameras within {int(eff_radius)}m of your pin.",
                )
            else:
                # Auto-expand to nearest street cameras if pin is in a sparse spot
                cameras = ranked_geo[: max(round1_k + round2_k, 12)]
                farthest = int(haversine_m(center_lat, center_lon, cameras[-1].latitude, cameras[-1].longitude))
                _emit(
                    "catalog_done",
                    f"Only {len(in_radius)} cams inside {int(eff_radius)}m → expanded to {len(cameras)} nearest street cameras (up to {farthest}m).",
                )
        else:
            _emit("catalog_done", f"Found {len(cameras)} online street intersection cameras in {area}.")

    client = [genai.Client(api_key=k) for k in get_gemini_api_keys()]
    note_store = NoteStore()
    total_telemetry = ModelTelemetry()
    inspected: list[InspectedCamera] = []
    inspected_lock_items: list[tuple[int, int, CameraInfo, ArtifactRef, Note, CameraObservation, float | None]] = []

    def _run_camera_batch(
        round_num: int,
        batch: list[tuple[CameraInfo, float | None]],
        prefetched_arts: dict[str, ArtifactRef] | None = None,
    ) -> None:
        nonlocal total_telemetry
        cam_by_idx = {str(i): pair for i, pair in enumerate(batch)}
        batch_results: dict[str, tuple[CameraInfo, ArtifactRef, Note, CameraObservation, float | None, ModelTelemetry]] = {}

        def _processor(spec: TaskSpec, _ctx: dict[str, Any]) -> ProcessorResult:
            idx_key = str(spec.context["idx"])
            cam, dist_m = cam_by_idx[idx_key]
            if prefetched_arts and cam.id in prefetched_arts:
                art = prefetched_arts[cam.id]
            elif cam.id in snapshot_frames:
                _emit("camera_fetch", f"[R{round_num}] SNAPSHOT → {cam.name}")
                src_path = snapshot_frames[cam.id]
                copied = img_dir / src_path.name
                copied.write_bytes(src_path.read_bytes())
                art = _artifact_from_local_jpeg(copied)
            else:
                _emit(
                    "camera_fetch",
                    f"[R{round_num}] GET /api/cameras/{cam.id[:8]}.../image → {cam.name}",
                )
                art = fetch_camera_jpeg(cam, img_dir)
            obs, tel = _inspect_camera_with_hydrogen(client, cam, art, query)
            disq_note = f" [DISQUALIFIED: {obs.disqualifier}]" if obs.disqualifier != "none" else ""
            _emit(
                "camera_done",
                f"[R{round_num}] gemini-4-hydrogen ({tel.latency_ms/1000:.1f}s) → {cam.name}: {obs.verdict} ({obs.match_score:.2f}){disq_note}",
            )
            cert = "CERTIFIED" if obs.verdict != "unclear_camera" else "UNCERTIFIED"
            note = Note(
                note_id=note_store.next_note_id(),
                certification=cert,
                sources=[cam.image_url],
                artifacts=[art],
                verifier_name="nyc_dot_live_camera_verifier",
                hypothesis_family=f"round_{round_num}_camera_sweep",
                requested_under=round_num - 1,
                summary=f"{cam.name} ({cam.area}): {obs.verdict} (score={obs.match_score:.2f}) — {obs.summary}",
                evidence={
                    "camera_id": cam.id,
                    "camera_name": cam.name,
                    "area": cam.area,
                    "verdict": obs.verdict,
                    "match_score": obs.match_score,
                    "curb_visibility": obs.curb_visibility,
                    "is_legal_parking_street": obs.is_legal_parking_street,
                    "open_curb_spot_visible": obs.open_curb_spot_visible,
                    "disqualifier": obs.disqualifier,
                    "key_detail": obs.key_detail,
                },
                row_count=1,
            )
            batch_results[idx_key] = (cam, art, note, obs, dist_m, tel)
            return ProcessorResult(
                output={"camera_id": cam.id, "verdict": obs.verdict, "match_score": obs.match_score},
                notes=[note],
                prompt_tokens=tel.prompt_tokens,
                completion_tokens=tel.completion_tokens,
                model_calls=tel.model_calls,
            )

        graph = TaskGraph(
            [
                TaskSpec(
                    task_id=f"r{round_num}_cam_{i}",
                    task_type="custom",
                    processor_name="inspect_camera",
                    context={"idx": str(i), "camera_id": cam.id},
                    dedup_key=f"r{round_num}:{cam.id}",
                )
                for i, (cam, _) in enumerate(batch)
            ]
        )
        runner = ARESRunner(
            note_store=note_store,
            processors={"inspect_camera": _processor},
        )
        runner.solve(graph=graph, max_concurrent=min(10, max(1, len(batch))))

        for i in range(len(batch)):
            key = str(i)
            if key not in batch_results:
                continue
            cam, art, note, obs, dist_m, tel = batch_results[key]
            total_telemetry = total_telemetry.plus(tel)
            inspected_lock_items.append((round_num, i + 1, cam, art, note, obs, dist_m))

    # --- Pre-fetch Round 1 JPEGs once so ARES & Gemini Solo start at the EXACT same instant ---
    if use_geo and center_lat is not None and center_lon is not None:
        # In pin/radius mode, Round 1 checks the closest cameras right around the user's pin
        r1_cams = sorted(
            cameras,
            key=lambda c: haversine_m(center_lat, center_lon, c.latitude, c.longitude),
        )[:round1_k]
    else:
        r1_cams = select_spread_cameras(cameras, k=round1_k)
    _emit("prefetch", f"Pre-fetching {len(r1_cams)} live street JPEGs in parallel...")
    prefetched_r1: dict[str, ArtifactRef] = {}

    def _prefetch_one(cam: CameraInfo) -> tuple[str, ArtifactRef | None]:
        try:
            if cam.id in snapshot_frames:
                src_path = snapshot_frames[cam.id]
                copied = img_dir / src_path.name
                copied.write_bytes(src_path.read_bytes())
                return cam.id, _artifact_from_local_jpeg(copied)
            return cam.id, fetch_camera_jpeg(cam, img_dir)
        except Exception:
            return cam.id, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, max(1, len(r1_cams)))) as pf_pool:
        for cid, art in pf_pool.map(_prefetch_one, r1_cams):
            if art is not None:
                prefetched_r1[cid] = art

    r1_ready_cams = [c for c in r1_cams if c.id in prefetched_r1]
    r1_photo_pairs = [(c, prefetched_r1[c.id]) for c in r1_ready_cams]

    # Start timer AFTER shared photo fetch so both agents race from t=0.0s on the exact same photos!
    t_race_start = time.perf_counter()

    # Launch Gemini Solo AT THE EXACT SAME TIME as ARES Round 1!
    solo_executor: concurrent.futures.ThreadPoolExecutor | None = None
    solo_future: concurrent.futures.Future[GeminiSoloBaseline] | None = None
    if include_solo_baseline and r1_photo_pairs:
        _emit(
            "solo_start",
            f"POST gemini-4-carbon (Gemini Solo, t=0): evaluating the same {len(r1_photo_pairs)} pre-fetched JPEGs...",
        )
        solo_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _run_solo() -> GeminiSoloBaseline:
            res_solo = run_gemini_solo_baseline(
                query=query,
                area=area,
                round1_items=r1_photo_pairs,
            )
            _emit(
                "solo_done",
                f"Gemini Solo finished in {res_solo.wall_clock_s}s → {res_solo.predicted_intersection} ({res_solo.confidence:.0%})",
            )
            return res_solo

        solo_future = solo_executor.submit(_run_solo)

    # --- Round 1: Coarse Parallel Sweep ---
    _emit(
        "round1_start",
        f"ARES Round 1 (t=0): dispatching {len(r1_ready_cams)} parallel gemini-4-hydrogen verifiers...",
    )
    _run_camera_batch(1, [(c, None) for c in r1_ready_cams], prefetched_arts=prefetched_r1)

    # Pick top Round-1 anchor camera by match_score
    r1_sorted = sorted(
        inspected_lock_items,
        key=lambda row: (row[5].verdict != "unclear_camera", row[5].match_score),
        reverse=True,
    )
    anchor_cam = r1_sorted[0][2] if r1_sorted else None

    # --- Round 2: Spatial Zoom-In (nearest online neighbors to anchor) ---
    if anchor_cam is not None and round2_k > 0:
        seen_ids = {row[2].id for row in inspected_lock_items}
        neighbors = select_nearest_cameras(anchor_cam, cameras, k=round2_k, exclude_ids=seen_ids)
        if neighbors:
            _emit(
                "round2_start",
                f"ARES Round 2: GPS zoom-in around {anchor_cam.name} → dispatching {len(neighbors)} nearest street cameras in parallel...",
            )
            _run_camera_batch(2, [(c, d) for c, d in neighbors])

    # Build InvestigationBoard (keep top 8 candidates on the active board so Decision-1 stays fast even for 18-36 cameras)
    all_inspected_objs: list[tuple[InspectedCamera, Note]] = []
    for round_num, idx, cam, art, note, obs, dist_m in inspected_lock_items:
        fname = _safe_field_name(round_num, idx, cam.name)
        item_obj = InspectedCamera(
            round_num=round_num,
            camera=cam,
            artifact=art,
            note_id=note.note_id,
            observation=obs,
            field_name=fname,
            distance_m_from_anchor=dist_m,
        )
        inspected.append(item_obj)
        all_inspected_objs.append((item_obj, note))

    top_scored = sorted(
        inspected,
        key=lambda x: (x.observation.verdict != "unclear_camera", x.observation.match_score),
        reverse=True,
    )
    board_top_names = {x.field_name for x in top_scored[:8]}
    updates: dict[str, dict[str, Any]] = {}
    for item_obj, note in all_inspected_objs:
        if item_obj.field_name not in board_top_names:
            continue
        obs = item_obj.observation
        updates[item_obj.field_name] = {
            "value": {
                "camera_name": item_obj.camera.name,
                "verdict": obs.verdict,
                "match_score": obs.match_score,
                "curb_visibility": obs.curb_visibility,
                "is_legal_parking_street": obs.is_legal_parking_street,
                "open_curb_spot_visible": obs.open_curb_spot_visible,
                "disqualifier": obs.disqualifier,
            },
            "sources": [note.note_id],
            "artifact_ids": [item_obj.artifact.artifact_id],
            "requested_under": item_obj.round_num - 1,
        }

    # Closed-set DecisionQuestion: top 4 highest-scoring cameras + none_verified
    options = ["none_verified"] + list(dict.fromkeys(x.camera.name for x in top_scored[:4]))
    question = DecisionQuestion(
        question_id="best_intersection",
        question=f"Which NYC intersection best satisfies '{query}' based ONLY on verified camera evidence (or 'none_verified' if unclear)?",
        options=options,
        question_type="multiple_choice",
    )

    initial_board = InvestigationBoard(
        board_version=0,
        decision=BoardDecision(
            question_id=question.question_id,
            question=question.question,
            answer="unknown",
        ),
        decisions={
            question.question_id: BoardDecision(
                question_id=question.question_id,
                question=question.question,
                answer="unknown",
            )
        },
        fields={},
    )
    _emit("fusion", f"fusion/patch.py: validating {len(updates)} board fields + SHA-256 JPEG receipts...")
    board, _patch_report = validate_and_merge_patch(
        initial_board,
        {"updates": updates, "rationale": "Grounded live NYC DOT camera observations."},
        note_store,
        strict=True,
        max_active_fields=max(12, len(updates)),
        compute_drift=False,
    )

    # Attach top 3 live JPEGs to gemini-decision-1 counterfactual probe
    top_artifacts = [x.artifact for x in top_scored[:3]]
    _emit(
        "probe",
        f"POST gemini-decision-1: base verdict + {len(board.fields)} parallel counterfactual 'Hide camera' ablations...",
    )
    probe_client = DecisionProbeClient()
    probe_result = probe_client.probe_board(
        board,
        [question],
        primary_question_id=question.question_id,
        run_group_ablations=False,
        max_concurrent=6,
        artifact_refs=top_artifacts,
    )
    total_telemetry = total_telemetry.plus(probe_result.telemetry)

    contact_sheet = render_nyc_contact_sheet(
        inspected,
        probe_result.field_influences,
        dest / "nyc_live_contact_sheet.png",
        query=query,
        decision_answer=probe_result.updated_board.decision.answer,
        decision_confidence=probe_result.updated_board.decision.confidence,
    )

    ares_wall_clock_s = round(time.perf_counter() - t_race_start, 2)
    _emit(
        "ares_done",
        f"ARES finished in {ares_wall_clock_s}s → {probe_result.updated_board.decision.answer} ({probe_result.updated_board.decision.confidence:.1%})",
    )

    solo_baseline: GeminiSoloBaseline | None = None
    if solo_future is not None:
        solo_baseline = solo_future.result()
    if solo_executor is not None:
        solo_executor.shutdown(wait=False)
    _emit(
        "done",
        f"ARES completed in {ares_wall_clock_s}s."
        if solo_baseline is None
        else f"Both completed! ARES={ares_wall_clock_s}s, Gemini Solo={solo_baseline.wall_clock_s}s.",
    )

    return NYCDemoResult(
        query=query,
        area=area,
        street_only=street_only,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        total_online_cameras_in_area=len(cameras),
        cameras_inspected_count=len(inspected),
        anchor_camera=anchor_cam,
        inspected=inspected,
        board=probe_result.updated_board,
        question=question,
        probe_result=probe_result,
        solo_baseline=solo_baseline,
        contact_sheet_path=str(contact_sheet),
        wall_clock_s=ares_wall_clock_s,
        telemetry=total_telemetry,
    )
