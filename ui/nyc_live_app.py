"""Live Interactive NYC Street-Camera ARES Demo Web App (`ui/nyc_live_app.py`).

Run with:
    uv run uvicorn ui.nyc_live_app:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

from collections import OrderedDict
import json
import queue
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from google import genai
from pydantic import BaseModel, Field

from domain.nyc_cameras import DEFAULT_VISION_MODEL, NYCDemoResult, run_live_nyc_search
from models.api_fallback import (
    call_fallback_text_json,
    get_fallback_vision_providers,
    get_gemini_api_keys,
)
from models.decision_probe import DecisionProbeClient, ablate_board_fields
from models.gemini_provider import _strip_json_fences

app = FastAPI(title="ARES Live NYC Street-Camera Search Demo")

RUN_ROOT = Path("benchmark/results/nyc_live_web")
# Start clean on container boot so old run artifacts don't accumulate
if RUN_ROOT.exists():
    shutil.rmtree(RUN_ROOT, ignore_errors=True)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

MAX_CACHED_RUNS = 10
_RUNS_LOCK = threading.Lock()
_RUNS: OrderedDict[str, NYCDemoResult] = OrderedDict()


def _store_run(run_id: str, res: NYCDemoResult) -> None:
    with _RUNS_LOCK:
        _RUNS[run_id] = res
        _RUNS.move_to_end(run_id)
        while len(_RUNS) > MAX_CACHED_RUNS:
            old_id, _ = _RUNS.popitem(last=False)
            shutil.rmtree(RUN_ROOT / old_id, ignore_errors=True)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "vision_model": DEFAULT_VISION_MODEL,
        "gemini_keys_configured": len(get_gemini_api_keys()),
        "fallback_providers": [p[0] for p in get_fallback_vision_providers()],
    }


class LiveSearchRequest(BaseModel):
    area: str = "Manhattan"
    query: str = "open legal street parking spot along a normal city curb"
    keyword: str | None = None
    street_only: bool = True
    round1_k: int = Field(default=12, ge=2, le=24)
    round2_k: int = Field(default=6, ge=0, le=12)
    center_lat: float | None = None
    center_lon: float | None = None
    radius_m: float | None = None


class AblateRequest(BaseModel):
    run_id: str
    hidden_fields: list[str] = Field(default_factory=list)


class AddressResolveRequest(BaseModel):
    address: str = Field(min_length=2, max_length=200)


class ResolvedAddress(BaseModel):
    valid: bool
    normalized_address: str = ""
    borough: str = "Manhattan"
    latitude: float | None = None
    longitude: float | None = None
    reason: str = ""


def _resolve_nyc_address_with_llm(raw_address: str) -> ResolvedAddress:
    """Validate and geocode a user-entered NYC address/intersection via LLM (with multi-key + provider fallback)."""
    cleaned = raw_address.strip()
    if len(cleaned) < 2:
        return ResolvedAddress(valid=False, reason="Please enter an NYC street address or intersection.")

    prompt = (
        f"User input: {cleaned!r}\n\n"
        "Determine if this is a real, recognizable street address, intersection, landmark, or location "
        "within the 5 boroughs of New York City (Manhattan, Brooklyn, Queens, Bronx, Staten Island).\n"
        "- If it is gibberish, too vague (e.g. just 'NYC' or 'parking'), outside New York City, or not a real place, "
        "set valid=false and explain briefly in reason.\n"
        "- If it IS a valid NYC location, set valid=true, normalized_address (clean concise NYC address/intersection), "
        "borough (one of Manhattan, Brooklyn, Queens, Bronx, Staten Island), and accurate latitude & longitude."
    )
    system_instruction = (
        "You are a strict NYC geocoder and address validator. Reject anything outside NYC or nonsensical. "
        "Output strict JSON matching the schema."
    )
    schema = ResolvedAddress.model_json_schema()

    errors: list[str] = []
    parsed: ResolvedAddress | None = None

    for idx, key in enumerate(get_gemini_api_keys()):
        try:
            client = genai.Client(api_key=key)
            interaction = client.interactions.create(
                model=DEFAULT_VISION_MODEL,
                store=False,
                input=prompt,
                system_instruction=system_instruction,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                generation_config={"thinking_level": "low"},
            )
            raw = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
            parsed = ResolvedAddress.model_validate_json(raw)
            break
        except Exception as exc:
            errors.append(f"gemini[{idx}]: {exc}")

    if parsed is None:
        for provider, prov_key, prov_model in get_fallback_vision_providers():
            try:
                raw_text = call_fallback_text_json(
                    provider,
                    prov_key,
                    prov_model,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    schema=schema,
                )
                parsed = ResolvedAddress.model_validate_json(_strip_json_fences(raw_text))
                break
            except Exception as exc:
                errors.append(f"{provider}: {exc}")

    if parsed is None:
        raise HTTPException(
            status_code=503,
            detail="Address validation service is temporarily unavailable: " + " | ".join(errors or ["no API keys"]),
        )

    # Hard geographic bounding-box check for NYC 5 boroughs
    if parsed.valid:
        if (
            parsed.latitude is None
            or parsed.longitude is None
            or not (40.47 <= parsed.latitude <= 40.92 and -74.26 <= parsed.longitude <= -73.70)
        ):
            return ResolvedAddress(
                valid=False,
                reason="Location is outside New York City bounds.",
            )
    return parsed


@app.post("/api/resolve-address")
def api_resolve_address(req: AddressResolveRequest) -> dict[str, Any]:
    resolved = _resolve_nyc_address_with_llm(req.address)
    if not resolved.valid:
        raise HTTPException(
            status_code=400,
            detail=resolved.reason or "Not a recognized NYC address or intersection.",
        )
    return resolved.model_dump()


def _serialize_run(run_id: str, res: NYCDemoResult) -> dict[str, Any]:
    payload = res.model_dump(mode="json")
    payload["run_id"] = run_id
    for item in payload["inspected"]:
        fname = Path(item["artifact"]["path"]).name
        item["image_endpoint"] = f"/api/runs/{run_id}/frames/{fname}"
        item["influence"] = res.probe_result.field_influences.get(item["field_name"], 0.0)
    return payload


@app.post("/api/search")
def api_search(req: LiveSearchRequest) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:10]
    run_dir = RUN_ROOT / run_id
    try:
        res = run_live_nyc_search(
            query=req.query,
            area=req.area,
            keyword=req.keyword or None,
            street_only=req.street_only,
            include_solo_baseline=False,
            round1_k=req.round1_k,
            round2_k=req.round2_k,
            center_lat=req.center_lat,
            center_lon=req.center_lon,
            radius_m=req.radius_m,
            out_dir=run_dir,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _store_run(run_id, res)
    return _serialize_run(run_id, res)


@app.get("/api/search/stream")
def api_search_stream(
    area: str = Query(default="Manhattan"),
    query: str = Query(default="open legal street parking spot along a normal city curb"),
    keyword: str | None = Query(default=None),
    street_only: bool = Query(default=True),
    round1_k: int = Query(default=12, ge=2, le=24),
    round2_k: int = Query(default=6, ge=0, le=12),
    center_lat: float | None = Query(default=None),
    center_lon: float | None = Query(default=None),
    radius_m: float | None = Query(default=None),
) -> StreamingResponse:
    run_id = uuid.uuid4().hex[:10]
    run_dir = RUN_ROOT / run_id
    ev_q: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _on_event(ev: dict[str, Any]) -> None:
        ev_q.put({"type": "log", **ev})

    def _worker() -> None:
        try:
            res = run_live_nyc_search(
                query=query,
                area=area,
                keyword=keyword or None,
                street_only=street_only,
                include_solo_baseline=False,
                round1_k=round1_k,
                round2_k=round2_k,
                center_lat=center_lat,
                center_lon=center_lon,
                radius_m=radius_m,
                out_dir=run_dir,
                on_event=_on_event,
            )
            _store_run(run_id, res)
            ev_q.put({"type": "result", "data": _serialize_run(run_id, res)})
        except Exception as exc:
            ev_q.put({"type": "error", "detail": str(exc)})
        finally:
            ev_q.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    def _gen():
        while True:
            item = ev_q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/frames/{filename}")
def api_frame(run_id: str, filename: str) -> FileResponse:
    safe_name = Path(filename).name
    fpath = RUN_ROOT / run_id / "frames" / safe_name
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(str(fpath), media_type="image/jpeg")


@app.post("/api/ablate")
def api_ablate(req: AblateRequest) -> dict[str, Any]:
    res = _RUNS.get(req.run_id)
    if res is None or res.question is None:
        raise HTTPException(status_code=404, detail="Run not found")

    ablated_board = (
        ablate_board_fields(res.board, req.hidden_fields)
        if req.hidden_fields
        else res.board
    )
    hidden_set = set(req.hidden_fields)
    remaining_arts = [
        x.artifact
        for x in sorted(
            res.inspected,
            key=lambda i: (i.observation.verdict != "unclear_camera", i.observation.match_score),
            reverse=True,
        )
        if x.field_name not in hidden_set
    ][:3]

    probe = DecisionProbeClient()
    ev = probe.evaluate_board(
        ablated_board,
        [res.question],
        artifact_refs=remaining_arts,
    )
    q_ev = ev.answers[res.question.question_id]
    base_conf = res.board.decision.confidence
    base_ans = res.board.decision.answer
    return {
        "hidden_fields": req.hidden_fields,
        "base_answer": base_ans,
        "base_confidence": base_conf,
        "ablated_answer": q_ev.answer,
        "ablated_confidence": q_ev.confidence,
        "confidence_delta": round(q_ev.confidence - base_conf, 4),
        "answer_flipped": q_ev.answer != base_ans,
        "probabilities": q_ev.probabilities,
    }


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ARES Live NYC Street-Camera Search</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {
    --bg: #0f1419; --card: #182028; --border: #2a3642;
    --text: #eef2f6; --muted: #93a1b0;
    --green: #1b9e77; --orange: #d95f02; --purple: #7570b3; --red: #e41a1c;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); padding: 20px; line-height: 1.45;
  }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { margin: 0 0 6px; font-size: 22px; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 16px; }
  .controls {
    display: grid; grid-template-columns: 150px 1fr 170px auto; gap: 10px;
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px;
    align-items: end;
  }
  label { display: block; font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.04em; }
  select, input[type=text] {
    width: 100%; background: #10161d; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 9px 10px; font-size: 13px;
  }
  button.primary {
    background: var(--green); color: white; border: none; border-radius: 8px;
    padding: 10px 18px; font-weight: 700; font-size: 13px; cursor: pointer; white-space: nowrap;
  }
  button.primary:disabled { opacity: 0.55; cursor: wait; }
  .check-row { grid-column: 1 / -1; display: flex; gap: 16px; align-items: center; font-size: 13px; color: var(--muted); margin-top: 2px; }
  .check-row label { display: inline-flex; align-items: center; gap: 6px; margin: 0; text-transform: none; font-size: 13px; color: var(--text); cursor: pointer; }
  .status { margin: 14px 0 8px; font-size: 13px; color: #7dd3fc; min-height: 20px; }
  .trace-box {
    background: #0b0f14; border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; margin-bottom: 16px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px; color: #cbd5e1; max-height: 170px; overflow-y: auto; display: none;
  }
  .trace-line { margin: 2px 0; }
  .trace-time { color: #64748b; margin-right: 8px; }
  .trace-step { color: #5eead4; font-weight: 700; margin-right: 8px; }
  .pipelines {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px 14px; margin-bottom: 14px; display: flex; flex-direction: column; gap: 10px;
  }
  .pipe-row { display: grid; grid-template-columns: 120px 1fr; gap: 10px; align-items: center; }
  .pipe-label { font-size: 11px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }
  .pipe-label.ares { color: #5eead4; }
  .pipe-label.solo { color: #fdba74; }
  .pipe-stages { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .stage {
    flex: 1; min-width: 120px; background: #10161d; border: 1px solid var(--border);
    border-radius: 8px; padding: 7px 9px; font-size: 11.5px; transition: all 0.2s ease;
    opacity: 0.55;
  }
  .stage .st-title { font-weight: 700; color: var(--text); }
  .stage .st-sub { font-size: 10.5px; color: var(--muted); margin-top: 2px; }
  .stage.active { opacity: 1; border-color: #38bdf8; box-shadow: 0 0 0 1px rgba(56,189,248,0.35); }
  .stage.done-ares { opacity: 1; border-color: var(--green); background: rgba(27,158,119,0.12); }
  .stage.done-solo { opacity: 1; border-color: var(--orange); background: rgba(217,95,2,0.12); }
  .arrow { color: #475569; font-weight: 700; font-size: 12px; user-select: none; }
  .compare { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; }
  .panel {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px;
  }
  .panel.ares { border-left: 4px solid var(--green); }
  .panel.solo { border-left: 4px solid var(--orange); }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700;
    margin-bottom: 6px;
  }
  .badge.green { background: rgba(27,158,119,0.2); color: #5eead4; }
  .badge.orange { background: rgba(217,95,2,0.22); color: #fdba74; }
  .badge.gray { background: rgba(148,163,184,0.2); color: #cbd5e1; }
  .verdict { font-size: 20px; font-weight: 800; margin: 4px 0; }
  .meta { font-size: 12px; color: var(--muted); }
  .ablate-banner {
    background: rgba(117,112,179,0.16); border: 1px solid var(--purple); border-radius: 10px;
    padding: 12px 14px; margin-bottom: 16px; font-size: 13px; display: none;
  }
  .cost-bar-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px 16px; margin-bottom: 16px; display: none;
  }
  .cost-bar-head {
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
    gap: 8px; margin-bottom: 10px;
  }
  .cost-bar-title { font-size: 11px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
  .cost-bar-ratio { font-size: 11.5px; color: #7dd3fc; font-weight: 600; }
  .cost-row {
    display: grid; grid-template-columns: 110px 1fr 220px; gap: 10px; align-items: center;
    margin-bottom: 8px;
  }
  .cost-row:last-child { margin-bottom: 0; }
  .cost-label { font-size: 11.5px; font-weight: 700; }
  .cost-label.ares { color: #5eead4; }
  .cost-label.solo { color: #fdba74; }
  .cost-track {
    height: 14px; background: #10161d; border: 1px solid var(--border);
    border-radius: 999px; overflow: hidden; position: relative;
  }
  .cost-fill {
    height: 100%; border-radius: 999px; width: 0%;
    transition: width 0.25s ease;
  }
  .cost-fill.ares { background: linear-gradient(90deg, #15803d, #1b9e77, #2dd4bf); }
  .cost-fill.solo { background: linear-gradient(90deg, #b45309, #d95f02, #fb923c); }
  .cost-stats { font-size: 11.5px; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .cost-stats b { color: var(--text); }
  .results-cols {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start;
  }
  .result-col {
    background: rgba(24,32,40,0.45); border: 1px solid var(--border); border-radius: 12px; padding: 12px;
  }
  .result-col.ares-col { border-top: 3px solid var(--green); }
  .result-col.solo-col { border-top: 3px solid var(--orange); }
  .col-header {
    display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;
    font-size: 11.5px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase;
  }
  .col-header.ares { color: #5eead4; }
  .col-header.solo { color: #fdba74; }
  .col-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(235px, 1fr)); gap: 12px; }
  .cam-card {
    background: var(--card); border: 2px solid var(--border); border-radius: 12px; overflow: hidden;
    display: flex; flex-direction: column;
  }
  .cam-card.strong_match { border-color: var(--green); }
  .cam-card.partial_match { border-color: var(--orange); }
  .cam-card.solo-pick { border-color: var(--orange); }
  .cam-card.hidden-clue { opacity: 0.48; border-style: dashed; border-color: var(--purple); }
  .img-wrap { position: relative; width: 100%; height: 205px; background: #000; overflow: hidden; }
  .img-wrap img { width: 100%; height: 100%; object-fit: fill; display: block; }
  .spot-svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .mode-switch {
    display: inline-flex; background: #10161d; border: 1px solid var(--border);
    border-radius: 10px; padding: 3px; gap: 3px;
  }
  .mode-btn {
    border: none; background: transparent; color: var(--muted); font-size: 12px;
    font-weight: 700; padding: 6px 12px; border-radius: 7px; cursor: pointer;
  }
  .mode-btn.active-live { background: rgba(27,158,119,0.25); color: #5eead4; }
  .mode-btn.active-off { background: rgba(117,112,179,0.28); color: #ddd6fe; }
  .header-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 10px; }
  .map-panel {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 10px 12px; margin-bottom: 12px;
  }
  .map-bar { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; font-size: 12px; color: var(--muted); }
  .map-btns { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .chip-btn {
    background: #10161d; color: var(--text); border: 1px solid var(--border); border-radius: 7px;
    padding: 5px 10px; font-size: 11.5px; font-weight: 600; cursor: pointer;
  }
  .chip-btn:hover { border-color: var(--green); }
  .chip-btn.geo { border-color: #38bdf8; color: #7dd3fc; }
  #nycMap { width: 100%; height: 245px; border-radius: 9px; border: 1px solid var(--border); background: #0b0f14; }
  #nycMap .leaflet-tile-pane { filter: invert(92%) hue-rotate(180deg) brightness(92%) contrast(92%); }
  .cam-body { padding: 10px 12px; flex: 1; display: flex; flex-direction: column; gap: 6px; }
  .cam-title { font-weight: 700; font-size: 13px; }
  .cam-desc { font-size: 12px; color: var(--muted); flex: 1; }
  .cam-foot { display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-top: 4px; }
  button.hide-btn {
    background: transparent; color: #c4b5fd; border: 1px solid var(--purple); border-radius: 6px;
    padding: 5px 9px; font-size: 11px; font-weight: 600; cursor: pointer;
  }
  button.hide-btn.active { background: var(--purple); color: white; }
  @media (max-width: 900px) {
    .controls, .compare, .results-cols { grid-template-columns: 1fr; }
    .cost-row { grid-template-columns: 95px 1fr; }
    .cost-stats { grid-column: 1 / -1; text-align: left; }
  }
</style>
</head>
<body>
<div class="wrap">
  <div class="header-row">
    <div>
      <h1>ARES Live NYC Street-Camera Search</h1>
      <div class="sub" style="margin-bottom:0;">2-round parallel street search + deterministic parking rules + <code>gemini-decision-1</code> counterfactuals</div>
    </div>
    <div class="mode-switch">
      <span class="mode-btn active-live" style="cursor:default;">● LIVE NYC DOT</span>
    </div>
  </div>

  <div class="map-panel">
    <div class="map-bar" style="flex-wrap:wrap;gap:8px;">
      <div><b>Pick location:</b> enter an NYC address or click the map (<span id="pinLabel">40.7233, -73.9985</span>)</div>
      <div class="map-btns" style="flex:1;justify-content:flex-end;min-width:280px;">
        <input id="addressInput" type="text" placeholder="NYC address or intersection (e.g. 350 5th Ave, Houston &amp; Bowery)" style="max-width:340px;padding:6px 10px;font-size:12px;" onkeydown="if(event.key==='Enter'){event.preventDefault();resolveAddress();}" />
        <button id="addressBtn" class="chip-btn" onclick="resolveAddress()">Set Pin</button>
        <button class="chip-btn geo" onclick="useMyLocation()">📍 Use My Location</button>
        <select id="radiusSelect" onchange="updateMapCircle()" style="width:auto;padding:5px 8px;font-size:11.5px;">
          <option value="220">~2 blocks (220m)</option>
          <option value="450" selected>~4 blocks (450m)</option>
          <option value="900">~8 blocks (900m)</option>
          <option value="0">Whole Borough</option>
        </select>
      </div>
    </div>
    <div id="addressMsg" style="display:none;font-size:12px;margin:0 0 8px;padding:6px 10px;border-radius:8px;background:#10161d;border:1px solid var(--border);"></div>
    <div id="nycMap"></div>
  </div>

  <div class="controls" style="grid-template-columns: 140px 1fr 140px 115px auto;">
    <div>
      <label>Borough</label>
      <select id="area">
        <option value="Manhattan" selected>Manhattan</option>
        <option value="Brooklyn">Brooklyn</option>
        <option value="Queens">Queens</option>
        <option value="Bronx">Bronx</option>
        <option value="Staten Island">Staten Island</option>
      </select>
    </div>
    <div>
      <label>Search Query</label>
      <input id="query" type="text" value="open legal street parking spot along a normal city curb" />
    </div>
    <div>
      <label>Street Filter</label>
      <input id="keyword" type="text" placeholder="e.g. Broadway" />
    </div>
    <div>
      <label>Cameras</label>
      <select id="camBudget">
        <option value="6,3">9 (6+3)</option>
        <option value="12,6" selected>18 (12+6)</option>
        <option value="16,8">24 (16+8)</option>
        <option value="24,12">36 (24+12)</option>
      </select>
    </div>
    <div>
      <button id="runBtn" class="primary" onclick="runSearch()">Launch Search</button>
    </div>
    <div class="check-row">
      <label><input id="streetOnly" type="checkbox" checked /> Streets only</label>
      <label><input id="showRejected" type="checkbox" onchange="rerenderGrid()" /> Show rejected cameras</label>
    </div>
  </div>

  <div id="status" class="status">Enter an NYC address or click the map to drop a pin, then click <b>Launch Search</b>.</div>

  <div class="pipelines">
    <div class="pipe-row">
      <div class="pipe-label ares">ARES Search</div>
      <div class="pipe-stages">
        <div class="stage" id="st_catalog"><div class="st-title">1. Filter</div><div class="st-sub" id="st_catalog_sub">waiting...</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_r1"><div class="st-title" id="st_r1_title">2. Sweep (×12)</div><div class="st-sub" id="st_r1_sub">parallel</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_r2"><div class="st-title" id="st_r2_title">3. GPS Zoom (×6)</div><div class="st-sub" id="st_r2_sub">nearest streets</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_fusion"><div class="st-title">4. Rules</div><div class="st-sub" id="st_fusion_sub">patch.py</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_probe"><div class="st-title">5. Decision-1</div><div class="st-sub" id="st_probe_sub">calibrated</div></div>
      </div>
    </div>
    <div class="pipe-row" id="soloPipeRow" style="display:none;">
      <div class="pipe-label solo">Gemini Solo</div>
      <div class="pipe-stages">
        <div class="stage" id="st_solo_in"><div class="st-title" id="st_solo_title">1. Same 12 R1 Photos</div><div class="st-sub" id="st_solo_in_sub">no Round-2 zoom</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_solo_out"><div class="st-title">2. 1-Shot Guess</div><div class="st-sub" id="st_solo_out_sub">no rules / no probe</div></div>
      </div>
    </div>
  </div>

  <details id="traceDetails" open style="margin-bottom:14px;">
    <summary style="cursor:pointer;font-size:12px;color:#93a1b0;margin-bottom:6px;">Live API Request Trace</summary>
    <div id="traceBox" class="trace-box" style="margin-bottom:0;"></div>
  </details>

  <div id="costBarCard" class="cost-bar-card">
    <div class="cost-bar-head">
      <div class="cost-bar-title">Est. API Cost &amp; Time to Find</div>
      <div id="costBarRatio" class="cost-bar-ratio"></div>
    </div>
    <div class="cost-row">
      <div class="cost-label ares">ARES Verified</div>
      <div class="cost-track"><div id="aresCostFill" class="cost-fill ares"></div></div>
      <div id="aresCostStats" class="cost-stats"><b>0.0s</b> · running...</div>
    </div>
    <div class="cost-row" id="soloCostRow" style="display:none;">
      <div class="cost-label solo">Gemini Solo</div>
      <div class="cost-track"><div id="soloCostFill" class="cost-fill solo"></div></div>
      <div id="soloCostStats" class="cost-stats"><b>0.0s</b> · running...</div>
    </div>
  </div>

  <div id="compare" class="compare" style="display:none;grid-template-columns:1fr;">
    <div class="panel ares">
      <span class="badge green">ARES (2-ROUND SEARCH + VERIFIED)</span>
      <div id="aresVerdict" class="verdict"></div>
      <div id="aresMeta" class="meta"></div>
    </div>
    <div class="panel solo" id="soloComparePanel" style="display:none;">
      <span class="badge orange" id="soloBadge">GEMINI SOLO (12 R1 PHOTOS · NO ZOOM)</span>
      <div id="soloVerdict" class="verdict"></div>
      <div id="soloMeta" class="meta"></div>
    </div>
  </div>

  <div id="ablateBanner" class="ablate-banner"></div>

  <div id="resultsCols" class="results-cols" style="display:none;grid-template-columns:1fr;">
    <div class="result-col ares-col">
      <div class="col-header ares">
        <span>ARES Results</span>
        <span id="aresColCount" class="badge green" style="margin:0;">0 verified</span>
      </div>
      <div id="aresGrid" class="col-grid"></div>
    </div>
    <div class="result-col solo-col" id="soloResultCol" style="display:none;">
      <div class="col-header solo">
        <span>Gemini Solo Result</span>
        <span id="soloColCount" class="badge orange" style="margin:0;">1-shot pick</span>
      </div>
      <div id="soloGrid" class="col-grid"></div>
    </div>
  </div>
</div>

<script>
let currentRunId = null;
let currentData = null;
let hiddenFields = new Set();
let pinLat = 40.7233;
let pinLon = -73.9985;
let pinName = "SoHo";
let nycMap = null;
let pinMarker = null;
let radiusCircle = null;
let camLayer = null;
let curR1Target = 12;
let curR2Target = 6;

function initMap() {
  if (typeof L === "undefined") return;
  nycMap = L.map("nycMap", { zoomControl: true }).setView([pinLat, pinLon], 15);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(nycMap);
  camLayer = L.layerGroup().addTo(nycMap);
  updatePinOnMap(false);
  nycMap.on("click", (e) => {
    pinLat = Number(e.latlng.lat.toFixed(5));
    pinLon = Number(e.latlng.lng.toFixed(5));
    pinName = "Dropped Pin";
    updatePinOnMap(false);
  });
}

function updatePinOnMap(recenter) {
  document.getElementById("pinLabel").textContent = `${pinName} · ${pinLat.toFixed(4)}, ${pinLon.toFixed(4)}`;
  if (!nycMap) return;
  if (pinMarker) nycMap.removeLayer(pinMarker);
  pinMarker = L.circleMarker([pinLat, pinLon], {
    radius: 8, color: "#38bdf8", fillColor: "#38bdf8", fillOpacity: 0.9, weight: 2
  }).addTo(nycMap).bindPopup(`<b>${pinName}</b><br/>${pinLat.toFixed(4)}, ${pinLon.toFixed(4)}`);
  updateMapCircle();
  if (recenter) nycMap.setView([pinLat, pinLon], 15);
}

function updateMapCircle() {
  if (!nycMap) return;
  if (radiusCircle) { nycMap.removeLayer(radiusCircle); radiusCircle = null; }
  const r = Number(document.getElementById("radiusSelect").value || 0);
  if (r > 0) {
    radiusCircle = L.circle([pinLat, pinLon], {
      radius: r, color: "#1b9e77", fillColor: "#1b9e77", fillOpacity: 0.12, weight: 1.5, dashArray: "5,5"
    }).addTo(nycMap);
  }
}

async function resolveAddress() {
  const input = document.getElementById("addressInput");
  const btn = document.getElementById("addressBtn");
  const msg = document.getElementById("addressMsg");
  const raw = (input.value || "").trim();
  if (!raw) {
    msg.style.display = "block";
    msg.style.color = "#fca5a5";
    msg.textContent = "Please enter an NYC street address or intersection.";
    return;
  }
  btn.disabled = true;
  btn.textContent = "Checking...";
  msg.style.display = "block";
  msg.style.color = "#93a1b0";
  msg.textContent = "Validating NYC address...";
  try {
    const resp = await fetch("/api/resolve-address", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ address: raw })
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      throw new Error(data.detail || data.error || `Rejected (HTTP ${resp.status})`);
    }
    pinLat = Number(data.latitude);
    pinLon = Number(data.longitude);
    pinName = data.normalized_address || raw;
    if (data.borough) {
      const areaEl = document.getElementById("area");
      for (const opt of areaEl.options) {
        if (opt.value.toLowerCase() === String(data.borough).toLowerCase()) {
          areaEl.value = opt.value;
          break;
        }
      }
    }
    if (Number(document.getElementById("radiusSelect").value) === 0) {
      document.getElementById("radiusSelect").value = "450";
    }
    updatePinOnMap(true);
    msg.style.color = "#86efac";
    msg.textContent = `Pin set to ${pinName} (${data.borough || "NYC"}). Click Launch Search when ready.`;
  } catch (err) {
    msg.style.color = "#fca5a5";
    msg.textContent = `Address rejected: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Set Pin";
  }
}

function useMyLocation() {
  if (!navigator.geolocation) {
    alert("Geolocation is not supported by your browser.");
    return;
  }
  document.getElementById("status").innerHTML = "Requesting your browser GPS coordinates...";
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      pinLat = Number(pos.coords.latitude.toFixed(5));
      pinLon = Number(pos.coords.longitude.toFixed(5));
      pinName = "My Location";
      if (Number(document.getElementById("radiusSelect").value) === 0) {
        document.getElementById("radiusSelect").value = "220";
      }
      updatePinOnMap(true);
      document.getElementById("status").innerHTML = "Location pin set. Click <b>Launch Search</b> when ready.";
    },
    (err) => {
      document.getElementById("status").innerHTML = `<span style="color:#fca5a5;">Location blocked (${err.message}). Enter an address or click anywhere on the map to drop a pin instead!</span>`;
    },
    { enableHighAccuracy: true, timeout: 8000 }
  );
}

function plotCamerasOnMap(inspected) {
  if (!nycMap || !camLayer) return;
  camLayer.clearLayers();
  for (const item of inspected) {
    const v = item.observation.verdict;
    const color = v === "strong_match" ? "#22c55e" : (v === "partial_match" ? "#f97316" : "#64748b");
    const m = L.circleMarker([item.camera.latitude, item.camera.longitude], {
      radius: v === "strong_match" ? 8 : 6,
      color: "#ffffff", weight: 1.5, fillColor: color, fillOpacity: 0.92
    }).addTo(camLayer);
    m.bindPopup(`<b>${item.camera.name}</b><br/>R${item.round_num} · ${v}<br/>${item.observation.summary}`);
  }
}

function formatLocalTime(isoStr) {
  try {
    return new Date(isoStr).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
  } catch {
    return isoStr || "";
  }
}

let activeStream = null;
let r1Count = 0;
let r2Count = 0;
let costTimerId = null;
let searchStartPerf = 0;
let soloDoneTimeS = null;
let aresDoneTimeS = null;

// Estimated tier rates ($ per 1M tokens): hydrogen/decision-1 (lightweight) vs carbon (frontier)
function estimateCostUsd(tel, tier) {
  if (!tel) return 0;
  const p = Number(tel.prompt_tokens || 0);
  const c = Number(tel.completion_tokens || 0);
  const inRate = tier === "carbon" ? 2.00 : 0.10;
  const outRate = tier === "carbon" ? 12.00 : 0.40;
  return (p * inRate + c * outRate) / 1_000_000;
}

function formatUsd(usd) {
  if (usd <= 0) return "$0.000";
  if (usd < 0.01) return `~$${usd.toFixed(4)}`;
  return `~$${usd.toFixed(3)}`;
}

function updateCostBars(aresS, soloS, aresTel, soloTel, isFinal) {
  const card = document.getElementById("costBarCard");
  card.style.display = "block";
  const aVal = Math.max(0, Number(aresS || 0));
  const sVal = Math.max(0, Number(soloS || 0));
  const ratioEl = document.getElementById("costBarRatio");

  if (isFinal) {
    const aCost = estimateCostUsd(aresTel, "hydrogen");
    const sCost = estimateCostUsd(soloTel, "carbon");
    const maxCost = Math.max(aCost, sCost, 0.0001);
    document.getElementById("aresCostFill").style.width = `${Math.min(100, (aCost / maxCost) * 100).toFixed(1)}%`;
    document.getElementById("soloCostFill").style.width = `${Math.min(100, (sCost / maxCost) * 100).toFixed(1)}%`;
    document.getElementById("aresCostStats").innerHTML =
      `<b>${formatUsd(aCost)}</b> · ${aVal.toFixed(1)}s`;
    document.getElementById("soloCostStats").innerHTML =
      `<b>${formatUsd(sCost)}</b> · ${sVal.toFixed(1)}s`;
    if (aCost > 0 && sCost > 0) {
      if (sCost >= aCost) {
        ratioEl.textContent = `ARES cost ${(sCost / aCost).toFixed(1)}× less (hydrogen + fusion + decision-1 vs carbon)`;
      } else {
        ratioEl.textContent = `ARES cost ${(aCost / sCost).toFixed(1)}× Solo (hydrogen + fusion + decision-1 vs carbon)`;
      }
    } else {
      ratioEl.textContent = `ARES ${aVal.toFixed(1)}s vs Solo ${sVal.toFixed(1)}s`;
    }
    return;
  }

  const maxVal = Math.max(aVal, sVal, 1.0);
  document.getElementById("aresCostFill").style.width = `${Math.min(100, (aVal / maxVal) * 100).toFixed(1)}%`;
  document.getElementById("soloCostFill").style.width = `${Math.min(100, (sVal / maxVal) * 100).toFixed(1)}%`;
  document.getElementById("aresCostStats").innerHTML =
    `<b>${aVal.toFixed(1)}s</b>${aresDoneTimeS !== null ? " · done" : " · searching..."}`;
  document.getElementById("soloCostStats").innerHTML =
    `<b>${sVal.toFixed(1)}s</b>${soloDoneTimeS !== null ? " · done" : " · guessing..."}`;
  ratioEl.textContent = "Live elapsed wall-clock time...";
}

function startCostTimer() {
  stopCostTimer();
  searchStartPerf = performance.now();
  soloDoneTimeS = null;
  aresDoneTimeS = null;
  updateCostBars(0, 0, null, null, false);
  costTimerId = setInterval(() => {
    const nowS = (performance.now() - searchStartPerf) / 1000;
    const aS = aresDoneTimeS !== null ? aresDoneTimeS : nowS;
    const sS = soloDoneTimeS !== null ? soloDoneTimeS : nowS;
    updateCostBars(aS, sS, null, null, false);
  }, 150);
}

function stopCostTimer() {
  if (costTimerId) {
    clearInterval(costTimerId);
    costTimerId = null;
  }
}

function setStage(id, state, subText) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.remove("active", "done-ares", "done-solo");
  if (state) el.classList.add(state);
  if (subText !== undefined) {
    const sub = document.getElementById(id + "_sub");
    if (sub) sub.textContent = subText;
  }
}

function resetStages() {
  r1Count = 0;
  r2Count = 0;
  const [r1k, r2k] = document.getElementById("camBudget").value.split(",");
  curR1Target = Number(r1k);
  curR2Target = Number(r2k);
  document.getElementById("st_r1_title").textContent = `2. Sweep (×${curR1Target})`;
  document.getElementById("st_r2_title").textContent = `3. GPS Zoom (×${curR2Target})`;
  document.getElementById("st_solo_title").textContent = `1. Same ${curR1Target} R1 Photos`;
  document.getElementById("soloBadge").textContent = `GEMINI SOLO (${curR1Target} R1 PHOTOS · NO ZOOM)`;
  setStage("st_catalog", "", "waiting...");
  setStage("st_r1", "", "parallel");
  setStage("st_r2", "", "nearest streets");
  setStage("st_fusion", "", "patch.py");
  setStage("st_probe", "", "calibrated");
  setStage("st_solo_in", "", `${curR1Target} R1 JPEGs, no zoom`);
  setStage("st_solo_out", "", "no rules / no probe");
}

function updateStagesFromEvent(step, message, elapsed) {
  const t = `+${Number(elapsed || 0).toFixed(1)}s`;
  if (step === "catalog") {
    setStage("st_catalog", "active", "fetching NYC DOT API...");
  } else if (step === "catalog_done") {
    setStage("st_catalog", "done-ares", `${message.slice(0, 34)} (${t})`);
  } else if (step === "round1_start") {
    setStage("st_r1", "active", `0/${curR1Target} checked...`);
  } else if (step === "camera_done") {
    if (message.startsWith("[R1]")) {
      r1Count += 1;
      setStage("st_r1", "active", `${r1Count}/${curR1Target} checked (${t})`);
    } else if (message.startsWith("[R2]")) {
      r2Count += 1;
      setStage("st_r2", "active", `${r2Count}/${curR2Target} checked (${t})`);
    }
  } else if (step === "round2_start") {
    setStage("st_r1", "done-ares", `${r1Count}/${curR1Target} done (${t})`);
    setStage("st_r2", "active", `0/${curR2Target} neighbors...`);
  } else if (step === "fusion") {
    setStage("st_r1", "done-ares", `${r1Count}/${curR1Target} done`);
    setStage("st_r2", "done-ares", `${r2Count}/${curR2Target} done`);
    setStage("st_fusion", "active", "verifying SHA-256 + rules...");
  } else if (step === "probe") {
    setStage("st_fusion", "done-ares", `0 invariant violations (${t})`);
    setStage("st_probe", "active", "running ablations...");
  } else if (step === "ares_done") {
    aresDoneTimeS = Number(elapsed || 0);
    setStage("st_probe", "done-ares", `calibrated verdict (${t})`);
  } else if (step === "solo_start") {
    setStage("st_solo_in", "active", `${curR1Target} R1 JPEGs in 1 call...`);
    setStage("st_solo_out", "active", "racing ARES...");
  } else if (step === "solo_done") {
    soloDoneTimeS = Number(elapsed || 0);
    setStage("st_solo_in", "done-solo", `${curR1Target} R1 JPEGs only`);
    setStage("st_solo_out", "done-solo", `1-shot guess (${t})`);
  } else if (step === "done") {
    setStage("st_solo_in", "done-solo", `${curR1Target} R1 JPEGs only`);
  }
}

function appendTrace(step, message, elapsed) {
  const box = document.getElementById("traceBox");
  box.style.display = "block";
  const line = document.createElement("div");
  line.className = "trace-line";
  line.innerHTML = `<span class="trace-time">[+${Number(elapsed || 0).toFixed(2)}s]</span><span class="trace-step">${step}</span>${message}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function runSearch() {
  if (activeStream) {
    activeStream.close();
    activeStream = null;
  }
  const btn = document.getElementById("runBtn");
  const status = document.getElementById("status");
  const box = document.getElementById("traceBox");
  btn.disabled = true;
  hiddenFields = new Set();
  resetStages();
  startCostTimer();
  document.getElementById("ablateBanner").style.display = "none";
  box.innerHTML = "";
  box.style.display = "block";
  status.innerHTML = "Streaming live requests...";

  const [r1k, r2k] = document.getElementById("camBudget").value.split(",");
  const rad = Number(document.getElementById("radiusSelect").value || 0);
  const params = new URLSearchParams({
    area: document.getElementById("area").value,
    query: document.getElementById("query").value,
    street_only: document.getElementById("streetOnly").checked ? "true" : "false",
    round1_k: r1k,
    round2_k: r2k,
  });
  if (rad > 0) {
    params.set("center_lat", String(pinLat));
    params.set("center_lon", String(pinLon));
    params.set("radius_m", String(rad));
  }
  const kw = document.getElementById("keyword").value.trim();
  if (kw) params.set("keyword", kw);

  const es = new EventSource(`/api/search/stream?${params.toString()}`);
  activeStream = es;

  es.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === "log") {
      appendTrace(msg.step, msg.message, msg.elapsed_s);
      updateStagesFromEvent(msg.step, msg.message, msg.elapsed_s);
      status.innerHTML = `<b>[+${Number(msg.elapsed_s || 0).toFixed(1)}s]</b> ${msg.message}`;
    } else if (msg.type === "result") {
      const data = msg.data;
      currentRunId = data.run_id;
      currentData = data;
      renderResult(data);
      const liveTime = formatLocalTime(data.timestamp_utc);
      const matches = data.inspected.filter(i => i.observation.verdict === "strong_match" || i.observation.verdict === "partial_match").length;
      const rejected = data.inspected.length - matches;
      setStage("st_fusion", "done-ares", `${rejected} pruned · 0 violations`);
      setStage("st_probe", "done-ares", `${(data.board.decision.confidence * 100).toFixed(1)}% (${data.wall_clock_s}s)`);
      if (data.solo_baseline) {
        setStage("st_solo_out", "done-solo", `${(data.solo_baseline.confidence * 100).toFixed(0)}% (${data.solo_baseline.wall_clock_s}s)`);
      }
      document.getElementById("traceDetails").open = false;
      status.innerHTML =
        `Live @ <b>${liveTime}</b> — <b>${matches}</b> verified spot(s) found (${rejected} non-matches/non-streets pruned from ${data.total_online_cameras_in_area} street cameras).`;
      stopCostTimer();
      es.close();
      activeStream = null;
      btn.disabled = false;
    } else if (msg.type === "error") {
      status.innerHTML = `<span style="color:#fca5a5;">Error: ${msg.detail}</span>`;
      stopCostTimer();
      es.close();
      activeStream = null;
      btn.disabled = false;
    }
  };

  es.onerror = () => {
    status.innerHTML = `<span style="color:#fca5a5;">Rate limit reached or connection interrupted. Please wait ~20 seconds before retrying.</span>`;
    stopCostTimer();
    es.close();
    activeStream = null;
    btn.disabled = false;
  };
}

function renderResult(data) {
  document.getElementById("compare").style.display = "grid";
  document.getElementById("resultsCols").style.display = "grid";
  const dec = data.board.decision;
  const liveTime = formatLocalTime(data.timestamp_utc);
  document.getElementById("aresVerdict").textContent = dec.answer;
  document.getElementById("aresMeta").innerHTML =
    `<b>${(dec.confidence * 100).toFixed(1)}%</b> calibrated confidence &nbsp;·&nbsp; <b>${data.wall_clock_s}s</b> &nbsp;·&nbsp; live @ ${liveTime}`;

  const soloS = data.solo_baseline ? data.solo_baseline.wall_clock_s : 0;
  updateCostBars(
    data.wall_clock_s,
    soloS,
    data.telemetry,
    data.solo_baseline ? data.solo_baseline.telemetry : null,
    true
  );

  if (data.solo_baseline) {
    const s = data.solo_baseline;
    document.getElementById("soloVerdict").textContent = s.predicted_intersection;
    document.getElementById("soloMeta").innerHTML =
      `<b>${(s.confidence * 100).toFixed(0)}%</b> self-reported &nbsp;·&nbsp; <b>${s.wall_clock_s}s</b> &nbsp;·&nbsp; ${curR1Target} R1 photos only<br/>${s.reasoning}`;
  }

  plotCamerasOnMap(data.inspected || []);
  rerenderGrid();
}

function normalizeCamName(name) {
  return String(name || "")
    .replace(/\\((manhattan|brooklyn|queens|bronx|staten island|nyc)\\)\\s*$/i, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function findSoloMatchedCamera(data) {
  if (!data || !data.solo_baseline || !data.solo_baseline.predicted_intersection) return null;
  const pred = normalizeCamName(data.solo_baseline.predicted_intersection);
  if (!pred || pred === "none verified" || pred === "none") return null;
  const inspected = data.inspected || [];
  let exact = inspected.find(item => normalizeCamName(item.camera.name) === pred);
  if (exact) return exact;
  let partial = inspected.find(item => {
    const cn = normalizeCamName(item.camera.name);
    return cn && (pred.includes(cn) || cn.includes(pred));
  });
  return partial || null;
}

function buildCircleSvg(box, strokeColor, fillColor) {
  if (Array.isArray(box) && box.length === 4) {
    const [ymin, xmin, ymax, xmax] = box;
    const cx = (xmin + xmax) / 2;
    const cy = (ymin + ymax) / 2;
    const rx = Math.max(45, (xmax - xmin) / 2);
    const ry = Math.max(45, (ymax - ymin) / 2);
    return `
      <svg class="spot-svg" viewBox="0 0 1000 1000" preserveAspectRatio="none">
        <ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"
          fill="${fillColor}" stroke="${strokeColor}" stroke-width="14" />
      </svg>
    `;
  }
  return "";
}

function buildSpotOverlay(item) {
  const v = item.observation.verdict;
  if (v === "strong_match" || v === "partial_match") {
    return buildCircleSvg(item.observation.spot_box, "#22c55e", "rgba(27,158,119,0.20)");
  }
  return "";
}

function rerenderGrid() {
  if (!currentData) return;
  const showRejected = document.getElementById("showRejected").checked;
  const liveTime = formatLocalTime(currentData.timestamp_utc);
  const aresGrid = document.getElementById("aresGrid");
  const soloGrid = document.getElementById("soloGrid");
  aresGrid.innerHTML = "";
  soloGrid.innerHTML = "";

  const matchedCount = currentData.inspected.filter(
    i => i.observation.verdict === "strong_match" || i.observation.verdict === "partial_match"
  ).length;
  document.getElementById("aresColCount").textContent = `${matchedCount} verified`;

  const visibleItems = currentData.inspected
    .filter(item => {
      const isMatch = item.observation.verdict === "strong_match" || item.observation.verdict === "partial_match";
      return showRejected || isMatch;
    })
    .sort((a, b) => {
      const rank = v => (v === "strong_match" ? 2 : (v === "partial_match" ? 1 : 0));
      const rd = rank(b.observation.verdict) - rank(a.observation.verdict);
      return rd !== 0 ? rd : (b.influence - a.influence);
    });

  if (visibleItems.length === 0) {
    aresGrid.innerHTML = `<div style="grid-column:1/-1;color:#93a1b0;padding:16px;background:#182028;border-radius:10px;font-size:12.5px;">No verified open parking spots in these ${currentData.inspected.length} cameras right now (ARES honestly abstains instead of guessing). Check <b>Show rejected cameras</b> above to inspect them.</div>`;
  } else {
    for (const item of visibleItems) {
      const v = item.observation.verdict;
      const badgeCls = v === "strong_match" ? "green" : (v === "partial_match" ? "orange" : "gray");
      const roundTag = item.round_num === 2 ? "R2 Zoom" : "R1";
      const card = document.createElement("div");
      card.className = `cam-card ${v}` + (hiddenFields.has(item.field_name) ? " hidden-clue" : "");
      card.id = `card_${item.field_name}`;
      const disq = item.observation.disqualifier && item.observation.disqualifier !== "none"
        ? `<span class="badge" style="background:rgba(228,26,28,0.22);color:#fca5a5;">✕ ${item.observation.disqualifier}</span>`
        : "";
      const isHidden = hiddenFields.has(item.field_name);
      card.innerHTML = `
        <div class="img-wrap">
          <img src="${item.image_endpoint}" alt="${item.camera.name}" />
          ${buildSpotOverlay(item)}
        </div>
        <div class="cam-body">
          <div>
            <span class="badge ${badgeCls}">${roundTag} · ${v.replace("_", " ")}</span>
            <span class="badge gray">impact ${(item.influence * 100).toFixed(0)}%</span>
            ${disq}
          </div>
          <div class="cam-title">${item.camera.name}</div>
          <div class="cam-desc">${item.observation.summary}</div>
          <div class="cam-foot">
            <span class="meta">${liveTime}</span>
            <button class="hide-btn ${isHidden ? "active" : ""}" id="btn_${item.field_name}" onclick="toggleHide('${item.field_name}')">${isHidden ? "Hidden (Restore)" : "Hide camera"}</button>
          </div>
        </div>
      `;
      aresGrid.appendChild(card);
    }
  }

  // Right column: Gemini Solo's pick (with the actual camera photo it guessed)
  const solo = currentData.solo_baseline;
  const soloItem = findSoloMatchedCamera(currentData);
  if (!solo) {
    soloGrid.innerHTML = `<div style="color:#93a1b0;padding:16px;background:#182028;border-radius:10px;font-size:12.5px;">Gemini Solo baseline was not run.</div>`;
  } else if (soloItem) {
    const sv = soloItem.observation.verdict;
    const agreed = sv === "strong_match" || sv === "partial_match";
    const verifierBadge = agreed
      ? `<span class="badge green">✓ ARES verifier: ${sv.replace("_", " ")}</span>`
      : `<span class="badge" style="background:rgba(228,26,28,0.22);color:#fca5a5;">✕ ARES rejected (${sv.replace("_", " ")})</span>`;
    const disq = soloItem.observation.disqualifier && soloItem.observation.disqualifier !== "none"
      ? `<span class="badge" style="background:rgba(228,26,28,0.22);color:#fca5a5;">✕ ${soloItem.observation.disqualifier}</span>`
      : "";
    document.getElementById("soloColCount").textContent = agreed ? "matched R1 cam" : "rejected by ARES";
    const card = document.createElement("div");
    card.className = "cam-card solo-pick";
    const soloOverlay = buildCircleSvg(solo.spot_box, "#f97316", "rgba(217,95,2,0.20)");
    card.innerHTML = `
      <div class="img-wrap">
        <img src="${soloItem.image_endpoint}" alt="${soloItem.camera.name}" />
        ${soloOverlay}
      </div>
      <div class="cam-body">
        <div>
          <span class="badge orange">1-Shot Pick · ${(solo.confidence * 100).toFixed(0)}%</span>
          ${verifierBadge}
          ${disq}
        </div>
        <div class="cam-title">${soloItem.camera.name}</div>
        <div class="cam-desc"><b>Solo reasoning:</b> ${solo.reasoning}<br/><span style="color:#94a3b8;"><b>ARES check:</b> ${soloItem.observation.summary}</span></div>
        <div class="cam-foot">
          <span class="meta">Found in ${solo.wall_clock_s}s · no Round-2 zoom</span>
        </div>
      </div>
    `;
    soloGrid.appendChild(card);
  } else {
    document.getElementById("soloColCount").textContent = "no camera match";
    soloGrid.innerHTML = `
      <div style="color:#cbd5e1;padding:16px;background:#182028;border:2px dashed var(--orange);border-radius:12px;font-size:12.5px;">
        <div style="margin-bottom:6px;"><span class="badge orange">1-Shot Pick · ${(solo.confidence * 100).toFixed(0)}%</span></div>
        <div style="font-weight:700;font-size:14px;margin-bottom:6px;">${solo.predicted_intersection}</div>
        <div style="color:#93a1b0;">${solo.reasoning}</div>
      </div>
    `;
  }
}

async function toggleHide(fieldName) {
  if (!currentRunId) return;
  const card = document.getElementById(`card_${fieldName}`);
  const btn = document.getElementById(`btn_${fieldName}`);
  if (hiddenFields.has(fieldName)) {
    hiddenFields.delete(fieldName);
    card.classList.remove("hidden-clue");
    btn.classList.remove("active");
    btn.textContent = "Hide camera";
  } else {
    hiddenFields.add(fieldName);
    card.classList.add("hidden-clue");
    btn.classList.add("active");
    btn.textContent = "Hidden (Restore)";
  }

  const banner = document.getElementById("ablateBanner");
  banner.style.display = "block";
  banner.innerHTML = "Re-probing <code>gemini-decision-1</code> with hidden camera(s)...";

  try {
    const resp = await fetch("/api/ablate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ run_id: currentRunId, hidden_fields: Array.from(hiddenFields) })
    });
    const r = await resp.json();
    if (!resp.ok) throw new Error(r.detail || "Ablation failed");
    const deltaPct = (r.confidence_delta * 100).toFixed(1);
    const sign = r.confidence_delta >= 0 ? "+" : "";
    const flipMsg = r.answer_flipped
      ? ` <b style="color:#fca5a5;">VERDICT FLIPPED → ${r.ablated_answer}</b>`
      : ` (verdict stayed <b>${r.ablated_answer}</b>)`;
    banner.innerHTML =
      `<b>Live Counterfactual Probe (${r.hidden_fields.length} camera hidden):</b> ` +
      `Confidence went from <b>${(r.base_confidence * 100).toFixed(1)}%</b> → ` +
      `<b>${(r.ablated_confidence * 100).toFixed(1)}%</b> (${sign}${deltaPct}%)` + flipMsg;
  } catch (err) {
    banner.innerHTML = `<span style="color:#fca5a5;">Ablation error: ${err.message}</span>`;
  }
}

// Initialize interactive NYC map (user launches search manually)
window.addEventListener("DOMContentLoaded", () => {
  initMap();
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)
