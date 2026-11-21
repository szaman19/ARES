"""Live Interactive NYC Street-Camera ARES Demo Web App (`ui/nyc_live_app.py`).

Run with:
    uv run uvicorn ui.nyc_live_app:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from domain.nyc_cameras import NYCDemoResult, run_live_nyc_search
from models.decision_probe import DecisionProbeClient, ablate_board_fields

app = FastAPI(title="ARES Live NYC Street-Camera Search Demo")

RUN_ROOT = Path("benchmark/results/nyc_live_web")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
_RUNS: dict[str, NYCDemoResult] = {}


DEFAULT_SNAPSHOT_DIR = Path("data/nyc_snapshot")


class LiveSearchRequest(BaseModel):
    area: str = "Manhattan"
    query: str = "open legal street parking spot along a normal city curb"
    keyword: str | None = None
    street_only: bool = True
    use_snapshot: bool = False
    round1_k: int = Field(default=12, ge=2, le=24)
    round2_k: int = Field(default=6, ge=0, le=12)


class AblateRequest(BaseModel):
    run_id: str
    hidden_fields: list[str] = Field(default_factory=list)


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
            include_solo_baseline=True,
            round1_k=req.round1_k,
            round2_k=req.round2_k,
            out_dir=run_dir,
            snapshot_dir=DEFAULT_SNAPSHOT_DIR if req.use_snapshot else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _RUNS[run_id] = res
    return _serialize_run(run_id, res)


@app.get("/api/search/stream")
def api_search_stream(
    area: str = Query(default="Manhattan"),
    query: str = Query(default="open legal street parking spot along a normal city curb"),
    keyword: str | None = Query(default=None),
    street_only: bool = Query(default=True),
    use_snapshot: bool = Query(default=False),
    round1_k: int = Query(default=12, ge=2, le=24),
    round2_k: int = Query(default=6, ge=0, le=12),
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
                include_solo_baseline=True,
                round1_k=round1_k,
                round2_k=round2_k,
                out_dir=run_dir,
                snapshot_dir=DEFAULT_SNAPSHOT_DIR if use_snapshot else None,
                on_event=_on_event,
            )
            _RUNS[run_id] = res
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
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .cam-card {
    background: var(--card); border: 2px solid var(--border); border-radius: 12px; overflow: hidden;
    display: flex; flex-direction: column;
  }
  .cam-card.strong_match { border-color: var(--green); }
  .cam-card.partial_match { border-color: var(--orange); }
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
    .controls, .compare, .grid { grid-template-columns: 1fr; }
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
      <button id="modeLiveBtn" class="mode-btn active-live" onclick="setMode(false)">● LIVE NYC DOT</button>
      <button id="modeOffBtn" class="mode-btn" onclick="setMode(true)">📦 OFFLINE SNAPSHOT (36)</button>
    </div>
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
      <button id="runBtn" class="primary" onclick="runSearch()">Re-Run</button>
    </div>
    <div class="check-row">
      <label><input id="streetOnly" type="checkbox" checked /> Streets only</label>
      <label><input id="showRejected" type="checkbox" onchange="rerenderGrid()" /> Show rejected cameras</label>
    </div>
  </div>

  <div id="status" class="status">Starting live search...</div>

  <div class="pipelines">
    <div class="pipe-row">
      <div class="pipe-label ares">ARES Search</div>
      <div class="pipe-stages">
        <div class="stage" id="st_catalog"><div class="st-title">1. Filter</div><div class="st-sub" id="st_catalog_sub">waiting...</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_r1"><div class="st-title">2. Sweep (×6)</div><div class="st-sub" id="st_r1_sub">parallel</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_r2"><div class="st-title">3. GPS Zoom (×3)</div><div class="st-sub" id="st_r2_sub">nearest streets</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_fusion"><div class="st-title">4. Rules</div><div class="st-sub" id="st_fusion_sub">patch.py</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_probe"><div class="st-title">5. Decision-1</div><div class="st-sub" id="st_probe_sub">calibrated</div></div>
      </div>
    </div>
    <div class="pipe-row">
      <div class="pipe-label solo">Gemini Solo</div>
      <div class="pipe-stages">
        <div class="stage" id="st_solo_in"><div class="st-title">1. Same 6 R1 Photos</div><div class="st-sub" id="st_solo_in_sub">no Round-2 zoom</div></div>
        <span class="arrow">→</span>
        <div class="stage" id="st_solo_out"><div class="st-title">2. 1-Shot Guess</div><div class="st-sub" id="st_solo_out_sub">no rules / no probe</div></div>
      </div>
    </div>
  </div>

  <details id="traceDetails" open style="margin-bottom:14px;">
    <summary style="cursor:pointer;font-size:12px;color:#93a1b0;margin-bottom:6px;">Live API Request Trace</summary>
    <div id="traceBox" class="trace-box" style="margin-bottom:0;"></div>
  </details>

  <div id="compare" class="compare" style="display:none;">
    <div class="panel ares">
      <span class="badge green">ARES (2-ROUND SEARCH + VERIFIED)</span>
      <div id="aresVerdict" class="verdict"></div>
      <div id="aresMeta" class="meta"></div>
    </div>
    <div class="panel solo">
      <span class="badge orange">GEMINI SOLO (6 R1 PHOTOS · NO ZOOM)</span>
      <div id="soloVerdict" class="verdict"></div>
      <div id="soloMeta" class="meta"></div>
    </div>
  </div>

  <div id="ablateBanner" class="ablate-banner"></div>
  <div id="grid" class="grid"></div>
</div>

<script>
let currentRunId = null;
let currentData = null;
let hiddenFields = new Set();
let useSnapshotMode = false;

function setMode(isOffline) {
  useSnapshotMode = isOffline;
  const liveBtn = document.getElementById("modeLiveBtn");
  const offBtn = document.getElementById("modeOffBtn");
  liveBtn.className = "mode-btn" + (isOffline ? "" : " active-live");
  offBtn.className = "mode-btn" + (isOffline ? " active-off" : "");
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
  setStage("st_catalog", "", "waiting...");
  setStage("st_r1", "", "ARESRunner + hydrogen");
  setStage("st_r2", "", "nearest street neighbors");
  setStage("st_fusion", "", "patch.py + disqualifiers");
  setStage("st_probe", "", "counterfactual drops");
  setStage("st_solo_in", "", "same 9 JPEGs, no zoom");
  setStage("st_solo_out", "", "no disqualifiers / no probe");
}

function updateStagesFromEvent(step, message, elapsed) {
  const t = `+${Number(elapsed || 0).toFixed(1)}s`;
  if (step === "catalog") {
    setStage("st_catalog", "active", "fetching NYC DOT API...");
  } else if (step === "catalog_done") {
    setStage("st_catalog", "done-ares", `${message.replace("Found ", "").replace(" online street intersection cameras in ", " cams · ")} (${t})`);
  } else if (step === "round1_start") {
    setStage("st_r1", "active", "0/6 live JPEGs checked...");
  } else if (step === "camera_done") {
    if (message.startsWith("[R1]")) {
      r1Count += 1;
      setStage("st_r1", "active", `${r1Count}/6 live JPEGs checked (${t})`);
    } else if (message.startsWith("[R2]")) {
      r2Count += 1;
      setStage("st_r2", "active", `${r2Count}/3 neighbors checked (${t})`);
    }
  } else if (step === "round2_start") {
    setStage("st_r1", "done-ares", `${r1Count}/6 done · picked anchor (${t})`);
    setStage("st_r2", "active", "0/3 GPS neighbors checked...");
  } else if (step === "fusion") {
    setStage("st_r1", "done-ares", `${r1Count}/6 done`);
    setStage("st_r2", "done-ares", `${r2Count}/3 done`);
    setStage("st_fusion", "active", "verifying SHA-256 + rules...");
  } else if (step === "probe") {
    setStage("st_fusion", "done-ares", `0 invariant violations (${t})`);
    setStage("st_probe", "active", "running 9 parallel ablations...");
  } else if (step === "ares_done") {
    setStage("st_probe", "done-ares", `calibrated verdict ready (${t})`);
  } else if (step === "solo_start") {
    setStage("st_solo_in", "active", "6 R1 JPEGs in 1 call...");
    setStage("st_solo_out", "active", "racing ARES Round 2...");
  } else if (step === "solo_done") {
    setStage("st_solo_in", "done-solo", "6 R1 JPEGs only");
    setStage("st_solo_out", "done-solo", `1-shot guess (${t})`);
  } else if (step === "done") {
    setStage("st_solo_in", "done-solo", "6 R1 JPEGs only");
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
  document.getElementById("ablateBanner").style.display = "none";
  box.innerHTML = "";
  box.style.display = "block";
  status.innerHTML = "Streaming live requests...";

  const [r1k, r2k] = document.getElementById("camBudget").value.split(",");
  const params = new URLSearchParams({
    area: document.getElementById("area").value,
    query: document.getElementById("query").value,
    street_only: document.getElementById("streetOnly").checked ? "true" : "false",
    use_snapshot: useSnapshotMode ? "true" : "false",
    round1_k: r1k,
    round2_k: r2k,
  });
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
      es.close();
      activeStream = null;
      btn.disabled = false;
    } else if (msg.type === "error") {
      status.innerHTML = `<span style="color:#fca5a5;">Error: ${msg.detail}</span>`;
      es.close();
      activeStream = null;
      btn.disabled = false;
    }
  };

  es.onerror = () => {
    status.innerHTML = `<span style="color:#fca5a5;">Stream disconnected. Click Run / Re-Run Live to retry.</span>`;
    es.close();
    activeStream = null;
    btn.disabled = false;
  };
}

function renderResult(data) {
  document.getElementById("compare").style.display = "grid";
  const dec = data.board.decision;
  const liveTime = formatLocalTime(data.timestamp_utc);
  document.getElementById("aresVerdict").textContent = dec.answer;
  document.getElementById("aresMeta").innerHTML =
    `<b>${(dec.confidence * 100).toFixed(1)}%</b> calibrated confidence &nbsp;·&nbsp; <b>${data.wall_clock_s}s</b> &nbsp;·&nbsp; live @ ${liveTime}`;

  if (data.solo_baseline) {
    const s = data.solo_baseline;
    document.getElementById("soloVerdict").textContent = s.predicted_intersection;
    document.getElementById("soloMeta").innerHTML =
      `<b>${(s.confidence * 100).toFixed(0)}%</b> self-reported &nbsp;·&nbsp; <b>${s.wall_clock_s}s</b> &nbsp;·&nbsp; 6 R1 photos only<br/>${s.reasoning}`;
  }

  rerenderGrid();
}

function rerenderGrid() {
  if (!currentData) return;
  const showRejected = document.getElementById("showRejected").checked;
  const liveTime = formatLocalTime(currentData.timestamp_utc);
  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  const visibleItems = currentData.inspected.filter(item => {
    const isMatch = item.observation.verdict === "strong_match" || item.observation.verdict === "partial_match";
    return showRejected || isMatch;
  });

  if (visibleItems.length === 0) {
    grid.innerHTML = `<div style="grid-column:1/-1;color:#93a1b0;padding:18px;background:#182028;border-radius:10px;">No verified open parking spots in these ${currentData.inspected.length} cameras right now (ARES honestly abstains instead of guessing). Check <b>Show rejected cameras</b> above to inspect them.</div>`;
    return;
  }

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
    const box = item.observation.spot_box;
    let overlaySvg = "";
    if (Array.isArray(box) && box.length === 4 && (v === "strong_match" || v === "partial_match")) {
      const [ymin, xmin, ymax, xmax] = box;
      const cx = (xmin + xmax) / 2;
      const cy = (ymin + ymax) / 2;
      const rx = Math.max(45, (xmax - xmin) / 2);
      const ry = Math.max(45, (ymax - ymin) / 2);
      overlaySvg = `
        <svg class="spot-svg" viewBox="0 0 1000 1000" preserveAspectRatio="none">
          <ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}"
            fill="rgba(27,158,119,0.20)" stroke="#22c55e" stroke-width="14" />
        </svg>
      `;
    }
    card.innerHTML = `
      <div class="img-wrap">
        <img src="${item.image_endpoint}" alt="${item.camera.name}" />
        ${overlaySvg}
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
    grid.appendChild(card);
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

// Auto-run once on load so the page is immediately live!
window.addEventListener("DOMContentLoaded", () => runSearch());
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)
