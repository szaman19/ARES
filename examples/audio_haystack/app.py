"""ARES Audio Haystack ("Is This Your Card?") — Single-App FastAPI Server.

Cloudflare Containers ready:
- Single FastAPI app bound to `0.0.0.0` on `$PORT` (default 8080).
- All credentials read from environment variables (`GEMINI_API_KEY`).
- Ephemeral-safe: 1,000-card haystack deterministically initializes on startup (~1.5s)
  and per-run mic uploads live in `$ARES_AUDIO_WORKDIR` (defaults to OS temp dir).

Run locally:
    uv run uvicorn examples.audio_haystack.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import json
import os
import queue
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from examples.audio_haystack.domain import (
    DEFAULT_TOTAL_CARDS,
    AudioHaystackStore,
    ScrambledRunContext,
    generate_sample_voice_wav,
    run_audio_haystack_race,
)

load_dotenv()

TOTAL_CARDS = int(os.environ.get("ARES_AUDIO_TOTAL_CARDS", str(DEFAULT_TOTAL_CARDS)))
MAX_CACHED_RUNS = 24


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    get_store()
    yield


app = FastAPI(title="ARES Audio Haystack — Is This Your Card?", lifespan=_lifespan)

_STORE: AudioHaystackStore | None = None
_STORE_LOCK = threading.Lock()
_RUNS: OrderedDict[str, ScrambledRunContext] = OrderedDict()
_RUNS_LOCK = threading.Lock()


def get_store() -> AudioHaystackStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = AudioHaystackStore(total_cards=TOTAL_CARDS)
    return _STORE


def _put_run(ctx: ScrambledRunContext) -> None:
    with _RUNS_LOCK:
        _RUNS[ctx.run_id] = ctx
        while len(_RUNS) > MAX_CACHED_RUNS:
            _RUNS.popitem(last=False)


def _get_run(run_id: str) -> ScrambledRunContext:
    with _RUNS_LOCK:
        ctx = _RUNS.get(run_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Run not found (container may have restarted).")
    return ctx


@app.get("/health")
def health() -> dict[str, Any]:
    store = get_store()
    return {
        "status": "ok",
        "total_cards": store.total_cards,
        "build_ms": store.build_ms,
        "has_gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.get("/api/haystack/info")
def api_haystack_info() -> dict[str, Any]:
    store = get_store()
    return {
        "total_cards": store.total_cards,
        "build_ms": store.build_ms,
        "has_gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.get("/api/haystack/cards/{card_num}/audio")
def api_card_audio(
    card_num: int,
    run_id: str | None = Query(default=None),
) -> Response:
    try:
        if run_id:
            ctx = _get_run(run_id)
            wav_bytes = ctx.get_card_wav(card_num)
        else:
            wav_bytes = get_store().get_base_wav(card_num)
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/runs/{run_id}/user-audio")
def api_user_audio(run_id: str) -> Response:
    ctx = _get_run(run_id)
    return Response(
        content=ctx.user_wav_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/runs/{run_id}/spectrogram/{filename}")
def api_spectrogram(run_id: str, filename: str) -> FileResponse:
    ctx = _get_run(run_id)
    safe_name = Path(filename).name
    fpath = ctx.run_dir / "artifacts" / safe_name
    if not fpath.is_file():
        raise HTTPException(status_code=404, detail="Spectrogram artifact not found.")
    return FileResponse(str(fpath), media_type="image/png")


class CreateRunRequest(BaseModel):
    clue: str = "Find the real human voice recording"
    use_sample: bool = False
    audio_wav_b64: str | None = None


@app.post("/api/runs")
def api_create_run(req: CreateRunRequest) -> dict[str, Any]:
    store = get_store()
    if req.use_sample or not req.audio_wav_b64:
        wav_bytes = generate_sample_voice_wav()
    else:
        try:
            wav_bytes = base64.b64decode(req.audio_wav_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid base64 WAV: {exc}") from exc
        if len(wav_bytes) < 44:
            raise HTTPException(status_code=400, detail="Uploaded WAV audio is too short.")

    run_id = uuid.uuid4().hex[:10]
    try:
        ctx = ScrambledRunContext(
            run_id=run_id,
            clue=req.clue,
            store=store,
            user_wav_bytes=wav_bytes,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not decode WAV audio: {exc}") from exc

    _put_run(ctx)
    return {
        "run_id": run_id,
        "clue": ctx.clue,
        "total_cards": ctx.total_cards,
        "secret_card_num": ctx.secret_card_num,
        "secret_card_id": ctx.secret_card_id,
        "user_speech_likeness": ctx.user_features.speech_likeness_index,
        "user_audio_url": f"/api/runs/{run_id}/user-audio",
    }


@app.get("/api/runs/{run_id}/stream")
def api_stream_race(
    run_id: str,
    ares_top_k: int = Query(default=12, ge=4, le=20),
    regular_budget: int = Query(default=10, ge=3, le=20),
) -> StreamingResponse:
    ctx = _get_run(run_id)
    ev_q: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def _on_event(ev: dict[str, Any]) -> None:
        ev_q.put({"type": "event", **ev})

    def _worker() -> None:
        try:
            race = run_audio_haystack_race(
                ctx,
                ares_top_k=ares_top_k,
                regular_budget=regular_budget,
                on_event=_on_event,
            )
            payload = race.model_dump(mode="json")
            if race.ares.spectrogram_artifact is not None:
                fname = Path(race.ares.spectrogram_artifact.path).name
                payload["ares"]["spectrogram_url"] = f"/api/runs/{run_id}/spectrogram/{fname}"
            else:
                payload["ares"]["spectrogram_url"] = None
            ev_q.put({"type": "result", "data": payload})
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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML_PAGE


_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ARES Audio Haystack — Is This Your Card?</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131a26;
    --border: #243044;
    --text: #e6edf7;
    --muted: #8b9bb4;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.45;
  }
  .wrap { max-width: 1360px; margin: 0 auto; padding: 20px; }
  header {
    display: flex; align-items: center; justify-content: space-between;
    gap: 16px; flex-wrap: wrap; margin-bottom: 18px;
  }
  h1 { margin: 0; font-size: 1.45rem; letter-spacing: -0.02em; }
  .subtitle { color: var(--muted); font-size: 0.92rem; margin-top: 2px; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 11px; border-radius: 999px; font-size: 0.8rem;
    background: #172031; border: 1px solid var(--border); color: var(--muted);
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 16px;
  }
  .controls-grid {
    display: grid; grid-template-columns: 1.1fr 1.3fr auto; gap: 14px; align-items: end;
  }
  @media (max-width: 900px) { .controls-grid { grid-template-columns: 1fr; } }
  label { display: block; font-size: 0.78rem; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
  input[type=text] {
    width: 100%; padding: 10px 12px; border-radius: 9px;
    border: 1px solid var(--border); background: #0c121d; color: var(--text); font-size: 0.95rem;
  }
  .btn-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button {
    cursor: pointer; border: none; border-radius: 9px; padding: 10px 15px;
    font-weight: 600; font-size: 0.9rem; transition: transform .08s, opacity .15s;
  }
  button:active { transform: scale(0.98); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-rec { background: var(--red); color: #fff; }
  .btn-rec.recording { animation: pulse 1s infinite; }
  .btn-sec { background: #1e293b; color: var(--text); border: 1px solid var(--border); }
  .btn-go { background: linear-gradient(135deg, #22c55e, #0ea5e9); color: #041016; font-weight: 700; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,.55); } 50% { box-shadow: 0 0 0 10px rgba(239,68,68,0); } }
  #waveCanvas {
    width: 100%; height: 48px; background: #0c121d;
    border: 1px solid var(--border); border-radius: 8px; display: block;
  }
  .secret-bar {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; flex-wrap: wrap; padding: 10px 14px; border-radius: 10px;
    background: #0f1726; border: 1px dashed #334155; margin-top: 12px;
  }
  .secret-num {
    font-family: ui-monospace, SFMono-Regular, monospace; font-weight: 800;
    font-size: 1.05rem; padding: 3px 10px; border-radius: 6px;
    background: #1e293b; color: var(--accent); letter-spacing: 0.06em;
  }
  .secret-num.blurred { filter: blur(6px); user-select: none; }
  .race-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  }
  @media (max-width: 960px) { .race-grid { grid-template-columns: 1fr; } }
  .engine-head {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px;
  }
  .engine-title { font-weight: 700; font-size: 1.05rem; display: flex; align-items: center; gap: 8px; }
  .badge {
    font-size: 0.72rem; padding: 2px 8px; border-radius: 999px; font-weight: 700; text-transform: uppercase;
  }
  .badge-ares { background: rgba(34,197,94,.18); color: var(--green); border: 1px solid rgba(34,197,94,.4); }
  .badge-reg { background: rgba(245,158,11,.18); color: var(--amber); border: 1px solid rgba(245,158,11,.4); }
  .tile-grid {
    display: grid; grid-template-columns: repeat(40, 1fr); gap: 2px;
    background: #090d14; padding: 8px; border-radius: 10px; border: 1px solid var(--border);
  }
  .tile {
    aspect-ratio: 1; border-radius: 2px; background: #263246; cursor: pointer;
    transition: background .18s, transform .15s, opacity .2s;
  }
  .tile:hover { transform: scale(1.6); z-index: 5; background: #94a3b8; }
  .tile.pruned { background: #111827; opacity: 0.35; }
  .tile.survivor { background: #0284c7; box-shadow: 0 0 4px rgba(56,189,248,.6); }
  .tile.listening { background: #a855f7; animation: pulse 0.8s infinite; }
  .tile.checked-miss { background: #b45309; }
  .tile.checked-hit { background: #22c55e; box-shadow: 0 0 8px #22c55e; transform: scale(1.35); z-index: 4; }
  .tile.reg-miss { background: #dc2626; }
  .tile.winner { background: #22c55e !important; box-shadow: 0 0 12px #22c55e; transform: scale(1.65); z-index: 6; }
  .log-box {
    margin-top: 12px; height: 340px; min-height: 220px; resize: vertical;
    overflow-y: auto; font-family: ui-monospace, monospace;
    font-size: 0.82rem; line-height: 1.5; background: #090d14; border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px; color: #cbd5e1;
  }
  .log-line { margin-bottom: 5px; word-break: break-word; }
  .reveal-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px;
  }
  @media (max-width: 900px) { .reveal-grid { grid-template-columns: 1fr; } }
  .reveal-card {
    border-radius: 12px; padding: 16px; border: 1px solid var(--border); background: #0e1522;
  }
  .reveal-card.hit { border-color: rgba(34,197,94,.65); background: rgba(34,197,94,.07); }
  .reveal-card.miss { border-color: rgba(239,68,68,.55); background: rgba(239,68,68,.06); }
  .big-card-num {
    font-family: ui-monospace, monospace; font-size: 2rem; font-weight: 900;
    letter-spacing: 0.04em; margin: 4px 0 8px;
  }
  .metrics-row {
    display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted); margin-top: 10px;
  }
  .metrics-row strong { color: var(--text); }
  audio { width: 100%; height: 34px; margin-top: 8px; }
  .spec-img {
    width: 100%; border-radius: 10px; border: 1px solid var(--border); margin-top: 12px;
  }
  .legend { display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.75rem; color: var(--muted); margin-top: 8px; }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>🎩 ARES Audio Haystack — "Is This Your Card?"</h1>
      <div class="subtitle">Record your voice → hide it randomly inside 100,000 numbered audio cards → race ARES Swarm vs. Autonomous Carbon</div>
    </div>
    <div class="pill" id="statusPill">Loading 100,000-card hat...</div>
  </header>

  <div class="card">
    <div class="controls-grid">
      <div>
        <label>1. Record Your Voice (3–5 sec)</label>
        <div class="btn-row" style="margin-bottom:8px;">
          <button id="recBtn" class="btn-rec">🎙 Record Mic</button>
          <button id="sampleBtn" class="btn-sec" title="Use a built-in synthetic voice if no mic is available">Use Sample Clip</button>
          <span id="recStatus" style="font-size:0.82rem;color:var(--muted);">No recording yet</span>
        </div>
        <canvas id="waveCanvas" width="420" height="48"></canvas>
        <audio id="previewAudio" controls style="display:none;"></audio>
      </div>

      <div>
        <label>2. Search Clue (what did you say?)</label>
        <input id="clueInput" type="text" value="Find the real human voice recording" placeholder="e.g. 'The blue server in rack four is overheating'" />
        <div style="font-size:0.78rem;color:var(--muted);margin-top:6px;">
          Tip: Say any fun phrase into the mic, type a short clue here, and see which engine pulls your exact numbered card!
        </div>
      </div>

      <div>
        <div class="btn-row" style="flex-direction:column;align-items:stretch;">
          <button id="shuffleBtn" class="btn-sec" disabled style="padding:11px 16px;font-size:0.92rem;">
            🔀 Shuffle into Hat (Try Finding It Yourself!)
          </button>
          <button id="raceBtn" class="btn-go" disabled style="padding:13px 18px;font-size:0.96rem;">
            🏁 Start ARES vs. Carbon Race!
          </button>
        </div>
      </div>
    </div>

    <div class="secret-bar" id="secretBar" style="display:none;">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span style="color:var(--muted);font-size:0.85rem;">Hidden on secret card:</span>
        <span id="secretNum" class="secret-num blurred">#??????</span>
        <button id="peekBtn" class="btn-sec" style="padding:4px 10px;font-size:0.78rem;">👁 Peek</button>
      </div>
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
        <span style="font-size:0.8rem;color:var(--muted);">Try pulling a card yourself:</span>
        <input id="manualCardInput" type="text" placeholder="e.g. 42137" style="width:105px;padding:5px 8px;font-size:0.82rem;" />
        <button id="manualListenBtn" class="btn-sec" style="padding:5px 10px;font-size:0.8rem;">▶ Listen</button>
        <button id="randomListenBtn" class="btn-sec" style="padding:5px 10px;font-size:0.8rem;">🎲 Pull Random</button>
        <span id="humanTries" style="font-size:0.8rem;color:var(--amber);">0 tries</span>
        <span id="tileNowPlaying" style="font-size:0.8rem;color:var(--accent);"></span>
      </div>
    </div>
  </div>

  <div class="race-grid">
    <div class="card">
      <div class="engine-head">
        <div class="engine-title">⚡ ARES Parallel Swarm <span class="badge badge-ares">DuckDB + Parallel Gemini</span></div>
        <div id="aresStat" style="font-size:0.8rem;color:var(--muted);">Ready</div>
      </div>
      <div id="aresGrid" class="tile-grid"></div>
      <div class="legend">
        <span><i class="dot" style="background:#111827;border:1px solid #334155;"></i>Pruned by DuckDB</span>
        <span><i class="dot" style="background:#0284c7;"></i>Top Survivors</span>
        <span><i class="dot" style="background:#a855f7;"></i>Listening (Parallel)</span>
        <span><i class="dot" style="background:#22c55e;"></i>Pulled Card!</span>
      </div>
      <div id="aresLog" class="log-box"></div>
    </div>

    <div class="card">
      <div class="engine-head">
        <div class="engine-title">🤖 Regular Carbon Agent <span class="badge badge-reg">Autonomous SQL + Audio</span></div>
        <div id="regStat" style="font-size:0.8rem;color:var(--muted);">Ready</div>
      </div>
      <div id="regGrid" class="tile-grid"></div>
      <div class="legend">
        <span><i class="dot" style="background:#263246;"></i>Unchecked</span>
        <span><i class="dot" style="background:#a855f7;"></i>Listening Now</span>
        <span><i class="dot" style="background:#dc2626;"></i>Miss</span>
        <span><i class="dot" style="background:#22c55e;"></i>Pulled Card!</span>
      </div>
      <div id="regLog" class="log-box"></div>
    </div>
  </div>

  <div class="card" id="revealSection" style="display:none;">
    <h2 style="margin:0 0 6px;font-size:1.25rem;">🎩 Final Reveal</h2>
    <div id="revealSub" style="color:var(--muted);font-size:0.88rem;"></div>
    <div id="magicCardHero" style="margin:16px 0;display:flex;justify-content:center;"></div>
    <div class="reveal-grid">
      <div class="reveal-card" id="aresReveal"></div>
      <div class="reveal-card" id="regReveal"></div>
    </div>
    <div id="specContainer"></div>
  </div>
</div>

<audio id="tilePlayer" style="display:none;"></audio>

<script>
let totalCards = 100000;
const GRID_BUCKETS = 1000;
let recordedWavBlob = null;
let useSampleClip = false;
let activeRunId = null;
let secretCardNum = null;
let secretRevealed = false;
let currentSource = null;

// Mic recording state (pure Web Audio PCM -> WAV, no ffmpeg needed)
let audioCtx = null, mediaStream = null, scriptNode = null, pcmChunks = [], isRecording = false, recTimer = null;

const statusPill = document.getElementById('statusPill');
const recBtn = document.getElementById('recBtn');
const sampleBtn = document.getElementById('sampleBtn');
const recStatus = document.getElementById('recStatus');
const previewAudio = document.getElementById('previewAudio');
const shuffleBtn = document.getElementById('shuffleBtn');
const raceBtn = document.getElementById('raceBtn');
const manualCardInput = document.getElementById('manualCardInput');
const manualListenBtn = document.getElementById('manualListenBtn');
const randomListenBtn = document.getElementById('randomListenBtn');
const humanTriesEl = document.getElementById('humanTries');
let humanTries = 0;
const clueInput = document.getElementById('clueInput');
const secretBar = document.getElementById('secretBar');
const secretNumEl = document.getElementById('secretNum');
const peekBtn = document.getElementById('peekBtn');
const aresGrid = document.getElementById('aresGrid');
const regGrid = document.getElementById('regGrid');
const aresLog = document.getElementById('aresLog');
const regLog = document.getElementById('regLog');
const aresStat = document.getElementById('aresStat');
const regStat = document.getElementById('regStat');
const tilePlayer = document.getElementById('tilePlayer');
const tileNowPlaying = document.getElementById('tileNowPlaying');
const waveCanvas = document.getElementById('waveCanvas');
const waveCtx = waveCanvas.getContext('2d');

function drawWave(samples) {
  const w = waveCanvas.width, h = waveCanvas.height;
  waveCtx.fillStyle = '#0c121d';
  waveCtx.fillRect(0, 0, w, h);
  waveCtx.strokeStyle = '#38bdf8';
  waveCtx.lineWidth = 1.5;
  waveCtx.beginPath();
  const step = Math.max(1, Math.floor(samples.length / w));
  for (let x = 0; x < w; x++) {
    const v = samples[x * step] || 0;
    const y = (0.5 - v * 0.45) * h;
    if (x === 0) waveCtx.moveTo(x, y); else waveCtx.lineTo(x, y);
  }
  waveCtx.stroke();
}

function encodeWav(floatSamples, sampleRate) {
  const buf = new ArrayBuffer(44 + floatSamples.length * 2);
  const view = new DataView(buf);
  const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + floatSamples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, floatSamples.length * 2, true);
  let off = 44;
  for (let i = 0; i < floatSamples.length; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, floatSamples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function cardToBucket(cardNum) {
  const perBucket = Math.max(1, Math.ceil(totalCards / GRID_BUCKETS));
  return Math.min(GRID_BUCKETS, Math.max(1, Math.ceil(cardNum / perBucket)));
}

function buildGrid(container, prefix) {
  container.innerHTML = '';
  const frag = document.createDocumentFragment();
  const perBucket = Math.max(1, Math.ceil(totalCards / GRID_BUCKETS));
  for (let b = 1; b <= GRID_BUCKETS; b++) {
    const startCard = (b - 1) * perBucket + 1;
    const endCard = Math.min(totalCards, b * perBucket);
    const d = document.createElement('div');
    d.className = 'tile';
    d.id = `${prefix}_${b}`;
    d.title = perBucket > 1
      ? `Cards #${startCard.toLocaleString()}–#${endCard.toLocaleString()} — click to sample audio`
      : `Card #${String(startCard).padStart(4, '0')} — click to listen`;
    d.onclick = () => playCard(startCard);
    frag.appendChild(d);
  }
  container.appendChild(frag);
}

function playCard(cardNum) {
  const clamped = Math.max(1, Math.min(totalCards, Number(cardNum) || 1));
  const url = activeRunId
    ? `/api/haystack/cards/${clamped}/audio?run_id=${encodeURIComponent(activeRunId)}`
    : `/api/haystack/cards/${clamped}/audio`;
  tilePlayer.src = url;
  tilePlayer.play().catch(() => {});
  const padLen = String(totalCards).length;
  const label = `#${String(clamped).padStart(padLen, '0')}`;
  if (activeRunId) {
    humanTries += 1;
    const hit = secretCardNum && clamped === secretCardNum;
    humanTriesEl.textContent = hit
      ? `🎉 You found it in ${humanTries} tries!`
      : `${humanTries} ${humanTries === 1 ? 'try' : 'tries'} / ${totalCards.toLocaleString()}`;
    tileNowPlaying.textContent = hit ? `▶ ${label} (YOUR VOICE!!)` : `▶ Playing ${label} (nope!)`;
  } else {
    tileNowPlaying.textContent = `▶ Playing ${label}`;
  }
}

function appendLog(el, text) {
  const div = document.createElement('div');
  div.className = 'log-line';
  div.textContent = text;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

async function init() {
  const res = await fetch('/api/haystack/info');
  const data = await res.json();
  totalCards = data.total_cards;
  statusPill.textContent = `${totalCards.toLocaleString()} audio cards ready (${data.build_ms}ms) • Gemini ${data.has_gemini_key ? 'OK' : 'MISSING KEY'}`;
  shuffleBtn.textContent = `🔀 Shuffle into ${totalCards.toLocaleString()} Cards (Try Yourself!)`;
  raceBtn.textContent = `🏁 Start ARES vs. Carbon Race!`;
  buildGrid(aresGrid, 'a');
  buildGrid(regGrid, 'r');
  drawWave(new Float32Array(128));
}

async function startRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(mediaStream);
    scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
    pcmChunks = [];
    scriptNode.onaudioprocess = (e) => {
      if (!isRecording) return;
      const ch = e.inputBuffer.getChannelData(0);
      pcmChunks.push(new Float32Array(ch));
      drawWave(ch);
    };
    src.connect(scriptNode);
    scriptNode.connect(audioCtx.destination);
    isRecording = true;
    useSampleClip = false;
    recBtn.textContent = '⏹ Stop Recording';
    recBtn.classList.add('recording');
    recStatus.textContent = 'Recording... speak clearly (auto-stops at 4s)';
    recTimer = setTimeout(() => { if (isRecording) stopRecording(); }, 4000);
  } catch (err) {
    recStatus.textContent = `Mic error: ${err.message}. Try "Use Sample Clip"!`;
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  clearTimeout(recTimer);
  recBtn.textContent = '🎙 Re-Record Mic';
  recBtn.classList.remove('recording');
  const sr = audioCtx ? audioCtx.sampleRate : 16000;
  if (scriptNode) scriptNode.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  if (audioCtx) audioCtx.close();

  const totalLen = pcmChunks.reduce((acc, c) => acc + c.length, 0);
  const merged = new Float32Array(totalLen);
  let off = 0;
  for (const c of pcmChunks) { merged.set(c, off); off += c.length; }
  recordedWavBlob = encodeWav(merged, sr);
  drawWave(merged);
  previewAudio.src = URL.createObjectURL(recordedWavBlob);
  previewAudio.style.display = 'block';
  recStatus.textContent = `Recorded ${(totalLen / sr).toFixed(1)}s WAV — shuffle into the hat!`;
  activeRunId = null;
  shuffleBtn.disabled = false;
  raceBtn.disabled = false;
}

recBtn.onclick = () => isRecording ? stopRecording() : startRecording();

sampleBtn.onclick = () => {
  useSampleClip = true;
  recordedWavBlob = null;
  previewAudio.style.display = 'none';
  recStatus.textContent = 'Using built-in sample voice clip — shuffle into the hat!';
  activeRunId = null;
  shuffleBtn.disabled = false;
  raceBtn.disabled = false;
};

manualListenBtn.onclick = () => {
  const raw = (manualCardInput.value || '').replace(/[^0-9]/g, '');
  if (!raw) return;
  playCard(parseInt(raw, 10));
};

randomListenBtn.onclick = () => {
  const r = Math.floor(Math.random() * totalCards) + 1;
  manualCardInput.value = String(r);
  playCard(r);
};

async function shuffleIntoHat() {
  shuffleBtn.disabled = true;
  document.getElementById('revealSection').style.display = 'none';
  buildGrid(aresGrid, 'a');
  buildGrid(regGrid, 'r');
  aresLog.innerHTML = '';
  regLog.innerHTML = '';
  humanTries = 0;
  humanTriesEl.textContent = `0 tries / ${totalCards.toLocaleString()}`;
  tileNowPlaying.textContent = '';

  let audioB64 = null;
  if (!useSampleClip && recordedWavBlob) {
    const ab = await recordedWavBlob.arrayBuffer();
    const bytes = new Uint8Array(ab);
    let bin = '';
    for (let i = 0; i < bytes.length; i += 8192) {
      bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
    }
    audioB64 = btoa(bin);
  }

  const createRes = await fetch('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      clue: clueInput.value || 'Find the real human voice recording',
      use_sample: useSampleClip,
      audio_wav_b64: audioB64,
    }),
  });
  if (!createRes.ok) {
    const err = await createRes.json().catch(() => ({ detail: 'Upload failed' }));
    alert(err.detail || 'Failed to shuffle');
    shuffleBtn.disabled = false;
    return null;
  }
  const runMeta = await createRes.json();
  activeRunId = runMeta.run_id;
  secretCardNum = runMeta.secret_card_num;
  secretRevealed = false;
  secretNumEl.textContent = runMeta.secret_card_id;
  secretNumEl.classList.add('blurred');
  peekBtn.textContent = '👁 Peek';
  secretBar.style.display = 'flex';
  appendLog(aresLog, `Shuffled your voice into ${runMeta.total_cards.toLocaleString()} numbered cards! Click any tile or pull random cards to try finding it yourself, then hit Start Race!`);
  appendLog(regLog, `Shuffled your voice into ${runMeta.total_cards.toLocaleString()} numbered cards! Ready when you hit Start Race.`);
  shuffleBtn.disabled = false;
  return runMeta;
}

shuffleBtn.onclick = () => shuffleIntoHat();

peekBtn.onclick = () => {
  secretRevealed = !secretRevealed;
  secretNumEl.classList.toggle('blurred', !secretRevealed);
  peekBtn.textContent = secretRevealed ? '🙈 Hide' : '👁 Peek';
};

raceBtn.onclick = async () => {
  if (currentSource) { currentSource.close(); currentSource = null; }
  raceBtn.disabled = true;
  shuffleBtn.disabled = true;
  document.getElementById('revealSection').style.display = 'none';

  if (!activeRunId) {
    const meta = await shuffleIntoHat();
    if (!meta) { raceBtn.disabled = false; shuffleBtn.disabled = false; return; }
  }
  aresStat.textContent = 'Racing...';
  regStat.textContent = 'Racing...';

  const es = new EventSource(`/api/runs/${encodeURIComponent(activeRunId)}/stream`);
  currentSource = es;

  es.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === 'event') {
      if (ev.engine === 'ares') handleAresEvent(ev);
      if (ev.engine === 'regular') handleRegEvent(ev);
    } else if (ev.type === 'result') {
      es.close();
      currentSource = null;
      renderFinalReveal(ev.data);
      raceBtn.disabled = false;
      shuffleBtn.disabled = false;
    } else if (ev.type === 'error') {
      es.close();
      currentSource = null;
      appendLog(aresLog, `ERROR: ${ev.detail}`);
      raceBtn.disabled = false;
      shuffleBtn.disabled = false;
    }
  };
  es.onerror = () => {
    es.close();
    currentSource = null;
    raceBtn.disabled = false;
    shuffleBtn.disabled = false;
  };
};

function handleAresEvent(ev) {
  if (ev.phase === 'pruning') {
    aresStat.textContent = 'Parallel SQL swarm...';
    appendLog(aresLog, ev.message);
  } else if (ev.phase === 'sql_query') {
    appendLog(aresLog, `[R${ev.round} SQL:${ev.worker}] ${ev.sql}`);
  } else if (ev.phase === 'sql_error') {
    appendLog(aresLog, `[SQL err:${ev.worker}] ${ev.error}`);
  } else if (ev.phase === 'pruned') {
    const survBuckets = new Set(ev.survivors.map(cardToBucket));
    for (let b = 1; b <= GRID_BUCKETS; b++) {
      const el = document.getElementById(`a_${b}`);
      if (!el) continue;
      el.className = survBuckets.has(b) ? 'tile survivor' : 'tile pruned';
    }
    aresStat.textContent = `Pruned ${ev.pruned_count.toLocaleString()} in ${ev.prune_ms}ms → ${ev.survivors.length} survivors`;
    appendLog(aresLog, ev.message);
  } else if (ev.phase === 'listening') {
    const el = document.getElementById(`a_${cardToBucket(ev.card_num)}`);
    if (el) el.className = 'tile listening';
  } else if (ev.phase === 'card_checked') {
    const el = document.getElementById(`a_${cardToBucket(ev.card_num)}`);
    const hit = ev.is_human_speech && ev.match_score >= 0.6;
    if (el) el.className = hit ? 'tile checked-hit' : 'tile checked-miss';
    appendLog(aresLog, `${ev.card_id}: ${ev.is_human_speech ? 'VOICE' : 'noise'} (${Math.round(ev.match_score*100)}%) — ${ev.transcript}`);
  } else if (ev.phase === 'probing') {
    aresStat.textContent = 'Counterfactual probe...';
    appendLog(aresLog, ev.message);
  } else if (ev.phase === 'done') {
    const s = ev.summary;
    aresStat.textContent = `Done in ${(s.wall_clock_ms/1000).toFixed(1)}s → ${s.picked_card_id}`;
    if (s.picked_card_num) {
      const el = document.getElementById(`a_${cardToBucket(s.picked_card_num)}`);
      if (el) el.className = 'tile winner';
    }
  }
}

function handleRegEvent(ev) {
  if (ev.phase === 'starting') {
    regStat.textContent = 'Planning strategy...';
    appendLog(regLog, ev.message);
  } else if (ev.phase === 'sql_query') {
    regStat.textContent = `Running SQL #${ev.query_num}...`;
    appendLog(regLog, `[SQL #${ev.query_num}] ${ev.sql}`);
  } else if (ev.phase === 'listening') {
    const el = document.getElementById(`r_${cardToBucket(ev.card_num)}`);
    if (el) el.className = 'tile listening';
    regStat.textContent = `Step ${ev.step}: listening to ${ev.card_id}...`;
  } else if (ev.phase === 'card_checked') {
    const el = document.getElementById(`r_${cardToBucket(ev.card_num)}`);
    const hit = ev.is_human_speech && ev.match_score >= 0.65;
    if (el) el.className = hit ? 'tile winner' : 'tile reg-miss';
    appendLog(regLog, `[#${ev.step}] ${ev.card_id}: ${ev.is_human_speech ? 'VOICE' : 'miss'} (${Math.round(ev.match_score*100)}%) — ${ev.transcript}`);
  } else if (ev.phase === 'done') {
    const s = ev.summary;
    regStat.textContent = `Done in ${(s.wall_clock_ms/1000).toFixed(1)}s → ${s.picked_card_id}`;
  }
}

function renderFinalReveal(data) {
  // Automatically reveal the secret card now that the race is over!
  secretRevealed = true;
  secretNumEl.classList.remove('blurred');
  peekBtn.textContent = '🙈 Hide';

  const sec = document.getElementById('revealSection');
  sec.style.display = 'block';
  document.getElementById('revealSub').innerHTML =
    `Your voice was hidden on <strong>Card ${data.secret_card_id}</strong> out of ${data.total_cards} cards. Press Play below to hear each agent's pull!`;

  const a = data.ares;
  const r = data.regular;

  // Big theatrical playing-card tile: "IS THIS YOUR CARD?" + #XXXXXX + voice note player!
  const aresPicked = Boolean(a.picked_card_num);
  const heroCardNum = aresPicked ? a.picked_card_num : r.picked_card_num;
  const heroCardId = aresPicked ? a.picked_card_id : (r.picked_card_num ? r.picked_card_id : '—');
  const heroMatched = aresPicked ? a.found_target : r.found_target;
  let heroTranscript = '';
  if (aresPicked && a.inspected) {
    const m = a.inspected.find(x => x.card_num === a.picked_card_num);
    heroTranscript = m ? m.transcript : '';
  } else if (r.picked_card_num && r.steps) {
    const m = r.steps.find(x => x.card_num === r.picked_card_num);
    heroTranscript = m ? m.transcript : '';
  }
  const heroBy = aresPicked ? 'ARES Swarm' : (r.picked_card_num ? 'Carbon Agent' : 'Neither');
  document.getElementById('magicCardHero').innerHTML = `
    <div style="
      width:340px; max-width:100%; border-radius:18px; padding:22px 24px; text-align:center;
      background: linear-gradient(160deg, #172538, #0d1420);
      border: 2px solid ${heroMatched ? '#22c55e' : '#38bdf8'};
      box-shadow: 0 14px 40px rgba(0,0,0,0.45), 0 0 28px ${heroMatched ? 'rgba(34,197,94,0.22)' : 'rgba(56,189,248,0.18)'};
    ">
      <div style="font-size:0.78rem;letter-spacing:0.14em;text-transform:uppercase;color:var(--accent);font-weight:800;">
        🃏 IS THIS YOUR CARD?
      </div>
      <div style="font-family:ui-monospace,monospace;font-size:2.5rem;font-weight:900;margin:10px 0 6px;color:${heroMatched ? '#4ade80' : '#f8fafc'};letter-spacing:0.06em;">
        ${heroCardId}
      </div>
      <div style="font-size:0.84rem;color:#cbd5e1;margin-bottom:10px;">
        ${heroTranscript ? `“${heroTranscript}”` : 'Pulled out of ' + data.total_cards.toLocaleString() + ' audio clips'}
      </div>
      ${heroCardNum ? `<audio controls src="/api/haystack/cards/${heroCardNum}/audio?run_id=${encodeURIComponent(data.run_id)}"></audio>` : ''}
      <div style="font-size:0.74rem;color:var(--muted);margin-top:8px;">
        ${heroMatched ? `✅ Exact match pulled by ${heroBy}!` : 'Pulled from the ' + data.total_cards.toLocaleString() + '-card hat'}
      </div>
    </div>
  `;

  const aresEl = document.getElementById('aresReveal');
  aresEl.className = `reveal-card ${a.found_target ? 'hit' : 'miss'}`;
  aresEl.innerHTML = `
    <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);">⚡ ARES Swarm Pulled:</div>
    <div class="big-card-num" style="color:${a.found_target ? 'var(--green)' : 'var(--red)'};">
      ${a.picked_card_id} ${a.found_target ? '✅ IS THIS YOUR CARD!' : '❌ Missed'}
    </div>
    <div style="font-size:0.86rem;color:#cbd5e1;">${a.rationale}</div>
    ${a.picked_card_num ? `<audio controls src="/api/haystack/cards/${a.picked_card_num}/audio?run_id=${encodeURIComponent(data.run_id)}"></audio>` : ''}
    <div class="metrics-row">
      <span>Time: <strong>${(a.wall_clock_ms/1000).toFixed(1)}s</strong></span>
      <span>Tool Calls: <strong>${a.tool_calls}</strong></span>
      <span>LLM Calls: <strong>${a.model_calls}</strong></span>
      <span>Tokens: <strong>${a.total_tokens.toLocaleString()}</strong></span>
      <span>Est. Cost: <strong>$${(a.estimated_cost_usd || 0).toFixed(4)}</strong></span>
      <span>Probe Conf: <strong>${Math.round(a.confidence*100)}%</strong></span>
    </div>
  `;

  const regEl = document.getElementById('regReveal');
  regEl.className = `reveal-card ${r.found_target ? 'hit' : 'miss'}`;
  regEl.innerHTML = `
    <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted);">🐢 Regular Agent Pulled:</div>
    <div class="big-card-num" style="color:${r.found_target ? 'var(--green)' : 'var(--red)'};">
      ${r.picked_card_id} ${r.found_target ? '✅ Found It!' : '❌ Stumped / Budget Exhausted'}
    </div>
    <div style="font-size:0.86rem;color:#cbd5e1;">${r.rationale}</div>
    ${r.picked_card_num ? `<audio controls src="/api/haystack/cards/${r.picked_card_num}/audio?run_id=${encodeURIComponent(data.run_id)}"></audio>` : ''}
    <div class="metrics-row">
      <span>Time: <strong>${(r.wall_clock_ms/1000).toFixed(1)}s</strong></span>
      <span>Tool Calls: <strong>${r.tool_calls}</strong></span>
      <span>LLM Calls: <strong>${r.model_calls}</strong></span>
      <span>Tokens: <strong>${r.total_tokens.toLocaleString()}</strong></span>
      <span>Est. Cost: <strong>$${(r.estimated_cost_usd || 0).toFixed(4)}</strong></span>
    </div>
  `;

  const specContainer = document.getElementById('specContainer');
  if (a.spectrogram_url) {
    specContainer.innerHTML = `<img class="spec-img" src="${a.spectrogram_url}" alt="Deterministic Spectrogram Comparison" />`;
  } else {
    specContainer.innerHTML = '';
  }
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("examples.audio_haystack.app:app", host="0.0.0.0", port=port)
