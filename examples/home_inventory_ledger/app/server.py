"""FastAPI server for Home Inventory Ledger (`app/server.py`).

Endpoints:
- `GET  /`                           -> Single-page interactive Capture -> Verdict -> Ask -> Refilm app
- `GET  /api/config`                 -> Active Gemini 4 & Decision-1 model IDs, live API key status, Shop MCP status
- `POST /api/policy`                 -> Extract Coverage C limit from photographed declarations page (`gemini-4-carbon`)
- `POST /api/walkthrough`            -> Process live recorded or uploaded walkthrough video + policy into `LedgerBoard`
- `POST /api/demo/walkthrough`       -> Run pipeline on generated sample desk clip (live if API key set, else disclosed replay)
- `POST /api/session/{id}/followup`  -> Process 5s close-up follow-up clip or spoken answer -> flips badge to CERTIFIED
- `POST /api/session/{id}/demo-followup` -> Run follow-up clip (`Model A2992`) on active session
- `GET  /api/session/{id}`           -> Fetch current `LedgerBoard` + recent progress events
- `POST /api/session/{id}/cart`      -> Build 'Rebuild My Home' Shop replacement cart (`shop.dev` bonus track)
- `GET  /api/session/{id}/export.csv`  -> Download Schedule of Loss CSV (re-checks 10-min listing freshness)
- `GET  /api/session/{id}/export.html` -> Open printable PDF-ready Schedule of Loss with embedded visual proof
- `GET  /api/frame/{session_id}/{filename}` -> Serve extracted item proof crops/frames safely
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from commerce.shop_mcp import ShopMCPClient
from contracts import LedgerBoard, PolicyInfo
from eval.generate_dev_clips import ensure_dev_assets
from ledger.export import (
    export_schedule_csv,
    export_schedule_html,
    refresh_stale_prices_for_export,
)
from ledger.loop import InventoryLedgerPipeline
from ledger.verifiers import policy_limit_parse
from models.carbon import CarbonClient
from models.provider import GenAIRuntime

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSIONS_DIR = ROOT_DIR / "data" / "sessions"
UPLOADS_DIR = ROOT_DIR / "data" / "uploads"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Home Inventory Ledger",
    description="Multimodal Pre-Loss Home Inventory & Coverage Adequacy Agent (Gemini 4 Carbon + Hydrogen + Decision-1 + shop.dev)",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory session store for fast live demo interactions
SESSIONS: dict[str, LedgerBoard] = {}
SESSION_EVENTS: dict[str, list[dict]] = {}
SESSION_SHOPS: dict[str, ShopMCPClient] = {}


def _resolve_replay_flag(requested_replay: bool | None) -> bool:
    load_dotenv(ROOT_DIR / ".env", override=False)
    if requested_replay is True:
        return True
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not env_key or env_key == "your_gemini_api_key_here":
        # Automatically fall back to disclosed replay mode when no API key is set yet
        return True
    if requested_replay is False:
        return False
    return os.getenv("LEDGER_REPLAY_MODE", "false").lower() in {"1", "true", "yes"}


def _serialize_board_for_ui(board: LedgerBoard) -> dict:
    data = board.model_dump(mode="json")
    # Convert local frame paths to browser-loadable `/api/media/...` URLs
    for iid, item_dict in data.get("items", {}).items():
        fp = item_dict.get("best_frame_path")
        if fp and Path(fp).exists():
            rel = Path(fp).resolve()
            try:
                sub = rel.relative_to(SESSIONS_DIR.resolve())
                item_dict["frame_url"] = f"/api/media/{sub.as_posix()}"
            except ValueError:
                item_dict["frame_url"] = None
        else:
            item_dict["frame_url"] = None
    data["events"] = SESSION_EVENTS.get(board.session_id, [])
    return data


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config() -> dict:
    load_dotenv(ROOT_DIR / ".env", override=False)
    rt = GenAIRuntime()
    eval_json_path = ROOT_DIR / "eval" / "results" / "latest_eval.json"
    eval_summary = None
    if eval_json_path.exists():
        import json

        try:
            eval_summary = json.loads(eval_json_path.read_text())
        except Exception:
            eval_summary = None

    return {
        "has_live_gemini_key": rt.has_live_credentials,
        "default_mode": "live" if rt.has_live_credentials else "replay",
        "models": {
            "understander": rt.carbon_model,
            "extractor": rt.hydrogen_model,
            "decision_probe": rt.decision_model,
        },
        "shop_mcp_url": os.getenv("SHOP_MCP_URL", "https://mcp.shop.com"),
        "latest_eval": eval_summary,
    }


@app.get("/api/media/{file_path:path}")
def serve_session_media(file_path: str) -> FileResponse:
    target = (SESSIONS_DIR / file_path).resolve()
    if not str(target).startswith(str(SESSIONS_DIR.resolve())) or not target.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(target)


@app.post("/api/policy")
async def parse_policy_document(
    file: UploadFile | None = File(default=None),
    use_sample: bool = Form(default=False),
    replay: bool | None = Form(default=None),
) -> dict:
    is_replay = _resolve_replay_flag(replay)
    rt = GenAIRuntime(replay_mode=is_replay)
    carbon = CarbonClient(rt)

    if use_sample or file is None:
        assets = ensure_dev_assets(ROOT_DIR / "eval" / "dev_clips")
        img_path = Path(assets["policy_path"])
    else:
        ext = Path(file.filename or "policy.jpg").suffix or ".jpg"
        img_path = UPLOADS_DIR / f"policy_{uuid.uuid4().hex[:8]}{ext}"
        with img_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

    info: PolicyInfo = carbon.extract_policy_only(img_path)
    valid, details = policy_limit_parse(info.coverage_c_limit, info.deductible)
    return {
        "policy": info.model_dump(mode="json"),
        "verifier_passed": valid,
        "verifier_details": details,
        "mode": "replay" if is_replay else "live",
    }


@app.post("/api/walkthrough")
async def process_walkthrough(
    video: UploadFile = File(...),
    policy_image: UploadFile | None = File(default=None),
    coverage_limit: float | None = Form(default=None),
    deductible: float | None = Form(default=None),
    expected_rooms: str = Form(default=""),
    replay: bool | None = Form(default=None),
) -> dict:
    session_id = f"hil_{uuid.uuid4().hex[:8]}"
    ext = Path(video.filename or "walkthrough.webm").suffix or ".webm"
    vpath = UPLOADS_DIR / f"{session_id}{ext}"
    with vpath.open("wb") as out:
        shutil.copyfileobj(video.file, out)

    p_path: Path | None = None
    if policy_image is not None and policy_image.filename:
        p_ext = Path(policy_image.filename).suffix or ".jpg"
        p_path = UPLOADS_DIR / f"{session_id}_policy{p_ext}"
        with p_path.open("wb") as out:
            shutil.copyfileobj(policy_image.file, out)

    is_replay = _resolve_replay_flag(replay)
    rt = GenAIRuntime(replay_mode=is_replay)
    shop = ShopMCPClient(replay_mode=is_replay)
    pipeline = InventoryLedgerPipeline(work_dir=SESSIONS_DIR, runtime=rt, shop_client=shop)

    SESSION_EVENTS[session_id] = []

    def _on_prog(stage: str, msg: str, meta: dict) -> None:
        SESSION_EVENTS[session_id].append({"stage": stage, "message": msg, "meta": meta})

    rooms_list = [r.strip() for r in expected_rooms.split(",") if r.strip()] or None
    board = pipeline.run_walkthrough(
        video_path=vpath,
        session_id=session_id,
        dec_page_path=p_path,
        manual_coverage_limit=coverage_limit,
        manual_deductible=deductible,
        expected_rooms=rooms_list,
        progress_cb=_on_prog,
    )
    SESSIONS[session_id] = board
    SESSION_SHOPS[session_id] = shop
    return _serialize_board_for_ui(board)


@app.post("/api/demo/walkthrough")
async def run_sample_walkthrough(
    coverage_limit: float = Form(default=7500.0),
    replay: bool | None = Form(default=None),
) -> dict:
    assets = ensure_dev_assets(ROOT_DIR / "eval" / "dev_clips")
    session_id = f"demo_{uuid.uuid4().hex[:6]}"
    is_replay = _resolve_replay_flag(replay)
    rt = GenAIRuntime(replay_mode=is_replay)
    shop = ShopMCPClient(replay_mode=is_replay)
    pipeline = InventoryLedgerPipeline(work_dir=SESSIONS_DIR, runtime=rt, shop_client=shop)

    SESSION_EVENTS[session_id] = []

    def _on_prog(stage: str, msg: str, meta: dict) -> None:
        SESSION_EVENTS[session_id].append({"stage": stage, "message": msg, "meta": meta})

    board = pipeline.run_walkthrough(
        video_path=assets["walkthrough_path"],
        session_id=session_id,
        dec_page_path=assets["policy_path"],
        manual_coverage_limit=coverage_limit,
        manual_deductible=500.0,
        progress_cb=_on_prog,
    )
    SESSIONS[session_id] = board
    SESSION_SHOPS[session_id] = shop
    return _serialize_board_for_ui(board)


@app.post("/api/session/{session_id}/followup")
async def process_followup(
    session_id: str,
    request_id: str = Form(...),
    video: UploadFile | None = File(default=None),
    spoken_answer: str | None = Form(default=None),
    replay: bool | None = Form(default=None),
) -> dict:
    board = SESSIONS.get(session_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Session not found")

    fpath: Path | None = None
    if video is not None and video.filename:
        ext = Path(video.filename).suffix or ".webm"
        fpath = UPLOADS_DIR / f"{session_id}_fup_{board.board_version + 1}{ext}"
        with fpath.open("wb") as out:
            shutil.copyfileobj(video.file, out)

    is_replay = _resolve_replay_flag(replay if replay is not None else (board.telemetry.mode == "replay"))
    rt = GenAIRuntime(replay_mode=is_replay)
    shop = SESSION_SHOPS.get(session_id) or ShopMCPClient(replay_mode=is_replay)
    pipeline = InventoryLedgerPipeline(work_dir=SESSIONS_DIR, runtime=rt, shop_client=shop)

    def _on_prog(stage: str, msg: str, meta: dict) -> None:
        SESSION_EVENTS.setdefault(session_id, []).append({"stage": stage, "message": msg, "meta": meta})

    updated = pipeline.run_followup(
        board=board,
        followup_video_path=fpath,
        request_id=request_id,
        spoken_or_text_answer=spoken_answer,
        progress_cb=_on_prog,
    )
    SESSIONS[session_id] = updated
    return _serialize_board_for_ui(updated)


@app.post("/api/session/{session_id}/demo-followup")
async def run_sample_followup(
    session_id: str,
    request_id: str | None = Form(default=None),
) -> dict:
    board = SESSIONS.get(session_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Session not found")

    assets = ensure_dev_assets(ROOT_DIR / "eval" / "dev_clips")
    is_replay = board.telemetry.mode == "replay"
    rt = GenAIRuntime(replay_mode=is_replay)
    shop = SESSION_SHOPS.get(session_id) or ShopMCPClient(replay_mode=is_replay)
    pipeline = InventoryLedgerPipeline(work_dir=SESSIONS_DIR, runtime=rt, shop_client=shop)

    req_id = request_id or (board.requests[0].request_id if board.requests else None)
    updated = pipeline.run_followup(
        board=board,
        followup_video_path=assets["followup_path"],
        request_id=req_id,
    )
    SESSIONS[session_id] = updated
    return _serialize_board_for_ui(updated)


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> dict:
    board = SESSIONS.get(session_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _serialize_board_for_ui(board)


@app.post("/api/session/{session_id}/cart")
def build_replacement_cart(session_id: str) -> dict:
    board = SESSIONS.get(session_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Session not found")
    shop = SESSION_SHOPS.get(session_id) or ShopMCPClient(replay_mode=(board.telemetry.mode == "replay"))
    refreshed = refresh_stale_prices_for_export(board, shop)
    SESSIONS[session_id] = refreshed
    return shop.create_replacement_cart(list(refreshed.items.values()))


@app.get("/api/session/{session_id}/export.csv")
def download_csv(session_id: str) -> PlainTextResponse:
    board = SESSIONS.get(session_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Session not found")
    shop = SESSION_SHOPS.get(session_id)
    refreshed = refresh_stale_prices_for_export(board, shop)
    SESSIONS[session_id] = refreshed
    csv_text = export_schedule_csv(refreshed)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="schedule_of_loss_{session_id}.csv"'},
    )


@app.get("/api/session/{session_id}/export.html", response_class=HTMLResponse)
def view_printable_report(session_id: str) -> HTMLResponse:
    board = SESSIONS.get(session_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Session not found")
    shop = SESSION_SHOPS.get(session_id)
    refreshed = refresh_stale_prices_for_export(board, shop)
    SESSIONS[session_id] = refreshed
    return HTMLResponse(content=export_schedule_html(refreshed))
