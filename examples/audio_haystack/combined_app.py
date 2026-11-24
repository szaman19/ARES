"""Combined ARES Multimodal Demo Server (`examples/audio_haystack/combined_app.py`).

Serves both ARES demos from a single FastAPI / Cloudflare Containers instance:
- `/`       : Landing hub with links to both live demos
- `/nyc`    : Live Interactive NYC Street-Camera Map (`examples/nyc_cameras/app.py`)
- `/audio`  : 100,000-Card Audio Haystack "Is This Your Card?" (`examples/audio_haystack/app.py`)
- `/api/*`  : All API endpoints from both apps (zero route collisions)
- `/health` : Combined container readiness check

Run locally:
    uv run uvicorn examples.audio_haystack.combined_app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute

from examples.audio_haystack import app as audio_haystack_app
from examples.nyc_cameras import app as nyc_live_app


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    audio_haystack_app.get_store()
    yield


app = FastAPI(title="ARES Multimodal Demos — NYC Live Map & Audio Haystack", lifespan=_lifespan)


def _mount_api_routes(target: FastAPI, source: FastAPI) -> None:
    """Register all non-root, non-health API routes from a sub-app onto the combined app."""
    for route in source.routes:
        if isinstance(route, APIRoute) and route.path not in ("/", "/health"):
            target.routes.append(route)


_mount_api_routes(app, nyc_live_app.app)
_mount_api_routes(app, audio_haystack_app.app)


def _inject_switcher(html: str, active: str) -> str:
    """Inject a compact top navigation bar to jump between both demos."""
    nyc_style = (
        "background:#1b9e77;color:#06120e;border-color:#1b9e77;"
        if active == "nyc"
        else "background:#16202c;color:#cbd5e1;border-color:#2a3642;"
    )
    audio_style = (
        "background:#38bdf8;color:#06131f;border-color:#38bdf8;"
        if active == "audio"
        else "background:#16202c;color:#cbd5e1;border-color:#2a3642;"
    )
    bar = (
        '<div style="position:sticky;top:0;z-index:9999;display:flex;align-items:center;'
        'justify-content:space-between;gap:12px;padding:8px 18px;background:rgba(11,15,23,0.94);'
        'backdrop-filter:blur(8px);border-bottom:1px solid #243044;font-family:system-ui,-apple-system,sans-serif;">'
        '<a href="/" style="color:#e6edf7;text-decoration:none;font-weight:700;font-size:13px;letter-spacing:0.02em;">'
        'ARES Multimodal Suite</a>'
        '<div style="display:flex;gap:8px;align-items:center;">'
        f'<a href="/nyc" style="padding:5px 12px;border-radius:999px;font-size:12px;font-weight:600;'
        f'text-decoration:none;border:1px solid;{nyc_style}">NYC Live Map</a>'
        f'<a href="/audio" style="padding:5px 12px;border-radius:999px;font-size:12px;font-weight:600;'
        f'text-decoration:none;border:1px solid;{audio_style}">Audio Haystack</a>'
        '</div></div>'
    )
    if "<body>" in html:
        return html.replace("<body>", f"<body>\n{bar}", 1)
    return bar + html


@app.get("/health")
def health() -> dict[str, Any]:
    store = audio_haystack_app.get_store()
    return {
        "ok": True,
        "status": "ok",
        "apps": ["nyc", "audio"],
        "audio_total_cards": store.total_cards,
        "audio_build_ms": store.build_ms,
        "has_gemini_key": bool(os.environ.get("GEMINI_API_KEY")),
    }


@app.get("/nyc", response_class=HTMLResponse)
def nyc_index() -> str:
    return _inject_switcher(nyc_live_app.HTML_PAGE, active="nyc")


@app.get("/audio", response_class=HTMLResponse)
def audio_index() -> str:
    return _inject_switcher(audio_haystack_app._HTML_PAGE, active="audio")


@app.get("/", response_class=HTMLResponse)
def hub_index() -> str:
    return _HUB_HTML


_HUB_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>ARES Multimodal Demo Suite</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131a26;
    --border: #243044;
    --text: #e6edf7;
    --muted: #8b9bb4;
    --green: #22c55e;
    --sky: #38bdf8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    min-height: 100vh;
    font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: radial-gradient(circle at 20% 0%, #152238 0%, var(--bg) 55%);
    color: var(--text);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 28px;
  }
  .wrap { max-width: 960px; width: 100%; }
  .badge {
    display: inline-block;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: #172031;
    border: 1px solid var(--border);
    color: var(--sky);
    margin-bottom: 14px;
  }
  h1 { margin: 0 0 8px; font-size: 2.1rem; letter-spacing: -0.025em; }
  .sub { color: var(--muted); font-size: 1.02rem; margin: 0 0 28px; max-width: 680px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
  .card {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    text-decoration: none;
    color: inherit;
    transition: transform 0.14s ease, border-color 0.14s ease, box-shadow 0.14s ease;
  }
  .card:hover {
    transform: translateY(-3px);
    border-color: var(--sky);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
  }
  .card.nyc:hover { border-color: var(--green); }
  .tag {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 10px;
  }
  .nyc .tag { color: var(--green); }
  .audio .tag { color: var(--sky); }
  .card h2 { margin: 0 0 8px; font-size: 1.35rem; }
  .card p { margin: 0 0 18px; color: var(--muted); font-size: 0.94rem; line-height: 1.5; }
  .cta {
    font-weight: 600;
    font-size: 0.92rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .nyc .cta { color: var(--green); }
  .audio .cta { color: var(--sky); }
</style>
</head>
<body>
  <div class="wrap">
    <div class="badge">Cloudflare Containers &middot; ARES Live Demos</div>
    <h1>ARES Multimodal Search Suite</h1>
    <p class="sub">
      Parallel Hypothesis Swarm, Deterministic Certification, and Counterfactual Decision Probing
      running live on real NYC street cameras and a 100,000-card audio haystack.
    </p>
    <div class="grid">
      <a class="card nyc" href="/nyc">
        <div>
          <div class="tag">Vision &middot; Live Geospatial</div>
          <h2>NYC Live Street-Camera Map</h2>
          <p>
            Drop a pin anywhere in NYC, scan live DOT traffic cameras in parallel with ARES vs. Gemini Solo,
            and inspect counterfactual evidence ablation on the winning curb.
          </p>
        </div>
        <div class="cta">Launch NYC Live Map &rarr;</div>
      </a>
      <a class="card audio" href="/audio">
        <div>
          <div class="tag">Audio &middot; 100k-Card Haystack</div>
          <h2>Audio Haystack (&ldquo;Is This Your Card?&rdquo;)</h2>
          <p>
            Record your voice or use a sample clip hidden among 100,000 numbered audio cards.
            Watch ARES DuckDB acoustic magnets + parallel Gemini swarm race an autonomous agent.
          </p>
        </div>
        <div class="cta">Launch Audio Haystack &rarr;</div>
      </a>
    </div>
  </div>
</body>
</html>
"""
