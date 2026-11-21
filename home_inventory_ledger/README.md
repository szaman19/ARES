# Home Inventory Ledger

> **Film your things for 60 seconds. We tell you today whether your insurance covers them, back every certified price with a live Shopify listing, and tell you the exact next shot to film to settle your coverage verdict.**

Built for the **Gemini 4 NYC Hackathon** (2026-11-21):
- **Primary Problem Statement:** **5 Sigma** (reliable multimodal multi-agent verification with 0% invented certified prices by construction, evaluated against `B0` and `B1` baselines with 95% Wilson intervals and an iterative failure log in [`eval/FAILURES.md`](eval/FAILURES.md)).
- **Multimodal Media & Stress Ladders (Element 119):** Raw walkthrough video, spoken audio narration, and photographed policy declarations pages are the sole inputs; degradation break-points across lighting, motion blur, speed, and audio noise are measured in [`eval/STRESS.md`](eval/STRESS.md).
- **Bonus Track:** **Build with Commerce (Shopify / `shop.dev`)** — live pricing via `https://mcp.shop.com` (`search_catalog`, `get_product`, 10-minute price TTL compliance) plus a one-click **"Rebuild My Home"** replacement cart manifest.

---

## 1. Model Roles (Gemini 4 + Decisions API)

Every model in the pipeline has a single, non-overlapping job:

| Model | Module | Clear Job in the System |
|---|---|---|
| **`gemini-4-carbon`** | [`models/carbon.py`](models/carbon.py) | **Global Multimodal Understander:** Watches the full walkthrough video + raw narration audio + photographed insurance declarations page in one pass to extract chronological room/area segments, timestamped spoken claims, and the Coverage C limit. Also serves as the single-call baseline `B0` in [`eval/baseline.py`](eval/baseline.py). |
| **`gemini-4-hydrogen`** | [`models/hydrogen.py`](models/hydrogen.py) | **Parallel Segment Extractors:** Concurrent workers (`ThreadPoolExecutor`) inspect keyframes per segment, read visibly legible hardware labels (strictly never guessing unseen characters), count simultaneous items per frame, and write ranked Shopify catalog search magnets. Also runs incremental 4s follow-up label extraction. |
| **`gemini-decision-1`** *(Decisions API)* | [`models/decision.py`](models/decision.py) | **Closed-Set Calibrated Verdicts & Counterfactual VoI Probe:** (1) Scores candidate Shop listings against item crops via `noul` (yes/no) questions with zero generative output tokens; (2) Evaluates coverage adequacy (`choice` + `noul` completeness); (3) Probes **one counterfactual board state per call** (ablating item fields to `"unknown"` and low/high bounds) to rank **Value-of-Information (`voi_usd`)** capture requests. |

---

## 2. Why Raw Video + Audio + Policy Image Matters (Beyond a Text Box)

1. **Zero Invented Prices (`CERTIFIED` vs. `UNCERTIFIED`):** An LLM reading a label is never trusted alone. Certification requires passing pure-Python deterministic verifiers ([`ledger/verifiers.py`](ledger/verifiers.py): `frame_quality`, `model_number_format` across 22 brands, `price_sanity` blocking `$24` case traps, `listing_freshness` <= 10 min, and `catalog_exact_match` on SKU/title) enforced by append-only board invariants I1–I7 ([`ledger/board.py`](ledger/board.py)).
2. **Spoken vs. Visual Contradiction Catching:** When a user narrates *"those are my Sony headphones"* while panning past earcups whose logo visibly reads **BOSE**, [`ledger/resolve.py`](ledger/resolve.py) aligns timestamps (`+-6s`) and flags `narration_conflict`, preserving visual ground truth.
3. **Active Value-of-Information (`VoI`) Re-Capture Loop:** Instead of dumping a static item table, Decision-1 computes which uncertain item swings the coverage verdict most and asks for one targeted 4-second shot (*"Show me the model label underneath the Apple MacBook Pro — worth +-$1,100"*), flipping the badge to `CERTIFIED` live.

---

## 3. Quickstart (15 Seconds)

```bash
# 1. Create virtualenv & install dependencies
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 2. Configure API key (optional for offline replay test; required for live Gemini 4 + Shop MCP)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=...

# 3. Run unit + invariant + end-to-end tests
.venv/bin/pytest -q

# 4. Launch the interactive web app (FastAPI + WebRTC camera/mic, no Streamlit)
.venv/bin/uvicorn app.server:app --reload --port 8000
```

Open **http://localhost:8000** on your laptop or phone.

---

## 4. Live 5 Sigma Evaluation Results (5 Unseen Open-Licensed Real Videos)

Evaluated **100% live** (`replay_mode=false`) across 5 unseen external room & desk videos downloaded via [`eval/datasets/download_external_clips.py`](eval/datasets/download_external_clips.py) (Wikimedia Commons Public Domain & CC-BY, `0` team-authored examples):

| System | Off-Market Prices >±15% (95% Wilson CI) | Uncited Point Prices | Duplicate Rows | Item F1 |
|---|---|---|---|---|
| **B0** · Single `gemini-4-carbon` call | **50/73 (68.5%, CI [57%, 78%])** | **73/73 (100.0%)** | 0/73 (0.0%) | 0.00 |
| **B1** · Single call + naive top-1 Shop MCP | **49/73 (67.1%, CI [56%, 77%])** | 0/73 (0.0%) | 0/73 (0.0%) | 0.00 |
| **Ledger** (`Carbon` + `Hydrogen ×N` + `Shop MCP` + `Verifiers` + `Decision-1`) | **0% false certifications** | **0/86 (0.0%, CI [0%, 4%])** | 10/86 (11.6%, CI [6%, 20%]) | **0.28** |

### Reproducing the Evaluation & Stress Ladders

```bash
# 1. Download external open-licensed benchmark videos into git-ignored eval/raw_clips/
.venv/bin/python -m eval.datasets.download_external_clips

# 2. Run B0 vs. B1 vs. Ledger live across the manifest (+ optional --stress ladders)
.venv/bin/python -m eval.run_eval --manifest eval/test_manifest.json
```

See:
- [`eval/FAILURES.md`](eval/FAILURES.md) — 4 concrete failure modes, root causes, verifier/invariant fixes, and before/after impact.
- [`eval/STRESS.md`](eval/STRESS.md) — Lighting, motion blur, audio messiness, and catalog coverage break-point ladders.

---

## 5. What We Built Today & Originality Statement (Hackathon Rule 5)

**100% of the code in this repository was written from scratch during the Gemini 4 NYC Hackathon on 2026-11-21** under the [MIT License](LICENSE). No external unlicensed code was copied (the conceptual pattern of separating certified verifier notes from uncertified ranges and counterfactual probing was cleanly designed and implemented from scratch in this repo).

| Module Built Today | Responsibility |
|---|---|
| [`contracts.py`](contracts.py) | Frozen/versioned Pydantic v2 contracts (`Frame`, `RoomSegment`, `SpokenClaim`, `ItemObservation`, `Listing`, `Note`, `LedgerBoard`, `CoverageDecision`, `CaptureRequest`). |
| [`ingest/frames.py`](ingest/frames.py), [`ingest/audio.py`](ingest/audio.py) | Deterministic `ffmpeg` keyframe sampler, 64-bit perceptual `dHash` deduplication, 2D Laplacian sharpness & luminance gates, cropper, and 16 kHz WAV extractor. |
| [`models/provider.py`](models/provider.py), [`models/carbon.py`](models/carbon.py), [`models/hydrogen.py`](models/hydrogen.py), [`models/decision.py`](models/decision.py) | Official `google-genai` v3.6+ clients using `client.interactions.create` (`gemini-4-carbon`, `gemini-4-hydrogen`) and `client.decisions.create` (`gemini-decision-1` closed-set `noul` & `choice` questions). |
| [`commerce/shop_mcp.py`](commerce/shop_mcp.py), [`commerce/cache.py`](commerce/cache.py) | JSON-RPC 2.0 client for `https://mcp.shop.com` (`search_catalog`, `get_product`, `update_cart`) + 10-minute price TTL cache. |
| [`ledger/verifiers.py`](ledger/verifiers.py), [`ledger/board.py`](ledger/board.py), [`ledger/resolve.py`](ledger/resolve.py), [`ledger/voi.py`](ledger/voi.py), [`ledger/loop.py`](ledger/loop.py), [`ledger/export.py`](ledger/export.py) | 9 pure-Python verifiers, Invariants I1–I7 (`apply_patch` + `rebuild_board` zero-drift check), cross-angle dedup + contradiction detector, TV counterfactual VoI ranker, orchestrator, and CSV/printable HTML Schedule of Loss exporters. |
| [`app/server.py`](app/server.py), [`app/static/`](app/static/index.html) | Verdict-first interactive FastAPI + WebRTC camera/microphone web app (capture -> verdict -> active VoI ask -> 4s refilm -> badge flips `CERTIFIED` -> Shop replacement cart). |
| [`eval/`](eval/run_eval.py), [`tests/`](tests/) | Baselines `B0` & `B1`, Wilson CIs, `ffmpeg` adversarial transforms, pre-run labeler CLI, dev video synthesizer, and full `pytest` verification suite. |
