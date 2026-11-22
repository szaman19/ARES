# Home Inventory Ledger: Project Plan & Agent Spec

> **One-liner:** Walk through your home for three minutes and get an inventory your insurer will accept. Every price is backed by a real listing, and you find out today whether you're underinsured, not after the fire.

**Event:** Gemini 4 NYC Hackathon (2026-11-21). Submissions due 5:00 PM ET.
**Tracks:** Problem Statement Two, *Element 119* (video and audio are the core input) and Problem Statement One, *5 Sigma* (held-out evaluation against a baseline), plus the **Shopify Commerce bonus** (shop.dev catalog for pricing, Shop Pay for replacement checkout).
**Status:** execution-ready spec for a ~4-hour build by 1–4 people, with or without coding agents.

---

## Table of Contents

1. [Problem](#1-problem)
2. [Users, Buyers & Business Model](#2-users-buyers--business-model)
3. [Why This Needs Gemini 4 Multimodal](#3-why-this-needs-gemini-4-multimodal)
4. [Product Experience](#4-product-experience)
5. [Core Idea: Certified vs. Uncertified Evidence](#5-core-idea-certified-vs-uncertified-evidence)
6. [System Architecture](#6-system-architecture)
7. [Model Roles](#7-model-roles)
8. [Data Contracts](#8-data-contracts)
9. [Pipeline, Step by Step](#9-pipeline-step-by-step)
10. [Deterministic Verifiers](#10-deterministic-verifiers)
11. [Invariants](#11-invariants)
12. [Coverage Decision, Counterfactuals & Value of Information](#12-coverage-decision-counterfactuals--value-of-information)
13. [Shopify / shop.dev Integration](#13-shopify--shopdev-integration)
14. [Evaluation Plan (5 Sigma)](#14-evaluation-plan-5-sigma)
15. [Stress Testing: Where It Breaks (Element 119)](#15-stress-testing-where-it-breaks-element-119)
16. [Live Demo Script](#16-live-demo-script)
17. [Repository Layout](#17-repository-layout)
18. [Build Schedule](#18-build-schedule)
19. [Coding-Agent Kickoff Prompts](#19-coding-agent-kickoff-prompts)
20. [Cut List](#20-cut-list)
21. [Risks & Mitigations](#21-risks--mitigations)
22. [Privacy, Safety & Ethics](#22-privacy-safety--ethics)
23. [Hackathon Compliance Checklist](#23-hackathon-compliance-checklist)
24. [Pitch Narrative & Judge Q&A](#24-pitch-narrative--judge-qa)
25. [Roadmap Beyond the Hackathon](#25-roadmap-beyond-the-hackathon)
26. [Open Questions](#26-open-questions)

---

## 1. Problem

### 1.1 The moment it matters
After a total loss from fire, flood or burglary, an insurer typically asks the policyholder for a **schedule of loss**: an itemized list of everything they owned, with description, brand/model, age, quantity and replacement cost. Survivors have to rebuild this list **from memory**, often while displaced. It can take months, and it routinely leaves money unclaimed because people forget what they owned or can't prove it.

The 2025 Los Angeles wildfires made this very public. Thousands of households faced itemization demands for homes that no longer existed, and California regulators pushed insurers to relax itemization requirements. *(Verify the specific regulatory actions before citing them on stage.)*

### 1.2 The problem nobody notices: underinsurance
Contents coverage ("Coverage C" on a homeowners policy, or personal property on a renters policy) is usually set by default as a percentage of the dwelling limit. Almost nobody checks whether it covers what they actually own. Industry and consumer-advocate sources commonly estimate that **about two in three homes are underinsured**. *(Verify the source and figure, e.g. CoreLogic or United Policyholders, before the pitch.)*

### 1.3 Why existing solutions fail
| Today's option | Why it fails |
|---|---|
| Spreadsheet or paper list | Tedious, so nobody does it, and it goes stale |
| Inventory apps (photo per item, manual typing) | Too much friction; abandoned after a few items |
| "Just film your house" advice | The video exists, but it isn't itemized, priced or accepted as a schedule of loss |
| Single-call LLM on a video | Double-counts, misses items, **invents prices** and brands, gives no provenance, and is confidently wrong |
| Public adjuster after the loss | Takes 10–15% of the claim and still works from memory |

### 1.4 What we deliver
A **structured, cited and verified inventory** built from one casual walkthrough video plus narration, with:
- one row per *physical* item, de-duplicated across angles and rooms,
- a timestamped frame as visual proof for each item,
- a replacement price that is **CERTIFIED** (traced to a live catalog listing with an exact model match) or explicitly **UNCERTIFIED** (an estimate with a range),
- a **coverage verdict** against the user's actual policy limit,
- a ranked list of **"show me X" requests** that most reduce uncertainty in that verdict.

---

## 2. Users, Buyers & Business Model

### 2.1 Personas
| Persona | Job to be done | Trigger |
|---|---|---|
| **Homeowner / renter** | "Am I covered? Can I prove what I own?" | Policy renewal, a move, news of a disaster |
| **Disaster survivor** | "Rebuild my list from old videos." | After a loss: run the pipeline on existing phone videos, social posts, real-estate tours |
| **Insurance agent / insurtech** | Right-size Coverage C, reduce claim friction and fraud | Onboarding, renewal |
| **Claims adjuster** | Faster, verifiable contents claims | First notice of loss |
| **Public adjuster** | Build schedules of loss faster | Each client engagement |
| **Movers, estate executors, divorce attorneys** | Valued inventories | Move, probate, asset division |

### 2.2 Revenue model
1. **B2B2C with insurers and insurtechs (main revenue):** per-inventory or per-policy fee. Insurers benefit from a *pre-loss* inventory: faster claims, less fraud, and correctly priced Coverage C (an upsell when someone is underinsured).
2. **Consumer freemium:** free verdict ("you're likely underinsured by $X"); paid export, annual refresh and vault storage.
3. **Shopify commerce loop:** after a claim pays out, the ledger turns into a **replacement cart**. Purchases through shop.dev earn commission, and the user rebuys exactly what they had in a few taps.
4. **Professional tier:** public adjusters and estate professionals pay per project.

### 2.3 Why now
- Climate-driven catastrophe losses and insurer pullbacks in CA, FL and other states make coverage adequacy a mainstream worry.
- Video and audio-native models (Gemini 4) can finally itemize a messy walkthrough in a single pass.
- Agent-accessible commerce catalogs (shop.dev / Shop MCP) give **real, citable prices** instead of model guesses.

---

## 3. Why This Needs Gemini 4 Multimodal

The judges' Element 119 question is: *"What does your project get from the raw video, audio or images that a text description would lose?"*

| Signal | Only available from raw media |
|---|---|
| **What exists** | Items nobody would think to type (the second monitor, the stand mixer behind the toaster) |
| **How many** | Counting identical items (4 dining chairs) and dedup across camera angles |
| **Exact identity** | Model numbers on labels, logos, distinctive design features |
| **Condition / age cues** | Wear, generation-specific design (e.g. TV bezel style), which affects depreciation |
| **Narration** | Owner knowledge ("bought in 2023", "that's real walnut") aligned to the frame it describes |
| **Contradiction** | Narration says "Dyson V15" but the frame shows a V8, which is flagged |
| **Coverage gaps** | "You never showed the closet": spatial completeness only video can reveal |
| **Policy document** | Coverage C limit read from a photographed declarations page |

A text description of the home loses counts, model identity, contradictions and completeness, which are exactly the things that decide a claim.

---

## 4. Product Experience

### 4.1 Happy path
1. **Policy:** the user uploads or photographs their **declarations page**. The system extracts the Coverage C / personal property limit and the deductible. If none is provided, it asks the user to enter a limit.
2. **Capture:** the user records (or uploads) a walkthrough video and narrates freely.
3. **Processing (live progress):** rooms detected → items streaming into the ledger → prices resolving → certification badges flipping.
4. **Ask:** the system shows the top 1–3 capture requests, each with its dollar impact:
   - "Show the label on the back of the living-room TV (±$1,400 uncertainty)."
   - "Open the bedroom closet: no items seen there yet."
   - "Is the sofa leather or fabric? (±$900)" (answerable by voice)
5. **Re-capture:** the user films the short follow-up clip. Only that clip is processed, and the ledger updates incrementally.
6. **Verdict:**
   > Certified contents: **$48,200** · Estimated total: **$61k–$74k** · Your Coverage C: **$50,000**
   > **Likely underinsured by $11k–$24k** (confidence 0.87). The verdict hinges on: TV, sofa, laptop.
7. **Export:** a schedule of loss as CSV and PDF. Each row has a thumbnail, timestamp, status, price, listing URL and `retrieved_at`.
8. **(Post-loss) Replace:** "Rebuild my home" turns the ledger into a Shop cart.

### 4.2 Screens (keep the UI minimal: the agent is the product, not a dashboard)
- **Capture:** record/upload, plus a policy photo slot.
- **Ledger:** item rows (thumbnail · name · qty · room · status badge · price/range · source link), with the ask panel pinned at the top.
- **Verdict card:** coverage vs. totals, the items it hinges on, and the export button.

---

## 5. Core Idea: Certified vs. Uncertified Evidence

This is the pattern carried over from ARES, applied to home inventory. We **reimplement it from scratch**; see the compliance section.

| Status | Meaning | Example |
|---|---|---|
| **CERTIFIED** | A deterministic check ties the claim to hard evidence | Model number read from a label frame **and** an exact match in a live catalog listing → price = that listing's price |
| **UNCERTIFIED** | A model inference, not deterministically confirmed | "Looks like a 65-inch Samsung QLED" → price range from the spread of similar listings |
| **UNKNOWN** | No usable evidence yet | An item seen too briefly to identify; a room never filmed |

Rules:
- An LLM reading a label is **not** certification on its own. Certification needs a **deterministic** cross-check (catalog exact-match plus format validation).
- Narration is evidence, but it is UNCERTIFIED unless corroborated visually or by a label.
- Totals are reported **split**: certified total, uncertified range, and unknown count. They are never blended into one number without a label.

---

## 6. System Architecture

```
                ┌──────────────────────────────────────────────────────────┐
  video.mp4 ──► │ INGEST (deterministic)                                   │
  dec_page.jpg  │  ffmpeg: keyframes @1 fps + scene-change frames          │
                │  audio track → WAV                                        │
                │  perceptual hash per frame; blur/darkness score          │
                └───────────────┬──────────────────────────────────────────┘
                                │
                ┌───────────────▼──────────────────────────────────────────┐
                │ UNDERSTAND (Carbon)                                       │
                │  • full-video pass → room segments [start,end,room_label] │
                │  • timestamped narration transcript + spoken claims       │
                │  • declarations page → Coverage C limit, deductible       │
                └───────────────┬──────────────────────────────────────────┘
                                │  one task per room segment
         ┌──────────────────────┼──────────────────────┐
   ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
   │ Hydrogen  │          │ Hydrogen  │          │ Hydrogen  │   PARALLEL WORKERS
   │ segment 1 │          │ segment 2 │          │ segment N │   → ItemObservation[]
   └─────┬─────┘          └─────┬─────┘          └─────┬─────┘     + catalog queries
         └──────────────────────┼──────────────────────┘
                                │
                ┌───────────────▼──────────────────────────────────────────┐
                │ RESOLVE                                                   │
                │  dedup/track (deterministic gates + Carbon merge proposals)│
                │  narration ↔ frame alignment, contradiction flags          │
                └───────────────┬──────────────────────────────────────────┘
                                │
                ┌───────────────▼──────────────────────────────────────────┐
                │ PRICE                                                      │
                │  Shop MCP search_catalog / get_product → top-k listings    │
                │  Decision-1: "is listing k the same item?" (closed set)    │
                │  VERIFIERS: model-number exact match, price sanity, etc.   │
                │  → CERTIFIED price | UNCERTIFIED range | UNKNOWN           │
                └───────────────┬──────────────────────────────────────────┘
                                │
                ┌───────────────▼──────────────────────────────────────────┐
                │ LEDGER BOARD (versioned, cited, invariant-checked)         │
                └───────────────┬──────────────────────────────────────────┘
                                │
                ┌───────────────▼──────────────────────────────────────────┐
                │ DECIDE + PROBE (Decision-1)                                │
                │  "Is Coverage C ≥ replacement total?" + confidence         │
                │  counterfactual: item → low/high/unknown → verdict swing   │
                │  VoI ranking → capture requests                            │
                └───────────────┬──────────────────────────────────────────┘
                                │
                     ask user ◄─┴─► export CSV/PDF · Shop cart
```

### 6.1 Loop
```
seed (policy + video) → understand → extract (parallel) → resolve → price/verify
   → board → decide → probe → ask → [follow-up clip] → extract(clip) → … → stop
```
**Stop conditions:**
- **Confident verdict:** verdict confidence ≥ 0.85 **and** max item VoI < $250 (or < 2% of the limit).
- **Budget:** max 3 ask rounds or the wall-clock cap. Output the best verdict with explicit UNCERTIFIED blockers.

---

## 7. Model Roles

The hackathon requires **at least two Gemini 4 models, each with a clear job**. Confirm the exact model IDs and their strengths (long video, live audio, latency) in the *Build with Gemini 4 Builder Guide* at kickoff.

| Role | Model | Job | Why this model |
|---|---|---|---|
| **Understander / fuser** | `gemini-4-carbon` | Full-video pass, room segmentation, narration transcript, declarations-page extraction, dedup merge proposals, final ledger patching | Strongest long-context multimodal reasoning |
| **Parallel extractors** | `gemini-4-hydrogen` | Per-segment item extraction from keyframes and the clip, label reading, catalog query writing | Fast and cheap, so N segments run concurrently |
| **Decision probe** | `gemini-decision-1` (Decisions API) | Closed-set judgments with calibrated 0–1 confidence: listing-match acceptance, the coverage verdict, counterfactual variants (one board variant per call) | Calibrated confidence makes the counterfactual and VoI meaningful |
| *(optional)* **Capture coach** | Gemini live voice model (see guide) | Real-time spoken guidance: "closer on the label", "pan left" | Makes the ask loop hands-free |
| **Baseline** | `gemini-4-carbon`, single call | "List every item in this video with a replacement price" | Honest comparison |

**Decisions API constraints** (from the ARES reference design; re-verify in the docs):
- Closed-set questions (yes/no, multiple choice, bounded score) with 0–1 confidence.
- Up to ~1,000 questions against one shared input, ~256K-token input.
- **One counterfactual board state per call.** Don't pack variants into one prompt.
- Ablated fields are set to the explicit string `"unknown"`, not removed.

---

## 8. Data Contracts

`contracts.py` (Pydantic v2). Freeze these in the first 30 minutes; everything else builds on them.

```python
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

Cert = Literal["CERTIFIED", "UNCERTIFIED", "UNKNOWN"]
EvidenceKind = Literal["visual", "spoken", "label_read", "catalog_match", "user_answer"]

class Frame(BaseModel):
    model_config = ConfigDict(frozen=True)
    frame_id: str            # f"{video_id}@{ts_ms}"
    video_id: str
    ts_ms: int
    path: str                # local JPEG
    phash: str               # perceptual hash (hex)
    blur_score: float        # variance of Laplacian; low = blurry
    brightness: float

class RoomSegment(BaseModel):
    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    room_label: str          # "kitchen", "living_room", ...

class SpokenClaim(BaseModel):
    claim_id: str
    ts_ms: int
    text: str                # "that's a Dyson V15, bought 2023"
    brand: str | None = None
    model: str | None = None
    purchase_year: int | None = None
    material: str | None = None

class ItemObservation(BaseModel):
    model_config = ConfigDict(frozen=True)
    obs_id: str
    segment_id: str
    frame_id: str
    bbox: tuple[float, float, float, float] | None = None   # normalized x,y,w,h
    category: str            # "television", "vacuum", "sofa", ...
    description: str         # "65in flat TV, thin bezel, wall-mounted"
    brand: str | None = None
    model_text: str | None = None      # raw text read from a label, if any
    quantity: int = 1
    condition: Literal["new", "good", "worn", "damaged", "unknown"] = "unknown"
    evidence: list[EvidenceKind]
    produced_by: str         # "gemini-4-hydrogen"

class Listing(BaseModel):
    model_config = ConfigDict(frozen=True)
    listing_id: str
    title: str
    url: str
    price: float
    currency: str = "USD"
    merchant: str | None = None
    specs: dict = Field(default_factory=dict)
    image_url: str | None = None
    retrieved_at: str        # ISO-8601; price cache TTL ≤ 10 min

class Note(BaseModel):
    """Immutable, append-only evidence record. The source of truth."""
    model_config = ConfigDict(frozen=True)
    note_id: str
    item_id: str
    certification: Literal["CERTIFIED", "UNCERTIFIED"]
    kind: Literal["identity", "price", "quantity", "contradiction", "coverage"]
    verifier: str | None     # name of the deterministic verifier, if any
    frame_ids: list[str] = []
    listing_ids: list[str] = []
    claim_ids: list[str] = []
    value: dict              # e.g. {"model": "QN65Q80C"} or {"price": 1799.99}
    summary: str
    requested_under: int     # board_version at dispatch

class LedgerItem(BaseModel):
    item_id: str
    room: str
    category: str
    display_name: str
    quantity: int = 1
    identity_status: Cert = "UNKNOWN"
    price_status: Cert = "UNKNOWN"
    unit_price: float | None = None          # set only if CERTIFIED
    price_low: float | None = None           # set for UNCERTIFIED
    price_high: float | None = None
    best_frame_id: str | None = None
    sources: list[str] = []                  # note_ids
    flags: list[str] = []                    # "narration_conflict", "needs_appraisal", ...
    influence: float = 0.0                   # counterfactual verdict influence
    voi_usd: float = 0.0                     # expected verdict-relevant uncertainty

class CoverageDecision(BaseModel):
    coverage_c_limit: float | None
    deductible: float | None
    certified_total: float
    estimate_low: float
    estimate_high: float
    verdict: Literal["adequate", "likely_underinsured", "underinsured", "unknown"]
    confidence: float
    hinges_on: list[str]                     # item_ids
    certified_share: float                   # share of point estimate that is CERTIFIED

class CaptureRequest(BaseModel):
    request_id: str
    item_id: str | None
    room: str | None
    instruction: str                         # "Show the label on the back of the TV"
    voi_usd: float
    answer_mode: Literal["video", "voice", "text"]

class LedgerBoard(BaseModel):
    board_version: int = 0
    items: dict[str, LedgerItem] = {}
    rooms_seen: list[str] = []
    rooms_missing: list[str] = []
    decision: CoverageDecision | None = None
    requests: list[CaptureRequest] = []
```

---

## 9. Pipeline, Step by Step

### 9.1 Ingest (deterministic, no LLM)
- `ffmpeg -i in.mp4 -vf "fps=1" frames/%06d.jpg` plus scene-change frames (`select='gt(scene,0.3)'`).
- Drop near-duplicate frames by phash Hamming distance ≤ 6 (cuts frames to send to the model by ~50–70%).
- Compute a blur score (variance of Laplacian) and brightness. Flag unusable frames instead of sending them.
- Extract audio: `ffmpeg -i in.mp4 -vn -ac 1 -ar 16000 audio.wav`.
- Long videos: cap at N keyframes per segment (e.g. 24) and chunk the timeline.

### 9.2 Understand (Carbon)
- Input: the whole video (or the keyframe sequence plus audio) and the declarations page image.
- Output (structured JSON): `RoomSegment[]`, `SpokenClaim[]`, `{coverage_c_limit, deductible}`.
- Validation: segments must be in time order and non-overlapping, and the limit must be a positive number. Retry once on schema failure.

### 9.3 Extract (Hydrogen, parallel)
- One task per `RoomSegment`, with `max_concurrent = 4–6`.
- Input: the segment's keyframes (≤ 24), the spoken claims in that time window, and a category taxonomy.
- Output: `ItemObservation[]` with a frame reference, bbox, category, brand/model text if visible, quantity, and evidence kinds.
- Prompt rules: **never invent model numbers**; `model_text` only if characters are visibly legible; prefer "unknown" to guessing.
- Also outputs 1–3 **catalog queries** per observation (e.g. `"Samsung 65 inch QLED Q80C"`).

### 9.4 Resolve
**Dedup / tracking:**
1. Candidate pairs: same category **and** (same segment, or adjacent segments with |Δt| < 20 s, or the same room revisited).
2. Deterministic gates: crop phash distance, bbox position continuity within a segment, and conflicting brand/model → never merge.
3. Carbon reviews the remaining ambiguous pairs with both crops side by side and answers merge or keep.
4. Quantities: identical items seen together in the same frame (4 chairs) are kept as one `LedgerItem` with `quantity = 4`.

**Narration alignment:**
- Attach each `SpokenClaim` to the observation nearest in time with a compatible category (window ±5 s).
- If the claim's brand/model contradicts a legible label → flag `narration_conflict` and keep the label value.

### 9.5 Price & Verify
For each `LedgerItem`:
1. Run its queries against Shop MCP `search_catalog` → top-k (k = 5–10) listings; `get_product` for specs where needed.
2. **Decision-1 match question** (one shared input, many closed-set questions): "Is listing *i* the same product as the item in this crop? (yes/no)", with confidence.
3. **Verifiers** (§10) decide the status:
   - Exact model-number match on a matched listing → **CERTIFIED**, `unit_price = listing.price`.
   - Category/brand match only → **UNCERTIFIED**, `price_low/high` = 25th/75th percentile of matched listings.
   - No usable listings → **UNKNOWN** (+ `needs_appraisal` for antiques/art/jewelry categories).
4. Write an immutable `Note` for every decision. The board only ever cites notes.

### 9.6 Decide, Probe, Ask
See §12.

### 9.7 Follow-up clips
- A follow-up clip is tagged with the `CaptureRequest.request_id` that prompted it.
- Only that clip is ingested and extracted. Resulting observations are linked to the target `item_id` first, falling back to a general dedup.
- The board version increments; stale arrivals (`requested_under < board_version - 1`) are logged.

---

## 10. Deterministic Verifiers

All verifiers are pure Python. **No LLM calls.** Each returns `(passed: bool, details: dict)`.

| Verifier | Checks | Certifies |
|---|---|---|
| `model_number_format` | Read string matches known brand patterns (regex table for ~20 top brands: Samsung, LG, Sony, Apple, Dyson, KitchenAid, …); normalized (strip spaces/dashes, uppercase) | Prerequisite for identity |
| `catalog_exact_match` | Normalized model string appears in listing title or specs (`mpn`, `model`, `sku`) | **Identity + price → CERTIFIED** |
| `price_sanity` | Listing price within [0.3×, 3×] the category median from the same search; not a part or accessory ("remote for", "replacement filter", "case") | Blocks accessory mismatches |
| `listing_freshness` | `now - retrieved_at ≤ 10 min` at export time, otherwise re-fetch | Compliance with shop.dev caching rules |
| `quantity_consistency` | Quantity ≤ max simultaneous count seen in any single frame for that item track | Blocks double-counting |
| `room_coverage` | Expected rooms (from the user or typical layout) vs. rooms seen; closets and cabinets opened? | Produces `rooms_missing` |
| `frame_quality` | Blur/brightness thresholds on the best frame for each item | Downgrades identity from blurry frames |
| `policy_limit_parse` | Extracted limit is numeric and plausible ($5k–$2M); deductible < limit | Coverage inputs |
| `total_arithmetic` | Totals recomputed from item rows; must match the board | Report integrity |

---

## 11. Invariants

Enforced in `ledger/board.py::apply_patch(board, patch, notes)`. Any violation rejects the patch and is logged.

1. **I1 Provenance:** every non-UNKNOWN item field cites ≥ 1 existing `note_id`; every CERTIFIED price cites a `Listing` and a passing `catalog_exact_match` note.
2. **I2 Weakest link:** an item's status = the weakest certification among its cited notes. A merge takes the union of sources and the weakest status. A model can never upgrade certification.
3. **I3 Notes are the truth:** notes are immutable and append-only. `rebuild_board(notes)` reproduces the board deterministically, and drift between patched and rebuilt boards is logged (it must be 0).
4. **I4 Context vs. verification:** board state may go into model prompts as context, but **never** into verifier inputs.
5. **I5 Staleness:** every task and note records `requested_under`.
6. **I6 No invented prices:** `unit_price` is set **only** when `price_status == CERTIFIED`. Uncertified items carry ranges only.
7. **I7 Bounded probe board:** the probe sees at most ~40 highest-value items. Low-value items (< $50 and influence < 0.01) are summarized as a single "misc" line.

---

## 12. Coverage Decision, Counterfactuals & Value of Information

### 12.1 Coverage verdict
Point estimate `E = certified_total + Σ midpoint(uncertified ranges)`. Range `[L, H] = [certified_total + Σ low, certified_total + Σ high]`.

Decision-1 receives a compact board (items, statuses, prices/ranges, limit, deductible) and answers:
- **Q1** `verdict ∈ {adequate, likely_underinsured, underinsured}`
- **Q2** "Is the inventory complete enough to judge?" yes/no (catches missing rooms)
- **Q3** for each top item: "Is this item's identity sufficiently verified?" yes/no

Deterministic rule as a fallback and sanity check: `H ≤ limit → adequate`; `L > limit → underinsured`; otherwise → likely_underinsured / ambiguous. If Decision-1 disagrees with the deterministic rule, show both and flag it. That disagreement is good demo material.

### 12.2 Counterfactual influence
For each item *i* among the top-M by uncertainty width (M ≈ 8–12), run **separate** Decision-1 calls:
- `ablate`: item *i* → `"unknown"`
- `low`: item *i* → `price_low`
- `high`: item *i* → `price_high`

`influence_i = TV(P_base(verdict), P_variant(verdict))` maximized over variants (TV = total variation distance). Normalize across items. `hinges_on` = the top 3 by influence.

Report `certified_share` = the share of total influence carried by CERTIFIED items. A verdict that rests mostly on uncertified items is labeled as such.

### 12.3 Value of information → capture requests
For each uncertain item: `voi_usd_i = (price_high - price_low) × quantity × influence_i`, plus a fixed bonus for missing rooms (e.g. a closet the user never opened).

Map each top-VoI item to a concrete instruction by evidence gap:
| Gap | Request |
|---|---|
| Brand/model unknown, electronics/appliance | "Show the label on the back/bottom of the {item}." |
| Material ambiguity (furniture) | Voice: "Is the {item} leather or fabric?" |
| Quantity ambiguity | "Pan slowly across the {room} {area} so I can count the {items}." |
| Room never seen | "Open the {closet/cabinet} in the {room}." |
| Blurry best frame | "Hold steady on the {item} for 2 seconds." |

Return the top 1–3 requests, each with its `voi_usd`.

---

## 13. Shopify / shop.dev Integration

Based on shop.dev documentation reviewed 2026-11-21. **Re-verify tool names and parameters in the quickstart before coding.**

| Capability | Endpoint / tool | Use |
|---|---|---|
| Anonymous catalog search | Shop MCP `https://mcp.shop.com/` (JSON-RPC MCP): `search_catalog`, `get_product`, `lookup_catalog` | Find candidate listings and prices; no API key needed |
| Higher-rate catalog | Global Catalog API `https://catalog.shopify.com/api/ucp/mcp` | Optional API key from the Shopify Dev Dashboard → bearer token (60 min). Requires a published agent profile at a public HTTPS URL |
| Cart / checkout | Shop MCP cart tools; `shop` CLI (`npm i -g @shopify/shop`) | "Rebuild my home" replacement cart |
| Agent payments | Shop Pay spend requests (single-use credentials; the card isn't exposed to the agent) | Checkout without handling card data |
| Identity | Sign in with Shop (OAuth) | Optional personalization and order tracking |

**Caching rules to respect:** product metadata ≤ 24 h; **price and availability ≤ 10 min**; never cache commissions. Every `Listing` carries `retrieved_at`, and the export re-fetches stale prices.

**Search strategy:**
1. Exact query: `"{brand} {model_text}"`.
2. Descriptive query: `"{brand} {size} {category} {key feature}"`.
3. Generic query: `"{category} {material} {size}"` (gives the uncertified range).
Filters: ship-to US, in stock, new condition where supported.

**Catalog coverage gaps are expected** (big-box-only appliance brands, furniture from non-Shopify retailers). These become UNCERTIFIED or UNKNOWN with a clear reason. **Never silently guess.**

**Commerce loop (demo beat):** select items → `cart create` with the same or equivalent listings → show the Shop Pay checkout link. Don't complete a real purchase on stage unless you mean to.

---

## 14. Evaluation Plan (5 Sigma)

### 14.1 Questions the judges will ask
- *What's your baseline, and how much better are you?*
- *Is your test set realistic and unseen?*
- *Where do you still fail, and what did you change after seeing failures?*

### 14.2 Baseline
**B0, single call:** `gemini-4-carbon` with the full video: *"List every distinct item in this home video with brand/model if visible, quantity, and an estimated replacement price in USD. Return JSON."* Same frames/video, same time budget.

**B1, single call + catalog (optional, a stronger baseline):** B0's items, each searched once on Shop MCP with the top-1 result's price taken. This isolates the value of verification and matching from the value of search access alone.

### 14.3 Test sets (none written by the team to pass)
| Set | Source | Size | Ground truth | Notes |
|---|---|---|---|---|
| **T1 External** | **EPIC-KITCHENS-100** (real head-mounted kitchen video) + VISOR annotations | 8–10 video segments, 2–5 min each | Object nouns from narrations / VISOR masks → item presence and counts | Licensed **non-commercial only**. Download individual videos, not the full ~700 GB set |
| **T2 Fresh captures** | Teammates and volunteers film real rooms with their phones | 4–6 videos | Labeled **before** any system run by someone who doesn't tune prompts: item list, counts, and true model numbers where labels are visible | Messy cases: low light, motion, reflections, clutter, items in cabinets |
| **T3 Adversarial (scripted, not hand-picked)** | Automatic transforms of T2: frame drops, blur, darkening, audio noise, and **planted contradictory narration** (TTS clip "this is a Dyson V15" spliced over a V8 frame) | 1 variant per transform per video | Inherited from T2 + known plant | Tests robustness and contradiction catching |
| **T4 Live** | The judge's table at demo time | 1 | Judge confirms | Unprepared input |

Optional external sets if time allows: Charades (crowd-filmed real home videos, non-commercial), ARKitScenes (real home scans with object boxes, check the license).

### 14.4 Matching predictions to ground truth
- Item matching: category normalized to a shared taxonomy (~60 classes) and Hungarian matching on (category, time overlap). Report precision / recall / F1.
- Count accuracy: |pred qty − true qty| per matched item.
- Identity accuracy (T2 only, items with visible labels): exact normalized model-number match.

### 14.5 Metrics
| # | Metric | Definition | Expectation |
|---|---|---|---|
| M1 | Item F1 | vs. T1/T2 ground truth | System ≥ baseline (parallel segment extraction gives better recall) |
| M2 | Duplicate rate | Predicted items mapping to an already-matched GT item | Baseline higher (double-counts across angles) |
| M3 | **Hallucinated-price rate** | Share of priced items whose price can't be traced to any real listing within ±15% | Baseline high; system 0% for CERTIFIED by construction (I6) |
| M4 | Hallucinated model-number rate | Model numbers output that don't match a legible label | Baseline > system |
| M5 | Identity accuracy | Exact model match on labeled items | System > baseline |
| M6 | Certified coverage | CERTIFIED $ / total point-estimate $ | Report and improve after asks |
| M7 | Ask efficiency | Change in certified coverage per capture request | Shows the VoI loop works |
| M8 | Contradiction catch rate | Planted narration conflicts flagged (T3) | System ≫ baseline |
| M9 | Verdict calibration | Decision-1 confidence vs. correctness on synthetic limit sweeps | Reliability plot |
| M10 | Influence validity | Rank correlation between counterfactual influence and true $ uncertainty contribution | Positive |
| M11 | Cost / latency | Tokens, model calls, seconds per minute of video | Report honestly |

### 14.6 Verdict ground truth (synthetic sweep)
For each T2 home, compute the true replacement total from labeled items and manually looked-up prices. Sweep the policy limit across `{0.5, 0.8, 1.0, 1.2, 1.5} × true_total` → correct verdicts are known. Measure verdict accuracy and calibration for the system vs. a baseline (B0 total vs. limit).

### 14.7 Failure log (required deliverable)
Keep `eval/FAILURES.md`: each failure with input, expected, got, root cause, the change made, and the before/after metric. **At least three entries by submission.** Judges explicitly ask for this.

### 14.8 Results table template
| System | Item F1 | Dup rate | Halluc. price | Halluc. model # | Identity acc. | Contradiction catch | Certified $ share | s / video-min |
|---|---|---|---|---|---|---|---|---|
| B0 single call | | | | | | | n/a | |
| B1 single call + search | | | | | | | | |
| **Ledger (no asks)** | | | | | | | | |
| **Ledger (+1 ask round)** | | | | | | | | |

---

## 15. Stress Testing: Where It Breaks (Element 119)

Push until it fails, then show what changed.

| Axis | Ladder | Expected break | Mitigation to try |
|---|---|---|---|
| Video length | 1 → 3 → 10 → 20 min | Context/latency, segment drift | Segment-level chunking, keyframe dedup |
| Clutter | Tidy desk → messy garage | Recall drops, wrong merges | Smaller segments, more keyframes per segment |
| Lighting | Daylight → lamp → near-dark | Label reading fails | `frame_quality` gate + "turn on the light" ask |
| Motion | Steady → walking → whip-pans | Blur, missed items | Scene-change frames, blur filter |
| Audio | Clean → TV in background → two people talking | Narration misattributed | Time-window alignment ±5 s, require category compatibility |
| Screens/mirrors | TV showing products, mirror reflections | Phantom items | Prompt rule + duplicate-in-mirror check (bbox symmetry heuristic) |
| Catalog coverage | Common electronics → niche furniture → antiques | No listings | Honest UNKNOWN + `needs_appraisal` |
| Concurrency | 2 → 6 → 12 parallel workers | Rate limits | Backoff, cap `max_concurrent` |

Record the break point for each axis in `eval/STRESS.md` with a screenshot.

---

## 16. Live Demo Script

**Total: 3:00.** Rehearse at least three times on fresh tables.

| Time | Beat | On screen |
|---|---|---|
| 0:00–0:20 | **Hook:** "After the LA fires, survivors had to list everything they owned from memory. Two in three homes are underinsured, and nobody knows until it's too late." | Title slide |
| 0:20–0:35 | Photograph a sample declarations page → limit extracted | Coverage C = $X |
| 0:35–1:05 | **Judge's table:** film it live for ~20 s while narrating ("that's a MacBook, maybe 2022") | Video capture |
| 1:05–1:30 | Items stream in, mostly UNCERTIFIED with ranges; a planted or real narration conflict is flagged | Ledger with badges |
| 1:30–1:55 | **Ask:** "Show the bottom label of the laptop (±$900)." Film it → flips to **CERTIFIED** with the listing link | Badge flip, $ range collapses |
| 1:55–2:15 | **Verdict + counterfactual:** "Verdict hinges on 3 items. Certified share 74%." | Verdict card |
| 2:15–2:40 | **Proof:** results table vs. baseline, plus one honest failure | Eval slide |
| 2:40–3:00 | **Business:** insurer B2B2C plus the Shopify "rebuild my home" cart | Cart → Shop Pay link |

**Backup plan:** a pre-recorded walkthrough video plus cached model responses (`--replay` flag), used only if the venue Wi-Fi fails. Say so if you use it.

---

## 17. Repository Layout

```
home-inventory-ledger/
├── README.md                  # what it is, how to run, WHAT WE BUILT TODAY
├── PLAN.md                    # this file
├── pyproject.toml             # uv; google-genai, pydantic, duckdb(optional), opencv-python, imagehash, httpx, fastapi
├── .env.example               # GEMINI_API_KEY=..., SHOP_CATALOG_KEY= (optional)
├── contracts.py               # §8
├── ingest/
│   ├── frames.py              # ffmpeg keyframes, phash, blur, brightness
│   └── audio.py               # audio extraction
├── models/
│   ├── carbon.py              # understand(), merge_review(), patch()
│   ├── hydrogen.py            # extract_segment(), queries
│   └── decision.py            # match(), verdict(), probe() (one variant per call)
├── commerce/
│   ├── shop_mcp.py            # JSON-RPC client: search_catalog, get_product, cart
│   └── cache.py               # TTL cache honoring 10-min price rule
├── ledger/
│   ├── resolve.py             # dedup/tracking, narration alignment
│   ├── verifiers.py           # §10
│   ├── board.py               # apply_patch() with invariants I1–I7, rebuild_board()
│   ├── voi.py                 # influence, VoI, capture requests
│   └── loop.py                # orchestrator
├── app/
│   ├── server.py              # FastAPI: /upload, /policy, /followup, /export, SSE progress
│   └── static/index.html      # minimal capture + ledger + verdict UI (no Streamlit)
├── eval/
│   ├── datasets/              # download scripts only; no copyrighted media committed
│   ├── labels/                # T2 ground-truth JSON (labeled before runs)
│   ├── transforms.py          # T3 adversarial generator
│   ├── baseline.py            # B0, B1
│   ├── run_eval.py            # metrics M1–M11 → results/*.json
│   ├── FAILURES.md
│   └── STRESS.md
└── tests/
    ├── test_verifiers.py
    ├── test_board_invariants.py
    └── test_voi.py
```

Note: the hackathon bans **Streamlit apps** and **dashboard-as-main-feature**. Use a minimal FastAPI + static HTML UI centered on the capture → ask → verdict loop.

---

## 18. Build Schedule

Assumes a start at ~1:00 PM and **submission at 5:00 PM**. Adjust for team size.

| Time | Track A: Perception | Track B: Commerce & Verification | Track C: Eval & Demo |
|---|---|---|---|
| 1:00–1:20 | Freeze `contracts.py`; repo skeleton; confirm model IDs and API keys | Smoke-test Shop MCP `search_catalog` | Start downloading the EPIC subset; record 4 T2 videos; labeler starts labeling |
| 1:20–2:15 | `ingest/frames.py`; Carbon `understand()`; Hydrogen `extract_segment()` in parallel | `shop_mcp.py`, TTL cache, `verifiers.py` (format, exact match, price sanity) | `baseline.py` (B0) running on T1/T2 |
| 2:15–3:00 | `resolve.py` dedup + narration alignment | Decision-1 `match()`; `board.py` invariants + tests | `run_eval.py` matching + M1–M5 |
| 3:00–3:45 | Integrate `loop.py` end to end; follow-up clip path | `voi.py` + Decision-1 `verdict()` / `probe()` | First full eval run → `FAILURES.md`; T3 transforms |
| 3:45–4:20 | `app/` UI: capture, ledger, ask, verdict | Shop cart demo beat | Fix the top failure, re-run, before/after numbers |
| 4:20–4:45 | **Freeze code.** Demo rehearsals ×3 on fresh tables | README "built today" section | Slides: problem, demo, results, failures, business |
| 4:45–5:00 | **Submit:** public repo, video/slides, README | | |

**Go/no-go gate at 3:00:** if end-to-end isn't producing a ledger from a video, cut to the core (§20) immediately.

---

## 19. Coding-Agent Kickoff Prompts

Dispatch in parallel worktrees once `contracts.py` is committed.

**Agent P1, Perception (`worktree-perception`):**
> Implement `ingest/frames.py` (ffmpeg 1 fps + scene-change keyframes, phash dedup with Hamming ≤ 6, Laplacian blur score, brightness) and `ingest/audio.py`. Implement `models/carbon.py::understand(video_path, dec_page_path) -> (list[RoomSegment], list[SpokenClaim], PolicyInfo)` and `models/hydrogen.py::extract_segment(segment, frames, claims) -> list[ItemObservation]` using the google-genai SDK with JSON-schema output. Hydrogen must never invent model numbers. Run segments concurrently (`max_concurrent` configurable). Include mocked unit tests and one live smoke test gated by env var.

**Agent P2, Commerce & Verifiers (`worktree-commerce`):**
> Implement `commerce/shop_mcp.py`, a JSON-RPC MCP client for `https://mcp.shop.com/` exposing `search_catalog(query, filters)`, `get_product(id)`, and cart creation, returning `Listing` objects with `retrieved_at`. Add `commerce/cache.py` (price TTL 10 min, metadata 24 h). Implement `ledger/verifiers.py` per PLAN §10 as pure functions with exhaustive pytest coverage, including a brand model-number regex table for ≥ 15 brands.

**Agent P3, Board & Decisions (`worktree-ledger`):**
> Implement `ledger/board.py::apply_patch()` enforcing invariants I1–I7 from PLAN §11, plus `rebuild_board(notes)` and a drift metric. Implement `models/decision.py` with `match(item_crop, listings)`, `verdict(board)` and `probe(board, item_ids)` on the Gemini Decisions API, one counterfactual board state per call and ablated fields set to the string "unknown". Implement `ledger/voi.py` per PLAN §12. 100% pass on invariant tests.

**Agent P4, Eval (`worktree-eval`):**
> Implement `eval/baseline.py` (B0 single Carbon call; B1 = B0 + top-1 Shop search), `eval/transforms.py` (blur, darken, frame-drop, audio noise, TTS contradiction splice), and `eval/run_eval.py` computing metrics M1–M11 from PLAN §14 with Hungarian matching on a ~60-class taxonomy. Output `eval/results/*.json` and a markdown results table. Do not tune anything on T2 labels.

---

## 20. Cut List

Cut from the top when behind schedule.

1. Live voice capture coach.
2. B1 baseline (keep B0).
3. T3 TTS contradiction splice (keep blur/darken transforms).
4. Shop cart / checkout beat (keep catalog pricing; mention the loop on a slide).
5. Multi-round asks (keep a single ask round).
6. Counterfactual low/high variants (keep `"unknown"` ablation only).
7. Room-coverage verifier.

**Never cut:** the CERTIFIED/UNCERTIFIED split with no invented prices (I6), the baseline comparison on unseen video, the failure log, and a live judge-table capture.

---

## 21. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Shop catalog lacks many home items | High | Medium | Present as honest UNCERTIFIED/UNKNOWN; it's a talking point, not a bug. Use B1 to show search alone isn't enough |
| Model IDs / Decisions API differ from assumptions | Medium | High | Verify in the first 20 min; keep a provider shim |
| EPIC-KITCHENS download too slow | Medium | Medium | Pick 8 short videos immediately; fall back to T2 + Charades clips |
| Rate limits during parallel extraction | Medium | Medium | Backoff, cap concurrency, cache responses by frame-hash key |
| Venue Wi-Fi fails during demo | Medium | High | `--replay` mode with cached responses; disclose it if used |
| Dedup errors (over-merge or under-merge) | High | Medium | Conservative merging; show the dup-rate metric honestly |
| "Image analyzer" anti-project perception | Low–Med | High | Frame as a decision agent: coverage verdict, counterfactual, asks, commerce loop |
| Price accuracy disputes (sale prices, used vs. new) | Medium | Low | Show the listing URL and timestamp; label prices "replacement cost, new" |

---

## 22. Privacy, Safety & Ethics

- **Home video is sensitive.** Process it without keeping it: delete raw video after extraction unless the user opts into vault storage. Blur faces, documents and screens in exported thumbnails.
- **No financial or legal advice:** the verdict is informational ("likely underinsured"), with a prompt to talk to their agent. It's not a coverage determination.
- **No fraud enablement:** the ledger only lists items seen in the video, and everything is timestamped. Narration can't add items without visual evidence (they stay UNCERTIFIED and flagged).
- **Data licensing:** EPIC-KITCHENS / Charades are used for non-commercial evaluation only, and no dataset media is committed to the repo. T2 videos are recorded with consent and contain no third-party faces.
- **Commerce honesty:** no purchase without explicit user confirmation; Shop Pay spend requests keep card data away from the agent.

---

## 23. Hackathon Compliance Checklist

- [ ] Public repository.
- [ ] ≥ 2 Gemini 4 models, each with a clear job (Carbon, Hydrogen, Decision-1) documented in the README.
- [ ] Media is central (video, audio, policy document image).
- [ ] **All code written during the event.** The README states that the *design pattern* (certified vs. uncertified evidence, counterfactual probing, VoI) was inspired by the public ARES-hack plan, and that **no ARES code was copied** (that repo has no license).
- [ ] README section **"What we built today"** listing every module.
- [ ] The demo shows only features built today.
- [ ] Evaluation includes unseen external data (T1) and is not made up entirely of team-written examples.
- [ ] Not on the anti-project list: not a Streamlit app, a dashboard-as-product, an image analyzer, or medical advice.
- [ ] Shopify bonus: shop.dev catalog pricing + cart beat documented.

---

## 24. Pitch Narrative & Judge Q&A

### 24.1 Pitch (≈ 45 s)
> "After the LA fires, thousands of families were asked to list every fork, lamp and laptop they'd lost, from memory. Most homes are underinsured, and nobody finds out until it's too late.
> Home Inventory Ledger turns a three-minute walkthrough video into an inventory your insurer will accept. Gemini 4 sees and hears your home, parallel agents itemize each room, and every price is either **certified** against a live Shopify listing or clearly marked as an estimate. We never invent a price.
> Then it tells you whether you're covered, which items that answer hinges on, and the one shot to film next that would settle it.
> Against a single Gemini call, we cut hallucinated prices from X% to 0 and duplicates from Y% to Z, on video we'd never seen. And when disaster strikes, the same list becomes your replacement cart."

### 24.2 Likely questions
| Question | Answer |
|---|---|
| *Why not one Gemini call?* | It double-counts, invents model numbers and prices, and can't tell you what's verified. Show M2/M3/M4. |
| *What does video give you over a typed list?* | Counts, model labels, completeness ("you never opened the closet"), and contradiction catching against narration. |
| *Why Decision-1?* | Calibrated closed-set confidence makes the counterfactual "what does this verdict hinge on" meaningful, and drives the VoI asks. |
| *What if an item isn't on Shopify?* | It stays UNCERTIFIED with a range or UNKNOWN with "needs appraisal". Honest by design (show a live example). |
| *Where does it fail?* | Show `FAILURES.md` and `STRESS.md`: dark rooms, mirrors, antiques, very long videos. |
| *Business model?* | Insurers pay for pre-loss inventories (faster claims, right-sized coverage); consumers freemium; commission on replacement purchases. |
| *Fraud risk?* | Every item is tied to a timestamped frame; narration can't add items; certified identity needs a visible label plus a catalog exact match. |

---

## 25. Roadmap Beyond the Hackathon

1. **Post-loss reconstruction mode:** build the ledger from *existing* media (old phone videos, social posts, real-estate listing tours, receipts in email).
2. **Depreciation / actual cash value:** age estimation from design cues and narration; replacement cost vs. ACV columns.
3. **Receipt and email ingestion:** purchase confirmations as CERTIFIED purchase-date and price evidence.
4. **Insurer integration:** export in carrier-specific schedule-of-loss formats; API for agents at renewal.
5. **Annual refresh:** a 60-second re-scan that diffs against last year's ledger.
6. **Additional price sources** beyond Shopify for categories it doesn't cover, with source-specific certification rules.
7. **Encrypted vault:** off-site storage with tamper-evident timestamps (hash chain) to strengthen pre-loss proof.

---

## 26. Open Questions

- Exact Gemini 4 model IDs and which one handles long video and live audio best (check the builder guide).
- Decisions API: image attachments allowed per call? Limits on question count and payload size?
- Shop MCP: exact `search_catalog` parameters, result fields (MPN/model in specs?), rate limits for anonymous use.
- EPIC-KITCHENS: which 8–10 videos give the best item variety in the shortest length?
- Should the demo use a sample declarations page (yes; never a real person's policy)?
- Team split: who owns the live demo phone and who owns the eval numbers?
