# Home Inventory Ledger, v2: Architecture & Scoping Notes

> **Implementation Note (4:25 PM Update):** Per **§0.1 & §7** below, **zero code from external repos was reused or copied** — 100% of this repository was implemented clean-room from scratch today under the [MIT License](LICENSE). Primary submission track: **Problem Statement One (*5 Sigma*)** + **Shopify (`shop.dev`) Commerce Bonus** (with Element-119 media stress ladders in [`eval/STRESS.md`](eval/STRESS.md)).

> **One-liner:** Film your stuff for 60 seconds. We tell you whether your insurance covers it, back every price with a real listing, and say which shot to film next to settle the answer.

Revised 2026-11-21, 1:40 PM, after re-reading `szaman19/ARES-hack` at commit `2de01ff` (1:27 PM). This replaces the strategy in [PLAN.md](PLAN.md); PLAN.md stays the reference for contracts, verifier details and the business case.

---

## 0. What changed and why (read this first)

| # | v1 assumption | What the ARES repo shows now | v2 decision |
|---|---|---|---|
| 1 | Reimplement certification, board and probe from scratch | ARES now has a tested, domain-agnostic core (`fusion/patch.py` invariants I1–I7, `run_investigation_loop`, `DecisionProbeClient` with influence + VoI, image attachments) and a **new-use-case guide** for adding domains without core edits | **Plug in as `domain/home_inventory.py`** and reuse the core. That frees ~2 of 3 build tracks for perception and eval. *Only valid if you're on the ARES team; see §0.1* |
| 2 | ARES is just inspiration | As-is, ARES fails three rules: input is synthetic Parquet (not media), its eval is **9 team-planted needles** ("evaluations made up entirely of examples the team wrote to pass" are banned), and **the baseline ties it 9/9**. ARES only wins on speed (34 s vs 87 s per case) and tokens | The home-inventory domain is **what makes ARES eligible**: real video and audio input, unseen external captures, and a baseline that clearly fails (invented prices, double counts) |
| 3 | Claim both tracks equally | The 5 Sigma criteria (baseline, messy unseen inputs, failure log) are ARES's DNA | **Primary: 5 Sigma.** Use an Element-119-style "where it breaks" ladder as the failure section, and keep the Shopify bonus |
| 4 | Decision-1 counts as one of the models | The rules list the Gemini 4 models as Argon, Carbon, Neon and Hydrogen. The Decisions API is extra | The ≥2-model requirement is met by **Carbon + Hydrogen with distinct jobs**. Decision-1 is on top of that |
| 5 | Whole-home walkthrough | Shopify catalog coverage is thin for furniture and appliances, and a 3-hour build can't eval whole homes | **Scope the demo and eval to electronics-dense scenes** (desk, entertainment center, go-bag). Highest value per item, legible labels, best catalog coverage. Whole home stays the product story |
| 6 | Eval on EPIC-KITCHENS + Hungarian matching on 60 classes | Too slow to download and map in the time left, and kitchen nouns don't carry prices | **Fresh captures from people not on the team** at the venue, labeled before any run, with a frozen dev/test split. EPIC is a stretch goal |
| 7 | 11 metrics | Judges will remember 3 | **Headline 3:** hallucinated-price rate, duplicate rate, contradiction catch. 4 supporting metrics |
| 8 | Open with the item list | An item list is exactly what an "image analyzer" looks like (a banned project) | **Open with the verdict, then the ask → refilm → badge-flips-CERTIFIED moment.** The item list is supporting evidence |

### 0.1 Ownership check (do this before writing code)
Repo authors are `Shehtab Zaman` and `raihans8379`, all commits from today, and there is **no LICENSE file**.
- **If you're on that team:** reuse is fine. It was built today, and the README must split "ARES core, built 10:05 AM–1:27 PM (commits `fd2e940`–`2de01ff`)" from "Ledger domain, built after 1:40 PM". Add an MIT `LICENSE`.
- **If you're not:** don't copy code. The no-license default plus the rule against "code you don't have rights to" means a disqualification risk. Either ask the authors to join or license it, or fall back to v1 (reimplement) and cut harder (§7).

---

## 1. How the idea maps onto ARES

ARES's tagline, *"the agents never read the haystack; they write the magnets"*, fits directly:
- **Haystack** = the global Shopify catalog (millions of listings, far beyond any context window).
- **Magnets** = catalog search queries that Hydrogen writes from what it saw in the video.
- **Deterministic verifiers** decide whether a listing *is* the item (model-number exact match), so a price is CERTIFIED or it isn't.
- **Decision-1** answers "are you covered?", probes which items the answer hinges on, and the top-VoI unknown becomes the **"film this next" request**.

```
video + narration + declarations-page photo
   │
   ├─ [NEW] ingest/frames.py        ffmpeg 1 fps + scene frames, phash dedup, blur score   (no LLM)
   ├─ [NEW] perceive/understand.py  Carbon: rooms/segments, timestamped narration claims, policy limit
   ├─ [NEW] perceive/extract.py     Hydrogen ×N parallel: ItemObservation[] per segment (frame ref, label text)
   ├─ [NEW] perceive/resolve.py     dedup + narration↔frame alignment → contradiction flags
   │
   ▼  build per-video Domain (top-12 items become the board's pivotal fields)
   ├─ [NEW]   domain/home_inventory.py   Domain plug-in
   ├─ [NEW]   adapters/shop_catalog.py   ExecutionBackend: search query = magnet → UNCERTIFIED note
   ├─ [REUSE] fusion.run_investigation_loop  propose (Hydrogen writes queries) → verify → fuse (Carbon) → decide → probe → acquire
   ├─ [REUSE] DecisionProbeClient        verdict + influence + VoI, one board per call, frames attached (≤3)
   │
   ├─ [NEW] ledger/asks.py          top-VoI field → "show the label on the back of the TV (±$1,400)"
   └─ [NEW] app/                    FastAPI + one HTML page: capture → verdict → ask → refilm
```

### 1.1 Domain plug-in contract (`domain/home_inventory.py`)

| `Domain` method | Implementation |
|---|---|
| `environment_description()` | Resolved item list (category, description, brand, `model_text`, best frame ts) + policy limit. Hydrogen reads this to write catalog queries |
| `magnet_rules()` | "A magnet is a plain product search string ≤ 120 chars. No URLs. Exact query `{brand} {model}`, then descriptive, then generic" |
| `create_backend()` | `ShopCatalogBackend`, built per the 5-method pattern in `tests/test_non_sql_domain.py` |
| `register_verifiers()` | `catalog_exact_match`, `price_sanity`, `quantity_consistency`, `frame_quality`, `policy_limit_parse` (pure Python, from PLAN.md §10). Each attaches the item's best frame as `ArtifactRef(kind="image")` so Decision-1 and the UI see visual proof |
| `pivotal_fields()` | `coverage_limit`, `rooms_missing`, and `item:<id>:price` for the **10 items with the widest price range × qty**. That is 12 fields, exactly the I7 cap. The rest roll up into a `misc_total` row (computed, not probed) |
| `required_signal_fields()` | `("coverage_limit",)`: the verdict can't be certified without a parsed limit |
| `decision_questions()` | Q1 `verdict` ∈ {adequate, likely_underinsured, underinsured, insufficient_evidence}; Q2 `sufficiently_verified` yes/no |
| `extract_field_patches_from_note()` | Deterministic: passing `catalog_exact_match` note → `item:<id>:price = listing.price`; uncertified search note → `{low, high}` from the 25th/75th percentile |
| `score()` | Against labels (§3) |

**Known gaps in reuse (budget for them):**
1. Fields are seeded statically, so **build the Domain object after extraction**, once per video. A follow-up clip only updates notes for the requested item; that field already exists.
2. ARES `voi` is a 0–1 decision swing. Rank asks by `voi × (high − low) × qty` to get a dollar figure.
3. `run_single_agent_baseline()` is a query-tool baseline. The video baseline (B0) is new, small code.
4. The probe is the cost driver (ARES: 128 Decision-1 calls / 9 cases). Cap at `max_rounds=2` live, and use M ≈ 8 items for low/high variants.

### 1.2 Model jobs (for the README and judges)
| Model | Single clear job |
|---|---|
| `gemini-4-carbon` | Watches the whole video once: room segments, narration transcript with timestamps, declarations-page limit. Fuses verifier notes into the board. Baseline B0 |
| `gemini-4-hydrogen` | Fast parallel workers: per-segment item extraction from frames (label reading, counts) and writing catalog-search magnets |
| `gemini-decision-1` | Calibrated closed-set calls: listing-match yes/no, coverage verdict, counterfactual variants → influence and VoI |
| *(stretch)* Neon or Argon | Check the builder guide at kickoff. If one is the live voice model, use it for a spoken capture coach ("closer on the label"). Don't plan around it |

---

## 2. The demo (3:00), verdict first

| Time | Beat |
|---|---|
| 0:00–0:15 | Hook: "After the LA fires, families had to list everything they'd lost from memory. Most homes are underinsured and nobody finds out until after." |
| 0:15–0:30 | Photograph a **sample** declarations page → `Coverage C = $15,000` |
| 0:30–0:55 | **Film the judge's table** for 20 s while narrating, including one wrong claim ("that's a Sony" on non-Sony headphones) |
| 0:55–1:20 | **Verdict card first:** "Likely underinsured. Certified $6.2k, estimated $13k–$19k. Hinges on: laptop, monitor, headphones." Contradiction flag on the headphones |
| 1:20–1:50 | **Hero beat:** "Show me the label under the laptop (±$1,100)." Film 5 s → badge flips **CERTIFIED**, listing link appears, range collapses, verdict confidence moves |
| 1:50–2:30 | **Proof slide:** B0 vs B1 vs Ledger on unseen captures. Headline: invented prices X% → 0%, duplicates Y% → Z%, contradictions caught n/N vs 0/N. Plus one honest failure and its fix |
| 2:30–3:00 | Business (insurer B2B2C) + "rebuild my home" Shop cart link (bonus track) |

Backup: `--replay` with cached responses for a pre-recorded clip, disclosed out loud if used.

---

## 3. Evaluation (5 Sigma), sized to 3 hours

### 3.1 Systems compared
- **B0, single call:** Carbon gets the full video: "List every item with brand/model, quantity and replacement price, as JSON."
- **B1, single call + search:** B0's items, each priced from the top-1 Shop result. This isolates *verification* from *search access*. Keep it: it's the honest strong baseline.
- **Ledger, no asks** and **Ledger, +1 ask round** (the ask round is simulated by the clip owner filming the requested shot).

### 3.2 Test set (no team-written examples)
| Set | How | Size | Ground truth |
|---|---|---|---|
| **Dev** | Team films 3 clips | 3 | Used for prompt tuning. **Never reported** |
| **Test: strangers** | 2:00–3:15: ask other hackers and volunteers to film *their own* desk or bag for 30–90 s on their phone and narrate naturally. Consent line in a form; no faces | 8–12 | A teammate who **never edits prompts** labels each clip **before any system run**: item list, counts, and a photo of each visible model label |
| **Test: perturbed** | Scripted transforms of the stranger clips: blur, darken, 2× speed, audio noise. Not hand-picked | 1 per transform | Inherited |
| **Live** | Judge's table | 1 | Judge confirms |

**Prompt freeze at 3:45 PM.** Run the test set once after freezing. Anything changed after that goes in FAILURES.md as a before/after with a re-run.

### 3.3 Metrics
| Metric | Definition | Why judges care |
|---|---|---|
| **Hallucinated-price rate** (headline) | Share of priced items with no real listing within ±15% (checked by a script against Shop search) | B0 invents prices; Ledger is 0% on CERTIFIED by construction, and the claim is *measured*, not asserted |
| **Duplicate rate** (headline) | Predicted rows mapping to an already-matched GT item | Multi-angle double counting is the classic video failure |
| **Contradiction catch** (headline) | Narration-vs-label conflicts flagged / present | Only possible with raw audio + video, so it also answers Element 119's question |
| Item recall / precision | Labeler matches rows with a tiny CLI (~2 min per clip; ~10 items per clip makes Hungarian matching unnecessary) | Completeness |
| Model-number accuracy | Exact normalized match on labeled items | Identity |
| Verdict accuracy + calibration | Sweep the limit over {0.5, 0.8, 1.0, 1.2, 1.5} × true total per clip | Decision quality; Decision-1 confidence curve |
| Cost / latency | Calls, tokens, seconds per clip | Honesty |

n is small (~100 items). Report counts with Wilson intervals, not bare percentages.

### 3.4 Where it breaks (failure section)
Three ladders only, each with a screenshot in `eval/STRESS.md`: **lighting** (daylight → lamp → near dark), **motion** (steady → walking → whip pan), **length** (30 s → 3 min → 10 min). Plus catalog coverage (laptop → desk chair → antique) shown as honest UNKNOWN. Keep **FAILURES.md** with ≥3 entries: input, expected, got, cause, change, before/after.

### 3.5 Clip collection playbook (owner: Track C, starts 2:00 PM)

**Why it's on the critical path.** Every number on the proof slide depends on clips that (a) nobody on the team wrote, (b) were labeled before any system run, and (c) were in hand before the 3:45 prompt freeze. People get harder to reach as the deadline nears: after ~3:00 other teams are heads-down, and at 4:30 nobody films anything for you. Labeling also takes ~5 min per clip, so 10 clips is ~50 min of one person's time. Start late and you either have no test set or one you labeled after seeing outputs, which is the exact thing judges are told to look for.

**Setup (1:50–2:00, 10 min)**
- A shared upload folder (Drive or Dropbox link, or AirDrop to the collector's phone). **Not in the repo.** Raw clips never get committed.
- `eval/labels/` in the repo, holding one JSON per clip (template below).
- The consent line, read aloud or pasted in chat; the contributor replies "yes" with a first name or handle:
  > "Your clip is used only to test our hackathon project. It won't be published or committed to our public repo and will be deleted after the event. Please film only your own things, no faces, and no screens or papers with personal info."
- Collector = labeler is fine. The labeler must **never edit prompts or look at system outputs** before the freeze.

**What each contributor films (2 short clips, ~2 min of their time)**
1. **Walkthrough (30–90 s):** "Film your desk, bag or setup like you're showing a friend, and narrate what things are ('that's my MacBook, got it in 2022'). Don't tidy anything." No coaching on angles or lighting. Variety comes from different people, not from instructions.
2. **Label reel (10–20 s):** close-ups of the model labels or undersides of the 2–4 most valuable items. This is the **ground truth** for model numbers, and also the **stand-in for the ask round**: when the system asks for item X's label, it gets that item's segment from the reel (only filmed before any run, so it's fair). This removes the need to track contributors down later.
3. **Blind contradiction (ask about half of them):** "Somewhere in the narration, misname one item on purpose, and tell only our labeler which one." The contributor picks it, so it isn't team-authored. It feeds the contradiction-catch metric.

**Quotas**
- Target **10 test clips from ≥6 different people**, minimum 6. Ten clips from two people is really two data points.
- **Parallel channel (send at 1:50):** text 4–5 friends outside the venue asking for a 60 s narrated clip of their living room or desk, plus a label reel. These are genuinely unseen, closer to the real product (homes, not hackathon tables), and arrive while you're busy building.
- Recruiting line: *"Got 2 minutes? Film your desk for 60 seconds on your phone and we'll show you what it's all worth at the end."* Start with neighboring tables, then organizers and volunteers.

**Labeling protocol (as clips arrive, ~5 min each)**
- Assign `clip_id` (`t01`, `t02`, …) and rename the file. Note conditions: lighting, motion, background noise.
- Write `eval/labels/<clip_id>.json`:
  ```json
  {
    "clip_id": "t04",
    "contributor": "handle-only",
    "conditions": {"lighting": "lamp", "motion": "walking", "audio": "crowd noise"},
    "items": [
      {"category": "laptop", "brand": "Apple", "model": "A2338", "qty": 1, "first_seen_s": 6, "price_usd": 999},
      {"category": "headphones", "brand": "Bose", "model": null, "qty": 1, "first_seen_s": 21, "price_usd": null}
    ],
    "planted_contradiction": {"item_index": 1, "said": "Sony"}
  }
  ```
- `price_usd`: fill in only for the top ~5 items by value (look them up manually after 3:15). It's needed for the verdict-sweep ground truth.
- **Commit each label file before the system ever runs on that clip.** Git timestamps are your proof to judges that labels came first.

**Hard cutoffs**
- **3:15:** collection closes for the main test set. Later clips go into a `late/` set, still usable if labeled before their run.
- **3:20:** commit `eval/test_manifest.json` (the clip IDs in the test set). The 3 team dev clips are excluded and never reported.
- Prompt engineers **don't watch test clips** before the 3:45 freeze.

**Common snags**
- iPhone `.mov`/HEVC and vertical video are fine (ffmpeg handles both). Ask for ≤90 s to keep uploads quick on venue Wi-Fi.
- Someone films venue property or another person's stuff: keep it out of the set, since they can't consent for it.
- Too few volunteers by 2:45: lean on the friends channel and organizers. Don't fill the gap with team clips in the test set.

---

## 4. Schedule (now → 5:00 PM)

Three tracks. With two people, merge C into A and B, but **someone must start collecting stranger clips at 2:00**: it's the one thing you can't make up later.

| Time | A: Perception | B: Domain + commerce | C: Eval + demo |
|---|---|---|---|
| 1:45–2:00 | Confirm model IDs (builder guide); ffmpeg works | **Shop MCP smoke test:** does `search_catalog` return MPN/model fields? (Decides whether `catalog_exact_match` matches title or specs) | Consent form; recruit capturers; film 3 dev clips |
| 2:00–2:45 | `ingest/frames.py`, Carbon `understand()`, Hydrogen `extract()` in parallel | `ShopCatalogBackend` + verifiers + unit tests | Collect stranger clips; labeler labels as they arrive; `eval/baseline.py` B0 |
| 2:45–3:20 | `resolve.py` (dedup, narration alignment) | `domain/home_inventory.py`; run `run_investigation_loop` on dev clip 1 end to end | B1; `score.py`; transforms |
| **3:20 gate** | **One dev clip → board with ≥1 CERTIFIED price and a verdict.** If not, cut to §5 now | | |
| 3:20–3:45 | Follow-up clip path (requested item only) | `ledger/asks.py`; verdict + hinges_on | First B0/B1 runs on the test set |
| **3:45** | **Prompt freeze** | | Ledger runs on the test set |
| 3:45–4:20 | `app/` single page: capture → verdict → ask → refilm | Shop cart link (bonus) | Results table, FAILURES.md, STRESS.md |
| 4:20–4:45 | **Code freeze.** 3 rehearsals on unfamiliar tables | README "built today" + model jobs | Slides: hook, demo, results, failure, business |
| 4:45–5:00 | **Submit:** public repo, LICENSE, README, slides/video | | |

---

## 5. Cut list (top first)
1. Live voice coach (Neon/Argon).
2. Shop cart beat → one slide.
3. Perturbed test set → keep darken only.
4. Low/high counterfactual variants → keep "unknown" ablation.
5. Multi-room segmentation → treat the clip as one segment, chunked by time.
6. Carbon fusion → deterministic patches only via `extract_field_patches_from_note` (the ARES invariants still hold).

**Never cut:** CERTIFIED vs UNCERTIFIED with no invented prices, B0 on unseen stranger clips, FAILURES.md, the live refilm → CERTIFIED flip.

---

## 6. Rules compliance (re-checked against the participant guide)

| Rule | How v2 meets it |
|---|---|
| ≥2 Gemini 4 models, each with a clear job | Carbon (understand + fuse + baseline), Hydrogen (parallel extraction + magnets); Decision-1 in addition |
| Beyond a text box; media central | Video, narration audio and a declarations-page photo are the only inputs; frames are the proof artifacts |
| Eval on inputs you didn't prepare | Stranger captures, labeled before runs, prompt freeze, dev set never reported |
| Not "scripted inputs only" | Judge's table, live |
| Public repo, new work, demo only today's work | ARES core and Ledger domain both built today; README timeline with commit hashes; the demo shows Ledger features |
| Rights to code, data and assets | §0.1 ownership check; stranger clips with consent; no dataset media committed; sample (not real) declarations page |
| Anti-projects: image analyzer, dashboard, Streamlit, medical | Output is a coverage *decision* plus an active ask loop, not labels on an image; one page centered on capture → verdict, not a dashboard; FastAPI + static HTML |

---

## 7. If you can't reuse ARES (not on the team)
Follow v1's architecture, but shrink the board to plain Python: `notes: list[Note]`, `rebuild(notes)`, and three invariants only (I1 provenance, I2 weakest link, I6 no invented prices). Probe = "unknown" ablation on the top 6 items. Drop Carbon fusion (cut item 6). Everything in §2–§5 still applies.
