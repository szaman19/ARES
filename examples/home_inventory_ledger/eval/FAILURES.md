# Failure Log & Iterative Engineering Fixes (`5 Sigma` Deliverable)

Per the **5 Sigma** track requirements (*"Where does your system still fail, and what did you change after seeing those failures?"*), this log records concrete failure modes discovered on development/perturbed captures, root causes, code changes made, and measured before/after impact.

---

## Failure #1 — Accessory Listing Trap Certifying a $24 Case as a $1,999 Laptop

- **Input:** Desk walkthrough clip (`dev_01_desk`) where follow-up clip revealed `Model A2992` under an Apple MacBook Pro 14".
- **Expected:** `CERTIFIED` replacement price ~$1,999 from an actual MacBook Pro A2992 listing, or `UNCERTIFIED` range if only generic laptops returned.
- **Got (before fix):** `catalog_exact_match` matched `"Hard Shell Protective Case for MacBook Pro 14 A2992 Clear Cover"` ($24.99) because the normalized string `A2992` appeared verbatim in the title!
- **Root Cause:** Pure substring/token model matching ignores whether the listing is the host device or a cheap accessory compatible *with* that model number.
- **Change Made (`ledger/verifiers.py::price_sanity`):**
  1. Added deterministic accessory/parts blocker (`"case for"`, `"cover for"`, `"remote for"`, `"replacement"`, `"screen protector"`, `"earpads for"`, `"for parts"`, etc.) whenever the item's own category is not an accessory.
  2. Added a `[0.30×, 3.00×]` peer-median price band check across non-accessory search results before any listing can be accepted as `CERTIFIED` or included in the `UNCERTIFIED` 25th–75th percentile range.
- **Measured Impact:** Accessory false-certification rate on `A2992` / `WH-1000XM5` / `ILCE-7M4` queries dropped from **1/4 (25%) → 0/4 (0%)**; `test_verifiers.py::test_price_sanity_blocks_accessory_trap` guards against regression.

---

## Failure #2 — Multi-Angle Pan Double-Counting & Summing Quantities Across Frames

- **Input:** Continuous 20-second desk pan (`dev_02_pan`) revisiting the monitor and laptop twice while sweeping left-to-right-to-left.
- **Expected:** `1× Monitor`, `1× Laptop`.
- **Got (Baseline B0 & early extractor):** Single-call baseline B0 emitted duplicate rows when the camera returned to the desk corner; naive track merging summed `quantity = 1 + 1 = 2`.
- **Root Cause:** Summing `obs.quantity` across sequential keyframes treats two temporal sightings of the same object as two physical units.
- **Change Made (`ledger/resolve.py` + `ledger/verifiers.py::quantity_consistency`):**
  1. Track deduplication merges compatible same-category sightings within the same room segment (`Δt ≤ 25s` and non-conflicting brand/model).
  2. Enforced `quantity = max(simultaneous_counts_per_frame)` backed by deterministic verifier `quantity_consistency` (an item's count cannot exceed the maximum number of simultaneous instances seen in a single frame unless distinct serial/model numbers exist).
- **Measured Impact:** Duplicate rate on multi-angle pans dropped to **0%** on dev captures while preserving true multi-unit counts (e.g., `2× studio monitors` visible together in one frame).

---

## Failure #3 — Severe Low-Light & Motion Blur Hallucinating Model Characters

- **Input:** Perturbed clip (`darken_severe` and `blur_severe` from `eval/transforms.py`), simulating dim bedroom lamps and fast whip-pans.
- **Expected:** Downgrade unreadable labels to `UNCERTIFIED` and issue a `"Hold steady with better light"` capture request.
- **Got (before fix):** Vision worker occasionally guessed plausible characters on low-contrast labels (e.g., `A2338` vs `A2992`).
- **Root Cause:** Even with prompt instructions ("never guess"), extreme blur/darkness degrades OCR reliability without a deterministic pixel gate.
- **Change Made (`ingest/frames.py` + `ledger/verifiers.py::frame_quality` + `ledger/loop.py`):**
  1. Computed discrete 2D Laplacian variance (`blur_score`) and mean luminance (`brightness`) on every extracted frame.
  2. Added `frame_quality` deterministic gate (`blur_score >= 22.0`, `brightness >= 28.0`): even if a model reports `model_text`, certification is **blocked** when `frame_quality` fails, keeping the item `UNCERTIFIED` with flag `frame_too_blurry` / `frame_too_dark` and routing `voi.py` to request a steady, lit close-up.
- **Measured Impact:** Eliminated false `CERTIFIED` upgrades on `blur_severe` and `darken_severe` stress rungs (0 false certifications), converting silent OCR errors into explicit user capture requests.

---

## Failure #4 — Non-Shopify / Antique Items Returning Unrelated Generic Merchandise

- **Input:** Acoustic guitar with no visible headstock brand and vintage/custom furniture items where Shopify merchant coverage varies.
- **Expected:** Honest `UNCERTIFIED` range or `UNKNOWN` (`needs_appraisal`) — never a fabricated point price.
- **Root Cause:** Taking top-1 search result blindly (as in Baseline `B1`) assigns an arbitrary listing's exact price to an unverified item.
- **Change Made (`ledger/board.py` Invariant I6 + `models/decision.py`):**
  1. Enforced **Invariant I6**: `unit_price` is structurally forbidden (`InvariantViolationError`) unless `price_status == "CERTIFIED"` via `catalog_exact_match`.
  2. `Decision-1` filters candidate listings via closed-set `noul` visual/spec match probability (`P(same_category_product) >= 0.50`); if fewer than 2 credible listings remain, status stays `UNKNOWN` (+ `needs_appraisal` for art/antiques/jewelry).
