# Element 119 Multimodal Stress Ladders — Where It Breaks

Per **Problem Statement Two (Element 119)** and **5 Sigma** (*"push them as far as they go... Then show us where it breaks"*), `eval/transforms.py` and `eval/run_eval.py --stress` evaluate four controlled degradation ladders on identical underlying scenes.

---

## 1. Lighting Ladder (`daylight` → `darken_mild` → `darken_severe`)

| Rung | Transform (`eval/transforms.py`) | Mean Brightness | What Holds Up | Where It Breaks | System Response |
|---|---|---|---|---|---|
| **L0 Baseline** | Original indoor capture | ~115–165 | Category detection, brand logos, close-up model labels, narration alignment | — | Full `CERTIFIED` on visible labels |
| **L1 Lamp / Dim** | `darken_mild` (`brightness=-0.22, gamma=0.60`) | ~55–80 | Object recall, counts, large logos (Bose, Sony, Apple), audio claims | Fine 6pt etched aluminum text (`Model A2992`) loses contrast | Stays `UNCERTIFIED` range; Decision-1 VoI asks for lit label close-up |
| **L2 Near-Dark** | `darken_severe` (`brightness=-0.42, gamma=0.35`) | < 28 | Spoken narration claims still parsed from raw audio | Dark peripherals blend into desk shadow; `frame_quality` gate trips (`brightness < 28`) | `frame_quality` verifier blocks any false `CERTIFIED` upgrade; emits `"Hold steady with better light"` ask |

---

## 2. Motion & Sharpness Ladder (`steady` → `blur_mild` → `blur_severe`)

| Rung | Transform | Laplacian `blur_score` | What Holds Up | Where It Breaks | System Response |
|---|---|---|---|---|---|
| **M0 Handheld** | Normal pan | 80–450 | Keyframe dedup (`dHash`) drops redundant frames by ~55–65% | Occasional mid-pan motion blur frame | `ingest/frames.py` automatically replaces blurry frame with sharper neighbor within Hamming ≤ 10 |
| **M1 Fast Walk** | `blur_mild` (`boxblur=4:1`) | 25–55 | Coarse categories, counts, silhouette dedup | Small SKU/serial strings unreadable | `UNCERTIFIED` 25th–75th percentile range preserved; 0 invented prices |
| **M2 Whip-Pan** | `blur_severe` (`boxblur=11:2`) | < 15 | Audio claims + largest items (monitor, guitar) | Small items missed; OCR impossible | `frame_quality` (`blur_score < 22`) blocks certification; prompts 3-second steady re-capture |

---

## 3. Audio Messiness & Contradiction Ladder (`clean` → `audio_noise` → `contradiction`)

| Rung | Condition | What Raw Multimodal Captures (vs. Text) | Break Point |
|---|---|---|---|
| **A0 Clear Narration** | Owner narrates purchase year & brand while panning | Timestamped alignment (`±6s`) links spoken claims to specific visual tracks | — |
| **A1 Crowd / Pink Noise** | `audio_noise` (`anoisesrc` mixed over speech) | Carbon still transcribes primary speech and timestamps | Overlapping simultaneous speakers > 0 dB SNR can misattribute brand to adjacent item |
| **A2 Spoken Contradiction** | Narrator says *"Sony headphones"* while frame shows **Bose** logo | **Caught (`narration_conflict` flag)**: visual/label identity overrides spoken misnaming | If logo is completely occluded AND unlit, spoken brand remains `UNCERTIFIED` (`brand_from_narration_only` flag) — never `CERTIFIED` |

---

## 4. Catalog Coverage Ladder (`mainstream electronics` → `unbranded peripherals` → `antiques / custom`)

| Category Tier | Example | Shop MCP Result | Ledger Output |
|---|---|---|---|
| **Tier 1: Exact Model Electronics** | Sony α7 IV (`ILCE-7M4`), Dell `U2723QE`, MacBook Pro `A2992` | Live listing + SKU/title exact match | **CERTIFIED** exact replacement price + clickable merchant URL + `retrieved_at` |
| **Tier 2: Branded / Unlabeled Items** | Apple iPhone (generation unclear), Bose headphones (no model #) | Multiple generation listings | **UNCERTIFIED** 25th–75th percentile range + VoI capture request if verdict hinges on it |
| **Tier 3: Custom / Antiques / Off-Catalog** | Luthier guitar, heirloom art, custom furniture | No credible product match (`Decision-1` match prob < 0.50) | **UNKNOWN** + `needs_appraisal` flag (never guesses a number) |
