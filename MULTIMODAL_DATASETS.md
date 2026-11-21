# Safe Multimodal Datasets & Git Worktree Guide for ARES

This branch (`jb-multimodal-datasets`) stages **4 vetted, publicly available, laptop-safe multimodal datasets** (structured DuckDB/Parquet tables + real `.png`/`.jpg`/`.wav` media + withheld ground-truth labels) and wires them directly into the existing ARES search & counterfactual probing stack via [`domain/multimodal_datasets.py`](domain/multimodal_datasets.py) and [`scripts/stage_multimodal_datasets.py`](scripts/stage_multimodal_datasets.py).

---

## 1. Git Worktrees & Branch Layout Found

| Worktree Path | Branch | Role & Status |
| --- | --- | --- |
| `/Users/jiddybearz/dev/ARES-hack` | `jb-B2-ui` | Active primary worktree (UI server, charts, and probe visualizations). Left **100% untouched**. |
| `/Users/jiddybearz/dev/ARES-hack/.worktrees/multimodal-datasets` | `jb-multimodal-datasets` | Isolated git worktree branched cleanly from `origin/main` containing the multimodal dataset catalog, hardened downloader, ARES search adapter, tests, and local `.data_cache/` (**95.1 MB** staged across all 4 datasets). |

* **Related remote/local branches inspected:** `origin/main` (Wave 1 + Gate 1 merged), `origin/sz-A1` (Person A SQL guards, schema views, and deterministic plot artifacts), `jb-B1-gate1` (Person B models/fusion/probe).
* **Zero Merge Conflicts:** This branch adds only self-contained files (`domain/multimodal_datasets.py`, `scripts/stage_multimodal_datasets.py`, `tests/test_multimodal_datasets.py`, `MULTIMODAL_DATASETS.md`, and `.data_cache/` in `.gitignore`) with zero new package dependencies.

---

## 2. Top 4 Safe Multimodal Datasets (Tabular + Real Media + Decoys)

### Option 1: ZTF `BTSbot` Supernova Transient Alerts — *Astronomy / Space*
* **Direct Link:** https://huggingface.co/datasets/MultimodalUniverse/btsbot *(Official NeurIPS Datasets & Benchmarks Multimodal Universe + Caltech/Northwestern Zwicky Transient Facility release)*
* **What's inside:**
  * **Table (`observations`):** Photometric & alert metrics (`magpsf`, `sigmapsf`, `diffmaglim`, `dsnrms`, `ssnrms`), ML alert scores (`drb`, `new_drb`, `acai_h/v/o/n/b`), PSF & CCD subtraction diagnostics (`fwhm`, `chipsf`, `nbad`, `nneg`, `mindtoedge`), host-galaxy context (`sgscore1`, `distpsnr1`), and light-curve history (`ndethist`, `ncovhist`, `age`, `days_since_peak`, `is_rise`).
  * **Media:** 3-panel `63×63` telescope sky cutout stamps (`Science` | `Reference` | `Difference`) rendered to lossless `.png`.
  * **Ground-truth label (`ground_truth.json`):** Spectroscopically confirmed supernova (`is_true_event = true`) vs. non-SN alert (`false`).
* **Why it's cool:** High-SNR brightness jumps with `drb > 0.99` look like newborn supernovae in the table, until inspecting the `Science / Reference / Difference` sky stamps separates a real point-source supernova offset from a host galaxy from **recurring variable stars, AGN nuclear flares, cosmic-ray hits, and CCD subtraction ghosts**.
* **Size & Download method:** Full split is 3 Parquet shards (~1.6 GB each), or streamed via DuckDB `httpfs` HTTP Range requests into a compact **~3–30 MB** local `.parquet` + `.png` slice without downloading the full 4.8 GB.
* **Exact safe download command:**
  ```bash
  uv run python scripts/stage_multimodal_datasets.py --dataset btsbot --cache-dir ./.data_cache --media-limit 60
  ```

---

### Option 2: Caltech Camera Traps + MegaDetector v5 — *Wildlife Ecology & Motion Sensors*
* **Direct Link:** https://lila.science/datasets/caltech-camera-traps/ *(Hosted on Google Cloud `storage.googleapis.com/public-datasets-lila` & AWS Open Data Source Cooperative)*
* **What's inside:**
  * **Table (`observations`):** **243,100** motion-triggered Southwest US camera-trap captures with `location_id` (140 stations), `captured_at`, `hour_local`, `seq_id`, `seq_num_frames`, `frame_num`, `width`, `height`, joined with automated MegaDetector v5a scores (`max_detection_conf`, `n_detections`, `max_bbox_area_frac`) and `media_url`.
  * **Media:** Full-resolution daytime RGB & nighttime infrared camera-trap `.jpg` photos.
  * **Ground-truth label (`ground_truth.json`):** Biologist species annotations across 21 wildlife categories (`bobcat`, `mountain_lion`, `coyote`, `fox`, `deer`, etc.) vs. `empty` false motion triggers.
* **Why it's cool:** Sun-heated desert brush, wind gusts, and shifting rock shadows trigger multi-frame bursts and fool automated detector numbers (`max_detection_conf >= 0.80` on **10,000+ `empty` decoys**), whereas inspecting the actual `.jpg` immediately distinguishes swaying grass from a nocturnal bobcat or coyote.
* **Size & Download method:** Full 243,100-row table + MegaDetector scores is only **~14 MB** compressed (**~50 MB** with 36 pre-cached `.jpg` photos), and any other row's `.jpg` (~150–500 KB) is fetched **on-demand over HTTPS** only when ARES inspects that candidate.
* **Exact safe download command:**
  ```bash
  uv run python scripts/stage_multimodal_datasets.py --dataset caltech_camera_traps --cache-dir ./.data_cache --media-limit 36
  ```

---

### Option 3: Pyronear Early Wildfire Lookout Network (`pyro-sdis`) — *Environmental / Wildfire Early Warning*
* **Direct Link:** https://huggingface.co/datasets/pyronear/pyro-sdis *(Official Pyronear Fire Detection organization release)*
* **What's inside:**
  * **Table (`observations`):** **1,129** mountaintop fire-lookout tower alert frames with `partner` (`biobio`, `engie`, `sdis`), `camera` & `azimuth_deg`, `captured_at`, `hour_obs`, frame dimensions, and deterministic optical telemetry (`brightness_mean`, `horizon_brightness`, `sky_ground_contrast`, `gray_haze_fraction`, `warm_glare_fraction`).
  * **Media:** Real mountaintop fire-lookout `.jpg` photographs (~85–140 KB per frame).
  * **Ground-truth label (`ground_truth.json`):** Confirmed early wildfire smoke plume (`is_true_event = true`, with plume box count) vs. verified hard-negative false alarms (`false`).
* **Why it's cool:** Automated lookout alerts spike on `gray_haze_fraction` and bright horizon contrast during dry afternoons, where viewing the actual tower `.jpg` separates a **rising forest smoke column** from **low valley fog banks, cumulus cloud bases, harvest dust, or lens flare**.
* **Size & Download method:** Entire validation Parquet shard is **122 MB** remotely (**~6–25 MB** locally with all 1,129 tabular rows + balanced real/decoy `.jpg` frames).
* **Exact safe download command:**
  ```bash
  uv run python scripts/stage_multimodal_datasets.py --dataset pyro_sdis --cache-dir ./.data_cache --media-limit 60
  ```

---

### Option 4: UrbanSound8K Acoustic Sensor Network — *Urban Audio Sensors + `.wav` Clips*
* **Direct Link:** https://huggingface.co/datasets/danavery/urbansound8K *(NYU MARL UrbanSound8K benchmark in native Hugging Face Parquet)*
* **What's inside:**
  * **Table (`observations`):** Acoustic trigger metadata (`slice_file_name`, `freesound_id`, `start_s`, `end_s`, `duration_s`, `salience`, `fold`, `sample_rate_hz`) plus deterministic waveform/FFT sensor telemetry (`rms_db`, `peak_amp`, `crest_factor_db`, `zero_crossing_rate`, `spectral_centroid_hz`, `high_freq_energy_ratio`).
  * **Media:** Real `.wav` field audio clips (`<= 4s`) **plus** deterministic 2-panel waveform + spectrogram `.png` cards so ARES's vision-enabled `DecisionProbeClient` can also inspect acoustic candidates visually out of the box.
  * **Ground-truth label (`ground_truth.json`):** Emergency acoustic signatures (`gun_shot`, `siren`) vs. urban noise decoys (`jackhammer`, `car_horn`, `dog_bark`, `drilling`, `street_music`, etc.).
* **Why it's cool:** Short, high-crest-factor impulsive transients (`crest_factor_db > 15 dB`, `salience = 2`) look like gunshots in acoustic sensor tables, until inspecting the `.wav` clip / spectrogram reveals rhythmic `jackhammer` impacts, a `car_horn`, or a `dog_bark`.
* **Size & Download method:** Single remote shard is ~434 MB; streamed slice of 450 tabular rows + verified `.wav` clips + spectrogram `.png` cards is **~36 MB** locally.
* **Exact safe download command:**
  ```bash
  uv run python scripts/stage_multimodal_datasets.py --dataset urbansound8k --cache-dir ./.data_cache --media-limit 60
  ```

---

## 3. Minimal-Effort Integration with Existing ARES Search & Probing

`MultimodalSearchDataset` plugs directly into `DuckDBTrustedRunner`, `NoteStore`, `VerifierRegistry`, `GeminiModelProvider`, and `DecisionProbeClient` with **zero changes to `ares/`, `fusion/`, or `models/`**:

```python
from ares.store import NoteStore
from domain.multimodal_datasets import MultimodalSearchDataset

# 1. Open any staged dataset ("btsbot", "caltech_camera_traps", "pyro_sdis", "urbansound8k")
ds = MultimodalSearchDataset.open("btsbot", cache_root="./.data_cache")
store = NoteStore()

# 2. Filter tabular haystack with ARES DuckDBTrustedRunner (ground truth is hermetically withheld!)
rows, sql_note = ds.search_sql(
    """
    SELECT record_id, object_id, magpsf, drb, ndethist, sgscore1
    FROM observations
    WHERE drb > 0.99 AND magpsf < 19.5 AND media_relpath != ''
    ORDER BY ndethist ASC
    LIMIT 5
    """,
    note_store=store,
)

# 3. Run deterministic TRUSTED media verifier -> emits CERTIFIED Note + verified PNG/JPG ArtifactRef
verifier_result, media_note = ds.inspect_candidate(rows[0]["record_id"], note_store=store)
print(media_note.artifacts[0])  # Ready for GeminiModelProvider & DecisionProbeClient!

# 4. Score final predictions against withheld ground_truth.json
metrics = ds.score_predictions({rows[0]["record_id"]: True})
print(metrics)
```

---

## 4. Strict Safety Controls & One-Command Cleanup

* **Allowlisted HTTPS hosts only:** `huggingface.co` (+ official HF CDN), `storage.googleapis.com`, `lila.science`, `lilawildlife.blob.core.windows.net`.
* **Safe formats only:** `.parquet`, `.json`, `.png`, `.jpg`, `.wav`. Explicitly blocks `.pkl`, `.pt`, `.bin`, `.sh`, zip-slip paths, decompression bombs (`Image.MAX_IMAGE_PIXELS = 25_000_000`), and `trust_remote_code`.
* **Magic-byte & SHA-256 verification:** Every downloaded or rendered file has its binary header verified before write and its SHA-256 logged in `.data_cache/<dataset>/manifest.json`.
* **Instant cleanup:** Everything staged lives inside `./.data_cache/` (gitignored):
  ```bash
  rm -rf ./.data_cache
  ```
