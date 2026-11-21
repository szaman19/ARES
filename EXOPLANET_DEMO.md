# ARES × NASA TESS: *Planet Hunter vs. Cosmic Impostors*

Flagship real-data domain plug-in (`ExoplanetTriageDomain`) built strictly on top of ARES's domain-agnostic narrow waist (`contracts.py`, `ares/`, `fusion/`, `models/`) with zero core library edits.

---

## Architecture Overview

- **Haystack & Ground Truth:** Real NASA TESS SPOC 2-minute cadence FITS photometry (`_lc.fits`), Target Pixel File CCD stamps (`_tp.fits`), and Gaia/TIC stellar parameters staged into partitioned Parquet shards (`lightcurves`, `stellar_catalog`, `pixel_stamps`), with official NASA TFOPWG dispositions (`CP`/`KP`, `FP`, `PC`/`APC`) strictly stripped from all Parquet tables.
- **Deterministic `TRUSTED` Physics Verifiers (`domain/exoplanet_triage.py`):**
  1. `verify_periodic_transit_snr` — Periodic photometric dip significance (`>= 2` transits, depth `>= 150 ppm`, `SNR >= 7.0`).
  2. `verify_odd_even_eclipse_symmetry` — Odd vs. even transit depth consistency (`< 3.0 sigma`) and phase-`0.5` secondary occultation rejection (`< 4.5 sigma`).
  3. `verify_keplerian_density_consistency` — Kepler's 3rd Law transit-implied stellar density vs. Gaia/TIC host density (`[0.18x, 5.5x]`).
  4. `verify_ccd_centroid_lock` — Sub-pixel `diff_snr^2`-weighted centroid lock (`<= 0.45 px`) and aperture isolation (`CROWDSAP >= 0.80`).
  5. `verify_exoplanet_confirmation` — Compound 4-way confirmation gate + deterministic SHA-256 2×2 dark-mode Observatory PNG card emission (`domain/exoplanet_artifacts.py`).
- **Live Stage Observatory UI (`app/observatory_server.py` + `app/static/index.html`):** Blinded Judge Sector Menu, instant deterministic scan, full Gemini 4 + Decision-1 + Baseline runner, counterfactual CCD stamp ablation toggle, and official NASA Exoplanet Archive ground-truth reveal.

---

## Quickstart Commands

```bash
# 1. Stage real NASA TESS FITS & Parquet shards (one-time setup)
.venv/bin/python scripts/fetch_real_tess_raw.py
.venv/bin/python scripts/stage_tess_parquet.py
.venv/bin/python scripts/build_cases_manifest.py

# 2. Run unit & domain integration tests
.venv/bin/python -m pytest tests/test_exoplanet_triage.py -v

# 3. Launch the Live Stage Observatory Web UI
.venv/bin/python app/observatory_server.py --port 8765

# 4. Run the 9-case Real NASA TESS Benchmark (ARES vs. Single-Agent Carbon SQL Baseline)
.venv/bin/python scripts/run_exoplanet_bench.py
```

---

## Verified Stage 1 Benchmark Scoreboard (`exoplanet_tess_stage1.json`)

| Metric | ARES Swarm + Decision-1 | Single-Agent Carbon SQL Baseline |
| :--- | :---: | :---: |
| **Overall Pass / Action Accuracy** | **9 / 9 (100.0%)** | 9 / 9 (100.0%) |
| **Cosmic Impostors Avoided (`look_alike_decoy`)** | **3 / 3** | 3 / 3 |
| **False Certainty (`under_determined`)** | **0** | 0 |
| **Mean Wall-Clock Time / Case** | **27.2 s** (**4.0× faster**) | 109.2 s |
| **Total Tokens Across 9 Cases** | **564,869** (**2.84× fewer**) | 1,606,034 |
| **Provenance & Drift Audit** | **`ungrounded=0`, `illegal_cert_upgrades=0`, `max_drift=0.0`** | Unaudited |
| **Gate 1 Verdict** | **`GO (PASSED)`** | — |
