# NASA TESS Exoplanet Triage Example

Scientific tabular + multimodal search example (`ExoplanetTriageDomain`) built strictly on top of ARES's narrow waist (`ares.contracts`, `ares.engine`, `ares.context`, `ares.loop`, `ares.adapters.DuckDBBackend`) with zero core library edits.

## Architecture

- **Haystack (`lightcurves`, `stellar_catalog`, `pixel_stamps`)**: Partitioned Parquet shards (either real NASA TESS SPOC FITS photometry + 11×11 Target Pixel File stamps staged via `scripts/`, or deterministic offline synthetic fallback when `data/tess_staged/` is absent). Official NASA dispositions are strictly excluded from all queryable tables.
- **4 Deterministic `TRUSTED` Physics & Spatial Verifiers (`domain.py`)**:
  1. `verify_periodic_transit_snr`: Periodic photometric dip significance (`>= 2` transits, depth `>= 150 ppm`, `SNR >= 7.0`).
  2. `verify_odd_even_eclipse_symmetry`: Odd vs. even transit depth consistency (`< 3.0 sigma`) and phase-`0.5` secondary occultation rejection (`< 4.5 sigma`).
  3. `verify_keplerian_density_consistency`: Kepler's 3rd Law transit-implied stellar density vs. Gaia/TIC host density (`[0.18x, 5.5x]`).
  4. `verify_ccd_centroid_lock`: Sub-pixel `diff_snr^2`-weighted centroid lock (`<= 0.45 px`) and aperture isolation (`CROWDSAP >= 0.80`).
  5. `verify_exoplanet_confirmation`: Compound 4-way gate + deterministic SHA-256 2×2 Observatory PNG card emission (`artifacts.py`).
- **Observatory UI (`observatory_server.py` + `static/index.html`)**: Local HTTP server with deterministic offline verifier preview, SSE-streamed ARES runs, and benchmark comparison tables.

## Commands

```bash
# Optional: download and stage real NASA TESS FITS & Parquet shards
uv run python examples/exoplanet_triage/scripts/fetch_real_tess_raw.py
uv run python examples/exoplanet_triage/scripts/stage_tess_parquet.py
uv run python examples/exoplanet_triage/scripts/build_cases_manifest.py

# Launch the local Observatory Web UI (supports offline deterministic preview)
uv run python examples/exoplanet_triage/observatory_server.py --port 8765

# Run the 9-case NASA TESS benchmark (requires GEMINI_API_KEY)
uv run python examples/exoplanet_triage/scripts/run_bench.py
```
