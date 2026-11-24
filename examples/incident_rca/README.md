# Incident Root-Cause Analysis Example (DuckDB + Parquet)

Demonstrates ARES orchestrating a search over partitioned software telemetry Parquet shards (`service_logs`, `metrics`, `traces`) via `ares.adapters.DuckDBBackend`:

- **3-of-3 Multi-Clue Needles**: Aligning upstream error retries, metric baseline z-score breaches, and downstream trace timeouts on shared `trace_id` / `shard_id`.
- **2-of-3 Look-Alike Decoys**: High-volume partial decoys are pruned deterministically by `verify_three_way_root_cause` without hallucinating root causes.
- **Under-Determined Cases**: Missing signals remain `UNCERTIFIED` and terminate honestly as `honest_uncertified` with zero false certainty.
- **Deterministic Visual Artifacts**: `artifacts.py` and `charts.py` render headless Matplotlib Agg PNGs with SHA-256 provenance.

## Commands

```bash
# Run the 9-case benchmark (requires GEMINI_API_KEY)
uv run python -m examples.incident_rca.benchmark

# Regenerate diagnostic charts from saved benchmark results (offline)
uv run python -m examples.incident_rca.generate_charts
```
