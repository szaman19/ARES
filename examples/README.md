# ARES Examples

Each directory under `examples/` showcases a domain plug-in, execution backend, or application built around ARES (or contributed alongside it):

| Example | Modality / Backend | Highlights | Run |
| :--- | :--- | :--- | :--- |
| [`codebase_audit/`](codebase_audit/README.md) | Custom non-SQL `ASTGraphBackend` | Pure-Python JSON call/lock graph, 3-of-3 vs. 2-of-3 decoy, `DeterministicBoardEvaluator` | `uv run python examples/codebase_audit/run.py` |
| [`incident_rca/`](incident_rca/README.md) | Tabular Parquet via `DuckDBBackend` | Software telemetry 3-way join benchmark + headless Agg diagnostic plots & charts | `uv run python -m examples.incident_rca.generate_charts` |
| [`exoplanet_triage/`](exoplanet_triage/README.md) | Tabular + 11×11 CCD stamps via `DuckDBBackend` | NASA TESS photometry + CCD centroid vetting + local Observatory Web UI | `uv run python examples/exoplanet_triage/observatory_server.py` |
| [`nyc_cameras/`](nyc_cameras/README.md) | Multimodal live JPEGs + `gemini-4-hydrogen` | 2-round spatial sweep + zoom-in over NYC DOT cameras + FastAPI UI | `uv run uvicorn examples.nyc_cameras.app:app --port 8765` |
| [`audio_haystack/`](audio_haystack/README.md) | Audio WAVs + DuckDB acoustic table | 100k-card audio haystack search race, spectrogram rendering + FastAPI UI | `uv run uvicorn examples.audio_haystack.app:app --port 8080` |
| [`external_evaluator/`](external_evaluator/README.md) | Optional `BoardEvaluator` plug-in | Shows how to plug `DecisionProbeClient` (`gemini-decision-1`) outside core `ares` | `uv run python examples/external_evaluator/run_example.py --mock` |
| [`home_inventory_ledger/`](home_inventory_ledger/README.md) | Standalone multimodal application | Independent contributor project with its own `pyproject.toml` and `LICENSE` | See `examples/home_inventory_ledger/README.md` |
