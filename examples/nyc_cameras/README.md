# NYC Live Street-Camera Search Example

Multimodal live-camera search built on ARES's parallel `ARESRunner`, `NoteStore`, `ares.context` provenance validation, `gemini-4-hydrogen` vision inspection, and `DeterministicCameraEvaluator` (with zero decision-probe coupling):

1. **Coarse Parallel Spatial Sweep**: Samples `k` spatially spread street intersections in parallel across ~964 online NYC DOT cameras.
2. **Spatial Zoom-In**: Picks the highest-signal Round-1 intersection and dispatches nearest-neighbor camera checks in parallel.
3. **Context Management (`ares.context`)**: Validates grounded camera observations, SHA-256 JPEG receipts (`ArtifactRef`), and bounds the active `InvestigationBoard`.
4. **Pluggable Evaluation**: Uses `DeterministicCameraEvaluator` by default (or any `BoardEvaluator`).

## Commands

```bash
# Run CLI search (requires GEMINI_API_KEY)
uv run python examples/nyc_cameras/run.py --area Manhattan --snapshot-dir examples/nyc_cameras/fixtures/nyc_snapshot

# Launch interactive FastAPI + Leaflet web UI (requires GEMINI_API_KEY)
uv run uvicorn examples.nyc_cameras.app:app --host 127.0.0.1 --port 8765
```
