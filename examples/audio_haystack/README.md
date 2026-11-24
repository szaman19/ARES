# ARES Audio Haystack ("Is This Your Card?")

DuckDB acoustic-feature pruning + parallel WAV inspection across up to 100,000 synthetic/recorded audio cards.

## Run

Standalone FastAPI UI:

```bash
uv run uvicorn examples.audio_haystack.app:app --host 0.0.0.0 --port 8080
```

Combined NYC Cameras + Audio Haystack UI:

```bash
uv run uvicorn examples.audio_haystack.combined_app:app --host 0.0.0.0 --port 8080
```
