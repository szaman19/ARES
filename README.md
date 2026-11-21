# ARES: Parallel Hypothesis Swarm, Deterministic Certification & Counterfactual Decision Probing

> **The agents never read the haystack; they write the magnets.**

ARES is a modality- and backend-agnostic architecture for parallel hypothesis exploration (`gemini-4-hydrogen`), deterministic `TRUSTED` verification, provenance-enforced investigation board fusion (`gemini-4-carbon`), and counterfactual reliance + Value-of-Information (VoI) probing (`gemini-decision-1`).

## Repository Layout

- `contracts.py` — Narrow-waist data contracts & protocols (`InvestigationBoard`, `BoardField`, `Note`, `ArtifactRef`, `HypothesisSpec`, `ExecutionBackend`, `Domain`, `VerifierRegistry`).
- `ares/` — Backend- and domain-agnostic core library (`solve(graph=...)`, `ARESRunner`, `TaskGraph`, `NoteStore`, content-hash deduplication, taint-checked standalone verifier execution).
- `adapters/` — Pluggable magnet & verifier execution backends (`DuckDBBackend` + `sql_guard.py` for partitioned Parquet views).
- `fusion/` & `models/` — Domain-agnostic `InvestigationBoard` invariant enforcement (I1–I7), 8-step investigation loop (`run_investigation_loop`), `GeminiModelProvider`, and `DecisionProbeClient`.
- `domain/` & `benchmark/` — Domain plug-ins and benchmark runners (`MicroBenchDomain` 9-case incident root-cause benchmark + deterministic 3-panel PNG slice plots).
- `NEW_USE_CASE_GUIDE.md` & `.agents/skills/ares-new-use-case/` — Agentic playbook and workspace skill for onboarding new tasks and modalities with zero core library edits.
- `tests/` — Unit, boundary-hygiene, non-SQL domain, and end-to-end loop test suites (`pytest`).

## Quickstart

```bash
uv sync
uv run pytest -v
```
