# ARES Hackathon: Parallel Hypothesis Swarm, Deterministic Certification & Counterfactual Decision Probing

> **The agents never read the haystack; they write the magnets.**

ARES executes parallel SQL hypothesis generation (`gemini-4-hydrogen`), deterministic `TRUSTED` DuckDB verification over local partitioned Parquet shards, provenance-enforced investigation board fusion (`gemini-4-carbon`), and counterfactual reliance + value-of-information (VoI) probing (`gemini-decision-1`).

## Repository Layout

- `contracts.py` — Narrow-waist data contracts (`InvestigationBoard`, `BoardField`, `Note`, `ArtifactRef`, `HypothesisQuerySpec`, `Domain`, `VerifierRegistry`).
- `ares/` — Domain-agnostic ARES library (`solve(graph=...)`, `TaskGraph`, `NoteStore`, read-only SQL root guards, `EXPLAIN` cost estimation, `DuckDBTrustedRunner`).
- `domain/` — Domain plug-ins (`MicroBenchDomain` 9-case Stage 1 incident benchmark, `TRUSTED` DuckDB verifiers, deterministic 3-panel candidate PNG slice plots).
- `tests/` — Unit and integration test suites (`pytest`).
- `ARES_plan.md` & `ARES_MULTIMODAL_ADDENDUM.md` — Full architecture, invariants (I1–I7), and multimodal perception specification.
- `NEW_USE_CASE_GUIDE.md` — Agentic playbook for building new tasks and domains with 100% reuse of `ares/` and `contracts.py`.

## Quickstart

```bash
uv sync
uv run pytest -v
```
