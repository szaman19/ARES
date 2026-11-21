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
- `home_inventory_ledger/` — **Multimodal Home Inventory & Coverage Adequacy Agent** (`gemini-4-carbon` video/audio/policy understanding + parallel `gemini-4-hydrogen` item/label extractors + live Shopify `https://mcp.shop.com` pricing & replacement cart + 9 deterministic verifiers + `gemini-decision-1` coverage verdict & counterfactual Value-of-Information re-capture loop + live 5 Sigma evaluation on unseen open-licensed videos). See [`home_inventory_ledger/README.md`](home_inventory_ledger/README.md).

## Hackathon Build Timeline (2026-11-21)

- **ARES Core & Scientific/Live Domains (`fd2e940`–`d628cb9`):** Built today during the Gemini 4 NYC Hackathon (`ares/`, `fusion/`, `models/`, `domain/`, `ui/`).
- **Home Inventory Ledger Multimodal + Shopify Commerce Module (`home_inventory_ledger/`):** Built today during the afternoon sprint (see [`home_inventory_ledger/README.md`](home_inventory_ledger/README.md) for module-by-module breakdown, live 5 Sigma evaluation table, and WebRTC live demo quickstart).

## Quickstart

```bash
uv sync
uv run pytest -v
```
