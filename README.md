# ARES

> **The agents never read the haystack; they write the magnets.**

ARES is a modality- and backend-agnostic framework that turns complex **search problems** over large haystacks into orchestrated **parallel agent swarms**, verified by **deterministic checkers**, over **automatically managed, provenance-enforced context**.

Instead of stuffing raw repositories, telemetry tables, sky cutouts, or media streams into a single prompt, ARES dispatches parallel worker agents to propose lightweight queries and checks ("magnets") through a Task DAG, runs deterministic verifiers behind strict context isolation, and synthesizes findings into a compact, auditable state board that automatically tracks staleness, prevents ungrounded upgrades, and archives low-priority clutter.

---

## Core Architecture

### 1. Search Problems Over Arbitrary Haystacks
ARES separates **search & evidence gathering** from storage and modality via a narrow waist (`ares.contracts`):
- **Track A — Pure-Python Verifiers (`backend=None`):** Execute deterministic Python/CLI/RDKit/solver verifiers directly via `ares.solve(..., backend=None)` with zero query language setup.
- **Track B — Custom Non-SQL `ExecutionBackend`:** Plug in any domain explorer (AST/call-graph inspector, SMARTS chemistry filter, simulator, or HTTP probe) by implementing `ExecutionBackend` (see [`examples/codebase_audit/`](examples/codebase_audit/README.md)).
- **Track C — Tabular Parquet (`ares.adapters.DuckDBBackend`):** Auto-registers Parquet directories as read-only SQL views protected by `sql_guard.py` and `EXPLAIN`-based cost estimation (not imported by top-level `ares`).

### 2. Swarm & Agent Orchestration (`ares.engine`, `ares.loop`)
- **`TaskGraph`, `TaskSpec`, `ARESRunner`, `solve()`:** Executes heterogeneous `"propose"`, `"query"`, `"verify"`, and `"custom"` tasks with `max_concurrent` worker parallelism, dependency scheduling (`depends_on`), cheapest-first cost ordering, sandbox root guards (`GuardValidationError` -> `status="rejected_guard"`), and SHA-256 content-hash deduplication (`status="skipped_duplicate"`).
- **`run_investigation_loop()` / `run_search_loop()`:** Coordinates multi-round exploration, parallel verification, context synthesis, board evaluation, gap-targeted follow-ups (`propose_followup_tasks`), and honest termination (`certified_success` vs. `honest_uncertified`).

### 3. Automatic Context Management (`ares.context`, `ares.store`)
Agents never accumulate unbounded, unverified scratchpads. ARES maintains two decoupled state layers enforcing Invariants I1–I7:
- **Immutable `NoteStore` (`ares.store`):** Append-only record of every query and verifier output (`CERTIFIED` vs. `UNCERTIFIED`, provenance `sources`, `ArtifactRef` digests, `content_hash`, and `requested_under` board version).
- **Bounded `InvestigationBoard` (`ares.context`):** Managed by `validate_and_merge_patch()`, `deduplicate_fields()`, `archive_stale_fields()`, and `rebuild_board_from_notes()`:
  - **I1 (Strict Provenance & Groundedness):** Every non-`"unknown"` field cites valid `Note` IDs and `produced_by_verifier=True` artifacts; certified values are deterministically verified against cited notes.
  - **I2 (Weakest-Link Certification & Alias Dedup):** Exploratory probes are always `UNCERTIFIED`; a board field is `"certified"` iff all cited notes are `CERTIFIED`, and duplicate alias fields collapse without losing weakest-link taint.
  - **I3 (Zero-Drift Replay):** `rebuild_board_from_notes()` reconstructs the live board from `NoteStore` alone with `board_drift == 0.0`.
  - **I4 (Context-vs-Verifier Isolation):** Proposal `TaskSpec.context` is never passed to `TRUSTED` verifiers.
  - **I5 (Staleness Tracking):** `requested_under` and `stale_rounds` track whether evidence predates newer board versions.
  - **I6 (Root Guards & Content-Hash Dedup):** Every magnet validates sandbox/budget bounds before dispatch and deduplicates by canonical content hash.
  - **I7 (Bounded Active Context):** `archive_stale_fields()` automatically evicts stale, low-priority distractor fields and enforces `max_active_fields` (default `<= 12`) so the active working context stays focused and compact.

### Pluggable Board Evaluation (`BoardEvaluator`)
Core ARES has **no dependency on an external decision module**. `ares.context.DeterministicBoardEvaluator` evaluates required certified signals, computes per-field `priority` and `uncertainty`, populates closed-set `BoardConclusion` answers, and runs completely offline. Any custom scoring backend can plug into `run_investigation_loop(..., evaluator=...)` by implementing `ares.contracts.BoardEvaluator` (see [`examples/external_evaluator/`](examples/external_evaluator/README.md) for an optional counterfactual evaluator kept outside core `ares`).

---

## Repository Layout

| Path | Role |
| :--- | :--- |
| [`ares/`](ares/) | Consolidated core library (`contracts.py`, `engine.py`, `store.py`, `context.py`, `loop.py`), optional `adapters/` (`DuckDBBackend`, `sql_guard.py`), and `models/` (`GeminiModelProvider`) |
| [`examples/`](examples/README.md) | Domain plug-ins, non-SQL & Parquet benchmarks, web UI demos, optional external evaluator, and contributor apps |
| [`docs/`](docs/new_use_case_guide.md) | Architecture and new-domain extension guide |
| [`tests/`](tests/) | Offline-first unit, boundary-hygiene, context-management, loop, and example test suites (`pytest`) |

---

## Quickstart

Install dependencies with [uv](https://docs.astral.sh/uv/) and run the offline test suite:

```bash
uv sync
uv run pytest -v
```

Run the self-contained non-SQL codebase audit example (100% offline, zero API keys required):

```bash
uv run python examples/codebase_audit/run.py
```

Minimal pure-Python verifier orchestration (`backend=None`):

```python
from ares import (
    InMemoryVerifierRegistry,
    NoteStore,
    TaskGraph,
    TaskSpec,
    VerifierResult,
    solve,
)


def verify_invariant(params: dict[str, object]) -> VerifierResult:
    target = str(params["candidate"])
    passed = target == "session_pool"
    return VerifierResult(
        verifier_name="verify_invariant",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidate": target, "passed": passed},
        row_count=1 if passed else 0,
        summary=f"Checked {target}: passed={passed}",
    )


registry = InMemoryVerifierRegistry()
registry.register("verify_invariant", verify_invariant, trusted=True)

report = solve(
    tasks=[
        TaskSpec(
            task_id="verify_session_pool",
            task_type="verify",
            verifier_name="verify_invariant",
            verifier_params={"candidate": "session_pool"},
        )
    ],
    graph=TaskGraph(),
    backend=None,
    registry=registry,
    note_store=NoteStore(),
)
print(report.usage.certified_notes, report.new_notes[0].summary)
```

---

## Examples

| Example | Modality / Pattern | Highlights |
| :--- | :--- | :--- |
| [`examples/codebase_audit/`](examples/codebase_audit/README.md) | Custom non-SQL `ASTGraphBackend` | Offline concurrency/call-graph audit distinguishing 3-of-3 true positives from 2-of-3 decoys |
| [`examples/incident_rca/`](examples/incident_rca/README.md) | Tabular Parquet via `DuckDBBackend` | Multi-clue incident root-cause benchmark + headless deterministic PNG comparison & calibration plots |
| [`examples/exoplanet_triage/`](examples/exoplanet_triage/README.md) | Tabular + 11×11 CCD stamps via `DuckDBBackend` | NASA TESS photometry & centroid vetting + local Observatory Web UI |
| [`examples/nyc_cameras/`](examples/nyc_cameras/README.md) | Multimodal live images + `gemini-4-hydrogen` | 2-round spatial sweep and zoom-in over NYC traffic camera feeds + FastAPI UI |
| [`examples/audio_haystack/`](examples/audio_haystack/README.md) | Audio WAVs + DuckDB acoustic table | 100,000-card audio haystack pruning, spectrogram rendering, and FastAPI UI |
| [`examples/external_evaluator/`](examples/external_evaluator/README.md) | Optional `BoardEvaluator` plug-in | Isolated example implementing `BoardEvaluator` via an external counterfactual probing API |
| [`examples/home_inventory_ledger/`](examples/home_inventory_ledger/README.md) | Standalone contributor application | Independent multimodal home-inventory & coverage project with its own `pyproject.toml` and `LICENSE` |

See [`docs/new_use_case_guide.md`](docs/new_use_case_guide.md) to build your own `Domain` plug-in without modifying `ares/`.

---

## License & Attribution

Core ARES is licensed under the MIT License (`Copyright (c) 2026 Shehtab Zaman` — see [`LICENSE`](LICENSE)).

Contributor additions under `examples/` may retain their own separate license and/or copyright notices; where present (such as [`examples/home_inventory_ledger/LICENSE`](examples/home_inventory_ledger/LICENSE)), those directory-level license terms apply to that addition.
