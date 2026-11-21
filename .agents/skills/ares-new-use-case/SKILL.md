---
name: ares-new-use-case
description: >-
  Guides the agent to design, scaffold, and verify new domains, benchmarks, or
  real-world tasks on top of the ARES library while maximizing code reuse and
  preserving the strict backend- and domain-agnostic narrow-waist boundary. Use
  whenever the user asks to add a new use case, adapt ARES to a new modality
  (codebases, chemistry, simulations, formal solvers, or tabular Parquet), write
  TRUSTED verifiers or ExecutionBackends, or dispatch parallel agents for a new
  ARES domain.
---

# ARES New Use Case Skill

ARES is **100% modality-, backend-, and domain-agnostic** across `contracts.py`, `ares/`, `fusion/`, and `models/`. SQL over Parquet (`adapters/duckdb_backend.py`) is only **one optional `ExecutionBackend` adapter**—never leak SQL, DuckDB, Parquet, or domain-specific field names into `ares/`, `contracts.py`, `fusion/`, or `models/`.

Read [NEW_USE_CASE_GUIDE.md](../../../NEW_USE_CASE_GUIDE.md) and the non-SQL reference test [test_non_sql_domain.py](../../../tests/test_non_sql_domain.py) for complete copy-pasteable templates.

---

## Step 1: Pick the Execution Track (Zero Edits to `ares/`, `fusion/`, or `models/`)

1. **Track A — Pure-Python Verifiers (`backend=None`):**
   - For tasks verified directly by Python/CLI/RDKit/pytest/SMT checks without an ad-hoc query language, return `None` from `domain.create_backend(workspace_dir)` (or call `ares.solve(..., backend=None, registry=..., note_store=...)`).
   - `ARESRunner` natively executes `TaskSpec(task_type="verify")` against `VerifierRegistry` (`fn(params) -> VerifierResult`) with Invariant I4 isolation, Invariant I2 taint checking, and Invariant I6 content-hash deduplication.
2. **Track B — Custom Non-SQL `ExecutionBackend` (`adapters/<modality>.py`):**
   - For codebases/AST, graphs, simulators, or custom DSLs, implement `ExecutionBackend` from [contracts.py](../../../contracts.py) (`validate_query`, `compute_content_hash`, `estimate_cost`, `execute_query`, `execute_verifier`) and return it from `domain.create_backend(workspace_dir)`.
3. **Track C — Tabular Parquet (`DuckDBBackend`):**
   - Place Parquet shards under `<workspace_dir>/<valid_identifier>/*.parquet` and return `DuckDBBackend(workspace_dir)` from `domain.create_backend(workspace_dir)`. Views auto-register with zero DDL.

---

## Step 2: Implement the `Domain` Plug-In (`domain/<use_case>.py`)

Implement the `Domain` protocol from [contracts.py](../../../contracts.py) while keeping all domain terms strictly inside `domain/<use_case>.py`:

1. **Prompts & Guard Rules:** Implement `environment_description()` and `magnet_rules()` so `GeminiModelProvider` (and `run_single_agent_baseline()`) adapts automatically to your modality.
2. **Canonical Fields:** Implement `pivotal_fields()` and `required_signal_fields()` plus `extract_field_patches_from_note(note)` so `fusion/patch.py` and `fusion/loop.py` seed, ground, protect, and audit your board fields with zero hardcoded strings in `fusion/`.
3. **Verifiers & Deterministic Slice Artifacts:**
   - Register single-signal verifiers (fast pruning + exposing 2-of-3 decoys) and a compound multi-signal verifier (`trusted=True`, zero LLM calls, never reading `TaskSpec.context`).
   - Optional visual slices: use thread-safe `matplotlib.figure.Figure` + `FigureCanvasAgg` (never global `pyplot`), `metadata={"Software": "ARES"}` for reproducible `sha256`, cap $\le 3$ candidates per round, and set `produced_by_verifier=True` (see [domain/artifacts.py](../../../domain/artifacts.py)).
4. **Closed-Set `DecisionQuestion`s:**
   - Always include `"none_verified"` (or `"insufficient_evidence"`) in candidate `options` and a `yes_no` verification gate question (`sufficiently_verified`).
   - Attach optional `instructions` and `option_criteria` directly onto `DecisionQuestion` if custom action rubrics are needed; otherwise `DecisionProbeClient` generates symmetric multi-signal rubrics from `required_signal_fields()`.

---

## Step 3: Validate Boundary Hygiene & End-to-End Loop

Run `uv run pytest -v` and confirm:
1. `test_strict_library_and_implementation_boundaries` passes (`git diff ares/ fusion/ models/ contracts.py` stays free of modality or domain leaks).
2. Guard violations raise `GuardValidationError` (`status="rejected_guard"`) and duplicate magnets/verifiers short-circuit (`status="skipped_duplicate"`).
3. Multi-clue cases reach `certified_success`, 2-of-3 look-alike decoys are avoided, and under-determined cases terminate as `honest_uncertified` with `final_board_drift == 0.0` and zero false certainty.
