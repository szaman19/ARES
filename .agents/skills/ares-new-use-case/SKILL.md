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

ARES is **100% modality-, backend-, and domain-agnostic** across `ares/contracts.py`, `ares/engine.py`, `ares/store.py`, `ares/context.py`, `ares/loop.py`, and `ares/models/`. SQL over Parquet (`ares/adapters/duckdb_backend.py`) is only **one optional `ExecutionBackend` adapter**—never leak SQL, DuckDB, Parquet, external decision modules, or domain-specific field names into core `ares/`.

Read [docs/new_use_case_guide.md](../../../docs/new_use_case_guide.md) and the non-SQL reference domain [examples/codebase_audit/domain.py](../../../examples/codebase_audit/domain.py) (plus [tests/test_codebase_audit.py](../../../tests/test_codebase_audit.py)) for complete copy-pasteable templates.

---

## Step 1: Pick the Execution Track (Zero Edits to `ares/`)

1. **Track A — Pure-Python Verifiers (`backend=None`):**
   - For tasks verified directly by Python/CLI/RDKit/pytest/SMT checks without an ad-hoc query language, return `None` from `domain.create_backend(workspace_dir)` (or call `ares.solve(..., backend=None, registry=..., note_store=...)`).
   - `ARESRunner` natively executes `TaskSpec(task_type="verify")` against `VerifierRegistry` (`fn(params) -> VerifierResult`) with Invariant I4 isolation, Invariant I2 taint checking, and Invariant I6 content-hash deduplication.
2. **Track B — Custom Non-SQL `ExecutionBackend` (`examples/<use_case>/domain.py`):**
   - For codebases/AST, graphs, simulators, or custom DSLs, implement `ExecutionBackend` from [ares/contracts.py](../../../ares/contracts.py) (`validate_query`, `compute_content_hash`, `estimate_cost`, `execute_query`, `execute_verifier`) and return it from `domain.create_backend(workspace_dir)`.
3. **Track C — Tabular Parquet (`DuckDBBackend`):**
   - Place Parquet shards under `<workspace_dir>/<valid_identifier>/*.parquet` and return `DuckDBBackend(workspace_dir)` from `ares.adapters` in `domain.create_backend(workspace_dir)`. Views auto-register with zero DDL.

---

## Step 2: Implement the `Domain` Plug-In (`examples/<use_case>/domain.py`)

Implement the `Domain` protocol from [ares/contracts.py](../../../ares/contracts.py) while keeping all domain terms strictly inside `examples/<use_case>/`:

1. **Prompts & Guard Rules:** Implement `environment_description()` and `magnet_rules()` so `GeminiModelProvider` (and `run_single_agent_baseline()`) adapts automatically to your modality.
2. **Canonical Fields & Automatic Context Bounds:** Implement `pivotal_fields()` and `required_signal_fields()` plus `extract_field_patches_from_note(note)` so `ares/context.py` and `ares/loop.py` seed, ground, protect (from alias collapse and stale-field archiving), and audit your board fields with zero hardcoded strings in `ares/`.
3. **Verifiers & Deterministic Slice Artifacts:**
   - Register single-signal verifiers (fast pruning + exposing 2-of-3 decoys) and a compound multi-signal verifier (`trusted=True`, zero LLM calls, never reading `TaskSpec.context`).
   - Optional visual slices: use thread-safe `matplotlib.figure.Figure` + `FigureCanvasAgg` (never global `pyplot`), `metadata={"Software": "ARES"}` for reproducible `sha256`, cap `<= 3` candidates per round, and set `produced_by_verifier=True` (see [examples/incident_rca/artifacts.py](../../../examples/incident_rca/artifacts.py)).
4. **Closed-Set `SearchQuestion`s & Evaluation:**
   - Implement `search_questions() -> list[SearchQuestion]`. Always include `"none_verified"` (or `"insufficient_evidence"`) in candidate `options` plus a `yes_no` verification gate question (`sufficiently_verified`).
   - Optionally implement `resolve_answers(board, questions, primary_target) -> dict[str, str]` on your `Domain` so ARES's built-in `DeterministicBoardEvaluator` resolves secondary remediation/action questions offline without any external decision API.
   - If you need an external or custom evaluator, implement `BoardEvaluator` outside `ares/` (see [examples/external_evaluator/](../../../examples/external_evaluator/README.md)) and pass `evaluator=...` to `run_investigation_loop()`.

---

## Step 3: Validate Boundary Hygiene & End-to-End Loop

Run `uv run pytest -v` and confirm:
1. `test_strict_library_and_implementation_boundaries` in [tests/test_engine.py](../../../tests/test_engine.py) passes (`ares/` remains free of modality, SQL, external decision-module, or domain leaks).
2. Guard violations raise `GuardValidationError` (`status="rejected_guard"`) and duplicate magnets/verifiers short-circuit (`status="skipped_duplicate"`).
3. Stale low-priority distractor fields archive automatically via `archive_stale_fields()` without disturbing protected pivotal fields.
4. Multi-clue cases reach `certified_success`, 2-of-3 look-alike decoys are avoided, and under-determined cases terminate as `honest_uncertified` with `final_board_drift == 0.0` and zero false certainty.
