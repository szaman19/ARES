# ARES New Use Case Guide: Backend-Agnostic Reuse & Extension Playbook

> **Core Mantra:** *The agents never read the haystack; they write the magnets.*
>
> ARES takes structured or unstructured **search problems**, **stitches together parallel agent swarms and deterministic verifiers** via a Task DAG (`solve(graph=...)`), and performs **automatic context management** over an immutable `NoteStore` and a bounded, provenance-enforced `InvestigationBoard` (`ares.context`).
>
> SQL over Parquet (`ares/adapters/duckdb_backend.py`) is simply **one optional `ExecutionBackend` adapter** alongside pure-Python verifiers (`backend=None`), AST/codebase analyzers, chemistry/physics checks, or formal SAT/SMT solvers. External scoring APIs are likewise optional: core ARES ships with a built-in deterministic evaluator (`DeterministicBoardEvaluator`), and any custom `BoardEvaluator` lives cleanly outside core `ares/`.

---

## 1. When Is a Task a High-Leverage Fit for ARES?

ARES outperforms a single tool-using prompt whenever a search problem has four structural properties—regardless of modality:

1. **The Haystack Exceeds Context (or Raw Inspection Is Noisy):** Candidate spaces live in large codebases, molecular/materials libraries, simulation parameter spaces, live sensor feeds, or multi-GB datasets.
2. **Branching Multi-Clue Hypotheses:** Confirming a true positive requires aligning **multiple independent constraints** (e.g., static call-graph reachability + lock-order cycle + deterministic race repro) rather than one obvious lookup.
3. **Plausible Look-Alike Decoys & Incomplete Evidence:** High-salience candidates satisfy 1 or 2 signals while failing a 3rd structural check—or the 3rd signal is genuinely unavailable and ARES must terminate honestly as `UNCERTIFIED` (`"none_verified"`).
4. **Automatic Context Discipline:** Long-running agent searches accumulate noise; ARES keeps only grounded, provenance-backed facts in a bounded working context, tracks staleness, and archives low-priority clutter automatically via `archive_stale_fields()`.

---

## 2. Consolidated Package Architecture & Reuse Map

Every core module under `ares/` (except the optional `ares/adapters/` package) is **100% free of SQL, DuckDB, Parquet, external decision modules, and domain-specific strings** (enforced continuously by `test_strict_library_and_implementation_boundaries` in [`tests/test_engine.py`](../tests/test_engine.py)):

| Layer | Modules | What You Reuse Untouched | Boundary Rule |
| :--- | :--- | :--- | :--- |
| **1. Narrow-Waist Contracts** | [`ares/contracts.py`](../ares/contracts.py) | `Note`, `ArtifactRef`, `HypothesisSpec`, `CostEstimate`, `GuardValidationError`, `VerifierResult`, `VerifierRegistry`, `InMemoryVerifierRegistry`, `ExecutionBackend`, `InvestigationBoard`, `BoardField`, `BoardConclusion`, `SearchQuestion`, `BoardEvaluationResult`, `BoardEvaluator`, `Domain` | **100% Reuse.** Zero SQL or domain assumptions. |
| **2. Core Task DAG Engine & Store** | [`ares/engine.py`](../ares/engine.py), [`ares/store.py`](../ares/store.py) | `solve()`, `ARESRunner`, `TaskGraph`, `TaskSpec` (`"propose"`, `"query"`, `"verify"`, `"custom"`), `ProcessorResult`, `NoteStore`, `canonical_content_hash()` | **100% Reuse (Do Not Edit).** Runs standalone `verify` tasks when `backend=None` or delegates to any `ExecutionBackend`. Unaware of board state. |
| **3. Pluggable Execution Adapters** | [`ares/adapters/duckdb_backend.py`](../ares/adapters/duckdb_backend.py), [`ares/adapters/sql_guard.py`](../ares/adapters/sql_guard.py), or domain backends | `DuckDBBackend` + `SQLGuardConfig` for tabular Parquet; implement `ExecutionBackend` (5 methods) for custom query languages, or return `None` for verifier-only tasks. | **Pluggable per Modality.** Never import adapter modules inside top-level `ares/__init__.py`, `engine.py`, `context.py`, `loop.py`, or `models/`. |
| **4. Automatic Context Management, Swarm Loop & Models** | [`ares/context.py`](../ares/context.py), [`ares/loop.py`](../ares/loop.py), [`ares/models/gemini.py`](../ares/models/gemini.py) | `run_investigation_loop()` / `run_search_loop()`, `validate_and_merge_patch()`, `deduplicate_fields()`, `archive_stale_fields()`, `rebuild_board_from_notes()`, `DeterministicBoardEvaluator`, `GeminiModelProvider` (including `run_single_agent_baseline()`) | **100% Reuse (Do Not Edit).** Parameterized dynamically via your `Domain` plug-in and optional `BoardEvaluator`. |

---

## 3. Three Implementation Tracks for New Use Cases

### Track A: Zero-Backend Pure-Python Verifiers (`backend=None`)
When hypotheses are tested directly by deterministic Python functions, compilers, `pytest`, RDKit, or formal solvers, you do not need an `ExecutionBackend`—`ares.solve(..., backend=None)` executes `task_type="verify"` natively with Invariant I4 context isolation, Invariant I2 taint checking, and Invariant I6 content-hash deduplication:

```python
from typing import Any
from ares import (
    InMemoryVerifierRegistry,
    NoteStore,
    TaskGraph,
    TaskSpec,
    VerifierResult,
    solve,
)


def verify_conformer_strain(params: dict[str, Any]) -> VerifierResult:
    """Deterministic TRUSTED verifier: receives ONLY `params`, never `TaskSpec.context` (I4)."""
    mol_id = params["mol_id"]
    strain_kcal = 2.4  # Replace with RDKit / OpenMM deterministic check
    passed = strain_kcal < 5.0
    return VerifierResult(
        verifier_name="verify_conformer_strain",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"mol_id": mol_id, "strain_kcal": strain_kcal},
        row_count=1 if passed else 0,
        summary=f"{mol_id} strain={strain_kcal:.1f} kcal/mol",
    )


registry = InMemoryVerifierRegistry()
registry.register("verify_conformer_strain", verify_conformer_strain, trusted=True)
store = NoteStore()
graph = TaskGraph()

report = solve(
    tasks=[
        TaskSpec(
            task_id="check_mol_01",
            task_type="verify",
            verifier_name="verify_conformer_strain",
            verifier_params={"mol_id": "mol_01"},
            requested_under=0,
        )
    ],
    graph=graph,
    backend=None,  # Zero SQL/DuckDB
    registry=registry,
    note_store=store,
)
```

---

### Track B: Custom Non-SQL `ExecutionBackend` (AST, Graph, Chemistry, Simulator)
If workers also author lightweight exploratory "magnet" expressions (e.g., AST/call-graph queries over a codebase, SMARTS substructure filters, circuit parameter sweeps), implement the 5-method `ExecutionBackend` protocol from [`ares/contracts.py`](../ares/contracts.py) (see complete working reference in [`examples/codebase_audit/domain.py`](../examples/codebase_audit/domain.py)):

1. `validate_query(query) -> str` — Enforce read-only/sandbox safety; raise `GuardValidationError` on violation (`status="rejected_guard"`).
2. `compute_content_hash(query, *, hypothesis_family, verifier_name, params) -> str` — Canonical SHA-256 hash (`canonical_content_hash(...)`).
3. `estimate_cost(query) -> CostEstimate` — Return `CostEstimate(estimated_units=..., estimated_bytes=..., within_budget=...)` so `ARESRunner` automatically rejects out-of-budget probes and runs cheapest magnets first.
4. `execute_query(...) -> tuple[Any, Note]` — Execute exploratory probe and **always** emit `certification="UNCERTIFIED"`.
5. `execute_verifier(...) -> tuple[VerifierResult, Note]` — Run `registry.get(verifier_name)` against the haystack (never passing `TaskSpec.context`) and stamp `CERTIFIED` iff `trusted=True` and `result.passed=True`.

Return your backend from `domain.create_backend(workspace_dir)`, and `run_investigation_loop(domain, workspace_dir)` + `run_single_agent_baseline(...)` work end-to-end with zero changes to `ares/`.

---

### Track C: Tabular / Columnar Datasets (Reusing `DuckDBBackend`)
When your haystack lives in tabular files (astronomy catalogs, clinical/omics cohorts, telemetry logs), return `DuckDBBackend(workspace_dir)` from `domain.create_backend(workspace_dir)`:

1. **Zero-DDL View Discovery:** Place Parquet shards in `<workspace_dir>/<table_name>/*.parquet`. `DuckDBBackend` automatically registers a DuckDB `VIEW` for every valid subdirectory and dynamically picks up newly appended shards.
2. **Built-In SQL Root Guards:** `validate_readonly_sql()` and `estimate_query_cost()` in [`ares/adapters/sql_guard.py`](../ares/adapters/sql_guard.py) block non-`SELECT` statements, filesystem escapes, and remote URIs, and sort queries cheapest-first via `EXPLAIN`.
3. **Reference Implementation:** Copy the pattern in [`examples/incident_rca/domain.py`](../examples/incident_rca/domain.py) and [`examples/incident_rca/artifacts.py`](../examples/incident_rca/artifacts.py).

---

## 4. Implementing a Full `Domain` Plug-In (`examples/<use_case>/domain.py`)

To run `run_investigation_loop()` (or `run_search_loop()`), implement `Domain` ([`ares/contracts.py`](../ares/contracts.py)):

- `environment_description() -> str` & `magnet_rules() -> str`: Describe the haystack entities/views and syntax constraints for worker prompts.
- `create_backend(workspace_dir) -> ExecutionBackend | None`: Return `DuckDBBackend(workspace_dir)`, your custom `ExecutionBackend`, or `None` (verifier-only mode).
- `pivotal_fields() -> list[str]` & `required_signal_fields() -> tuple[str, ...]`: Declare canonical board fields seeded at `board_version=0`, protected from alias collapse and automatic archiving, and checked before `certified_success`.
- `register_verifiers(registry)`: Register fast single-signal verifiers (to prune candidates and expose 2-of-3 decoys) plus a compound multi-signal verifier (optionally emitting deterministic `FigureCanvasAgg` PNG `ArtifactRef`s per [`examples/incident_rca/artifacts.py`](../examples/incident_rca/artifacts.py)).
- `extract_field_patches_from_note(note) -> dict[str, Any] | None`: Deterministically map verifier `Note` outputs onto canonical board fields (`None` falls back to generic `note_{hypothesis_family}`).
- `search_questions() -> list[SearchQuestion]`: Closed-set questions the investigation board answers. Always include `"none_verified"` (or `"insufficient_evidence"`) in candidate `options` plus a `yes_no` verification gate question (`sufficiently_verified`).
- Optional `resolve_answers(board, questions, primary_target) -> dict[str, str]`: Hook used by the built-in `DeterministicBoardEvaluator` to map a verified primary target onto secondary follow-up/remediation questions without requiring any LLM or external API call.
- Optional custom `BoardEvaluator`: Pass `evaluator=MyCustomEvaluator()` to `run_investigation_loop(domain, workspace_dir, evaluator=...)` if you want bespoke or external model scoring (see [`examples/external_evaluator/`](../examples/external_evaluator/README.md)) without editing `ares/`.

---

## 5. Non-Negotiable Architectural Invariants (I1–I7)

1. **Strict Layer Hygiene:** `ares/engine.py` and `ares/store.py` know zero SQL, zero DuckDB, and zero `InvestigationBoard`. Core `ares/` (`contracts.py`, `context.py`, `loop.py`, `models/`) knows zero SQL/DuckDB, zero external decision module symbols, and zero domain-specific field/entity names.
2. **I1 (Strict Provenance):** Every non-`"unknown"` `BoardField` cites >= 1 valid `note_id` and valid `produced_by_verifier=True` `artifact_id`s in `NoteStore`, with certified values deterministically grounded in note evidence.
3. **I2 (Weakest-Link Certification):** Exploratory `query` runs and LLM interpretations are always `UNCERTIFIED`; a `BoardField` is `"certified"` iff **all** cited notes are `"CERTIFIED"`.
4. **I3 (Immutable Source of Truth & Zero-Drift Replay):** `NoteStore` is append-only; `rebuild_board_from_notes(...)` must match the live patched board (`final_board_drift == 0.0`).
5. **I4 (Context vs. Verifier Isolation):** `TaskSpec.context` is never passed to `TRUSTED` verifiers.
6. **I5 (Staleness Tracking):** `TaskSpec.requested_under` and `BoardField.stale_rounds` track `board_version` lag across rounds.
7. **I6 (Root Guards & Content-Hash Dedup):** Every automated magnet passes `validate_query` + `estimate_cost` and deduplicates by content hash (`skipped_duplicate`).
8. **I7 (Bounded Active Context & Verifier Slices):** `archive_stale_fields()` keeps active `board.fields` bounded (`max_active_fields <= 12` by default) by archiving stale, low-`priority` non-protected fields; verifier plots cap at `<= 3` per round and use headless `Figure` + `FigureCanvasAgg` (never global `matplotlib.pyplot`).

---

## 6. Cross-Modality Use-Case Blueprints

| Use Case & Modality | `create_backend()` | True Multi-Clue Needle (3-Way Signal) | Plausible 2-of-3 Decoy | `TRUSTED` Verifiers & Slice Artifact | Closed-Set `SearchQuestion`s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Codebase Concurrency Audit** | Custom AST/Call-Graph `ExecutionBackend` ([`examples/codebase_audit/domain.py`](../examples/codebase_audit/domain.py)) | Entrypoint reachability + lock-order inversion + deterministic stress repro test | Unreachable legacy helper or lock warning that fails race reproduction | Static reachability + lock-cycle check + stress repro verifier | `(1)` Buggy module (`none_verified`), `(2)` Patch action, `(3)` Verified (`yes`/`no`) |
| **2. Lead Molecule / Materials Triage** | `None` (pure-Python verifiers) or SMARTS backend | Pharmacophore match + clean pH-7.4 alerts + low-strain 3D conformer | Reactive hit or high-strain conformer looking potent in 2D | Deterministic RDKit + strain physics verifier; 2D structure + profile PNG | `(1)` Lead compound (`none_verified`), `(2)` Action (`synthesize`, `redesign`, `reject`), `(3)` Verified (`yes`/`no`) |
| **3. Control / Circuit Fault Isolation** | Parameter-sweep backend | Sensor phase lag + actuator saturation + stability-margin violation | Normal step-response overshoot where stability margin holds | Deterministic ODE/SPICE/SMT invariant verifier; Bode/phase PNG | `(1)` Faulty subsystem, `(2)` Action (`detune_gain`, `swap_sensor`, `hold_safe_mode`), `(3)` Verified (`yes`/`no`) |
| **4. Telemetry RCA or Astronomy Triage** | `DuckDBBackend(workspace_dir)` ([`examples/incident_rca/`](../examples/incident_rca/README.md), [`examples/exoplanet_triage/`](../examples/exoplanet_triage/README.md)) | 3-way multi-table join or multi-sector lightcurve + clean centroid offset | 2-of-3 look-alike spike or eclipsing binary failing 3rd structural check | 3-way SQL / photometry join verifier + 3-panel Agg PNG | `(1)` Target (`none_verified`), `(2)` Action / follow-up, `(3)` Verified (`yes`/`no`) |

---

## 7. Pre-Flight Verification Checklist

Run `uv run pytest -v` before marking any new use case complete:
- [ ] `test_strict_library_and_implementation_boundaries` in `tests/test_engine.py` passes (`ares/` stays 100% free of domain/backend leaks and external decision-module coupling).
- [ ] Unsafe or out-of-budget magnets raise `GuardValidationError` and record `status="rejected_guard"`.
- [ ] Equivalent magnets/verifiers across rounds short-circuit as `status="skipped_duplicate"`.
- [ ] Automatic context archiving keeps active `board.fields` bounded while protecting `pivotal_fields()` and `required_signal_fields()`.
- [ ] 3-of-3 needles reach `certified_success`, 2-of-3 decoys are avoided, and under-determined cases terminate as `honest_uncertified` with `final_board_drift == 0.0` and zero false certainty.
