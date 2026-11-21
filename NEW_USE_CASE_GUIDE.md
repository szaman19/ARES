# ARES New Use Case Guide: Backend-Agnostic Reuse & Extension Playbook

> **Core Mantra:** *The agents never read the haystack; they write the magnets.*
> ARES is **strictly modality-, backend-, and domain-agnostic**. At its narrow waist, ARES provides a parallel task DAG (`solve(graph=...)`), deterministic `CERTIFIED` vs. `UNCERTIFIED` taint tracking, content-hash deduplication, and an immutable provenance store (`NoteStore`), paired with a generic `InvestigationBoard` fusion engine and counterfactual Value-of-Information (VoI) decision probe.
>
> SQL over Parquet (`adapters/duckdb_backend.py`) is simply **one optional `ExecutionBackend` adapter** alongside pure-Python verifiers (`backend=None`), AST/codebase analyzers, chemistry/physics engines, or formal SAT/SMT solvers.

---

## 1. When Is a Task a High-Leverage Fit for ARES?

ARES outperforms a single tool-using LLM whenever a problem has four structural properties—regardless of modality:

1. **The Haystack Exceeds Context (or Raw Inspection Is Noisy):** Candidate spaces live in large codebases, molecular/materials libraries, simulation parameter spaces, APIs, or multi-GB datasets.
2. **Branching Multi-Clue Hypotheses:** Confirming a true positive requires aligning **multiple independent constraints** (e.g., static call-graph reachability + lock-order cycle + deterministic race repro) rather than one obvious lookup.
3. **Plausible Look-Alike Decoys & Incomplete Evidence:** High-salience candidates satisfy 1 or 2 signals while failing a 3rd structural check—or the 3rd signal is genuinely unavailable and ARES must terminate honestly as `UNCERTIFIED` (`"none_verified"`).
4. **Calibrated Closed-Set Decisions:** Downstream action requires knowing *which facts are deterministically `CERTIFIED` vs. `UNCERTIFIED`*, *which evidence fields drive the conclusion* (`influence`), and *which unknown is highest value to test next* (`voi`).

---

## 2. Clean Four-Layer Architecture & Reuse Map

Every file outside `domain/` and `adapters/` is **100% free of SQL, DuckDB, Parquet, and domain-specific strings** (enforced continuously by `test_strict_library_and_implementation_boundaries` in [`tests/test_ares_verifiers.py`](tests/test_ares_verifiers.py)):

| Layer | Modules | What You Reuse Untouched | Boundary Rule |
| :--- | :--- | :--- | :--- |
| **1. Narrow-Waist Contracts** | [`contracts.py`](contracts.py) | `Note`, `ArtifactRef`, `HypothesisSpec`, `CostEstimate`, `GuardValidationError`, `VerifierResult`, `VerifierRegistry`, `InMemoryVerifierRegistry`, `ExecutionBackend`, `InvestigationBoard`, `DecisionQuestion`, `Domain` | **100% Reuse.** Zero SQL or domain assumptions. |
| **2. Core ARES Engine** | [`ares/engine.py`](ares/engine.py), [`ares/store.py`](ares/store.py) | `solve()`, `ARESRunner`, `TaskGraph`, `TaskSpec` (`"propose"`, `"query"`, `"verify"`, `"custom"`), `ProcessorResult`, `NoteStore`, `canonical_content_hash()` | **100% Reuse (Do Not Edit).** Natively runs standalone pure-Python `verify` tasks when `backend=None` or delegates to any `ExecutionBackend`. |
| **3. Pluggable Execution Adapters** | [`adapters/duckdb_backend.py`](adapters/duckdb_backend.py), [`adapters/sql_guard.py`](adapters/sql_guard.py), or custom `adapters/<modality>.py` | `DuckDBBackend` + `SQLGuardConfig` for tabular Parquet; implement `ExecutionBackend` (5 methods) for custom query languages, or use `backend=None` for verifier/processor-only tasks. | **Pluggable per Modality.** Never import adapter modules inside `ares/`, `fusion/`, or `models/`. |
| **4. Generic Fusion & Decision Probe** | [`fusion/patch.py`](fusion/patch.py), [`fusion/loop.py`](fusion/loop.py), [`models/gemini_provider.py`](models/gemini_provider.py), [`models/decision_probe.py`](models/decision_probe.py) | `run_investigation_loop()`, `validate_and_merge_patch()`, `rebuild_board_from_notes()`, `GeminiModelProvider` (including `run_single_agent_baseline()`), `DecisionProbeClient` | **100% Reuse (Do Not Edit).** Parameterized dynamically via your `Domain` plug-in. |

---

## 3. Three Implementation Tracks for New Use Cases

### Track A: Zero-Backend Pure-Python Verifiers (`backend=None`)
When hypotheses are tested directly by deterministic Python functions, compilers, `pytest`, RDKit, or formal solvers, you don't even need an `ExecutionBackend`—`ares.solve(..., backend=None)` executes `task_type="verify"` natively with Invariant I4 context isolation, Invariant I2 taint checking, and Invariant I6 content-hash deduplication:

```python
from typing import Any
from ares import NoteStore, TaskGraph, TaskSpec, solve
from contracts import InMemoryVerifierRegistry, VerifierResult


def verify_conformer_strain(params: dict[str, Any]) -> VerifierResult:
    """Deterministic TRUSTED verifier: receives ONLY `params`, never `TaskSpec.context` (I4)."""
    mol_id = params["mol_id"]
    strain_kcal = 2.4  # Replace with RDKit / OpenMM deterministic physics check
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
    backend=None,  # Zero SQL/DuckDB!
    registry=registry,
    note_store=store,
)
```

---

### Track B: Custom Non-SQL `ExecutionBackend` (AST, Graph, Chemistry, Simulator)
If workers also author lightweight exploratory "magnet" expressions (e.g., `tree-sitter`/ripgrep queries over a monorepo, SMARTS substructure filters, circuit parameter sweeps), implement the 5-method `ExecutionBackend` protocol from [`contracts.py`](contracts.py) (see complete working reference in [`tests/test_non_sql_domain.py`](tests/test_non_sql_domain.py)):

1. `validate_query(query) -> str` — Enforce read-only/sandbox safety; raise `GuardValidationError` on violation (`status="rejected_guard"`).
2. `compute_content_hash(query, *, hypothesis_family, verifier_name, params) -> str` — Canonical SHA-256 hash (`canonical_content_hash(...)`).
3. `estimate_cost(query) -> CostEstimate` — Return `CostEstimate(estimated_units=..., estimated_bytes=..., within_budget=...)` so `ARESRunner` automatically rejects out-of-budget probes and runs cheapest magnets first.
4. `execute_query(...) -> tuple[Any, Note]` — Execute exploratory probe and **always** emit `certification="UNCERTIFIED"`.
5. `execute_verifier(...) -> tuple[VerifierResult, Note]` — Run `registry.get(verifier_name)` against the haystack (never passing `TaskSpec.context`) and stamp `CERTIFIED` iff `trusted=True` and `result.passed=True`.

Return your backend from `domain.create_backend(workspace_dir)`, and `run_investigation_loop(domain, workspace_dir)` + `run_single_agent_baseline(...)` work end-to-end with zero changes to `ares/`, `fusion/`, or `models/`!

---

### Track C: Tabular / Columnar Datasets (Reusing `DuckDBBackend`)
When your haystack lives in tabular files (astronomy catalogs, clinical/omics cohorts, financial ticks, manufacturing telemetry, incident logs), return `DuckDBBackend(workspace_dir)` from `domain.create_backend(workspace_dir)`:

1. **Zero-DDL View Discovery:** Place Parquet shards in `<workspace_dir>/<table_name>/*.parquet`. `DuckDBBackend` automatically registers a DuckDB `VIEW` for every valid subdirectory and dynamically picks up newly appended shards.
2. **Built-In SQL Root Guards:** `validate_readonly_sql()` and `estimate_query_cost()` block non-`SELECT` statements, filesystem escapes, and remote URIs, and sort queries cheapest-first via `EXPLAIN`.
3. **Reference Implementation:** Copy the pattern in [`domain/micro_bench.py`](domain/micro_bench.py) and [`domain/artifacts.py`](domain/artifacts.py).

---

## 4. Implementing a Full `Domain` Plug-In (`domain/<use_case>.py`)

To run the full 8-step `run_investigation_loop()` (and `benchmark/run_micro_bench.py`), implement `Domain` ([`contracts.py`](contracts.py)):

- `environment_description() -> str` & `magnet_rules() -> str`: Describe the haystack entities/views and syntax constraints for `gemini-4-hydrogen` / `gemini-4-carbon`.
- `create_backend(workspace_dir) -> ExecutionBackend | None`: Return `DuckDBBackend(workspace_dir)`, your custom `ExecutionBackend`, or `None` (verifier-only mode).
- `pivotal_fields() -> list[str]` & `required_signal_fields() -> tuple[str, ...]`: Declare canonical board fields seeded at `board_version=0`, protected from alias-collapse/eviction, and checked before `certified_success`.
- `register_verifiers(registry)`: Register fast single-signal verifiers (to prune candidates and expose 2-of-3 decoys) plus a compound multi-signal verifier (optionally emitting deterministic `FigureCanvasAgg` PNG `ArtifactRef`s per [`domain/artifacts.py`](domain/artifacts.py)).
- `extract_field_patches_from_note(note) -> dict[str, Any] | None`: Deterministically map verifier `Note` outputs onto canonical board fields (`None` falls back to generic `note_{hypothesis_family}`).
- `decision_questions() -> list[DecisionQuestion]`: Always include `"none_verified"` (or `"insufficient_evidence"`) in candidate options plus a `yes_no` verification gate question (`sufficiently_verified`). Attach optional `instructions` and `option_criteria` directly onto `DecisionQuestion` when domain-specific remediation rules apply; otherwise `DecisionProbeClient` synthesizes symmetric multi-signal rubrics automatically from `required_signal_fields()`.

---

## 5. Non-Negotiable Architectural Invariants (I1–I7)

1. **Strict Layer Hygiene:** `ares/` knows zero SQL, zero DuckDB, and zero `InvestigationBoard`. `contracts.py`, `fusion/`, and `models/` know zero SQL/DuckDB and zero domain-specific field/entity names.
2. **I1 (Strict Provenance):** Every non-`"unknown"` `BoardField` cites $\ge 1$ valid `note_id` and valid `produced_by_verifier=True` `artifact_id`s in `NoteStore`, with certified values deterministically grounded in note evidence.
3. **I2 (Weakest-Link Certification):** Exploratory `query` runs and LLM interpretations are always `UNCERTIFIED`; a `BoardField` is `"certified"` iff **all** cited notes are `"CERTIFIED"`.
4. **I3 (Immutable Source of Truth):** `NoteStore` is append-only; `rebuild_board_from_notes(...)` must match the live patched board (`final_board_drift == 0.0`).
5. **I4 (Context vs. Verifier Isolation):** `TaskSpec.context` is never passed to `TRUSTED` verifiers.
6. **I5 (Staleness Tracking):** `TaskSpec.requested_under` tracks `board_version` at dispatch.
7. **I6 (Root Guards & Content-Hash Dedup):** Every automated magnet passes `validate_query` + `estimate_cost` and deduplicates by content hash (`skipped_duplicate`).
8. **I7 (Bounded Active Board & 1-Board-Per-Call Probing):** Active `board.fields` stay $\le 12$, verifier plots cap at $\le 3$ per round (rendered via thread-safe `FigureCanvasAgg`, never global `matplotlib.pyplot`), and `DecisionProbeClient` evaluates **one board variant per API call** using explicit `"unknown"` field ablation.

---

## 6. Cross-Modality Use-Case Blueprints

| Use Case & Modality | `create_backend()` | True Multi-Clue Needle (3-Way Signal) | Plausible 2-of-3 Decoy | `TRUSTED` Verifiers & Slice Artifact | Closed-Set `DecisionQuestion`s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Monorepo Concurrency / Security Audit** *(Codebase)* | Custom AST/Call-Graph `ExecutionBackend` (see [`tests/test_non_sql_domain.py`](tests/test_non_sql_domain.py)) | Entrypoint reachability + lock-order inversion + deterministic stress/exploit repro test | Unreachable legacy helper or lock warning that fails deterministic race reproduction | Static reachability + lock-cycle check + sandboxed `pytest` verifier; call-graph PNG | `(1)` Buggy module (`none_verified`), `(2)` Patch action (`reorder_locks`, `add_guard`, `escalate`), `(3)` Verified (`yes`/`no`) |
| **2. Lead Molecule / Materials Triage** *(Chemistry / Physics)* | `None` (pure-Python verifiers) or SMARTS descriptor backend | Target pharmacophore match + clean PAINS/pH-7.4 ionization + low-strain 3D conformer (< 5 kcal/mol) | Reactive PAINS hit or high-strain conformer that looks potent in 2D | Deterministic RDKit + ETKDGv3 physics verifier; 2D structure + torsion profile PNG | `(1)` Lead compound (`none_verified`), `(2)` Action (`synthesize`, `redesign`, `reject`), `(3)` Verified (`yes`/`no`) |
| **3. Control / Circuit Fault Isolation** *(Simulation / Solver)* | Parameter-sweep backend | Sensor phase lag + actuator saturation + stability-margin violation under bounded disturbance | Normal step-response overshoot where stability margin holds | Deterministic ODE/SPICE/SMT invariant verifier; Bode/phase-portrait PNG | `(1)` Faulty subsystem, `(2)` Action (`detune_gain`, `swap_sensor`, `hold_safe_mode`), `(3)` Verified (`yes`/`no`) |
| **4. Astronomy Transient or Genomics Cohort Discovery** *(Tabular Parquet)* | `DuckDBBackend(workspace_dir)` | Multi-band flux rise + stationary centroid + clean housekeeping (or case enrichment + cis-eQTL + clean QC) | Satellite glint with centroid drift, or batch artifact failing lane QC | 3-way SQL join verifier + 3-panel lightcurve/locus PNG | `(1)` Candidate object/gene (`none_verified`), `(2)` Follow-up (`spectroscopy` / `wetlab` / `reject`), `(3)` Verified (`yes`/`no`) |

---

## 7. Pre-Flight Verification Checklist

Run `uv run pytest -v` before marking any new use case complete:
- [ ] `test_strict_library_and_implementation_boundaries` passes (`ares/`, `contracts.py`, `fusion/`, and `models/` remain 100% free of domain/backend leaks).
- [ ] Unsafe or out-of-budget magnets raise `GuardValidationError` and record `status="rejected_guard"`.
- [ ] Equivalent magnets/verifiers across rounds short-circuit as `status="skipped_duplicate"`.
- [ ] 3-of-3 needles reach `certified_success`, 2-of-3 decoys are rejected, and under-determined cases terminate as `honest_uncertified` with `final_board_drift == 0.0` and zero false certainty.
