# ARES v2 Roadmap: Automated Reasoning State Machines

> **ARES = Automated Reasoning State Machines.**  
> Pushdown-automaton (`Spawn` / `Done` / `Suspend` / `Fail`) + dependency-DAG execution for parallel reasoning, deterministic verification, and long or broken contexts — without requiring long-context frontier models.

---

## Core Architectural Principles

1. **PDA + Dependency DAG (`ares/machine.py`):**
   - Tasks execute in isolated local frames (`task.input` + `deps` outputs) rather than an ever-growing chat transcript.
   - Any state handler can dynamically push subtasks and an optional continuation (`Spawn(tasks=[...], then=...)`), pop a compact result (`Done(output)`), pause (`Suspend`), or fail/retry (`Fail`).
   - Ready tasks (all `depends_on` `"done"`) execute in parallel via `ThreadPoolExecutor` with optional `dedup_key` deduplication.
2. **O(1) Context & Crash Resilience (`ares/store.py`):**
   - `ArtifactStore` content-addresses large blobs and automatically spills outputs exceeding `budget.max_inline_bytes` to disk/store, passing a tiny `{__ares_ref__: ...}` handle downstream.
   - `CheckpointStore` atomically snapshots `state.json` and appends `events.jsonl` after every wave so runs survive crashes, rate limits, and model handoffs via `StateMachine.resume(run_dir)`.
3. **Strict Boundary Hygiene:**
   - Core `ares/` depends **only** on Python stdlib + `pydantic`.
   - Zero `InvestigationBoard`, `Domain`, `SearchQuestion`, SQL/DuckDB, web servers, or mandatory LLM SDK coupling in core `ares/`.
   - Optional extensions (`ares/ext/llm/`, `ares/ext/git.py`) and domain tools (`duckdb`, `httpx`, `pytest` subprocesses) live strictly outside core `ares`.
4. **SDK-First, Thin CLI (`ares/cli.py`):**
   - `ares run`, `ares resume`, `ares inspect` (`--tree`, `--tail`), and `ares artifact` (`--offset`, `--limit`) wrap the Python SDK cleanly using the standard checkpoint, event stream, and content-addressed artifact store.

---

## Step-by-Step Checklist

### Step 0: Archive & Clean-Slate Branch (Kill Context Taint)
- [x] Tag hackathon baseline as `v0-hackathon-archive` (`835a9d9`).
- [x] Branch `v2/core-state-machine` from `origin/main` (`835a9d9`) and purge legacy hackathon code, docs, and skills.
- [x] Trim `pyproject.toml` to core `pydantic` only with optional `[llm]`, `[examples]`, and `dev` extras.
- [x] Write `PLAN_V2.md` and `docs/SPEC.md`.

### Step 1: Core SDK Spec (`docs/SPEC.md`)
- [x] Define `Task`, `TaskSpec`, `Transition` (`Done`, `Spawn`, `Suspend`, `Fail`), `RunBudget`, `Event`, `RunResult`.
- [x] Specify PDA `Spawn(..., then=...)` waiting/popping semantics, `ArtifactStore` auto-spill, and `CheckpointStore` crash/resume guarantees.

### Step 2: Core SDK & Optional Extensions + Offline Tests
- [x] Implement `ares/types.py` (191 LOC), `ares/store.py` (271 LOC), `ares/machine.py` (1,072 LOC), `ares/cli.py` (398 LOC), and `ares/__init__.py` (41 LOC) — 1,973 LOC total core (including `join="any"` race cancellation, `Usage` token/cost accounting & budgets, `trace` events, `Task.deterministic` / `Task.artifact_refs` / `Transition.metadata` wiring, and `ares inspect --tree --tail` / `ares artifact`).
- [x] Implement optional `ares/ext/llm/` (652 LOC: `LLMClient` protocol, `OpenAICompatClient` via `httpx`, and tracing helpers) and `ares/ext/git.py` (268 LOC: `GitWorktreeSandbox`).
- [x] Pass offline tests across `test_types_and_store.py`, `test_machine_pda.py`, `test_resume_and_suspend.py`, `test_boundary_hygiene.py`, `test_cli.py`, and `test_llm_openai_compat.py`.

### Step 3: Port 4 Flagship Examples (Abstraction Litmus Tests)
- [x] **`examples/long_horizon_small_context/`** (`machine.py`: 791 LOC): Tail-recursive multi-hop chain (`H=4, 8, 12`) proving O(1) bounded prompt size vs. rolling-context overflow, plus mid-chain crash/resume, `Usage` recording, and `--mock` / `--base-url` / `--model` support (`test_example_long_horizon.py`).
- [x] **`examples/coarse_to_fine_search/`** (`machine.py`: 184 LOC): 2-level dynamic `Spawn` coarse-to-fine search using `duckdb` strictly inside the example, separating 3-signal targets from 2-of-3 decoys and honest unverified cases (`test_example_coarse_to_fine.py`).
- [x] **`examples/codegen_verify_repair/`** (`machine.py`: 566 LOC): Parallel competing codegen branches in isolated temp workspaces, deterministic `pytest` subprocess verification, bounded repair loop with compact/spilled failure feedback, optional `join="any"` race mode, `Usage` recording, `--base-url` / `--model` / `--no-mock` CLI flags, and certified artifact selection (`test_example_codegen_repair.py`).
- [x] **`examples/repo_fix/`** (`machine.py`: 536 LOC): Bounded-context multi-file repository search and parallel `GitWorktreeSandbox` repair with `join="any"` sibling cancellation, `Usage` accounting, `--base-url` / `--model` / `--no-mock` CLI flags, and certified unified git diff output (`test_example_repo_fix.py`).

### Step 4: Thin CLI & Clean Documentation
- [x] Verify `ares run`, `ares resume`, `ares inspect` (`--tree`, `--tail`), and `ares artifact` end-to-end against example machines.
- [x] Write clean `README.md` centered strictly on ARES v2 PDA state machines.

### Step 5: Full End-to-End Verification
- [x] Run full `uv run pytest -v`, all 4 example runners, and CLI smoke tests.
