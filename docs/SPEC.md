# ARES v2 Core SDK Specification

**ARES** is a Pushdown Automaton (PDA) + Dependency DAG execution engine for automated reasoning over long and broken contexts.

## 1. Strict Boundaries

- Core `ares` (`ares/types.py`, `ares/store.py`, `ares/machine.py`, `ares/cli.py`, `ares/__init__.py`) depends **only** on Python 3.11+ stdlib and `pydantic>=2.9.0`.
- Core `ares` MUST NOT import `ares.llm`, `duckdb`, `sql`, `httpx`, `openai`, `google.genai`, `fastapi`, `matplotlib`, `numpy`, `pyarrow`, or `uvicorn`.
- Core `ares` has no concept of `InvestigationBoard`, `Domain`, `SearchQuestion`, or query DSLs. Domain tools live inside state handlers or optional external packages.

## 2. Data Contracts (`ares/types.py`)

- **`TaskStatus`**: `"pending" | "running" | "waiting" | "done" | "failed" | "suspended"`
- **`ArtifactRef`** *(frozen)*: `ref_id`, `sha256`, `byte_size`, `mime_type`, `preview`, `path`
- **`TaskSpec`**: `id: str | None = None`, `state: str`, `input: dict[str, Any] = {}`, `depends_on: list[str] = []`, `dedup_key: str | None = None`, `metadata: dict[str, Any] = {}`
- **`Task`**: runtime frame including `id`, `state`, `input`, `depends_on`, `parent_id`, `depth`, `status`, `output`, `error`, `attempts`, `dedup_key`, `continuation_id`, `wait_for`, `suspend_reason`, `resume_key`, `complete_on_resume`, `resume_data`, `spilled_artifact_id`, `metadata`, `duration_ms`.
- **`Transition`** *(discriminated union)*:
  - `Done(output=None, artifacts=[], metadata={})` — Pop current task as `"done"`.
  - `Spawn(tasks=[], then=None, metadata={})` — Push child tasks and optional continuation `then`.
  - `Suspend(reason, resume_key=None, complete_on_resume=False, metadata={})` — Pause task cleanly.
  - `Fail(error, retryable=False, metadata={})` — Fail or retry up to `max_retries`.
- **`RunBudget`**: `max_tasks=500`, `max_depth=32`, `max_wall_clock_s=300.0`, `max_inline_bytes=4096`.
- **`Event`**: `seq`, `timestamp`, `event_type`, `task_id`, `payload`.
- **`RunResult`**: `run_id`, `status` (`"done" | "suspended" | "failed" | "budget_exceeded"`), `root_task_id`, `output`, `tasks`, `suspended_task_ids`, `failed_task_ids`, `run_dir`, `wall_clock_ms`.

## 3. PDA `Spawn` & Pop Semantics (`ares/machine.py`)

Handler signature:
```python
def handler(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Transition: ...
```
- `deps` maps each `dep_id` in `task.depends_on` to `tasks[dep_id].output` (auto-spilled handles when large).

When task `T` returns `Spawn(tasks=children, then=K)`:
1. **`children` non-empty, `then=K` present**:
   - Enqueue all `children` with `parent_id=T.id`, `depth=T.depth + 1`. Resolve intra-spawn `depends_on` references (`TaskSpec.id` within `children` or existing graph IDs). Reject cycles with `ValueError`.
   - Enqueue `K` with `parent_id=T.id`, `depth=T.depth + 1`, and `K.depends_on = dedup(K.depends_on + [c.id for c in children])`.
   - `T` transitions to `status="waiting"`, `T.continuation_id = K.id`.
   - When `K` transitions to `"done"` (including after nested `Spawn`s), `T` pops to `status="done"` with `T.output = K.output`.
2. **`children` non-empty, `then=None` (fork-join gather)**:
   - Enqueue all `children` with `parent_id=T.id`, `depth=T.depth + 1`.
   - `T` transitions to `status="waiting"`, `T.wait_for = [c.id for c in children]`.
   - Once all `wait_for` tasks reach `"done"`, `T` pops to `status="done"` with `T.output = {cid: tasks[cid].output for cid in T.wait_for}`.
3. **`children=[]`, `then=K` (tail-call)**:
   - Enqueue `K` with `parent_id=T.id`, `depth=T.depth + 1`.
   - `T` transitions to `status="waiting"`, `T.continuation_id = K.id`.
   - When `K` finishes `"done"`, `T` pops to `"done"` with `T.output = K.output`.
4. **`children=[]`, `then=None`**:
   - Invalid; treat as `Done(output=None)`.
5. **Failure propagation**:
   - If `T.continuation_id` or any task in `T.wait_for` transitions to `"failed"`, `T` transitions to `"failed"` (propagating `"failed"` up the parent chain).
6. **Fixed-point pop resolution**:
   - Pop propagation runs to a fixed point after each wave so arbitrarily nested continuations unwind cleanly.

## 4. Scheduler, Deduplication & Budgets

- Each wave dequeues all `"pending"` tasks whose `depends_on` are all `"done"`.
- If two ready tasks in the same wave share a `dedup_key` (or match an already `"done"` task's `dedup_key`), only the first executes; duplicates resolve on the next tick (or immediately if already `"done"`) as `"done"` with `output` copied from the completed task and emit `"task_deduped"`.
- Tasks run concurrently via `ThreadPoolExecutor(max_workers=max_concurrent)`.
- Enforce `budget.max_tasks` (on enqueue), `budget.max_depth` (on spawn), and `budget.max_wall_clock_s`.

## 5. Long & Broken Contexts (`ares/store.py`)

- **`ArtifactStore`**: Content-addressed store (in-memory or `<run_dir>/artifacts/`).
  - `maybe_spill(value, max_inline_bytes)` stores values whose serialized JSON size exceeds `max_inline_bytes` and returns a compact handle `{"__ares_ref__": ref_id, "byte_size": ..., "mime_type": ..., "preview": ...}`.
  - Downstream tasks inspect handles via `ctx.store.get_json(ref_id)`, `get_text(ref_id)`, or `read_slice(ref_id, offset, limit)`.
- **`CheckpointStore`**:
  - Saves `<run_dir>/state.json` atomically (`temp + os.replace`) after each wave and appends `Event`s to `<run_dir>/events.jsonl`.
  - On `StateMachine.resume(run_dir, resume_inputs=...)`:
    - Reloads `state.json` and `ArtifactStore` index; resets any `"running"` tasks (from a mid-flight crash) to `"pending"` without re-running `"done"` tasks.
    - Matches `"suspended"` tasks against `resume_inputs` (by `task.id` or `task.resume_key`): if `complete_on_resume=True`, marks `"done"` with the provided value; otherwise stores `task.resume_data` and marks `"pending"` to re-run.
