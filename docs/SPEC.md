# ARES v2 Core SDK Specification

**ARES** is a Pushdown Automaton (PDA) + Dependency DAG execution engine for automated reasoning over long and broken contexts.

## 1. Strict Boundaries

- Core `ares` (`ares/types.py`, `ares/store.py`, `ares/machine.py`, `ares/cli.py`, `ares/__init__.py`) depends **only** on Python 3.11+ stdlib and `pydantic>=2.9.0`.
- Core `ares` MUST NOT import `ares.llm`, `duckdb`, `sql`, `httpx`, `openai`, `google.genai`, `fastapi`, `matplotlib`, `numpy`, `pyarrow`, or `uvicorn`.
- Core `ares` has no concept of `InvestigationBoard`, `Domain`, `SearchQuestion`, or query DSLs. Domain tools live inside state handlers or optional external packages.

## 2. Data Contracts (`ares/types.py`)

- **`JoinMode`**: `"all" | "any"`
- **`TaskStatus`**: `"pending" | "running" | "waiting" | "done" | "failed" | "suspended" | "cancelled"`
- **`Usage`**: `prompt_tokens`, `completion_tokens`, `total_tokens`, `llm_calls`, `cost_usd` (with `.add()` / `+` helper).
- **`ArtifactRef`** *(frozen)*: `ref_id`, `sha256`, `byte_size`, `mime_type`, `preview`, `path`
- **`TaskSpec`**: `id: str | None = None`, `state: str`, `input: dict[str, Any] = {}`, `depends_on: list[str] = []`, `dedup_key: str | None = None`, `metadata: dict[str, Any] = {}`
- **`Task`**: runtime frame including `id`, `state`, `input`, `depends_on`, `parent_id`, `depth`, `status`, `output`, `error`, `attempts`, `dedup_key`, `join_mode`, `continuation_id`, `wait_for`, `suspend_reason`, `resume_key`, `complete_on_resume`, `resume_data`, `spilled_artifact_id`, `usage`, `metadata`, `duration_ms`.
- **`Transition`** *(discriminated union)*:
  - `Done(output=None, artifacts=[], usage=None, metadata={})` — Pop current task as `"done"`.
  - `Spawn(tasks=[], then=None, join="all", metadata={})` — Push child tasks and optional continuation `then`.
  - `Suspend(reason, resume_key=None, complete_on_resume=False, metadata={})` — Pause task cleanly.
  - `Fail(error, retryable=False, metadata={})` — Fail or retry up to `max_retries`.
- **`RunBudget`**: `max_tasks=500`, `max_depth=32`, `max_wall_clock_s=300.0`, `max_inline_bytes=4096`, `max_tokens=None`, `max_cost_usd=None`.
- **`Event`**: `seq`, `timestamp`, `event_type` (includes `"task_cancelled"`), `task_id`, `payload`.
- **`RunResult`**: `run_id`, `status` (`"done" | "suspended" | "failed" | "budget_exceeded"`), `root_task_id`, `output`, `tasks`, `suspended_task_ids`, `failed_task_ids`, `cancelled_task_ids`, `usage`, `run_dir`, `wall_clock_ms`.

## 3. PDA `Spawn`, Tail-Call Depth & Race Semantics (`ares/machine.py`)

Handler signature:
```python
def handler(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Transition: ...
```
- `deps` maps each `dep_id` in `task.depends_on` to `tasks[dep_id].output` (auto-spilled handles when large).
- Handlers can record incremental LLM telemetry via `ctx.record_usage(prompt_tokens=..., completion_tokens=..., llm_calls=..., cost_usd=...)` (or `Done(usage=...)`), accumulated onto `Task.usage` and `RunResult.usage`.

When task `T` returns `Spawn(tasks=children, then=K, join=...)`:
1. **`children` non-empty, `then=K` present, `join="all"`**:
   - Enqueue `children` with `parent_id=T.id`, `depth=T.depth + 1`. Reject cycles with `ValueError`.
   - Enqueue continuation `K` with `parent_id=T.id`, **at `depth=T.depth`** (continuation replaces `T`'s frame), and `K.depends_on = dedup(K.depends_on + [c.id for c in children])`.
   - `T` transitions to `status="waiting"`, `T.continuation_id = K.id`.
   - When `K` transitions to `"done"`, `T` pops to `status="done"` with `T.output = K.output`.
2. **`children` non-empty, `join="any"` (competitive race)**:
   - Enqueue `children` at `depth=T.depth + 1`, `T.join_mode = "any"`, `T.wait_for = [c.id for c in children]`. If `then=K` is provided, enqueue `K` dormant in `status="waiting"` (`T.continuation_id = K.id`).
   - As soon as any child finishes `"done"`, the first winner cancels all losing sibling subtrees recursively (`status="cancelled"`, emitting `"task_cancelled"`). If `K` is set, `winner.id` is attached to `K.depends_on` and `K` becomes `"pending"`; otherwise `T` pops directly with `winner.output`.
   - If all children fail/cancel, `T` (and dormant `K`) fails/cancels.
3. **`children` non-empty, `then=None`, `join="all"` (fork-join gather)**:
   - Enqueue `children` at `depth=T.depth + 1`, `T.wait_for = [c.id for c in children]`.
   - Once all `wait_for` tasks reach `"done"`, `T` pops to `"done"` with `T.output = {cid: tasks[cid].output for cid in T.wait_for}`.
4. **`children=[]`, `then=K` (O(1) tail-call)**:
   - Enqueue `K` with `parent_id=T.id`, **`depth=T.depth`** (constant stack depth; does not increment against `budget.max_depth`, remains bounded by `budget.max_tasks`).
   - `T` transitions to `status="waiting"`, `T.continuation_id = K.id`, and pops `K.output` on completion.
5. **Failure & cancellation propagation**:
   - Pending tasks whose `depends_on` contain a `"failed"` or `"cancelled"` dependency transition to `"failed"`.
   - Fixed-point `propagate_pops()` unwinds nested failures/cancellations and race winners cleanly after every wave.

## 4. Scheduler, Deduplication & Budgets

- Each wave dequeues all `"pending"` tasks whose `depends_on` are all `"done"`.
- Same-wave and cross-wave `dedup_key` deduplication resolves identical work once (`"task_deduped"`).
- Tasks run concurrently via `ThreadPoolExecutor(max_workers=max_concurrent)`.
- Enforce `budget.max_tasks`, `budget.max_depth` (on child depth expansion), `budget.max_wall_clock_s`, `budget.max_tokens`, and `budget.max_cost_usd`.

## 5. Long & Broken Contexts (`ares/store.py`)

- **`ArtifactStore`**: Content-addressed store (in-memory or `<run_dir>/artifacts/`) with `maybe_spill()` auto-spilling outputs larger than `max_inline_bytes` and `read_slice()` / `get_text()` / `get_json()`.
- **`CheckpointStore`**: Atomic `<run_dir>/state.json` + append-only `<run_dir>/events.jsonl` after every wave; `StateMachine.resume(run_dir, resume_inputs=...)` recovers `"running"` tasks and wakes matching `"suspended"` tasks.
