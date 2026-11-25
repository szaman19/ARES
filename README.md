# ARES

> **Automated Reasoning State Machines** — Pushdown-automaton (`Spawn` / `Done` / `Suspend` / `Fail`) + dependency-DAG orchestration for parallel reasoning, deterministic verification, and long or broken contexts.

ARES lets small- or bounded-context models and deterministic tools solve deep, branching tasks without stuffing an ever-growing transcript into a single prompt. Every state executes in an isolated local frame, pushes child tasks or continuations dynamically, spills large payloads to a content-addressed `ArtifactStore`, and checkpoints atomically so runs survive crashes and handoffs.

---

## Core Concepts

1. **PDA + Dependency DAG (`ares.machine.StateMachine`)**
   - Each state handler `(task, deps, ctx) -> Transition` sees only its local `task.input` and the compact outputs of `task.depends_on`.
   - **`Done(output, usage=None)`** completes the task and unblocks dependents.
   - **`Spawn(tasks=[...], then=Continuation | None, join="all" | "any")`** pushes parallel child tasks (optionally with inter-task `depends_on`) and pauses the parent in `"waiting"` until `then` (or all children) pops a result back. Tail calls (`Spawn(tasks=[], then=...)`) preserve stack depth `O(1)`, and `join="any"` races competing branches—cancelling slower siblings as soon as a winner completes.
   - **`Suspend(reason, resume_key=...)`** pauses cleanly for external input or manual intervention.
   - **`Fail(error, retryable=...)`** supports bounded retries (`max_retries`) and clean failure propagation.
2. **O(1) Bounded Context & Budgets (`ares.store.ArtifactStore`, `ares.types.RunBudget`)**
   - Outputs larger than `RunBudget.max_inline_bytes` automatically spill to content-addressed storage (`__ares_ref__` handle + short preview). Downstream states inspect slices on demand via `ctx.store.read_slice(...)` or `get_text(...)` / `get_json(...)`.
   - Handlers record token/call/cost telemetry via `ctx.record_usage(...)` against optional `RunBudget(max_tokens=..., max_cost_usd=...)`.
3. **Broken-Context Crash & Resume (`ares.store.CheckpointStore`)**
   - When `run_dir` is provided, every wave atomically checkpoints `state.json` and appends `events.jsonl`. `StateMachine.resume(run_dir)` resets interrupted `"running"` tasks to `"pending"` without re-running completed tasks.
4. **Zero Bloat Core**
   - Core `ares` depends only on `pydantic>=2.9.0`. Reusable optional extensions live in `ares.ext` (`ares.ext.llm.OpenAICompatClient`, `ares.ext.git.GitWorktreeSandbox`) and are never imported by core `ares`, while domain tools (e.g., `duckdb`, `pytest` subprocesses) live strictly inside callers/examples.

---

## Quickstart

```bash
uv sync --all-groups
uv run pytest -v
```

### Minimal Python Example

```python
from ares import Done, Spawn, StateMachine, Task, TaskContext, TaskSpec, Transition

m = StateMachine()

@m.state("plan")
def plan(task: Task, deps: dict, ctx: TaskContext) -> Transition:
    items = task.input["items"]
    return Spawn(
        tasks=[TaskSpec(state="square", input={"x": x}) for x in items],
        then=TaskSpec(state="sum_all"),
    )

@m.state("square", deterministic=True)
def square(task: Task, deps: dict, ctx: TaskContext) -> Transition:
    x = task.input["x"]
    return Done(x * x)

@m.state("sum_all", deterministic=True)
def sum_all(task: Task, deps: dict, ctx: TaskContext) -> Transition:
    return Done(sum(deps.values()))

res = m.run(TaskSpec(state="plan", input={"items": [2, 3, 4]}), max_concurrent=3)
print(res.status, res.output)  # done 29
```

---

## CLI

```bash
# Run a state machine from module:attr (instance or zero-arg factory)
uv run ares run examples.codegen_verify_repair.machine:build_machine \
  --state plan --input '{"task": "slugify"}' --run-dir /tmp/ares_run

# Inspect task DAG (--tree for indented parent-child view), statuses, artifacts, and usage
uv run ares inspect /tmp/ares_run --tree

# Print a stored artifact or slice by ref_id
uv run ares artifact /tmp/ares_run art_<id> --offset 0 --limit 1024

# Resume a crashed or suspended run
uv run ares resume /tmp/ares_run \
  --machine examples.codegen_verify_repair.machine:build_machine
```

---

## Examples

| Example | What It Demonstrates | Run |
| :--- | :--- | :--- |
| [`examples/long_horizon_small_context/`](examples/long_horizon_small_context/run.py) | Tail-recursive `H`-hop chains (`H=4, 8, 12`) keeping O(1) prompt size while a rolling-context baseline overflows; crash/resume + optional `OpenAICompatClient` (`--base-url`) | `uv run python -m examples.long_horizon_small_context.run` |
| [`examples/coarse_to_fine_search/`](examples/coarse_to_fine_search/run.py) | 2-level dynamic `Spawn` coarse-to-fine search using `duckdb` strictly inside the example; separates 3-signal targets from 2-of-3 decoys | `uv run python -m examples.coarse_to_fine_search.run` |
| [`examples/codegen_verify_repair/`](examples/codegen_verify_repair/run.py) | Parallel competing codegen branches (with optional `race=True` `join="any"` cancellation), isolated `pytest` subprocess verification, and bounded repair loops over spilled failure summaries | `uv run python -m examples.codegen_verify_repair.run` |
| [`examples/repo_fix/`](examples/repo_fix/run.py) | Multi-file bounded repository navigation (`locate` -> parallel `inspect_file`) + parallel `GitWorktreeSandbox` patching/repair (`join="any"`) producing a verified `git diff` artifact | `uv run python -m examples.repo_fix.run` |

---

## License

MIT — see [`LICENSE`](LICENSE).
