# ARES

**Automated Reasoning State Machines** — an execution runtime for **Propose → Verify → Refine** scientific reasoning pipelines designed to work with small local models and provide full step-by-step visibility.

---

## What It Is Suited For

- **Propose → Verify → Refine loops:** Iteratively proposing and refining candidate solvers, models, or code against external deterministic checks.
- **Coarse-to-fine screening:** Narrowing large paper corpora or multi-table datasets to top candidates, verifying them blindly, and returning `unverified` when evidence is incomplete.
- **Long or human-gated workflows:** Pipelines that must survive crashes, track token/cost usage, or pause for scientist sign-off and new data without restarting from scratch.

---

## Approach (At a Glance)

- **Isolated local frames:** Each step receives only its direct inputs rather than an ever-growing chat transcript, keeping prompt size bounded.
- **Out-of-line data:** Large datasets, logs, and documents stay on disk; downstream steps inspect bounded slices on demand.
- **Blind verification:** External tools and test suites evaluate candidates without seeing the generator's prose.
- **Parallel branching & checkpoints:** Competing branches can run concurrently with early stopping on first verified success, wave-level checkpointing, and pause/resume.

For internal state-machine semantics, data contracts, and boundary rules, see [`docs/SPEC.md`](docs/SPEC.md).

---

## Installation

```bash
pip install ares-sdk
# With optional OpenAI-compatible LLM client (httpx)
pip install "ares-sdk[llm]"

# Or with uv
uv add ares-sdk
```

---

## Quickstart

For local development from this repository:

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
    return Done(output=x * x)

@m.state("sum_all", deterministic=True)
def sum_all(task: Task, deps: dict, ctx: TaskContext) -> Transition:
    return Done(output=sum(deps.values()))

res = m.run(TaskSpec(state="plan", input={"items": [2, 3, 4]}), max_concurrent=3)
print(res.status, res.output)  # done 29
```

---

## CLI & Visualization

```bash
# Run a state machine
uv run ares run examples.codegen_verify_repair.machine:build_machine \
  --state plan --input '{"task": "slugify"}' --run-dir /tmp/ares_run

# Inspect DAG tree, statuses, artifacts, and recent events
uv run ares inspect /tmp/ares_run --tree

# Read a stored artifact or slice
uv run ares artifact /tmp/ares_run art_<id> --offset 0 --limit 1024

# Resume a crashed or paused run
uv run ares resume /tmp/ares_run \
  --machine examples.codegen_verify_repair.machine:build_machine

# Export a checkpointed run to an interactive single-file HTML trace viewer
uv run python -m ares.ext.viz /tmp/ares_run -o /tmp/trace.html
```

---

## Examples

| Example | Focus | Run |
| :--- | :--- | :--- |
| [`examples/long_horizon_small_context/`](examples/long_horizon_small_context/run.py) | Bounded-context multi-hop chains vs. rolling-transcript overflow; crash/resume + `--base-url` | `uv run python -m examples.long_horizon_small_context.run` |
| [`examples/coarse_to_fine_search/`](examples/coarse_to_fine_search/run.py) | 2-level coarse-to-fine DuckDB screening; separates 3-signal targets from 2-of-3 decoys | `uv run python -m examples.coarse_to_fine_search.run` |
| [`examples/codegen_verify_repair/`](examples/codegen_verify_repair/run.py) | Parallel codegen, isolated `pytest` verification, and bounded refinement over failure slices | `uv run python -m examples.codegen_verify_repair.run` |
| [`examples/repo_fix/`](examples/repo_fix/run.py) | Multi-file repository search and parallel git-worktree patching producing a verified diff | `uv run python -m examples.repo_fix.run` |

---

## License

MIT — see [`LICENSE`](LICENSE).
