"""Integration tests for StateMachine PDA, DAG dependencies, parallelism, dedup, and budgets."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ares import (
    ArtifactStore,
    Done,
    Event,
    Fail,
    RunBudget,
    Spawn,
    StateMachine,
    Task,
    TaskContext,
    TaskSpec,
)


def test_nested_three_level_spawn_and_then_unwinding() -> None:
    machine = StateMachine()

    @machine.state("lvl1")
    def lvl1(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[TaskSpec(id="c1", state="lvl2", input={"x": 10})],
            then=TaskSpec(id="k1", state="join_add", input={"bonus": 1}),
        )

    @machine.state("lvl2")
    def lvl2(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[TaskSpec(id="c2", state="lvl3", input={"x": task.input["x"]})],
            then=TaskSpec(id="k2", state="join_add", input={"bonus": 2}),
        )

    @machine.state("lvl3")
    def lvl3(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="leaf_a", state="leaf", input={"v": task.input["x"] + 1}),
                TaskSpec(id="leaf_b", state="leaf", input={"v": task.input["x"] + 2}),
            ],
            then=TaskSpec(id="k3", state="join_add", input={"bonus": 3}),
        )

    @machine.state("leaf")
    def leaf(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=task.input["v"])

    @machine.state("join_add")
    def join_add(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=sum(deps.values()) + task.input["bonus"])

    res = machine.run(TaskSpec(id="root", state="lvl1"))
    assert res.status == "done"
    # k3 = (11 + 12) + 3 = 26 -> c2 pops 26 -> k2 = 26 + 2 = 28 -> c1 pops 28 -> k1 = 28 + 1 = 29
    assert res.output == 29
    assert res.tasks["root"].depth == 0
    assert res.tasks["c1"].depth == 1
    assert res.tasks["c2"].depth == 2
    assert res.tasks["leaf_a"].depth == 3
    assert all(res.tasks[tid].status == "done" for tid in ["root", "c1", "c2", "k1", "k2", "k3"])


def test_tail_call_spawn_empty_tasks_with_then() -> None:
    machine = StateMachine()

    @machine.state("hop")
    def hop(task: Task, deps: dict, ctx: TaskContext):
        remaining = task.input["remaining"]
        acc = task.input["acc"]
        if remaining == 0:
            return Done(output=acc)
        return Spawn(
            tasks=[],
            then=TaskSpec(
                state="hop",
                input={"remaining": remaining - 1, "acc": acc + remaining},
            ),
        )

    res = machine.run(TaskSpec(id="start", state="hop", input={"remaining": 4, "acc": 0}))
    assert res.status == "done"
    assert res.output == 10
    assert len(res.tasks) == 5


def test_fork_join_gather_and_failure_propagation() -> None:
    machine = StateMachine()

    @machine.state("scatter")
    def scatter(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="w1", state="work", input={"val": 3}),
                TaskSpec(id="w2", state="work", input={"val": 4}),
            ],
            then=None,
        )

    @machine.state("work")
    def work(task: Task, deps: dict, ctx: TaskContext) -> Done | Fail:
        if task.input.get("fail"):
            return Fail(error="worker exploded")
        return Done(output=task.input["val"] * 2)

    res = machine.run(TaskSpec(id="root", state="scatter"))
    assert res.status == "done"
    assert res.output == {"w1": 6, "w2": 8}

    # Failure propagation up parent chain
    fail_machine = StateMachine()

    @fail_machine.state("root_s")
    def root_s(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="ok", state="leaf", input={"fail": False}),
                TaskSpec(id="bad", state="leaf", input={"fail": True}),
            ],
            then=None,
        )

    @fail_machine.state("leaf")
    def leaf_s(task: Task, deps: dict, ctx: TaskContext) -> Done | Fail:
        if task.input["fail"]:
            return Fail(error="boom")
        return Done(output=1)

    fail_res = fail_machine.run(TaskSpec(id="r", state="root_s"))
    assert fail_res.status == "failed"
    assert fail_res.tasks["r"].status == "failed"
    assert "boom" in (fail_res.tasks["r"].error or "")


def test_intra_spawn_depends_on_ordering_and_parallel_barrier() -> None:
    machine = StateMachine()
    barrier = threading.Barrier(3, timeout=3.0)
    order: list[str] = []
    lock = threading.Lock()

    @machine.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        # Note: step_b listed before step_a to prove topological readiness, not list order.
        return Spawn(
            tasks=[
                TaskSpec(id="step_b", state="after", depends_on=["p1", "p2", "p3"]),
                TaskSpec(id="p1", state="parallel"),
                TaskSpec(id="p2", state="parallel"),
                TaskSpec(id="p3", state="parallel"),
            ],
            then=TaskSpec(id="final", state="collect"),
        )

    @machine.state("parallel")
    def parallel(task: Task, deps: dict, ctx: TaskContext) -> Done:
        barrier.wait()
        with lock:
            order.append(task.id)
        return Done(output=task.id.upper())

    @machine.state("after")
    def after(task: Task, deps: dict, ctx: TaskContext) -> Done:
        with lock:
            order.append(task.id)
        return Done(output=sorted(deps.values()))

    @machine.state("collect")
    def collect(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=deps["step_b"])

    res = machine.run(TaskSpec(id="root", state="root"), max_concurrent=3)
    assert res.status == "done"
    assert res.output == ["P1", "P2", "P3"]
    assert order[-1] == "step_b"
    assert set(order[:3]) == {"p1", "p2", "p3"}


def test_cycle_and_duplicate_id_detection() -> None:
    machine = StateMachine()

    @machine.state("noop")
    def noop(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=1)

    with pytest.raises(ValueError):
        machine.run(TaskSpec(id="self", state="noop", depends_on=["self"]))

    cyc_machine = StateMachine()

    @cyc_machine.state("bad_spawn")
    def bad_spawn(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="a", state="noop", depends_on=["b"]),
                TaskSpec(id="b", state="noop", depends_on=["a"]),
            ]
        )

    cyc_machine.register_state("noop", lambda t, d, c: Done(output=1))
    with pytest.raises(ValueError, match="Cycle"):
        cyc_machine.run(TaskSpec(id="root", state="bad_spawn"))

    dup_machine = StateMachine()

    @dup_machine.state("dup_spawn")
    def dup_spawn(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="dup", state="noop"),
                TaskSpec(id="dup", state="noop"),
            ]
        )

    dup_machine.register_state("noop", lambda t, d, c: Done(output=1))
    with pytest.raises(ValueError, match="Duplicate"):
        dup_machine.run(TaskSpec(id="root", state="dup_spawn"))


def test_deduplication_same_wave_and_across_waves() -> None:
    machine = StateMachine()
    calls = 0
    call_lock = threading.Lock()
    events: list[Event] = []

    @machine.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="a1", state="compute", input={"n": 5}, dedup_key="k5"),
                TaskSpec(id="a2", state="compute", input={"n": 5}, dedup_key="k5"),
                TaskSpec(
                    id="a3",
                    state="compute",
                    input={"n": 5},
                    dedup_key="k5",
                    depends_on=["a1"],
                ),
            ],
            then=TaskSpec(id="sum", state="sum_deps"),
        )

    @machine.state("compute")
    def compute(task: Task, deps: dict, ctx: TaskContext) -> Done:
        nonlocal calls
        with call_lock:
            calls += 1
        return Done(output=task.input["n"] * 10)

    @machine.state("sum_deps")
    def sum_deps(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=deps["a1"] + deps["a2"] + deps["a3"])

    res = machine.run(TaskSpec(id="root", state="root"), on_event=events.append)
    assert res.status == "done"
    assert res.output == 150
    assert calls == 1
    dedup_events = [ev for ev in events if ev.event_type == "task_deduped"]
    assert {ev.task_id for ev in dedup_events} == {"a2", "a3"}

    # Automatic deduplicate=True mode
    auto_machine = StateMachine()
    auto_calls = 0

    @auto_machine.state("root")
    def auto_root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="x1", state="auto_work", input={"q": "same"}),
                TaskSpec(id="x2", state="auto_work", input={"q": "same"}),
            ],
            then=None,
        )

    @auto_machine.state("auto_work", deduplicate=True)
    def auto_work(task: Task, deps: dict, ctx: TaskContext) -> Done:
        nonlocal auto_calls
        auto_calls += 1
        return Done(output="cached")

    auto_res = auto_machine.run(TaskSpec(id="r", state="root"))
    assert auto_res.status == "done"
    assert auto_res.output == {"x1": "cached", "x2": "cached"}
    assert auto_calls == 1


def test_retries_and_unhandled_exceptions() -> None:
    machine = StateMachine()
    attempts_seen = 0

    @machine.state("flaky", max_retries=2)
    def flaky(task: Task, deps: dict, ctx: TaskContext):
        nonlocal attempts_seen
        attempts_seen += 1
        if attempts_seen < 3:
            return Fail(error=f"transient {attempts_seen}", retryable=True)
        return Done(output="recovered")

    res = machine.run(TaskSpec(id="r", state="flaky"))
    assert res.status == "done"
    assert res.output == "recovered"
    assert res.tasks["r"].attempts == 3

    err_machine = StateMachine()

    @err_machine.state("boom")
    def boom(task: Task, deps: dict, ctx: TaskContext):
        raise RuntimeError("unexpected crash")

    err_res = err_machine.run(TaskSpec(id="r", state="boom"))
    assert err_res.status == "failed"
    assert "unexpected crash" in (err_res.tasks["r"].error or "")


def test_budget_limits_enforced(tmp_path: Path) -> None:
    machine = StateMachine()

    @machine.state("recurse")
    def recurse(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[TaskSpec(state="recurse")],
            then=None,
        )

    depth_res = machine.run(
        TaskSpec(id="root", state="recurse"),
        run_dir=tmp_path / "depth_run",
        budget=RunBudget(max_depth=2, max_tasks=100),
    )
    assert depth_res.status == "budget_exceeded"
    assert (tmp_path / "depth_run" / "state.json").exists()

    fan_machine = StateMachine()

    @fan_machine.state("fan")
    def fan(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(tasks=[TaskSpec(state="leaf") for _ in range(5)])

    fan_machine.register_state("leaf", lambda t, d, c: Done(output=1))
    tasks_res = fan_machine.run(
        TaskSpec(id="root", state="fan"),
        budget=RunBudget(max_tasks=3),
    )
    assert tasks_res.status == "budget_exceeded"

    slow_machine = StateMachine()

    @slow_machine.state("slow")
    def slow(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        time.sleep(0.05)
        return Spawn(tasks=[TaskSpec(state="leaf")])

    slow_machine.register_state("leaf", lambda t, d, c: Done(output=1))
    wall_res = slow_machine.run(
        TaskSpec(id="root", state="slow"),
        budget=RunBudget(max_wall_clock_s=0.02),
    )
    assert wall_res.status == "budget_exceeded"


def test_auto_spill_and_read_slice_in_downstream_task(tmp_path: Path) -> None:
    machine = StateMachine()
    big_payload = "HEADER:" + ("A" * 300) + ":TAIL"

    @machine.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[TaskSpec(id="producer", state="produce")],
            then=TaskSpec(id="consumer", state="consume"),
        )

    @machine.state("produce")
    def produce(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=big_payload)

    @machine.state("consume")
    def consume(task: Task, deps: dict, ctx: TaskContext) -> Done:
        handle = deps["producer"]
        assert ArtifactStore.is_artifact_handle(handle)
        ref_id = ArtifactStore.extract_ref_id(handle)
        assert ref_id is not None
        head = ctx.store.read_slice(ref_id, offset=0, limit=7)
        tail = ctx.store.read_slice(ref_id, offset=len(big_payload) - 5, limit=5)
        return Done(output=f"{head}|{tail}")

    res = machine.run(
        TaskSpec(id="root", state="root"),
        run_dir=tmp_path / "spill_run",
        budget=RunBudget(max_inline_bytes=64),
    )
    assert res.status == "done"
    assert res.output == "HEADER:|:TAIL"
    assert res.tasks["producer"].spilled_artifact_id is not None
