"""Integration tests for StateMachine PDA, DAG dependencies, parallelism, dedup, and budgets."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ares import (
    ArtifactRef,
    ArtifactStore,
    CheckpointStore,
    Done,
    Event,
    Fail,
    RunBudget,
    Spawn,
    StateMachine,
    Suspend,
    Task,
    TaskContext,
    TaskSpec,
    Usage,
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


def test_tail_call_preserves_o1_depth_across_50_hops() -> None:
    machine = StateMachine()

    @machine.state("step")
    def step(task: Task, deps: dict, ctx: TaskContext):
        remaining = int(task.input["remaining"])
        if remaining == 0:
            return Done(output=task.input["acc"])
        return Spawn(
            tasks=[],
            then=TaskSpec(
                state="step",
                input={"remaining": remaining - 1, "acc": task.input["acc"] + 1},
            ),
        )

    res = machine.run(
        TaskSpec(id="root", state="step", input={"remaining": 50, "acc": 0}),
        budget=RunBudget(max_depth=3, max_tasks=100),
    )
    assert res.status == "done"
    assert res.output == 50
    assert len(res.tasks) == 51
    assert all(t.depth == 0 for t in res.tasks.values())


def test_pending_continuation_fails_when_child_fails() -> None:
    machine = StateMachine()
    cont_ran = False

    @machine.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="ok", state="leaf", input={"fail": False}),
                TaskSpec(id="bad", state="leaf", input={"fail": True}),
            ],
            then=TaskSpec(id="cont", state="join"),
        )

    @machine.state("leaf")
    def leaf(task: Task, deps: dict, ctx: TaskContext):
        if task.input["fail"]:
            return Fail(error="leaf_boom")
        return Done(output=42)

    @machine.state("join")
    def join(task: Task, deps: dict, ctx: TaskContext) -> Done:
        nonlocal cont_ran
        cont_ran = True
        return Done(output=deps)

    res = machine.run(TaskSpec(id="root", state="root"))
    assert res.status == "failed"
    assert not cont_ran
    assert res.tasks["cont"].status == "failed"
    assert res.tasks["root"].status == "failed"
    assert "leaf_boom" in (res.tasks["cont"].error or "")


def test_race_join_any_without_continuation_cancels_slow_subtree() -> None:
    machine = StateMachine()
    fast_done = threading.Event()
    slow_child_ran = threading.Event()
    events: list[Event] = []

    @machine.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="fast", state="fast_branch"),
                TaskSpec(id="slow", state="slow_branch"),
            ],
            then=None,
            join="any",
        )

    @machine.state("fast_branch")
    def fast_branch(task: Task, deps: dict, ctx: TaskContext) -> Done:
        fast_done.set()
        return Done(output="winner_payload")

    @machine.state("slow_branch")
    def slow_branch(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        # Spawn a nested child that must be cancelled before executing Wave 2.
        fast_done.wait(timeout=2.0)
        return Spawn(
            tasks=[TaskSpec(id="slow_leaf", state="slow_leaf_state")],
            then=TaskSpec(id="slow_cont", state="slow_leaf_state"),
        )

    @machine.state("slow_leaf_state")
    def slow_leaf_state(task: Task, deps: dict, ctx: TaskContext) -> Done:
        slow_child_ran.set()
        return Done(output="never")

    res = machine.run(
        TaskSpec(id="root", state="root"),
        max_concurrent=2,
        on_event=events.append,
    )
    assert res.status == "done"
    assert res.output == "winner_payload"
    assert not slow_child_ran.is_set()
    assert set(res.cancelled_task_ids) >= {"slow", "slow_leaf", "slow_cont"}
    assert any(ev.event_type == "task_cancelled" for ev in events)


def test_race_join_any_with_continuation_and_all_fail() -> None:
    machine = StateMachine()
    seen_deps: dict[str, object] = {}

    @machine.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="b_fail", state="branch_fail"),
                TaskSpec(id="b_win", state="branch_win"),
                TaskSpec(id="b_lose", state="branch_lose"),
            ],
            then=TaskSpec(id="pick", state="pick_winner"),
            join="any",
        )

    @machine.state("branch_fail")
    def branch_fail(task: Task, deps: dict, ctx: TaskContext) -> Fail:
        return Fail(error="early_fail")

    @machine.state("branch_win")
    def branch_win(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output={"branch": "b_win", "score": 99})

    @machine.state("branch_lose")
    def branch_lose(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(tasks=[TaskSpec(id="lose_sub", state="branch_win")])

    @machine.state("pick_winner")
    def pick_winner(task: Task, deps: dict, ctx: TaskContext) -> Done:
        seen_deps.update(deps)
        return Done(output={"picked": next(iter(deps.values()))})

    res = machine.run(TaskSpec(id="root", state="root"), max_concurrent=3)
    assert res.status == "done"
    assert res.output == {"picked": {"branch": "b_win", "score": 99}}
    assert list(seen_deps.keys()) == ["b_win"]
    assert "b_lose" in res.cancelled_task_ids
    assert "lose_sub" in res.cancelled_task_ids

    # All children fail -> root fails and dormant continuation is cancelled.
    fail_machine = StateMachine()

    @fail_machine.state("root")
    def fail_root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[
                TaskSpec(id="f1", state="boom"),
                TaskSpec(id="f2", state="boom"),
            ],
            then=TaskSpec(id="never_run", state="cont"),
            join="any",
        )

    fail_machine.register_state("boom", lambda t, d, c: Fail(error="all_bad"))
    fail_machine.register_state("cont", lambda t, d, c: Done(output="unreachable"))

    fail_res = fail_machine.run(TaskSpec(id="root", state="root"), max_concurrent=2)
    assert fail_res.status == "failed"
    assert fail_res.tasks["never_run"].status == "cancelled"


def test_usage_recording_and_token_cost_budgets() -> None:
    machine = StateMachine()

    @machine.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        ctx.record_usage(prompt_tokens=10, completion_tokens=5, llm_calls=1, cost_usd=0.01)
        return Spawn(
            tasks=[TaskSpec(id="c1", state="child"), TaskSpec(id="c2", state="child")],
            then=TaskSpec(id="join", state="join"),
        )

    @machine.state("child")
    def child(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(
            output=1,
            usage=Usage(prompt_tokens=20, completion_tokens=10, llm_calls=1, cost_usd=0.02),
        )

    @machine.state("join")
    def join(task: Task, deps: dict, ctx: TaskContext) -> Done:
        ctx.record_usage(prompt_tokens=5, completion_tokens=5, llm_calls=1, cost_usd=0.005)
        return Done(output=sum(deps.values()))

    ok_res = machine.run(TaskSpec(id="root", state="root"))
    assert ok_res.status == "done"
    assert ok_res.usage.prompt_tokens == 55
    assert ok_res.usage.completion_tokens == 30
    assert ok_res.usage.total_tokens == 85
    assert ok_res.usage.llm_calls == 4
    assert ok_res.usage.cost_usd == pytest.approx(0.055)

    # Exceed max_tokens
    tok_res = machine.run(
        TaskSpec(id="root", state="root"),
        budget=RunBudget(max_tokens=40),
    )
    assert tok_res.status == "budget_exceeded"
    assert tok_res.tasks["join"].status == "pending"

    # Exceed max_cost_usd
    cost_res = machine.run(
        TaskSpec(id="root", state="root"),
        budget=RunBudget(max_cost_usd=0.03),
    )
    assert cost_res.status == "budget_exceeded"


def test_trace_events_done_artifacts_metadata_and_deterministic(tmp_path: Path) -> None:
    machine = StateMachine()
    events: list[Event] = []

    @machine.state("det_root", deterministic=True)
    def det_root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        ctx.emit("trace", {"phase": "root_enter", "step": 1})
        return Spawn(
            tasks=[TaskSpec(id="worker", state="nondet_worker", metadata={"origin": "root"})],
            then=TaskSpec(id="det_join", state="det_join"),
            metadata={"spawned_by": "det_root"},
        )

    @machine.state("nondet_worker")
    def nondet_worker(task: Task, deps: dict, ctx: TaskContext) -> Done:
        ref1 = ctx.store.put_text("artifact one", name="one.txt")
        ref2 = ctx.store.put_text("artifact two", name="two.txt")
        ctx.emit("trace", {"phase": "worker_artifacts", "count": 2})
        return Done(
            output="work_done",
            artifacts=[ref1, ref1, ref2],
            metadata={"worker_status": "ok"},
        )

    @machine.state("det_join", deterministic=True)
    def det_join(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=f"joined:{deps['worker']}", metadata={"finalized": True})

    run_dir = tmp_path / "obs_run"
    res = machine.run(
        TaskSpec(id="root", state="det_root", metadata={"seed": 7}),
        run_dir=run_dir,
        on_event=events.append,
    )
    assert res.status == "done"
    assert res.output == "joined:work_done"

    # deterministic flag propagated from @machine.state(..., deterministic=True)
    assert res.tasks["root"].deterministic is True
    assert res.tasks["worker"].deterministic is False
    assert res.tasks["det_join"].deterministic is True

    # Transition.metadata merged onto Task.metadata across Spawn and Done
    assert res.tasks["root"].metadata == {"seed": 7, "spawned_by": "det_root"}
    assert res.tasks["worker"].metadata == {"origin": "root", "worker_status": "ok"}
    assert res.tasks["det_join"].metadata == {"finalized": True}

    # Done.artifacts persisted and deduped by ref_id on Task.artifact_refs
    worker_refs = res.tasks["worker"].artifact_refs
    assert len(worker_refs) == 2
    assert all(isinstance(r, ArtifactRef) for r in worker_refs)
    assert len({r.ref_id for r in worker_refs}) == 2

    # Trace events & task_done artifacts payload recorded both in callback and events.jsonl
    trace_events = [ev for ev in events if ev.event_type == "trace"]
    assert [(ev.task_id, ev.payload) for ev in trace_events] == [
        ("root", {"phase": "root_enter", "step": 1}),
        ("worker", {"phase": "worker_artifacts", "count": 2}),
    ]
    worker_done_ev = next(
        ev for ev in events if ev.event_type == "task_done" and ev.task_id == "worker"
    )
    assert "artifacts" in worker_done_ev.payload
    assert set(worker_done_ev.payload["artifacts"]) == {r.ref_id for r in worker_refs}

    persisted_events = CheckpointStore(run_dir).load_events()
    assert any(ev.event_type == "trace" for ev in persisted_events)

    # Verify Suspend and Fail also merge Transition.metadata
    sf_machine = StateMachine()

    @sf_machine.state("suspend_then_fail")
    def suspend_then_fail(task: Task, deps: dict, ctx: TaskContext):
        if ctx.resume_data is None:
            return Suspend(reason="pause", resume_key="k", metadata={"paused": True})
        return Fail(error="rejected", metadata={"failed_after_resume": True})

    sf_dir = tmp_path / "sf_run"
    r_susp = sf_machine.run(TaskSpec(id="sf", state="suspend_then_fail"), run_dir=sf_dir)
    assert r_susp.status == "suspended"
    assert r_susp.tasks["sf"].metadata == {"paused": True}

    r_fail = sf_machine.resume(sf_dir, resume_inputs={"k": "reject"})
    assert r_fail.status == "failed"
    assert r_fail.tasks["sf"].metadata == {"paused": True, "failed_after_resume": True}


def test_fresh_run_resets_stale_events_jsonl_in_reused_run_dir(tmp_path: Path) -> None:
    machine = StateMachine()

    @machine.state("echo")
    def echo(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=task.input["val"])

    run_dir = tmp_path / "reused_dir"
    r1 = machine.run(TaskSpec(id="first_root", state="echo", input={"val": 1}), run_dir=run_dir)
    assert r1.status == "done"
    ev1 = CheckpointStore(run_dir).load_events()
    assert any(ev.task_id == "first_root" for ev in ev1)

    r2 = machine.run(TaskSpec(id="second_root", state="echo", input={"val": 2}), run_dir=run_dir)
    assert r2.status == "done"
    ev2 = CheckpointStore(run_dir).load_events()
    assert ev2[0].seq == 1
    assert all(ev.task_id != "first_root" for ev in ev2)
    assert any(ev.task_id == "second_root" for ev in ev2)
