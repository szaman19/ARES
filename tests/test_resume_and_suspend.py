"""Tests for crash recovery, Suspend/resume, and nested parent pop on resume."""

from __future__ import annotations

from pathlib import Path

from ares import (
    ArtifactStore,
    CheckpointStore,
    Done,
    RunBudget,
    Spawn,
    StateMachine,
    Suspend,
    Task,
    TaskContext,
    TaskSpec,
    Usage,
)


def test_crash_recovery_resets_running_without_rerunning_done(tmp_path: Path) -> None:
    run_dir = tmp_path / "crash_run"
    counts = {"step1": 0, "step2": 0, "join": 0}

    def build_machine() -> StateMachine:
        m = StateMachine()

        @m.state("root")
        def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
            return Spawn(
                tasks=[
                    TaskSpec(id="step1", state="step1"),
                    TaskSpec(id="step2", state="step2"),
                ],
                then=TaskSpec(id="join", state="join"),
            )

        @m.state("step1")
        def step1(task: Task, deps: dict, ctx: TaskContext) -> Done:
            counts["step1"] += 1
            return Done(output="from_step1")

        @m.state("step2")
        def step2(task: Task, deps: dict, ctx: TaskContext) -> Done:
            counts["step2"] += 1
            return Done(output="from_step2")

        @m.state("join")
        def join(task: Task, deps: dict, ctx: TaskContext) -> Done:
            counts["join"] += 1
            return Done(output=f"{deps['step1']}+{deps['step2']}")

        return m

    # Run once cleanly to produce valid checkpoint artifacts, then simulate a mid-flight crash
    # where step1 was already done, step2 was stuck "running", and join was still "pending".
    m1 = build_machine()
    initial = m1.run(TaskSpec(id="root", state="root"), run_dir=run_dir)
    assert initial.status == "done"
    assert counts == {"step1": 1, "step2": 1, "join": 1}

    cp = CheckpointStore(run_dir)
    snap = cp.load_state()
    assert snap is not None
    snap["tasks"]["root"]["status"] = "waiting"
    snap["tasks"]["root"]["output"] = None
    snap["tasks"]["step2"]["status"] = "running"
    snap["tasks"]["step2"]["output"] = None
    snap["tasks"]["join"]["status"] = "pending"
    snap["tasks"]["join"]["output"] = None
    cp.save_state(snap)

    counts = {"step1": 0, "step2": 0, "join": 0}
    m2 = build_machine()
    resumed = m2.resume(run_dir)
    assert resumed.status == "done"
    assert resumed.output == "from_step1+from_step2"
    assert counts["step1"] == 0  # Done task must never be re-executed.
    assert counts["step2"] == 1
    assert counts["join"] == 1


def test_suspend_complete_on_resume_false_and_true(tmp_path: Path) -> None:
    # Case 1: complete_on_resume=False -> task re-runs with ctx.resume_data
    run_dir_1 = tmp_path / "suspend_rerun"
    m1 = StateMachine()

    @m1.state("gate")
    def gate(task: Task, deps: dict, ctx: TaskContext):
        if ctx.resume_data is None:
            return Suspend(
                reason="awaiting human sign-off",
                resume_key="human_gate",
                complete_on_resume=False,
            )
        return Done(output={"approved_by": ctx.resume_data["user"]})

    r1 = m1.run(TaskSpec(id="g", state="gate"), run_dir=run_dir_1)
    assert r1.status == "suspended"
    assert r1.suspended_task_ids == ["g"]

    # Resuming without matching input keeps task suspended.
    r1_noop = m1.resume(run_dir_1, resume_inputs={"other_key": 123})
    assert r1_noop.status == "suspended"
    assert r1_noop.suspended_task_ids == ["g"]

    r1_done = m1.resume(run_dir_1, resume_inputs={"human_gate": {"user": "alice"}})
    assert r1_done.status == "done"
    assert r1_done.output == {"approved_by": "alice"}

    # Case 2: complete_on_resume=True inside nested Spawn + then chain + spilled output
    run_dir_2 = tmp_path / "suspend_nested"
    m2 = StateMachine()

    @m2.state("outer")
    def outer(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[TaskSpec(id="inner_parent", state="inner")],
            then=TaskSpec(id="outer_then", state="finalize"),
        )

    @m2.state("inner")
    def inner(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        return Spawn(
            tasks=[TaskSpec(id="waiter", state="external_wait")],
            then=TaskSpec(id="inner_then", state="read_external"),
        )

    @m2.state("external_wait")
    def external_wait(task: Task, deps: dict, ctx: TaskContext) -> Suspend:
        return Suspend(
            reason="wait for external blob",
            resume_key="ext_blob",
            complete_on_resume=True,
        )

    @m2.state("read_external")
    def read_external(task: Task, deps: dict, ctx: TaskContext) -> Done:
        val = deps["waiter"]
        assert ArtifactStore.is_artifact_handle(val)
        ref_id = ArtifactStore.extract_ref_id(val)
        assert ref_id is not None
        return Done(output=f"len={len(ctx.store.get_text(ref_id))}")

    @m2.state("finalize")
    def finalize(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output=f"final:{deps['inner_parent']}")

    r2 = m2.run(
        TaskSpec(id="root", state="outer"),
        run_dir=run_dir_2,
        budget=RunBudget(max_inline_bytes=32),
    )
    assert r2.status == "suspended"
    assert r2.suspended_task_ids == ["waiter"]

    big_external = "Z" * 100
    r2_done = m2.resume(run_dir_2, resume_inputs={"ext_blob": big_external})
    assert r2_done.status == "done"
    assert r2_done.output == "final:len=100"
    assert r2_done.tasks["waiter"].spilled_artifact_id is not None
    assert r2_done.tasks["inner_parent"].status == "done"
    assert r2_done.tasks["root"].status == "done"


def test_resume_preserves_usage_and_race_cancellation(tmp_path: Path) -> None:
    run_dir = tmp_path / "resume_usage_race"
    m = StateMachine()

    @m.state("root")
    def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
        ctx.record_usage(prompt_tokens=15, completion_tokens=5, llm_calls=1, cost_usd=0.01)
        return Spawn(
            tasks=[
                TaskSpec(id="gated", state="gated"),
                TaskSpec(id="never_branch", state="gated_slow"),
            ],
            then=TaskSpec(id="after", state="after"),
            join="any",
        )

    @m.state("gated")
    def gated(task: Task, deps: dict, ctx: TaskContext):
        if ctx.resume_data is None:
            return Suspend(reason="need token", resume_key="gate_key")
        return Done(
            output=ctx.resume_data,
            usage=Usage(prompt_tokens=20, completion_tokens=10, llm_calls=1, cost_usd=0.02),
        )

    @m.state("gated_slow")
    def gated_slow(task: Task, deps: dict, ctx: TaskContext) -> Suspend:
        return Suspend(reason="slower gate", resume_key="slow_key")

    @m.state("after")
    def after(task: Task, deps: dict, ctx: TaskContext) -> Done:
        return Done(output={"won": deps["gated"]})

    r1 = m.run(TaskSpec(id="root", state="root"), run_dir=run_dir, max_concurrent=2)
    assert r1.status == "suspended"
    assert set(r1.suspended_task_ids) == {"gated", "never_branch"}
    assert r1.usage.total_tokens == 20

    r2 = m.resume(run_dir, resume_inputs={"gate_key": "ok"}, max_concurrent=2)
    assert r2.status == "done"
    assert r2.output == {"won": "ok"}
    assert "never_branch" in r2.cancelled_task_ids
    assert r2.usage.total_tokens == 50
    assert r2.usage.llm_calls == 2
