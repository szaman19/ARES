"""End-to-end tests for the thin ARES v2 CLI (`run`, `resume`, `inspect`)."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

import ares
from ares import ArtifactStore
from ares.cli import main

_MACHINE_MODULE = '''\
from ares import Done, Spawn, StateMachine, Suspend, Task, TaskContext, TaskSpec

machine = StateMachine()


@machine.state("entry")
def entry(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
    return Spawn(
        tasks=[
            TaskSpec(id="prep", state="prep", input=task.input),
            TaskSpec(id="gate", state="gate"),
        ],
        then=TaskSpec(id="finish", state="finish"),
    )


@machine.state("prep")
def prep(task: Task, deps: dict, ctx: TaskContext) -> Done:
    text = task.input.get("text", "default")
    return Done(output=text * task.input.get("repeat", 1))


@machine.state("gate")
def gate(task: Task, deps: dict, ctx: TaskContext):
    if ctx.resume_data is None:
        return Suspend(reason="waiting approval", resume_key="approval")
    return Done(output=ctx.resume_data)


@machine.state("finish")
def finish(task: Task, deps: dict, ctx: TaskContext) -> Done:
    return Done(output={"prep_ref": deps["prep"]["__ares_ref__"], "gate": deps["gate"]})


def build_machine() -> StateMachine:
    return machine
'''


def _write_temp_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    mod_name = "temp_cli_machine"
    (tmp_path / f"{mod_name}.py").write_text(_MACHINE_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    return mod_name


def test_cli_run_resume_inspect_json_and_human(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod_name = _write_temp_machine(tmp_path, monkeypatch)
    run_dir = tmp_path / "run_state"

    # 1. Run until suspended (using module:attr instance and spilling prep output).
    rc = main(
        [
            "run",
            f"{mod_name}:machine",
            "--state",
            "entry",
            "--id",
            "root_1",
            "--input",
            '{"text": "ABCD", "repeat": 20}',
            "--run-dir",
            str(run_dir),
            "--max-concurrent",
            "2",
            "--max-tasks",
            "50",
            "--max-inline-bytes",
            "64",
            "--json",
        ]
    )
    assert rc == 0
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["status"] == "suspended"
    assert run_out["root_task_id"] == "root_1"
    assert run_out["suspended_task_ids"] == ["gate"]
    assert run_out["tasks"]["prep"]["spilled_artifact_id"] is not None

    # 2. Inspect in human-readable mode while suspended.
    rc_inspect_human = main(["inspect", str(run_dir)])
    assert rc_inspect_human == 0
    inspect_text = capsys.readouterr().out
    assert "status=suspended" in inspect_text
    assert "Artifacts:" in inspect_text
    assert "root_1" in inspect_text

    # 3. Resume using factory callable and --provide, emitting JSON.
    rc_resume = main(
        [
            "resume",
            str(run_dir),
            "--machine",
            f"{mod_name}:build_machine",
            "--provide",
            'approval={"approved": true, "reviewer": "bob"}',
            "--max-concurrent",
            "2",
            "--json",
        ]
    )
    assert rc_resume == 0
    resume_out = json.loads(capsys.readouterr().out)
    assert resume_out["status"] == "done"
    assert ArtifactStore.is_artifact_handle(resume_out["output"])
    final_obj = ArtifactStore(run_dir / "artifacts").get_json(
        ArtifactStore.extract_ref_id(resume_out["output"])  # type: ignore[arg-type]
    )
    assert final_obj["gate"] == {"approved": True, "reviewer": "bob"}
    assert final_obj["prep_ref"].startswith("art_")

    # 4. Inspect final state in JSON mode.
    rc_inspect_json = main(["inspect", str(run_dir), "--json"])
    assert rc_inspect_json == 0
    inspect_json = json.loads(capsys.readouterr().out)
    assert inspect_json["status"] == "done"
    assert inspect_json["root_task_id"] == "root_1"
    assert len(inspect_json["artifacts"]) >= 1
    assert any(ev["event_type"] == "run_finished" for ev in inspect_json["events"])


def test_cli_run_human_output_and_error_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod_name = _write_temp_machine(tmp_path, monkeypatch)

    rc = main(
        [
            "run",
            f"{mod_name}:build_machine",
            "--state",
            "prep",
            "--input",
            '{"text": "hi", "repeat": 2}',
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "task_started" in out
    assert "task_done" in out
    assert "status=done" in out
    assert '"hihi"' in out

    # Inspect missing directory returns non-zero cleanly.
    rc_missing = main(["inspect", str(tmp_path / "does_not_exist")])
    assert rc_missing == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_artifact_tree_and_token_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod_name = _write_temp_machine(tmp_path, monkeypatch)
    run_dir = tmp_path / "tree_art_run"

    rc_run = main(
        [
            "run",
            f"{mod_name}:machine",
            "--state",
            "entry",
            "--id",
            "root_1",
            "--input",
            '{"text": "WXYZ", "repeat": 30}',
            "--run-dir",
            str(run_dir),
            "--max-inline-bytes",
            "32",
            "--max-tokens",
            "1000",
            "--json",
        ]
    )
    assert rc_run == 0
    run_out = json.loads(capsys.readouterr().out)
    prep_ref = run_out["tasks"]["prep"]["spilled_artifact_id"]
    assert prep_ref is not None

    # inspect --tree
    rc_tree = main(["inspect", str(run_dir), "--tree"])
    assert rc_tree == 0
    tree_out = capsys.readouterr().out
    assert "Task Tree:" in tree_out
    assert "root_1 [entry]" in tree_out
    assert "prep [prep]" in tree_out
    assert f"ref={prep_ref}" in tree_out

    # ares artifact full and sliced
    rc_art_full = main(["artifact", str(run_dir), prep_ref])
    assert rc_art_full == 0
    assert capsys.readouterr().out == "WXYZ" * 30

    rc_art_slice = main(["artifact", str(run_dir), prep_ref, "--offset", "1", "--limit", "4"])
    assert rc_art_slice == 0
    assert capsys.readouterr().out == "XYZW"

    rc_art_missing = main(["artifact", str(run_dir), "art_missing"])
    assert rc_art_missing == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_inspect_tail_and_tree_det_arts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod_source = '''\
from ares import Done, Spawn, StateMachine, Task, TaskContext, TaskSpec

machine = StateMachine()


@machine.state("root", deterministic=True)
def root(task: Task, deps: dict, ctx: TaskContext) -> Spawn:
    ctx.emit("trace", {"note": "start"})
    return Spawn(
        tasks=[TaskSpec(id=f"w{i}", state="worker", input={"i": i}) for i in range(4)],
        then=TaskSpec(id="agg", state="agg"),
    )


@machine.state("worker")
def worker(task: Task, deps: dict, ctx: TaskContext) -> Done:
    ref = ctx.store.put_text(f"payload-{task.input['i']}")
    return Done(output=task.input["i"], artifacts=[ref])


@machine.state("agg", deterministic=True)
def agg(task: Task, deps: dict, ctx: TaskContext) -> Done:
    return Done(output=sum(deps.values()))
'''
    mod_name = "temp_tail_machine"
    (tmp_path / f"{mod_name}.py").write_text(mod_source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    run_dir = tmp_path / "tail_run"

    rc_run = main(
        [
            "run",
            f"{mod_name}:machine",
            "--state",
            "root",
            "--id",
            "root_1",
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc_run == 0
    capsys.readouterr()

    # Default --tail 10 hides earliest events when total > 10.
    assert main(["inspect", str(run_dir), "--tree"]) == 0
    default_out = capsys.readouterr().out
    assert "root_1 [root] (done)" in default_out
    assert " det" in default_out
    assert "arts=1" in default_out
    assert "[   1] " not in default_out

    # --tail 3 shows only the last 3 events.
    assert main(["inspect", str(run_dir), "--tail", "3"]) == 0
    tail3_out = capsys.readouterr().out
    event_lines = [ln for ln in tail3_out.splitlines() if ln.strip().startswith("[")]
    assert len(event_lines) == 3

    # --tail 0 shows all events starting from [   1].
    assert main(["inspect", str(run_dir), "--tail", "0"]) == 0
    tail_all_out = capsys.readouterr().out
    assert "[   1] " in tail_all_out
    assert "trace" in tail_all_out


def test_cli_version_flag_and_pyproject_sync(capsys: pytest.CaptureFixture[str]) -> None:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as fh:
        data = tomllib.load(fh)
    assert ares.__version__ == data["project"]["version"]

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    assert f"ares {ares.__version__}" in capsys.readouterr().out
