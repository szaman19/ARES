"""Offline end-to-end tests for examples/codegen_verify_repair."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ares import ArtifactStore, RunBudget, TaskSpec
from ares.cli import main as cli_main
from ares.llm import OpenAICompatClient
from examples.codegen_verify_repair import TASKS, build_machine
from examples.codegen_verify_repair.run import main as demo_main


def _unwrap(value: Any, store: ArtifactStore) -> Any:
    if ArtifactStore.is_artifact_handle(value):
        ref_id = ArtifactStore.extract_ref_id(value)
        if ref_id is not None:
            return store.get_json(ref_id)
    return value


def test_slugify_parallel_verify_repair_and_certify(tmp_path: Path) -> None:
    run_dir = tmp_path / "slugify_run"
    machine = build_machine(max_attempts=3)
    result = machine.run(
        TaskSpec(state="plan", input={"task": "slugify"}),
        run_dir=run_dir,
        max_concurrent=4,
    )

    assert result.status == "done"
    store = ArtifactStore(run_dir / "artifacts")
    output = _unwrap(result.output, store)
    assert output["certified"] is True
    assert output["task"] == "slugify"
    assert output["selected_branch"] == "branch_0"
    assert output["attempts"] == 2
    code_ref_id = output["code_ref_id"]
    assert isinstance(code_ref_id, str) and code_ref_id.startswith("art_")

    # Verify branch outcomes recorded in select_best output.
    by_branch = {item["branch_id"]: item for item in output["branches"]}
    assert set(by_branch) == {"branch_0", "branch_1", "branch_2"}
    assert by_branch["branch_0"]["certified"] is True
    assert by_branch["branch_0"]["attempt"] == 2
    assert by_branch["branch_1"]["certified"] is False
    assert by_branch["branch_1"]["attempt"] == 3
    assert by_branch["branch_2"]["certified"] is True
    assert by_branch["branch_2"]["attempt"] == 3

    # Verify branch_0 failed attempt 1 in real pytest and passed attempt 2.
    decide_tasks = [t for t in result.tasks.values() if t.state == "decide_repair"]
    b0_attempts = sorted(
        [t for t in decide_tasks if t.input["branch_id"] == "branch_0"],
        key=lambda t: int(t.input["attempt"]),
    )
    assert [int(t.input["attempt"]) for t in b0_attempts] == [1, 2]

    b0_v1 = _unwrap(result.tasks[b0_attempts[0].depends_on[0]].output, store)
    b0_v2 = _unwrap(result.tasks[b0_attempts[1].depends_on[0]].output, store)
    assert b0_v1["passed"] is False
    assert b0_v1["returncode"] != 0
    assert "AssertionError" in b0_v1["failure_summary"]
    assert b0_v2["passed"] is True
    assert b0_v2["returncode"] == 0

    # Verify Invariant I4: verify_pytest receives ONLY task and code_ref_id.
    verify_tasks = [t for t in result.tasks.values() if t.state == "verify_pytest"]
    assert len(verify_tasks) == 8  # branch_0: 2, branch_1: 3, branch_2: 3
    for vtask in verify_tasks:
        assert set(vtask.input.keys()) == {"task", "code_ref_id"}

    # Verify certified code retrieved from ArtifactStore executes correctly.
    certified_code = store.get_text(code_ref_id)
    namespace: dict[str, Any] = {}
    exec(certified_code, namespace)
    slugify_fn = namespace["slugify"]
    assert slugify_fn("  Hello, World!!  ") == "hello-world"
    assert slugify_fn("Café & Résumé") == "cafe-resume"
    assert slugify_fn("---!!!") == ""


def test_rle_encode_task_and_spilled_failure_slice(tmp_path: Path) -> None:
    run_dir = tmp_path / "rle_run"
    machine = build_machine(max_attempts=3)
    # Small max_inline_bytes forces task outputs (including verify_pytest) to spill.
    result = machine.run(
        TaskSpec(state="plan", input={"task": "rle_encode"}),
        run_dir=run_dir,
        budget=RunBudget(max_inline_bytes=128),
    )

    assert result.status == "done"
    store = ArtifactStore(run_dir / "artifacts")
    output = _unwrap(result.output, store)
    assert output["certified"] is True
    assert output["task"] == "rle_encode"
    assert output["selected_branch"] == "branch_0"
    assert output["attempts"] == 2

    certified_code = store.get_text(output["code_ref_id"])
    namespace: dict[str, Any] = {}
    exec(certified_code, namespace)
    rle_encode = namespace["rle_encode"]
    assert rle_encode("") == []
    assert rle_encode("aaabbc") == [("a", 3), ("b", 2), ("c", 1)]
    assert rle_encode("aba") == [("a", 1), ("b", 1), ("a", 1)]


def test_max_attempts_one_returns_uncertified_honestly(tmp_path: Path) -> None:
    run_dir = tmp_path / "all_fail_run"
    machine = build_machine(max_attempts=1)
    result = machine.run(
        TaskSpec(state="plan", input={"task": "slugify"}),
        run_dir=run_dir,
    )

    assert result.status == "done"
    store = ArtifactStore(run_dir / "artifacts")
    output = _unwrap(result.output, store)
    assert output["certified"] is False
    assert output["selected_branch"] is None
    assert output["code_ref_id"] is None
    assert len(output["branches"]) == 3
    assert all(branch["certified"] is False for branch in output["branches"])
    assert all(branch["attempt"] == 1 for branch in output["branches"])

    # Exactly 3 verification runs (no extra repair attempts).
    verify_tasks = [t for t in result.tasks.values() if t.state == "verify_pytest"]
    assert len(verify_tasks) == 3


def test_generate_reads_spilled_failure_summary_via_read_slice(tmp_path: Path) -> None:
    run_dir = tmp_path / "spilled_summary"
    store = ArtifactStore(run_dir / "artifacts")
    long_failure = ("FAILED test_solution.py::test_edge - AssertionError\n" * 40).strip()
    spilled_handle, ref = store.maybe_spill(long_failure, max_inline_bytes=128)
    assert ref is not None

    machine = build_machine(max_attempts=2)
    result = machine.run(
        TaskSpec(
            state="generate",
            input={
                "task": "slugify",
                "branch_id": "branch_0",
                "attempt": 2,
                "max_attempts": 2,
                "failure_summary": spilled_handle,
            },
        ),
        run_dir=run_dir,
    )
    assert result.status == "done"
    out = _unwrap(result.output, ArtifactStore(run_dir / "artifacts"))
    assert out["certified"] is True
    assert out["branch_id"] == "branch_0"
    assert out["attempt"] == 2


def test_cli_run_and_inspect(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert "slugify" in TASKS and "rle_encode" in TASKS
    run_dir = tmp_path / "cli_run"

    rc_run = cli_main(
        [
            "run",
            "examples.codegen_verify_repair.machine:build_machine",
            "--state",
            "plan",
            "--input",
            '{"task": "slugify"}',
            "--run-dir",
            str(run_dir),
            "--json",
        ]
    )
    assert rc_run == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["status"] == "done"
    assert run_payload["output"]["certified"] is True
    assert run_payload["output"]["selected_branch"] == "branch_0"

    rc_inspect = cli_main(["inspect", str(run_dir), "--json"])
    assert rc_inspect == 0
    inspect_payload = json.loads(capsys.readouterr().out)
    assert inspect_payload["status"] == "done"
    assert len(inspect_payload["artifacts"]) >= 1
    assert any(ev["event_type"] == "run_finished" for ev in inspect_payload["events"])


def test_optional_live_llm_codegen_path(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_msg = body["messages"][-1]["content"]
        if "branch_0" in user_msg and "repair attempt" not in user_msg:
            code = "```python\ndef rle_encode(text: str) -> list[tuple[str, int]]:\n    return []\n```"
        elif "branch_0" in user_msg:
            code = (
                "```python\n"
                "def rle_encode(text: str) -> list[tuple[str, int]]:\n"
                "    if not text:\n"
                "        return []\n"
                "    runs = []\n"
                "    cur, count = text[0], 1\n"
                "    for ch in text[1:]:\n"
                "        if ch == cur:\n"
                "            count += 1\n"
                "        else:\n"
                "            runs.append((cur, count))\n"
                "            cur, count = ch, 1\n"
                "    runs.append((cur, count))\n"
                "    return runs\n"
                "```"
            )
        else:
            code = "def rle_encode(text: str) -> list[tuple[str, int]]:\n    return [('bad', 0)]\n"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": code}}]},
        )

    client = OpenAICompatClient(
        base_url="http://mock.local/v1",
        model="mock-coder",
        transport=httpx.MockTransport(handler),
    )
    run_dir = tmp_path / "llm_run"
    machine = build_machine(llm=client, max_attempts=2)
    result = machine.run(
        TaskSpec(state="plan", input={"task": "rle_encode"}),
        run_dir=run_dir,
    )
    assert result.status == "done"
    out = _unwrap(result.output, ArtifactStore(run_dir / "artifacts"))
    assert out["certified"] is True
    assert out["selected_branch"] == "branch_0"
    assert out["attempts"] == 2


def test_demo_runner_rle_encode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = demo_main(["--task", "rle_encode", "--run-dir", str(tmp_path / "demo_rle")])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert "task=rle_encode" in stdout
    assert "certified=True" in stdout
    assert "selected_branch=branch_0" in stdout


def test_race_mode_cancels_slower_branches_after_branch_0_passes(tmp_path: Path) -> None:
    all_res = build_machine(max_attempts=3, race=False).run(
        TaskSpec(state="plan", input={"task": "slugify"}),
        run_dir=tmp_path / "no_race",
        max_concurrent=3,
    )
    race_res = build_machine(max_attempts=3, race=True).run(
        TaskSpec(state="plan", input={"task": "slugify"}),
        run_dir=tmp_path / "with_race",
        max_concurrent=3,
    )

    assert race_res.status == "done"
    store = ArtifactStore(tmp_path / "with_race" / "artifacts")
    out = _unwrap(race_res.output, store)
    assert out["certified"] is True
    assert out["selected_branch"] == "branch_0"
    assert out["attempts"] == 2
    assert len(race_res.cancelled_task_ids) > 0
    done_verifies_race = [
        t for t in race_res.tasks.values() if t.state == "verify_pytest" and t.status == "done"
    ]
    done_verifies_all = [
        t for t in all_res.tasks.values() if t.state == "verify_pytest" and t.status == "done"
    ]
    assert len(done_verifies_race) < len(done_verifies_all)
