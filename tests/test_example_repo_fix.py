"""Offline end-to-end tests for ``examples/repo_fix`` multi-file bounded repair."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from ares import ArtifactStore, TaskSpec
from ares.cli import main as cli_main
from ares.ext.llm import OpenAICompatClient
from ares.ext.git import GitWorktreeSandbox
from examples.repo_fix import build_machine, create_sample_repo
from examples.repo_fix.run import main as demo_main


def _unwrap(value: Any, store: ArtifactStore) -> Any:
    if ArtifactStore.is_artifact_handle(value):
        ref_id = ArtifactStore.extract_ref_id(value)
        if ref_id is not None:
            return store.get_json(ref_id)
    return value


def test_repo_fix_bounded_search_parallel_race_and_patch_certification(tmp_path: Path) -> None:
    repo_dir = create_sample_repo(tmp_path / "billing_repo")
    run_dir = tmp_path / "repo_fix_run"

    # Confirm baseline tests actually fail before ARES runs.
    with GitWorktreeSandbox(repo_dir) as wt_base:
        base_res = wt_base.run_command([sys.executable, "-m", "pytest", "-q"])
        assert not base_res.passed

    res = build_machine(max_attempts=3, race=True).run(
        TaskSpec(state="locate", input={"repo_dir": str(repo_dir)}),
        run_dir=run_dir,
        max_concurrent=4,
    )
    assert res.status == "done"

    store = ArtifactStore(run_dir / "artifacts")
    out = _unwrap(res.output, store)
    assert out["certified"] is True
    assert out["selected_branch"] == "branch_0"
    assert out["attempts"] == 2
    assert set(out["relevant_files"]) == {"billing/discounts.py", "billing/invoice.py"}
    assert "billing/formatting.py" in out["ignored_files"]
    assert "billing/logging_utils.py" in out["ignored_files"]
    assert len(res.cancelled_task_ids) > 0
    assert res.usage.llm_calls > 0

    # Verify Invariant I4: verify_worktree receives ONLY repo_dir and patch_ref_id.
    verify_tasks = [t for t in res.tasks.values() if t.state == "verify_worktree"]
    assert len(verify_tasks) >= 2
    for vt in verify_tasks:
        assert set(vt.input.keys()) == {"repo_dir", "patch_ref_id"}

    # Source repo stays untouched and zero ephemeral worktrees linger.
    git_status = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert git_status.stdout.strip() == ""
    wt_list = subprocess.run(
        ["git", "-C", str(repo_dir), "worktree", "list", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert wt_list.stdout.count("worktree ") == 1

    # Applying the certified patch artifact in a fresh worktree passes 100% of pytest.
    patch_text = store.get_text(out["patch_ref_id"])
    assert "apply_discount" in patch_text and "compute_invoice_total" in patch_text
    with GitWorktreeSandbox(repo_dir) as wt_check:
        wt_check.apply_patch(patch_text)
        check_res = wt_check.run_command([sys.executable, "-m", "pytest", "-q"])
        assert check_res.passed, check_res.stdout + check_res.stderr


def test_repo_fix_honest_uncertified_when_max_attempts_one(tmp_path: Path) -> None:
    repo_dir = create_sample_repo(tmp_path / "billing_fail")
    run_dir = tmp_path / "fail_run"

    res = build_machine(max_attempts=1, race=False).run(
        TaskSpec(state="locate", input={"repo_dir": str(repo_dir)}),
        run_dir=run_dir,
        max_concurrent=3,
    )
    assert res.status == "done"
    out = _unwrap(res.output, ArtifactStore(run_dir / "artifacts"))
    assert out["certified"] is False
    assert out["selected_branch"] is None
    assert out["patch_ref_id"] is None


def test_repo_fix_mocked_live_llm_and_cli_integration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_dir = create_sample_repo(tmp_path / "billing_live")

    fixed_discounts = (
        "from billing.currency import round_cents\n\n"
        "def apply_discount(amount_cents: int, discount_percent: float) -> int:\n"
        "    if discount_percent < 0:\n"
        "        raise ValueError('negative')\n"
        "    clamped = min(float(discount_percent), 100.0)\n"
        "    return max(0, amount_cents - round_cents(amount_cents * (clamped / 100.0)))\n"
    )
    fixed_invoice = (
        "from billing.currency import round_cents\n"
        "from billing.discounts import apply_discount\n\n"
        "def compute_invoice_total(line_items, *, discount_percent=0.0, tax_rate=0.0) -> int:\n"
        "    if tax_rate < 0:\n"
        "        raise ValueError('neg tax')\n"
        "    subtotal = sum(u * q for u, q in line_items)\n"
        "    disc = apply_discount(subtotal, discount_percent)\n"
        "    return 0 if disc == 0 else disc + round_cents(disc * tax_rate)\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        sys_msg = body["messages"][0]["content"]
        user_msg = body["messages"][-1]["content"]
        if "Classify whether a single file slice" in sys_msg:
            is_rel = ("billing/discounts.py" in user_msg) or ("billing/invoice.py" in user_msg)
            payload = {"relevant": is_rel, "reason": "matched" if is_rel else "distractor"}
        elif "branch_0" in user_msg:
            payload = {
                "edits": [
                    {"path": "billing/discounts.py", "content": fixed_discounts},
                    {"path": "billing/invoice.py", "content": fixed_invoice},
                ],
                "rationale": "Fix both discount clamping and pre-tax ordering",
            }
        else:
            payload = {"edits": [], "rationale": "no-op"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
            },
        )

    client = OpenAICompatClient(
        base_url="http://mock.local/v1",
        model="mock-repo-fixer",
        transport=httpx.MockTransport(handler),
    )
    run_dir = tmp_path / "live_run"
    res = build_machine(llm=client, max_attempts=2, race=True).run(
        TaskSpec(state="locate", input={"repo_dir": str(repo_dir)}),
        run_dir=run_dir,
        max_concurrent=3,
    )
    assert res.status == "done"
    out = _unwrap(res.output, ArtifactStore(run_dir / "artifacts"))
    assert out["certified"] is True
    assert out["selected_branch"] == "branch_0"
    assert res.usage.total_tokens > 0

    # CLI run + inspect --tree + artifact check on offline build_machine
    cli_run_dir = tmp_path / "cli_repo_run"
    rc_run = cli_main(
        [
            "run",
            "examples.repo_fix.machine:build_machine",
            "--state",
            "locate",
            "--input",
            json.dumps({"repo_dir": str(repo_dir)}),
            "--run-dir",
            str(cli_run_dir),
            "--json",
        ]
    )
    assert rc_run == 0
    cli_payload = json.loads(capsys.readouterr().out)
    patch_ref_id = cli_payload["output"]["patch_ref_id"]
    assert cli_payload["output"]["certified"] is True

    rc_tree = cli_main(["inspect", str(cli_run_dir), "--tree"])
    assert rc_tree == 0
    assert "locate" in capsys.readouterr().out

    rc_art = cli_main(["artifact", str(cli_run_dir), patch_ref_id])
    assert rc_art == 0
    assert "diff --git" in capsys.readouterr().out

    # Demo script check
    assert demo_main([]) == 0
    assert "certified=True" in capsys.readouterr().out
