"""Offline unit tests for ``ares.ext.git.GitWorktreeSandbox``."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ares.ext.git import CommandResult, GitWorktreeSandbox


def _git(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "Sandbox Test",
            "GIT_AUTHOR_EMAIL": "sandbox@ares.local",
            "GIT_COMMITTER_NAME": "Sandbox Test",
            "GIT_COMMITTER_EMAIL": "sandbox@ares.local",
        }
    )
    return subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("# Demo\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo.resolve()


def test_worktree_lifecycle_and_cleanup_on_exception(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    before = _git(repo, "worktree", "list", "--porcelain").stdout.count("worktree ")
    assert before == 1

    wt_path: Path | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with GitWorktreeSandbox(repo) as wt:
            wt_path = wt.path
            assert wt_path.exists()
            assert "pkg/mod.py" in wt.list_files()
            wt.write_file("pkg/mod.py", "VALUE = 999\n")
            raise RuntimeError("boom")

    assert wt_path is not None and not wt_path.exists()
    after = _git(repo, "worktree", "list", "--porcelain").stdout.count("worktree ")
    assert after == 1
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""
    assert (repo / "pkg" / "mod.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_path_traversal_rejected_and_file_ops(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    with GitWorktreeSandbox(repo) as wt:
        with pytest.raises(ValueError):
            wt.read_file("../outside.txt")
        with pytest.raises(ValueError):
            wt.write_file(str(tmp_path / "abs.txt"), "bad")
        with pytest.raises(ValueError):
            wt.replace_in_file("pkg/mod.py", "MISSING", "X")

        wt.replace_in_file("pkg/mod.py", "VALUE = 1", "VALUE = 2")
        assert wt.read_file("pkg/mod.py") == "VALUE = 2\n"
        assert wt.list_files(extensions=[".py"]) == ["pkg/mod.py"]


def test_diff_including_untracked_and_apply_patch_roundtrip(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    with GitWorktreeSandbox(repo) as wt1:
        wt1.replace_in_file("pkg/mod.py", "VALUE = 1", "VALUE = 42")
        wt1.write_file("pkg/extra.py", "EXTRA = 7\n")
        patch = wt1.diff()
        assert "VALUE = 42" in patch
        assert "EXTRA = 7" in patch

    # Apply the captured patch into a clean second worktree and verify both files.
    with GitWorktreeSandbox(repo) as wt2:
        wt2.apply_patch(patch)
        assert wt2.read_file("pkg/mod.py") == "VALUE = 42\n"
        assert wt2.read_file("pkg/extra.py") == "EXTRA = 7\n"

    assert _git(repo, "status", "--porcelain").stdout.strip() == ""


def test_run_command_pass_fail_and_timeout(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    with GitWorktreeSandbox(repo) as wt:
        ok: CommandResult = wt.run_command([sys.executable, "-c", "print('hello')"])
        assert ok.passed
        assert ok.stdout.strip() == "hello"

        bad = wt.run_command([sys.executable, "-c", "import sys; sys.exit(3)"])
        assert not bad.passed
        assert bad.returncode == 3

        timed = wt.run_command([sys.executable, "-c", "import time; time.sleep(1.0)"], timeout_s=0.05)
        assert not timed.passed
        assert timed.timed_out is True
