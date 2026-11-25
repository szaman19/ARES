"""Ephemeral git worktree sandbox for parallel code generation and deterministic verification."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel


class CommandResult(BaseModel):
    """Structured result from running a command inside a sandbox."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """Return True iff the command exited 0 without timing out."""
        return not self.timed_out and self.returncode == 0


class GitWorktreeSandbox:
    """Ephemeral detached ``git worktree`` sandbox with path-traversal guards."""

    def __init__(
        self,
        repo_dir: str | Path,
        *,
        base_ref: str = "HEAD",
        prefix: str = "ares_wt_",
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.base_ref = base_ref
        self.prefix = prefix
        self._temp_root: Path | None = None
        self._worktree_dir: Path | None = None

    @property
    def path(self) -> Path:
        """Return the active worktree directory path."""
        if self._worktree_dir is None:
            raise RuntimeError("GitWorktreeSandbox is not open; call open() or use 'with'.")
        return self._worktree_dir

    def open(self) -> Path:
        """Create and detach a temporary git worktree from ``repo_dir``."""
        if self._worktree_dir is not None:
            return self._worktree_dir
        if not (self.repo_dir / ".git").exists():
            raise ValueError(f"Not a git repository: {self.repo_dir}")

        temp_root = Path(tempfile.mkdtemp(prefix=self.prefix)).resolve()
        wt_dir = temp_root / "wt"
        proc = subprocess.run(
            ["git", "-C", str(self.repo_dir), "worktree", "add", "--detach", str(wt_dir), self.base_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise RuntimeError(
                f"Failed to create git worktree at {wt_dir}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        self._temp_root = temp_root
        self._worktree_dir = wt_dir.resolve()
        return self._worktree_dir

    def close(self) -> None:
        """Force-remove the ephemeral git worktree and prune stale metadata."""
        wt_dir = self._worktree_dir
        temp_root = self._temp_root
        self._worktree_dir = None
        self._temp_root = None

        if wt_dir is not None:
            subprocess.run(
                ["git", "-C", str(self.repo_dir), "worktree", "remove", "--force", str(wt_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
        subprocess.run(
            ["git", "-C", str(self.repo_dir), "worktree", "prune"],
            capture_output=True,
            text=True,
            check=False,
        )

    def __enter__(self) -> GitWorktreeSandbox:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _resolve(self, rel_path: str | Path) -> Path:
        root = self.path.resolve()
        rel = Path(rel_path)
        if rel.is_absolute():
            raise ValueError(f"Path must be relative to worktree: {rel_path!r}")
        # Check intermediate parents before resolving to reject symlink escapes too.
        candidate = root / rel
        resolved = candidate.resolve()
        if resolved != root and not resolved.is_relative_to(root):
            raise ValueError(f"Path escapes worktree sandbox: {rel_path!r}")
        return resolved

    def list_files(
        self,
        *,
        extensions: Sequence[str] | None = None,
        ignore_dirs: Sequence[str] = (".git", "__pycache__", ".venv", ".pytest_cache"),
    ) -> list[str]:
        """Return sorted worktree-relative POSIX file paths."""
        root = self.path.resolve()
        ignored = set(ignore_dirs)
        ext_set = (
            {e if e.startswith(".") else f".{e}" for e in extensions}
            if extensions is not None
            else None
        )
        results: list[str] = []
        for cur_root, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in ignored)
            cur_path = Path(cur_root)
            for fname in sorted(files):
                fpath = cur_path / fname
                if ext_set is not None and fpath.suffix not in ext_set:
                    continue
                results.append(fpath.relative_to(root).as_posix())
        return sorted(results)

    def read_file(self, rel_path: str | Path) -> str:
        """Read a UTF-8 file inside the worktree."""
        return self._resolve(rel_path).read_text(encoding="utf-8")

    def write_file(self, rel_path: str | Path, content: str) -> Path:
        """Write a UTF-8 file inside the worktree, creating parent directories if needed."""
        target = self._resolve(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def replace_in_file(self, rel_path: str | Path, old: str, new: str) -> None:
        """Replace ``old`` with ``new`` in a worktree file; raise ValueError if missing."""
        target = self._resolve(rel_path)
        text = target.read_text(encoding="utf-8")
        if old not in text:
            raise ValueError(f"Target text not found in {rel_path!r}")
        target.write_text(text.replace(old, new, 1), encoding="utf-8")

    def diff(self, *, include_untracked: bool = True) -> str:
        """Return unified ``git diff HEAD`` (optionally including new untracked files)."""
        wt = self.path
        if include_untracked:
            untracked_proc = subprocess.run(
                ["git", "-C", str(wt), "ls-files", "--others", "--exclude-standard"],
                capture_output=True,
                text=True,
                check=False,
            )
            untracked = [
                line.strip()
                for line in untracked_proc.stdout.splitlines()
                if line.strip()
                and "__pycache__" not in line
                and ".pytest_cache" not in line
                and not line.endswith(".pyc")
            ]
            if untracked:
                subprocess.run(
                    ["git", "-C", str(wt), "add", "-N", "--", *untracked],
                    capture_output=True,
                    text=True,
                    check=False,
                )

        diff_proc = subprocess.run(
            ["git", "-C", str(wt), "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.returncode != 0:
            raise RuntimeError(f"git diff failed: {diff_proc.stderr.strip()}")
        return diff_proc.stdout

    def apply_patch(self, patch_text: str) -> None:
        """Apply a unified diff patch inside the worktree."""
        if not patch_text.strip():
            return
        proc = subprocess.run(
            ["git", "-C", str(self.path), "apply", "--whitespace=nowarn", "-"],
            input=patch_text,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git apply failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )

    def run_command(
        self,
        cmd: Sequence[str],
        *,
        timeout_s: float = 30.0,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run a subprocess inside the active worktree with timeout capture."""
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=self.path,
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            return CommandResult(
                command=list(cmd),
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                timed_out=False,
                duration_ms=round(duration_ms, 3),
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            stdout = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return CommandResult(
                command=list(cmd),
                returncode=-1,
                stdout=stdout,
                stderr=(stderr + f"\nTimed out after {timeout_s}s").strip(),
                timed_out=True,
                duration_ms=round(duration_ms, 3),
            )
