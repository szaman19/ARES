"""CLI runner demonstrating multi-file bounded repo search & parallel GitWorktreeSandbox fix."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ares import ArtifactStore, TaskSpec
from examples.repo_fix.fixture import create_sample_repo
from examples.repo_fix.machine import build_machine


def _unwrap(value: Any, store: ArtifactStore) -> Any:
    if ArtifactStore.is_artifact_handle(value):
        ref_id = ArtifactStore.extract_ref_id(value)
        if ref_id is not None:
            return store.get_json(ref_id)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ARES repo_fix multi-file GitWorktreeSandbox demo")
    parser.add_argument("--no-race", action="store_true", help="Disable join='any' sibling cancellation")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    with tempfile.TemporaryDirectory(prefix="ares_repo_fix_") as tmpdir:
        tmp_path = Path(tmpdir)
        repo_dir = create_sample_repo(tmp_path / "sample_repo")
        run_dir = tmp_path / "ares_run"

        machine = build_machine(max_attempts=args.max_attempts, race=not args.no_race)
        res = machine.run(
            TaskSpec(state="locate", input={"repo_dir": str(repo_dir)}),
            run_dir=run_dir,
            max_concurrent=4,
        )

        store = ArtifactStore(run_dir / "artifacts")
        out = _unwrap(res.output, store)
        print(
            f"Run {res.run_id} | status={res.status} | wall_clock_ms={res.wall_clock_ms:.1f} | "
            f"llm_calls={res.usage.llm_calls} tokens={res.usage.total_tokens}"
        )
        print(f"Relevant files: {out.get('relevant_files')}")
        print(f"Ignored files:  {out.get('ignored_files')}")
        print(f"Cancelled tasks ({len(res.cancelled_task_ids)}): {res.cancelled_task_ids}")
        print(
            f"Selected: branch={out.get('selected_branch')} attempts={out.get('attempts')} "
            f"certified={out.get('certified')}"
        )
        patch_ref_id = out.get("patch_ref_id")
        if patch_ref_id:
            print("\nCertified unified git diff:")
            print("-" * 60)
            print(store.get_text(patch_ref_id).rstrip())
            print("-" * 60)
        return 0 if out.get("certified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
