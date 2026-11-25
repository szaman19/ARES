"""Runnable demo for the ARES codegen-verify-repair PDA + DAG example."""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from ares import ArtifactStore, RunResult, TaskSpec
from examples.codegen_verify_repair.machine import TASKS, build_machine


def _unwrap_output(value: Any, store: ArtifactStore) -> Any:
    if ArtifactStore.is_artifact_handle(value):
        ref_id = ArtifactStore.extract_ref_id(value)
        if ref_id is not None:
            return store.get_json(ref_id)
    return value


def _collect_branch_trajectories(
    result: RunResult,
    store: ArtifactStore,
) -> dict[str, list[dict[str, Any]]]:
    """Collect per-branch verification/repair attempt records from completed tasks."""
    verify_by_id: dict[str, dict[str, Any]] = {}
    for task in result.tasks.values():
        if task.state == "verify_pytest" and task.status == "done":
            verify_by_id[task.id] = _unwrap_output(task.output, store)

    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in result.tasks.values():
        if task.state != "decide_repair":
            continue
        branch_id = str(task.input["branch_id"])
        attempt = int(task.input["attempt"])
        code_ref_id = str(task.input["code_ref_id"])
        verify_out = next(
            (verify_by_id[dep_id] for dep_id in task.depends_on if dep_id in verify_by_id),
            {},
        )
        trajectories[branch_id].append(
            {
                "attempt": attempt,
                "passed": bool(verify_out.get("passed", False)),
                "returncode": verify_out.get("returncode"),
                "code_ref_id": code_ref_id,
            }
        )

    for attempts in trajectories.values():
        attempts.sort(key=lambda entry: int(entry["attempt"]))
    return dict(sorted(trajectories.items()))


def run_demo(
    task_name: str = "slugify",
    max_attempts: int = 3,
    run_dir: str | Path | None = None,
) -> RunResult:
    """Execute the codegen-verify-repair demo and print a structured report."""
    machine = build_machine(max_attempts=max_attempts)
    root = TaskSpec(state="plan", input={"task": task_name, "max_attempts": max_attempts})

    if run_dir is not None:
        resolved_dir = Path(run_dir)
        result = machine.run(root, run_dir=resolved_dir)
        _print_report(result, ArtifactStore(resolved_dir / "artifacts"))
        return result

    with tempfile.TemporaryDirectory(prefix="ares_codegen_demo_") as tmpdir:
        resolved_dir = Path(tmpdir)
        result = machine.run(root, run_dir=resolved_dir)
        _print_report(result, ArtifactStore(resolved_dir / "artifacts"))
        return result


def _print_report(result: RunResult, store: ArtifactStore) -> None:
    output = _unwrap_output(result.output, store) or {}
    print(
        f"Run {result.run_id} | task={output.get('task')} | "
        f"status={result.status} | wall_clock_ms={result.wall_clock_ms:.1f}"
    )

    trajectories = _collect_branch_trajectories(result, store)
    print("\nBranch repair trajectories:")
    for branch_id, entries in trajectories.items():
        steps = ", ".join(
            f"attempt {e['attempt']}: {'PASS' if e['passed'] else 'FAIL'} ({e['code_ref_id']})"
            for e in entries
        )
        print(f"  {branch_id}: {steps}")

    print("\nFinal selection:")
    print(f"  certified={output.get('certified')}")
    print(f"  selected_branch={output.get('selected_branch')}")
    print(f"  code_ref_id={output.get('code_ref_id')}")

    code_ref_id = output.get("code_ref_id")
    if code_ref_id:
        certified_code = store.get_text(str(code_ref_id))
        print("\nCertified code from ArtifactStore:")
        print("-" * 50)
        print(certified_code.rstrip())
        print("-" * 50)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the standalone codegen-verify-repair demo."""
    parser = argparse.ArgumentParser(
        description="Run the ARES codegen-verify-repair PDA + DAG demo.",
    )
    parser.add_argument(
        "--task",
        default="slugify",
        choices=sorted(TASKS),
        help="Built-in coding task to execute",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum codegen/repair attempts per parallel branch",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Optional persistent run directory (uses temporary dir if omitted)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_demo(task_name=args.task, max_attempts=args.max_attempts, run_dir=args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
