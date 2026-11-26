"""Thin CLI wrapper around the ARES v2 StateMachine SDK."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from ares.machine import StateMachine
from ares.store import ArtifactStore, CheckpointStore
from ares.types import ArtifactRef, Event, RunBudget, RunResult, Task, TaskSpec, Usage


def _parse_optional_int(raw: str) -> int | None:
    if raw.strip().lower() in {"none", "null"}:
        return None
    return int(raw)


def _parse_optional_float(raw: str) -> float | None:
    if raw.strip().lower() in {"none", "null"}:
        return None
    return float(raw)


def _load_machine(machine_ref: str) -> StateMachine:
    if ":" not in machine_ref:
        raise ValueError(
            f"Invalid machine reference {machine_ref!r}; expected 'module.path:attr'"
        )
    module_name, attr_path = machine_ref.split(":", 1)
    cwd = os.getcwd()
    if cwd and cwd not in sys.path:
        sys.path.insert(0, cwd)

    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr_path.split("."):
        target = getattr(target, part)

    if isinstance(target, StateMachine):
        return target
    if callable(target):
        instance = target()
        if isinstance(instance, StateMachine):
            return instance
        raise TypeError(
            f"Callable {machine_ref!r} returned {type(instance).__name__}, expected StateMachine"
        )
    raise TypeError(
        f"Reference {machine_ref!r} resolved to {type(target).__name__}, expected StateMachine"
    )


def _parse_provide_items(items: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --provide value {item!r}; expected <key>=<json>")
        key, raw_val = item.split("=", 1)
        if not key:
            raise ValueError(f"Empty key in --provide {item!r}")
        parsed[key] = json.loads(raw_val)
    return parsed


def _format_event_line(event: Event) -> str:
    task_part = f" task={event.task_id}" if event.task_id else ""
    payload_part = f" {json.dumps(event.payload, ensure_ascii=False)}" if event.payload else ""
    return f"[{event.seq:>4}] {event.event_type:<16}{task_part}{payload_part}"


def _total_usage(tasks: dict[str, Task]) -> Usage:
    total = Usage()
    for task in tasks.values():
        total = total.add(task.usage)
    return total


def _print_usage(usage: Usage) -> None:
    if usage.total_tokens > 0 or usage.llm_calls > 0 or usage.cost_usd > 0.0:
        print(
            f"Usage:     calls={usage.llm_calls} tokens={usage.total_tokens} "
            f"(prompt={usage.prompt_tokens}, completion={usage.completion_tokens}) "
            f"cost=${usage.cost_usd:.6f}"
        )


def _print_tasks_table(tasks: dict[str, Task]) -> None:
    print("\nTasks:")
    header = f"  {'ID':<16} {'State':<18} {'Status':<10} {'Depth':>5} {'Att':>3} {'Ms':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for task in tasks.values():
        print(
            f"  {task.id:<16} {task.state:<18} {task.status:<10} "
            f"{task.depth:>5} {task.attempts:>3} {task.duration_ms:>8.1f}"
        )


def _print_tasks_tree(tasks: dict[str, Task], root_task_id: str) -> None:
    children_map: dict[str | None, list[Task]] = {}
    for task in tasks.values():
        parent_key = task.parent_id if task.parent_id in tasks else None
        children_map.setdefault(parent_key, []).append(task)

    print("\nTask Tree:")
    visited: set[str] = set()

    def _walk(tid: str, prefix: str, is_last: bool, is_root: bool) -> None:
        if tid in visited or tid not in tasks:
            return
        visited.add(tid)
        task = tasks[tid]
        branch = "" if is_root else ("└─ " if is_last else "├─ ")
        extras: list[str] = [f"depth={task.depth}", f"{task.duration_ms:.1f}ms"]
        if task.deterministic:
            extras.append("det")
        if task.usage.total_tokens > 0:
            extras.append(f"tok={task.usage.total_tokens}")
        if task.artifact_refs:
            extras.append(f"arts={len(task.artifact_refs)}")
        if task.spilled_artifact_id:
            extras.append(f"ref={task.spilled_artifact_id}")
        print(f"  {prefix}{branch}{task.id} [{task.state}] ({task.status}) {' '.join(extras)}")

        kids = children_map.get(tid, [])
        child_prefix = prefix if is_root else (prefix + ("   " if is_last else "│  "))
        for idx, child in enumerate(kids):
            _walk(child.id, child_prefix, idx == len(kids) - 1, False)

    if root_task_id in tasks:
        _walk(root_task_id, "", True, True)
    for tid in tasks:
        if tid not in visited:
            _walk(tid, "", True, True)


def _print_artifacts(refs: list[ArtifactRef]) -> None:
    if not refs:
        return
    print("\nArtifacts:")
    for ref in refs:
        print(
            f"  {ref.ref_id} ({ref.byte_size} bytes, {ref.mime_type}) "
            f"preview={ref.preview!r}"
        )


def _print_run_summary(result: RunResult) -> None:
    print(f"\nRun {result.run_id}: status={result.status} wall_clock_ms={result.wall_clock_ms:.1f}")
    print(f"Root task: {result.root_task_id}")
    if result.run_dir:
        print(f"Run dir:   {result.run_dir}")
    _print_usage(result.usage)
    if result.suspended_task_ids:
        print(f"Suspended: {', '.join(result.suspended_task_ids)}")
    if result.failed_task_ids:
        print(f"Failed:    {', '.join(result.failed_task_ids)}")
    if result.cancelled_task_ids:
        print(f"Cancelled: {', '.join(result.cancelled_task_ids)}")
    _print_tasks_table(result.tasks)
    if result.run_dir:
        refs = ArtifactStore(Path(result.run_dir) / "artifacts").list_refs()
        _print_artifacts(refs)
    if result.output is not None:
        print(f"\nOutput: {json.dumps(result.output, ensure_ascii=False)}")


def _cmd_run(args: argparse.Namespace) -> int:
    machine = _load_machine(args.machine_ref)
    input_data = json.loads(args.input)
    if not isinstance(input_data, dict):
        raise ValueError("--input JSON must decode to an object/dict")

    budget = RunBudget(
        max_tasks=args.max_tasks,
        max_inline_bytes=args.max_inline_bytes,
        max_tokens=args.max_tokens,
        max_cost_usd=args.max_cost_usd,
    )
    root = TaskSpec(id=args.id, state=args.state, input=input_data)

    on_event = None if args.json else (lambda ev: print(_format_event_line(ev)))
    result = machine.run(
        root,
        run_dir=args.run_dir,
        max_concurrent=args.max_concurrent,
        budget=budget,
        on_event=on_event,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_run_summary(result)
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    machine = _load_machine(args.machine)
    resume_inputs = _parse_provide_items(args.provide)
    on_event = None if args.json else (lambda ev: print(_format_event_line(ev)))
    result = machine.resume(
        args.run_dir,
        resume_inputs=resume_inputs or None,
        max_concurrent=args.max_concurrent,
        on_event=on_event,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_run_summary(result)
    return 0


def _derive_status(tasks: dict[str, Task], root_task_id: str, events: list[Event]) -> str:
    for event in reversed(events):
        if event.event_type == "run_finished" and "status" in event.payload:
            return str(event.payload["status"])
    root = tasks.get(root_task_id)
    if root is not None and root.status in ("done", "failed"):
        return root.status
    if any(t.status == "suspended" for t in tasks.values()):
        return "suspended"
    if any(t.status == "failed" for t in tasks.values()):
        return "failed"
    return "running"


def _cmd_inspect(args: argparse.Namespace) -> int:
    run_path = Path(args.run_dir)
    checkpoint = CheckpointStore(run_path)
    snapshot = checkpoint.load_state()
    if snapshot is None:
        print(f"Error: no checkpoint state.json found in {run_path}", file=sys.stderr)
        return 1

    tasks = {
        tid: Task.model_validate(tdata)
        for tid, tdata in snapshot.get("tasks", {}).items()
    }
    events = checkpoint.load_events()
    refs = ArtifactStore(run_path / "artifacts").list_refs()
    run_id = str(snapshot.get("run_id", ""))
    root_task_id = str(snapshot.get("root_task_id", ""))
    status = _derive_status(tasks, root_task_id, events)
    usage = _total_usage(tasks)

    if args.json:
        payload = {
            "run_id": run_id,
            "root_task_id": root_task_id,
            "status": status,
            "run_dir": str(run_path),
            "usage": usage.model_dump(),
            "tasks": {tid: task.model_dump() for tid, task in tasks.items()},
            "events": [ev.model_dump() for ev in events],
            "artifacts": [ref.model_dump() for ref in refs],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Run {run_id}: status={status} root_task_id={root_task_id}")
        print(f"Run dir: {run_path}")
        _print_usage(usage)
        if args.tree:
            _print_tasks_tree(tasks, root_task_id)
        else:
            _print_tasks_table(tasks)
        _print_artifacts(refs)
        tail = int(getattr(args, "tail", 10))
        if tail < 0:
            raise ValueError(f"--tail must be >= 0, got {tail}")
        shown_events = events if tail == 0 else events[-tail:]
        print(f"\nEvents ({len(events)} total):")
        for ev in shown_events:
            print("  " + _format_event_line(ev))
    return 0


def _cmd_artifact(args: argparse.Namespace) -> int:
    artifacts_dir = Path(args.run_dir) / "artifacts"
    if not artifacts_dir.exists():
        print(f"Error: artifacts directory not found in {args.run_dir}", file=sys.stderr)
        return 1
    store = ArtifactStore(artifacts_dir)
    if store.get_ref(args.ref_id) is None:
        print(f"Error: unknown artifact ref {args.ref_id!r} in {args.run_dir}", file=sys.stderr)
        return 1
    if args.offset is not None or args.limit is not None:
        offset = 0 if args.offset is None else args.offset
        limit = 4096 if args.limit is None else args.limit
        sys.stdout.write(store.read_slice(args.ref_id, offset=offset, limit=limit))
    else:
        sys.stdout.write(store.get_text(args.ref_id))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level ArgumentParser for the ``ares`` CLI."""
    parser = argparse.ArgumentParser(
        prog="ares",
        description="ARES v2: Pushdown Automaton + DAG State Machine CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Execute a StateMachine from a root TaskSpec")
    run_p.add_argument("machine_ref", help="Machine reference in 'module.path:attr' format")
    run_p.add_argument("--state", required=True, help="Initial root state name")
    run_p.add_argument("--id", default=None, help="Optional root task ID")
    run_p.add_argument("--input", default="{}", help="JSON object for root task input")
    run_p.add_argument("--run-dir", default=None, help="Directory for checkpoints and artifacts")
    run_p.add_argument("--max-concurrent", type=int, default=4, help="Worker thread pool size")
    run_p.add_argument("--max-tasks", type=int, default=500, help="Maximum total tasks allowed")
    run_p.add_argument(
        "--max-inline-bytes",
        type=_parse_optional_int,
        default=4096,
        help="Inline output byte limit before spilling to ArtifactStore (or 'none')",
    )
    run_p.add_argument(
        "--max-tokens",
        type=_parse_optional_int,
        default=None,
        help="Maximum cumulative LLM tokens allowed before stopping (or 'none')",
    )
    run_p.add_argument(
        "--max-cost-usd",
        type=_parse_optional_float,
        default=None,
        help="Maximum cumulative LLM cost in USD allowed before stopping (or 'none')",
    )
    run_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    run_p.set_defaults(handler=_cmd_run)

    resume_p = subparsers.add_parser("resume", help="Resume a checkpointed run directory")
    resume_p.add_argument("run_dir", help="Checkpoint directory to resume")
    resume_p.add_argument(
        "--machine",
        required=True,
        help="Machine reference in 'module.path:attr' format",
    )
    resume_p.add_argument(
        "--provide",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="Resume input mapping task_id or resume_key to a JSON value (repeatable)",
    )
    resume_p.add_argument("--max-concurrent", type=int, default=4, help="Worker thread pool size")
    resume_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    resume_p.set_defaults(handler=_cmd_resume)

    inspect_p = subparsers.add_parser("inspect", help="Inspect a checkpointed run directory")
    inspect_p.add_argument("run_dir", help="Checkpoint directory to inspect")
    inspect_p.add_argument(
        "--tree",
        action="store_true",
        help="Display tasks as an indented parent-child DAG tree",
    )
    inspect_p.add_argument(
        "--tail",
        type=int,
        default=10,
        help="Number of trailing events to display (0 for all, default: 10)",
    )
    inspect_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")
    inspect_p.set_defaults(handler=_cmd_inspect)

    art_p = subparsers.add_parser("artifact", help="Print a stored artifact or slice by ref_id")
    art_p.add_argument("run_dir", help="Checkpoint directory containing artifacts/")
    art_p.add_argument("ref_id", help="Artifact reference ID (e.g. art_...)")
    art_p.add_argument("--offset", type=int, default=None, help="Character offset for slice")
    art_p.add_argument("--limit", type=int, default=None, help="Maximum characters to print")
    art_p.set_defaults(handler=_cmd_artifact)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.handler(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
