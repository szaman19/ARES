"""Pushdown Automaton (PDA) + Dependency DAG execution engine."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, TypeVar

from ares.store import ArtifactStore, CheckpointStore
from ares.types import (
    Done,
    Event,
    EventType,
    Fail,
    JoinMode,
    RunBudget,
    RunResult,
    Spawn,
    Suspend,
    Task,
    TaskSpec,
    Transition,
    Usage,
)

F = TypeVar("F", bound=Callable[..., Transition])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_dedup_key(task: Task, tasks: dict[str, Task]) -> str:
    payload = {
        "state": task.state,
        "input": task.input,
        "deps": {dep_id: tasks[dep_id].output for dep_id in sorted(task.depends_on)},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return f"auto:{hashlib.sha256(raw).hexdigest()}"


class TaskContext:
    """Execution context passed to each state handler invocation."""

    def __init__(
        self,
        *,
        store: ArtifactStore,
        resume_data: Any = None,
        task_id: str | None = None,
        run_id: str | None = None,
        emit_fn: Callable[[EventType, str | None, dict[str, Any] | None], Event] | None = None,
    ) -> None:
        self.store = store
        self.resume_data = resume_data
        self.task_id = task_id
        self.run_id = run_id
        self._emit_fn = emit_fn
        self._usage = Usage()
        self._usage_lock = Lock()

    def record_usage(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
        llm_calls: int = 1,
        cost_usd: float = 0.0,
    ) -> Usage:
        """Record LLM token, call, and cost usage for the current task."""
        resolved_total = (
            total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
        )
        delta = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=resolved_total,
            llm_calls=llm_calls,
            cost_usd=cost_usd,
        )
        with self._usage_lock:
            self._usage = self._usage.add(delta)
            return self._usage

    def recorded_usage(self) -> Usage:
        """Return cumulative Usage recorded on this TaskContext."""
        with self._usage_lock:
            return self._usage.model_copy()

    def emit(
        self,
        event_type: EventType,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        """Emit a structured lifecycle event scoped to the current task."""
        if self._emit_fn is None:
            return Event(
                seq=0,
                timestamp=_utc_now_iso(),
                event_type=event_type,
                task_id=self.task_id,
                payload=payload or {},
            )
        return self._emit_fn(event_type, self.task_id, payload)


@dataclass(slots=True)
class StateConfig:
    """Registered handler configuration for a named machine state."""

    name: str
    fn: Callable[[Task, dict[str, Any], TaskContext], Transition]
    deterministic: bool = False
    max_retries: int = 0
    deduplicate: bool = False


class _RunSession:
    """Mutable per-run coordinator owned by the main scheduler loop."""

    def __init__(
        self,
        *,
        machine: StateMachine,
        run_id: str,
        root_task_id: str,
        tasks: dict[str, Task],
        store: ArtifactStore,
        checkpoint: CheckpointStore | None,
        budget: RunBudget,
        max_concurrent: int,
        on_event: Callable[[Event], None] | None,
        task_seq: int = 0,
        event_seq: int = 0,
        run_dir: str | None = None,
    ) -> None:
        self.machine = machine
        self.run_id = run_id
        self.root_task_id = root_task_id
        self.tasks = tasks
        self.store = store
        self.checkpoint = checkpoint
        self.budget = budget
        self.max_concurrent = max(1, max_concurrent)
        self.on_event = on_event
        self.task_seq = task_seq
        self.event_seq = event_seq
        self.run_dir = run_dir
        self._event_lock = Lock()
        self._start_monotonic = time.monotonic()
        self.budget_exceeded = False

    def emit(
        self,
        event_type: EventType,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Event:
        with self._event_lock:
            self.event_seq += 1
            event = Event(
                seq=self.event_seq,
                timestamp=_utc_now_iso(),
                event_type=event_type,
                task_id=task_id,
                payload=dict(payload) if payload else {},
            )
            if self.checkpoint is not None:
                self.checkpoint.append_event(event)
            if self.on_event is not None:
                self.on_event(event)
            return event

    def save_checkpoint(self) -> None:
        if self.checkpoint is None:
            return
        self.emit("checkpoint_saved", payload={"task_count": len(self.tasks)})
        snapshot = {
            "version": 2,
            "run_id": self.run_id,
            "root_task_id": self.root_task_id,
            "task_seq": self.task_seq,
            "event_seq": self.event_seq,
            "budget": self.budget.model_dump(),
            "tasks": {tid: task.model_dump() for tid, task in self.tasks.items()},
        }
        self.checkpoint.save_state(snapshot)

    def _next_task_id(self) -> str:
        while True:
            self.task_seq += 1
            candidate = f"task_{self.task_seq}"
            if candidate not in self.tasks:
                return candidate

    def _ancestor_ids(self, task_id: str | None) -> set[str]:
        ancestors: set[str] = set()
        cur = task_id
        while cur is not None:
            ancestors.add(cur)
            parent = self.tasks.get(cur)
            cur = parent.parent_id if parent is not None else None
        return ancestors

    def _validate_no_cycles(self, candidate_tasks: dict[str, Task]) -> None:
        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited

        def dfs(node_id: str) -> None:
            mark = state.get(node_id, 0)
            if mark == 1:
                raise ValueError(f"Cycle detected in task dependencies involving {node_id!r}")
            if mark == 2:
                return
            state[node_id] = 1
            node = candidate_tasks[node_id]
            for dep_id in node.depends_on:
                dfs(dep_id)
            if node.continuation_id is not None:
                dfs(node.continuation_id)
            for child_id in node.wait_for:
                dfs(child_id)
            state[node_id] = 2

        for tid in candidate_tasks:
            if state.get(tid, 0) == 0:
                dfs(tid)

    def enqueue_batch(
        self,
        specs: list[TaskSpec],
        *,
        parent_id: str | None,
        depth: int,
        then_depth: int | None = None,
        then_spec: TaskSpec | None = None,
        join_mode: JoinMode = "all",
    ) -> tuple[list[Task], Task | None]:
        """Materialize specs + optional ``then`` continuation and validate DAG invariants."""
        eff_then_depth = depth if then_depth is None else then_depth
        assigned_ids: list[str] = []
        batch_ids: set[str] = set()

        def allocate_id(spec: TaskSpec) -> str:
            if spec.id is not None:
                if spec.id in self.tasks or spec.id in batch_ids:
                    raise ValueError(f"Duplicate task id: {spec.id!r}")
                batch_ids.add(spec.id)
                return spec.id
            while True:
                candidate = self._next_task_id()
                if candidate not in batch_ids:
                    batch_ids.add(candidate)
                    return candidate

        for spec in specs:
            assigned_ids.append(allocate_id(spec))

        then_id: str | None = None
        if then_spec is not None:
            then_id = allocate_id(then_spec)

        new_tasks: dict[str, Task] = {}
        ancestors = self._ancestor_ids(parent_id)

        for spec, tid in zip(specs, assigned_ids, strict=True):
            deps = list(dict.fromkeys(spec.depends_on))
            new_tasks[tid] = Task(
                id=tid,
                state=spec.state,
                input=dict(spec.input),
                depends_on=deps,
                parent_id=parent_id,
                depth=depth,
                dedup_key=spec.dedup_key,
                metadata=dict(spec.metadata),
            )

        then_task: Task | None = None
        if then_spec is not None and then_id is not None:
            if join_mode == "any":
                then_deps = list(dict.fromkeys(then_spec.depends_on))
                then_status: Any = "waiting"
            else:
                then_deps = list(dict.fromkeys([*then_spec.depends_on, *assigned_ids]))
                then_status = "pending"
            then_task = Task(
                id=then_id,
                state=then_spec.state,
                input=dict(then_spec.input),
                depends_on=then_deps,
                parent_id=parent_id,
                depth=eff_then_depth,
                status=then_status,
                dedup_key=then_spec.dedup_key,
                metadata=dict(then_spec.metadata),
            )
            new_tasks[then_id] = then_task

        combined = {**self.tasks, **new_tasks}
        for tid, task in new_tasks.items():
            for dep_id in task.depends_on:
                if dep_id == tid:
                    raise ValueError(f"Task {tid!r} cannot depend on itself")
                if dep_id not in combined:
                    raise ValueError(f"Task {tid!r} depends on unknown task {dep_id!r}")
                if dep_id in ancestors and combined[dep_id].status != "done":
                    raise ValueError(
                        f"Task {tid!r} depends on active ancestor {dep_id!r}, creating a cycle"
                    )

        self._validate_no_cycles(combined)

        self.tasks.update(new_tasks)
        children = [new_tasks[cid] for cid in assigned_ids]
        for task in children:
            self.emit(
                "task_enqueued",
                task_id=task.id,
                payload={"state": task.state, "depth": task.depth, "parent_id": parent_id},
            )
        if then_task is not None:
            self.emit(
                "task_enqueued",
                task_id=then_task.id,
                payload={
                    "state": then_task.state,
                    "depth": then_task.depth,
                    "parent_id": parent_id,
                    "is_continuation": True,
                },
            )
        return children, then_task

    def _has_cancelled_ancestor(self, task: Task) -> bool:
        cur = task.parent_id
        while cur is not None:
            parent = self.tasks.get(cur)
            if parent is None:
                return False
            if parent.status == "cancelled":
                return True
            cur = parent.parent_id
        return False

    def _cancel_subtree(self, root_id: str, *, reason: str) -> None:
        if root_id not in self.tasks:
            return
        children_by_parent: dict[str, list[str]] = {}
        for tid, t in self.tasks.items():
            if t.parent_id is not None:
                children_by_parent.setdefault(t.parent_id, []).append(tid)

        queue: list[str] = [root_id]
        visited: set[str] = set()
        while queue:
            tid = queue.pop(0)
            if tid in visited:
                continue
            visited.add(tid)
            t = self.tasks.get(tid)
            if t is None:
                continue
            for child_id in children_by_parent.get(tid, ()):
                if child_id not in visited:
                    queue.append(child_id)
            if t.continuation_id is not None and t.continuation_id not in visited:
                queue.append(t.continuation_id)
            for wf_id in t.wait_for:
                if wf_id not in visited:
                    queue.append(wf_id)

            if t.status in ("pending", "waiting", "suspended", "running"):
                t.status = "cancelled"
                t.suspend_reason = None
                self.emit(
                    "task_cancelled",
                    task_id=t.id,
                    payload={"reason": reason},
                )

    def propagate_pops(self) -> None:
        """Run fixed-point PDA pop propagation, dependency failure, and race resolution."""
        changed = True
        while changed:
            changed = False

            # 1. Propagate dependency failures/cancellations to pending tasks.
            for task in list(self.tasks.values()):
                if task.status != "pending":
                    continue
                failed_dep = next(
                    (
                        self.tasks[dep_id]
                        for dep_id in task.depends_on
                        if self.tasks[dep_id].status in ("failed", "cancelled")
                    ),
                    None,
                )
                if failed_dep is not None:
                    task.status = "failed"
                    task.error = (
                        f"Dependency {failed_dep.id} {failed_dep.status}: {failed_dep.error or ''}"
                    ).strip()
                    self.emit(
                        "task_failed",
                        task_id=task.id,
                        payload={"failed_dependency": failed_dep.id, "error": task.error},
                    )
                    changed = True

            # 2. Resolve waiting PDA parents (race join="any" first, then join="all").
            for task in list(self.tasks.values()):
                if task.status != "waiting":
                    continue

                if task.join_mode == "any" and task.wait_for:
                    children = [self.tasks[cid] for cid in task.wait_for]
                    winner = next((c for c in children if c.status == "done"), None)
                    if winner is not None:
                        for other in children:
                            if other.id != winner.id:
                                self._cancel_subtree(other.id, reason=f"race_won_by:{winner.id}")
                        if task.continuation_id is not None:
                            cont = self.tasks[task.continuation_id]
                            if winner.id not in cont.depends_on:
                                cont.depends_on.append(winner.id)
                            if cont.status == "waiting":
                                cont.status = "pending"
                            task.wait_for = []
                            task.join_mode = "all"
                        else:
                            task.output = winner.output
                            task.spilled_artifact_id = winner.spilled_artifact_id
                            task.wait_for = []
                            task.status = "done"
                            self.emit(
                                "task_done",
                                task_id=task.id,
                                payload={"race_winner": winner.id},
                            )
                        changed = True
                        continue

                    if all(c.status in ("failed", "cancelled") for c in children):
                        first_fail = next((c for c in children if c.status == "failed"), children[0])
                        err = first_fail.error or "All race children failed"
                        if task.continuation_id is not None:
                            self._cancel_subtree(
                                task.continuation_id,
                                reason="all_race_children_failed",
                            )
                        task.status = "failed"
                        task.error = err
                        task.wait_for = []
                        self.emit(
                            "task_failed",
                            task_id=task.id,
                            payload={"failed_child": first_fail.id, "error": err},
                        )
                        changed = True
                        continue

                    # Race still active; wait for a winner or all-failed.
                    continue

                if task.continuation_id is not None:
                    cont = self.tasks[task.continuation_id]
                    if cont.status == "done":
                        task.output = cont.output
                        task.spilled_artifact_id = cont.spilled_artifact_id
                        task.status = "done"
                        self.emit(
                            "task_done",
                            task_id=task.id,
                            payload={"popped_from": cont.id},
                        )
                        changed = True
                    elif cont.status == "failed":
                        task.status = "failed"
                        task.error = cont.error
                        self.emit(
                            "task_failed",
                            task_id=task.id,
                            payload={"failed_child": cont.id, "error": cont.error},
                        )
                        changed = True
                    elif cont.status == "cancelled":
                        task.status = "cancelled"
                        self.emit(
                            "task_cancelled",
                            task_id=task.id,
                            payload={"reason": f"continuation_cancelled:{cont.id}"},
                        )
                        changed = True
                elif task.wait_for:
                    children = [self.tasks[cid] for cid in task.wait_for]
                    failed_children = [c for c in children if c.status == "failed"]
                    if failed_children:
                        first_fail = failed_children[0]
                        task.status = "failed"
                        task.error = first_fail.error or f"Child task {first_fail.id} failed"
                        self.emit(
                            "task_failed",
                            task_id=task.id,
                            payload={"failed_child": first_fail.id, "error": task.error},
                        )
                        changed = True
                    elif all(c.status == "done" for c in children):
                        gathered = {c.id: c.output for c in children}
                        output_val, ref = self.store.maybe_spill(
                            gathered,
                            self.budget.max_inline_bytes,
                        )
                        task.output = output_val
                        if ref is not None:
                            task.spilled_artifact_id = ref.ref_id
                        task.status = "done"
                        self.emit(
                            "task_done",
                            task_id=task.id,
                            payload={"gathered": list(task.wait_for)},
                        )
                        changed = True

    def _done_by_dedup_key(self) -> dict[str, Task]:
        mapping: dict[str, Task] = {}
        for task in self.tasks.values():
            if task.status == "done" and task.dedup_key is not None and task.dedup_key not in mapping:
                mapping[task.dedup_key] = task
        return mapping

    def _select_ready_wave(self) -> list[Task]:
        """Resolve immediate deduplications and return the next concurrent wave."""
        while True:
            done_dedup = self._done_by_dedup_key()
            wave_keys: set[str] = set()
            wave: list[Task] = []
            resolved_any = False

            for task in self.tasks.values():
                if task.status != "pending":
                    continue
                if not all(self.tasks[dep_id].status == "done" for dep_id in task.depends_on):
                    continue

                cfg = self.machine._states.get(task.state)
                if task.dedup_key is None and cfg is not None and cfg.deduplicate:
                    task.dedup_key = _compute_dedup_key(task, self.tasks)

                key = task.dedup_key
                if key is not None:
                    if key in done_dedup:
                        source = done_dedup[key]
                        task.output = source.output
                        task.spilled_artifact_id = source.spilled_artifact_id
                        task.status = "done"
                        self.emit(
                            "task_deduped",
                            task_id=task.id,
                            payload={"dedup_key": key, "source_task_id": source.id},
                        )
                        resolved_any = True
                        continue
                    if key in wave_keys:
                        # Defer duplicate in the same wave so it deduplicates next tick.
                        continue
                    wave_keys.add(key)

                wave.append(task)

            if resolved_any:
                self.propagate_pops()
                continue
            return wave

    def _execute_task(
        self, task_snapshot: Task, deps: dict[str, Any]
    ) -> tuple[Transition, float, Usage]:
        start = time.perf_counter()
        cfg = self.machine._states.get(task_snapshot.state)
        if cfg is None:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return (
                Fail(error=f"Unknown state: {task_snapshot.state!r}", retryable=False),
                elapsed_ms,
                Usage(),
            )

        ctx = TaskContext(
            store=self.store,
            resume_data=task_snapshot.resume_data,
            task_id=task_snapshot.id,
            run_id=self.run_id,
            emit_fn=self.emit,
        )
        try:
            result = cfg.fn(task_snapshot, deps, ctx)
            if not isinstance(result, (Done, Spawn, Suspend, Fail)):
                transition: Transition = Fail(
                    error=f"State {task_snapshot.state!r} returned invalid transition {type(result).__name__}",
                    retryable=False,
                )
            else:
                transition = result
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            transition = Fail(error=message, retryable=False)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return transition, elapsed_ms, ctx.recorded_usage()

    def _apply_transition(self, task: Task, transition: Transition) -> None:
        if isinstance(transition, Done):
            output_val, ref = self.store.maybe_spill(
                transition.output,
                self.budget.max_inline_bytes,
            )
            task.output = output_val
            if ref is not None:
                task.spilled_artifact_id = ref.ref_id
            task.error = None
            task.status = "done"
            self.emit(
                "task_done",
                task_id=task.id,
                payload={"spilled": ref.ref_id if ref is not None else None},
            )
            return

        if isinstance(transition, Suspend):
            task.status = "suspended"
            task.suspend_reason = transition.reason
            task.resume_key = transition.resume_key
            task.complete_on_resume = transition.complete_on_resume
            self.emit(
                "task_suspended",
                task_id=task.id,
                payload={"reason": transition.reason, "resume_key": transition.resume_key},
            )
            return

        if isinstance(transition, Fail):
            cfg = self.machine._states.get(task.state)
            max_retries = cfg.max_retries if cfg is not None else 0
            task.error = transition.error
            if transition.retryable and task.attempts <= max_retries:
                task.status = "pending"
                self.emit(
                    "task_retried",
                    task_id=task.id,
                    payload={
                        "attempt": task.attempts,
                        "max_retries": max_retries,
                        "error": transition.error,
                    },
                )
            else:
                task.status = "failed"
                self.emit(
                    "task_failed",
                    task_id=task.id,
                    payload={"error": transition.error, "attempts": task.attempts},
                )
            return

        # Spawn transition
        children_specs = list(transition.tasks)
        then_spec = transition.then
        eff_join_mode: JoinMode = transition.join if children_specs else "all"

        if not children_specs and then_spec is None:
            task.output = None
            task.status = "done"
            self.emit("task_done", task_id=task.id, payload={"empty_spawn": True})
            return

        new_count = len(children_specs) + (1 if then_spec is not None else 0)
        if children_specs and task.depth + 1 > self.budget.max_depth:
            task.status = "pending"
            self.budget_exceeded = True
            return
        if len(self.tasks) + new_count > self.budget.max_tasks:
            task.status = "pending"
            self.budget_exceeded = True
            return

        children, then_task = self.enqueue_batch(
            children_specs,
            parent_id=task.id,
            depth=task.depth + 1,
            then_depth=task.depth,
            then_spec=then_spec,
            join_mode=eff_join_mode,
        )
        child_ids = [c.id for c in children]
        task.status = "waiting"
        task.join_mode = eff_join_mode
        if eff_join_mode == "any":
            task.wait_for = child_ids
            if then_task is not None:
                task.continuation_id = then_task.id
            self.emit(
                "task_spawned",
                task_id=task.id,
                payload={
                    "children": child_ids,
                    "then": then_task.id if then_task is not None else None,
                    "join": "any",
                },
            )
        elif then_task is not None:
            task.continuation_id = then_task.id
            self.emit(
                "task_spawned",
                task_id=task.id,
                payload={"children": child_ids, "then": then_task.id},
            )
        else:
            task.wait_for = child_ids
            self.emit(
                "task_spawned",
                task_id=task.id,
                payload={"children": child_ids, "then": None},
            )

    def _wall_clock_exceeded(self) -> bool:
        return (time.monotonic() - self._start_monotonic) > self.budget.max_wall_clock_s

    def _total_usage(self) -> Usage:
        total = Usage()
        for task in self.tasks.values():
            total = total.add(task.usage)
        return total

    def _usage_budget_exceeded(self) -> bool:
        if self.budget.max_tokens is None and self.budget.max_cost_usd is None:
            return False
        total = self._total_usage()
        if self.budget.max_tokens is not None and total.total_tokens > self.budget.max_tokens:
            return True
        if self.budget.max_cost_usd is not None and total.cost_usd > self.budget.max_cost_usd:
            return True
        return False

    def run_loop(self) -> RunResult:
        root_task = self.tasks[self.root_task_id]

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            while root_task.status not in ("done", "failed", "cancelled"):
                if self._wall_clock_exceeded() or self._usage_budget_exceeded():
                    self.budget_exceeded = True
                    break

                wave = self._select_ready_wave()
                if root_task.status in ("done", "failed", "cancelled"):
                    break
                if not wave:
                    break

                if self._wall_clock_exceeded() or self._usage_budget_exceeded():
                    self.budget_exceeded = True
                    break

                futures: list[tuple[Task, Future[tuple[Transition, float, Usage]]]] = []
                for task in wave:
                    task.status = "running"
                    task.attempts += 1
                    self.emit(
                        "task_started",
                        task_id=task.id,
                        payload={"state": task.state, "attempt": task.attempts},
                    )
                    deps = {dep_id: self.tasks[dep_id].output for dep_id in task.depends_on}
                    snapshot = task.model_copy(deep=True)
                    futures.append((task, executor.submit(self._execute_task, snapshot, deps)))

                for task, future in futures:
                    transition, elapsed_ms, recorded_usage = future.result()
                    task.duration_ms += elapsed_ms
                    task.usage = task.usage.add(recorded_usage)
                    if isinstance(transition, Done) and transition.usage is not None:
                        task.usage = task.usage.add(transition.usage)
                    if task.status == "cancelled" or self._has_cancelled_ancestor(task):
                        if task.status != "cancelled":
                            task.status = "cancelled"
                            self.emit(
                                "task_cancelled",
                                task_id=task.id,
                                payload={"reason": "ancestor_cancelled"},
                            )
                    else:
                        self._apply_transition(task, transition)

                self.propagate_pops()
                if self._usage_budget_exceeded():
                    self.budget_exceeded = True
                self.save_checkpoint()

                if self.budget_exceeded:
                    break

        suspended_ids = [tid for tid, t in self.tasks.items() if t.status == "suspended"]
        failed_ids = [tid for tid, t in self.tasks.items() if t.status == "failed"]
        cancelled_ids = [tid for tid, t in self.tasks.items() if t.status == "cancelled"]
        total_usage = self._total_usage()

        if self.budget_exceeded:
            status: Any = "budget_exceeded"
        elif root_task.status == "done":
            status = "done"
        elif root_task.status in ("failed", "cancelled"):
            status = "failed"
        elif suspended_ids:
            status = "suspended"
        elif failed_ids:
            status = "failed"
        else:
            status = "failed"

        wall_clock_ms = (time.monotonic() - self._start_monotonic) * 1000.0
        self.emit(
            "run_finished",
            task_id=self.root_task_id,
            payload={"status": status, "wall_clock_ms": wall_clock_ms},
        )
        self.save_checkpoint()

        return RunResult(
            run_id=self.run_id,
            status=status,
            root_task_id=self.root_task_id,
            output=root_task.output if status == "done" else None,
            tasks=self.tasks,
            suspended_task_ids=suspended_ids,
            failed_task_ids=failed_ids,
            cancelled_task_ids=cancelled_ids,
            usage=total_usage,
            run_dir=self.run_dir,
            wall_clock_ms=wall_clock_ms,
        )


class StateMachine:
    """Pushdown Automaton + Dependency DAG state machine orchestrator."""

    def __init__(self) -> None:
        self._states: dict[str, StateConfig] = {}

    def register_state(
        self,
        name: str,
        fn: Callable[[Task, dict[str, Any], TaskContext], Transition],
        *,
        deterministic: bool = False,
        max_retries: int = 0,
        deduplicate: bool = False,
    ) -> StateConfig:
        """Register a state handler imperatively."""
        if name in self._states:
            raise ValueError(f"State {name!r} is already registered")
        cfg = StateConfig(
            name=name,
            fn=fn,
            deterministic=deterministic,
            max_retries=max_retries,
            deduplicate=deduplicate,
        )
        self._states[name] = cfg
        return cfg

    def state(
        self,
        name: str,
        *,
        deterministic: bool = False,
        max_retries: int = 0,
        deduplicate: bool = False,
    ) -> Callable[[F], F]:
        """Decorator registering a state handler on this StateMachine."""

        def decorator(fn: F) -> F:
            self.register_state(
                name,
                fn,
                deterministic=deterministic,
                max_retries=max_retries,
                deduplicate=deduplicate,
            )
            return fn

        return decorator

    def run(
        self,
        root: TaskSpec,
        *,
        run_dir: str | Path | None = None,
        max_concurrent: int = 4,
        budget: RunBudget | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> RunResult:
        """Execute a root TaskSpec to completion, suspension, failure, or budget limit."""
        resolved_budget = budget or RunBudget()
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        resolved_run_dir: str | None = None
        checkpoint: CheckpointStore | None = None
        if run_dir is not None:
            run_path = Path(run_dir)
            resolved_run_dir = str(run_path)
            checkpoint = CheckpointStore(run_path)
            store = ArtifactStore(run_path / "artifacts")
        else:
            store = ArtifactStore(None)

        session = _RunSession(
            machine=self,
            run_id=run_id,
            root_task_id=root.id or "task_1",
            tasks={},
            store=store,
            checkpoint=checkpoint,
            budget=resolved_budget,
            max_concurrent=max_concurrent,
            on_event=on_event,
            run_dir=resolved_run_dir,
        )

        if resolved_budget.max_tasks < 1 or resolved_budget.max_depth < 0:
            root_tasks, _ = session.enqueue_batch([root], parent_id=None, depth=0)
            session.root_task_id = root_tasks[0].id
            session.budget_exceeded = True
            session.save_checkpoint()
            session.emit(
                "run_finished",
                task_id=session.root_task_id,
                payload={"status": "budget_exceeded", "wall_clock_ms": 0.0},
            )
            return RunResult(
                run_id=run_id,
                status="budget_exceeded",
                root_task_id=session.root_task_id,
                output=None,
                tasks=session.tasks,
                run_dir=resolved_run_dir,
                wall_clock_ms=0.0,
            )

        root_tasks, _ = session.enqueue_batch([root], parent_id=None, depth=0)
        session.root_task_id = root_tasks[0].id
        session.save_checkpoint()
        return session.run_loop()

    def resume(
        self,
        run_dir: str | Path,
        *,
        resume_inputs: dict[str, Any] | None = None,
        max_concurrent: int = 4,
        budget: RunBudget | None = None,
        on_event: Callable[[Event], None] | None = None,
    ) -> RunResult:
        """Resume a checkpointed run from ``run_dir``, recovering crashes and suspensions."""
        run_path = Path(run_dir)
        checkpoint = CheckpointStore(run_path)
        snapshot = checkpoint.load_state()
        if snapshot is None:
            raise FileNotFoundError(f"No checkpoint state found in {run_path}")

        store = ArtifactStore(run_path / "artifacts")
        tasks = {
            tid: Task.model_validate(tdata)
            for tid, tdata in snapshot.get("tasks", {}).items()
        }
        resolved_budget = (
            budget
            if budget is not None
            else RunBudget.model_validate(snapshot.get("budget", {}))
        )
        prior_events = checkpoint.load_events()
        event_seq = max(int(snapshot.get("event_seq", 0)), len(prior_events))
        task_seq = int(snapshot.get("task_seq", len(tasks)))

        session = _RunSession(
            machine=self,
            run_id=str(snapshot["run_id"]),
            root_task_id=str(snapshot["root_task_id"]),
            tasks=tasks,
            store=store,
            checkpoint=checkpoint,
            budget=resolved_budget,
            max_concurrent=max_concurrent,
            on_event=on_event,
            task_seq=task_seq,
            event_seq=event_seq,
            run_dir=str(run_path),
        )

        # Recover mid-flight crashed tasks without touching already-done tasks.
        for task in session.tasks.values():
            if task.status == "running":
                task.status = "pending"

        # Apply external resume_inputs to suspended tasks.
        if resume_inputs:
            for task in session.tasks.values():
                if task.status != "suspended":
                    continue
                matched = False
                value: Any = None
                if task.id in resume_inputs:
                    matched = True
                    value = resume_inputs[task.id]
                elif task.resume_key is not None and task.resume_key in resume_inputs:
                    matched = True
                    value = resume_inputs[task.resume_key]

                if not matched:
                    continue

                if task.complete_on_resume:
                    output_val, ref = store.maybe_spill(
                        value,
                        resolved_budget.max_inline_bytes,
                    )
                    task.output = output_val
                    if ref is not None:
                        task.spilled_artifact_id = ref.ref_id
                    task.suspend_reason = None
                    task.status = "done"
                    session.emit(
                        "task_done",
                        task_id=task.id,
                        payload={"resumed": True},
                    )
                else:
                    task.resume_data = value
                    task.suspend_reason = None
                    task.status = "pending"

        session.propagate_pops()
        session.save_checkpoint()
        return session.run_loop()
