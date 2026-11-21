"""Domain-agnostic ARES Task DAG engine and parallel solve(graph=...) runner.

Respects the ARES Narrow-Waist Boundary Rule (ARES Plan §3.2):
- Provides TaskSpec, TaskGraph, solve(graph=...), max_concurrent parallel dispatch,
  streaming/progress callbacks, populated UsageReport, root-guard validation,
  content-hash deduplication, and cheapest-first DuckDB execution.
- Completely unaware of investigation boards, fusion patches, or domain-specific schemas.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ares.duckdb_runner import DuckDBTrustedRunner
from ares.sql_guard import (
    QueryCostEstimate,
    SQLValidationError,
    query_content_hash,
    validate_readonly_sql,
)
from ares.store import NoteStore
from contracts import HypothesisQuerySpec, Note, VerifierRegistry


TaskType = Literal["propose", "sql_query", "verify", "custom"]
TaskStatus = Literal[
    "pending",
    "running",
    "completed",
    "skipped_duplicate",
    "rejected_guard",
    "failed",
]
EventType = Literal[
    "task_started",
    "task_completed",
    "task_skipped",
    "task_rejected",
    "task_failed",
    "note_emitted",
]


class TaskSpec(BaseModel):
    """Generic unit of work submitted through the ARES narrow waist."""

    task_id: str
    task_type: TaskType = "custom"
    depends_on: list[str] = Field(default_factory=list)
    requested_under: int = Field(default=0, ge=0)
    priority: int = 0
    # Proposal context only (Invariant I4: never passed to TRUSTED verifiers)
    context: dict[str, Any] = Field(default_factory=dict)
    hypothesis_spec: HypothesisQuerySpec | None = None
    verifier_name: str | None = None
    verifier_params: dict[str, Any] = Field(default_factory=dict)
    processor_name: str | None = None
    dedup_key: str | None = None


class TaskNode(BaseModel):
    """Runtime execution state for a TaskSpec inside a TaskGraph."""

    spec: TaskSpec
    status: TaskStatus = "pending"
    query_hash: str = ""
    cost_estimate: QueryCostEstimate | None = None
    note_ids: list[str] = Field(default_factory=list)
    output: Any = None
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)


class ProcessorResult(BaseModel):
    """Optional structured return type for custom/propose stateless processors."""

    output: Any = None
    notes: list[Note] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)


class UsageReport(BaseModel):
    """Populated execution & resource telemetry returned by solve()."""

    total_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)
    skipped_duplicate_tasks: int = Field(default=0, ge=0)
    rejected_guard_tasks: int = Field(default=0, ge=0)
    sql_queries_executed: int = Field(default=0, ge=0)
    verifiers_executed: int = Field(default=0, ge=0)
    notes_emitted: int = Field(default=0, ge=0)
    certified_notes: int = Field(default=0, ge=0)
    uncertified_notes: int = Field(default=0, ge=0)
    artifacts_produced: int = Field(default=0, ge=0)
    total_prompt_tokens: int = Field(default=0, ge=0)
    total_completion_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    wall_clock_ms: float = Field(default=0.0, ge=0.0)


class SolveEvent(BaseModel):
    """Streaming progress event emitted during solve()."""

    event_type: EventType
    task_id: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskGraph:
    """Extensible directed acyclic graph (DAG) of TaskNodes."""

    def __init__(self, tasks: list[TaskSpec] | None = None) -> None:
        self.nodes: dict[str, TaskNode] = {}
        self._order: list[str] = []
        if tasks:
            self.extend(tasks)

    def add_task(self, spec: TaskSpec) -> TaskNode:
        """Add a TaskSpec to the graph, enforcing unique IDs and acyclicity."""
        if spec.task_id in self.nodes:
            raise ValueError(f"Duplicate task_id in TaskGraph: {spec.task_id}")
        if spec.task_id in spec.depends_on:
            raise ValueError(f"Task cannot depend on itself: {spec.task_id}")
        node = TaskNode(spec=spec)
        self.nodes[spec.task_id] = node
        self._order.append(spec.task_id)
        if self._has_cycle():
            del self.nodes[spec.task_id]
            self._order.pop()
            raise ValueError(f"Adding task '{spec.task_id}' introduces a cycle in TaskGraph.")
        return node

    def extend(self, specs: list[TaskSpec]) -> list[TaskNode]:
        """Extend an existing TaskGraph with additional tasks."""
        return [self.add_task(s) for s in specs]

    def _has_cycle(self) -> bool:
        state: dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited

        def _dfs(tid: str) -> bool:
            state[tid] = 1
            node = self.nodes.get(tid)
            if node is not None:
                for dep in node.spec.depends_on:
                    if dep not in self.nodes:
                        continue
                    dep_state = state.get(dep, 0)
                    if dep_state == 1:
                        return True
                    if dep_state == 0 and _dfs(dep):
                        return True
            state[tid] = 2
            return False

        for tid in self.nodes:
            if state.get(tid, 0) == 0 and _dfs(tid):
                return True
        return False

    def validate_dependencies(self) -> None:
        """Raise ValueError if any task depends on an unknown task_id."""
        for tid, node in self.nodes.items():
            for dep in node.spec.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"Task '{tid}' depends on unknown task '{dep}'.")

    def ready_tasks(self) -> list[TaskNode]:
        """Return pending tasks whose dependencies have all reached a terminal state."""
        terminal = {"completed", "skipped_duplicate", "rejected_guard", "failed"}
        ready: list[TaskNode] = []
        for tid in self._order:
            node = self.nodes[tid]
            if node.status != "pending":
                continue
            if all(
                dep in self.nodes and self.nodes[dep].status in terminal
                for dep in node.spec.depends_on
            ):
                ready.append(node)
        return ready

    def pending_count(self) -> int:
        return sum(1 for n in self.nodes.values() if n.status == "pending")

    def is_complete(self) -> bool:
        return self.pending_count() == 0 and all(
            n.status != "running" for n in self.nodes.values()
        )


class RunReport(BaseModel):
    """Structured report returned by ARES solve()."""

    model_config = {"arbitrary_types_allowed": True}

    graph: TaskGraph
    new_notes: list[Note] = Field(default_factory=list)
    usage: UsageReport = Field(default_factory=UsageReport)
    errors: dict[str, str] = Field(default_factory=dict)


ProcessorFn = Callable[[TaskSpec, dict[str, Any]], Any]


class ARESRunner:
    """Stateless parallel TaskGraph executor for ARES."""

    def __init__(
        self,
        *,
        duckdb_runner: DuckDBTrustedRunner | None = None,
        registry: VerifierRegistry | None = None,
        note_store: NoteStore | None = None,
        processors: dict[str, ProcessorFn] | None = None,
    ) -> None:
        self.duckdb_runner = duckdb_runner
        self.registry = registry
        self.note_store = note_store if note_store is not None else NoteStore()
        self.processors: dict[str, ProcessorFn] = dict(processors or {})

    def _compute_dedup_key(self, spec: TaskSpec) -> str:
        """Compute deterministic content-hash deduplication key for a TaskSpec."""
        if spec.dedup_key:
            return spec.dedup_key

        if spec.task_type == "sql_query" and spec.hypothesis_spec is not None:
            effective_vname = (
                spec.hypothesis_spec.verifier_name if self.registry is not None else None
            )
            return query_content_hash(
                spec.hypothesis_spec.query_sql,
                hypothesis_family=spec.hypothesis_spec.hypothesis_family,
                verifier_name=effective_vname,
            )

        if spec.task_type == "verify":
            vname = spec.verifier_name or (
                spec.hypothesis_spec.verifier_name if spec.hypothesis_spec else ""
            )
            qsql = spec.hypothesis_spec.query_sql if spec.hypothesis_spec else ""
            hfam = spec.hypothesis_spec.hypothesis_family if spec.hypothesis_spec else ""
            params_repr = json.dumps(spec.verifier_params, sort_keys=True, default=str)
            base = qsql if qsql.strip() else f"VERIFIER:{vname}:{params_repr}"
            return query_content_hash(
                base,
                hypothesis_family=hfam,
                verifier_name=vname or None,
            )

        return ""

    def solve(
        self,
        tasks: list[TaskSpec] | None = None,
        *,
        graph: TaskGraph | None = None,
        max_concurrent: int = 4,
        on_event: Callable[[SolveEvent], None] | None = None,
        cost_sort: bool = True,
    ) -> RunReport:
        """Execute pending tasks in `graph` (optionally extending with `tasks`).

        Supports:
        - Extending an existing TaskGraph across rounds via `solve(graph=g, tasks=...)`
        - Bounded parallel dispatch (`max_concurrent`)
        - Read-only SQL root guards & EXPLAIN cost estimation (cheapest-first)
        - Content-hash deduplication against NoteStore and prior tasks in the graph
        - Streaming progress callbacks (`on_event`)
        """
        t_start = time.perf_counter()
        active_graph = graph if graph is not None else TaskGraph()
        if tasks:
            active_graph.extend(tasks)
        active_graph.validate_dependencies()

        event_lock = threading.Lock()
        seen_keys_lock = threading.Lock()
        seen_dedup_keys: set[str] = set()

        # Pre-seed seen keys from already completed/skipped nodes in an extended graph
        for existing in active_graph.nodes.values():
            if existing.status in ("completed", "skipped_duplicate") and existing.query_hash:
                seen_dedup_keys.add(existing.query_hash)

        new_notes: list[Note] = []
        notes_lock = threading.Lock()
        errors: dict[str, str] = {}
        usage = UsageReport()

        def _emit(event_type: EventType, task_id: str, **payload: Any) -> None:
            if on_event is None:
                return
            ev = SolveEvent(event_type=event_type, task_id=task_id, payload=payload)
            with event_lock:
                on_event(ev)

        workers = max(1, int(max_concurrent))

        while True:
            ready = active_graph.ready_tasks()
            if not ready:
                break

            # Pre-flight phase for ready batch: validate guards, dedup, and EXPLAIN cost-sort
            dispatchable: list[tuple[int, int, int, int, TaskNode]] = []

            for idx, node in enumerate(ready):
                spec = node.spec
                usage.total_tasks += 1

                # 1. Read-only SQL guard & EXPLAIN cost estimation for SQL tasks
                if spec.task_type == "sql_query":
                    if spec.hypothesis_spec is None:
                        node.status = "rejected_guard"
                        node.error = "sql_query task missing hypothesis_spec."
                        errors[spec.task_id] = node.error
                        usage.rejected_guard_tasks += 1
                        _emit("task_rejected", spec.task_id, reason=node.error)
                        continue
                    try:
                        cfg = self.duckdb_runner.guard_config if self.duckdb_runner else None
                        validate_readonly_sql(spec.hypothesis_spec.query_sql, cfg)
                    except SQLValidationError as exc:
                        node.status = "rejected_guard"
                        node.error = str(exc)
                        errors[spec.task_id] = node.error
                        usage.rejected_guard_tasks += 1
                        _emit("task_rejected", spec.task_id, reason=node.error)
                        continue

                # 2. Content-hash deduplication (Invariant I6)
                dkey = self._compute_dedup_key(spec)
                if dkey:
                    node.query_hash = dkey
                    with seen_keys_lock:
                        already_seen = dkey in seen_dedup_keys or self.note_store.has_query_hash(dkey)
                        if already_seen:
                            node.status = "skipped_duplicate"
                            usage.skipped_duplicate_tasks += 1
                            existing_note = self.note_store.get_by_query_hash(dkey)
                            if existing_note is not None:
                                node.note_ids.append(existing_note.note_id)
                            _emit(
                                "task_skipped",
                                spec.task_id,
                                reason="duplicate_content_hash",
                                query_hash=dkey,
                            )
                            continue
                        seen_dedup_keys.add(dkey)

                # 3. EXPLAIN cost estimation for cheapest-first ordering
                est_bytes = 0
                est_rows = 0
                if (
                    cost_sort
                    and spec.task_type == "sql_query"
                    and spec.hypothesis_spec is not None
                    and self.duckdb_runner is not None
                ):
                    try:
                        cost = self.duckdb_runner.estimate_cost(spec.hypothesis_spec.query_sql)
                        node.cost_estimate = cost
                        if not cost.within_budget:
                            node.status = "rejected_guard"
                            node.error = cost.rejection_reason or "Query exceeded EXPLAIN cost cap."
                            errors[spec.task_id] = node.error
                            usage.rejected_guard_tasks += 1
                            _emit("task_rejected", spec.task_id, reason=node.error)
                            continue
                        est_bytes = cost.estimated_bytes
                        est_rows = cost.estimated_rows
                    except Exception as exc:
                        node.status = "rejected_guard"
                        node.error = f"EXPLAIN failed: {exc}"
                        errors[spec.task_id] = node.error
                        usage.rejected_guard_tasks += 1
                        _emit("task_rejected", spec.task_id, reason=node.error)
                        continue

                dispatchable.append((spec.priority, est_bytes, est_rows, idx, node))

            if not dispatchable:
                continue

            # Cheapest-first within equal priority
            dispatchable.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            ordered_nodes = [item[4] for item in dispatchable]

            def _execute_node(node: TaskNode) -> tuple[TaskNode, dict[str, int]]:
                spec = node.spec
                node.status = "running"
                _emit("task_started", spec.task_id, task_type=spec.task_type)
                t0 = time.perf_counter()
                delta = {
                    "sql_queries": 0,
                    "verifiers": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "model_calls": 0,
                }
                try:
                    if spec.task_type == "sql_query":
                        if self.duckdb_runner is None:
                            raise RuntimeError("sql_query task requires duckdb_runner.")
                        hspec = spec.hypothesis_spec
                        assert hspec is not None
                        if hspec.verifier_name and self.registry is not None:
                            vres, note = self.duckdb_runner.execute_verifier(
                                self.registry,
                                hspec.verifier_name,
                                params=spec.verifier_params,
                                requested_under=spec.requested_under,
                                hypothesis_family=hspec.hypothesis_family,
                                query_sql=hspec.query_sql,
                                query_hash=node.query_hash,
                                note_store=self.note_store,
                            )
                            delta["verifiers"] += 1
                            node.output = vres
                        else:
                            qres, note = self.duckdb_runner.execute_readonly_query(
                                hspec.query_sql,
                                requested_under=spec.requested_under,
                                hypothesis_family=hspec.hypothesis_family,
                                query_hash=node.query_hash,
                                note_store=self.note_store,
                            )
                            delta["sql_queries"] += 1
                            node.output = qres
                        node.note_ids.append(note.note_id)
                        with notes_lock:
                            new_notes.append(note)
                        _emit(
                            "note_emitted",
                            spec.task_id,
                            note_id=note.note_id,
                            certification=note.certification,
                            artifacts=len(note.artifacts),
                        )

                    elif spec.task_type == "verify":
                        if self.duckdb_runner is None or self.registry is None:
                            raise RuntimeError(
                                "verify task requires both duckdb_runner and registry."
                            )
                        vname = spec.verifier_name or (
                            spec.hypothesis_spec.verifier_name if spec.hypothesis_spec else None
                        )
                        if not vname:
                            raise ValueError("verify task missing verifier_name.")
                        # Invariant I4: never pass spec.context to TRUSTED verifiers
                        vres, note = self.duckdb_runner.execute_verifier(
                            self.registry,
                            vname,
                            params=spec.verifier_params,
                            requested_under=spec.requested_under,
                            hypothesis_family=(
                                spec.hypothesis_spec.hypothesis_family
                                if spec.hypothesis_spec
                                else ""
                            ),
                            query_sql=(
                                spec.hypothesis_spec.query_sql if spec.hypothesis_spec else ""
                            ),
                            query_hash=node.query_hash,
                            note_store=self.note_store,
                        )
                        delta["verifiers"] += 1
                        node.output = vres
                        node.note_ids.append(note.note_id)
                        with notes_lock:
                            new_notes.append(note)
                        _emit(
                            "note_emitted",
                            spec.task_id,
                            note_id=note.note_id,
                            certification=note.certification,
                            artifacts=len(note.artifacts),
                        )

                    else:
                        # "propose" or "custom" stateless processor
                        pname = spec.processor_name or spec.task_type
                        proc = self.processors.get(pname)
                        if proc is None:
                            raise KeyError(f"No processor registered for '{pname}'.")
                        dep_outputs = {
                            dep: active_graph.nodes[dep].output
                            for dep in spec.depends_on
                            if dep in active_graph.nodes
                        }
                        raw_out = proc(spec, dep_outputs)
                        if isinstance(raw_out, ProcessorResult):
                            node.output = raw_out.output
                            delta["prompt_tokens"] += raw_out.prompt_tokens
                            delta["completion_tokens"] += raw_out.completion_tokens
                            delta["model_calls"] += raw_out.model_calls
                            for emitted in raw_out.notes:
                                self.note_store.append(emitted)
                                node.note_ids.append(emitted.note_id)
                                with notes_lock:
                                    new_notes.append(emitted)
                                _emit(
                                    "note_emitted",
                                    spec.task_id,
                                    note_id=emitted.note_id,
                                    certification=emitted.certification,
                                    artifacts=len(emitted.artifacts),
                                )
                        else:
                            node.output = raw_out

                    node.status = "completed"
                    node.duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    _emit("task_completed", spec.task_id, duration_ms=node.duration_ms)
                except Exception as exc:
                    node.status = "failed"
                    node.error = str(exc)
                    node.duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                    _emit("task_failed", spec.task_id, error=node.error)
                return node, delta

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_execute_node, node) for node in ordered_nodes]
                for fut in concurrent.futures.as_completed(futures):
                    finished_node, delta = fut.result()
                    if finished_node.status == "completed":
                        usage.completed_tasks += 1
                    elif finished_node.status == "failed":
                        usage.failed_tasks += 1
                        if finished_node.error:
                            errors[finished_node.spec.task_id] = finished_node.error
                    usage.sql_queries_executed += delta["sql_queries"]
                    usage.verifiers_executed += delta["verifiers"]
                    usage.total_prompt_tokens += delta["prompt_tokens"]
                    usage.total_completion_tokens += delta["completion_tokens"]
                    usage.model_calls += delta["model_calls"]

        # Populate note & artifact telemetry
        usage.notes_emitted = len(new_notes)
        usage.certified_notes = sum(1 for n in new_notes if n.certification == "CERTIFIED")
        usage.uncertified_notes = sum(1 for n in new_notes if n.certification == "UNCERTIFIED")
        usage.artifacts_produced = sum(len(n.artifacts) for n in new_notes)
        usage.wall_clock_ms = round((time.perf_counter() - t_start) * 1000.0, 3)

        return RunReport(
            graph=active_graph,
            new_notes=new_notes,
            usage=usage,
            errors=errors,
        )


def solve(
    tasks: list[TaskSpec] | None = None,
    *,
    graph: TaskGraph | None = None,
    duckdb_runner: DuckDBTrustedRunner | None = None,
    registry: VerifierRegistry | None = None,
    note_store: NoteStore | None = None,
    processors: dict[str, ProcessorFn] | None = None,
    max_concurrent: int = 4,
    on_event: Callable[[SolveEvent], None] | None = None,
    cost_sort: bool = True,
) -> RunReport:
    """Narrow-waist functional entry point for ARES parallel TaskGraph execution."""
    runner = ARESRunner(
        duckdb_runner=duckdb_runner,
        registry=registry,
        note_store=note_store,
        processors=processors,
    )
    return runner.solve(
        tasks=tasks,
        graph=graph,
        max_concurrent=max_concurrent,
        on_event=on_event,
        cost_sort=cost_sort,
    )


__all__ = [
    "ARESRunner",
    "EventType",
    "ProcessorFn",
    "ProcessorResult",
    "RunReport",
    "SolveEvent",
    "TaskGraph",
    "TaskNode",
    "TaskSpec",
    "TaskStatus",
    "TaskType",
    "UsageReport",
    "solve",
]
