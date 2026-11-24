"""Backend- and domain-agnostic ARES Task DAG engine and parallel `solve(graph=...)` runner.

Respects the ARES Narrow-Waist Boundary Rule:
- Provides TaskSpec, TaskGraph, solve(graph=...), max_concurrent parallel dispatch,
  streaming/progress callbacks, populated UsageReport, pluggable ExecutionBackend
  guard & cheapest-first cost sorting, content-hash deduplication, and standalone
  or backend-backed TRUSTED verifier execution.
- Completely unaware of board state, patch merging, domain schemas, or
  any specific query language / storage engine.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ares.contracts import (
    CertificationStatus,
    CostEstimate,
    ExecutionBackend,
    GuardValidationError,
    HypothesisSpec,
    Note,
    VerifierRegistry,
    VerifierResult,
)
from ares.store import NoteStore


TaskType = Literal["propose", "query", "verify", "custom"]
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


def canonical_content_hash(
    query: str = "",
    *,
    hypothesis_family: str = "",
    verifier_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Compute a modality-agnostic SHA-256 content hash for deduplication (Invariant I6)."""
    payload = json.dumps(
        {
            "query": query.strip(),
            "hypothesis_family": hypothesis_family.strip(),
            "verifier_name": verifier_name or "",
            "params": params or {},
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_verifier_trusted(registry: VerifierRegistry, verifier_name: str) -> bool:
    """Return True if a registered verifier is marked `trusted=True`."""
    meta_fn = getattr(registry, "metadata", None)
    if callable(meta_fn):
        try:
            meta = meta_fn(verifier_name)
            if isinstance(meta, dict) and "trusted" in meta:
                return bool(meta["trusted"])
        except Exception:
            return False
    return True


class TaskSpec(BaseModel):
    """Generic unit of work submitted through the ARES narrow waist."""

    task_id: str
    task_type: TaskType = "custom"
    depends_on: list[str] = Field(default_factory=list)
    requested_under: int = Field(default=0, ge=0)
    priority: int = 0
    # Proposal context only (Invariant I4: never passed to TRUSTED verifiers)
    context: dict[str, Any] = Field(default_factory=dict)
    hypothesis_spec: HypothesisSpec | None = None
    verifier_name: str | None = None
    verifier_params: dict[str, Any] = Field(default_factory=dict)
    processor_name: str | None = None
    dedup_key: str | None = None


class TaskNode(BaseModel):
    """Runtime execution state for a TaskSpec inside a TaskGraph."""

    spec: TaskSpec
    status: TaskStatus = "pending"
    content_hash: str = ""
    cost_estimate: CostEstimate | None = None
    note_ids: list[str] = Field(default_factory=list)
    output: Any = None
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)


class ProcessorResult(BaseModel):
    """Structured return type for custom/propose stateless processors."""

    output: Any = None
    notes: list[Note] = Field(default_factory=list)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)


class UsageReport(BaseModel):
    """Populated execution & resource telemetry returned by `solve()`."""

    total_tasks: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)
    skipped_duplicate_tasks: int = Field(default=0, ge=0)
    rejected_guard_tasks: int = Field(default=0, ge=0)
    queries_executed: int = Field(default=0, ge=0)
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
    """Streaming progress event emitted during `solve()`."""

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
        state: dict[str, int] = {}

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
    """Structured report returned by ARES `solve()`."""

    model_config = {"arbitrary_types_allowed": True}

    graph: TaskGraph
    new_notes: list[Note] = Field(default_factory=list)
    usage: UsageReport = Field(default_factory=UsageReport)
    errors: dict[str, str] = Field(default_factory=dict)


ProcessorFn = Callable[[TaskSpec, dict[str, Any]], Any]


class ARESRunner:
    """Stateless, backend-agnostic parallel TaskGraph executor for ARES."""

    def __init__(
        self,
        *,
        backend: ExecutionBackend | None = None,
        registry: VerifierRegistry | None = None,
        note_store: NoteStore | None = None,
        processors: dict[str, ProcessorFn] | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.note_store = note_store if note_store is not None else NoteStore()
        self.processors: dict[str, ProcessorFn] = dict(processors or {})

    def _compute_dedup_key(self, spec: TaskSpec) -> str:
        """Compute deterministic content-hash deduplication key for a TaskSpec."""
        if spec.dedup_key:
            return spec.dedup_key

        hasher = (
            self.backend.compute_content_hash
            if self.backend is not None
            else canonical_content_hash
        )

        if spec.task_type == "query" and spec.hypothesis_spec is not None:
            effective_vname = (
                spec.hypothesis_spec.verifier_name if self.registry is not None else None
            )
            return hasher(
                spec.hypothesis_spec.query,
                hypothesis_family=spec.hypothesis_spec.hypothesis_family,
                verifier_name=effective_vname,
                params=spec.hypothesis_spec.params,
            )

        if spec.task_type == "verify":
            vname = spec.verifier_name or (
                spec.hypothesis_spec.verifier_name if spec.hypothesis_spec else ""
            )
            q_str = spec.hypothesis_spec.query if spec.hypothesis_spec else ""
            hfam = spec.hypothesis_spec.hypothesis_family if spec.hypothesis_spec else ""
            merged_params = {
                **(spec.hypothesis_spec.params if spec.hypothesis_spec else {}),
                **spec.verifier_params,
            }
            return hasher(
                q_str,
                hypothesis_family=hfam,
                verifier_name=vname or None,
                params=merged_params,
            )

        return ""

    def _execute_standalone_verifier(
        self,
        verifier_name: str,
        *,
        params: dict[str, Any] | None = None,
        requested_under: int = 0,
        hypothesis_family: str = "",
        query: str = "",
        content_hash: str = "",
    ) -> tuple[VerifierResult, Note]:
        """Execute a pure-Python VerifierRegistry function when `backend is None`.

        Enforces Invariant I4 (only explicit `params` are passed; never `spec.context`)
        and Invariant I2 taint discipline (untrusted or failing verifiers emit UNCERTIFIED).
        """
        assert self.registry is not None
        verifier_fn = self.registry.get(verifier_name)
        trusted = is_verifier_trusted(self.registry, verifier_name)
        effective_params = dict(params) if params else {}
        effective_hash = content_hash or canonical_content_hash(
            query,
            hypothesis_family=hypothesis_family,
            verifier_name=verifier_name,
            params=effective_params,
        )

        result: VerifierResult = verifier_fn(effective_params)

        certification: CertificationStatus = (
            result.certification if (trusted and result.passed) else "UNCERTIFIED"
        )
        if certification != result.certification:
            result = result.model_copy(update={"certification": certification})

        note = Note(
            note_id=self.note_store.next_note_id(),
            certification=certification,
            sources=[f"verifier:{verifier_name}"],
            artifacts=list(result.artifacts),
            verifier_name=verifier_name,
            hypothesis_family=hypothesis_family,
            query=query,
            content_hash=effective_hash,
            requested_under=requested_under,
            summary=result.summary,
            evidence=dict(result.evidence),
            row_count=result.row_count,
        )
        self.note_store.append(note)
        return result, note

    def solve(
        self,
        tasks: list[TaskSpec] | None = None,
        *,
        graph: TaskGraph | None = None,
        max_concurrent: int = 4,
        on_event: Callable[[SolveEvent], None] | None = None,
        cost_sort: bool = True,
    ) -> RunReport:
        """Execute pending tasks in `graph` (optionally extending with `tasks`)."""
        t_start = time.perf_counter()
        active_graph = graph if graph is not None else TaskGraph()
        if tasks:
            active_graph.extend(tasks)
        active_graph.validate_dependencies()

        event_lock = threading.Lock()
        seen_keys_lock = threading.Lock()
        seen_dedup_keys: set[str] = set()

        for existing in active_graph.nodes.values():
            if existing.status in ("completed", "skipped_duplicate") and existing.content_hash:
                seen_dedup_keys.add(existing.content_hash)

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

            dispatchable: list[tuple[int, int, int, int, TaskNode]] = []

            for idx, node in enumerate(ready):
                spec = node.spec
                usage.total_tasks += 1

                # 1. Pluggable root-guard validation for automated magnet queries (Invariant I6)
                if spec.task_type == "query":
                    if spec.hypothesis_spec is None:
                        node.status = "rejected_guard"
                        node.error = "query task missing hypothesis_spec."
                        errors[spec.task_id] = node.error
                        usage.rejected_guard_tasks += 1
                        _emit("task_rejected", spec.task_id, reason=node.error)
                        continue
                    if self.backend is None:
                        node.status = "rejected_guard"
                        node.error = "query task requires an ExecutionBackend."
                        errors[spec.task_id] = node.error
                        usage.rejected_guard_tasks += 1
                        _emit("task_rejected", spec.task_id, reason=node.error)
                        continue
                    try:
                        self.backend.validate_query(spec.hypothesis_spec.query)
                    except GuardValidationError as exc:
                        node.status = "rejected_guard"
                        node.error = str(exc)
                        errors[spec.task_id] = node.error
                        usage.rejected_guard_tasks += 1
                        _emit("task_rejected", spec.task_id, reason=node.error)
                        continue

                # 2. Content-hash deduplication (Invariant I6)
                dkey = self._compute_dedup_key(spec)
                if dkey:
                    node.content_hash = dkey
                    with seen_keys_lock:
                        already_seen = (
                            dkey in seen_dedup_keys or self.note_store.has_content_hash(dkey)
                        )
                        if already_seen:
                            node.status = "skipped_duplicate"
                            usage.skipped_duplicate_tasks += 1
                            existing_note = self.note_store.get_by_content_hash(dkey)
                            if existing_note is not None:
                                node.note_ids.append(existing_note.note_id)
                            _emit(
                                "task_skipped",
                                spec.task_id,
                                reason="duplicate_content_hash",
                                content_hash=dkey,
                            )
                            continue
                        seen_dedup_keys.add(dkey)

                # 3. Pluggable pre-execution cost estimation for cheapest-first ordering
                est_bytes = 0
                est_units = 0
                if (
                    cost_sort
                    and spec.task_type == "query"
                    and spec.hypothesis_spec is not None
                    and self.backend is not None
                ):
                    try:
                        cost = self.backend.estimate_cost(spec.hypothesis_spec.query)
                        node.cost_estimate = cost
                        if not cost.within_budget:
                            node.status = "rejected_guard"
                            node.error = cost.rejection_reason or "Query exceeded cost guard cap."
                            errors[spec.task_id] = node.error
                            usage.rejected_guard_tasks += 1
                            _emit("task_rejected", spec.task_id, reason=node.error)
                            continue
                        est_bytes = cost.estimated_bytes
                        est_units = cost.estimated_units
                    except Exception as exc:
                        node.status = "rejected_guard"
                        node.error = f"Cost estimation failed: {exc}"
                        errors[spec.task_id] = node.error
                        usage.rejected_guard_tasks += 1
                        _emit("task_rejected", spec.task_id, reason=node.error)
                        continue

                dispatchable.append((spec.priority, est_bytes, est_units, idx, node))

            if not dispatchable:
                continue

            dispatchable.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            ordered_nodes = [item[4] for item in dispatchable]

            def _execute_node(node: TaskNode) -> tuple[TaskNode, dict[str, int]]:
                spec = node.spec
                node.status = "running"
                _emit("task_started", spec.task_id, task_type=spec.task_type)
                t0 = time.perf_counter()
                delta = {
                    "queries": 0,
                    "verifiers": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "model_calls": 0,
                }
                try:
                    if spec.task_type == "query":
                        if self.backend is None:
                            raise RuntimeError("query task requires an ExecutionBackend.")
                        hspec = spec.hypothesis_spec
                        assert hspec is not None
                        merged_params = {**hspec.params, **spec.verifier_params}
                        if hspec.verifier_name and self.registry is not None:
                            vres, note = self.backend.execute_verifier(
                                self.registry,
                                hspec.verifier_name,
                                params=merged_params,
                                requested_under=spec.requested_under,
                                hypothesis_family=hspec.hypothesis_family,
                                query=hspec.query,
                                content_hash=node.content_hash,
                                note_store=self.note_store,
                            )
                            delta["verifiers"] += 1
                            node.output = vres
                        else:
                            qres, note = self.backend.execute_query(
                                hspec.query,
                                params=merged_params,
                                requested_under=spec.requested_under,
                                hypothesis_family=hspec.hypothesis_family,
                                content_hash=node.content_hash,
                                note_store=self.note_store,
                            )
                            delta["queries"] += 1
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
                        if self.registry is None:
                            raise RuntimeError("verify task requires a VerifierRegistry.")
                        vname = spec.verifier_name or (
                            spec.hypothesis_spec.verifier_name if spec.hypothesis_spec else None
                        )
                        if not vname:
                            raise ValueError("verify task missing verifier_name.")

                        hfam = (
                            spec.hypothesis_spec.hypothesis_family if spec.hypothesis_spec else ""
                        )
                        q_str = spec.hypothesis_spec.query if spec.hypothesis_spec else ""
                        merged_params = {
                            **(spec.hypothesis_spec.params if spec.hypothesis_spec else {}),
                            **spec.verifier_params,
                        }
                        # Invariant I4: never pass spec.context to TRUSTED verifiers
                        if self.backend is not None:
                            vres, note = self.backend.execute_verifier(
                                self.registry,
                                vname,
                                params=merged_params,
                                requested_under=spec.requested_under,
                                hypothesis_family=hfam,
                                query=q_str,
                                content_hash=node.content_hash,
                                note_store=self.note_store,
                            )
                        else:
                            vres, note = self._execute_standalone_verifier(
                                vname,
                                params=merged_params,
                                requested_under=spec.requested_under,
                                hypothesis_family=hfam,
                                query=q_str,
                                content_hash=node.content_hash,
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
                    usage.queries_executed += delta["queries"]
                    usage.verifiers_executed += delta["verifiers"]
                    usage.total_prompt_tokens += delta["prompt_tokens"]
                    usage.total_completion_tokens += delta["completion_tokens"]
                    usage.model_calls += delta["model_calls"]

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
    backend: ExecutionBackend | None = None,
    registry: VerifierRegistry | None = None,
    note_store: NoteStore | None = None,
    processors: dict[str, ProcessorFn] | None = None,
    max_concurrent: int = 4,
    on_event: Callable[[SolveEvent], None] | None = None,
    cost_sort: bool = True,
) -> RunReport:
    """Narrow-waist functional entry point for ARES parallel TaskGraph execution."""
    runner = ARESRunner(
        backend=backend,
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
    "canonical_content_hash",
    "is_verifier_trusted",
    "solve",
]
