"""ARES narrow-waist library package.

100% backend- and domain-agnostic: exposes Task DAG execution (`solve(graph=...)`),
parallel worker dispatch (`max_concurrent`), streaming progress callbacks, content-hash
deduplication, taint-checked verifier execution, and immutable `NoteStore`.
"""

from ares.engine import (
    ARESRunner,
    EventType,
    ProcessorFn,
    ProcessorResult,
    RunReport,
    SolveEvent,
    TaskGraph,
    TaskNode,
    TaskSpec,
    TaskStatus,
    TaskType,
    UsageReport,
    canonical_content_hash,
    is_verifier_trusted,
    solve,
)
from ares.store import NoteStore

__all__ = [
    "ARESRunner",
    "EventType",
    "NoteStore",
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
