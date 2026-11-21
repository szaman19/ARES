"""ARES narrow-waist library package.

Exposes domain-agnostic Task DAG execution (`solve(graph=...)`), parallel
worker dispatch (`max_concurrent`), streaming progress callbacks, immutable
`NoteStore`, and guarded DuckDB `TRUSTED` query & verifier execution.
"""

from ares.duckdb_runner import DuckDBTrustedRunner, QueryExecutionResult
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
    solve,
)
from ares.sql_guard import (
    QueryCostEstimate,
    SQLGuardConfig,
    SQLValidationError,
    estimate_query_cost,
    normalize_sql,
    query_content_hash,
    validate_readonly_sql,
)
from ares.store import NoteStore

__all__ = [
    "ARESRunner",
    "DuckDBTrustedRunner",
    "EventType",
    "NoteStore",
    "ProcessorFn",
    "ProcessorResult",
    "QueryCostEstimate",
    "QueryExecutionResult",
    "RunReport",
    "SQLGuardConfig",
    "SQLValidationError",
    "SolveEvent",
    "TaskGraph",
    "TaskNode",
    "TaskSpec",
    "TaskStatus",
    "TaskType",
    "UsageReport",
    "estimate_query_cost",
    "normalize_sql",
    "query_content_hash",
    "solve",
    "validate_readonly_sql",
]
