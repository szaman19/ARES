"""Pluggable execution backends (`ExecutionBackend` implementations) for ARES."""

from ares.adapters.duckdb_backend import DuckDBBackend, QueryExecutionResult
from ares.adapters.sql_guard import (
    SQLGuardConfig,
    SQLValidationError,
    estimate_query_cost,
    normalize_sql,
    sql_content_hash,
    validate_readonly_sql,
)

__all__ = [
    "DuckDBBackend",
    "QueryExecutionResult",
    "SQLGuardConfig",
    "SQLValidationError",
    "estimate_query_cost",
    "normalize_sql",
    "sql_content_hash",
    "validate_readonly_sql",
]
