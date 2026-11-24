"""Read-only SQL validation, normalization, content-hash dedup, and DuckDB EXPLAIN cost estimation.

Lives inside the pluggable `ares.adapters` layer (`ares/adapters/sql_guard.py`), implementing
ARES Invariant I6 (Root Guards & Deduplication) for tabular DuckDB + Parquet use cases
without coupling the core `ares` engine or context layer to SQL or DuckDB.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, Field

from ares.contracts import CostEstimate, GuardValidationError


class SQLValidationError(GuardValidationError):
    """Raised when a SQL query violates read-only root guards."""


class SQLGuardConfig(BaseModel):
    """Root-guard limits for automated DuckDB SQL queries (Invariant I6)."""

    max_estimated_rows: int = Field(default=50_000_000, gt=0)
    max_estimated_bytes: int = Field(default=15_000_000_000, gt=0)
    max_result_rows: int = Field(default=500, gt=0)
    avg_bytes_per_row_fallback: int = Field(default=256, gt=0)
    allowed_data_dir: str | None = None
    allowed_tables: tuple[str, ...] | None = None


_FORBIDDEN_KEYWORDS: frozenset[str] = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "REPLACE",
        "TRUNCATE",
        "COPY",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "SET",
        "RESET",
        "LOAD",
        "INSTALL",
        "EXPORT",
        "IMPORT",
        "CALL",
        "EXECUTE",
        "PREPARE",
        "DEALLOCATE",
        "VACUUM",
        "CHECKPOINT",
        "GRANT",
        "REVOKE",
        "MERGE",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "TRANSACTION",
        "FORCE",
        "COMMENT",
        "USE",
        "PIVOT_WIDER",
    }
)

_FORBIDDEN_FUNCTIONS: frozenset[str] = frozenset(
    {
        "getenv",
        "duckdb_secrets",
        "which_secret",
        "write_parquet",
        "read_blob",
        "read_text",
        "read_csv",
        "read_csv_auto",
        "sniff_csv",
        "read_json",
        "read_json_auto",
        "read_json_objects",
        "read_ndjson",
        "read_ndjson_auto",
        "read_ndjson_objects",
        "parquet_metadata",
        "parquet_schema",
        "parquet_file_metadata",
        "parquet_kv_metadata",
        "glob",
        "http_get",
        "http_post",
    }
)

_FORBIDDEN_URI_PREFIXES: tuple[str, ...] = (
    "http://",
    "https://",
    "s3://",
    "gs://",
    "gcs://",
    "az://",
    "abfss://",
    "hf://",
    "ftp://",
)

_SQL_KEYWORDS_TO_NORMALIZE: frozenset[str] = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "AND",
        "OR",
        "NOT",
        "IN",
        "IS",
        "NULL",
        "AS",
        "ON",
        "JOIN",
        "INNER",
        "LEFT",
        "RIGHT",
        "FULL",
        "OUTER",
        "CROSS",
        "GROUP",
        "BY",
        "HAVING",
        "ORDER",
        "ASC",
        "DESC",
        "LIMIT",
        "OFFSET",
        "DISTINCT",
        "WITH",
        "UNION",
        "ALL",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "BETWEEN",
        "LIKE",
        "ILIKE",
        "CAST",
        "COUNT",
        "SUM",
        "AVG",
        "MIN",
        "MAX",
        "NULLIF",
        "COALESCE",
        "OVER",
        "PARTITION",
    }
)


def _strip_sql_comments(sql: str) -> str:
    """Remove -- line comments and /* ... */ block comments while preserving string literals."""
    out: list[str] = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_single:
            out.append(ch)
            if ch == "'" and nxt == "'":
                out.append(nxt)
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue

        if in_double:
            out.append(ch)
            if ch == '"' and nxt == '"':
                out.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            out.append(ch)
            i += 1
            continue

        if ch == "-" and nxt == "-":
            i += 2
            while i < n and sql[i] != "\n":
                i += 1
            out.append(" ")
            continue

        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            out.append(" ")
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _split_literals_and_code(sql: str) -> list[tuple[bool, str]]:
    """Split SQL into alternating segments of (is_literal, text)."""
    segments: list[tuple[bool, str]] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    in_quote: str | None = None

    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        if in_quote is not None:
            buf.append(ch)
            if ch == in_quote and nxt == in_quote:
                buf.append(nxt)
                i += 2
                continue
            if ch == in_quote:
                segments.append((True, "".join(buf)))
                buf = []
                in_quote = None
            i += 1
            continue

        if ch in ("'", '"'):
            if buf:
                segments.append((False, "".join(buf)))
                buf = []
            in_quote = ch
            buf.append(ch)
            i += 1
            continue

        buf.append(ch)
        i += 1

    if buf:
        segments.append((in_quote is not None, "".join(buf)))
    return segments


def normalize_sql(sql: str) -> str:
    """Produce a canonical normalized SQL string for content-hash deduplication."""
    cleaned = _strip_sql_comments(sql).strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()

    segments = _split_literals_and_code(cleaned)
    normalized_parts: list[str] = []

    def _upper_kw(match: re.Match[str]) -> str:
        word = match.group(0)
        return word.upper() if word.upper() in _SQL_KEYWORDS_TO_NORMALIZE else word

    for is_lit, text in segments:
        if is_lit:
            normalized_parts.append(text)
        else:
            collapsed = re.sub(r"\s+", " ", text)
            collapsed = re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", _upper_kw, collapsed)
            normalized_parts.append(collapsed)

    return "".join(normalized_parts).strip()


def sql_content_hash(
    sql: str,
    *,
    hypothesis_family: str = "",
    verifier_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA-256 content hash over normalized SQL + metadata."""
    norm = normalize_sql(sql) if sql.strip() else ""
    payload = json.dumps(
        {
            "sql": norm,
            "hypothesis_family": hypothesis_family.strip(),
            "verifier_name": verifier_name or "",
            "params": params or {},
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_parquet_paths(sql: str) -> list[str]:
    """Extract string-literal paths passed to read_parquet(...) or parquet_scan(...)."""
    call_pattern = re.compile(
        r"\b(?:read_parquet|parquet_scan)\s*\(([^)]*)\)",
        re.IGNORECASE,
    )
    paths: list[str] = []
    for args_str in call_pattern.findall(sql):
        paths.extend(re.findall(r"'([^']*)'", args_str))
    return paths


def validate_readonly_sql(
    sql: str,
    config: SQLGuardConfig | None = None,
) -> str:
    """Validate that SQL is a single read-only SELECT/WITH statement safe under root guards."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("SQL query must be a non-empty string.")

    cleaned = _strip_sql_comments(sql).strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()

    if not cleaned:
        raise SQLValidationError("SQL query is empty after stripping comments.")

    segments = _split_literals_and_code(cleaned)

    if segments and segments[-1][0]:
        last_lit = segments[-1][1]
        if len(last_lit) < 2 or last_lit[0] != last_lit[-1]:
            raise SQLValidationError("Unclosed string or identifier literal in SQL.")

    code_only = " ".join(text for is_lit, text in segments if not is_lit)
    if ";" in code_only:
        raise SQLValidationError("Multi-statement SQL is forbidden.")
    if re.search(r"\$[A-Za-z0-9_]*\$", code_only):
        raise SQLValidationError("Dollar-quoted strings are forbidden in guarded SQL.")

    code_with_slits = "".join(
        " __SLIT__ " if (is_lit and text.startswith("'")) else text
        for is_lit, text in segments
    )
    if re.search(r"\b(?:FROM|JOIN)\s+__SLIT__", code_with_slits, re.IGNORECASE):
        raise SQLValidationError(
            "Direct file replacement scans (FROM/JOIN '...') are forbidden; "
            "query registered table views or read_parquet(...) within allowed_data_dir."
        )

    first_token_match = re.match(r"^\s*([A-Za-z_]+)", code_only)
    if not first_token_match or first_token_match.group(1).upper() not in ("SELECT", "WITH"):
        raise SQLValidationError("Only read-only SELECT or WITH queries are permitted.")

    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code_only)
    for tok in tokens:
        up = tok.upper()
        if up in _FORBIDDEN_KEYWORDS:
            raise SQLValidationError(f"Forbidden SQL keyword detected: {up}")
        if tok.lower() in _FORBIDDEN_FUNCTIONS:
            raise SQLValidationError(f"Forbidden DuckDB function detected: {tok}")

    for is_lit, lit_text in segments:
        if not is_lit:
            continue
        inner = lit_text[1:-1].strip().lower()
        for prefix in _FORBIDDEN_URI_PREFIXES:
            if inner.startswith(prefix):
                raise SQLValidationError(f"Remote URI access is forbidden in SQL: {prefix}")

    if config and config.allowed_data_dir:
        allowed_root = Path(config.allowed_data_dir).resolve()
        for raw_path in _extract_parquet_paths(cleaned):
            non_glob = raw_path.split("*")[0].split("?")[0]
            candidate = Path(non_glob)
            if not candidate.is_absolute():
                candidate = (allowed_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            try:
                candidate.relative_to(allowed_root)
            except ValueError as exc:
                raise SQLValidationError(
                    f"Parquet path escapes allowed data directory: {raw_path}"
                ) from exc

    return normalize_sql(cleaned)


def _walk_explain_json(node: Any, cardinalities: list[int], ops: list[str]) -> None:
    if isinstance(node, list):
        for item in node:
            _walk_explain_json(item, cardinalities, ops)
        return
    if not isinstance(node, dict):
        return

    name = node.get("name") or node.get("operator_name")
    if isinstance(name, str):
        ops.append(name.strip())

    extra = node.get("extra_info")
    if isinstance(extra, dict):
        est = extra.get("Estimated Cardinality")
        if est is not None:
            try:
                cardinalities.append(int(str(est).lstrip("~").strip()))
            except ValueError:
                pass

    if "cardinality" in node:
        try:
            cardinalities.append(int(node["cardinality"]))
        except (TypeError, ValueError):
            pass

    for child_key in ("children", "plans"):
        if child_key in node:
            _walk_explain_json(node[child_key], cardinalities, ops)


def _estimate_dir_parquet_bytes(data_dir: str | None) -> int:
    if not data_dir:
        return 0
    root = Path(data_dir)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*.parquet") if p.is_file())


def estimate_query_cost(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    data_dir: str | None = None,
    config: SQLGuardConfig | None = None,
) -> CostEstimate:
    """Estimate rows and bytes scanned using DuckDB EXPLAIN before executing a query."""
    cfg = config or SQLGuardConfig()
    norm_sql = validate_readonly_sql(sql, cfg)

    cardinalities: list[int] = []
    ops: list[str] = []
    plan_summary = ""

    try:
        explain_rows = con.execute(f"EXPLAIN (FORMAT JSON) {norm_sql}").fetchall()
        for _, plan_val in explain_rows:
            if isinstance(plan_val, str):
                parsed = json.loads(plan_val)
                _walk_explain_json(parsed, cardinalities, ops)
        plan_summary = " -> ".join(ops[:10])
    except Exception:
        text_rows = con.execute(f"EXPLAIN {norm_sql}").fetchall()
        plan_text = "\n".join(str(r[1]) for r in text_rows if len(r) > 1)
        for match in re.findall(r"~?(\d+)\s+Rows", plan_text, re.IGNORECASE):
            cardinalities.append(int(match))
        plan_summary = plan_text[:240]

    estimated_rows = max(cardinalities) if cardinalities else 0
    estimated_bytes = estimated_rows * cfg.avg_bytes_per_row_fallback
    if estimated_bytes == 0 and data_dir:
        estimated_bytes = _estimate_dir_parquet_bytes(data_dir)

    within_budget = True
    rejection_reason: str | None = None
    if estimated_rows > cfg.max_estimated_rows:
        within_budget = False
        rejection_reason = (
            f"Estimated rows ({estimated_rows}) exceed guard cap ({cfg.max_estimated_rows})."
        )
    elif estimated_bytes > cfg.max_estimated_bytes:
        within_budget = False
        rejection_reason = (
            f"Estimated bytes ({estimated_bytes}) exceed guard cap ({cfg.max_estimated_bytes})."
        )

    return CostEstimate(
        estimated_units=estimated_rows,
        estimated_bytes=estimated_bytes,
        plan_summary=plan_summary,
        within_budget=within_budget,
        rejection_reason=rejection_reason,
    )


__all__ = [
    "SQLGuardConfig",
    "SQLValidationError",
    "estimate_query_cost",
    "normalize_sql",
    "sql_content_hash",
    "validate_readonly_sql",
]
