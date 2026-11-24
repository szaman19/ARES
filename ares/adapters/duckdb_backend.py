"""DuckDB + Parquet `ExecutionBackend` adapter for tabular ARES domains.

Implements the narrow-waist `ExecutionBackend` protocol (`ares/contracts.py`) with:
- Automatic zero-DDL discovery of `<data_dir>/<table>/*.parquet` views
- Read-only SQL validation & path sandboxing (Invariant I6)
- DuckDB EXPLAIN cost estimation & cheapest-first ordering
- Content-hash deduplication (Invariant I6)
- Taint/certification enforcement (only TRUSTED verifiers can emit CERTIFIED notes)
- Immutable Note emission with ArtifactRef support (Invariant I3)
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, Field

from ares.adapters.sql_guard import (
    SQLGuardConfig,
    SQLValidationError,
    estimate_query_cost,
    sql_content_hash,
    validate_readonly_sql,
)
from ares.contracts import (
    CertificationStatus,
    CostEstimate,
    HypothesisSpec,
    Note,
    VerifierRegistry,
    VerifierResult,
)
from ares.engine import is_verifier_trusted
from ares.store import NoteStore


def _json_safe(value: Any) -> Any:
    """Convert DuckDB cell values into JSON-serializable Python primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class QueryExecutionResult(BaseModel):
    """Structured result of a single guarded DuckDB read-only SQL execution."""

    query: str
    content_hash: str
    cost_estimate: CostEstimate
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    truncated: bool = False
    duration_ms: float = Field(default=0.0, ge=0.0)


class DuckDBBackend:
    """Pluggable DuckDB `ExecutionBackend` for tabular Parquet domains."""

    def __init__(
        self,
        data_dir: str,
        *,
        guard_config: SQLGuardConfig | None = None,
        threads_per_connection: int = 2,
    ) -> None:
        self.data_dir = str(Path(data_dir).resolve())
        base_cfg = guard_config or SQLGuardConfig()
        if base_cfg.allowed_data_dir is None:
            base_cfg = base_cfg.model_copy(update={"allowed_data_dir": self.data_dir})
        self.guard_config = base_cfg
        self.threads_per_connection = max(1, int(threads_per_connection))

    def _open_connection(self) -> duckdb.DuckDBPyConnection:
        """Open an isolated in-memory DuckDB connection with dynamic Parquet views."""
        con = duckdb.connect(database=":memory:")
        con.execute(f"SET threads TO {self.threads_per_connection}")

        root = Path(self.data_dir)
        if root.exists() and root.is_dir():
            for subdir in sorted(root.iterdir()):
                if not subdir.is_dir():
                    continue
                if not any(subdir.glob("*.parquet")):
                    continue
                table_name = subdir.name
                if not table_name.isidentifier():
                    continue
                if (
                    self.guard_config.allowed_tables is not None
                    and table_name not in self.guard_config.allowed_tables
                ):
                    continue
                glob_str = str(subdir / "*.parquet").replace("'", "''")
                con.execute(
                    f'CREATE OR REPLACE VIEW "{table_name}" AS '
                    f"SELECT * FROM read_parquet('{glob_str}')"
                )
        return con

    def validate_query(self, query: str) -> str:
        """Validate read-only SQL safety under `self.guard_config`."""
        return validate_readonly_sql(query, self.guard_config)

    def compute_content_hash(
        self,
        query: str,
        *,
        hypothesis_family: str = "",
        verifier_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Compute deterministic SHA-256 content hash over normalized SQL + metadata."""
        return sql_content_hash(
            query,
            hypothesis_family=hypothesis_family,
            verifier_name=verifier_name,
            params=params,
        )

    def estimate_cost(
        self,
        query: str,
        *,
        config: SQLGuardConfig | None = None,
    ) -> CostEstimate:
        """Validate SQL and run DuckDB EXPLAIN cost estimation."""
        cfg = config or self.guard_config
        con = self._open_connection()
        try:
            return estimate_query_cost(con, query, data_dir=self.data_dir, config=cfg)
        finally:
            con.close()

    def execute_query(
        self,
        query: str,
        *,
        params: dict[str, Any] | None = None,
        requested_under: int = 0,
        hypothesis_family: str = "",
        content_hash: str = "",
        note_store: NoteStore | None = None,
        config: SQLGuardConfig | None = None,
    ) -> tuple[QueryExecutionResult, Note]:
        """Validate, cost-check, and execute a read-only SQL query, emitting an UNCERTIFIED Note."""
        del params
        cfg = config or self.guard_config
        norm_sql = validate_readonly_sql(query, cfg)
        c_hash = content_hash or sql_content_hash(
            norm_sql, hypothesis_family=hypothesis_family
        )

        con = self._open_connection()
        try:
            cost = estimate_query_cost(con, norm_sql, data_dir=self.data_dir, config=cfg)
            if not cost.within_budget:
                raise SQLValidationError(
                    cost.rejection_reason or "Query exceeded EXPLAIN cost budget."
                )

            t0 = time.perf_counter()
            fetch_limit = cfg.max_result_rows + 1
            capped_sql = f"SELECT * FROM ({norm_sql}) AS _ares_sub LIMIT {fetch_limit}"
            cursor = con.execute(capped_sql)
            columns = [desc[0] for desc in (cursor.description or [])]
            raw_rows = cursor.fetchall()
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        finally:
            con.close()

        truncated = len(raw_rows) > cfg.max_result_rows
        if truncated:
            raw_rows = raw_rows[: cfg.max_result_rows]

        rows = [
            {col: _json_safe(val) for col, val in zip(columns, row)}
            for row in raw_rows
        ]
        exec_result = QueryExecutionResult(
            query=norm_sql,
            content_hash=c_hash,
            cost_estimate=cost,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            duration_ms=duration_ms,
        )

        note_id = (
            note_store.next_note_id()
            if note_store is not None
            else f"n_q_{c_hash[:8]}"
        )
        note = Note(
            note_id=note_id,
            certification="UNCERTIFIED",
            sources=[f"query:{c_hash[:12]}"],
            artifacts=[],
            verifier_name=None,
            hypothesis_family=hypothesis_family,
            query=norm_sql,
            content_hash=c_hash,
            requested_under=requested_under,
            summary=(
                f"Ad-hoc read-only query returned {len(rows)} row(s)"
                + (" (truncated)" if truncated else "")
                + f" in {duration_ms:.1f}ms."
            ),
            evidence={
                "columns": columns,
                "rows": rows,
                "truncated": truncated,
                "estimated_units": cost.estimated_units,
                "estimated_bytes": cost.estimated_bytes,
                "duration_ms": duration_ms,
            },
            row_count=len(rows),
        )
        if note_store is not None:
            note_store.append(note)
        return exec_result, note

    def execute_verifier(
        self,
        registry: VerifierRegistry,
        verifier_name: str,
        *,
        params: dict[str, Any] | None = None,
        requested_under: int = 0,
        hypothesis_family: str = "",
        query: str = "",
        content_hash: str = "",
        note_store: NoteStore | None = None,
    ) -> tuple[VerifierResult, Note]:
        """Execute a registered deterministic DuckDB verifier over local Parquet shards."""
        verifier_fn = registry.get(verifier_name)
        trusted = is_verifier_trusted(registry, verifier_name)
        effective_params = dict(params) if params else {}

        norm_sql = ""
        if query.strip():
            norm_sql = validate_readonly_sql(query, self.guard_config)
        effective_hash = content_hash or sql_content_hash(
            norm_sql,
            hypothesis_family=hypothesis_family,
            verifier_name=verifier_name,
            params=effective_params,
        )

        con = self._open_connection()
        try:
            result = verifier_fn(con, self.data_dir, effective_params)
        finally:
            con.close()

        certification: CertificationStatus = (
            result.certification if (trusted and result.passed) else "UNCERTIFIED"
        )
        if certification != result.certification:
            result = result.model_copy(update={"certification": certification})

        note_id = (
            note_store.next_note_id()
            if note_store is not None
            else f"n_ver_{effective_hash[:8]}"
        )
        note = Note(
            note_id=note_id,
            certification=certification,
            sources=[f"verifier:{verifier_name}"],
            artifacts=list(result.artifacts),
            verifier_name=verifier_name,
            hypothesis_family=hypothesis_family,
            query=norm_sql,
            content_hash=effective_hash,
            requested_under=requested_under,
            summary=result.summary,
            evidence=dict(result.evidence),
            row_count=result.row_count,
        )
        if note_store is not None:
            note_store.append(note)
        return result, note

    def plan_and_execute_batch(
        self,
        specs: list[HypothesisSpec],
        *,
        registry: VerifierRegistry | None = None,
        note_store: NoteStore | None = None,
        max_concurrent: int = 4,
        artifact_dir: str | None = None,
    ) -> tuple[list[Note], dict[str, Any]]:
        """Validate, deduplicate, cost-sort (cheapest-first), and execute a batch of specs."""
        store = note_store if note_store is not None else NoteStore()
        seen_hashes: set[str] = set()
        planned: list[tuple[int, int, int, HypothesisSpec, str, str]] = []

        skipped_duplicates = 0
        rejected_guard = 0
        rejection_reasons: list[str] = []

        for idx, spec in enumerate(specs):
            try:
                norm_sql = validate_readonly_sql(spec.query, self.guard_config)
            except SQLValidationError as exc:
                rejected_guard += 1
                rejection_reasons.append(f"spec[{idx}]: {exc}")
                continue

            c_hash = sql_content_hash(
                norm_sql,
                hypothesis_family=spec.hypothesis_family,
                verifier_name=spec.verifier_name,
                params=spec.params,
            )
            if c_hash in seen_hashes or store.has_content_hash(c_hash):
                skipped_duplicates += 1
                continue
            seen_hashes.add(c_hash)

            try:
                cost = self.estimate_cost(norm_sql)
            except Exception as exc:
                rejected_guard += 1
                rejection_reasons.append(f"spec[{idx}] EXPLAIN failed: {exc}")
                continue

            if not cost.within_budget:
                rejected_guard += 1
                rejection_reasons.append(f"spec[{idx}]: {cost.rejection_reason}")
                continue

            planned.append(
                (cost.estimated_bytes, cost.estimated_units, idx, spec, norm_sql, c_hash)
            )

        planned.sort(key=lambda item: (item[0], item[1], item[2]))

        def _run_one(
            entry: tuple[int, int, int, HypothesisSpec, str, str],
        ) -> Note:
            _, _, _, spec, norm_sql, c_hash = entry
            if spec.verifier_name and registry is not None:
                params = dict(spec.params)
                if artifact_dir is not None:
                    params["artifact_dir"] = artifact_dir
                _, note = self.execute_verifier(
                    registry,
                    spec.verifier_name,
                    params=params,
                    requested_under=spec.requested_under,
                    hypothesis_family=spec.hypothesis_family,
                    query=norm_sql,
                    content_hash=c_hash,
                    note_store=store,
                )
                return note
            _, note = self.execute_query(
                norm_sql,
                params=spec.params,
                requested_under=spec.requested_under,
                hypothesis_family=spec.hypothesis_family,
                content_hash=c_hash,
                note_store=store,
            )
            return note

        notes: list[Note] = []
        workers = max(1, int(max_concurrent))
        if planned:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_run_one, entry) for entry in planned]
                for fut in futures:
                    notes.append(fut.result())

        stats = {
            "submitted": len(specs),
            "executed": len(notes),
            "skipped_duplicates": skipped_duplicates,
            "rejected_guard": rejected_guard,
            "rejection_reasons": rejection_reasons,
            "execution_order_hashes": [entry[5] for entry in planned],
        }
        return notes, stats


__all__ = [
    "DuckDBBackend",
    "QueryExecutionResult",
]
