"""Incident Root-Cause Analysis Example (`MicroBenchDomain`).

Implements the modality-agnostic `Domain` protocol (`ares/contracts.py`) using the
pluggable `DuckDBBackend` (`ares/adapters/duckdb_backend.py`) over partitioned Parquet
shards (`service_logs`, `metrics`, `traces`) for 9 incident root-cause test cases:
- 3x Multi-Clue Join cases (`multi_clue_join`)
- 3x Look-Alike Decoy cases (`look_alike_decoy`)
- 3x Under-Determined cases (`under_determined`)

All incident-specific field names, verifier-to-field mappings, SQL magnet rules,
and remediation resolvers live strictly inside this domain plug-in.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ares.adapters.duckdb_backend import DuckDBBackend
from ares.contracts import (
    ArtifactRef,
    ExecutionBackend,
    GroundTruth,
    InvestigationBoard,
    NeedleSpec,
    Note,
    SearchQuestion,
    VerifierRegistry,
    VerifierResult,
)
from examples.incident_rca.artifacts import render_candidate_slice_plot

SERVICES: tuple[str, ...] = (
    "api-gateway",
    "checkout-svc",
    "auth-svc",
    "payments-db",
    "inventory-db",
    "cache-redis",
    "search-indexer",
    "notification-worker",
)

INCIDENT_PIVOTAL_FIELDS: list[str] = [
    "logs_retry_spike",
    "metrics_zscore_breach",
    "traces_error_cluster",
    "three_way_root_cause",
]
INCIDENT_REQUIRED_SIGNALS: tuple[str, ...] = (
    "logs_retry_spike",
    "metrics_zscore_breach",
    "traces_error_cluster",
)

UPSTREAM_CALL_MAP: dict[str, tuple[str, ...]] = {
    "api-gateway": ("checkout-svc", "auth-svc", "search-indexer"),
    "checkout-svc": ("payments-db", "inventory-db", "cache-redis", "notification-worker"),
    "auth-svc": ("cache-redis", "payments-db"),
    "search-indexer": ("inventory-db", "cache-redis"),
}

SHARD_IDS: tuple[str, ...] = ("shard-01", "shard-02", "shard-03", "shard-04")
REGIONS: tuple[str, ...] = ("us-east-1", "us-west-2", "eu-west-1")

METRIC_BASELINES: dict[str, tuple[float, float]] = {
    "pool_utilization_pct": (42.0, 5.0),
    "p99_latency_ms": (65.0, 10.0),
    "cpu_pct": (38.0, 6.0),
    "lock_wait_ms": (12.0, 3.0),
    "queue_depth": (25.0, 5.0),
}

OPERATIONS_BY_SERVICE: dict[str, tuple[str, ...]] = {
    "api-gateway": ("http.route", "http.proxy"),
    "checkout-svc": ("order.submit", "order.validate"),
    "auth-svc": ("auth.verify_token", "auth.refresh_session"),
    "payments-db": ("db.acquire_lock", "db.commit_tx"),
    "inventory-db": ("db.reserve_sku", "db.acquire_lock"),
    "cache-redis": ("cache.get", "cache.mget"),
    "search-indexer": ("index.query", "index.refresh"),
    "notification-worker": ("queue.publish", "notify.dispatch"),
}

BASE_START_TS = datetime(2026, 11, 21, 13, 0, 0, tzinfo=timezone.utc)
HORIZON_SECONDS = 7200  # 2-hour window: 13:00:00Z to 15:00:00Z

SERVICE_LOGS_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("us", tz="UTC")),
        ("service", pa.string()),
        ("downstream_service", pa.string()),
        ("trace_id", pa.string()),
        ("shard_id", pa.string()),
        ("region", pa.string()),
        ("level", pa.string()),
        ("error_code", pa.string()),
        ("retry_count", pa.int32()),
        ("latency_ms", pa.float64()),
        ("message", pa.string()),
    ]
)

METRICS_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("us", tz="UTC")),
        ("service", pa.string()),
        ("shard_id", pa.string()),
        ("region", pa.string()),
        ("metric_name", pa.string()),
        ("value", pa.float64()),
        ("baseline_mean", pa.float64()),
        ("baseline_std", pa.float64()),
        ("host", pa.string()),
    ]
)

TRACES_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("us", tz="UTC")),
        ("trace_id", pa.string()),
        ("span_id", pa.string()),
        ("parent_service", pa.string()),
        ("service", pa.string()),
        ("shard_id", pa.string()),
        ("operation", pa.string()),
        ("status", pa.string()),
        ("error_kind", pa.string()),
        ("duration_ms", pa.float64()),
    ]
)

FAULT_METRIC_MAP: dict[str, tuple[str, str, str]] = {
    "db_lock_timeout": (
        "lock_wait_ms",
        "LOCK_TIMEOUT",
        "retrying downstream database transaction after lock wait timeout",
    ),
    "pool_exhaustion": (
        "pool_utilization_pct",
        "POOL_EXHAUSTED",
        "retrying downstream call after connection pool wait exceeded threshold",
    ),
    "auth_token_drift": (
        "p99_latency_ms",
        "DEADLINE_EXCEEDED",
        "retrying auth verification after upstream deadline exceeded",
    ),
    "unverified_db_drift": (
        "lock_wait_ms",
        "LOCK_TIMEOUT",
        "retrying downstream database query after transient timeout",
    ),
    "unverified_auth_errors": (
        "p99_latency_ms",
        "DEADLINE_EXCEEDED",
        "retrying auth verification after token validation error",
    ),
    "unverified_cache_latency": (
        "p99_latency_ms",
        "DEADLINE_EXCEEDED",
        "retrying cache lookup after elevated tail latency",
    ),
}

_VERIFIER_FIELD_MAP: dict[str, str] = {
    "verify_upstream_retry_spike": "logs_retry_spike",
    "verify_metric_zscore_breach": "metrics_zscore_breach",
    "verify_trace_error_cluster": "traces_error_cluster",
    "verify_three_way_root_cause": "three_way_root_cause",
}


def _parse_iso_utc(ts_str: str) -> datetime:
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_epoch_us(dt: datetime) -> int:
    return int(dt.timestamp() * 1_000_000)


def _stable_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _shard_for_case(case_id: str) -> str:
    idx = _stable_seed(17, case_id) % len(SHARD_IDS)
    return SHARD_IDS[idx]


def _decoy_shard_for_case(case_id: str) -> str:
    idx = (_stable_seed(17, case_id) + 1) % len(SHARD_IDS)
    return SHARD_IDS[idx]


def _extract_candidate_services(note: Note) -> str | list[str]:
    candidates = note.evidence.get("candidates")
    if note.row_count == 0 or not isinstance(candidates, list) or len(candidates) == 0:
        return "none"
    services = [
        str(c["service"])
        for c in candidates
        if isinstance(c, dict) and c.get("service") is not None
    ]
    if not services:
        return "none"
    if note.row_count == 1 or len(services) == 1:
        return services[0]
    return services


def extract_incident_note_fields(note: Note) -> dict[str, Any] | None:
    """Deterministic verifier-note -> canonical board field extractor for incident logs."""
    vname = note.verifier_name
    if vname not in _VERIFIER_FIELD_MAP:
        return None

    fname = _VERIFIER_FIELD_MAP[vname]
    out: dict[str, Any] = {}

    if vname == "verify_three_way_root_cause":
        roots = note.evidence.get("verified_root_causes")
        if isinstance(roots, list):
            if len(roots) == 0 or note.row_count == 0:
                out[fname] = "none"
            elif len(roots) == 1:
                out[fname] = str(roots[0])
            else:
                out[fname] = [str(r) for r in roots]
        else:
            out[fname] = _extract_candidate_services(note)
    else:
        out[fname] = _extract_candidate_services(note)

    if vname == "verify_metric_zscore_breach" and note.certification == "CERTIFIED":
        candidates = note.evidence.get("candidates")
        if isinstance(candidates, list) and len(candidates) > 0:
            if len(candidates) == 1:
                c0 = candidates[0]
                if isinstance(c0, dict) and c0.get("metric_name") is not None:
                    out["metric_anomaly_type"] = str(c0["metric_name"])
            else:
                mapped = {
                    str(c["service"]): str(c["metric_name"])
                    for c in candidates
                    if isinstance(c, dict)
                    and c.get("service") is not None
                    and c.get("metric_name") is not None
                }
                if mapped:
                    out["metric_anomaly_type"] = mapped

    return out


def _build_default_needle_menu() -> list[NeedleSpec]:
    """Return the 9 canonical Stage 1 micro-benchmark cases."""
    return [
        # --- 3x Multi-Clue Join Cases ---
        NeedleSpec(
            case_id="mc_01_payments_db_lock",
            category="multi_clue_join",
            scenario_type="db_lock_timeout",
            target_entity="payments-db",
            subtlety="medium",
            description=(
                "Correlated checkout-svc retry spike, payments-db lock_wait_ms breach, "
                "and LOCK_TIMEOUT spans on shared trace_ids."
            ),
            expected_decisions={
                "root_cause": "payments-db",
                "recommended_action": "rollback",
                "sufficiently_verified": "yes",
            },
            seed=101,
            metadata={
                "upstream_service": "checkout-svc",
                "onset_ts": "2026-11-21T14:03:00Z",
                "window_seconds": 120,
            },
        ),
        NeedleSpec(
            case_id="mc_02_auth_pool_exhaustion",
            category="multi_clue_join",
            scenario_type="pool_exhaustion",
            target_entity="auth-svc",
            subtlety="high",
            description=(
                "Subtle api-gateway retry cluster aligned with auth-svc pool saturation "
                "and POOL_EXHAUSTED trace spans."
            ),
            expected_decisions={
                "root_cause": "auth-svc",
                "recommended_action": "scale_pool",
                "sufficiently_verified": "yes",
            },
            seed=102,
            metadata={
                "upstream_service": "api-gateway",
                "onset_ts": "2026-11-21T13:32:00Z",
                "window_seconds": 120,
            },
        ),
        NeedleSpec(
            case_id="mc_03_inventory_db_lock",
            category="multi_clue_join",
            scenario_type="db_lock_timeout",
            target_entity="inventory-db",
            subtlety="medium",
            description=(
                "checkout-svc retries joining inventory-db lock_wait_ms surge and "
                "downstream LOCK_TIMEOUT spans."
            ),
            expected_decisions={
                "root_cause": "inventory-db",
                "recommended_action": "restart",
                "sufficiently_verified": "yes",
            },
            seed=103,
            metadata={
                "upstream_service": "checkout-svc",
                "onset_ts": "2026-11-21T14:22:00Z",
                "window_seconds": 120,
            },
        ),
        # --- 3x Look-Alike Decoy Cases ---
        NeedleSpec(
            case_id="decoy_01_payments_vs_cache",
            category="look_alike_decoy",
            scenario_type="pool_exhaustion",
            target_entity="payments-db",
            decoy_entity="cache-redis",
            subtlety="high",
            description=(
                "High-volume cache-redis decoy matches logs+metrics (2/3) with healthy traces; "
                "subtler payments-db needle matches all 3 signals."
            ),
            expected_decisions={
                "root_cause": "payments-db",
                "recommended_action": "scale_pool",
                "sufficiently_verified": "yes",
            },
            seed=201,
            metadata={
                "upstream_service": "checkout-svc",
                "decoy_upstream_service": "checkout-svc",
                "onset_ts": "2026-11-21T13:45:00Z",
                "window_seconds": 120,
            },
        ),
        NeedleSpec(
            case_id="decoy_02_auth_vs_search",
            category="look_alike_decoy",
            scenario_type="auth_token_drift",
            target_entity="auth-svc",
            decoy_entity="search-indexer",
            subtlety="high",
            description=(
                "High-volume search-indexer decoy matches logs+traces (2/3) with normal metrics; "
                "subtler auth-svc needle matches all 3 signals."
            ),
            expected_decisions={
                "root_cause": "auth-svc",
                "recommended_action": "rollback",
                "sufficiently_verified": "yes",
            },
            seed=202,
            metadata={
                "upstream_service": "api-gateway",
                "decoy_upstream_service": "api-gateway",
                "onset_ts": "2026-11-21T14:10:00Z",
                "window_seconds": 120,
            },
        ),
        NeedleSpec(
            case_id="decoy_03_inventory_vs_notify",
            category="look_alike_decoy",
            scenario_type="db_lock_timeout",
            target_entity="inventory-db",
            decoy_entity="notification-worker",
            subtlety="high",
            description=(
                "High-volume notification-worker decoy matches metrics+traces (2/3) without "
                "upstream retry logs; subtler inventory-db needle matches all 3 signals."
            ),
            expected_decisions={
                "root_cause": "inventory-db",
                "recommended_action": "restart",
                "sufficiently_verified": "yes",
            },
            seed=203,
            metadata={
                "upstream_service": "checkout-svc",
                "decoy_upstream_service": "checkout-svc",
                "onset_ts": "2026-11-21T13:20:00Z",
                "window_seconds": 120,
            },
        ),
        # --- 3x Under-Determined Cases ---
        NeedleSpec(
            case_id="ud_01_missing_traces",
            category="under_determined",
            scenario_type="unverified_db_drift",
            target_entity="payments-db",
            missing_signal="traces",
            subtlety="medium",
            description=(
                "Upstream retries and payments-db metric breach exist, but downstream "
                "trace spans show no correlated errors (must remain UNCERTIFIED)."
            ),
            expected_decisions={
                "root_cause": "none_verified",
                "recommended_action": "escalate",
                "sufficiently_verified": "no",
            },
            seed=301,
            metadata={
                "upstream_service": "checkout-svc",
                "onset_ts": "2026-11-21T13:55:00Z",
                "window_seconds": 120,
            },
        ),
        NeedleSpec(
            case_id="ud_02_missing_metrics",
            category="under_determined",
            scenario_type="unverified_auth_errors",
            target_entity="auth-svc",
            missing_signal="metrics",
            subtlety="medium",
            description=(
                "api-gateway retries and auth-svc trace errors align, but auth-svc "
                "metrics stay within normal baseline (must remain UNCERTIFIED)."
            ),
            expected_decisions={
                "root_cause": "none_verified",
                "recommended_action": "escalate",
                "sufficiently_verified": "no",
            },
            seed=302,
            metadata={
                "upstream_service": "api-gateway",
                "onset_ts": "2026-11-21T14:30:00Z",
                "window_seconds": 120,
            },
        ),
        NeedleSpec(
            case_id="ud_03_missing_logs",
            category="under_determined",
            scenario_type="unverified_cache_latency",
            target_entity="cache-redis",
            missing_signal="logs",
            subtlety="medium",
            description=(
                "cache-redis metrics breach and trace errors exist, but no upstream "
                "retry spike is logged (must remain UNCERTIFIED)."
            ),
            expected_decisions={
                "root_cause": "none_verified",
                "recommended_action": "escalate",
                "sufficiently_verified": "no",
            },
            seed=303,
            metadata={
                "upstream_service": "checkout-svc",
                "onset_ts": "2026-11-21T13:15:00Z",
                "window_seconds": 120,
            },
        ),
    ]


class MicroBenchDomain:
    """Stage 1 micro-benchmark domain for software incident root-cause analysis."""

    name: str = "incident_logs_micro_bench"

    def __init__(self, num_background_shards: int = 4) -> None:
        self._num_background_shards = max(1, num_background_shards)
        self._menu = _build_default_needle_menu()

    def create_backend(self, workspace_dir: str) -> ExecutionBackend:
        """Instantiate the tabular DuckDB + Parquet execution backend."""
        return DuckDBBackend(workspace_dir)

    def pivotal_fields(self) -> list[str]:
        return list(INCIDENT_PIVOTAL_FIELDS)

    def required_signal_fields(self) -> tuple[str, ...]:
        return INCIDENT_REQUIRED_SIGNALS

    def extract_field_patches_from_note(self, note: Note) -> dict[str, Any] | None:
        return extract_incident_note_fields(note)

    def generate(self, out_dir: str, size_gb: float = 0.02, seed: int = 42) -> None:
        """Write deterministic background haystack as partitioned Parquet shards."""
        root = Path(out_dir)
        for table_name in ("service_logs", "metrics", "traces"):
            table_dir = root / table_name
            table_dir.mkdir(parents=True, exist_ok=True)
            for existing in table_dir.glob("*.parquet"):
                existing.unlink()

        scale = max(1.0, size_gb / 0.02)
        total_logs = int(80_000 * scale)
        total_traces = int(60_000 * scale)
        num_shards = self._num_background_shards

        self._generate_background_logs(root / "service_logs", total_logs, num_shards, seed)
        self._generate_background_metrics(root / "metrics", num_shards, seed, scale)
        self._generate_background_traces(root / "traces", total_traces, num_shards, seed)

    def _generate_background_logs(
        self,
        table_dir: Path,
        total_rows: int,
        num_shards: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(_stable_seed(seed, "bg_logs"))
        start_us = _to_epoch_us(BASE_START_TS)

        pairs: list[tuple[str, str]] = [
            (up, down)
            for up, downs in UPSTREAM_CALL_MAP.items()
            for down in downs
        ]
        pair_up = np.array([p[0] for p in pairs])
        pair_down = np.array([p[1] for p in pairs])
        shards_arr = np.array(SHARD_IDS)
        regions_arr = np.array(REGIONS)

        rows_per_shard = total_rows // num_shards
        remainder = total_rows % num_shards
        offset = 0

        for shard_idx in range(num_shards):
            n = rows_per_shard + (1 if shard_idx < remainder else 0)
            ts_offsets_s = rng.integers(0, HORIZON_SECONDS, size=n, dtype=np.int64)
            ts_us = (
                start_us
                + ts_offsets_s * 1_000_000
                + rng.integers(0, 1_000_000, size=n, dtype=np.int64)
            )
            ts_us.sort()

            pair_idx = rng.integers(0, len(pairs), size=n)
            services = pair_up[pair_idx]
            downstreams = pair_down[pair_idx]
            shard_ids = shards_arr[rng.integers(0, len(SHARD_IDS), size=n)]
            regions = regions_arr[rng.integers(0, len(REGIONS), size=n)]

            u = rng.random(n)
            levels = np.where(u < 0.95, "INFO", np.where(u < 0.985, "WARN", "ERROR"))
            error_codes = np.where(
                levels == "INFO",
                "OK",
                np.where(levels == "WARN", "RATE_LIMIT", "CONN_RESET"),
            )
            retry_counts = np.where(
                levels == "INFO",
                0,
                np.where(levels == "WARN", 1, rng.integers(1, 3, size=n)),
            ).astype(np.int32)
            latencies = np.round(
                np.where(
                    levels == "INFO",
                    rng.normal(45.0, 12.0, size=n).clip(5.0, 220.0),
                    rng.normal(180.0, 45.0, size=n).clip(40.0, 500.0),
                ),
                2,
            )
            messages = np.where(
                levels == "INFO",
                "request completed normally",
                np.where(
                    levels == "WARN",
                    "transient downstream throttle",
                    "single attempt reset",
                ),
            )
            trace_ids = [f"tr-bg-{offset + i:08d}" for i in range(n)]
            offset += n

            table = pa.table(
                {
                    "ts": pa.array(ts_us, type=pa.timestamp("us", tz="UTC")),
                    "service": pa.array(services, type=pa.string()),
                    "downstream_service": pa.array(downstreams, type=pa.string()),
                    "trace_id": pa.array(trace_ids, type=pa.string()),
                    "shard_id": pa.array(shard_ids, type=pa.string()),
                    "region": pa.array(regions, type=pa.string()),
                    "level": pa.array(levels, type=pa.string()),
                    "error_code": pa.array(error_codes, type=pa.string()),
                    "retry_count": pa.array(retry_counts, type=pa.int32()),
                    "latency_ms": pa.array(latencies, type=pa.float64()),
                    "message": pa.array(messages, type=pa.string()),
                },
                schema=SERVICE_LOGS_SCHEMA,
            )
            pq.write_table(table, table_dir / f"part_{shard_idx:04d}.parquet")

    def _generate_background_metrics(
        self,
        table_dir: Path,
        num_shards: int,
        seed: int,
        scale: float,
    ) -> None:
        rng = np.random.default_rng(_stable_seed(seed, "bg_metrics"))
        start_us = _to_epoch_us(BASE_START_TS)
        step_seconds = 15 if scale <= 1.5 else max(5, int(15 / scale))
        num_buckets = HORIZON_SECONDS // step_seconds
        metric_names = ("pool_utilization_pct", "p99_latency_ms", "lock_wait_ms")

        bucket_ts = np.array(
            [start_us + i * step_seconds * 1_000_000 for i in range(num_buckets)],
            dtype=np.int64,
        )

        service_splits = np.array_split(np.array(SERVICES), num_shards)
        for shard_idx, svc_group in enumerate(service_splits):
            if len(svc_group) == 0:
                continue
            ts_list: list[np.ndarray] = []
            svc_list: list[np.ndarray] = []
            shd_list: list[np.ndarray] = []
            reg_list: list[np.ndarray] = []
            mname_list: list[np.ndarray] = []
            val_list: list[np.ndarray] = []
            mean_list: list[np.ndarray] = []
            std_list: list[np.ndarray] = []
            host_list: list[np.ndarray] = []

            for svc in svc_group:
                for shd in SHARD_IDS:
                    reg = REGIONS[_stable_seed(seed, f"{svc}:{shd}") % len(REGIONS)]
                    host = f"{svc}-{shd}.internal"
                    for mname in metric_names:
                        b_mean, b_std = METRIC_BASELINES[mname]
                        z = rng.normal(0.0, 0.85, size=num_buckets).clip(-2.4, 2.4)
                        vals = np.round(np.maximum(0.5, b_mean + z * b_std), 3)

                        ts_list.append(bucket_ts)
                        svc_list.append(np.full(num_buckets, svc))
                        shd_list.append(np.full(num_buckets, shd))
                        reg_list.append(np.full(num_buckets, reg))
                        mname_list.append(np.full(num_buckets, mname))
                        val_list.append(vals)
                        mean_list.append(np.full(num_buckets, b_mean, dtype=np.float64))
                        std_list.append(np.full(num_buckets, b_std, dtype=np.float64))
                        host_list.append(np.full(num_buckets, host))

            table = pa.table(
                {
                    "ts": pa.array(np.concatenate(ts_list), type=pa.timestamp("us", tz="UTC")),
                    "service": pa.array(np.concatenate(svc_list), type=pa.string()),
                    "shard_id": pa.array(np.concatenate(shd_list), type=pa.string()),
                    "region": pa.array(np.concatenate(reg_list), type=pa.string()),
                    "metric_name": pa.array(np.concatenate(mname_list), type=pa.string()),
                    "value": pa.array(np.concatenate(val_list), type=pa.float64()),
                    "baseline_mean": pa.array(np.concatenate(mean_list), type=pa.float64()),
                    "baseline_std": pa.array(np.concatenate(std_list), type=pa.float64()),
                    "host": pa.array(np.concatenate(host_list), type=pa.string()),
                },
                schema=METRICS_SCHEMA,
            )
            pq.write_table(table, table_dir / f"part_{shard_idx:04d}.parquet")

    def _generate_background_traces(
        self,
        table_dir: Path,
        total_rows: int,
        num_shards: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(_stable_seed(seed, "bg_traces"))
        start_us = _to_epoch_us(BASE_START_TS)

        pairs: list[tuple[str, str]] = [
            (up, down)
            for up, downs in UPSTREAM_CALL_MAP.items()
            for down in downs
        ]
        pair_up = np.array([p[0] for p in pairs])
        pair_down = np.array([p[1] for p in pairs])
        shards_arr = np.array(SHARD_IDS)

        rows_per_shard = total_rows // num_shards
        remainder = total_rows % num_shards
        offset = 0

        for shard_idx in range(num_shards):
            n = rows_per_shard + (1 if shard_idx < remainder else 0)
            ts_offsets_s = rng.integers(0, HORIZON_SECONDS, size=n, dtype=np.int64)
            ts_us = (
                start_us
                + ts_offsets_s * 1_000_000
                + rng.integers(0, 1_000_000, size=n, dtype=np.int64)
            )
            ts_us.sort()

            pair_idx = rng.integers(0, len(pairs), size=n)
            parents = pair_up[pair_idx]
            services = pair_down[pair_idx]
            shard_ids = shards_arr[rng.integers(0, len(SHARD_IDS), size=n)]

            operations = np.array([OPERATIONS_BY_SERVICE[s][0] for s in services])
            is_err = rng.random(n) < 0.015
            statuses = np.where(is_err, "ERROR", "OK")
            error_kinds = np.where(is_err, "CONN_REFUSED", "")
            durations = np.round(
                np.where(
                    is_err,
                    rng.normal(280.0, 60.0, size=n).clip(120.0, 490.0),
                    rng.normal(32.0, 10.0, size=n).clip(3.0, 160.0),
                ),
                2,
            )
            trace_ids = [f"tr-bg-{offset + i:08d}" for i in range(n)]
            span_ids = [f"sp-bg-{offset + i:08d}" for i in range(n)]
            offset += n

            table = pa.table(
                {
                    "ts": pa.array(ts_us, type=pa.timestamp("us", tz="UTC")),
                    "trace_id": pa.array(trace_ids, type=pa.string()),
                    "span_id": pa.array(span_ids, type=pa.string()),
                    "parent_service": pa.array(parents, type=pa.string()),
                    "service": pa.array(services, type=pa.string()),
                    "shard_id": pa.array(shard_ids, type=pa.string()),
                    "operation": pa.array(operations, type=pa.string()),
                    "status": pa.array(statuses, type=pa.string()),
                    "error_kind": pa.array(error_kinds, type=pa.string()),
                    "duration_ms": pa.array(durations, type=pa.float64()),
                },
                schema=TRACES_SCHEMA,
            )
            pq.write_table(table, table_dir / f"part_{shard_idx:04d}.parquet")

    def plant(self, out_dir: str, needle: NeedleSpec) -> GroundTruth:
        """Append fresh Parquet shards containing the needle (and decoy if configured)."""
        root = Path(out_dir)
        for table_name in ("service_logs", "metrics", "traces"):
            (root / table_name).mkdir(parents=True, exist_ok=True)

        meta = needle.metadata or {}
        svc = needle.target_entity
        upstream_svc = str(meta.get("upstream_service", "checkout-svc"))
        onset_str = str(meta.get("onset_ts", "2026-11-21T14:00:00Z"))
        window_seconds = int(meta.get("window_seconds", 120))

        rng = np.random.default_rng(_stable_seed(needle.seed, needle.case_id))
        onset_dt = _parse_iso_utc(onset_str)
        end_dt = onset_dt + timedelta(seconds=window_seconds)
        target_shard = _shard_for_case(needle.case_id)
        region = REGIONS[_stable_seed(needle.seed, f"region:{needle.case_id}") % len(REGIONS)]

        suffix = hashlib.sha256(f"{needle.case_id}:{needle.seed}".encode("utf-8")).hexdigest()[:8]

        log_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []

        event_count = {"low": 36, "medium": 26, "high": 18}[needle.subtlety]

        include_logs = needle.missing_signal != "logs"
        include_metrics = needle.missing_signal != "metrics"
        include_traces = needle.missing_signal != "traces"

        metric_name, trace_error_kind, log_msg = FAULT_METRIC_MAP.get(
            needle.scenario_type,
            ("p99_latency_ms", "DEADLINE_EXCEEDED", "retrying downstream call after timeout"),
        )

        trace_prefix = hashlib.sha1(f"{needle.case_id}:t".encode("utf-8")).hexdigest()[:6]
        shared_trace_ids = [f"tr-{trace_prefix}-{i:04d}" for i in range(event_count)]

        offsets_s = np.sort(rng.uniform(2.0, window_seconds - 2.0, size=event_count))

        if include_logs:
            for tid, off in zip(shared_trace_ids, offsets_s):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=float(off)))
                log_rows.append(
                    {
                        "ts": ts_us,
                        "service": upstream_svc,
                        "downstream_service": svc,
                        "trace_id": tid,
                        "shard_id": target_shard,
                        "region": region,
                        "level": "ERROR",
                        "error_code": "UPSTREAM_RETRY",
                        "retry_count": int(rng.integers(3, 6)),
                        "latency_ms": round(float(rng.uniform(950.0, 2200.0)), 2),
                        "message": log_msg,
                    }
                )

        if include_metrics:
            b_mean, b_std = METRIC_BASELINES[metric_name]
            num_buckets = max(6, window_seconds // 15)
            for b in range(num_buckets):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=b * 15))
                z_val = float(rng.uniform(5.2, 7.5))
                metric_rows.append(
                    {
                        "ts": ts_us,
                        "service": svc,
                        "shard_id": target_shard,
                        "region": region,
                        "metric_name": metric_name,
                        "value": round(b_mean + z_val * b_std, 3),
                        "baseline_mean": b_mean,
                        "baseline_std": b_std,
                        "host": f"{svc}-{target_shard}.internal",
                    }
                )

        if include_traces:
            op = OPERATIONS_BY_SERVICE.get(svc, ("rpc.call",))[0]
            for idx, (tid, off) in enumerate(zip(shared_trace_ids, offsets_s)):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=float(off) + 0.05))
                trace_rows.append(
                    {
                        "ts": ts_us,
                        "trace_id": tid,
                        "span_id": f"sp-{trace_prefix}-{idx:04d}",
                        "parent_service": upstream_svc,
                        "service": svc,
                        "shard_id": target_shard,
                        "operation": op,
                        "status": "ERROR",
                        "error_kind": trace_error_kind,
                        "duration_ms": round(float(rng.uniform(1100.0, 2400.0)), 2),
                    }
                )

        if needle.category == "look_alike_decoy" and needle.decoy_entity:
            self._append_decoy_rows(
                needle=needle,
                rng=rng,
                onset_dt=onset_dt,
                window_seconds=window_seconds,
                upstream_svc=upstream_svc,
                region=region,
                log_rows=log_rows,
                metric_rows=metric_rows,
                trace_rows=trace_rows,
            )

        planted_files: list[str] = []
        if log_rows:
            p = root / "service_logs" / f"part_append_{suffix}.parquet"
            pq.write_table(self._rows_to_table(log_rows, SERVICE_LOGS_SCHEMA), p)
            planted_files.append(str(p))
        if metric_rows:
            p = root / "metrics" / f"part_append_{suffix}.parquet"
            pq.write_table(self._rows_to_table(metric_rows, METRICS_SCHEMA), p)
            planted_files.append(str(p))
        if trace_rows:
            p = root / "traces" / f"part_append_{suffix}.parquet"
            pq.write_table(self._rows_to_table(trace_rows, TRACES_SCHEMA), p)
            planted_files.append(str(p))

        is_verified = needle.category != "under_determined"
        pivotal_clues: list[str] = []
        if include_logs:
            pivotal_clues.append("logs_retry_spike")
        if include_metrics:
            pivotal_clues.append("metrics_zscore_breach")
        if include_traces:
            pivotal_clues.append("traces_error_cluster")

        expected_action = (
            needle.expected_decisions.get("recommended_action", "escalate")
            if is_verified
            else "escalate"
        )
        return GroundTruth(
            case_id=needle.case_id,
            category=needle.category,
            primary_target=svc if is_verified else None,
            expected_decisions={
                "root_cause": svc if is_verified else "none_verified",
                "recommended_action": expected_action,
                "sufficiently_verified": "yes" if is_verified else "no",
            },
            is_sufficiently_verified=is_verified,
            expected_status="CERTIFIED" if is_verified else "UNCERTIFIED",
            pivotal_fields=pivotal_clues,
            decoy_target=needle.decoy_entity,
            planted_artifacts=planted_files,
            metadata={
                "onset_ts": onset_dt.isoformat().replace("+00:00", "Z"),
                "end_ts": end_dt.isoformat().replace("+00:00", "Z"),
                "shard_id": target_shard,
                "shared_trace_ids": shared_trace_ids,
            },
        )

    def _append_decoy_rows(
        self,
        *,
        needle: NeedleSpec,
        rng: np.random.Generator,
        onset_dt: datetime,
        window_seconds: int,
        upstream_svc: str,
        region: str,
        log_rows: list[dict[str, Any]],
        metric_rows: list[dict[str, Any]],
        trace_rows: list[dict[str, Any]],
    ) -> None:
        assert needle.decoy_entity is not None
        decoy_svc = needle.decoy_entity
        decoy_up = str((needle.metadata or {}).get("decoy_upstream_service", upstream_svc))
        decoy_shard = _decoy_shard_for_case(needle.case_id)
        decoy_prefix = hashlib.sha1(f"{needle.case_id}:d".encode("utf-8")).hexdigest()[:6]

        decoy_count = 80
        decoy_tids = [f"tr-{decoy_prefix}-{i:04d}" for i in range(decoy_count)]
        offsets_s = np.sort(rng.uniform(1.0, window_seconds - 1.0, size=decoy_count))

        missing_choice = _stable_seed(99, needle.case_id) % 3
        if "vs_cache" in needle.case_id:
            decoy_missing = "traces"
        elif "vs_search" in needle.case_id:
            decoy_missing = "metrics"
        elif "vs_notify" in needle.case_id:
            decoy_missing = "logs"
        else:
            decoy_missing = ("traces", "metrics", "logs")[missing_choice]

        if decoy_missing != "logs":
            for tid, off in zip(decoy_tids, offsets_s):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=float(off)))
                log_rows.append(
                    {
                        "ts": ts_us,
                        "service": decoy_up,
                        "downstream_service": decoy_svc,
                        "trace_id": tid,
                        "shard_id": decoy_shard,
                        "region": region,
                        "level": "ERROR",
                        "error_code": "UPSTREAM_RETRY",
                        "retry_count": int(rng.integers(3, 6)),
                        "latency_ms": round(float(rng.uniform(700.0, 1500.0)), 2),
                        "message": "retrying downstream call after elevated response time",
                    }
                )

        if decoy_missing != "metrics":
            b_mean, b_std = METRIC_BASELINES["p99_latency_ms"]
            num_buckets = max(6, window_seconds // 15)
            for b in range(num_buckets):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=b * 15))
                z_val = float(rng.uniform(5.5, 8.0))
                metric_rows.append(
                    {
                        "ts": ts_us,
                        "service": decoy_svc,
                        "shard_id": decoy_shard,
                        "region": region,
                        "metric_name": "p99_latency_ms",
                        "value": round(b_mean + z_val * b_std, 3),
                        "baseline_mean": b_mean,
                        "baseline_std": b_std,
                        "host": f"{decoy_svc}-{decoy_shard}.internal",
                    }
                )

        if decoy_missing != "traces":
            op = OPERATIONS_BY_SERVICE.get(decoy_svc, ("rpc.call",))[0]
            for idx, (tid, off) in enumerate(zip(decoy_tids, offsets_s)):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=float(off) + 0.04))
                trace_rows.append(
                    {
                        "ts": ts_us,
                        "trace_id": tid,
                        "span_id": f"sp-{decoy_prefix}-{idx:04d}",
                        "parent_service": decoy_up,
                        "service": decoy_svc,
                        "shard_id": decoy_shard,
                        "operation": op,
                        "status": "ERROR",
                        "error_kind": "DEADLINE_EXCEEDED",
                        "duration_ms": round(float(rng.uniform(950.0, 1900.0)), 2),
                    }
                )

    @staticmethod
    def _rows_to_table(rows: list[dict[str, Any]], schema: pa.Schema) -> pa.Table:
        cols: dict[str, pa.Array] = {}
        for field in schema:
            values = [r[field.name] for r in rows]
            cols[field.name] = pa.array(values, type=field.type)
        return pa.table(cols, schema=schema)

    def needle_menu(self) -> list[NeedleSpec]:
        """Return the 9 selectable micro-benchmark needle definitions."""
        return [n.model_copy(deep=True) for n in self._menu]

    def environment_description(self) -> str:
        """Tables, columns, join keys, and semantics for Hydrogen/Carbon prompts."""
        return (
            "DuckDB SQL table views (auto-registered; query by view name directly):\n"
            "1. service_logs:\n"
            "   - ts (TIMESTAMP UTC), service (VARCHAR, upstream caller), "
            "downstream_service (VARCHAR, callee), trace_id (VARCHAR), shard_id (VARCHAR), "
            "region (VARCHAR), level ('INFO'|'WARN'|'ERROR'), error_code (VARCHAR), "
            "retry_count (INTEGER), latency_ms (DOUBLE), message (VARCHAR).\n"
            "2. metrics:\n"
            "   - ts (TIMESTAMP UTC), service (VARCHAR), shard_id (VARCHAR), region (VARCHAR), "
            "metric_name ('pool_utilization_pct'|'p99_latency_ms'|'lock_wait_ms'), "
            "value (DOUBLE), baseline_mean (DOUBLE), baseline_std (DOUBLE), host (VARCHAR).\n"
            "   - Baseline z-score = (value - baseline_mean) / baseline_std.\n"
            "3. traces:\n"
            "   - ts (TIMESTAMP UTC), trace_id (VARCHAR), span_id (VARCHAR), "
            "parent_service (VARCHAR), service (VARCHAR, downstream callee), "
            "shard_id (VARCHAR), operation (VARCHAR), status ('OK'|'ERROR'), "
            "error_kind (VARCHAR), duration_ms (DOUBLE).\n"
            "Join keys: service_logs.trace_id = traces.trace_id AND "
            "service_logs.shard_id = traces.shard_id AND "
            "service_logs.downstream_service = traces.service; align metrics on "
            "(service, shard_id) within +/- 180s of the incident window.\n"
            "A CERTIFIED root cause requires all 3 signals aligned on the same (service, shard_id) "
            "within a narrow window: (1) >=12 upstream ERROR retries (retry_count >= 3), "
            "(2) >=4 metric buckets with z-score >= 4.0, and (3) >=12 correlated ERROR trace spans "
            "with duration_ms >= 800ms on shared trace_ids."
        )

    def magnet_rules(self) -> str:
        """DuckDB read-only SQL syntax & guard rules for automated swarm queries."""
        return (
            "1. Reference DuckDB table view names directly (`service_logs`, `metrics`, `traces`). "
            "NEVER call `read_parquet(...)` and NEVER include `<data_dir>` placeholders.\n"
            "2. Emit single read-only `SELECT` or `WITH ... SELECT` queries only (no semicolons, "
            "no DDL/DML).\n"
            "3. Bound result sets with `GROUP BY` / `ORDER BY` / `LIMIT <= 50`."
        )

    def search_questions(self) -> list[SearchQuestion]:
        """Closed-set search questions and domain criteria for the board."""
        rc_options = [*SERVICES, "none_verified"]
        rc_criteria: dict[str, str] = {
            "none_verified": (
                "Select if ANY of the three required signal fields (`logs_retry_spike`, "
                "`metrics_zscore_breach`, `traces_error_cluster`) is missing, has "
                "status != 'certified', or has value in ('unknown', 'none'), OR if no single "
                "component appears in all three."
            )
        }
        for svc in SERVICES:
            rc_criteria[svc] = (
                f"Select if all three required signal fields (`logs_retry_spike`, "
                f"`metrics_zscore_breach`, `traces_error_cluster`) have status='certified' "
                f"and '{svc}' is present in ALL THREE required signal fields (as the string "
                "value or inside the candidate list)."
            )

        action_criteria = {
            "escalate": (
                "Select when ANY of `logs_retry_spike`, `metrics_zscore_breach`, or "
                "`traces_error_cluster` is missing, has status != 'certified', or has value in "
                "('unknown', 'none'), or when no single component appears in all three."
            ),
            "rollback": (
                "Select ONLY when a component is certified across all three required signals and "
                "certified evidence (`metric_anomaly_type`, `fault_type`, or `candidate_remediation`) "
                "for that verified component indicates a deployment/token regression or payments-db "
                "lock regression warranting rollback (e.g. auth-svc with p99_latency_ms / "
                "auth_token_drift, payments-db with lock_wait_ms, or candidate_remediation='rollback')."
            ),
            "restart": (
                "Select ONLY when a component is certified across all three required signals and "
                "certified evidence (`metric_anomaly_type`, `fault_type`, or `candidate_remediation`) "
                "for that verified component indicates a transient lock/resource wedge warranting "
                "restart (e.g. inventory-db with lock_wait_ms or candidate_remediation='restart')."
            ),
            "scale_pool": (
                "Select ONLY when a component is certified across all three required signals and "
                "certified evidence (`metric_anomaly_type`, `fault_type`, or `candidate_remediation`) "
                "for that verified component indicates connection pool exhaustion "
                "(pool_utilization_pct, POOL_EXHAUSTED, pool_exhaustion, or "
                "candidate_remediation='scale_pool')."
            ),
        }

        return [
            SearchQuestion(
                question_id="root_cause",
                question="Which component is the verified root cause?",
                options=rc_options,
                question_type="multiple_choice",
                instructions=(
                    "Which component is the verified root cause? Evaluate strictly from the board's "
                    "`fields` (`value` and `status`). The three required deterministic signal fields "
                    "are `logs_retry_spike`, `metrics_zscore_breach`, and `traces_error_cluster` "
                    "(a signal field may hold either a single service string or a list of candidate "
                    "services when a partial 2-of-3 decoy co-exists with a 3-of-3 root cause). "
                    "A component is verified if and only if ALL THREE required signal fields have "
                    "`status='certified'` and that component appears in all three."
                ),
                option_criteria=rc_criteria,
            ),
            SearchQuestion(
                question_id="recommended_action",
                question="What is the recommended remediation action?",
                options=["rollback", "restart", "scale_pool", "escalate"],
                question_type="multiple_choice",
                instructions=(
                    "What is the recommended remediation action? Select a concrete remediation ONLY "
                    "when a single root cause component appears across all three `status='certified'` "
                    "required signal fields (`logs_retry_spike`, `metrics_zscore_breach`, "
                    "`traces_error_cluster`); otherwise select 'escalate'. Inspect `metric_anomaly_type` "
                    "for that verified component to distinguish pool exhaustion vs lock/latency regressions."
                ),
                option_criteria=action_criteria,
            ),
            SearchQuestion(
                question_id="sufficiently_verified",
                question="Is the root cause sufficiently verified by all three deterministic signals?",
                options=["yes", "no"],
                question_type="yes_no",
                instructions=(
                    "Is the root cause sufficiently verified by all three deterministic signals? "
                    "Evaluate strictly from `logs_retry_spike`, `metrics_zscore_breach`, and "
                    "`traces_error_cluster`. Answer true when at least one component "
                    "appears across ALL THREE required signal fields where each of those three fields "
                    "has status='certified'. Answer false whenever ANY of those three "
                    "fields is missing, has status != 'certified', or has value in ('none', 'unknown')."
                ),
            ),
        ]

    def resolve_answers(
        self,
        board: InvestigationBoard,
        questions: Sequence[SearchQuestion],
        primary_target: str | None,
    ) -> dict[str, str]:
        """Deterministically resolve remediation action from certified metric anomaly metadata."""
        del questions
        if primary_target is None:
            return {"recommended_action": "escalate", "sufficiently_verified": "no"}

        metric_name = ""
        anom_fld = board.fields.get("metric_anomaly_type")
        if anom_fld is not None and anom_fld.status == "certified":
            if isinstance(anom_fld.value, str):
                metric_name = anom_fld.value
            elif isinstance(anom_fld.value, dict):
                metric_name = str(anom_fld.value.get(primary_target, ""))

        rem_fld = board.fields.get("candidate_remediation")
        if rem_fld is not None and rem_fld.status == "certified" and isinstance(rem_fld.value, str):
            if rem_fld.value in ("rollback", "restart", "scale_pool"):
                return {"recommended_action": rem_fld.value, "sufficiently_verified": "yes"}

        if metric_name in ("pool_utilization_pct", "POOL_EXHAUSTED", "pool_exhaustion"):
            action = "scale_pool"
        elif primary_target == "inventory-db" and metric_name == "lock_wait_ms":
            action = "restart"
        elif metric_name in ("lock_wait_ms", "p99_latency_ms"):
            action = "rollback"
        else:
            action = "rollback"
        return {"recommended_action": action, "sufficiently_verified": "yes"}

    def hypothesis_families(self) -> list[str]:
        return [
            "upstream_retry_spike",
            "metric_baseline_zscore_breach",
            "downstream_trace_timeout_cluster",
            "cross_service_three_way_join",
            "decoy_disambiguation_check",
        ]

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register(
            "verify_upstream_retry_spike",
            verify_upstream_retry_spike,
            description="Detects >=12 upstream ERROR retry logs (retry_count >= 3) on (downstream_service, shard_id).",
            trusted=True,
        )
        registry.register(
            "verify_metric_zscore_breach",
            verify_metric_zscore_breach,
            description="Detects >=4 metric buckets with z-score >= 4.0 on (service, shard_id).",
            trusted=True,
        )
        registry.register(
            "verify_trace_error_cluster",
            verify_trace_error_cluster,
            description="Detects >=12 ERROR trace spans with duration_ms >= 800ms on (service, shard_id).",
            trusted=True,
        )
        registry.register(
            "verify_three_way_root_cause",
            verify_three_way_root_cause,
            description=(
                "Compound 3-way verifier joining service_logs retries, traces error spans "
                "on shared trace_ids, and aligned metrics z-score breaches."
            ),
            trusted=True,
        )

    def score(self, board: dict[str, Any], truth: GroundTruth) -> dict[str, Any]:
        """Evaluate final board answer, certification accuracy, and candidate narrowing."""
        conc = board.get("conclusion") or board.get("decision") or {}
        concs_map = board.get("conclusions") or board.get("decisions") or {}

        rc_conc = concs_map.get("root_cause", conc)
        act_conc = concs_map.get("recommended_action", {})
        ver_conc = concs_map.get("sufficiently_verified", {})

        predicted_rc = rc_conc.get("answer", conc.get("answer", "unknown"))
        predicted_action = act_conc.get("answer", "escalate")
        predicted_verified_raw = ver_conc.get("answer")

        confidence = float(conc.get("confidence", 0.0))
        verified_flag = conc.get("verified")
        meta = conc.get("metadata") or {}
        uncertified_share = float(
            meta.get(
                "uncertified_influence_share",
                conc.get("uncertified_influence_share", 0.0 if verified_flag else 1.0),
            )
        )
        unknown_voi_max = float(
            meta.get(
                "unknown_voi_max",
                conc.get("unknown_voi_max", 0.0 if verified_flag else 1.0),
            )
        )

        if verified_flag is not None:
            board_certified = (
                bool(verified_flag)
                and confidence >= 0.85
                and predicted_rc not in (None, "", "unknown", "none_verified")
                and (predicted_verified_raw in (None, "yes"))
            )
        else:
            board_certified = (
                confidence >= 0.85
                and uncertified_share <= 0.15
                and unknown_voi_max < 0.20
                and predicted_rc not in (None, "", "unknown", "none_verified")
                and (predicted_verified_raw in (None, "yes"))
            )

        if truth.is_sufficiently_verified:
            root_cause_correct = predicted_rc == truth.primary_target
        else:
            root_cause_correct = (
                not board_certified
                or predicted_rc in (None, "none_verified", "unknown")
            )

        expected_action = truth.expected_decisions.get("recommended_action", "escalate")
        action_correct = predicted_action == expected_action
        certification_correct = board_certified == truth.is_sufficiently_verified
        false_certainty = (not truth.is_sufficiently_verified) and board_certified
        decoy_fallen_for = (
            truth.decoy_target is not None
            and predicted_rc == truth.decoy_target
        )
        decoy_avoided = (
            None if truth.decoy_target is None else (not decoy_fallen_for)
        )

        fields = board.get("fields", {})
        pivotal_set = set(truth.pivotal_fields)
        pivotal_priorities = [
            float(f.get("priority", f.get("influence", 0.0)))
            for fname, f in fields.items()
            if fname in pivotal_set
        ]
        distractor_priorities = [
            float(f.get("priority", f.get("influence", 0.0)))
            for fname, f in fields.items()
            if fname not in pivotal_set and f.get("status") != "unknown"
        ]
        pivotal_priority_mean = (
            float(np.mean(pivotal_priorities)) if pivotal_priorities else None
        )
        distractor_priority_mean = (
            float(np.mean(distractor_priorities)) if distractor_priorities else None
        )

        total_candidates = len(SERVICES)
        if predicted_rc in SERVICES:
            candidate_narrowing = round(1.0 - (1.0 / total_candidates), 4)
        else:
            candidate_narrowing = 0.0

        overall_pass = (
            root_cause_correct
            and certification_correct
            and not decoy_fallen_for
            and not false_certainty
        )

        return {
            "case_id": truth.case_id,
            "category": truth.category,
            "expected_status": truth.expected_status,
            "predicted_root_cause": predicted_rc,
            "predicted_action": predicted_action,
            "confidence": confidence,
            "certified_field_ratio": float(conc.get("certified_field_ratio", 0.0)),
            "board_certified": board_certified,
            "root_cause_correct": root_cause_correct,
            "action_correct": action_correct,
            "certification_correct": certification_correct,
            "false_certainty": false_certainty,
            "decoy_fallen_for": decoy_fallen_for,
            "decoy_avoided": decoy_avoided,
            "pivotal_priority_mean": pivotal_priority_mean,
            "distractor_priority_mean": distractor_priority_mean,
            "candidate_narrowing": candidate_narrowing,
            "overall_pass": overall_pass,
        }


# ----------------------------------------------------------------------
# Deterministic TRUSTED DuckDB Verifiers (No LLM Calls)
# ----------------------------------------------------------------------


def _parquet_glob(data_dir: str, table: str) -> str:
    path = Path(data_dir) / table / "*.parquet"
    return str(path).replace("'", "''")


def verify_upstream_retry_spike(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    min_retries = int(params.get("min_retries", 3))
    min_count = int(params.get("min_count", 12))
    service_filter = params.get("service")

    logs_glob = _parquet_glob(data_dir, "service_logs")
    where_service = ""
    query_params: list[Any] = [min_retries, min_count]
    if service_filter:
        where_service = "AND downstream_service = ?"
        query_params = [min_retries, service_filter, min_count]

    sql = f"""
        SELECT
            downstream_service AS service,
            shard_id,
            COUNT(*) AS retry_log_count,
            COUNT(DISTINCT trace_id) AS distinct_traces,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts
        FROM read_parquet('{logs_glob}')
        WHERE level = 'ERROR'
          AND retry_count >= ?
          {where_service}
        GROUP BY downstream_service, shard_id
        HAVING COUNT(DISTINCT trace_id) >= ?
        ORDER BY distinct_traces DESC, service ASC
    """
    rows = con.execute(sql, query_params).fetchall()
    candidates = [
        {
            "service": r[0],
            "shard_id": r[1],
            "retry_log_count": int(r[2]),
            "distinct_traces": int(r[3]),
            "first_ts": r[4].isoformat() if r[4] else None,
            "last_ts": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]
    passed = len(candidates) > 0
    return VerifierResult(
        verifier_name="verify_upstream_retry_spike",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidates": candidates},
        row_count=len(candidates),
        summary=(
            f"Found {len(candidates)} (service, shard) retry spike(s)."
            if passed
            else "No upstream retry spike met threshold."
        ),
    )


def verify_metric_zscore_breach(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    min_z = float(params.get("min_zscore", 4.0))
    min_buckets = int(params.get("min_buckets", 4))
    service_filter = params.get("service")

    metrics_glob = _parquet_glob(data_dir, "metrics")
    where_service = ""
    query_params: list[Any] = [min_z, min_buckets]
    if service_filter:
        where_service = "AND service = ?"
        query_params = [min_z, service_filter, min_buckets]

    sql = f"""
        SELECT
            service,
            shard_id,
            metric_name,
            COUNT(*) AS breach_buckets,
            MAX((value - baseline_mean) / NULLIF(baseline_std, 0)) AS max_zscore,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts
        FROM read_parquet('{metrics_glob}')
        WHERE baseline_std > 0
          AND ((value - baseline_mean) / baseline_std) >= ?
          {where_service}
        GROUP BY service, shard_id, metric_name
        HAVING COUNT(*) >= ?
        ORDER BY max_zscore DESC, service ASC
    """
    rows = con.execute(sql, query_params).fetchall()
    candidates = [
        {
            "service": r[0],
            "shard_id": r[1],
            "metric_name": r[2],
            "breach_buckets": int(r[3]),
            "max_zscore": round(float(r[4]), 3),
            "first_ts": r[5].isoformat() if r[5] else None,
            "last_ts": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]
    passed = len(candidates) > 0
    return VerifierResult(
        verifier_name="verify_metric_zscore_breach",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidates": candidates},
        row_count=len(candidates),
        summary=(
            f"Found {len(candidates)} metric z-score breach(es)."
            if passed
            else "No metric z-score breach met threshold."
        ),
    )


def verify_trace_error_cluster(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    min_duration_ms = float(params.get("min_duration_ms", 800.0))
    min_count = int(params.get("min_count", 12))
    service_filter = params.get("service")

    traces_glob = _parquet_glob(data_dir, "traces")
    where_service = ""
    query_params: list[Any] = [min_duration_ms, min_count]
    if service_filter:
        where_service = "AND service = ?"
        query_params = [min_duration_ms, service_filter, min_count]

    sql = f"""
        SELECT
            service,
            shard_id,
            COUNT(*) AS error_spans,
            COUNT(DISTINCT trace_id) AS distinct_traces,
            MAX(duration_ms) AS max_duration_ms,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts
        FROM read_parquet('{traces_glob}')
        WHERE status = 'ERROR'
          AND duration_ms >= ?
          {where_service}
        GROUP BY service, shard_id
        HAVING COUNT(DISTINCT trace_id) >= ?
        ORDER BY distinct_traces DESC, service ASC
    """
    rows = con.execute(sql, query_params).fetchall()
    candidates = [
        {
            "service": r[0],
            "shard_id": r[1],
            "error_spans": int(r[2]),
            "distinct_traces": int(r[3]),
            "max_duration_ms": round(float(r[4]), 2),
            "first_ts": r[5].isoformat() if r[5] else None,
            "last_ts": r[6].isoformat() if r[6] else None,
        }
        for r in rows
    ]
    passed = len(candidates) > 0
    return VerifierResult(
        verifier_name="verify_trace_error_cluster",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={"candidates": candidates},
        row_count=len(candidates),
        summary=(
            f"Found {len(candidates)} downstream trace error cluster(s)."
            if passed
            else "No downstream trace error cluster met threshold."
        ),
    )


def verify_three_way_root_cause(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    params: dict[str, Any] | None = None,
) -> VerifierResult:
    params = params or {}
    min_retries = int(params.get("min_retries", 3))
    min_duration_ms = float(params.get("min_duration_ms", 800.0))
    min_z = float(params.get("min_zscore", 4.0))
    min_traces = int(params.get("min_traces", 12))
    min_buckets = int(params.get("min_buckets", 4))
    window_pad_s = int(params.get("window_pad_seconds", 180))

    logs_glob = _parquet_glob(data_dir, "service_logs")
    metrics_glob = _parquet_glob(data_dir, "metrics")
    traces_glob = _parquet_glob(data_dir, "traces")

    sql = f"""
        WITH log_trace_join AS (
            SELECT
                l.downstream_service AS service,
                l.shard_id,
                COUNT(DISTINCT l.trace_id) AS correlated_traces,
                MIN(l.ts) AS window_start,
                MAX(l.ts) AS window_end,
                MAX(t.duration_ms) AS max_trace_ms
            FROM read_parquet('{logs_glob}') l
            INNER JOIN read_parquet('{traces_glob}') t
                ON l.trace_id = t.trace_id
               AND l.shard_id = t.shard_id
               AND l.downstream_service = t.service
            WHERE l.level = 'ERROR'
              AND l.retry_count >= ?
              AND t.status = 'ERROR'
              AND t.duration_ms >= ?
            GROUP BY l.downstream_service, l.shard_id
            HAVING COUNT(DISTINCT l.trace_id) >= ?
        ),
        metric_breaches AS (
            SELECT
                service,
                shard_id,
                metric_name,
                ts,
                (value - baseline_mean) / baseline_std AS zscore
            FROM read_parquet('{metrics_glob}')
            WHERE baseline_std > 0
              AND ((value - baseline_mean) / baseline_std) >= ?
        )
        SELECT
            lt.service,
            lt.shard_id,
            lt.correlated_traces,
            lt.window_start,
            lt.window_end,
            lt.max_trace_ms,
            COUNT(mb.ts) AS aligned_metric_buckets,
            MAX(mb.zscore) AS max_zscore
        FROM log_trace_join lt
        INNER JOIN metric_breaches mb
            ON lt.service = mb.service
           AND lt.shard_id = mb.shard_id
           AND mb.ts BETWEEN lt.window_start - INTERVAL '{window_pad_s} seconds'
                         AND lt.window_end + INTERVAL '{window_pad_s} seconds'
        GROUP BY
            lt.service,
            lt.shard_id,
            lt.correlated_traces,
            lt.window_start,
            lt.window_end,
            lt.max_trace_ms
        HAVING COUNT(mb.ts) >= ?
        ORDER BY lt.correlated_traces DESC, max_zscore DESC, lt.service ASC
    """
    rows = con.execute(
        sql,
        [min_retries, min_duration_ms, min_traces, min_z, min_buckets],
    ).fetchall()

    candidates = [
        {
            "service": r[0],
            "shard_id": r[1],
            "correlated_traces": int(r[2]),
            "window_start": r[3].isoformat() if r[3] else None,
            "window_end": r[4].isoformat() if r[4] else None,
            "max_trace_ms": round(float(r[5]), 2),
            "aligned_metric_buckets": int(r[6]),
            "max_zscore": round(float(r[7]), 3),
        }
        for r in rows
    ]
    passed = len(candidates) > 0
    artifacts: list[ArtifactRef] = []
    artifact_dir = params.get("artifact_dir")
    if passed and artifact_dir:
        max_artifacts = max(1, min(int(params.get("max_artifacts", 3)), 3))
        for cand in candidates[:max_artifacts]:
            artifacts.append(
                render_candidate_slice_plot(
                    con=con,
                    data_dir=data_dir,
                    service=cand["service"],
                    shard_id=cand["shard_id"],
                    out_dir=str(artifact_dir),
                    window_start=cand["window_start"],
                    window_end=cand["window_end"],
                    window_pad_seconds=window_pad_s,
                )
            )

    return VerifierResult(
        verifier_name="verify_three_way_root_cause",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={
            "verified_root_causes": [c["service"] for c in candidates],
            "candidates": candidates,
        },
        artifacts=artifacts,
        row_count=len(candidates),
        summary=(
            f"CERTIFIED 3-way root cause(s): {[c['service'] for c in candidates]}"
            if passed
            else "UNCERTIFIED: No component satisfied all three aligned signals."
        ),
    )
