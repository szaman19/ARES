"""Stage 1 Micro-Benchmark Domain (ARES Plan §6–§7, Agent A1).

Generates deterministic partitioned Parquet shards (`service_logs`, `metrics`,
`traces`) for 9 incident root-cause test cases:
- 3x Multi-Clue Join cases (`multi_clue_join`)
- 3x Look-Alike Decoy cases (`look_alike_decoy`)
- 3x Under-Determined cases (`under_determined`)

Conforms strictly to the `Domain` protocol in `contracts.py` / `domain/protocol.py`.
No ground-truth annotations or labels are ever written into Parquet columns.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from contracts import (
    DecisionQuestion,
    GroundTruth,
    NeedleSpec,
    VerifierRegistry,
    VerifierResult,
)

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
    # fault_type -> (metric_name, trace_error_kind, log_message)
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


def _build_default_needle_menu() -> list[NeedleSpec]:
    """Return the 9 canonical Stage 1 micro-benchmark cases."""
    return [
        # --- 3x Multi-Clue Join Cases ---
        NeedleSpec(
            case_id="mc_01_payments_db_lock",
            category="multi_clue_join",
            fault_type="db_lock_timeout",
            service="payments-db",
            upstream_service="checkout-svc",
            onset_ts="2026-11-21T14:03:00Z",
            window_seconds=120,
            subtlety="medium",
            description=(
                "Correlated checkout-svc retry spike, payments-db lock_wait_ms breach, "
                "and LOCK_TIMEOUT spans on shared trace_ids."
            ),
            recommended_action="rollback",
            seed=101,
        ),
        NeedleSpec(
            case_id="mc_02_auth_pool_exhaustion",
            category="multi_clue_join",
            fault_type="pool_exhaustion",
            service="auth-svc",
            upstream_service="api-gateway",
            onset_ts="2026-11-21T13:32:00Z",
            window_seconds=120,
            subtlety="high",
            description=(
                "Subtle api-gateway retry cluster aligned with auth-svc pool saturation "
                "and POOL_EXHAUSTED trace spans."
            ),
            recommended_action="scale_pool",
            seed=102,
        ),
        NeedleSpec(
            case_id="mc_03_inventory_db_lock",
            category="multi_clue_join",
            fault_type="db_lock_timeout",
            service="inventory-db",
            upstream_service="checkout-svc",
            onset_ts="2026-11-21T14:22:00Z",
            window_seconds=120,
            subtlety="medium",
            description=(
                "checkout-svc retries joining inventory-db lock_wait_ms surge and "
                "downstream LOCK_TIMEOUT spans."
            ),
            recommended_action="restart",
            seed=103,
        ),
        # --- 3x Look-Alike Decoy Cases ---
        NeedleSpec(
            case_id="decoy_01_payments_vs_cache",
            category="look_alike_decoy",
            fault_type="pool_exhaustion",
            service="payments-db",
            upstream_service="checkout-svc",
            decoy_service="cache-redis",
            decoy_upstream_service="checkout-svc",
            onset_ts="2026-11-21T13:45:00Z",
            window_seconds=120,
            subtlety="high",
            description=(
                "High-volume cache-redis decoy matches logs+metrics (2/3) with healthy traces; "
                "subtler payments-db needle matches all 3 signals."
            ),
            recommended_action="scale_pool",
            seed=201,
        ),
        NeedleSpec(
            case_id="decoy_02_auth_vs_search",
            category="look_alike_decoy",
            fault_type="auth_token_drift",
            service="auth-svc",
            upstream_service="api-gateway",
            decoy_service="search-indexer",
            decoy_upstream_service="api-gateway",
            onset_ts="2026-11-21T14:10:00Z",
            window_seconds=120,
            subtlety="high",
            description=(
                "High-volume search-indexer decoy matches logs+traces (2/3) with normal metrics; "
                "subtler auth-svc needle matches all 3 signals."
            ),
            recommended_action="rollback",
            seed=202,
        ),
        NeedleSpec(
            case_id="decoy_03_inventory_vs_notify",
            category="look_alike_decoy",
            fault_type="db_lock_timeout",
            service="inventory-db",
            upstream_service="checkout-svc",
            decoy_service="notification-worker",
            decoy_upstream_service="checkout-svc",
            onset_ts="2026-11-21T13:20:00Z",
            window_seconds=120,
            subtlety="high",
            description=(
                "High-volume notification-worker decoy matches metrics+traces (2/3) without "
                "upstream retry logs; subtler inventory-db needle matches all 3 signals."
            ),
            recommended_action="restart",
            seed=203,
        ),
        # --- 3x Under-Determined Cases ---
        NeedleSpec(
            case_id="ud_01_missing_traces",
            category="under_determined",
            fault_type="unverified_db_drift",
            service="payments-db",
            upstream_service="checkout-svc",
            missing_clue="traces",
            onset_ts="2026-11-21T13:55:00Z",
            window_seconds=120,
            subtlety="medium",
            description=(
                "Upstream retries and payments-db metric breach exist, but downstream "
                "trace spans show no correlated errors (must remain UNCERTIFIED)."
            ),
            recommended_action="escalate",
            seed=301,
        ),
        NeedleSpec(
            case_id="ud_02_missing_metrics",
            category="under_determined",
            fault_type="unverified_auth_errors",
            service="auth-svc",
            upstream_service="api-gateway",
            missing_clue="metrics",
            onset_ts="2026-11-21T14:30:00Z",
            window_seconds=120,
            subtlety="medium",
            description=(
                "api-gateway retries and auth-svc trace errors align, but auth-svc "
                "metrics stay within normal baseline (must remain UNCERTIFIED)."
            ),
            recommended_action="escalate",
            seed=302,
        ),
        NeedleSpec(
            case_id="ud_03_missing_logs",
            category="under_determined",
            fault_type="unverified_cache_latency",
            service="cache-redis",
            upstream_service="checkout-svc",
            missing_clue="logs",
            onset_ts="2026-11-21T13:15:00Z",
            window_seconds=120,
            subtlety="medium",
            description=(
                "cache-redis metrics breach and trace errors exist, but no upstream "
                "retry spike is logged (must remain UNCERTIFIED)."
            ),
            recommended_action="escalate",
            seed=303,
        ),
    ]


class MicroBenchDomain:
    """Stage 1 micro-benchmark domain for software incident root-cause analysis."""

    name: str = "incident_logs_micro_bench"

    def __init__(self, num_background_shards: int = 4) -> None:
        self._num_background_shards = max(1, num_background_shards)
        self._menu = _build_default_needle_menu()

    # ------------------------------------------------------------------
    # Domain protocol: generate()
    # ------------------------------------------------------------------

    def generate(self, out_dir: str, size_gb: float = 0.02, seed: int = 42) -> None:
        """Write deterministic background haystack as partitioned Parquet shards.

        At the default micro-benchmark scale (`size_gb <= 0.05`), generates
        ~180,000 total rows (~80K service_logs, ~40K metrics, ~60K traces)
        across partitioned shards in under ~1 second.
        """
        root = Path(out_dir)
        for table_name in ("service_logs", "metrics", "traces"):
            table_dir = root / table_name
            table_dir.mkdir(parents=True, exist_ok=True)
            for existing in table_dir.glob("*.parquet"):
                existing.unlink()

        scale = max(1.0, size_gb / 0.02)
        total_logs = int(80_000 * scale)
        total_traces = int(60_000 * scale)
        # Metrics are deterministic 15s buckets across services x shards x metrics
        # 480 buckets * 8 services * 4 shards * 3 core metrics = 46,080 rows at scale=1
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

        # Build valid (upstream, downstream) pairs
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
            ts_us = start_us + ts_offsets_s * 1_000_000 + rng.integers(0, 1_000_000, size=n, dtype=np.int64)
            ts_us.sort()

            pair_idx = rng.integers(0, len(pairs), size=n)
            services = pair_up[pair_idx]
            downstreams = pair_down[pair_idx]
            shard_ids = shards_arr[rng.integers(0, len(SHARD_IDS), size=n)]
            regions = regions_arr[rng.integers(0, len(REGIONS), size=n)]

            # Ambient noise: ~95% INFO, ~3.5% WARN, ~1.5% ERROR; retry_count <= 2 in background
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
                np.where(levels == "WARN", "transient downstream throttle", "single attempt reset"),
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

        # Partition by service groups across num_shards files
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
                        # Bounded normal noise: |z| <= 2.4 so background never triggers z >= 4.0 breach
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
            ts_us = start_us + ts_offsets_s * 1_000_000 + rng.integers(0, 1_000_000, size=n, dtype=np.int64)
            ts_us.sort()

            pair_idx = rng.integers(0, len(pairs), size=n)
            parents = pair_up[pair_idx]
            services = pair_down[pair_idx]
            shard_ids = shards_arr[rng.integers(0, len(SHARD_IDS), size=n)]

            operations = np.array([OPERATIONS_BY_SERVICE[s][0] for s in services])
            is_err = rng.random(n) < 0.015
            statuses = np.where(is_err, "ERROR", "OK")
            error_kinds = np.where(is_err, "CONN_REFUSED", "")
            # Ambient errors stay below 500ms so they don't form >= 800ms lock/timeout clusters
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

    # ------------------------------------------------------------------
    # Domain protocol: plant()
    # ------------------------------------------------------------------

    def plant(self, out_dir: str, needle: NeedleSpec) -> GroundTruth:
        """Append fresh Parquet shards containing the needle (and decoy if configured).

        Does NOT modify existing background shards. Filenames use an opaque hash
        suffix so file listings do not leak ground-truth case identifiers.
        """
        root = Path(out_dir)
        for table_name in ("service_logs", "metrics", "traces"):
            (root / table_name).mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(_stable_seed(needle.seed, needle.case_id))
        onset_dt = _parse_iso_utc(needle.onset_ts)
        end_dt = onset_dt + timedelta(seconds=needle.window_seconds)
        target_shard = _shard_for_case(needle.case_id)
        region = REGIONS[_stable_seed(needle.seed, f"region:{needle.case_id}") % len(REGIONS)]

        suffix = hashlib.sha256(f"{needle.case_id}:{needle.seed}".encode("utf-8")).hexdigest()[:8]

        log_rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []

        # Determine event count based on subtlety (still >= 12 verifier minimum)
        event_count = {"low": 36, "medium": 26, "high": 18}[needle.subtlety]

        include_logs = needle.missing_clue != "logs"
        include_metrics = needle.missing_clue != "metrics"
        include_traces = needle.missing_clue != "traces"

        metric_name, trace_error_kind, log_msg = FAULT_METRIC_MAP.get(
            needle.fault_type,
            ("p99_latency_ms", "DEADLINE_EXCEEDED", "retrying downstream call after timeout"),
        )

        # Generate opaque trace IDs (no "needle" substring!)
        trace_prefix = hashlib.sha1(f"{needle.case_id}:t".encode("utf-8")).hexdigest()[:6]
        shared_trace_ids = [f"tr-{trace_prefix}-{i:04d}" for i in range(event_count)]

        offsets_s = np.sort(rng.uniform(2.0, needle.window_seconds - 2.0, size=event_count))

        if include_logs:
            for tid, off in zip(shared_trace_ids, offsets_s):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=float(off)))
                log_rows.append(
                    {
                        "ts": ts_us,
                        "service": needle.upstream_service,
                        "downstream_service": needle.service,
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
            num_buckets = max(6, needle.window_seconds // 15)
            for b in range(num_buckets):
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=b * 15))
                z_val = float(rng.uniform(5.2, 7.5))
                metric_rows.append(
                    {
                        "ts": ts_us,
                        "service": needle.service,
                        "shard_id": target_shard,
                        "region": region,
                        "metric_name": metric_name,
                        "value": round(b_mean + z_val * b_std, 3),
                        "baseline_mean": b_mean,
                        "baseline_std": b_std,
                        "host": f"{needle.service}-{target_shard}.internal",
                    }
                )

        if include_traces:
            op = OPERATIONS_BY_SERVICE.get(needle.service, ("rpc.call",))[0]
            for idx, (tid, off) in enumerate(zip(shared_trace_ids, offsets_s)):
                # Trace span occurs within ~50ms of the upstream retry log
                ts_us = _to_epoch_us(onset_dt + timedelta(seconds=float(off) + 0.05))
                trace_rows.append(
                    {
                        "ts": ts_us,
                        "trace_id": tid,
                        "span_id": f"sp-{trace_prefix}-{idx:04d}",
                        "parent_service": needle.upstream_service,
                        "service": needle.service,
                        "shard_id": target_shard,
                        "operation": op,
                        "status": "ERROR",
                        "error_kind": trace_error_kind,
                        "duration_ms": round(float(rng.uniform(1100.0, 2400.0)), 2),
                    }
                )

        # Plant high-volume 2-of-3 decoy for look_alike_decoy cases
        if needle.category == "look_alike_decoy" and needle.decoy_service:
            self._append_decoy_rows(
                needle=needle,
                rng=rng,
                onset_dt=onset_dt,
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

        return GroundTruth(
            case_id=needle.case_id,
            category=needle.category,
            root_cause_component=needle.service if is_verified else None,
            recommended_action=needle.recommended_action if is_verified else "escalate",
            is_sufficiently_verified=is_verified,
            expected_status="CERTIFIED" if is_verified else "UNCERTIFIED",
            pivotal_clues=pivotal_clues,
            decoy_component=needle.decoy_service,
            onset_ts=onset_dt.isoformat().replace("+00:00", "Z"),
            end_ts=end_dt.isoformat().replace("+00:00", "Z"),
            shard_id=target_shard,
            shared_trace_ids=shared_trace_ids,
            planted_files=planted_files,
        )

    def _append_decoy_rows(
        self,
        *,
        needle: NeedleSpec,
        rng: np.random.Generator,
        onset_dt: datetime,
        region: str,
        log_rows: list[dict[str, Any]],
        metric_rows: list[dict[str, Any]],
        trace_rows: list[dict[str, Any]],
    ) -> None:
        """Plant a high-volume 2-of-3 decoy that misses exactly 1 signal."""
        assert needle.decoy_service is not None
        decoy_svc = needle.decoy_service
        decoy_up = needle.decoy_upstream_service or needle.upstream_service
        decoy_shard = _decoy_shard_for_case(needle.case_id)
        decoy_prefix = hashlib.sha1(f"{needle.case_id}:d".encode("utf-8")).hexdigest()[:6]

        # 3x-4x higher volume than the subtle true needle
        decoy_count = 80
        decoy_tids = [f"tr-{decoy_prefix}-{i:04d}" for i in range(decoy_count)]
        offsets_s = np.sort(rng.uniform(1.0, needle.window_seconds - 1.0, size=decoy_count))

        # Vary which 1 of 3 clues the decoy is missing based on case_id hash
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
            num_buckets = max(6, needle.window_seconds // 15)
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

    # ------------------------------------------------------------------
    # Domain protocol: metadata & prompts
    # ------------------------------------------------------------------

    def needle_menu(self) -> list[NeedleSpec]:
        """Return the 9 selectable micro-benchmark needle definitions."""
        return [n.model_copy(deep=True) for n in self._menu]

    def schema_description(self) -> str:
        """Tables, columns, join keys, and semantics for Hydrogen/Carbon prompts."""
        return (
            "DuckDB Parquet tables under <data_dir>:\n"
            "1. service_logs (<data_dir>/service_logs/*.parquet):\n"
            "   - ts (TIMESTAMP UTC), service (VARCHAR, upstream caller), "
            "downstream_service (VARCHAR, callee), trace_id (VARCHAR), shard_id (VARCHAR), "
            "region (VARCHAR), level ('INFO'|'WARN'|'ERROR'), error_code (VARCHAR), "
            "retry_count (INTEGER), latency_ms (DOUBLE), message (VARCHAR).\n"
            "2. metrics (<data_dir>/metrics/*.parquet):\n"
            "   - ts (TIMESTAMP UTC), service (VARCHAR), shard_id (VARCHAR), region (VARCHAR), "
            "metric_name ('pool_utilization_pct'|'p99_latency_ms'|'lock_wait_ms'), "
            "value (DOUBLE), baseline_mean (DOUBLE), baseline_std (DOUBLE), host (VARCHAR).\n"
            "   - Baseline z-score = (value - baseline_mean) / baseline_std.\n"
            "3. traces (<data_dir>/traces/*.parquet):\n"
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

    def decision_questions(self) -> list[DecisionQuestion]:
        """Closed-set questions the board must answer (ARES Plan §6)."""
        return [
            DecisionQuestion(
                question_id="root_cause",
                question="Which component is the verified root cause?",
                options=[*SERVICES, "none_verified"],
                question_type="multiple_choice",
            ),
            DecisionQuestion(
                question_id="recommended_action",
                question="What is the recommended remediation action?",
                options=["rollback", "restart", "scale_pool", "escalate"],
                question_type="multiple_choice",
            ),
            DecisionQuestion(
                question_id="sufficiently_verified",
                question="Is the root cause sufficiently verified by all three deterministic signals?",
                options=["yes", "no"],
                question_type="yes_no",
            ),
        ]

    def hypothesis_families(self) -> list[str]:
        """Initial failure/signal families for parallel swarm exploration."""
        return [
            "upstream_retry_spike",
            "metric_baseline_zscore_breach",
            "downstream_trace_timeout_cluster",
            "cross_service_three_way_join",
            "decoy_disambiguation_check",
        ]

    # ------------------------------------------------------------------
    # Domain protocol: register_verifiers()
    # ------------------------------------------------------------------

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        """Register deterministic TRUSTED DuckDB verifiers (no LLM calls)."""
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

    # ------------------------------------------------------------------
    # Domain protocol: score()
    # ------------------------------------------------------------------

    def score(self, board: dict[str, Any], truth: GroundTruth) -> dict[str, Any]:
        """Evaluate final board answer, certification accuracy, and candidate narrowing."""
        decision = board.get("decision", {})
        decisions_map = board.get("decisions", {})

        rc_dec = decisions_map.get("root_cause", decision)
        act_dec = decisions_map.get("recommended_action", {})
        ver_dec = decisions_map.get("sufficiently_verified", {})

        predicted_rc = rc_dec.get("answer", decision.get("answer", "unknown"))
        predicted_action = act_dec.get("answer", "escalate")
        predicted_verified_raw = ver_dec.get("answer")

        confidence = float(decision.get("confidence", 0.0))
        uncertified_share = float(decision.get("uncertified_influence_share", 1.0))
        unknown_voi_max = float(decision.get("unknown_voi_max", 1.0))

        # ARES Plan §5.2 Step 8 stop thresholds
        board_certified = (
            confidence >= 0.85
            and uncertified_share <= 0.15
            and unknown_voi_max < 0.20
            and predicted_rc not in (None, "", "unknown", "none_verified")
            and (predicted_verified_raw in (None, "yes"))
        )

        if truth.is_sufficiently_verified:
            root_cause_correct = predicted_rc == truth.root_cause_component
        else:
            root_cause_correct = (
                not board_certified
                or predicted_rc in (None, "none_verified", "unknown")
            )

        action_correct = predicted_action == truth.recommended_action
        certification_correct = board_certified == truth.is_sufficiently_verified
        false_certainty = (not truth.is_sufficiently_verified) and board_certified
        decoy_fallen_for = (
            truth.decoy_component is not None
            and predicted_rc == truth.decoy_component
        )
        decoy_avoided = (
            None if truth.decoy_component is None else (not decoy_fallen_for)
        )

        # Evaluate pivotal vs distractor field influence if fields are present
        fields = board.get("fields", {})
        pivotal_set = set(truth.pivotal_clues)
        pivotal_influences = [
            float(f.get("influence", 0.0))
            for fname, f in fields.items()
            if fname in pivotal_set
        ]
        distractor_influences = [
            float(f.get("influence", 0.0))
            for fname, f in fields.items()
            if fname not in pivotal_set and f.get("status") != "unknown"
        ]
        pivotal_influence_mean = (
            float(np.mean(pivotal_influences)) if pivotal_influences else None
        )
        distractor_influence_mean = (
            float(np.mean(distractor_influences)) if distractor_influences else None
        )

        # Candidate narrowing metric
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
            "uncertified_influence_share": uncertified_share,
            "board_certified": board_certified,
            "root_cause_correct": root_cause_correct,
            "action_correct": action_correct,
            "certification_correct": certification_correct,
            "false_certainty": false_certainty,
            "decoy_fallen_for": decoy_fallen_for,
            "decoy_avoided": decoy_avoided,
            "pivotal_influence_mean": pivotal_influence_mean,
            "distractor_influence_mean": distractor_influence_mean,
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
    """Verify whether any (downstream_service, shard_id) has >= 12 upstream retry ERROR logs."""
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
    """Verify whether any (service, shard_id) has >= 4 metric buckets with z-score >= 4.0."""
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
    """Verify whether any (service, shard_id) has >= 12 ERROR trace spans >= 800ms."""
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
    """Compound deterministic 3-way verifier across service_logs, metrics, and traces."""
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
    return VerifierResult(
        verifier_name="verify_three_way_root_cause",
        passed=passed,
        certification="CERTIFIED" if passed else "UNCERTIFIED",
        evidence={
            "verified_root_causes": [c["service"] for c in candidates],
            "candidates": candidates,
        },
        row_count=len(candidates),
        summary=(
            f"CERTIFIED 3-way root cause(s): {[c['service'] for c in candidates]}"
            if passed
            else "UNCERTIFIED: No component satisfied all three aligned signals."
        ),
    )
