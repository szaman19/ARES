"""Coarse-to-fine DuckDB anomaly search proving 2-level dynamic Spawn + then."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

import duckdb

from ares import Done, Spawn, StateMachine, Task, TaskContext, TaskSpec

CaseType = Literal["true_needle", "decoy_only", "under_determined"]

TOP_K = 3
ERROR_SPIKE_THRESHOLD = 15
LATENCY_DRIFT_P99_MS = 250.0


def create_anomaly_dataset(
    db_path: str | Path,
    *,
    case_type: CaseType = "true_needle",
    seed: int = 7,
) -> Path:
    """Create a local DuckDB database with 6 bucketed telemetry tables.

    Tables created: ``events``, ``service_metrics``, and ``trace_checks``.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    rng = random.Random(seed)
    events_rows: list[tuple[str, str, int, int]] = []
    metric_rows: list[tuple[str, str, float, float]] = []
    trace_rows: list[tuple[str, str, bool | None, int]] = []

    def add_entity(
        bucket: str, cid: str, err: int, total: int, p99: float, base: float, lock: bool | None, samples: int
    ) -> None:
        events_rows.append((bucket, cid, err, total))
        metric_rows.append((bucket, cid, p99, base))
        trace_rows.append((bucket, cid, lock, samples))

    for b_idx in range(6):
        bucket = f"bucket_{b_idx}"
        for s_idx in range(5):
            base = round(rng.uniform(35.0, 55.0), 2)
            p99 = round(base + rng.uniform(5.0, 40.0), 2)
            add_entity(bucket, f"svc-{b_idx}-{s_idx}", rng.randint(0, 5), rng.randint(400, 900), p99, base, False, rng.randint(5, 20))

    # Moderate 1-of-3 noise candidate to exercise top-K pruning and verification.
    add_entity("bucket_2", "auth-edge-01", 18, 850, 118.0, 48.0, False, 15)
    # Loud 2-of-3 decoy: high error + latency spikes, fails downstream_lock_confirmed.
    add_entity("bucket_1", "cache-redis-03", 145, 1000, 820.0, 45.0, False, 40)

    if case_type == "true_needle":
        # Subtle 3-of-3 true positive: lower coarse score than decoy, passes all 3 checks.
        add_entity("bucket_4", "payments-db-02", 26, 800, 340.0, 50.0, True, 35)
    elif case_type == "under_determined":
        # 2 signals exist, but the 3rd structural check is missing (NULL).
        add_entity("bucket_4", "payments-db-02", 26, 800, 340.0, 50.0, None, 0)

    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE events (bucket VARCHAR, candidate_id VARCHAR, error_count INTEGER, total_events INTEGER)")
        conn.execute("CREATE TABLE service_metrics (bucket VARCHAR, candidate_id VARCHAR, p99_ms DOUBLE, baseline_p99_ms DOUBLE)")
        conn.execute("CREATE TABLE trace_checks (bucket VARCHAR, candidate_id VARCHAR, downstream_lock_confirmed BOOLEAN, sample_count INTEGER)")
        conn.executemany("INSERT INTO events VALUES (?, ?, ?, ?)", events_rows)
        conn.executemany("INSERT INTO service_metrics VALUES (?, ?, ?, ?)", metric_rows)
        conn.executemany("INSERT INTO trace_checks VALUES (?, ?, ?, ?)", trace_rows)

    return path


def build_machine(db_path: str | Path) -> StateMachine:
    """Build a 2-level coarse-to-fine StateMachine backed by DuckDB."""
    default_db = str(db_path)
    machine = StateMachine()

    @machine.state("coarse_scan")
    def coarse_scan(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn:
        _ = deps, ctx
        db = str(task.input.get("db_path", default_db))
        with duckdb.connect(db, read_only=True) as conn:
            buckets = [r[0] for r in conn.execute("SELECT DISTINCT bucket FROM events ORDER BY bucket").fetchall()]
        children = [TaskSpec(state="probe_bucket", input={"bucket": b, "db_path": db}) for b in buckets]
        return Spawn(tasks=children, then=TaskSpec(state="pick_top_candidates", input={"db_path": db}))

    @machine.state("probe_bucket", deterministic=True)
    def probe_bucket(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = deps, ctx
        bucket = str(task.input["bucket"])
        db = str(task.input.get("db_path", default_db))
        query = """
            SELECT e.candidate_id, e.bucket,
                   ROUND(e.error_count + (m.p99_ms / m.baseline_p99_ms) * 10.0, 3) AS score
            FROM events e
            JOIN service_metrics m ON e.bucket = m.bucket AND e.candidate_id = m.candidate_id
            WHERE e.bucket = ?
            ORDER BY score DESC, e.candidate_id ASC
            LIMIT 2
        """
        with duckdb.connect(db, read_only=True) as conn:
            rows = conn.execute(query, [bucket]).fetchall()
            total_in_bucket = conn.execute("SELECT COUNT(*) FROM events WHERE bucket = ?", [bucket]).fetchone()[0]

        candidates = [{"candidate_id": r[0], "bucket": r[1], "score": float(r[2])} for r in rows]
        return Done(output={"bucket": bucket, "scanned_entities": int(total_in_bucket), "candidates": candidates})

    @machine.state("pick_top_candidates")
    def pick_top_candidates(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn:
        _ = ctx
        db = str(task.input.get("db_path", default_db))
        pooled: list[dict[str, Any]] = []
        total_scanned = 0
        for bucket_result in deps.values():
            total_scanned += int(bucket_result.get("scanned_entities", 0))
            pooled.extend(bucket_result.get("candidates", []))

        pooled.sort(key=lambda item: (-float(item["score"]), item["candidate_id"]))
        top_cands = pooled[:TOP_K]
        children = [
            TaskSpec(
                state="verify_candidate",
                input={"candidate_id": c["candidate_id"], "bucket": c["bucket"], "coarse_score": c["score"], "db_path": db},
            )
            for c in top_cands
        ]
        return Spawn(
            tasks=children,
            then=TaskSpec(state="finalize", input={"total_scanned": total_scanned, "round1_survivors": len(top_cands)}),
        )

    @machine.state("verify_candidate", deterministic=True)
    def verify_candidate(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = deps
        cid = str(task.input["candidate_id"])
        db = str(task.input.get("db_path", default_db))
        with duckdb.connect(db, read_only=True) as conn:
            err_row = conn.execute("SELECT bucket, error_count FROM events WHERE candidate_id = ?", [cid]).fetchone()
            met_row = conn.execute("SELECT p99_ms FROM service_metrics WHERE candidate_id = ?", [cid]).fetchone()
            trc_row = conn.execute("SELECT downstream_lock_confirmed FROM trace_checks WHERE candidate_id = ?", [cid]).fetchone()

        bucket = err_row[0] if err_row else task.input.get("bucket")
        error_count = int(err_row[1]) if err_row else 0
        p99_ms = float(met_row[0]) if met_row else 0.0
        lock_confirmed: bool | None = trc_row[0] if trc_row else None

        signals = {
            "signal_error_spike": error_count >= ERROR_SPIKE_THRESHOLD,
            "signal_latency_drift": p99_ms >= LATENCY_DRIFT_P99_MS,
            "signal_lock_trace": lock_confirmed,
        }
        all_3 = all(signals[k] is True for k in ("signal_error_spike", "signal_latency_drift", "signal_lock_trace"))
        receipt = {
            "candidate_id": cid,
            "bucket": bucket,
            "passed": all_3,
            "signals": signals,
            "metrics": {"error_count": error_count, "p99_ms": p99_ms, "downstream_lock_confirmed": lock_confirmed},
        }
        ref = ctx.store.put_json(receipt, name=f"receipt_{cid}.json")
        return Done(output={"candidate_id": cid, "bucket": bucket, "passed": all_3, "signals": signals, "receipt_ref": ref.ref_id})

    @machine.state("finalize")
    def finalize(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = ctx
        results = sorted(deps.values(), key=lambda r: r["candidate_id"])
        passed = [r for r in results if r["passed"]]
        rejected = [r["candidate_id"] for r in results if not r["passed"]]
        summary = {
            "total_scanned": task.input.get("total_scanned", 0),
            "round1_survivors": task.input.get("round1_survivors", len(results)),
            "rejected_decoys": rejected,
            "checked_candidates": results,
        }
        if len(passed) == 1:
            winner = passed[0]
            return Done(output={"status": "verified", "target": winner["candidate_id"], "receipt_ref": winner["receipt_ref"], "signals": winner["signals"], **summary})
        return Done(output={"status": "unverified", "target": None, **summary})

    return machine
