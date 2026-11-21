"""Deterministic visual artifact generation for surviving candidate slices.

Implements the Person A verifier-side PNG plotting requirement from
`ARES_MULTIMODAL_ADDENDUM.md` (§1, §2 I4/I7, §4):
- Generates compact, deterministic 3-panel PNG plots directly from Parquet shards
  using DuckDB + Matplotlib Agg (no LLM calls, no global pyplot state).
- Computes SHA-256 over the rendered PNG bytes and returns an `ArtifactRef`
  stamped `produced_by_verifier=True`.
"""

from __future__ import annotations

import hashlib
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from contracts import ArtifactRef

matplotlib.use("Agg")


def _parquet_glob(data_dir: str, table: str) -> str:
    return str(Path(data_dir) / table / "*.parquet").replace("'", "''")


def _parse_ts(val: str | datetime) -> datetime:
    if isinstance(val, datetime):
        dt = val
    else:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def render_candidate_slice_plot(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    service: str,
    shard_id: str,
    out_dir: str,
    *,
    window_start: str | datetime | None = None,
    window_end: str | datetime | None = None,
    window_pad_seconds: int = 180,
) -> ArtifactRef:
    """Render a deterministic 3-panel PNG plot for a surviving (service, shard_id) slice.

    Panels:
      1. Upstream retry ERROR log counts (15s buckets, retry_count >= 3)
      2. Metric baseline z-score time series (with z = 4.0 threshold)
      3. Downstream ERROR trace span durations in ms (with 800ms threshold)
    """
    logs_glob = _parquet_glob(data_dir, "service_logs")
    metrics_glob = _parquet_glob(data_dir, "metrics")
    traces_glob = _parquet_glob(data_dir, "traces")

    where_window = ""
    win_params: list[Any] = []
    if window_start is not None and window_end is not None:
        ws = _parse_ts(window_start) - timedelta(seconds=window_pad_seconds)
        we = _parse_ts(window_end) + timedelta(seconds=window_pad_seconds)
        where_window = "AND ts BETWEEN ? AND ?"
        win_params = [ws, we]

    # 1. Upstream retry counts in 15-second buckets
    log_rows = con.execute(
        f"""
        SELECT
            time_bucket(INTERVAL '15 seconds', ts) AS bucket_ts,
            COUNT(*) AS retry_errors
        FROM read_parquet('{logs_glob}')
        WHERE downstream_service = ?
          AND shard_id = ?
          AND level = 'ERROR'
          AND retry_count >= 3
          {where_window}
        GROUP BY bucket_ts
        ORDER BY bucket_ts ASC
        """,
        [service, shard_id, *win_params],
    ).fetchall()

    # 2. Max metric z-score per bucket
    metric_rows = con.execute(
        f"""
        SELECT
            ts,
            MAX((value - baseline_mean) / NULLIF(baseline_std, 0)) AS max_z
        FROM read_parquet('{metrics_glob}')
        WHERE service = ?
          AND shard_id = ?
          AND baseline_std > 0
          {where_window}
        GROUP BY ts
        ORDER BY ts ASC
        """,
        [service, shard_id, *win_params],
    ).fetchall()

    # 3. Downstream trace error spans
    trace_rows = con.execute(
        f"""
        SELECT
            ts,
            duration_ms
        FROM read_parquet('{traces_glob}')
        WHERE service = ?
          AND shard_id = ?
          AND status = 'ERROR'
          {where_window}
        ORDER BY ts ASC
        """,
        [service, shard_id, *win_params],
    ).fetchall()

    # Build deterministic figure without global pyplot state
    fig = Figure(figsize=(9.0, 6.0), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 1, sharex=True)
    fig.suptitle(
        f"Candidate Slice Telemetry: {service} ({shard_id})",
        fontsize=11,
        fontweight="bold",
    )

    # Panel 1: Upstream retries
    ax0 = axes[0]
    if log_rows:
        xs = [_parse_ts(r[0]) for r in log_rows]
        ys = [int(r[1]) for r in log_rows]
        ax0.bar(xs, ys, width=timedelta(seconds=12), color="#d95f02", alpha=0.85)
    ax0.set_ylabel("Retry Logs\n(count / 15s)", fontsize=8)
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.set_title("1. Upstream ERROR Retry Spike (retry_count >= 3)", fontsize=9, loc="left")

    # Panel 2: Metric z-score
    ax1 = axes[1]
    if metric_rows:
        xs = [_parse_ts(r[0]) for r in metric_rows]
        ys = [float(r[1]) for r in metric_rows]
        ax1.plot(xs, ys, color="#1b9e77", linewidth=1.6, marker="o", markersize=2.5)
    ax1.axhline(4.0, color="#e41a1c", linestyle="--", linewidth=1.0, label="z = 4.0")
    ax1.set_ylabel("Max Z-Score", fontsize=8)
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.set_title("2. Metric Baseline Z-Score", fontsize=9, loc="left")

    # Panel 3: Trace error spans
    ax2 = axes[2]
    if trace_rows:
        xs = [_parse_ts(r[0]) for r in trace_rows]
        ys = [float(r[1]) for r in trace_rows]
        ax2.scatter(xs, ys, color="#7570b3", s=18, alpha=0.85)
    ax2.axhline(800.0, color="#e41a1c", linestyle="--", linewidth=1.0, label="800 ms")
    ax2.set_ylabel("Span Duration\n(ms)", fontsize=8)
    ax2.set_xlabel("UTC Timestamp", fontsize=8)
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.set_title("3. Downstream ERROR Trace Spans", fontsize=9, loc="left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    buf = io.BytesIO()
    # Fixed PNG metadata guarantees identical bytes & sha256 across repeated runs
    fig.savefig(buf, format="png", dpi=100, metadata={"Software": "ARES"})
    png_bytes = buf.getvalue()
    sha256 = hashlib.sha256(png_bytes).hexdigest()

    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_svc = service.replace("/", "_")
    safe_shard = shard_id.replace("/", "_")
    out_path = dest_dir / f"slice_{safe_svc}_{safe_shard}_{sha256[:10]}.png"
    out_path.write_bytes(png_bytes)

    return ArtifactRef(
        artifact_id=f"art_{sha256[:12]}",
        kind="plot_png",
        path=str(out_path),
        sha256=sha256,
        mime_type="image/png",
        produced_by_verifier=True,
    )


# Consistent palette across all ARES deterministic charts
COLOR_CERTIFIED = "#1b9e77"
COLOR_UNCERTIFIED = "#d95f02"
COLOR_UNKNOWN = "#7570b3"
COLOR_THRESHOLD = "#e41a1c"
COLOR_NEUTRAL = "#666666"


def _query_candidate_signals(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    service: str,
    shard_id: str,
    *,
    window_start: str | datetime | None = None,
    window_end: str | datetime | None = None,
    window_pad_seconds: int = 180,
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Query raw 3-signal telemetry for a single `(service, shard_id)` slice."""
    logs_glob = _parquet_glob(data_dir, "service_logs")
    metrics_glob = _parquet_glob(data_dir, "metrics")
    traces_glob = _parquet_glob(data_dir, "traces")

    where_window = ""
    win_params: list[Any] = []
    if window_start is not None and window_end is not None:
        ws = _parse_ts(window_start) - timedelta(seconds=window_pad_seconds)
        we = _parse_ts(window_end) + timedelta(seconds=window_pad_seconds)
        where_window = "AND ts BETWEEN ? AND ?"
        win_params = [ws, we]

    log_rows = con.execute(
        f"""
        SELECT
            time_bucket(INTERVAL '15 seconds', ts) AS bucket_ts,
            COUNT(*) AS retry_errors
        FROM read_parquet('{logs_glob}')
        WHERE downstream_service = ?
          AND shard_id = ?
          AND level = 'ERROR'
          AND retry_count >= 3
          {where_window}
        GROUP BY bucket_ts
        ORDER BY bucket_ts ASC
        """,
        [service, shard_id, *win_params],
    ).fetchall()

    metric_rows = con.execute(
        f"""
        SELECT
            ts,
            MAX((value - baseline_mean) / NULLIF(baseline_std, 0)) AS max_z
        FROM read_parquet('{metrics_glob}')
        WHERE service = ?
          AND shard_id = ?
          AND baseline_std > 0
          {where_window}
        GROUP BY ts
        ORDER BY ts ASC
        """,
        [service, shard_id, *win_params],
    ).fetchall()

    trace_rows = con.execute(
        f"""
        SELECT
            ts,
            duration_ms
        FROM read_parquet('{traces_glob}')
        WHERE service = ?
          AND shard_id = ?
          AND status = 'ERROR'
          {where_window}
        ORDER BY ts ASC
        """,
        [service, shard_id, *win_params],
    ).fetchall()

    return log_rows, metric_rows, trace_rows


def _deterministic_signal_status(
    log_rows: list[tuple[Any, Any]],
    metric_rows: list[tuple[Any, Any]],
    trace_rows: list[tuple[Any, Any]],
    *,
    min_retry_logs: int = 12,
    min_metric_buckets: int = 4,
    min_zscore: float = 4.0,
    min_trace_spans: int = 12,
    min_duration_ms: float = 800.0,
) -> tuple[bool, bool, bool]:
    """Compute deterministic pass/fail per signal using SQL verifier thresholds (Invariant I4)."""
    logs_pass = sum(int(r[1]) for r in log_rows) >= min_retry_logs
    metrics_pass = sum(1 for r in metric_rows if float(r[1]) >= min_zscore) >= min_metric_buckets
    traces_pass = sum(1 for r in trace_rows if float(r[1]) >= min_duration_ms) >= min_trace_spans
    return logs_pass, metrics_pass, traces_pass


def render_candidate_comparison_plot(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    candidates: list[tuple[str, str]],
    out_dir: str,
    *,
    window_start: str | datetime | None = None,
    window_end: str | datetime | None = None,
    window_pad_seconds: int = 180,
) -> ArtifactRef:
    """Render a deterministic 3xN side-by-side comparison PNG for up to 3 candidate slices.

    Rows (shared y-axis per row):
      1. Signal 1: Upstream ERROR retry spike count / 15s (retry_count >= 3)
      2. Signal 2: Metric max z-score with dashed z = 4.0 threshold line
      3. Signal 3: Downstream ERROR trace duration in ms with dashed 800 ms threshold line

    Column headers show `service (shard_id)` plus a deterministic pass/fail badge
    strictly derived from SQL thresholds (never LLM output, preserving Invariant I4).
    """
    if not candidates:
        raise ValueError("candidates must contain at least one (service, shard_id) pair.")
    selected = list(candidates)[:3]
    ncols = len(selected)

    per_candidate_data: list[
        tuple[str, str, list[tuple[Any, Any]], list[tuple[Any, Any]], list[tuple[Any, Any]], tuple[bool, bool, bool]]
    ] = []
    for svc, shard in selected:
        log_rows, metric_rows, trace_rows = _query_candidate_signals(
            con,
            data_dir,
            svc,
            shard,
            window_start=window_start,
            window_end=window_end,
            window_pad_seconds=window_pad_seconds,
        )
        flags = _deterministic_signal_status(log_rows, metric_rows, trace_rows)
        per_candidate_data.append((svc, shard, log_rows, metric_rows, trace_rows, flags))

    fig = Figure(figsize=(9.0, 6.0), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, ncols, sharey="row", sharex="col", squeeze=False)
    fig.suptitle(
        "Multimodal Candidate Comparison (Deterministic Verifier Evidence)",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )

    for col_idx, (svc, shard, log_rows, metric_rows, trace_rows, flags) in enumerate(
        per_candidate_data
    ):
        logs_ok, metrics_ok, traces_ok = flags
        passed_count = int(logs_ok) + int(metrics_ok) + int(traces_ok)
        if passed_count == 3:
            badge_text = "3/3 PASS"
            badge_color = COLOR_CERTIFIED
            sig_color = COLOR_CERTIFIED
        elif passed_count > 0:
            badge_text = f"{passed_count}/3 PARTIAL"
            badge_color = COLOR_UNCERTIFIED
            sig_color = COLOR_UNCERTIFIED
        else:
            badge_text = "0/3 NONE"
            badge_color = COLOR_NEUTRAL
            sig_color = COLOR_NEUTRAL

        sig_summary = (
            f"L:{'OK' if logs_ok else 'NO'} "
            f"M:{'OK' if metrics_ok else 'NO'} "
            f"T:{'OK' if traces_ok else 'NO'}"
        )

        # Row 0: Signal 1 — Upstream ERROR retry spike count / 15s
        ax0 = axes[0, col_idx]
        if log_rows:
            xs = [_parse_ts(r[0]) for r in log_rows]
            ys = [int(r[1]) for r in log_rows]
            ax0.bar(
                xs,
                ys,
                width=timedelta(seconds=12),
                color=COLOR_CERTIFIED if logs_ok else COLOR_UNCERTIFIED,
                alpha=0.85,
            )
        else:
            ax0.text(
                0.5,
                0.5,
                "No retry spikes",
                transform=ax0.transAxes,
                ha="center",
                va="center",
                fontsize=7.5,
                color=COLOR_NEUTRAL,
            )
        ax0.set_title(
            f"{svc} ({shard})\n{badge_text}  [{sig_summary}]",
            fontsize=8.5,
            fontweight="bold",
            color=badge_color,
            pad=6,
        )
        ax0.grid(True, linestyle=":", alpha=0.5)
        if col_idx == 0:
            ax0.set_ylabel("Retry Logs\n(count / 15s)", fontsize=8)

        # Row 1: Signal 2 — Metric max z-score
        ax1 = axes[1, col_idx]
        if metric_rows:
            xs = [_parse_ts(r[0]) for r in metric_rows]
            ys = [float(r[1]) for r in metric_rows]
            ax1.plot(
                xs,
                ys,
                color=COLOR_CERTIFIED if metrics_ok else sig_color,
                linewidth=1.5,
                marker="o",
                markersize=2.2,
            )
        else:
            ax1.text(
                0.5,
                0.5,
                "No metric rows",
                transform=ax1.transAxes,
                ha="center",
                va="center",
                fontsize=7.5,
                color=COLOR_NEUTRAL,
            )
        ax1.axhline(4.0, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.0, label="z = 4.0")
        ax1.grid(True, linestyle=":", alpha=0.5)
        if col_idx == 0:
            ax1.set_ylabel("Metric Max\nZ-Score", fontsize=8)
            ax1.legend(loc="upper right", fontsize=6.5)

        # Row 2: Signal 3 — Downstream ERROR trace duration
        ax2 = axes[2, col_idx]
        if trace_rows:
            xs = [_parse_ts(r[0]) for r in trace_rows]
            ys = [float(r[1]) for r in trace_rows]
            ax2.scatter(
                xs,
                ys,
                color=COLOR_CERTIFIED if traces_ok else COLOR_UNKNOWN,
                s=16,
                alpha=0.85,
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "No ERROR traces",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=7.5,
                color=COLOR_NEUTRAL,
            )
        ax2.axhline(800.0, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.0, label="800 ms")
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.set_xlabel("UTC Time", fontsize=7.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.tick_params(axis="x", labelsize=7, rotation=20)
        if col_idx == 0:
            ax2.set_ylabel("ERROR Trace\nDuration (ms)", fontsize=8)
            ax2.legend(loc="upper right", fontsize=6.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, metadata={"Software": "ARES"})
    png_bytes = buf.getvalue()
    sha256 = hashlib.sha256(png_bytes).hexdigest()

    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = "_vs_".join(
        f"{svc.replace('/', '_')}-{shard.replace('/', '_')}" for svc, shard in selected
    )
    out_path = dest_dir / f"compare_{slug}_{sha256[:10]}.png"
    out_path.write_bytes(png_bytes)

    return ArtifactRef(
        artifact_id=f"art_{sha256[:12]}",
        kind="plot_png",
        path=str(out_path),
        sha256=sha256,
        mime_type="image/png",
        produced_by_verifier=True,
    )
