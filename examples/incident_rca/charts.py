"""Deterministic Live Demo & Benchmark Proof Charts (`examples/incident_rca/charts.py`).

Global Technical Constraints:
- Headless & thread-safe: Matplotlib OO API (`Figure` + `FigureCanvasAgg`, `Agg` backend);
  never imports `matplotlib.pyplot`.
- Byte-deterministic: `fig.savefig(..., format="png", dpi=100, metadata={"Software": "ARES"})`
  produces identical PNG bytes and SHA-256 across repeated runs.
- Compact & readable: `figsize=(9.0, 5.5)` at `100 dpi` (~40–100 KB per PNG).
- Consistent ARES color palette across all charts.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from ares.contracts import InvestigationBoard

matplotlib.use("Agg")

# Consistent ARES Color Palette
COLOR_CERTIFIED = "#1b9e77"    # teal/green: CERTIFIED / True signal / ARES
COLOR_UNCERTIFIED = "#d95f02"  # orange: UNCERTIFIED / Warning / Decoy / Distractor
COLOR_UNKNOWN = "#7570b3"      # purple: UNKNOWN / VoI target
COLOR_THRESHOLD = "#e41a1c"    # red: Threshold / Failure / Baseline alert
COLOR_NEUTRAL = "#666666"      # gray: Neutral / Baseline bar


def _save_deterministic_png(fig: Figure, out_path: str | Path, *, dpi: int = 100) -> Path:
    """Render `fig` to deterministic PNG bytes and write to `out_path`."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, metadata={"Software": "ARES"})
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf.getvalue())
    return dest


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute 1-based average ranks (handling ties) using pure NumPy."""
    n = values.size
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Pearson correlation coefficient without SciPy."""
    if x.size < 2 or y.size < 2:
        return None
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denom <= 1e-12:
        return None
    return float(np.clip(np.sum(dx * dy) / denom, -1.0, 1.0))


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    """Compute Spearman rank correlation coefficient without SciPy."""
    if x.size < 2 or y.size < 2:
        return None
    return _pearson_r(_average_ranks(x), _average_ranks(y))


def _calibration_metrics(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int], float, float]:
    """Compute reliability bin stats, ECE, and Brier score over `n_bins` equal-width bins."""
    n = confidences.size
    if n == 0:
        return [], [], [], 0.0, 0.0

    confs = np.clip(confidences.astype(float), 0.0, 1.0)
    corrs = correctness.astype(float)
    brier = float(np.mean((confs - corrs) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0

    for idx in range(n_bins):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        if idx < n_bins - 1:
            mask = (confs >= lo) & (confs < hi)
        else:
            mask = (confs >= lo) & (confs <= hi)
        count = int(np.sum(mask))
        if count > 0:
            mean_c = float(np.mean(confs[mask]))
            mean_a = float(np.mean(corrs[mask]))
            bin_confs.append(mean_c)
            bin_accs.append(mean_a)
            bin_counts.append(count)
            ece += (count / n) * abs(mean_a - mean_c)

    return bin_confs, bin_accs, bin_counts, float(ece), brier


# ----------------------------------------------------------------------
# 2. Board Influence Chart
# ----------------------------------------------------------------------


def render_board_priority_chart(
    board: InvestigationBoard,
    out_path: str | Path,
) -> Path:
    """Render horizontal bar chart of active `board.fields` sorted descending by priority.

    Encoding:
    - Bar color = field `status` (`certified` green, `uncertified` orange, `unknown` purple).
    - Known fields plot `priority`; `unknown` fields plot `uncertainty` (with secondary marker if both).
    - Each bar is annotated with exact value and source note count.
    - Subtitle displays `board_version`, primary conclusion `answer`, `confidence`, and
      `certified_field_ratio`.
    """
    fig = Figure(figsize=(9.0, 5.5), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    conc = board.conclusion
    subtitle = (
        f"v{board.board_version}  |  Conclusion: {conc.answer} ({conc.confidence:.1%})  |  "
        f"Certified Field Ratio: {conc.certified_field_ratio:.1%}  |  "
        f"Verified: {conc.verified}"
    )
    fig.suptitle(
        "Investigation Board — Field Priority & Uncertainty",
        fontsize=11.5,
        fontweight="bold",
        y=0.97,
    )
    ax.set_title(subtitle, fontsize=8.5, color="#333333", pad=10)

    items = sorted(
        board.fields.items(),
        key=lambda kv: (-float(kv[1].priority), -float(kv[1].uncertainty), kv[0]),
    )

    if not items:
        ax.text(
            0.5,
            0.5,
            "No active fields on InvestigationBoard",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=COLOR_NEUTRAL,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
        return _save_deterministic_png(fig, out_path)

    status_color_map = {
        "certified": COLOR_CERTIFIED,
        "uncertified": COLOR_UNCERTIFIED,
        "unknown": COLOR_UNKNOWN,
    }

    labels: list[str] = []
    bar_values: list[float] = []
    bar_colors: list[str] = []
    annotations: list[str] = []
    secondary_markers: list[tuple[int, float]] = []

    for idx, (fname, field) in enumerate(items):
        labels.append(fname)
        color = status_color_map.get(field.status, COLOR_NEUTRAL)
        bar_colors.append(color)
        n_notes = len(field.sources)
        note_lbl = f"{n_notes} note" if n_notes == 1 else f"{n_notes} notes"

        if field.status == "unknown":
            val = float(field.uncertainty)
            bar_values.append(val)
            annotations.append(f"unc={val:.3f} ({note_lbl})")
            if field.priority > 0.0:
                secondary_markers.append((idx, float(field.priority)))
        else:
            val = float(field.priority)
            bar_values.append(val)
            annotations.append(f"pri={val:.3f} ({note_lbl})")
            if field.uncertainty > 0.0:
                secondary_markers.append((idx, float(field.uncertainty)))

    y_pos = np.arange(len(items))
    bars = ax.barh(y_pos, bar_values, color=bar_colors, height=0.62, alpha=0.9, edgecolor="white")

    # Hatch unknown bars lightly so uncertainty bars are visually distinct even in grayscale
    for bar, (_, field) in zip(bars, items):
        if field.status == "unknown":
            bar.set_hatch("//")
            bar.set_edgecolor("#ffffff")

    if secondary_markers:
        mx = [m[1] for m in secondary_markers]
        my = [m[0] for m in secondary_markers]
        ax.scatter(
            mx,
            my,
            marker="D",
            s=36,
            color=COLOR_UNKNOWN,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
            label="Secondary Uncertainty / Priority",
        )

    max_val = max(
        [0.1] + bar_values + [m[1] for m in secondary_markers]
    )
    x_limit = min(1.32, max(0.35, max_val * 1.38))
    ax.set_xlim(0.0, x_limit)

    for idx, (val, ann) in enumerate(zip(bar_values, annotations)):
        ax.text(
            val + x_limit * 0.015,
            idx,
            ann,
            va="center",
            ha="left",
            fontsize=8,
            color="#222222",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Field Priority (known fields) / Uncertainty (unknown fields)", fontsize=8.5)
    ax.grid(True, axis="x", linestyle=":", alpha=0.55)
    ax.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=COLOR_CERTIFIED, label="certified (priority)"),
        Patch(facecolor=COLOR_UNCERTIFIED, label="uncertified (priority)"),
        Patch(facecolor=COLOR_UNKNOWN, hatch="//", edgecolor="white", label="unknown (uncertainty)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.92)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return _save_deterministic_png(fig, out_path)


render_board_influence_chart = render_board_priority_chart


# ----------------------------------------------------------------------
# 3. Confidence Calibration / Reliability Diagram
# ----------------------------------------------------------------------


def render_calibration_plot(
    records: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Render a 10-bin reliability diagram with ECE and Brier score."""
    fig = Figure(figsize=(9.0, 5.5), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    # Perfect calibration diagonal
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color=COLOR_THRESHOLD,
        linewidth=1.2,
        label="Perfect calibration (y = x)",
        zorder=2,
    )

    grouped: dict[str, list[tuple[float, bool]]] = {}
    for rec in records:
        sys_name = str(rec.get("system", "ares")).strip().lower() or "ares"
        conf = float(rec.get("confidence", 0.0))
        corr = bool(rec.get("correct", False))
        grouped.setdefault(sys_name, []).append((conf, corr))

    system_styles: dict[str, dict[str, Any]] = {
        "ares": {
            "label": "ARES",
            "color": COLOR_CERTIFIED,
            "marker": "o",
            "linestyle": "-",
            "offset_y": 8,
        },
        "baseline": {
            "label": "Baseline (gemini-4-carbon SQL)",
            "color": COLOR_NEUTRAL,
            "marker": "s",
            "linestyle": "--",
            "offset_y": -13,
        },
    }

    subtitle_parts: list[str] = []
    ordered_systems = [s for s in ("ares", "baseline") if s in grouped] + [
        s for s in sorted(grouped) if s not in ("ares", "baseline")
    ]

    for sys_name in ordered_systems:
        pairs = grouped[sys_name]
        confs = np.array([p[0] for p in pairs], dtype=float)
        corrs = np.array([1.0 if p[1] else 0.0 for p in pairs], dtype=float)
        b_confs, b_accs, b_counts, ece, brier = _calibration_metrics(confs, corrs, n_bins=10)

        style = system_styles.get(
            sys_name,
            {
                "label": sys_name,
                "color": COLOR_UNCERTIFIED,
                "marker": "^",
                "linestyle": "-.",
                "offset_y": 8,
            },
        )
        ax.plot(
            b_confs,
            b_accs,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=6.5,
            label=f"{style['label']} (n={len(pairs)})",
            zorder=4,
        )
        for bx, by, cnt in zip(b_confs, b_accs, b_counts):
            ax.annotate(
                f"n={cnt}",
                xy=(bx, by),
                xytext=(0, style["offset_y"]),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=style["color"],
                fontweight="bold",
            )
        subtitle_parts.append(
            f"{sys_name.upper()}: ECE={ece:.4f}, Brier={brier:.4f} (n={len(pairs)})"
        )

    if not ordered_systems:
        ax.text(
            0.5,
            0.5,
            "No calibration records provided",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=COLOR_NEUTRAL,
        )

    fig.suptitle(
        "Confidence Calibration Reliability Diagram (10 Equal-Width Bins)",
        fontsize=11.5,
        fontweight="bold",
        y=0.97,
    )
    ax.set_title(
        "  |  ".join(subtitle_parts) if subtitle_parts else "No records",
        fontsize=8.5,
        color="#333333",
        pad=10,
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.set_yticks(np.linspace(0.0, 1.0, 11))
    ax.set_xlabel("Mean Predicted Confidence", fontsize=9)
    ax.set_ylabel("Empirical Accuracy", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return _save_deterministic_png(fig, out_path)


# ----------------------------------------------------------------------
# 4. Counterfactual Influence vs. Ground-Truth Info Value Plot
# ----------------------------------------------------------------------


def render_influence_vs_voi_plot(
    records: Sequence[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Render scatter plot of Counterfactual Influence vs. Ground-Truth Info Value."""
    fig = Figure(figsize=(9.0, 5.5), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    pivotal_x: list[float] = []
    pivotal_y: list[float] = []
    distractor_x: list[float] = []
    distractor_y: list[float] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for rec in records:
        inf = float(rec.get("priority", rec.get("influence", 0.0)))
        gain = float(rec.get("true_info_gain", 0.0))
        is_piv = bool(rec.get("is_pivotal", False))
        all_x.append(inf)
        all_y.append(gain)
        if is_piv:
            pivotal_x.append(inf)
            pivotal_y.append(gain)
        else:
            distractor_x.append(inf)
            distractor_y.append(gain)

    x_arr = np.array(all_x, dtype=float)
    y_arr = np.array(all_y, dtype=float)
    rho = _spearman_rho(x_arr, y_arr)
    pearson = _pearson_r(x_arr, y_arr)

    if pivotal_x:
        ax.scatter(
            pivotal_x,
            pivotal_y,
            color=COLOR_CERTIFIED,
            s=52,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
            label=f"Pivotal clue (n={len(pivotal_x)})",
            zorder=4,
        )
    if distractor_x:
        ax.scatter(
            distractor_x,
            distractor_y,
            color=COLOR_UNCERTIFIED,
            s=46,
            marker="^",
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            label=f"Distractor / non-pivotal (n={len(distractor_x)})",
            zorder=3,
        )

    if x_arr.size >= 2 and float(np.std(x_arr)) > 1e-12:
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color=COLOR_UNKNOWN,
            linestyle="--",
            linewidth=1.6,
            label=f"Linear fit (y = {slope:.2f}x + {intercept:.2f})",
            zorder=2,
        )

    if not records:
        ax.text(
            0.5,
            0.5,
            "No influence vs. ground-truth info gain records provided",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color=COLOR_NEUTRAL,
        )

    rho_str = f"{rho:.3f}" if rho is not None else "n/a"
    r_str = f"{pearson:.3f}" if pearson is not None else "n/a"
    fig.suptitle(
        "Board Field Priority vs. Ground-Truth Information Value",
        fontsize=11.5,
        fontweight="bold",
        y=0.97,
    )
    ax.set_title(
        f"Spearman ρ = {rho_str}   |   Pearson r = {r_str}   |   Total fields = {len(records)}",
        fontsize=8.8,
        color="#333333",
        pad=10,
    )
    ax.set_xlabel("Board Field Priority", fontsize=9)
    ax.set_ylabel("Ground-Truth Candidate-Set Reduction / Info Gain", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return _save_deterministic_png(fig, out_path)


# ----------------------------------------------------------------------
# 5. Benchmark Summary Comparison Chart
# ----------------------------------------------------------------------


def _extract_rate(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        val = metrics.get(k)
        if val is not None:
            fval = float(val)
            return fval * 100.0 if fval <= 1.0 else fval
    return default


def _extract_wall_clock_s(metrics: dict[str, Any]) -> float:
    for key in ("median_wall_clock_s", "mean_wall_clock_s", "wall_clock_s"):
        if metrics.get(key) is not None:
            return float(metrics[key])
    for key in ("median_wall_clock_ms", "mean_wall_clock_ms", "wall_clock_ms"):
        if metrics.get(key) is not None:
            return float(metrics[key]) / 1000.0
    return 0.0


def render_benchmark_summary_chart(
    ares_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    out_path: str | Path,
) -> Path:
    """Render 2-panel headline comparison (`ARES` vs. `gemini-4-carbon + DuckDB SQL` baseline).

    Left panel: Grouped bar chart of rates (0–100%) — Needle Accuracy, Decoy-Avoidance Rate,
      Honest UNCERTIFIED Rate, and Verifier Prune Rate (ARES only).
    Right panel: Wall-Clock Time (s) and Total Tokens / Model Calls comparison.
    """
    fig = Figure(figsize=(9.0, 5.5), dpi=100, facecolor="white")
    FigureCanvasAgg(fig)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.30, hspace=0.45)

    ax_rates = fig.add_subplot(gs[:, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_tokens = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "ARES vs. Carbon SQL Baseline — Benchmark Summary",
        fontsize=11.5,
        fontweight="bold",
        y=0.97,
    )

    # --- Left Panel: Rates (0–100%) ---
    ares_needle = _extract_rate(ares_metrics, "needle_accuracy", "root_cause_accuracy", "overall_pass_rate")
    base_needle = _extract_rate(baseline_metrics, "needle_accuracy", "root_cause_accuracy", "overall_pass_rate")

    ares_decoy = _extract_rate(ares_metrics, "decoy_avoidance_rate")
    base_decoy = _extract_rate(baseline_metrics, "decoy_avoidance_rate")

    ares_honest = _extract_rate(ares_metrics, "honest_uncertified_rate", "certification_accuracy")
    base_honest = _extract_rate(baseline_metrics, "honest_uncertified_rate", "certification_accuracy")

    if ares_metrics.get("verifier_prune_rate") is not None:
        ares_prune = _extract_rate(ares_metrics, "verifier_prune_rate")
    elif ares_metrics.get("mean_candidates_pruned") is not None:
        # 8 candidate (service, shard) slices in micro-bench; normalize if raw count provided
        ares_prune = min(100.0, (float(ares_metrics["mean_candidates_pruned"]) / 8.0) * 100.0)
    else:
        ares_prune = 0.0

    categories = [
        "Needle\nAccuracy",
        "Decoy\nAvoidance",
        "Honest\nUNCERTIFIED",
        "Verifier\nPrune Rate",
    ]
    ares_vals = [ares_needle, ares_decoy, ares_honest, ares_prune]
    base_vals: list[float | None] = [base_needle, base_decoy, base_honest, None]

    x = np.arange(len(categories))
    width = 0.35

    ares_bars = ax_rates.bar(
        x - width / 2,
        ares_vals,
        width=width,
        color=COLOR_CERTIFIED,
        label="ARES",
        alpha=0.92,
    )
    for idx, bval in enumerate(base_vals):
        if bval is not None:
            bar = ax_rates.bar(
                x[idx] + width / 2,
                bval,
                width=width,
                color=COLOR_NEUTRAL,
                label="Baseline" if idx == 0 else None,
                alpha=0.88,
            )
            ax_rates.text(
                x[idx] + width / 2,
                bval + 1.8,
                f"{bval:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )
        else:
            ax_rates.text(
                x[idx] + width / 2,
                3.0,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=COLOR_NEUTRAL,
                style="italic",
            )

    for bar, aval in zip(ares_bars, ares_vals):
        ax_rates.text(
            bar.get_x() + bar.get_width() / 2.0,
            aval + 1.8,
            f"{aval:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold",
            color=COLOR_CERTIFIED,
        )

    ax_rates.set_xticks(x)
    ax_rates.set_xticklabels(categories, fontsize=8)
    ax_rates.set_ylim(0.0, 122.0)
    ax_rates.set_ylabel("Rate (%)", fontsize=8.5)
    ax_rates.set_title("Accuracy, Honesty & Pruning Rates", fontsize=9.5, fontweight="bold")
    ax_rates.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax_rates.set_axisbelow(True)
    ax_rates.legend(loc="upper right", ncol=2, fontsize=8, framealpha=0.92)

    # --- Right Top Panel: Wall-Clock Time (s) ---
    ares_time = _extract_wall_clock_s(ares_metrics)
    base_time = _extract_wall_clock_s(baseline_metrics)
    t_labels = ["ARES", "Baseline"]
    t_vals = [ares_time, base_time]
    t_colors = [COLOR_CERTIFIED, COLOR_NEUTRAL]

    t_bars = ax_time.bar(t_labels, t_vals, width=0.5, color=t_colors, alpha=0.9)
    max_t = max(1.0, max(t_vals))
    ax_time.set_ylim(0.0, max_t * 1.28)
    for bar, val in zip(t_bars, t_vals):
        ax_time.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max_t * 0.04,
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#222222",
        )
    time_title = "Median Wall-Clock Time (s)" if "median_wall_clock_s" in ares_metrics else "Wall-Clock Time / Case (s)"
    ax_time.set_title(time_title, fontsize=9, fontweight="bold")
    ax_time.set_ylabel("Seconds", fontsize=8)
    ax_time.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax_time.set_axisbelow(True)

    # --- Right Bottom Panel: Total Tokens (k) & Model Calls ---
    ares_tokens_k = float(ares_metrics.get("total_tokens", 0)) / 1000.0
    base_tokens_k = float(baseline_metrics.get("total_tokens", 0)) / 1000.0
    ares_calls = int(ares_metrics.get("total_model_calls", 0))
    base_calls = int(baseline_metrics.get("total_model_calls", 0))

    tok_vals = [ares_tokens_k, base_tokens_k]
    tok_bars = ax_tokens.bar(t_labels, tok_vals, width=0.5, color=t_colors, alpha=0.9)
    max_tok = max(1.0, max(tok_vals))
    ax_tokens.set_ylim(0.0, max_tok * 1.32)
    for bar, val, calls in zip(tok_bars, tok_vals, (ares_calls, base_calls)):
        ax_tokens.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + max_tok * 0.04,
            f"{val:.0f}k tok\n({calls} calls)",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222222",
        )
    ax_tokens.set_title("Total Tokens & Model Calls", fontsize=9, fontweight="bold")
    ax_tokens.set_ylabel("Tokens (thousands)", fontsize=8)
    ax_tokens.grid(True, axis="y", linestyle=":", alpha=0.55)
    ax_tokens.set_axisbelow(True)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.11, wspace=0.30, hspace=0.48)
    return _save_deterministic_png(fig, out_path)
