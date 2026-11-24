"""Dark-Mode 2x2 NASA TESS Observatory Diagnostic Renderer (`examples/exoplanet_triage/artifacts.py`).

Generates deterministic, SHA-256-pinned multimodal PNG cards directly from real
NASA TESS Parquet shards (`lightcurves`, `stellar_catalog`, `pixel_stamps`) using
thread-safe `FigureCanvasAgg` (zero LLM calls, zero global `pyplot` state).

Visual layout:
  1. Top-Left: Phase-Folded TESS 2-Minute Light Curve (ppm relative flux)
  2. Top-Right: Real 11x11 NASA CCD Difference Stamp + Aperture + Centroid Shift Vector
  3. Bottom-Left: Odd vs. Even Transit Depth Overlay & Secondary Occultation Check
  4. Bottom-Right: Kepler's 3rd Law Stellar Density Gauge (Transit vs. Gaia/TIC Host)
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from ares.contracts import ArtifactRef

matplotlib.use("Agg")

# Material Design 3 Dark Observatory Palette
BG_DARK = "#191C22"
SURFACE = "#22262F"
BORDER = "#3B4250"
TEXT_MAIN = "#E3E6ED"
TEXT_MUTED = "#9AA2B2"
CYAN = "#A8C7FA"
GREEN_CERT = "#8ED5B9"
AMBER_WARN = "#E5B586"
RED_ALERT = "#F2B8B5"
VIOLET = "#CBC3E8"
GOLD = "#E8C58E"


def _parquet_glob(data_dir: str, table: str) -> str:
    return str(Path(data_dir) / table / "*.parquet").replace("'", "''")


def _style_dark_axis(ax: Any, title: str) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(1.1)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.set_title(title, fontsize=9.5, fontweight="bold", color=TEXT_MAIN, loc="left", pad=6)


def _bin_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    n_bins: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bx: list[float] = []
    by: list[float] = []
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if int(np.sum(mask)) >= 2:
            bx.append(float(centers[i]))
            by.append(float(np.nanmedian(y[mask])))
    return np.array(bx, dtype=np.float64), np.array(by, dtype=np.float64)


def render_exoplanet_diagnostic_card(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    tic_id: str,
    out_dir: str,
    *,
    mask_ccd_panel: bool = False,
) -> ArtifactRef:
    """Render a deterministic 2x2 dark-mode NASA TESS diagnostic PNG for `tic_id`."""
    lc_glob = _parquet_glob(data_dir, "lightcurves")
    cat_glob = _parquet_glob(data_dir, "stellar_catalog")
    pix_glob = _parquet_glob(data_dir, "pixel_stamps")

    star_row = con.execute(
        f"""
        SELECT
            tic_id,
            sector,
            tmag,
            stellar_radius_rsun,
            stellar_mass_msun,
            stellar_density_gcc,
            teff_k,
            crowding_metric,
            candidate_period_days,
            candidate_t0_btjd,
            candidate_duration_hours
        FROM read_parquet('{cat_glob}')
        WHERE tic_id = ?
        LIMIT 1
        """,
        [tic_id],
    ).fetchone()

    if star_row is None:
        raise ValueError(f"tic_id '{tic_id}' not found in stellar_catalog")

    (
        _tic,
        sector,
        tmag,
        r_star,
        m_star,
        rho_gaia_gcc,
        teff_k,
        crowding,
        period_d,
        t0_btjd,
        dur_h,
    ) = star_row

    period_d = float(period_d or 3.0)
    t0_btjd = float(t0_btjd or 0.0)
    dur_h = max(0.5, float(dur_h or 2.5))
    rho_gaia_gcc = max(1e-3, float(rho_gaia_gcc or 1.41))
    crowding = float(crowding if crowding is not None else 1.0)

    lc_rows = con.execute(
        f"""
        SELECT btjd, rel_flux_ppm, flux_err_ppm
        FROM read_parquet('{lc_glob}')
        WHERE tic_id = ?
          AND quality_flag = 0
          AND rel_flux_ppm IS NOT NULL
        ORDER BY btjd ASC
        """,
        [tic_id],
    ).fetchall()

    pix_rows = con.execute(
        f"""
        SELECT
            row_idx,
            col_idx,
            mean_out_flux,
            diff_flux,
            diff_snr,
            aperture_mask,
            target_row,
            target_col
        FROM read_parquet('{pix_glob}')
        WHERE tic_id = ?
        ORDER BY row_idx ASC, col_idx ASC
        """,
        [tic_id],
    ).fetchall()

    btjd = np.array([float(r[0]) for r in lc_rows], dtype=np.float64)
    flux_ppm = np.array([float(r[1]) for r in lc_rows], dtype=np.float64)
    err_ppm = np.array([max(10.0, float(r[2] or 200.0)) for r in lc_rows], dtype=np.float64)

    # Phase folding around candidate_t0_btjd
    if btjd.size > 0:
        epoch_float = (btjd - t0_btjd) / period_d
        epoch_int = np.round(epoch_float).astype(np.int64)
        phase_days = (btjd - t0_btjd) - epoch_int * period_d
        phase_hours = phase_days * 24.0
        is_odd = (np.abs(epoch_int) % 2) == 1
    else:
        phase_hours = np.zeros(0, dtype=np.float64)
        is_odd = np.zeros(0, dtype=bool)

    half_dur = 0.5 * dur_h
    in_tr = np.abs(phase_hours) <= (0.85 * half_dur)
    out_tr = (np.abs(phase_hours) >= 1.5 * half_dur) & (np.abs(phase_hours) <= 4.0 * half_dur)

    if int(np.sum(in_tr)) >= 3 and int(np.sum(out_tr)) >= 5:
        base_med = float(np.median(flux_ppm[out_tr]))
        in_med = float(np.median(flux_ppm[in_tr]))
        depth_ppm = max(0.0, base_med - in_med)
        scatter = float(np.median(err_ppm[in_tr])) / math.sqrt(max(1, int(np.sum(in_tr))))
        snr_val = depth_ppm / max(15.0, scatter)
    else:
        depth_ppm = 0.0
        snr_val = 0.0

    # Odd vs Even depth
    odd_in = in_tr & is_odd
    even_in = in_tr & (~is_odd)
    odd_depth = float(-np.median(flux_ppm[odd_in])) if int(np.sum(odd_in)) >= 2 else depth_ppm
    even_depth = float(-np.median(flux_ppm[even_in])) if int(np.sum(even_in)) >= 2 else depth_ppm
    oe_err = max(40.0, float(np.median(err_ppm)) / math.sqrt(max(1, min(int(np.sum(odd_in)), int(np.sum(even_in))))))
    oe_sigma = abs(odd_depth - even_depth) / oe_err if (int(np.sum(odd_in)) >= 2 and int(np.sum(even_in)) >= 2) else 0.0

    # Secondary eclipse check near phase 0.5 * period_d
    if btjd.size > 0:
        sec_phase_h = (((btjd - (t0_btjd + 0.5 * period_d) + 0.5 * period_d) % period_d) - 0.5 * period_d) * 24.0
        sec_in = np.abs(sec_phase_h) <= (0.85 * half_dur)
        sec_out = (np.abs(sec_phase_h) >= 1.5 * half_dur) & (np.abs(sec_phase_h) <= 4.0 * half_dur)
        if int(np.sum(sec_in)) >= 3 and int(np.sum(sec_out)) >= 5:
            sec_depth = max(0.0, float(np.median(flux_ppm[sec_out]) - np.median(flux_ppm[sec_in])))
            sec_err = float(np.median(err_ppm[sec_in])) / math.sqrt(max(1, int(np.sum(sec_in))))
            sec_snr = sec_depth / max(20.0, sec_err)
        else:
            sec_snr = 0.0
    else:
        sec_snr = 0.0

    # Kepler's 3rd Law transit-implied stellar density (circular approximation, g/cm^3):
    # rho_star [g/cm^3] ~= 1.408 * (P_days / (pi * T_dur_days))^3 * (24 / 24) -> exact Seager & Mallen-Ornelas:
    # rho_star / rho_sun = ((2 * sqrt(k)) / sqrt((T_tot/P)^2 - (T_full/P)^2))^3 -> circular b=0 limit:
    # rho_transit_gcc = 1.408 * (1.806 * P_days^(1/3) / dur_hours)^3
    k_rp_rs = math.sqrt(min(0.25, max(1e-5, depth_ppm * 1e-6)))
    sin_term = math.sin(min(math.pi / 2.1, math.pi * (dur_h / 24.0) / max(0.2, period_d)))
    a_over_rstar = max(1.2, (1.0 + k_rp_rs) / max(1e-4, sin_term))
    # 3 * pi / (G * P^2) * (a/R*)^3 in g/cm^3
    p_sec = period_d * 86400.0
    rho_transit_gcc = (3.0 * math.pi / (6.6743e-8 * p_sec * p_sec)) * (a_over_rstar ** 3)
    rho_ratio = rho_transit_gcc / max(1e-4, rho_gaia_gcc)

    # Build Figure (Thread-safe FigureCanvasAgg)
    fig = Figure(figsize=(10.4, 7.0), dpi=105, facecolor=BG_DARK)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)

    fig.suptitle(
        f"NASA TESS Cross-Modal Vetting Card  ·  {tic_id} (Sector {sector}, Tmag={float(tmag):.2f})",
        fontsize=12.0,
        fontweight="bold",
        color=TEXT_MAIN,
        y=0.975,
    )

    # ------------------------------------------------------------------
    # Panel 1 (Top-Left): Phase-Folded TESS Light Curve
    # ------------------------------------------------------------------
    ax00 = axes[0, 0]
    _style_dark_axis(ax00, "1 · Phase-Folded TESS Photometry (2-Min Cadence)")
    win_h = max(3.5, min(10.0, 2.8 * dur_h))
    if phase_hours.size > 0:
        vis = np.abs(phase_hours) <= win_h
        ax00.scatter(
            phase_hours[vis],
            flux_ppm[vis],
            s=6,
            color=TEXT_MUTED,
            alpha=0.30,
            edgecolors="none",
            label="2-min cadence",
        )
        bx, by = _bin_curve(phase_hours[vis], flux_ppm[vis], -win_h, win_h, n_bins=30)
        if bx.size > 0:
            ax00.plot(bx, by, color=CYAN, linewidth=2.2, marker="o", markersize=3.5, label="Phase-binned")
    ax00.axhline(0.0, color=BORDER, linestyle="--", linewidth=1.0)
    ax00.axvspan(-half_dur, half_dur, color=CYAN, alpha=0.08)
    ax00.set_xlim(-win_h, win_h)
    if phase_hours.size > 0 and np.any(np.abs(phase_hours) <= win_h):
        vis_vals = flux_ppm[np.abs(phase_hours) <= win_h]
        y_lo = float(np.nanpercentile(vis_vals, 1.0))
        y_hi = float(np.nanpercentile(vis_vals, 99.0))
        y_span = max(800.0, y_hi - y_lo)
        ax00.set_ylim(y_lo - 0.28 * y_span, y_hi + 0.14 * y_span)
    ax00.set_xlabel("Phase from Mid-Transit (hours)", fontsize=8.5, color=TEXT_MUTED)
    ax00.set_ylabel("Relative Flux (ppm)", fontsize=8.5, color=TEXT_MUTED)
    ax00.grid(True, linestyle=":", color=BORDER, alpha=0.55)
    ax00.text(
        0.50,
        0.045,
        f"Depth = {depth_ppm:,.0f} ppm  ·  SNR = {snr_val:.1f}σ  ·  P = {period_d:.3f} d",
        transform=ax00.transAxes,
        ha="center",
        fontsize=7.8,
        fontweight="bold",
        color=CYAN,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": BG_DARK, "edgecolor": CYAN, "alpha": 0.92},
    )

    # ------------------------------------------------------------------
    # Panel 2 (Top-Right): Real 11x11 NASA CCD Difference Stamp
    # ------------------------------------------------------------------
    ax01 = axes[0, 1]
    _style_dark_axis(ax01, "2 · CCD Difference Stamp (Out − In)")

    if mask_ccd_panel:
        ax01.set_xticks([])
        ax01.set_yticks([])
        ax01.text(
            0.5,
            0.5,
            "ABLATED TO 'unknown'\n(Counterfactual Visual Reliance Probe)",
            transform=ax01.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=GOLD,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": BG_DARK, "edgecolor": GOLD, "linewidth": 1.6},
        )
    elif pix_rows:
        max_r = max(int(r[0]) for r in pix_rows) + 1
        max_c = max(int(r[1]) for r in pix_rows) + 1
        diff_img = np.zeros((max_r, max_c), dtype=np.float64)
        ap_mask = np.zeros((max_r, max_c), dtype=np.int32)
        t_row = float(pix_rows[0][6])
        t_col = float(pix_rows[0][7])

        for r_idx, c_idx, _mout, _dflux, dsnr, amask, _tr, _tc in pix_rows:
            diff_img[int(r_idx), int(c_idx)] = float(dsnr or 0.0)
            ap_mask[int(r_idx), int(c_idx)] = int(amask or 0)

        # Positive transit loss centroid weighted by positive diff_snr^2 for sub-pixel localization
        pos_weights = np.maximum(0.0, diff_img) ** 2
        w_sum = float(np.sum(pos_weights))
        if w_sum > 0:
            rr, cc = np.indices(diff_img.shape)
            cent_r = float(np.sum(rr * pos_weights) / w_sum)
            cent_c = float(np.sum(cc * pos_weights) / w_sum)
        else:
            cent_r, cent_c = t_row, t_col

        offset_px = math.hypot(cent_r - t_row, cent_c - t_col)
        peak_snr = float(np.max(diff_img)) if diff_img.size else 0.0

        ax01.imshow(diff_img, origin="lower", cmap="magma", interpolation="nearest", aspect="equal")

        # Draw SPOC optimal aperture pixel outlines (bit 1 -> & 2)
        for r_i in range(max_r):
            for c_i in range(max_c):
                if (ap_mask[r_i, c_i] & 2) != 0:
                    ax01.add_patch(
                        Rectangle(
                            (c_i - 0.5, r_i - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor=CYAN,
                            linewidth=1.1,
                            linestyle="--",
                            alpha=0.85,
                        )
                    )

        # Target star crosshair (+)
        ax01.plot(t_col, t_row, marker="+", markersize=13, markeredgewidth=2.2, color=CYAN, label="Target Star")

        # Determine spatial status badge
        if crowding < 0.80 or peak_snr < 5.5:
            badge_col = AMBER_WARN
            badge_txt = f"AMBIGUOUS · CROWDSAP={crowding:.2f} · Δr={offset_px:.2f}px"
            ax01.add_patch(Circle((cent_c, cent_r), radius=0.55, fill=False, edgecolor=AMBER_WARN, linewidth=2.0))
        elif offset_px <= 0.45:
            badge_col = GREEN_CERT
            badge_txt = f"TARGET LOCKED · Δr={offset_px:.2f}px · CROWDSAP={crowding:.2f}"
            ax01.add_patch(Circle((cent_c, cent_r), radius=0.48, fill=False, edgecolor=GREEN_CERT, linewidth=2.2))
        else:
            badge_col = RED_ALERT
            badge_txt = f"OFF-TARGET IMPOSTOR · Δr={offset_px:.2f}px (>0.45)"
            ax01.add_patch(Circle((cent_c, cent_r), radius=0.55, fill=False, edgecolor=RED_ALERT, linewidth=2.3))
            ax01.add_patch(
                FancyArrowPatch(
                    (t_col, t_row),
                    (cent_c, cent_r),
                    arrowstyle="->",
                    mutation_scale=14,
                    color=RED_ALERT,
                    linewidth=2.3,
                )
            )

        ax01.set_xlabel("CCD Column (21\"/px)", fontsize=8.5, color=TEXT_MUTED)
        ax01.set_ylabel("CCD Row (21\"/px)", fontsize=8.5, color=TEXT_MUTED)
        ax01.text(
            0.50,
            0.045,
            badge_txt,
            transform=ax01.transAxes,
            ha="center",
            fontsize=7.3,
            fontweight="bold",
            color=badge_col,
            bbox={"boxstyle": "round,pad=0.26", "facecolor": BG_DARK, "edgecolor": badge_col, "alpha": 0.92},
        )
    else:
        ax01.text(0.5, 0.5, "No TPF Pixel Stamp Available", transform=ax01.transAxes, ha="center", color=AMBER_WARN)

    # ------------------------------------------------------------------
    # Panel 3 (Bottom-Left): Odd vs. Even Transit Depth Overlay
    # ------------------------------------------------------------------
    ax10 = axes[1, 0]
    _style_dark_axis(ax10, "3 · Odd vs. Even Eclipse Symmetry & Phase 0.5 Check")
    if phase_hours.size > 0:
        vis_oe = np.abs(phase_hours) <= (2.2 * dur_h)
        bx_odd, by_odd = _bin_curve(
            phase_hours[vis_oe & is_odd],
            flux_ppm[vis_oe & is_odd],
            -2.2 * dur_h,
            2.2 * dur_h,
            n_bins=20,
        )
        bx_evn, by_evn = _bin_curve(
            phase_hours[vis_oe & (~is_odd)],
            flux_ppm[vis_oe & (~is_odd)],
            -2.2 * dur_h,
            2.2 * dur_h,
            n_bins=20,
        )
        if bx_odd.size > 0:
            ax10.plot(bx_odd, by_odd, color=CYAN, linewidth=2.0, marker="o", markersize=3.5, label=f"Odd ({odd_depth:,.0f} ppm)")
        if bx_evn.size > 0:
            ax10.plot(bx_evn, by_evn, color=AMBER_WARN, linewidth=2.0, linestyle="--", marker="s", markersize=3.5, label=f"Even ({even_depth:,.0f} ppm)")
        all_oe_y = np.concatenate([by_odd, by_evn]) if (bx_odd.size + bx_evn.size) > 0 else np.array([0.0, -1000.0])
        oe_span = max(600.0, float(np.nanmax(all_oe_y) - np.nanmin(all_oe_y)))
        ax10.set_ylim(float(np.nanmin(all_oe_y)) - 0.30 * oe_span, float(np.nanmax(all_oe_y)) + 0.28 * oe_span)
        ax10.legend(loc="upper right", fontsize=7.4, facecolor=BG_DARK, edgecolor=BORDER, labelcolor=TEXT_MAIN)

    ax10.axhline(0.0, color=BORDER, linestyle=":", linewidth=1.0)
    ax10.grid(True, linestyle=":", color=BORDER, alpha=0.55)
    ax10.set_xlabel("Phase from Mid-Transit (hours)", fontsize=8.5, color=TEXT_MUTED)
    ax10.set_ylabel("Relative Flux (ppm)", fontsize=8.5, color=TEXT_MUTED)

    oe_clean = oe_sigma < 3.0 and sec_snr < 4.5
    oe_col = GREEN_CERT if oe_clean else RED_ALERT
    oe_label = "SYMMETRIC" if oe_clean else "BINARY ASYMMETRY"
    ax10.text(
        0.50,
        0.045,
        f"{oe_label} · Odd-Even Δ={oe_sigma:.2f}σ · Sec={sec_snr:.1f}σ",
        transform=ax10.transAxes,
        ha="center",
        fontsize=7.5,
        fontweight="bold",
        color=oe_col,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": BG_DARK, "edgecolor": oe_col, "alpha": 0.92},
    )

    # ------------------------------------------------------------------
    # Panel 4 (Bottom-Right): Kepler's 3rd Law Stellar Density Check
    # ------------------------------------------------------------------
    ax11 = axes[1, 1]
    _style_dark_axis(ax11, "4 · Kepler's 3rd Law Stellar Density Consistency")

    log_gaia = math.log10(max(1e-3, rho_gaia_gcc))
    log_tran = math.log10(max(1e-3, rho_transit_gcc))
    density_ok = 0.18 <= rho_ratio <= 5.5

    # Acceptance zone around Gaia catalog stellar density
    ax11.axvspan(log_gaia + math.log10(0.18), log_gaia + math.log10(5.5), color=GREEN_CERT, alpha=0.14, label="Keplerian Consistent Zone")
    ax11.axvline(log_gaia, color=CYAN, linewidth=2.2, linestyle="-")
    ax11.barh(
        [1.05, 0.25],
        [log_gaia + 3.0, log_tran + 3.0],
        left=-3.0,
        height=0.34,
        color=[CYAN, GREEN_CERT if density_ok else RED_ALERT],
        alpha=0.88,
        edgecolor=TEXT_MAIN,
        linewidth=0.8,
    )
    ax11.set_yticks([0.25, 1.05])
    ax11.set_yticklabels(["Transit-Implied ρ★", "Gaia/TIC Host ρ★"], fontsize=8.5, color=TEXT_MAIN)
    ax11.set_ylim(-0.42, 1.50)
    x_lo = min(-2.2, log_gaia - 1.2, log_tran - 0.6)
    x_hi = max(1.5, log_gaia + 1.2, log_tran + 0.6)
    ax11.set_xlim(x_lo, x_hi)
    ax11.set_xlabel("log10( Stellar Density ρ★ [g/cm³] )", fontsize=8.5, color=TEXT_MUTED)
    ax11.grid(True, axis="x", linestyle=":", color=BORDER, alpha=0.55)

    dens_col = GREEN_CERT if density_ok else RED_ALERT
    dens_tag = "KEPLERIAN MATCH" if density_ok else "DENSITY VIOLATION"
    ax11.text(
        0.50,
        0.045,
        f"{dens_tag} · ρ_tr/ρ_Gaia={rho_ratio:.2f}× (R★={float(r_star):.2f}R☉, {int(teff_k)}K)",
        transform=ax11.transAxes,
        ha="center",
        fontsize=7.3,
        fontweight="bold",
        color=dens_col,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": BG_DARK, "edgecolor": dens_col, "alpha": 0.92},
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=105, facecolor=BG_DARK, metadata={"Software": "ARES"})
    png_bytes = buf.getvalue()
    sha256 = hashlib.sha256(png_bytes).hexdigest()

    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_tic = tic_id.replace("/", "_").replace(" ", "_")
    suffix = "masked" if mask_ccd_panel else "full"
    out_path = dest_dir / f"tess_{safe_tic}_{suffix}_{sha256[:10]}.png"
    out_path.write_bytes(png_bytes)

    return ArtifactRef(
        artifact_id=f"art_{sha256[:12]}",
        kind="plot_png",
        path=str(out_path),
        sha256=sha256,
        mime_type="image/png",
        produced_by_verifier=True,
    )
