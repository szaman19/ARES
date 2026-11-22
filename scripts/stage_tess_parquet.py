"""Convert downloaded real NASA TESS FITS files into partitioned Parquet shards (`data/tess_staged/`).

Reads `data/tess_raw/` (`nasa_toi_catalog.json`, `targets/*.fits`, `background/*.fits`),
extracts real 2-minute cadence `lightcurves`, `stellar_catalog` (with ALL ground-truth
disposition labels strictly stripped!), and 11x11 `pixel_stamps`, evaluates the exact
deterministic physics & spatial centroid metrics on each real NASA TOI, and curates:
  - 3x `multi_clue_join` cases (Real NASA Confirmed Planets `CP`/`KP`)
  - 3x `look_alike_decoy` cases (Loud Real NASA `FP` Cosmic Impostor + Subtle Real `CP`/`KP` Planet)
  - 3x `under_determined` cases (Real NASA `PC`/`APC` Crowded/Unconfirmed Candidates)
  - Additional real NASA holdout targets for live judge exploration.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domain.fits_utils import read_fits_hdus

RAW_DIR = REPO_ROOT / "data" / "tess_raw"
STAGED_DIR = REPO_ROOT / "data" / "tess_staged"

LIGHTCURVES_SCHEMA = pa.schema(
    [
        ("tic_id", pa.string()),
        ("sector", pa.int32()),
        ("cadence_no", pa.int32()),
        ("btjd", pa.float64()),
        ("pdcsap_flux", pa.float64()),
        ("rel_flux_ppm", pa.float64()),
        ("flux_err_ppm", pa.float64()),
        ("mom_centr1", pa.float64()),
        ("mom_centr2", pa.float64()),
        ("quality_flag", pa.int32()),
    ]
)

STELLAR_CATALOG_SCHEMA = pa.schema(
    [
        ("tic_id", pa.string()),
        ("sector", pa.int32()),
        ("tmag", pa.float64()),
        ("teff_k", pa.float64()),
        ("stellar_radius_rsun", pa.float64()),
        ("stellar_mass_msun", pa.float64()),
        ("stellar_density_gcc", pa.float64()),
        ("crowding_metric", pa.float64()),
        ("flux_fraction", pa.float64()),
        ("ra_deg", pa.float64()),
        ("dec_deg", pa.float64()),
        ("candidate_period_days", pa.float64()),
        ("candidate_t0_btjd", pa.float64()),
        ("candidate_duration_hours", pa.float64()),
    ]
)

PIXEL_STAMPS_SCHEMA = pa.schema(
    [
        ("tic_id", pa.string()),
        ("sector", pa.int32()),
        ("row_idx", pa.int32()),
        ("col_idx", pa.int32()),
        ("mean_out_flux", pa.float64()),
        ("mean_in_flux", pa.float64()),
        ("diff_flux", pa.float64()),
        ("diff_snr", pa.float64()),
        ("aperture_mask", pa.int32()),
        ("target_row", pa.float64()),
        ("target_col", pa.float64()),
    ]
)


def _detrend_to_ppm(
    btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    *,
    period_d: float | None = None,
    t0_btjd: float | None = None,
    dur_h: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize raw NASA SPOC `PDCSAP_FLUX` to continuum-detrended `rel_flux_ppm`.

    Masks in-transit cadences (if ephemeris is known) while estimating the slow
    stellar-rotation continuum via a sliding ~1.2-day median filter so real transit
    depths, odd/even asymmetries, and photon noise are preserved faithfully.
    """
    n = btjd.size
    rel_ppm = np.full(n, np.nan, dtype=np.float64)
    err_ppm = np.full(n, np.nan, dtype=np.float64)

    valid = np.isfinite(btjd) & np.isfinite(flux) & (flux > 0) & (quality == 0)
    if int(np.sum(valid)) < 30:
        return rel_ppm, err_ppm

    continuum_mask = valid.copy()
    if period_d and t0_btjd is not None and dur_h:
        epoch = np.round((btjd - t0_btjd) / period_d)
        phase_h = ((btjd - t0_btjd) - epoch * period_d) * 24.0
        in_dip = np.abs(phase_h) <= (0.75 * dur_h)
        # Also mask phase 0.5 so secondary eclipses are not detrended away!
        sec_phase_h = (((btjd - (t0_btjd + 0.5 * period_d) + 0.5 * period_d) % period_d) - 0.5 * period_d) * 24.0
        in_sec = np.abs(sec_phase_h) <= (0.75 * dur_h)
        continuum_mask = valid & (~in_dip) & (~in_sec)
        if int(np.sum(continuum_mask)) < 20:
            continuum_mask = valid.copy()

    t_cont = btjd[continuum_mask]
    f_cont = flux[continuum_mask]
    global_med = float(np.median(f_cont))

    # Interpolate local continuum across ~0.5-day knots
    t_min, t_max = float(np.nanmin(btjd[valid])), float(np.nanmax(btjd[valid]))
    n_knots = max(4, int((t_max - t_min) / 0.55))
    edges = np.linspace(t_min, t_max, n_knots + 1)
    knot_t: list[float] = []
    knot_f: list[float] = []
    for i in range(n_knots):
        m = (t_cont >= edges[i]) & (t_cont <= edges[i + 1])
        if int(np.sum(m)) >= 8:
            knot_t.append(float(np.median(t_cont[m])))
            knot_f.append(float(np.median(f_cont[m])))

    if len(knot_t) >= 2:
        baseline = np.interp(btjd, np.array(knot_t), np.array(knot_f))
    else:
        baseline = np.full(n, global_med, dtype=np.float64)

    finite_flux = np.isfinite(btjd) & np.isfinite(flux) & (baseline > 0)
    rel_ppm[finite_flux] = 1.0e6 * (flux[finite_flux] / baseline[finite_flux] - 1.0)

    finite_err = finite_flux & np.isfinite(flux_err) & (flux_err > 0)
    err_ppm[finite_err] = 1.0e6 * (flux_err[finite_err] / baseline[finite_err])
    if int(np.sum(finite_err)) < int(np.sum(finite_flux)):
        fallback_err = float(np.nanmedian(err_ppm[finite_err])) if np.any(finite_err) else 250.0
        err_ppm[finite_flux & (~finite_err)] = fallback_err

    return rel_ppm, err_ppm


def _extract_star_tables(
    lc_path: Path,
    tp_path: Path | None,
    *,
    toi_meta: dict[str, Any] | None = None,
) -> tuple[pa.Table, pa.Table, pa.Table] | None:
    """Parse one real NASA star's FITS file(s) into Arrow tables for the 3 domain views."""
    lc_hdus = read_fits_hdus(lc_path)
    if len(lc_hdus) < 2 or lc_hdus[1]["data"] is None:
        return None

    h0 = lc_hdus[0]["header"]
    h1 = lc_hdus[1]["header"]
    lc_data = lc_hdus[1]["data"]

    tid_int = int((toi_meta or {}).get("tid") or h0.get("TICID") or 0)
    if tid_int <= 0:
        return None
    tic_id = f"TIC-{tid_int}"
    sector = int((toi_meta or {}).get("sector") or h0.get("SECTOR") or 1)

    btjd = np.asarray(lc_data["TIME"], dtype=np.float64)
    cadence_no = np.asarray(lc_data["CADENCENO"], dtype=np.int32)
    flux = np.asarray(
        lc_data.get("PDCSAP_FLUX", lc_data.get("SAP_FLUX")), dtype=np.float64
    )
    flux_err = np.asarray(
        lc_data.get("PDCSAP_FLUX_ERR", lc_data.get("SAP_FLUX_ERR")), dtype=np.float64
    )
    quality = np.asarray(lc_data["QUALITY"], dtype=np.int32)
    mom1 = np.asarray(lc_data.get("MOM_CENTR1", np.full_like(btjd, np.nan)), dtype=np.float64)
    mom2 = np.asarray(lc_data.get("MOM_CENTR2", np.full_like(btjd, np.nan)), dtype=np.float64)

    keep = np.isfinite(btjd)
    if int(np.sum(keep)) < 200:
        return None

    btjd = btjd[keep]
    cadence_no = cadence_no[keep]
    flux = flux[keep]
    flux_err = flux_err[keep]
    quality = quality[keep]
    mom1 = mom1[keep]
    mom2 = mom2[keep]

    # Stellar & candidate ephemeris metadata (stripping ALL disposition labels!)
    tmag = float((toi_meta or {}).get("st_tmag") or h0.get("TESSMAG") or 11.0)
    teff_k = float((toi_meta or {}).get("st_teff") or h0.get("TEFF") or 5500.0)
    r_sun = float((toi_meta or {}).get("st_rad") or h0.get("RADIUS") or 1.0)
    r_sun = max(0.12, min(25.0, r_sun))
    logg = float((toi_meta or {}).get("st_logg") or h0.get("LOGG") or 4.438)
    # M/M_sun = (10^logg / 27420) * (R/R_sun)^2 ; rho_gcc = 1.408 * (M/M_sun) / (R/R_sun)^3
    m_sun = max(0.08, min(15.0, ((10.0 ** logg) / 27420.0) * (r_sun ** 2)))
    rho_gcc = 1.408 * m_sun / (r_sun ** 3)

    crowdsap = h1.get("CROWDSAP")
    crowding = float(crowdsap) if crowdsap is not None else float((toi_meta or {}).get("crowdsap") or 0.95)
    flfrcsap = h1.get("FLFRCSAP")
    flux_frac = float(flfrcsap) if flfrcsap is not None else 0.85
    ra_deg = float((toi_meta or {}).get("ra") or h0.get("RA_OBJ") or 0.0)
    dec_deg = float((toi_meta or {}).get("dec") or h0.get("DEC_OBJ") or 0.0)

    if toi_meta is not None:
        period_d: float | None = float(toi_meta["pl_orbper"])
        raw_t0 = float(toi_meta["pl_tranmid"])
        t0_btjd: float | None = raw_t0 - 2457000.0 if raw_t0 > 2400000.0 else raw_t0
        dur_h: float | None = float(toi_meta["pl_trandurh"])
    else:
        period_d = round(1.75 + (tid_int % 67) * 0.125, 5)
        t0_btjd = round(float(btjd[0]) + 0.65 + (tid_int % 11) * 0.09, 5)
        dur_h = round(1.8 + (tid_int % 5) * 0.35, 3)

    rel_ppm, err_ppm = _detrend_to_ppm(
        btjd,
        flux,
        flux_err,
        quality,
        period_d=period_d,
        t0_btjd=t0_btjd,
        dur_h=dur_h,
    )

    n_rows = btjd.size
    lc_table = pa.table(
        {
            "tic_id": pa.array([tic_id] * n_rows, type=pa.string()),
            "sector": pa.array(np.full(n_rows, sector, dtype=np.int32), type=pa.int32()),
            "cadence_no": pa.array(cadence_no, type=pa.int32()),
            "btjd": pa.array(btjd, type=pa.float64()),
            "pdcsap_flux": pa.array(
                np.where(np.isfinite(flux), flux, None), type=pa.float64()
            ),
            "rel_flux_ppm": pa.array(
                np.where(np.isfinite(rel_ppm), rel_ppm, None), type=pa.float64()
            ),
            "flux_err_ppm": pa.array(
                np.where(np.isfinite(err_ppm), err_ppm, None), type=pa.float64()
            ),
            "mom_centr1": pa.array(
                np.where(np.isfinite(mom1), mom1, None), type=pa.float64()
            ),
            "mom_centr2": pa.array(
                np.where(np.isfinite(mom2), mom2, None), type=pa.float64()
            ),
            "quality_flag": pa.array(quality, type=pa.int32()),
        },
        schema=LIGHTCURVES_SCHEMA,
    )

    cat_table = pa.table(
        {
            "tic_id": pa.array([tic_id], type=pa.string()),
            "sector": pa.array([sector], type=pa.int32()),
            "tmag": pa.array([round(tmag, 4)], type=pa.float64()),
            "teff_k": pa.array([round(teff_k, 1)], type=pa.float64()),
            "stellar_radius_rsun": pa.array([round(r_sun, 4)], type=pa.float64()),
            "stellar_mass_msun": pa.array([round(m_sun, 4)], type=pa.float64()),
            "stellar_density_gcc": pa.array([round(rho_gcc, 5)], type=pa.float64()),
            "crowding_metric": pa.array([round(crowding, 5)], type=pa.float64()),
            "flux_fraction": pa.array([round(flux_frac, 5)], type=pa.float64()),
            "ra_deg": pa.array([round(ra_deg, 6)], type=pa.float64()),
            "dec_deg": pa.array([round(dec_deg, 6)], type=pa.float64()),
            "candidate_period_days": pa.array([period_d], type=pa.float64()),
            "candidate_t0_btjd": pa.array([t0_btjd], type=pa.float64()),
            "candidate_duration_hours": pa.array([dur_h], type=pa.float64()),
        },
        schema=STELLAR_CATALOG_SCHEMA,
    )

    # Pixel stamp extraction from real _tp.fits if present
    pix_rows_dict: dict[str, list[Any]] = {f.name: [] for f in PIXEL_STAMPS_SCHEMA}
    if tp_path is not None and tp_path.is_file() and period_d and t0_btjd is not None and dur_h:
        tp_hdus = read_fits_hdus(tp_path)
        if len(tp_hdus) >= 3 and tp_hdus[1]["data"] is not None and tp_hdus[2]["data"] is not None:
            tp_data = tp_hdus[1]["data"]
            ap_img = np.asarray(tp_hdus[2]["data"], dtype=np.int32)
            tp_t = np.asarray(tp_data["TIME"], dtype=np.float64)
            tp_q = np.asarray(tp_data["QUALITY"], dtype=np.int32)
            tp_flux = np.asarray(tp_data["FLUX"], dtype=np.float64)  # (N, NY, NX)

            if tp_flux.ndim == 3 and ap_img.ndim == 2:
                ny, nx = tp_flux.shape[1], tp_flux.shape[2]
                # Crop to compact central 11x11 stamp around optimal aperture center if larger
                opt_mask = (ap_img & 2) != 0
                if np.any(opt_mask):
                    rr_all, cc_all = np.where(opt_mask)
                    cr = int(np.round(np.mean(rr_all)))
                    cc = int(np.round(np.mean(cc_all)))
                else:
                    cr, cc = ny // 2, nx // 2

                if ny > 13 or nx > 13:
                    r0 = max(0, min(ny - 11, cr - 5))
                    c0 = max(0, min(nx - 11, cc - 5))
                    r1 = min(ny, r0 + 11)
                    c1 = min(nx, c0 + 11)
                    tp_flux = tp_flux[:, r0:r1, c0:c1]
                    ap_img = ap_img[r0:r1, c0:c1]
                    ny, nx = tp_flux.shape[1], tp_flux.shape[2]

                good_t = np.isfinite(tp_t) & (tp_q == 0)
                epoch = np.round((tp_t - t0_btjd) / period_d)
                phase_h = ((tp_t - t0_btjd) - epoch * period_d) * 24.0
                in_m = good_t & (np.abs(phase_h) <= (0.425 * dur_h))
                out_m = good_t & (np.abs(phase_h) >= (0.75 * dur_h)) & (np.abs(phase_h) <= (2.2 * dur_h))

                if int(np.sum(in_m)) >= 3 and int(np.sum(out_m)) >= 8:
                    # Normalize per-epoch local pixel baseline to suppress spacecraft pointing drift
                    in_epochs = np.unique(epoch[in_m])
                    diff_per_epoch: list[np.ndarray] = []
                    out_frames = tp_flux[out_m]
                    mean_out = np.nanmedian(out_frames, axis=0)

                    for ep in in_epochs:
                        ep_in = in_m & (epoch == ep)
                        ep_out = out_m & (epoch == ep)
                        if int(np.sum(ep_in)) >= 2 and int(np.sum(ep_out)) >= 4:
                            loc_out = np.nanmedian(tp_flux[ep_out], axis=0)
                            loc_in = np.nanmedian(tp_flux[ep_in], axis=0)
                            diff_per_epoch.append(loc_out - loc_in)

                    if diff_per_epoch:
                        diff_stack = np.stack(diff_per_epoch, axis=0)
                        diff_img = np.nanmedian(diff_stack, axis=0)
                        if diff_stack.shape[0] >= 3:
                            mad = np.nanmedian(np.abs(diff_stack - diff_img), axis=0) * 1.4826
                            err_img = mad / math.sqrt(diff_stack.shape[0])
                        else:
                            mad_out = np.nanmedian(np.abs(out_frames - mean_out), axis=0) * 1.4826
                            err_img = mad_out / math.sqrt(max(1, int(np.sum(in_m))))
                    else:
                        mean_in = np.nanmedian(tp_flux[in_m], axis=0)
                        diff_img = mean_out - mean_in
                        mad_out = np.nanmedian(np.abs(out_frames - mean_out), axis=0) * 1.4826
                        err_img = mad_out / math.sqrt(max(1, int(np.sum(in_m))))

                    mean_in_final = mean_out - diff_img
                    safe_err = np.maximum(0.25, np.nan_to_num(err_img, nan=1.0))
                    diff_snr = np.nan_to_num(diff_img, nan=0.0) / safe_err

                    # Target star catalog/out-of-transit pixel center inside optimal SPOC aperture
                    ap_opt = (ap_img & 2) != 0
                    pos_out = np.maximum(0.0, np.nan_to_num(mean_out, nan=0.0))
                    target_weights = pos_out * ap_opt.astype(np.float64)
                    if float(np.sum(target_weights)) <= 0:
                        target_weights = pos_out
                    rr_grid, cc_grid = np.indices((ny, nx))
                    tw_sum = float(np.sum(target_weights))
                    t_row = float(np.sum(rr_grid * target_weights) / tw_sum) if tw_sum > 0 else (ny - 1) / 2.0
                    t_col = float(np.sum(cc_grid * target_weights) / tw_sum) if tw_sum > 0 else (nx - 1) / 2.0

                    for r_i in range(ny):
                        for c_i in range(nx):
                            pix_rows_dict["tic_id"].append(tic_id)
                            pix_rows_dict["sector"].append(sector)
                            pix_rows_dict["row_idx"].append(r_i)
                            pix_rows_dict["col_idx"].append(c_i)
                            pix_rows_dict["mean_out_flux"].append(float(np.nan_to_num(mean_out[r_i, c_i], nan=0.0)))
                            pix_rows_dict["mean_in_flux"].append(float(np.nan_to_num(mean_in_final[r_i, c_i], nan=0.0)))
                            pix_rows_dict["diff_flux"].append(float(np.nan_to_num(diff_img[r_i, c_i], nan=0.0)))
                            pix_rows_dict["diff_snr"].append(float(np.nan_to_num(diff_snr[r_i, c_i], nan=0.0)))
                            pix_rows_dict["aperture_mask"].append(int(ap_img[r_i, c_i]))
                            pix_rows_dict["target_row"].append(round(t_row, 5))
                            pix_rows_dict["target_col"].append(round(t_col, 5))

    pix_table = pa.table(
        {
            f.name: pa.array(pix_rows_dict[f.name], type=f.type)
            for f in PIXEL_STAMPS_SCHEMA
        },
        schema=PIXEL_STAMPS_SCHEMA,
    )
    return lc_table, cat_table, pix_table


def _evaluate_star_physics(target_dir: Path) -> dict[str, Any]:
    """Run the exact DuckDB physics queries on a single staged target star directory."""
    con = duckdb.connect(":memory:")
    lc_p = str(target_dir / "lightcurves.parquet").replace("'", "''")
    cat_p = str(target_dir / "stellar_catalog.parquet").replace("'", "''")
    pix_p = str(target_dir / "pixel_stamps.parquet").replace("'", "''")

    row = con.execute(
        f"""
        WITH folded AS (
            SELECT
                l.tic_id,
                s.candidate_period_days AS p_days,
                s.candidate_t0_btjd AS t0,
                s.candidate_duration_hours AS dur_h,
                s.stellar_density_gcc AS rho_gaia,
                s.crowding_metric,
                l.btjd,
                l.rel_flux_ppm,
                l.flux_err_ppm,
                CAST(ROUND((l.btjd - s.candidate_t0_btjd) / s.candidate_period_days) AS BIGINT) AS epoch_idx,
                ((l.btjd - s.candidate_t0_btjd)
                 - ROUND((l.btjd - s.candidate_t0_btjd) / s.candidate_period_days) * s.candidate_period_days
                ) * 24.0 AS phase_h,
                ((((l.btjd - (s.candidate_t0_btjd + 0.5 * s.candidate_period_days))
                   % s.candidate_period_days) + 1.5 * s.candidate_period_days)
                 % s.candidate_period_days - 0.5 * s.candidate_period_days
                ) * 24.0 AS sec_phase_h
            FROM read_parquet('{lc_p}') l
            INNER JOIN read_parquet('{cat_p}') s ON l.tic_id = s.tic_id
            WHERE l.quality_flag = 0
              AND l.rel_flux_ppm IS NOT NULL
              AND s.candidate_period_days IS NOT NULL
        ),
        stats AS (
            SELECT
                tic_id,
                ANY_VALUE(p_days) AS p_days,
                ANY_VALUE(dur_h) AS dur_h,
                ANY_VALUE(rho_gaia) AS rho_gaia,
                ANY_VALUE(crowding_metric) AS crowding_metric,
                COUNT(DISTINCT epoch_idx) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h) AS n_transits,
                COUNT_IF(ABS(phase_h) <= 0.425 * dur_h) AS n_in,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) BETWEEN 0.75 * dur_h AND 2.0 * dur_h) AS out_med,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h) AS in_med,
                MEDIAN(flux_err_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h) AS in_err,
                COUNT_IF(ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 1) AS n_odd,
                COUNT_IF(ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 0) AS n_even,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 1) AS odd_in_med,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h AND ABS(epoch_idx) % 2 = 0) AS even_in_med,
                COUNT_IF(ABS(sec_phase_h) <= 0.425 * dur_h) AS n_sec_in,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(sec_phase_h) BETWEEN 0.75 * dur_h AND 2.0 * dur_h) AS sec_out_med,
                MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(sec_phase_h) <= 0.425 * dur_h) AS sec_in_med
            FROM folded
            GROUP BY tic_id
        )
        SELECT * FROM stats
        """
    ).fetchone()

    pix_row = con.execute(
        f"""
        SELECT
            SUM(row_idx * POW(GREATEST(0.0, diff_snr), 2)) / NULLIF(SUM(POW(GREATEST(0.0, diff_snr), 2)), 0) AS cent_r,
            SUM(col_idx * POW(GREATEST(0.0, diff_snr), 2)) / NULLIF(SUM(POW(GREATEST(0.0, diff_snr), 2)), 0) AS cent_c,
            AVG(target_row) AS t_row,
            AVG(target_col) AS t_col,
            MAX(diff_snr) AS peak_pix_snr
        FROM read_parquet('{pix_p}')
        """
    ).fetchone()
    con.close()

    if row is None:
        return {"valid": False}

    (
        tic_id,
        p_days,
        dur_h,
        rho_gaia,
        crowding,
        n_transits,
        n_in,
        out_med,
        in_med,
        in_err,
        n_odd,
        n_even,
        odd_in_med,
        even_in_med,
        n_sec_in,
        sec_out_med,
        sec_in_med,
    ) = row

    if n_in < 4 or out_med is None or in_med is None:
        return {"valid": False}

    depth_ppm = max(0.0, float(out_med) - float(in_med))
    in_err_f = float(in_err or 250.0)
    transit_snr = depth_ppm / max(15.0, in_err_f / math.sqrt(max(1, int(n_in))))

    if n_odd >= 2 and n_even >= 2 and odd_in_med is not None and even_in_med is not None:
        oe_diff = abs(float(odd_in_med) - float(even_in_med))
        oe_err = max(40.0, in_err_f / math.sqrt(max(1, min(int(n_odd), int(n_even)))))
        oe_sigma = oe_diff / oe_err
    else:
        oe_sigma = 99.0

    if n_sec_in >= 3 and sec_out_med is not None and sec_in_med is not None:
        sec_depth = max(0.0, float(sec_out_med) - float(sec_in_med))
        sec_snr = sec_depth / max(20.0, in_err_f / math.sqrt(max(1, int(n_sec_in))))
    else:
        sec_snr = 0.0

    k = math.sqrt(min(0.25, max(1e-5, depth_ppm * 1e-6)))
    sin_term = math.sin(min(math.pi / 2.1, math.pi * (float(dur_h) / 24.0) / float(p_days)))
    a_r = max(1.2, (1.0 + k) / max(1e-4, sin_term))
    rho_tr = (3.0 * math.pi / (6.6743e-8 * ((float(p_days) * 86400.0) ** 2))) * (a_r ** 3)
    density_ratio = rho_tr / max(1e-4, float(rho_gaia))

    if pix_row and pix_row[0] is not None and pix_row[2] is not None:
        cent_offset = math.hypot(float(pix_row[0]) - float(pix_row[2]), float(pix_row[1]) - float(pix_row[3]))
        peak_pix_snr = float(pix_row[4] or 0.0)
    else:
        cent_offset = 99.0
        peak_pix_snr = 0.0

    pass_snr = int(n_transits) >= 2 and transit_snr >= 7.0 and depth_ppm >= 150.0
    pass_oe = pass_snr and (oe_sigma < 3.0) and (sec_snr < 4.5)
    pass_dens = pass_snr and (0.18 <= density_ratio <= 5.5)
    pass_cent = pass_snr and (cent_offset <= 0.45) and (peak_pix_snr >= 5.5) and (float(crowding) >= 0.80)

    return {
        "valid": True,
        "tic_id": tic_id,
        "depth_ppm": round(depth_ppm, 1),
        "transit_snr": round(transit_snr, 2),
        "n_transits": int(n_transits),
        "oe_sigma": round(oe_sigma, 3),
        "sec_snr": round(sec_snr, 2),
        "density_ratio": round(density_ratio, 3),
        "cent_offset_px": round(cent_offset, 3),
        "peak_pix_snr": round(peak_pix_snr, 2),
        "crowding_metric": round(float(crowding), 4),
        "pass_snr": pass_snr,
        "pass_oe": pass_oe,
        "pass_dens": pass_dens,
        "pass_cent": pass_cent,
        "pass_all_four": pass_snr and pass_oe and pass_dens and pass_cent,
    }


def _bg_star_triggers_snr(lc_t: pa.Table, cat_t: pa.Table) -> bool:
    """Return True if a background star's trial ephemeris accidentally passes transit SNR."""
    con = duckdb.connect(":memory:")
    con.register("lc", lc_t)
    con.register("cat", cat_t)
    row = con.execute(
        """
        WITH folded AS (
            SELECT
                s.candidate_duration_hours AS dur_h,
                l.rel_flux_ppm,
                l.flux_err_ppm,
                CAST(ROUND((l.btjd - s.candidate_t0_btjd) / s.candidate_period_days) AS BIGINT) AS epoch_idx,
                ((l.btjd - s.candidate_t0_btjd)
                 - ROUND((l.btjd - s.candidate_t0_btjd) / s.candidate_period_days) * s.candidate_period_days
                ) * 24.0 AS phase_h
            FROM lc l
            INNER JOIN cat s ON l.tic_id = s.tic_id
            WHERE l.quality_flag = 0
              AND l.rel_flux_ppm IS NOT NULL
              AND s.candidate_period_days IS NOT NULL
        )
        SELECT
            COUNT(DISTINCT epoch_idx) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h) AS n_transits,
            COUNT_IF(ABS(phase_h) <= 0.425 * dur_h) AS n_in,
            MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) BETWEEN 0.75 * dur_h AND 2.0 * dur_h) AS out_med,
            MEDIAN(rel_flux_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h) AS in_med,
            COALESCE(MEDIAN(flux_err_ppm) FILTER (WHERE ABS(phase_h) <= 0.425 * dur_h), 250.0) AS in_err
        FROM folded
        """
    ).fetchone()
    con.close()
    if row is None:
        return False
    n_transits, n_in, out_med, in_med, in_err = row
    if int(n_in or 0) < 4 or out_med is None or in_med is None:
        return False
    depth_ppm = max(0.0, float(out_med) - float(in_med))
    snr = depth_ppm / max(15.0, float(in_err or 250.0) / math.sqrt(max(1, int(n_in))))
    return int(n_transits or 0) >= 2 and snr >= 7.0 and depth_ppm >= 150.0


def main() -> None:
    manifest_path = RAW_DIR / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run scripts/fetch_real_tess_raw.py first: {manifest_path}")

    raw_manifest = json.loads(manifest_path.read_text())
    targets_meta: list[dict[str, Any]] = raw_manifest["targets"]
    bg_meta: list[dict[str, Any]] = raw_manifest["background"]

    if STAGED_DIR.exists():
        shutil.rmtree(STAGED_DIR)
    (STAGED_DIR / "background" / "lightcurves").mkdir(parents=True, exist_ok=True)
    (STAGED_DIR / "background" / "stellar_catalog").mkdir(parents=True, exist_ok=True)
    (STAGED_DIR / "background" / "pixel_stamps").mkdir(parents=True, exist_ok=True)
    (STAGED_DIR / "targets").mkdir(parents=True, exist_ok=True)

    # 1. Stage background TESS sector stars into 4 shards
    num_shards = 4
    bg_lc_shards: list[list[pa.Table]] = [[] for _ in range(num_shards)]
    bg_cat_shards: list[list[pa.Table]] = [[] for _ in range(num_shards)]
    bg_pix_shards: list[list[pa.Table]] = [[] for _ in range(num_shards)]
    total_bg_cadences = 0
    staged_bg_stars = 0

    skipped_bg_false_snr = 0
    for idx, bm in enumerate(bg_meta):
        lc_p = REPO_ROOT / bm["lc_local"]
        if not lc_p.is_file():
            continue
        parsed = _extract_star_tables(lc_p, None, toi_meta=None)
        if parsed is None:
            continue
        lc_t, cat_t, pix_t = parsed
        if _bg_star_triggers_snr(lc_t, cat_t):
            skipped_bg_false_snr += 1
            continue
        shard_i = staged_bg_stars % num_shards
        bg_lc_shards[shard_i].append(lc_t)
        bg_cat_shards[shard_i].append(cat_t)
        bg_pix_shards[shard_i].append(pix_t)
        total_bg_cadences += lc_t.num_rows
        staged_bg_stars += 1

    for s_i in range(num_shards):
        if bg_lc_shards[s_i]:
            pq.write_table(
                pa.concat_tables(bg_lc_shards[s_i]),
                STAGED_DIR / "background" / "lightcurves" / f"bg_part_{s_i:04d}.parquet",
            )
            pq.write_table(
                pa.concat_tables(bg_cat_shards[s_i]),
                STAGED_DIR / "background" / "stellar_catalog" / f"bg_part_{s_i:04d}.parquet",
            )
            pq.write_table(
                pa.concat_tables(bg_pix_shards[s_i]),
                STAGED_DIR / "background" / "pixel_stamps" / f"bg_part_{s_i:04d}.parquet",
            )

    print(
        f"[BG] Staged {staged_bg_stars} real quiet TESS stars "
        f"({total_bg_cadences:,} real 2-min cadence rows across {num_shards} Parquet shards, "
        f"skipped_accidental_snr={skipped_bg_false_snr})."
    )

    # 2. Stage & physics-profile all downloaded NASA TOI targets
    confirmed_planets: list[dict[str, Any]] = []
    vetted_impostors: list[dict[str, Any]] = []
    ambiguous_candidates: list[dict[str, Any]] = []

    for tm in targets_meta:
        lc_p = REPO_ROOT / tm["lc_local"]
        tp_p = REPO_ROOT / tm["tp_local"]
        if not (lc_p.is_file() and tp_p.is_file()):
            continue
        parsed = _extract_star_tables(lc_p, tp_p, toi_meta=tm)
        if parsed is None:
            continue
        lc_t, cat_t, pix_t = parsed
        tic_id = f"TIC-{tm['tid']}"
        tdir = STAGED_DIR / "targets" / tic_id
        tdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(lc_t, tdir / "lightcurves.parquet")
        pq.write_table(cat_t, tdir / "stellar_catalog.parquet")
        pq.write_table(pix_t, tdir / "pixel_stamps.parquet")

        phys = _evaluate_star_physics(tdir)
        if not phys.get("valid"):
            continue
        record = {
            **tm,
            "tic_id": tic_id,
            "cadence_rows": lc_t.num_rows,
            "physics": phys,
        }
        disp = tm["tfopwg_disp"]
        if disp in ("CP", "KP") and phys["pass_all_four"]:
            confirmed_planets.append(record)
        elif disp in ("FP", "FA") and phys["pass_snr"] and not phys["pass_all_four"]:
            vetted_impostors.append(record)
        elif disp in ("PC", "APC") and phys["pass_snr"] and not phys["pass_all_four"]:
            ambiguous_candidates.append(record)

    print(
        f"[PROFILED] Clean real NASA matches:\n"
        f"  Confirmed Planets (CP/KP passing all 4 checks): {len(confirmed_planets)}\n"
        f"  Cosmic Impostors (FP passing SNR, failing >=1 physics/CCD check): {len(vetted_impostors)}\n"
        f"  Unconfirmed Candidates (PC/APC passing SNR, ambiguous/incomplete): {len(ambiguous_candidates)}"
    )

    for label, grp in (
        ("CP/KP", confirmed_planets[:8]),
        ("FP", vetted_impostors[:10]),
        ("PC/APC", ambiguous_candidates[:10]),
    ):
        print(f"\n--- Top {label} ---")
        for r in grp:
            p = r["physics"]
            print(
                f"  {r['tic_id']} (TOI-{r['toi']}, {r['tfopwg_disp']}): "
                f"SNR={p['transit_snr']:.1f}, depth={p['depth_ppm']:.0f}ppm, "
                f"OE={p['oe_sigma']:.2f}s(sec={p['sec_snr']:.1f}s), "
                f"rho_rat={p['density_ratio']:.2f}, cent={p['cent_offset_px']:.2f}px, "
                f"crowd={p['crowding_metric']:.2f} | "
                f"checks=[SNR:{p['pass_snr']}, OE:{p['pass_oe']}, DENS:{p['pass_dens']}, CENT:{p['pass_cent']}]"
            )

    summary_out = STAGED_DIR / "profiled_targets.json"
    summary_out.write_text(
        json.dumps(
            {
                "background_stars": staged_bg_stars,
                "background_cadences": total_bg_cadences,
                "confirmed_planets": confirmed_planets,
                "vetted_impostors": vetted_impostors,
                "ambiguous_candidates": ambiguous_candidates,
            },
            indent=2,
        )
    )
    print(f"\n[SAVED] {summary_out}")


if __name__ == "__main__":
    main()
