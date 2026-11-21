#!/usr/bin/env python3
"""Safely stage real-world multimodal dataset slices into `./.data_cache/`.

All downloaded data goes strictly inside `--cache-dir` (default `./.data_cache`)
so the entire cache can be wiped at any time with:
    rm -rf ./.data_cache

Safety controls enforced:
- Explicit HTTPS host allowlist ONLY (`huggingface.co`, `storage.googleapis.com`, `lila.science`).
- Safe formats ONLY (`.parquet`, `.json`, `.png`, `.jpg`, `.wav`). Zero `.pkl`/`.pt`/`.sh` or `trust_remote_code`.
- Magic-byte verification + Pillow `MAX_IMAGE_PIXELS` bomb guard + SHA-256 manifest logging.
- Ground-truth labels written to `ground_truth.json` (never inside `observations/*.parquet`).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from domain.multimodal_datasets import (  # noqa: E402
    DATASET_CATALOG,
    SafetyViolationError,
    safe_extract_single_json_from_zip,
    safe_https_get_bytes,
    sha256_bytes,
    verify_magic_bytes,
    verify_safe_https_url,
    verify_safe_output_path,
)

matplotlib.use("Agg")
Image.MAX_IMAGE_PIXELS = 25_000_000  # Strict decompression-bomb ceiling


BTSBOT_PARQUET_URL = (
    "https://huggingface.co/datasets/MultimodalUniverse/btsbot/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
LILA_CCT_LABELS_ZIP_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_camera_traps.json.zip"
)
LILA_CCT_MD_ZIP_URL = (
    "https://lila.science/public/lila-md-results/"
    "caltech-camera-traps_mdv5a.0.0_results.json.zip"
)
LILA_CCT_IMAGE_BASE_URL = (
    "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images"
)
PYRO_SDIS_VAL_PARQUET_URL = (
    "https://huggingface.co/datasets/pyronear/pyro-sdis/resolve/"
    "main/data/val-00000-of-00001.parquet"
)
URBANSOUND_PARQUET_URL = (
    "https://huggingface.co/datasets/danavery/urbansound8K/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)


def _open_httpfs_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET threads TO 4;")
    return con


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _check_disk_budget(cache_root: Path, max_total_mb: float) -> None:
    used_mb = _directory_size_bytes(cache_root) / (1024 * 1024)
    if used_mb > max_total_mb:
        raise SafetyViolationError(
            f"Cache directory {cache_root} reached {used_mb:.1f} MB, exceeding "
            f"--max-total-mb={max_total_mb:.1f} MB safety ceiling."
        )


def _write_dataset_artifacts(
    ds_dir: Path,
    *,
    dataset_id: str,
    table: pa.Table,
    ground_truth_records: dict[str, dict[str, Any]],
    media_hashes: dict[str, str],
) -> dict[str, Any]:
    obs_dir = ds_dir / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = verify_safe_output_path(obs_dir / "part-0000.parquet", ds_dir)
    pq.write_table(table, parquet_path, compression="zstd")
    parquet_bytes = parquet_path.read_bytes()
    verify_magic_bytes(parquet_bytes, ".parquet", max_bytes=250 * 1024 * 1024)

    gt_payload = {
        "dataset_id": dataset_id,
        "record_count": len(ground_truth_records),
        "records": ground_truth_records,
    }
    gt_path = verify_safe_output_path(ds_dir / "ground_truth.json", ds_dir)
    gt_bytes = json.dumps(gt_payload, indent=2, sort_keys=True).encode("utf-8")
    gt_path.write_bytes(gt_bytes)

    pos_count = sum(1 for v in ground_truth_records.values() if v.get("is_true_event"))
    decoy_count = sum(
        1
        for v in ground_truth_records.values()
        if not v.get("is_true_event") and v.get("is_decoy", True)
    )
    manifest = {
        "dataset_id": dataset_id,
        "title": DATASET_CATALOG[dataset_id].title,
        "source_url": DATASET_CATALOG[dataset_id].source_url,
        "row_count": table.num_rows,
        "columns": table.column_names,
        "positive_count": pos_count,
        "decoy_count": decoy_count,
        "staged_media_files": len(media_hashes),
        "observations_parquet_sha256": sha256_bytes(parquet_bytes),
        "ground_truth_sha256": sha256_bytes(gt_bytes),
        "total_dataset_bytes": _directory_size_bytes(ds_dir),
        "media_sha256": media_hashes,
    }
    manifest_path = verify_safe_output_path(ds_dir / "manifest.json", ds_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# 1. Astronomy: ZTF BTSbot (MultimodalUniverse/btsbot)
# ---------------------------------------------------------------------------


def _render_btsbot_triplet_png(
    views: list[str],
    bands: list[str],
    arrays: list[Any],
    record_id: str,
) -> bytes:
    """Render deterministic 3-panel Science / Reference / Difference PNG sky cutout."""
    fig = Figure(figsize=(6.6, 2.4), dpi=110)
    canvas = FigureCanvasAgg(fig)
    n_panels = min(3, len(arrays))
    for idx in range(n_panels):
        ax = fig.add_subplot(1, 3, idx + 1)
        arr = np.asarray(arrays[idx], dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            vmin, vmax = np.percentile(finite, [5.0, 99.2])
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        ax.imshow(
            np.nan_to_num(arr, nan=float(vmin)),
            cmap="magma" if idx < 2 else "coolwarm",
            vmin=float(vmin),
            vmax=float(vmax),
            origin="lower",
        )
        vname = str(views[idx]) if idx < len(views) else f"view_{idx}"
        bname = str(bands[idx]).replace("b'", "").replace("'", "") if idx < len(bands) else ""
        ax.set_title(f"{vname} ({bname})", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        # Crosshair at transient alert position (center of 63x63 stamp)
        cy, cx = arr.shape[0] / 2.0, arr.shape[1] / 2.0
        ax.plot([cx - 6, cx - 2], [cy, cy], color="cyan", lw=0.9)
        ax.plot([cx + 2, cx + 6], [cy, cy], color="cyan", lw=0.9)
        ax.plot([cx, cx], [cy - 6, cy - 2], color="cyan", lw=0.9)
        ax.plot([cx, cx], [cy + 2, cy + 6], color="cyan", lw=0.9)

    fig.suptitle(f"ZTF Alert {record_id}", fontsize=9.0, y=0.98)
    fig.tight_layout()
    buf = io.BytesIO()
    canvas.print_png(buf)
    png_bytes = buf.getvalue()
    verify_magic_bytes(png_bytes, ".png")
    return png_bytes


def stage_btsbot(
    cache_root: Path,
    *,
    row_limit: int = 1200,
    media_limit: int = 120,
    max_total_mb: float = 600.0,
) -> dict[str, Any]:
    """Stage ZTF BTSbot supernova transient alerts + 3-panel cutout PNG stamps."""
    verify_safe_https_url(BTSBOT_PARQUET_URL)
    ds_dir = cache_root / "btsbot"
    media_dir = ds_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    con = _open_httpfs_duckdb()
    try:
        rows = con.execute(
            f"""
            SELECT
                CAST(object_id AS VARCHAR) || '_' || CAST( row_number() OVER () AS VARCHAR ) AS record_id,
                CAST("OBJECT_ID_" AS VARCHAR) AS object_id,
                CAST(jd AS DOUBLE) AS jd,
                CAST(magpsf AS DOUBLE) AS magpsf,
                CAST(sigmapsf AS DOUBLE) AS sigmapsf,
                CAST(diffmaglim AS DOUBLE) AS diffmaglim,
                CAST(dsnrms AS DOUBLE) AS dsnrms,
                CAST(ssnrms AS DOUBLE) AS ssnrms,
                CAST(drb AS DOUBLE) AS drb,
                CAST(new_drb AS DOUBLE) AS new_drb,
                CAST(fwhm AS DOUBLE) AS fwhm,
                CAST(chipsf AS DOUBLE) AS chipsf,
                CAST(nbad AS BIGINT) AS nbad,
                CAST(nneg AS BIGINT) AS nneg,
                CAST(mindtoedge AS DOUBLE) AS mindtoedge,
                CAST(sgscore1 AS DOUBLE) AS sgscore1,
                CAST(distpsnr1 AS DOUBLE) AS distpsnr1,
                CAST(ndethist AS BIGINT) AS ndethist,
                CAST(ncovhist AS BIGINT) AS ncovhist,
                CAST(age AS DOUBLE) AS age,
                CAST(days_since_peak AS DOUBLE) AS days_since_peak,
                CAST(acai_h AS DOUBLE) AS acai_h,
                CAST(acai_v AS DOUBLE) AS acai_v,
                CAST(acai_o AS DOUBLE) AS acai_o,
                CAST(acai_n AS DOUBLE) AS acai_n,
                CAST(acai_b AS DOUBLE) AS acai_b,
                CAST(is_rise AS BOOLEAN) AS is_rise,
                CAST(near_threshold AS BOOLEAN) AS near_threshold,
                CAST(is_SN AS BOOLEAN) AS is_sn_gt,
                CAST(label AS BIGINT) AS bts_label_gt,
                image."view" AS img_views,
                image.band AS img_bands,
                image."array" AS img_arrays
            FROM (
                SELECT * FROM read_parquet('{BTSBOT_PARQUET_URL}')
                LIMIT {int(row_limit)}
            )
            """
        ).fetchall()
        col_names = [d[0] for d in con.description]
    finally:
        con.close()

    # Select balanced media subset: half real supernovae, half high-drb non-SN decoys!
    half = max(1, media_limit // 2)
    pos_indices: list[int] = []
    decoy_indices: list[int] = []
    for i, r in enumerate(rows):
        rec = dict(zip(col_names, r))
        if rec["is_sn_gt"] and len(pos_indices) < half:
            pos_indices.append(i)
        elif (not rec["is_sn_gt"]) and (rec["drb"] or 0.0) > 0.90 and len(decoy_indices) < half:
            decoy_indices.append(i)
    selected_media_idx = set(pos_indices + decoy_indices)

    media_hashes: dict[str, str] = {}
    gt_records: dict[str, dict[str, Any]] = {}
    out_cols: dict[str, list[Any]] = {
        c: []
        for c in col_names
        if c not in ("is_sn_gt", "bts_label_gt", "img_views", "img_bands", "img_arrays")
    }
    out_cols["media_relpath"] = []

    for i, r in enumerate(rows):
        rec = dict(zip(col_names, r))
        rid = str(rec["record_id"])
        relpath = ""
        if i in selected_media_idx and rec["img_arrays"] is not None:
            png_bytes = _render_btsbot_triplet_png(
                rec["img_views"] or [],
                rec["img_bands"] or [],
                rec["img_arrays"],
                rid,
            )
            relpath = f"media/{rid}.png"
            out_file = verify_safe_output_path(ds_dir / relpath, ds_dir)
            out_file.write_bytes(png_bytes)
            media_hashes[relpath] = sha256_bytes(png_bytes)

        is_sn = bool(rec["is_sn_gt"])
        bts_label = int(rec["bts_label_gt"] or 0)
        gt_records[rid] = {
            "is_true_event": is_sn,
            "is_bright_sn_target": bool(bts_label == 1),
            "is_decoy": bool((not is_sn) and (rec["drb"] or 0.0) >= 0.85),
            "ground_truth_class": "supernova" if is_sn else "non_sn_decoy_or_artifact",
        }

        for k in out_cols:
            if k == "media_relpath":
                out_cols[k].append(relpath)
            else:
                out_cols[k].append(rec[k])

    _check_disk_budget(cache_root, max_total_mb)
    table = pa.Table.from_pydict(out_cols)
    return _write_dataset_artifacts(
        ds_dir,
        dataset_id="btsbot",
        table=table,
        ground_truth_records=gt_records,
        media_hashes=media_hashes,
    )


# ---------------------------------------------------------------------------
# 2. Wildlife Camera Traps: Caltech Camera Traps + MegaDetector v5 (LILA BC)
# ---------------------------------------------------------------------------


def stage_caltech_camera_traps(
    cache_root: Path,
    *,
    media_limit: int = 36,
    max_total_mb: float = 600.0,
) -> dict[str, Any]:
    """Stage all 243,100 Caltech Camera Trap rows + MegaDetector scores + sample JPEGs."""
    ds_dir = cache_root / "caltech_camera_traps"
    media_dir = ds_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    labels_zip = safe_https_get_bytes(LILA_CCT_LABELS_ZIP_URL, max_bytes=30 * 1024 * 1024)
    labels_doc = json.loads(safe_extract_single_json_from_zip(labels_zip).decode("utf-8"))

    md_zip = safe_https_get_bytes(LILA_CCT_MD_ZIP_URL, max_bytes=30 * 1024 * 1024)
    md_doc = json.loads(safe_extract_single_json_from_zip(md_zip).decode("utf-8"))

    cat_map: dict[int, str] = {
        int(c["id"]): str(c["name"]).lower().replace(" ", "_")
        for c in labels_doc.get("categories", [])
    }
    ann_by_img: dict[str, str] = {}
    for ann in labels_doc.get("annotations", []):
        img_id = str(ann["image_id"])
        cname = cat_map.get(int(ann["category_id"]), "unknown")
        # Prefer non-empty label if multi-label annotation exists
        if img_id not in ann_by_img or ann_by_img[img_id] == "empty":
            ann_by_img[img_id] = cname

    md_by_file: dict[str, tuple[float, int, float]] = {}
    for item in md_doc.get("images", []):
        fn = str(item.get("file", ""))
        dets = item.get("detections") or []
        max_conf = 0.0
        n_high = 0
        max_area = 0.0
        for d in dets:
            conf = float(d.get("conf", 0.0))
            if conf > max_conf:
                max_conf = conf
            if conf >= 0.20:
                n_high += 1
            bbox = d.get("bbox") or [0, 0, 0, 0]
            if len(bbox) == 4:
                area = float(bbox[2]) * float(bbox[3])
                if area > max_area:
                    max_area = area
        md_by_file[fn] = (round(max_conf, 4), n_high, round(max_area, 5))

    images_list = labels_doc.get("images", [])

    # Pick balanced media candidates to pre-cache:
    # 1) Confirmed predators/mammals (bobcat, coyote, mountain_lion, fox, deer) with high MD conf
    # 2) Natural false-alarm decoys: ground-truth `empty` frames where MegaDetector conf >= 0.75!
    half = max(1, media_limit // 2)
    interesting_species = {"bobcat", "coyote", "mountain_lion", "fox", "deer", "raccoon"}
    pos_candidates: list[dict[str, Any]] = []
    decoy_candidates: list[dict[str, Any]] = []

    for img in images_list:
        iid = str(img["id"])
        fn = str(img["file_name"])
        sp = ann_by_img.get(iid, "empty")
        max_conf, _, _ = md_by_file.get(fn, (0.0, 0, 0.0))
        if sp in interesting_species and max_conf >= 0.80 and len(pos_candidates) < half:
            pos_candidates.append(img)
        elif sp == "empty" and max_conf >= 0.80 and len(decoy_candidates) < half:
            decoy_candidates.append(img)
        if len(pos_candidates) >= half and len(decoy_candidates) >= half:
            break

    precached_relpaths: dict[str, str] = {}
    media_hashes: dict[str, str] = {}
    for img in pos_candidates + decoy_candidates:
        iid = str(img["id"])
        fn = str(img["file_name"])
        img_url = f"{LILA_CCT_IMAGE_BASE_URL}/{fn}"
        try:
            jpg_bytes = safe_https_get_bytes(img_url, max_bytes=10 * 1024 * 1024, timeout_s=20.0)
            verify_magic_bytes(jpg_bytes, ".jpg")
            relpath = f"media/{iid}.jpg"
            out_file = verify_safe_output_path(ds_dir / relpath, ds_dir)
            out_file.write_bytes(jpg_bytes)
            precached_relpaths[iid] = relpath
            media_hashes[relpath] = sha256_bytes(jpg_bytes)
        except Exception:
            # Skip individual transient network hiccups gracefully
            continue

    out_cols: dict[str, list[Any]] = {
        "record_id": [],
        "location_id": [],
        "captured_at": [],
        "hour_local": [],
        "seq_id": [],
        "seq_num_frames": [],
        "frame_num": [],
        "width": [],
        "height": [],
        "max_detection_conf": [],
        "n_detections": [],
        "max_bbox_area_frac": [],
        "media_url": [],
        "media_relpath": [],
    }
    gt_records: dict[str, dict[str, Any]] = {}

    for img in images_list:
        iid = str(img["id"])
        fn = str(img["file_name"])
        dt_str = str(img.get("date_captured", ""))
        hour_val = -1
        if len(dt_str) >= 13 and dt_str[11:13].isdigit():
            hour_val = int(dt_str[11:13])
        max_conf, n_det, max_area = md_by_file.get(fn, (0.0, 0, 0.0))
        sp = ann_by_img.get(iid, "empty")
        is_animal = sp not in ("empty", "car", "unknown")

        out_cols["record_id"].append(iid)
        out_cols["location_id"].append(str(img.get("location", "")))
        out_cols["captured_at"].append(dt_str)
        out_cols["hour_local"].append(hour_val)
        out_cols["seq_id"].append(str(img.get("seq_id", "")))
        out_cols["seq_num_frames"].append(int(img.get("seq_num_frames", 1)))
        out_cols["frame_num"].append(int(img.get("frame_num", 1)))
        out_cols["width"].append(int(img.get("width", 0)))
        out_cols["height"].append(int(img.get("height", 0)))
        out_cols["max_detection_conf"].append(max_conf)
        out_cols["n_detections"].append(n_det)
        out_cols["max_bbox_area_frac"].append(max_area)
        out_cols["media_url"].append(f"{LILA_CCT_IMAGE_BASE_URL}/{fn}")
        out_cols["media_relpath"].append(precached_relpaths.get(iid, ""))

        gt_records[iid] = {
            "is_true_event": is_animal,
            "species": sp,
            "is_decoy": bool((not is_animal) and max_conf >= 0.50),
        }

    _check_disk_budget(cache_root, max_total_mb)
    table = pa.Table.from_pydict(out_cols)
    return _write_dataset_artifacts(
        ds_dir,
        dataset_id="caltech_camera_traps",
        table=table,
        ground_truth_records=gt_records,
        media_hashes=media_hashes,
    )


# ---------------------------------------------------------------------------
# 3. Environmental / Wildfire: Pyronear Lookout Network (pyronear/pyro-sdis)
# ---------------------------------------------------------------------------


def _compute_lookout_image_features(jpg_bytes: bytes) -> dict[str, Any]:
    """Compute deterministic luminance, horizon-contrast, haze, and glare features."""
    with Image.open(io.BytesIO(jpg_bytes)) as im:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = rgb.shape
    lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mid = max(1, h // 2)
    upper = lum[:mid, :]
    lower = lum[mid:, :]

    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    sat = np.where(cmax > 1e-4, (cmax - cmin) / np.maximum(cmax, 1e-4), 0.0)

    # Low-saturation moderate-to-high brightness pixels (elevated on BOTH smoke plumes & fog/clouds!)
    gray_haze = float(np.mean((lum > 0.45) & (lum < 0.92) & (sat < 0.18)))
    # Warm sun glare pixels
    warm_glare = float(np.mean((rgb[:, :, 0] > 0.80) & ((rgb[:, :, 0] - rgb[:, :, 2]) > 0.20)))

    return {
        "width": int(w),
        "height": int(h),
        "brightness_mean": round(float(np.mean(lum)), 5),
        "brightness_std": round(float(np.std(lum)), 5),
        "horizon_brightness": round(float(np.mean(upper)), 5),
        "sky_ground_contrast": round(float(np.mean(upper) - np.mean(lower)), 5),
        "gray_haze_fraction": round(gray_haze, 5),
        "warm_glare_fraction": round(warm_glare, 5),
    }


def stage_pyro_sdis(
    cache_root: Path,
    *,
    media_limit: int = 160,
    max_total_mb: float = 600.0,
) -> dict[str, Any]:
    """Stage Pyronear wildfire lookout tower validation split + optical features + JPEGs."""
    verify_safe_https_url(PYRO_SDIS_VAL_PARQUET_URL)
    ds_dir = cache_root / "pyro_sdis"
    media_dir = ds_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    con = _open_httpfs_duckdb()
    try:
        rows = con.execute(
            f"""
            SELECT
                image_name,
                partner,
                camera,
                date,
                annotations,
                image.bytes AS jpg_bytes
            FROM read_parquet('{PYRO_SDIS_VAL_PARQUET_URL}')
            """
        ).fetchall()
    finally:
        con.close()

    half = max(1, media_limit // 2)
    saved_smoke = 0
    saved_decoy = 0

    out_cols: dict[str, list[Any]] = {
        "record_id": [],
        "partner": [],
        "camera": [],
        "azimuth_deg": [],
        "captured_at": [],
        "hour_obs": [],
        "width": [],
        "height": [],
        "jpg_bytes": [],
        "brightness_mean": [],
        "brightness_std": [],
        "horizon_brightness": [],
        "sky_ground_contrast": [],
        "gray_haze_fraction": [],
        "warm_glare_fraction": [],
        "media_relpath": [],
    }
    gt_records: dict[str, dict[str, Any]] = {}
    media_hashes: dict[str, str] = {}

    for image_name, partner, camera, date_str, annotations, raw_blob in rows:
        jpg_bytes = bytes(raw_blob)
        verify_magic_bytes(jpg_bytes, ".jpg")
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(str(image_name)).stem)
        ann_str = (annotations or "").strip()
        has_smoke = len(ann_str) > 0
        n_boxes = len([ln for ln in ann_str.splitlines() if ln.strip()]) if has_smoke else 0

        az_match = re.search(r"-(\d{1,3})$", str(camera or ""))
        azimuth_deg = int(az_match.group(1)) if az_match else -1
        hr_match = re.search(r"T(\d{2})", str(date_str or ""))
        hour_obs = int(hr_match.group(1)) if hr_match else -1

        feats = _compute_lookout_image_features(jpg_bytes)

        relpath = ""
        should_save = (has_smoke and saved_smoke < half) or (
            (not has_smoke) and saved_decoy < half
        )
        if should_save:
            relpath = f"media/{stem}.jpg"
            out_file = verify_safe_output_path(ds_dir / relpath, ds_dir)
            out_file.write_bytes(jpg_bytes)
            media_hashes[relpath] = sha256_bytes(jpg_bytes)
            if has_smoke:
                saved_smoke += 1
            else:
                saved_decoy += 1

        out_cols["record_id"].append(stem)
        out_cols["partner"].append(str(partner or ""))
        out_cols["camera"].append(str(camera or ""))
        out_cols["azimuth_deg"].append(azimuth_deg)
        out_cols["captured_at"].append(str(date_str or ""))
        out_cols["hour_obs"].append(hour_obs)
        out_cols["width"].append(feats["width"])
        out_cols["height"].append(feats["height"])
        out_cols["jpg_bytes"].append(len(jpg_bytes))
        out_cols["brightness_mean"].append(feats["brightness_mean"])
        out_cols["brightness_std"].append(feats["brightness_std"])
        out_cols["horizon_brightness"].append(feats["horizon_brightness"])
        out_cols["sky_ground_contrast"].append(feats["sky_ground_contrast"])
        out_cols["gray_haze_fraction"].append(feats["gray_haze_fraction"])
        out_cols["warm_glare_fraction"].append(feats["warm_glare_fraction"])
        out_cols["media_relpath"].append(relpath)

        gt_records[stem] = {
            "is_true_event": has_smoke,
            "n_smoke_boxes": n_boxes,
            "is_decoy": not has_smoke,
            "ground_truth_class": "wildfire_smoke" if has_smoke else "false_alarm_decoy",
        }

    _check_disk_budget(cache_root, max_total_mb)
    table = pa.Table.from_pydict(out_cols)
    return _write_dataset_artifacts(
        ds_dir,
        dataset_id="pyro_sdis",
        table=table,
        ground_truth_records=gt_records,
        media_hashes=media_hashes,
    )


# ---------------------------------------------------------------------------
# 4. Urban Acoustic Sensors + Audio: UrbanSound8K (danavery/urbansound8K)
# ---------------------------------------------------------------------------


def _decode_riff_wav_mono(wav_bytes: bytes) -> tuple[np.ndarray, int, int]:
    """Safely parse RIFF/WAVE PCM (8/16/24/32-bit) or IEEE float32 into [-1, 1] mono float64."""
    verify_magic_bytes(wav_bytes, ".wav")
    pos = 12
    fmt_tag = 1
    channels = 1
    sample_rate = 44100
    bits_per_sample = 16
    data_buf = b""

    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
        chunk_start = pos + 8
        chunk_end = min(len(wav_bytes), chunk_start + chunk_size)
        if chunk_id == b"fmt " and chunk_size >= 16:
            fmt_tag, channels, sample_rate, _, _, bits_per_sample = struct.unpack_from(
                "<HHIIHH", wav_bytes, chunk_start
            )
        elif chunk_id == b"data":
            data_buf = wav_bytes[chunk_start:chunk_end]
            break
        pos = chunk_start + chunk_size + (chunk_size % 2)

    if not data_buf or channels < 1:
        return np.zeros(1024, dtype=np.float64), sample_rate, max(1, channels)

    if fmt_tag == 3 and bits_per_sample == 32:
        samples = np.frombuffer(data_buf, dtype="<f4").astype(np.float64)
    elif bits_per_sample == 16:
        samples = np.frombuffer(data_buf, dtype="<i2").astype(np.float64) / 32768.0
    elif bits_per_sample == 24:
        n_samp = len(data_buf) // 3
        raw3 = np.frombuffer(data_buf[: n_samp * 3], dtype=np.uint8).reshape(n_samp, 3)
        signed = (
            raw3[:, 0].astype(np.int32)
            | (raw3[:, 1].astype(np.int32) << 8)
            | (raw3[:, 2].astype(np.int8).astype(np.int32) << 16)
        )
        samples = signed.astype(np.float64) / 8388608.0
    elif bits_per_sample == 32:
        samples = np.frombuffer(data_buf, dtype="<i4").astype(np.float64) / 2147483648.0
    elif bits_per_sample == 8:
        samples = (np.frombuffer(data_buf, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        samples = np.zeros(1024, dtype=np.float64)

    if channels > 1 and samples.size >= channels:
        n_frames = samples.size // channels
        samples = samples[: n_frames * channels].reshape(n_frames, channels).mean(axis=1)

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(samples, -1.0, 1.0), int(sample_rate), int(channels)


def _compute_audio_features(mono: np.ndarray, sr: int) -> dict[str, float]:
    """Compute deterministic acoustic sensor telemetry from mono waveform."""
    if mono.size < 16:
        return {
            "rms_db": -90.0,
            "peak_amp": 0.0,
            "crest_factor_db": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid_hz": 0.0,
            "high_freq_energy_ratio": 0.0,
        }
    rms = float(np.sqrt(np.mean(np.square(mono))) + 1e-9)
    peak = float(np.max(np.abs(mono)) + 1e-9)
    rms_db = 20.0 * math.log10(rms)
    crest_db = 20.0 * math.log10(peak / rms)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(mono)))))

    # Windowed FFT spectral summary
    win_n = min(mono.size, 65536)
    seg = mono[:win_n] * np.hanning(win_n)
    mag = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(win_n, d=1.0 / max(1, sr))
    mag_sum = float(np.sum(mag)) + 1e-12
    centroid = float(np.sum(freqs * mag) / mag_sum)
    power = np.square(mag)
    hf_ratio = float(np.sum(power[freqs >= 2500.0]) / (float(np.sum(power)) + 1e-12))

    return {
        "rms_db": round(rms_db, 3),
        "peak_amp": round(peak, 5),
        "crest_factor_db": round(crest_db, 3),
        "zero_crossing_rate": round(zcr, 5),
        "spectral_centroid_hz": round(centroid, 2),
        "high_freq_energy_ratio": round(hf_ratio, 5),
    }


def _render_audio_diagnostic_png(mono: np.ndarray, sr: int, record_id: str) -> bytes:
    """Render deterministic 2-panel waveform + log-spectrogram PNG card for ARES vision probe."""
    fig = Figure(figsize=(6.4, 3.4), dpi=110)
    canvas = FigureCanvasAgg(fig)

    ax1 = fig.add_subplot(2, 1, 1)
    step = max(1, mono.size // 2000)
    t = np.arange(0, mono.size, step) / max(1, sr)
    ax1.plot(t, mono[::step], color="#1f77b4", lw=0.7)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_ylabel("Amp", fontsize=8)
    ax1.set_title(f"Acoustic Trigger {record_id} ({sr} Hz)", fontsize=9)

    ax2 = fig.add_subplot(2, 1, 2)
    if mono.size >= 512:
        with np.errstate(divide="ignore"):
            ax2.specgram(mono + 1e-7, NFFT=512, Fs=max(1, sr), noverlap=256, cmap="magma")
    ax2.set_ylim(0, min(8000, sr // 2))
    ax2.set_xlabel("Time (s)", fontsize=8)
    ax2.set_ylabel("Freq (Hz)", fontsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    canvas.print_png(buf)
    png_bytes = buf.getvalue()
    verify_magic_bytes(png_bytes, ".png")
    return png_bytes


def stage_urbansound8k(
    cache_root: Path,
    *,
    row_limit: int = 450,
    media_limit: int = 60,
    max_total_mb: float = 600.0,
) -> dict[str, Any]:
    """Stage UrbanSound8K acoustic triggers + verified `.wav` clips + spectrogram `.png`s."""
    verify_safe_https_url(URBANSOUND_PARQUET_URL)
    ds_dir = cache_root / "urbansound8k"
    media_dir = ds_dir / "media"
    wav_dir = ds_dir / "audio"
    media_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    con = _open_httpfs_duckdb()
    try:
        rows = con.execute(
            f"""
            SELECT
                slice_file_name,
                fsID,
                "start",
                "end",
                salience,
                fold,
                "class" AS sound_class,
                audio.bytes AS wav_bytes
            FROM read_parquet('{URBANSOUND_PARQUET_URL}')
            LIMIT {int(row_limit)}
            """
        ).fetchall()
    finally:
        con.close()

    emergency_classes = {"gun_shot", "siren"}
    decoy_classes = {"jackhammer", "car_horn", "dog_bark", "drilling"}
    half = max(1, media_limit // 2)
    saved_pos = 0
    saved_decoy = 0

    out_cols: dict[str, list[Any]] = {
        "record_id": [],
        "slice_file_name": [],
        "freesound_id": [],
        "start_s": [],
        "end_s": [],
        "duration_s": [],
        "salience": [],
        "fold": [],
        "sample_rate_hz": [],
        "channels": [],
        "rms_db": [],
        "peak_amp": [],
        "crest_factor_db": [],
        "zero_crossing_rate": [],
        "spectral_centroid_hz": [],
        "high_freq_energy_ratio": [],
        "wav_relpath": [],
        "media_relpath": [],
    }
    gt_records: dict[str, dict[str, Any]] = {}
    media_hashes: dict[str, str] = {}

    for slice_fn, fsid, start_s, end_s, salience, fold, sound_cls, raw_wav in rows:
        wav_bytes = bytes(raw_wav)
        mono, sr, ch = _decode_riff_wav_mono(wav_bytes)
        feats = _compute_audio_features(mono, sr)
        rid = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(str(slice_fn)).stem)
        cls_name = str(sound_cls)
        is_emergency = cls_name in emergency_classes
        is_decoy = (not is_emergency) and (
            cls_name in decoy_classes or feats["crest_factor_db"] >= 14.0
        )

        wav_rel = ""
        png_rel = ""
        should_save = (is_emergency and saved_pos < half) or (is_decoy and saved_decoy < half)
        if should_save:
            wav_rel = f"audio/{rid}.wav"
            wav_file = verify_safe_output_path(ds_dir / wav_rel, ds_dir)
            wav_file.write_bytes(wav_bytes)
            media_hashes[wav_rel] = sha256_bytes(wav_bytes)

            png_bytes = _render_audio_diagnostic_png(mono, sr, rid)
            png_rel = f"media/{rid}.png"
            png_file = verify_safe_output_path(ds_dir / png_rel, ds_dir)
            png_file.write_bytes(png_bytes)
            media_hashes[png_rel] = sha256_bytes(png_bytes)

            if is_emergency:
                saved_pos += 1
            else:
                saved_decoy += 1

        dur = round(max(0.0, float(end_s) - float(start_s)), 4)
        out_cols["record_id"].append(rid)
        out_cols["slice_file_name"].append(str(slice_fn))
        out_cols["freesound_id"].append(int(fsid))
        out_cols["start_s"].append(round(float(start_s), 4))
        out_cols["end_s"].append(round(float(end_s), 4))
        out_cols["duration_s"].append(dur)
        out_cols["salience"].append(int(salience))
        out_cols["fold"].append(int(fold))
        out_cols["sample_rate_hz"].append(sr)
        out_cols["channels"].append(ch)
        out_cols["rms_db"].append(feats["rms_db"])
        out_cols["peak_amp"].append(feats["peak_amp"])
        out_cols["crest_factor_db"].append(feats["crest_factor_db"])
        out_cols["zero_crossing_rate"].append(feats["zero_crossing_rate"])
        out_cols["spectral_centroid_hz"].append(feats["spectral_centroid_hz"])
        out_cols["high_freq_energy_ratio"].append(feats["high_freq_energy_ratio"])
        out_cols["wav_relpath"].append(wav_rel)
        out_cols["media_relpath"].append(png_rel)

        gt_records[rid] = {
            "is_true_event": is_emergency,
            "sound_class": cls_name,
            "is_decoy": is_decoy,
        }

    _check_disk_budget(cache_root, max_total_mb)
    table = pa.Table.from_pydict(out_cols)
    return _write_dataset_artifacts(
        ds_dir,
        dataset_id="urbansound8k",
        table=table,
        ground_truth_records=gt_records,
        media_hashes=media_hashes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely download and stage curated multimodal datasets into ./.data_cache/"
    )
    parser.add_argument(
        "--dataset",
        choices=[*sorted(DATASET_CATALOG.keys()), "all"],
        default="all",
        help="Dataset ID to stage into --cache-dir (default: all)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".data_cache",
        help="Disposable directory for staged datasets (default: ./.data_cache)",
    )
    parser.add_argument(
        "--media-limit",
        type=int,
        default=60,
        help="Max verified media assets to pre-cache per dataset (default: 60)",
    )
    parser.add_argument(
        "--max-total-mb",
        type=float,
        default=600.0,
        help="Hard safety disk cap in MB across --cache-dir (default: 600 MB)",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    targets = sorted(DATASET_CATALOG.keys()) if args.dataset == "all" else [args.dataset]
    summaries: list[dict[str, Any]] = []

    for ds_id in targets:
        print(f"[stage] Staging {ds_id} into {cache_root / ds_id} ...", flush=True)
        if ds_id == "btsbot":
            manifest = stage_btsbot(
                cache_root,
                media_limit=args.media_limit,
                max_total_mb=args.max_total_mb,
            )
        elif ds_id == "caltech_camera_traps":
            manifest = stage_caltech_camera_traps(
                cache_root,
                media_limit=min(args.media_limit, 36),
                max_total_mb=args.max_total_mb,
            )
        elif ds_id == "pyro_sdis":
            manifest = stage_pyro_sdis(
                cache_root,
                media_limit=args.media_limit,
                max_total_mb=args.max_total_mb,
            )
        elif ds_id == "urbansound8k":
            manifest = stage_urbansound8k(
                cache_root,
                media_limit=args.media_limit,
                max_total_mb=args.max_total_mb,
            )
        else:
            raise ValueError(f"Unhandled dataset: {ds_id}")

        mb = manifest["total_dataset_bytes"] / (1024 * 1024)
        print(
            f"[done]  {ds_id}: {manifest['row_count']} rows, "
            f"{manifest['staged_media_files']} verified media files, "
            f"{manifest['positive_count']} positives / {manifest['decoy_count']} decoys "
            f"({mb:.1f} MB)",
            flush=True,
        )
        summaries.append(
            {
                "dataset_id": ds_id,
                "rows": manifest["row_count"],
                "media_files": manifest["staged_media_files"],
                "positives": manifest["positive_count"],
                "decoys": manifest["decoy_count"],
                "size_mb": round(mb, 2),
            }
        )

    total_mb = _directory_size_bytes(cache_root) / (1024 * 1024)
    print(f"\nAll requested datasets staged safely in {cache_root} ({total_mb:.1f} MB total).")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
