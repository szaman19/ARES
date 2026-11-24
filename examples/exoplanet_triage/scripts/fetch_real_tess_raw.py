"""Download real NASA Exoplanet Archive TOI catalog + real TESS S3 FITS files.

Zero authentication required:
1. Queries NASA Exoplanet Archive TAP (`SELECT * FROM toi`) for real human-vetted
   TESS dispositions:
     - `CP` / `KP` (Confirmed / Known Planets)
     - `FP` / `FA` (TFOP Vetted Astrophysical False Positives / Impostors)
     - `PC` / `APC` (Unconfirmed / Ambiguous Planetary Candidates)
2. Lists public TESS SPOC 2-minute cadence FITS keys from AWS Open Data
   (`https://stpubdata.s3.amazonaws.com`, bucket `stpubdata`, prefix `tess/public/tid/`).
3. Downloads matched `(_lc.fits, _tp.fits)` pairs across all 3 categories plus
   real non-TOI background TESS sector `_lc.fits` files into `data/tess_raw/`.
"""

from __future__ import annotations

import concurrent.futures
import csv
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.exoplanet_triage.fits_utils import read_fits_hdus

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
S3_BASE = "https://stpubdata.s3.amazonaws.com"
OUT_RAW_DIR = REPO_ROOT / "data" / "tess_raw"

# Sectors rich in confirmed planets, vetted eclipsing-binary FPs, and Kepler-field overlaps
TARGET_SECTORS = ("s0001", "s0002", "s0014", "s0026")

_LC_RE = re.compile(r"tess\d+-s(\d{4})-(\d{16})-\d+-s_lc\.fits$")
_TP_RE = re.compile(r"tess\d+-s(\d{4})-(\d{16})-\d+-s_tp\.fits$")


def _http_get_bytes(url: str, timeout: float = 30.0, retries: int = 3) -> bytes:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ARES-Exoplanet-Benchmark/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(0.8 * (attempt + 1))
    assert last_err is not None
    raise last_err


def fetch_toi_catalog() -> list[dict[str, Any]]:
    """Fetch the full NASA Exoplanet Archive TOI table via public TAP CSV."""
    cache_path = OUT_RAW_DIR / "nasa_toi_catalog.json"
    if cache_path.is_file():
        print(f"[TAP] Using cached TOI catalog: {cache_path}")
        return json.loads(cache_path.read_text())

    params = urllib.parse.urlencode(
        {
            "query": "select * from toi",
            "format": "csv",
        }
    )
    url = f"{TAP_URL}?{params}"
    print(f"[TAP] Downloading NASA Exoplanet Archive TOI catalog...")
    raw_csv = _http_get_bytes(url, timeout=60.0).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw_csv))
    rows = list(reader)
    print(f"[TAP] Retrieved {len(rows)} real TOI rows.")
    OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(rows))
    return rows


def list_s3_sector_files(sector: str, max_pages: int = 85) -> tuple[dict[int, str], dict[int, str]]:
    """List `_lc.fits` and `_tp.fits` S3 keys for a given TESS sector (`s0014`, etc.)."""
    cache_file = OUT_RAW_DIR / f"s3_index_{sector}.json"
    if cache_file.is_file():
        cached = json.loads(cache_file.read_text())
        return (
            {int(k): v for k, v in cached["lc"].items()},
            {int(k): v for k, v in cached["tp"].items()},
        )

    lc_map: dict[int, str] = {}
    tp_map: dict[int, str] = {}
    prefix = f"tess/public/tid/{sector}/"
    token: str | None = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    print(f"[S3] Listing TESS SPOC files in {sector}...")
    for _page in range(max_pages):
        q: dict[str, str] = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = f"{S3_BASE}/?{urllib.parse.urlencode(q)}"
        xml_bytes = _http_get_bytes(url, timeout=30.0)
        root = ET.fromstring(xml_bytes)

        for contents in root.findall("s3:Contents", ns):
            key_el = contents.find("s3:Key", ns)
            if key_el is None or not key_el.text:
                continue
            key = key_el.text
            m_lc = _LC_RE.search(key)
            if m_lc:
                lc_map[int(m_lc.group(2))] = key
                continue
            m_tp = _TP_RE.search(key)
            if m_tp:
                tp_map[int(m_tp.group(2))] = key

        is_trunc = root.find("s3:IsTruncated", ns)
        if is_trunc is None or is_trunc.text != "true":
            break
        next_tok = root.find("s3:NextContinuationToken", ns)
        if next_tok is None or not next_tok.text:
            break
        token = next_tok.text

    cache_file.write_text(
        json.dumps(
            {
                "lc": {str(k): v for k, v in lc_map.items()},
                "tp": {str(k): v for k, v in tp_map.items()},
            }
        )
    )
    print(f"[S3] {sector}: indexed {len(lc_map)} _lc.fits and {len(tp_map)} _tp.fits.")
    return lc_map, tp_map


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        f = float(val)
        return f
    except (ValueError, TypeError):
        return None


def _download_one(key: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 2880:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{S3_BASE}/{key}"
    data = _http_get_bytes(url, timeout=45.0)
    dest.write_bytes(data)
    return dest


def main() -> None:
    OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    toi_rows = fetch_toi_catalog()

    # Index TOIs by TIC ID (pick highest-SNR / cleanest entry per TIC)
    toi_by_tid: dict[int, dict[str, Any]] = {}
    for row in toi_rows:
        tid_raw = row.get("tid")
        disp = (row.get("tfopwg_disp") or "").strip().upper()
        per = _safe_float(row.get("pl_orbper"))
        dur = _safe_float(row.get("pl_trandurh"))
        dep = _safe_float(row.get("pl_trandep"))
        t0 = _safe_float(row.get("pl_tranmid"))
        srad = _safe_float(row.get("st_rad"))
        slogg = _safe_float(row.get("st_logg"))
        tmag = _safe_float(row.get("st_tmag"))

        if not tid_raw or disp not in ("CP", "KP", "FP", "FA", "PC", "APC"):
            continue
        if per is None or not (0.8 <= per <= 11.5):
            continue
        if dur is None or not (0.4 <= dur <= 10.0):
            continue
        if dep is None or dep < 180.0:
            continue
        if t0 is None or srad is None or slogg is None or tmag is None:
            continue
        if tmag > 13.5:
            continue

        tid = int(tid_raw)
        # Prefer larger transit depth if multiple planet candidates on same host
        prev = toi_by_tid.get(tid)
        if prev is None or float(dep) > float(prev["pl_trandep"]):
            toi_by_tid[tid] = {
                "tid": tid,
                "toi": row.get("toi"),
                "tfopwg_disp": disp,
                "pl_orbper": per,
                "pl_tranmid": t0,
                "pl_trandurh": dur,
                "pl_trandep": dep,
                "pl_rade": _safe_float(row.get("pl_rade")),
                "st_tmag": tmag,
                "st_rad": srad,
                "st_logg": slogg,
                "st_teff": _safe_float(row.get("st_teff")) or 5500.0,
                "ra": _safe_float(row.get("ra")),
                "dec": _safe_float(row.get("dec")),
            }

    print(f"[FILTER] {len(toi_by_tid)} quality short-period TOIs across CP/KP, FP, PC/APC.")

    # Index S3 sectors in parallel
    sector_indices: dict[str, tuple[dict[int, str], dict[int, str]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(list_s3_sector_files, sec): sec for sec in TARGET_SECTORS}
        for fut in concurrent.futures.as_completed(futs):
            sec = futs[fut]
            sector_indices[sec] = fut.result()

    # Match TOIs that have BOTH _lc.fits and _tp.fits in any indexed sector
    matched_by_group: dict[str, list[dict[str, Any]]] = {
        "planet": [],
        "impostor": [],
        "candidate": [],
    }
    seen_tids: set[int] = set()
    background_candidates: list[tuple[int, str, str]] = []

    for sec in TARGET_SECTORS:
        lc_map, tp_map = sector_indices[sec]
        common_tids = sorted(set(lc_map.keys()) & set(tp_map.keys()))
        for tid in common_tids:
            if tid in toi_by_tid and tid not in seen_tids:
                meta = dict(toi_by_tid[tid])
                meta["sector"] = int(sec[1:])
                meta["sector_tag"] = sec
                meta["lc_key"] = lc_map[tid]
                meta["tp_key"] = tp_map[tid]
                disp = meta["tfopwg_disp"]
                if disp in ("CP", "KP"):
                    matched_by_group["planet"].append(meta)
                elif disp in ("FP", "FA"):
                    matched_by_group["impostor"].append(meta)
                elif disp in ("PC", "APC"):
                    matched_by_group["candidate"].append(meta)
                seen_tids.add(tid)
            elif tid not in toi_by_tid and len(background_candidates) < 180:
                # Deterministic stride sampling of quiet non-TOI stars
                if tid % 97 == 0:
                    background_candidates.append((tid, sec, lc_map[tid]))

    for grp, items in matched_by_group.items():
        print(f"[MATCH] Group '{grp}': {len(items)} real TESS TOIs with (_lc.fits + _tp.fits)")

    # Sort planets by high SNR (depth / bright star), impostors by loud transit depth, candidates by varied depth
    matched_by_group["planet"].sort(
        key=lambda m: (m["pl_trandep"] / max(1.0, 10 ** (0.2 * (m["st_tmag"] - 8.0)))),
        reverse=True,
    )
    matched_by_group["impostor"].sort(key=lambda m: m["pl_trandep"], reverse=True)
    matched_by_group["candidate"].sort(key=lambda m: m["pl_trandep"], reverse=True)

    selected_targets = (
        matched_by_group["planet"][:18]
        + matched_by_group["impostor"][:22]
        + matched_by_group["candidate"][:18]
    )
    bg_selected = background_candidates[:120]

    download_jobs: list[tuple[str, Path]] = []
    for meta in selected_targets:
        tid = meta["tid"]
        sec = meta["sector_tag"]
        lc_dest = OUT_RAW_DIR / "targets" / f"tic_{tid}_{sec}_lc.fits"
        tp_dest = OUT_RAW_DIR / "targets" / f"tic_{tid}_{sec}_tp.fits"
        meta["lc_local"] = str(lc_dest.relative_to(REPO_ROOT))
        meta["tp_local"] = str(tp_dest.relative_to(REPO_ROOT))
        download_jobs.append((meta["lc_key"], lc_dest))
        download_jobs.append((meta["tp_key"], tp_dest))

    bg_manifest: list[dict[str, Any]] = []
    for tid, sec, lc_key in bg_selected:
        lc_dest = OUT_RAW_DIR / "background" / f"tic_{tid}_{sec}_lc.fits"
        bg_manifest.append(
            {
                "tid": tid,
                "sector": int(sec[1:]),
                "sector_tag": sec,
                "lc_local": str(lc_dest.relative_to(REPO_ROOT)),
            }
        )
        download_jobs.append((lc_key, lc_dest))

    print(f"[DOWNLOAD] Fetching {len(download_jobs)} real NASA FITS files via 12 parallel workers...")
    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        fut_map = {pool.submit(_download_one, key, dest): (key, dest) for key, dest in download_jobs}
        for fut in concurrent.futures.as_completed(fut_map):
            try:
                fut.result()
                completed += 1
                if completed % 25 == 0 or completed == len(download_jobs):
                    print(f"  progress: {completed}/{len(download_jobs)} FITS files downloaded")
            except Exception as exc:
                failed += 1
                key, _ = fut_map[fut]
                print(f"  WARN failed {key}: {exc}")

    # Verify readability of a sample target using pure-numpy reader
    valid_targets: list[dict[str, Any]] = []
    for meta in selected_targets:
        lc_p = REPO_ROOT / meta["lc_local"]
        tp_p = REPO_ROOT / meta["tp_local"]
        if not (lc_p.is_file() and tp_p.is_file()):
            continue
        try:
            lc_hdus = read_fits_hdus(lc_p)
            tp_hdus = read_fits_hdus(tp_p)
            if len(lc_hdus) >= 2 and len(tp_hdus) >= 3 and lc_hdus[1]["data"] is not None:
                # Record SPOC CROWDSAP & FLFRCSAP directly from NASA FITS header
                h1 = lc_hdus[1]["header"]
                meta["crowdsap"] = _safe_float(h1.get("CROWDSAP"))
                meta["flfrcsap"] = _safe_float(h1.get("FLFRCSAP"))
                valid_targets.append(meta)
        except Exception as exc:
            print(f"  WARN skipping corrupt FITS for TIC {meta['tid']}: {exc}")

    valid_bg = [
        b for b in bg_manifest if (REPO_ROOT / b["lc_local"]).is_file()
    ]

    manifest_path = OUT_RAW_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "targets": valid_targets,
                "background": valid_bg,
            },
            indent=2,
        )
    )
    print(
        f"[DONE] Verified {len(valid_targets)} real TOI targets (with _lc + _tp FITS) "
        f"and {len(valid_bg)} real background sector stars -> {manifest_path} (failed={failed})"
    )


if __name__ == "__main__":
    main()
