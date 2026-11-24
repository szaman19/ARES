"""Capture a real, timestamped snapshot of NYC DOT street-intersection cameras for offline fallback / replay.

Usage:
    uv run python -m examples.nyc_cameras.capture_snapshot --areas Manhattan Brooklyn --per-area 18 --out-dir examples/nyc_cameras/fixtures/nyc_snapshot
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from examples.nyc_cameras.domain import (
    CameraInfo,
    fetch_camera_catalog,
    fetch_camera_jpeg,
    select_spread_cameras,
)


def capture_snapshot(
    *,
    areas: list[str],
    per_area: int = 18,
    out_dir: str | Path = "examples/nyc_cameras/fixtures/nyc_snapshot",
    max_concurrent: int = 6,
) -> Path:
    dest = Path(out_dir)
    frames_dir = dest / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    selected: list[CameraInfo] = []
    for area in areas:
        cams = fetch_camera_catalog(area=area, street_only=True)
        picked = select_spread_cameras(cams, k=per_area)
        selected.extend(picked)

    records: list[dict[str, Any]] = []

    def _grab(cam: CameraInfo) -> dict[str, Any] | None:
        try:
            art = fetch_camera_jpeg(cam, frames_dir)
            return {
                "camera": cam.model_dump(mode="json"),
                "frame_file": Path(art.path).name,
                "sha256": art.sha256,
                "artifact_id": art.artifact_id,
            }
        except Exception as exc:
            print(f"  [skip] {cam.name}: {exc}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        for rec in pool.map(_grab, selected):
            if rec is not None:
                records.append(rec)

    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "areas": areas,
        "count": len(records),
        "cameras": records,
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Captured {len(records)} live NYC street-intersection JPEGs -> {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a real NYC DOT street-camera snapshot.")
    parser.add_argument(
        "--areas",
        nargs="+",
        default=["Manhattan", "Brooklyn"],
        help="NYC boroughs to sample.",
    )
    parser.add_argument(
        "--per-area",
        type=int,
        default=18,
        help="Number of spatially spread street intersections per borough.",
    )
    parser.add_argument(
        "--out-dir",
        default="examples/nyc_cameras/fixtures/nyc_snapshot",
        help="Destination directory for manifest.json and frames/.",
    )
    args = parser.parse_args()
    capture_snapshot(areas=args.areas, per_area=args.per_area, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
