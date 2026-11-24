"""Deterministic video frame extraction, perceptual hash deduplication, and quality scoring.

No LLM calls occur in this module. Uses ffmpeg (system PATH or imageio-ffmpeg)
plus Pillow and NumPy for 64-bit dHash, Laplacian sharpness variance, and luminance.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from contracts import Frame


def ffmpeg_exe() -> str:
    """Resolve ffmpeg binary from PATH or bundled imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # type: ignore

    return imageio_ffmpeg.get_ffmpeg_exe()


def compute_dhash_hex(img: Image.Image) -> str:
    """Compute 64-bit difference hash (dHash) as a 16-char lowercase hex string."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return f"{val:016x}"


def hamming_distance_hex(h1: str, h2: str) -> int:
    """Bitwise Hamming distance between two 64-bit hex perceptual hashes."""
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def compute_blur_and_brightness(img: Image.Image) -> tuple[float, float]:
    """Return (laplacian_variance, mean_brightness_0_255) on grayscale image."""
    # Standardize evaluation resolution so blur scores are comparable across phones/webcams
    gray = img.convert("L")
    w, h = gray.size
    if max(w, h) > 640:
        scale = 640.0 / max(w, h)
        gray = gray.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.float32)
    brightness = float(arr.mean())
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0, brightness
    # Discrete 2D Laplacian: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    lap = (
        arr[:-2, 1:-1]
        + arr[2:, 1:-1]
        + arr[1:-1, :-2]
        + arr[1:-1, 2:]
        - 4.0 * arr[1:-1, 1:-1]
    )
    blur_score = float(lap.var())
    return blur_score, brightness


def extract_frames(
    video_path: str | Path,
    out_dir: str | Path,
    video_id: str | None = None,
    sample_fps: float = 2.0,
    phash_hamming_thresh: int = 6,
    scene_jump_bits: int = 18,
    max_frames: int = 48,
    blur_min: float = 15.0,
    brightness_min: float = 20.0,
) -> tuple[list[Frame], int]:
    """Extract keyframes + scene shifts, deduplicate by perceptual hash, and score quality.

    Returns `(kept_frames, raw_extracted_count)`.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video not found: {vpath}")
    vid = video_id or vpath.stem
    out = Path(out_dir)
    raw_dir = out / "raw_frames"
    kept_dir = out / "frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    kept_dir.mkdir(parents=True, exist_ok=True)

    # Extract uniform candidate stream at sample_fps (e.g. 2 fps -> 500 ms step)
    cmd = [
        ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(vpath),
        "-vf",
        f"fps={sample_fps},scale='min(1280,iw)':-2",
        "-q:v",
        "3",
        str(raw_dir / "cand_%06d.jpg"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr.strip()}")

    cand_files = sorted(raw_dir.glob("cand_*.jpg"))
    raw_count = len(cand_files)
    if raw_count == 0:
        # Fallback: single frame grab for ultra-short clips
        cmd_single = [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(vpath),
            "-frames:v",
            "1",
            str(raw_dir / "cand_000001.jpg"),
        ]
        subprocess.run(cmd_single, check=False)
        cand_files = sorted(raw_dir.glob("cand_*.jpg"))
        raw_count = len(cand_files)

    step_ms = int(round(1000.0 / max(sample_fps, 0.1)))
    kept: list[Frame] = []
    last_kept_hash: str | None = None
    last_kept_ts_ms: int = -10_000

    for idx, fpath in enumerate(cand_files):
        ts_ms = idx * step_ms
        with Image.open(fpath) as im:
            phash = compute_dhash_hex(im)
            blur, bright = compute_blur_and_brightness(im)

        usable = blur >= blur_min and bright >= brightness_min

        if last_kept_hash is not None:
            dist = hamming_distance_hex(phash, last_kept_hash)
            # Keep if scene jumped sharply OR >=1s elapsed and not a near-duplicate
            is_scene_jump = dist >= scene_jump_bits
            is_cadence_tick = (ts_ms - last_kept_ts_ms >= 950) and (dist > phash_hamming_thresh)
            if not (is_scene_jump or is_cadence_tick):
                # However, if the previous frame was blurry and this frame is much sharper of same view, replace!
                if kept and not kept[-1].usable and usable and dist <= phash_hamming_thresh + 4:
                    prev = kept.pop()
                    Path(prev.path).unlink(missing_ok=True)
                else:
                    continue

        dest = kept_dir / f"{vid}_{ts_ms:07d}ms.jpg"
        shutil.copy2(fpath, dest)
        frame = Frame(
            frame_id=f"{vid}@{ts_ms}",
            video_id=vid,
            ts_ms=ts_ms,
            path=str(dest),
            phash=phash,
            blur_score=round(blur, 2),
            brightness=round(bright, 2),
            usable=usable,
        )
        kept.append(frame)
        last_kept_hash = phash
        last_kept_ts_ms = ts_ms

    # Clean up raw intermediate frames to save disk
    shutil.rmtree(raw_dir, ignore_errors=True)

    # Cap total frames for long walkthroughs while preserving timeline coverage
    if len(kept) > max_frames:
        indices = np.linspace(0, len(kept) - 1, max_frames, dtype=int)
        selected_set = set(int(i) for i in indices)
        pruned: list[Frame] = []
        for i, fr in enumerate(kept):
            if i in selected_set:
                pruned.append(fr)
            else:
                Path(fr.path).unlink(missing_ok=True)
        kept = pruned

    return kept, raw_count


def crop_bbox(
    frame_path: str | Path,
    bbox: tuple[float, float, float, float] | None,
    out_path: str | Path,
    padding_ratio: float = 0.08,
) -> str:
    """Crop normalized (x, y, w, h) region from frame_path with slight context padding."""
    src = Path(frame_path)
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if bbox is None:
            im.convert("RGB").save(dst, quality=90)
            return str(dst)
        W, H = im.size
        x, y, bw, bh = bbox
        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio
        x0 = int(max(0, (x - pad_x) * W))
        y0 = int(max(0, (y - pad_y) * H))
        x1 = int(min(W, (x + bw + pad_x) * W))
        y1 = int(min(H, (y + bh + pad_y) * H))
        if x1 - x0 < 16 or y1 - y0 < 16:
            im.convert("RGB").save(dst, quality=90)
        else:
            im.crop((x0, y0, x1, y1)).convert("RGB").save(dst, quality=92)
    return str(dst)
