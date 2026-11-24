"""Deterministic audio extraction to 16 kHz mono WAV via ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ingest.frames import ffmpeg_exe


def extract_audio_wav(
    video_path: str | Path,
    out_wav_path: str | Path,
    sample_rate: int = 16000,
) -> str | None:
    """Extract mono PCM WAV audio track from video.

    Returns path string if audio exists and succeeded, or `None` if the clip is silent/no-audio.
    """
    vpath = Path(video_path)
    out = Path(out_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(vpath),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 100:
        out.unlink(missing_ok=True)
        return None
    return str(out)
