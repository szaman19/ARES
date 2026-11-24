"""Scripted adversarial & Element-119 stress-test video transforms (PLAN §14.3 / §15).

Generates objective, un-cherry-picked perturbations from real unseen clips:
- Lighting ladder: `darken_mild` (gamma 0.55), `darken_severe` (brightness -0.45, gamma 0.35)
- Motion / blur ladder: `blur_mild` (boxblur 4:1), `blur_severe` (boxblur 10:2)
- Temporal / speed ladder: `speed_2x` (0.5*PTS + atempo=2.0)
- Audio messiness ladder: `audio_noise` (mix pink/white ambient crowd noise over narration)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from ingest.frames import ffmpeg_exe

TransformKind = Literal[
    "darken_mild",
    "darken_severe",
    "blur_mild",
    "blur_severe",
    "speed_2x",
    "audio_noise",
]


def apply_video_transform(
    src_video: str | Path,
    dst_video: str | Path,
    kind: TransformKind,
) -> str:
    """Apply a deterministic ffmpeg video/audio perturbation and write `dst_video`."""
    src = Path(src_video)
    dst = Path(dst_video)
    dst.parent.mkdir(parents=True, exist_ok=True)
    exe = ffmpeg_exe()

    if kind == "darken_mild":
        cmd = [
            exe, "-y", "-loglevel", "error", "-i", str(src),
            "-vf", "eq=brightness=-0.22:gamma=0.60",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(dst),
        ]
    elif kind == "darken_severe":
        cmd = [
            exe, "-y", "-loglevel", "error", "-i", str(src),
            "-vf", "eq=brightness=-0.42:gamma=0.35",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(dst),
        ]
    elif kind == "blur_mild":
        cmd = [
            exe, "-y", "-loglevel", "error", "-i", str(src),
            "-vf", "boxblur=4:1",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(dst),
        ]
    elif kind == "blur_severe":
        cmd = [
            exe, "-y", "-loglevel", "error", "-i", str(src),
            "-vf", "boxblur=11:2",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "copy", str(dst),
        ]
    elif kind == "speed_2x":
        cmd = [
            exe, "-y", "-loglevel", "error", "-i", str(src),
            "-filter_complex", "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(dst),
        ]
    elif kind == "audio_noise":
        cmd = [
            exe, "-y", "-loglevel", "error", "-i", str(src),
            "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.08",
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", str(dst),
        ]
    else:
        raise ValueError(f"Unsupported transform kind: {kind}")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Fallback for silent input videos on speed_2x / audio_noise
        if kind in {"speed_2x", "audio_noise"}:
            vf = "setpts=0.5*PTS" if kind == "speed_2x" else "null"
            fallback_cmd = [
                exe, "-y", "-loglevel", "error", "-i", str(src),
                "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", str(dst),
            ]
            subprocess.run(fallback_cmd, check=True)
        else:
            raise RuntimeError(f"Transform {kind} failed: {proc.stderr.strip()}")
    return str(dst)
