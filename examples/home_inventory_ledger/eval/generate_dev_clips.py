"""Generate local development/demo sample clips & policy declarations image.

IMPORTANT COMPLIANCE NOTE (Gemini 4 Hackathon Rule 1 & 5):
These generated clips live strictly in `eval/dev_clips/` for local engineering tests,
CI verification, and disclosed offline `--replay` backup. They are NEVER included in
the 5 Sigma held-out stranger evaluation manifest (`eval/test_manifest.json`).
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import tempfile
import wave

from PIL import Image, ImageDraw, ImageFont

from ingest.frames import ffmpeg_exe


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        ("/System/Library/Fonts/HelveticaNeue.ttc", 1 if bold else 0),
        ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    ]
    for p, idx in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=idx)
            except Exception:
                continue
    return ImageFont.load_default()


def generate_sample_declarations_page(out_path: str | Path, limit_usd: int = 8000, deductible_usd: int = 500) -> str:
    """Render a clearly-watermarked SAMPLE insurance declarations page image."""
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (900, 1160), (252, 250, 245))
    d = ImageDraw.Draw(im)

    d.rectangle((40, 40, 860, 1120), outline=(40, 50, 75), width=3)
    d.text((70, 70), "SAMPLE HOMEOWNERS / RENTERS POLICY", font=_font(28, True), fill=(20, 28, 48))
    d.text((70, 110), "DECLARATIONS PAGE — FOR DEMO & TESTING ONLY (NOT A REAL POLICY)", font=_font(16, True), fill=(190, 95, 15))
    d.line((70, 150, 830, 150), fill=(180, 186, 200), width=2)

    info_rows = [
        ("Policy Number:", "HO-SAMPLE-2026-NYC"),
        ("Named Insured:", "Sample Policyholder (Demo)"),
        ("Property Location:", "220 W 26th St, Apt 4B, New York, NY 10001"),
        ("Policy Period:", "01/01/2026 to 01/01/2027"),
    ]
    y = 180
    for k, v in info_rows:
        d.text((70, y), k, font=_font(18, True), fill=(55, 62, 80))
        d.text((290, y), v, font=_font(18, False), fill=(20, 28, 48))
        y += 38

    d.text((70, 360), "SECTION I — PROPERTY COVERAGES & LIMITS OF LIABILITY", font=_font(20, True), fill=(20, 28, 48))
    d.rectangle((70, 400, 830, 680), outline=(160, 168, 185), width=2)

    cov_rows = [
        ("Coverage A — Dwelling", "N/A (Renters / Condo Form)"),
        ("Coverage B — Other Structures", "$0"),
        ("Coverage C — Personal Property (Contents Replacement)", f"${limit_usd:,}"),
        ("Coverage D — Loss of Use", "$3,200"),
        ("Section I All-Perils Deductible", f"${deductible_usd:,}"),
    ]
    cy = 424
    for name, val in cov_rows:
        is_cov_c = "Coverage C" in name
        if is_cov_c:
            d.rectangle((76, cy - 8, 824, cy + 36), fill=(228, 240, 255))
        d.text((90, cy), name, font=_font(19, is_cov_c), fill=(15, 24, 42))
        d.text((640, cy), val, font=_font(21, True), fill=(15, 70, 165) if is_cov_c else (30, 36, 52))
        cy += 50

    d.text(
        (70, 730),
        "Valuation Clause: Replacement Cost Value (RCV) on documented personal property.\n"
        "High-value jewelry, fine art, and antiques subject to scheduled appraisal sub-limits.",
        font=_font(16, False),
        fill=(80, 88, 106),
    )
    im.save(dst, quality=92)
    return str(dst)


def _synthesize_narration_wav(out_wav: Path, duration_s: float, text: str) -> None:
    """Create spoken narration WAV via macOS `say` if available, else silent/tone WAV."""
    rate = 16000
    say_bin = "/usr/bin/say"
    if Path(say_bin).exists():
        try:
            tmp_wav = out_wav.with_suffix(".raw.wav")
            subprocess.run(
                [say_bin, "-v", "Samantha", "-r", "175", "-o", str(tmp_wav), "--data-format=LEI16@16000", text],
                check=True,
                capture_output=True,
            )
            with wave.open(str(tmp_wav), "rb") as r:
                frames = r.readframes(r.getnframes())
            tmp_wav.unlink(missing_ok=True)
            target_samples = int(duration_s * rate)
            raw_bytes = frames[: target_samples * 2]
            pad_samples = max(0, target_samples - len(raw_bytes) // 2)
            with wave.open(str(out_wav), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(raw_bytes + b"\x00\x00" * pad_samples)
            return
        except Exception:
            pass

    # Fallback pure PCM wave if `say` unavailable
    target_samples = int(duration_s * rate)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * target_samples)


def generate_dev_walkthrough_video(out_video: str | Path, duration_s: float = 10.0, fps: int = 12) -> str:
    """Render a 10s panning desk video with legible labels + spoken contradiction ('Sony' on Bose)."""
    dst = Path(out_video)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Wide 2200x800 canvas with crisp labels on Monitor & Camera, unlabeled MacBook, and Bose logo
    CW, CH = 2200, 820
    canvas = Image.new("RGB", (CW, CH), (212, 218, 228))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 520, CW, CH), fill=(148, 112, 78))
    d.rectangle((0, 520, CW, 534), fill=(112, 82, 56))

    # 1. Dell 27" Monitor with crisp model badge U2723QE
    d.rectangle((180, 110, 740, 440), fill=(26, 28, 34))
    d.rectangle((196, 126, 724, 420), fill=(48, 78, 128))
    d.text((425, 422), "DELL  U2723QE", font=_font(16, True), fill=(230, 235, 245))
    d.rectangle((430, 440, 490, 520), fill=(72, 76, 86))
    d.rectangle((360, 515, 560, 535), fill=(82, 86, 96))

    # 2. Apple MacBook Pro (top lid / open keyboard — NO model number visible yet!)
    d.polygon([(560, 740), (980, 740), (940, 620), (600, 620)], fill=(176, 180, 188))
    d.polygon([(600, 620), (940, 620), (920, 500), (620, 500)], fill=(130, 134, 144))
    d.text((725, 550), "MacBook Pro", font=_font(20, True), fill=(240, 242, 248))

    # 3. Keychron Mechanical Keyboard
    d.rounded_rectangle((1040, 630, 1380, 720), 10, fill=(44, 46, 54))
    for r in range(3):
        for c in range(11):
            d.rectangle((1058 + c * 28, 644 + r * 22, 1080 + c * 28, 660 + r * 22), fill=(88, 92, 104))

    # 4. Bose Headphones (narrator says "Sony headphones" -> contradiction!)
    d.arc((1450, 420, 1650, 620), 180, 360, fill=(28, 30, 36), width=18)
    d.rounded_rectangle((1438, 520, 1495, 625), 18, fill=(28, 30, 36))
    d.rounded_rectangle((1605, 520, 1662, 625), 18, fill=(28, 30, 36))
    d.text((1446, 562), "BOSE", font=_font(16, True), fill=(242, 244, 250))
    d.text((1612, 562), "BOSE", font=_font(16, True), fill=(242, 244, 250))

    # 5. Sony Alpha a7 IV Camera + SEL2470GM2 Lens with legible model plate
    d.rounded_rectangle((1760, 520, 2060, 700), 18, fill=(24, 24, 28))
    d.rectangle((1830, 485, 1930, 525), fill=(24, 24, 28))
    d.ellipse((1835, 540, 1985, 690), fill=(14, 16, 20), outline=(90, 94, 104), width=6)
    d.text((1780, 532), "SONY  ILCE-7M4", font=_font(18, True), fill=(245, 245, 250))
    d.text((1848, 605), "SEL2470GM2", font=_font(15, True), fill=(220, 225, 235))

    VW, VH = 960, 540
    tmp = Path(tempfile.mkdtemp(prefix="dev_clip_"))
    wav_path = tmp / "narration.wav"
    narration = (
        "Here is my desk setup with my Dell twenty seven inch monitor and my MacBook Pro from last year. "
        "Next to the keyboard are my Sony headphones, and on the right is my Sony Alpha seven four camera."
    )
    _synthesize_narration_wav(wav_path, duration_s, narration)

    total_frames = int(duration_s * fps)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(fps), "-i", "-",
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-shortest", str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in range(total_frames):
        p = f / max(1, total_frames - 1)
        # Smooth pan left -> right
        x0 = int((CW - VW) * p)
        y0 = int(140 + 8 * math.sin(f * 0.3))
        crop = canvas.crop((x0, y0, x0 + VW, y0 + VH))
        proc.stdin.write(crop.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed rendering dev walkthrough clip")
    wav_path.unlink(missing_ok=True)
    tmp.rmdir()
    return str(dst)


def generate_dev_followup_label_clip(out_video: str | Path, model_code: str = "A2992", duration_s: float = 4.0, fps: int = 10) -> str:
    """Render a 4-second close-up clip of the bottom aluminum chassis engraving of the laptop."""
    dst = Path(out_video)
    dst.parent.mkdir(parents=True, exist_ok=True)
    VW, VH = 960, 540

    base = Image.new("RGB", (VW, VH), (168, 172, 180))
    d = ImageDraw.Draw(base)
    d.rounded_rectangle((140, 90, 820, 450), 22, fill=(198, 202, 210), outline=(120, 124, 134), width=3)
    d.text((240, 150), "Designed by Apple in California  Assembled in China", font=_font(19, False), fill=(55, 58, 66))
    d.text((240, 215), f"MacBook Pro 14-inch   Model {model_code}", font=_font(34, True), fill=(22, 24, 30))
    d.text((240, 280), "Rated 20V === 4.7A max   FCC ID: BCGA2992   EMC 8407", font=_font(18, False), fill=(65, 68, 76))
    d.text((240, 340), "Serial: C02DEMO99X", font=_font(18, False), fill=(75, 78, 88))

    total_frames = int(duration_s * fps)
    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for f in range(total_frames):
        dx = int(3 * math.sin(f * 0.4))
        dy = int(2 * math.cos(f * 0.3))
        shifted = Image.new("RGB", (VW, VH), (168, 172, 180))
        shifted.paste(base, (dx, dy))
        proc.stdin.write(shifted.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed rendering follow-up label clip")
    return str(dst)


def ensure_dev_assets(base_dir: str | Path = "eval/dev_clips") -> dict[str, str]:
    """Ensure local sample declarations page, walkthrough clip, and follow-up clip exist."""
    bdir = Path(base_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    policy_jpg = bdir / "sample_declarations_7500.jpg"
    walk_mp4 = bdir / "dev_desk_walkthrough.mp4"
    followup_mp4 = bdir / "dev_laptop_label_a2992.mp4"

    if not policy_jpg.exists():
        generate_sample_declarations_page(policy_jpg, limit_usd=7500, deductible_usd=500)
    if not walk_mp4.exists():
        generate_dev_walkthrough_video(walk_mp4)
    if not followup_mp4.exists():
        generate_dev_followup_label_clip(followup_mp4, model_code="A2992")

    return {
        "policy_path": str(policy_jpg),
        "walkthrough_path": str(walk_mp4),
        "followup_path": str(followup_mp4),
    }


if __name__ == "__main__":
    assets = ensure_dev_assets()
    print("Generated dev assets:", assets)
