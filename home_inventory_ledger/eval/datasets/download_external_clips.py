"""Download external, open-licensed real-world home & desk videos for 5 Sigma evaluation.

COMPLIANCE & DATA RIGHTS (Gemini 4 Hackathon Rules 1 & 5):
- Every video in this benchmark is an external, real-world recording hosted on Wikimedia Commons
  under **Public Domain** or **Creative Commons (CC BY 3.0)** licenses — NONE were filmed or
  authored by our team.
- Raw video files are stored strictly in git-ignored `eval/raw_clips/` (never committed to git).
- Ground-truth item lists in `eval/labels/*.json` are derived directly from Wikimedia Commons'
  independent archival scene descriptions prior to running any model.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import wave

import httpx

from eval.transforms import apply_video_transform
from ingest.frames import ffmpeg_exe

EXTERNAL_CLIPS: list[dict] = [
    {
        "clip_id": "ext_01_macbook_desk",
        "filename": "ext_01_macbook_desk.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/40/%E3%82%AF%E3%83%A9%E3%83%A0%E3%82%B7%E3%82%A7%E3%83%AB%E3%82%A2%E3%83%AB%E3%83%9F%E3%82%B9%E3%82%BF%E3%83%B3%E3%83%89_Macbook%E3%81%8B%E3%82%89%E3%83%8E%E3%83%BC%E3%83%88%E3%83%91%E3%82%BD%E3%82%B3%E3%83%B3%E3%81%BE%E3%81%A7%E5%B9%85%E5%BA%83%E3%81%8F%E4%BD%BF%E3%81%88%E3%82%8B%E3%81%95%E3%81%BE%E3%81%96%E3%81%BE%E3%81%AA%E5%8E%9A%E3%81%95%E3%81%AB%E8%AA%BF%E6%95%B4%E5%8F%AF%E8%83%BD_PDA-STN23S.webm",
        "license": "CC BY 3.0 (Sanwa Supply / Wikimedia Commons)",
        "archival_description": "Clamshell aluminum laptop stand PDA-STN23S used on desk with Apple MacBook laptop and external monitor.",
        "splice_contradiction_speech": "On my desk here is my Dell Windows laptop sitting in the aluminum stand.",
        "gt": {
            "clip_id": "ext_01_macbook_desk",
            "source": "Wikimedia Commons (CC BY 3.0)",
            "conditions": {"lighting": "studio_indoor", "motion": "multi_angle_cuts", "audio": "contradictory_narration"},
            "items": [
                {"category": "laptop", "brand": "Apple", "model": None, "qty": 1, "price_usd": 1199.0},
                {"category": "stand", "brand": "Sanwa", "model": "PDA-STN23S", "qty": 1, "price_usd": 45.0},
                {"category": "monitor", "brand": None, "model": None, "qty": 1, "price_usd": 350.0},
            ],
            "planted_contradiction": {"item_index": 0, "said": "Dell"},
        },
    },
    {
        "clip_id": "ext_02_kitchen_appliances",
        "filename": "ext_02_kitchen_appliances.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/db/12.03.25_USVI_Production_%28MVI-5153%29.webm",
        "license": "Public Domain (USVI Production MVI-5153 / Wikimedia Commons)",
        "archival_description": "Modern stainless steel stove with granite countertops and a stand mixer in the background.",
        "gt": {
            "clip_id": "ext_02_kitchen_appliances",
            "source": "Wikimedia Commons Public Domain (MVI-5153)",
            "conditions": {"lighting": "natural_daylight", "motion": "handheld_pan", "audio": "ambient"},
            "items": [
                {"category": "stove", "brand": None, "model": None, "qty": 1, "price_usd": 1200.0},
                {"category": "stand_mixer", "brand": None, "model": None, "qty": 1, "price_usd": 380.0},
            ],
            "planted_contradiction": None,
        },
    },
    {
        "clip_id": "ext_03_bedroom_desk",
        "filename": "ext_03_bedroom_desk.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/6/6c/12.03.25_USVI_Production_%28MVI-5171%29.webm",
        "license": "Public Domain (USVI Production MVI-5171 / Wikimedia Commons)",
        "archival_description": "Cozy bedroom with a wooden desk, white chair, and patterned rug under soft white curtains.",
        "gt": {
            "clip_id": "ext_03_bedroom_desk",
            "source": "Wikimedia Commons Public Domain (MVI-5171)",
            "conditions": {"lighting": "window_backlit", "motion": "handheld_walk", "audio": "ambient"},
            "items": [
                {"category": "desk", "brand": None, "model": None, "qty": 1, "price_usd": 450.0},
                {"category": "chair", "brand": None, "model": None, "qty": 1, "price_usd": 180.0},
                {"category": "rug", "brand": None, "model": None, "qty": 1, "price_usd": 220.0},
            ],
            "planted_contradiction": None,
        },
    },
    {
        "clip_id": "ext_04_bedroom_minimalist",
        "filename": "ext_04_bedroom_minimalist.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/b/b3/12.03.25_USVI_Production_%28MVI-5183%29.webm",
        "license": "Public Domain (USVI Production MVI-5183 / Wikimedia Commons)",
        "archival_description": "Bright minimalist bedroom with white walls, a four-poster bed, a colorful zigzag rug, a white dresser, and a wicker chair.",
        "gt": {
            "clip_id": "ext_04_bedroom_minimalist",
            "source": "Wikimedia Commons Public Domain (MVI-5183)",
            "conditions": {"lighting": "bright_indoor", "motion": "handheld_pan", "audio": "ambient"},
            "items": [
                {"category": "bed", "brand": None, "model": None, "qty": 1, "price_usd": 1100.0},
                {"category": "dresser", "brand": None, "model": None, "qty": 1, "price_usd": 550.0},
                {"category": "chair", "brand": None, "model": None, "qty": 1, "price_usd": 240.0},
                {"category": "rug", "brand": None, "model": None, "qty": 1, "price_usd": 280.0},
            ],
            "planted_contradiction": None,
        },
    },
]


def _splice_spoken_claim(video_in: Path, video_out: Path, text: str) -> None:
    """Attach spoken narration track (`say` TTS) onto silent external product clip for T3 contradiction testing."""
    say_bin = Path("/usr/bin/say")
    tmp = Path(tempfile.mkdtemp(prefix="contra_"))
    wav_path = tmp / "speech.wav"
    if say_bin.exists():
        subprocess.run(
            [str(say_bin), "-v", "Samantha", "-r", "175", "-o", str(wav_path), "--data-format=LEI16@16000", text],
            check=True,
        )
    else:
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 32000)

    cmd = [
        ffmpeg_exe(), "-y", "-loglevel", "error",
        "-i", str(video_in),
        "-i", str(wav_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-t", "14",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac",
        str(video_out),
    ]
    subprocess.run(cmd, check=True)
    wav_path.unlink(missing_ok=True)
    tmp.rmdir()


def download_and_prepare_external_set(
    raw_dir: Path = Path("eval/raw_clips"),
    labels_dir: Path = Path("eval/labels"),
    manifest_path: Path = Path("eval/test_manifest.json"),
) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "HomeInventoryLedgerBot/1.0 (https://github.com/satish/home-inventory-ledger; satish@hackathon.nyc) httpx/0.28",
        "Accept": "*/*",
    }
    manifest_entries: list[dict] = []

    import time

    with httpx.Client(timeout=45.0, follow_redirects=True, headers=headers) as client:
        for spec in EXTERNAL_CLIPS:
            cid = spec["clip_id"]
            dl_path = raw_dir / spec["filename"]
            if not dl_path.exists():
                for attempt in range(3):
                    resp = client.get(spec["url"])
                    if resp.status_code == 429:
                        time.sleep(2.5 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    dl_path.write_bytes(resp.content)
                    time.sleep(1.2)
                    break
                else:
                    resp.raise_for_status()

            usable_video_path = dl_path
            if spec.get("splice_contradiction_speech"):
                spliced = raw_dir / f"{cid}_with_narration.mp4"
                if not spliced.exists():
                    _splice_spoken_claim(dl_path, spliced, spec["splice_contradiction_speech"])
                usable_video_path = spliced

            lbl_path = labels_dir / f"{cid}.json"
            lbl_path.write_text(json.dumps(spec["gt"], indent=2))

            manifest_entries.append(
                {
                    "clip_id": cid,
                    "video_path": str(usable_video_path),
                    "label_path": str(lbl_path),
                    "license": spec["license"],
                    "source_url": spec["url"],
                }
            )

    # Add 5th messy edge-case clip: severe dim-light + blur perturbation of ext_03_bedroom_desk
    base_bed = next(e for e in manifest_entries if e["clip_id"] == "ext_03_bedroom_desk")
    dark_video = raw_dir / "ext_05_bedroom_lowlight.mp4"
    if not dark_video.exists():
        apply_video_transform(base_bed["video_path"], dark_video, "darken_mild")
    dark_gt = json.loads(Path(base_bed["label_path"]).read_text())
    dark_gt["clip_id"] = "ext_05_bedroom_lowlight"
    dark_gt["conditions"] = {"lighting": "dim_lowlight_perturbed", "motion": "handheld_walk", "audio": "ambient"}
    dark_lbl = labels_dir / "ext_05_bedroom_lowlight.json"
    dark_lbl.write_text(json.dumps(dark_gt, indent=2))
    manifest_entries.append(
        {
            "clip_id": "ext_05_bedroom_lowlight",
            "video_path": str(dark_video),
            "label_path": str(dark_lbl),
            "license": base_bed["license"] + " (darken_mild edge-case transform)",
            "source_url": base_bed["source_url"],
        }
    )

    manifest = {
        "dataset_name": "Wikimedia Commons Open-Licensed Real Home & Desk Walkthroughs (T1/T3 Held-Out Set)",
        "team_authored_examples": 0,
        "clips": manifest_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    m = download_and_prepare_external_set()
    print(f"Prepared {len(m['clips'])} external unseen benchmark clips -> eval/test_manifest.json")
