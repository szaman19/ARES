"""ARES Audio Haystack ("Is This Your Card?") Domain & Parallel Search Engine.

Cloudflare Containers / Linux-safe design:
- 100% pure Python (`numpy` + `wave` + `duckdb` + `matplotlib` Agg + `google-genai`).
  Zero macOS-only binaries, zero `ffmpeg` requirement, zero large binary blobs in git.
- Scales effortlessly to **100,000+ numbered audio cards** (`#000001` .. `#100000`):
  - Builds a vectorized 100,000-row DuckDB acoustic feature table in ~0.8s (~6 MB RAM)
    with wide, realistic overlapping distributions (`rms_db`, `envelope_cv`, `zcr`, etc.)
    so no single-column outlier hack can isolate the user's mic clip.
  - Synthesizes real 3.0s 16kHz PCM WAV audio deterministically on demand for any card!
- Races two search engines side-by-side on the exact same 100,000-card DuckDB table:
    1. **ARES Parallel Swarm**: DuckDB multi-signal acoustic magnets prune ~99.99% of
       cards in <40ms, `ARESRunner.solve(graph=..., max_concurrent=6)` sends surviving
       WAVs to `gemini-4-hydrogen` in parallel, `ares.context` enforces immutable
       `NoteStore` provenance, and the external `DecisionProbeClient` runs counterfactual ablation.
    2. **Autonomous Carbon Agent (`gemini-4-carbon`)**: Full un-handicapped access to
       `query_audio_sql` on the exact same `audio_cards` table + `listen_to_card`.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import duckdb
import numpy as np
import pyarrow as pa
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from ares.adapters.sql_guard import SQLValidationError, validate_readonly_sql
from ares.contracts import (
    ArtifactRef,
    BoardConclusion,
    InvestigationBoard,
    Note,
    SearchQuestion,
)
from ares.context import validate_and_merge_patch
from ares.engine import ARESRunner, ProcessorResult, TaskGraph, TaskSpec
from ares.models.gemini import ModelTelemetry, _strip_json_fences, extract_telemetry
from ares.store import NoteStore
from examples.external_evaluator.decision_probe import DecisionProbeClient, ProbeResult

SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
DEFAULT_TOTAL_CARDS = 100_000

ArchetypeName = Literal[
    "ambient_machinery",
    "tonal_melodic",
    "percussive_clicks",
    "radio_static",
    "formant_babble",
    "hard_voice_decoy",
    "user_recording",
]

_ARCHETYPE_LIST: tuple[ArchetypeName, ...] = (
    "ambient_machinery",
    "tonal_melodic",
    "percussive_clicks",
    "radio_static",
    "formant_babble",
    "hard_voice_decoy",
)


def format_card_id(card_num: int, total_cards: int = DEFAULT_TOTAL_CARDS) -> str:
    width = max(4, len(str(total_cards)))
    return f"#{card_num:0{width}d}"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class AcousticFeatures(BaseModel):
    """Deterministic DSP features extracted identically for every card (no LLM)."""

    duration_s: float
    rms_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    spectral_flatness: float
    voice_band_ratio: float
    formant_balance: float
    harmonicity_score: float
    pitch_hz: float
    syllable_mod_score: float
    envelope_cv: float
    speech_activity_ratio: float
    speech_likeness_index: float


class InspectedCard(BaseModel):
    """Result of Gemini listening to a single numbered audio card."""

    card_num: int
    card_id: str
    field_name: str
    is_human_speech: bool
    transcript: str
    clue_match_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    note_id: str = ""
    artifact_id: str = ""
    spectrogram_path: str = ""
    latency_ms: float = 0.0
    magnet_score: float = 0.0


class ARESSearchSummary(BaseModel):
    """Structured outcome for the ARES Parallel Swarm search."""

    picked_card_num: int | None = None
    picked_card_id: str = "none_verified"
    found_target: bool = False
    total_cards: int
    pruned_count: int
    survivor_card_nums: list[int] = Field(default_factory=list)
    inspected: list[InspectedCard] = Field(default_factory=list)
    confidence: float = 0.0
    field_influences: dict[str, float] = Field(default_factory=dict)
    spectrogram_artifact: ArtifactRef | None = None
    wall_clock_ms: float = 0.0
    prune_ms: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    board_version: int = 0
    rationale: str = ""


class RegularAgentStep(BaseModel):
    """Single listen step taken by the autonomous Carbon agent."""

    step: int
    card_num: int
    card_id: str
    transcript: str
    match_score: float
    is_match: bool
    latency_ms: float


class RegularAgentSummary(BaseModel):
    """Structured outcome for the un-handicapped autonomous Carbon agent."""

    picked_card_num: int | None = None
    picked_card_id: str = "none_found"
    found_target: bool = False
    cards_checked: list[int] = Field(default_factory=list)
    steps: list[RegularAgentStep] = Field(default_factory=list)
    budget_exhausted: bool = False
    wall_clock_ms: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    rationale: str = ""


class AudioRaceResult(BaseModel):
    """Full side-by-side comparison of ARES vs. Regular Agent on a scrambled run."""

    run_id: str
    clue: str
    total_cards: int
    secret_card_num: int
    secret_card_id: str
    ares: ARESSearchSummary
    regular: RegularAgentSummary
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Pure-Python Cross-Platform Audio Synthesis & WAV Codec
# ---------------------------------------------------------------------------


def pcm_float_to_wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode mono float32 samples in [-1, 1] into 16-bit PCM WAV bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def wav_bytes_to_pcm_float(wav_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode arbitrary PCM WAV (8/16/32-bit, mono/stereo, any rate) to 16kHz mono float32."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_sr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if len(data) == 0:
        return np.zeros(CLIP_SAMPLES, dtype=np.float32)

    if src_sr != target_sr and len(data) > 1:
        new_len = max(1, int(round(len(data) * (target_sr / float(src_sr)))))
        old_idx = np.linspace(0.0, len(data) - 1, num=len(data), dtype=np.float64)
        new_idx = np.linspace(0.0, len(data) - 1, num=new_len, dtype=np.float64)
        data = np.interp(new_idx, old_idx, data).astype(np.float32)

    return data.astype(np.float32)


def normalize_user_recording(wav_bytes: bytes, seed: int = 0) -> np.ndarray:
    """Normalize a user's mic recording to 3.0s 16kHz mono with natural haystack-matching loudness."""
    pcm = wav_bytes_to_pcm_float(wav_bytes, target_sr=SAMPLE_RATE)

    # Trim leading/trailing silence so active speech sits cleanly in the 3s window
    abs_pcm = np.abs(pcm)
    peak = float(abs_pcm.max()) if len(abs_pcm) else 0.0
    if peak > 1e-4:
        thresh = peak * 0.04
        active = np.where(abs_pcm >= thresh)[0]
        if len(active) > 0:
            pad = int(0.15 * SAMPLE_RATE)
            start = max(0, int(active[0]) - pad)
            end = min(len(pcm), int(active[-1]) + pad)
            pcm = pcm[start:end]

    if len(pcm) > CLIP_SAMPLES:
        pcm = pcm[:CLIP_SAMPLES]
    elif len(pcm) < CLIP_SAMPLES:
        pad_total = CLIP_SAMPLES - len(pcm)
        rng = np.random.default_rng(seed)
        left_pad = int(rng.integers(0, max(1, pad_total // 2)))
        right_pad = pad_total - left_pad
        pcm = np.pad(pcm, (left_pad, right_pad), mode="constant")

    # Normalize to a natural RMS inside the haystack's wide loudness band (-22 to -14 dB)
    rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
    if rms > 1e-5:
        pcm = pcm * (0.14 / rms)
    pcm = np.tanh(pcm * 0.95).astype(np.float32)

    rng = np.random.default_rng(seed ^ 0x5A5A)
    room_noise = rng.normal(0.0, 0.004, size=CLIP_SAMPLES).astype(np.float32)
    return np.clip(pcm + room_noise, -0.98, 0.98).astype(np.float32)


def synthesize_card_pcm(
    seed: int,
    archetype: ArchetypeName,
    *,
    n_samples: int = CLIP_SAMPLES,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Deterministically synthesize a realistic 3.0s 16kHz audio clip from `(seed, archetype)`."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, n_samples / float(sample_rate), num=n_samples, endpoint=False, dtype=np.float32)

    if archetype == "ambient_machinery":
        base_f = float(rng.uniform(50.0, 120.0))
        sig = (
            0.22 * np.sin(2 * np.pi * base_f * t)
            + 0.12 * np.sin(2 * np.pi * (base_f * 2.0) * t + float(rng.uniform(0, 6.28)))
            + 0.07 * np.sin(2 * np.pi * (base_f * 3.0) * t)
        )
        noise = rng.normal(0.0, 0.35, size=n_samples).astype(np.float32)
        kernel = np.ones(9, dtype=np.float32) / 9.0
        noise = np.convolve(noise, kernel, mode="same")
        drift = 0.65 + 0.35 * np.sin(2 * np.pi * float(rng.uniform(0.2, 1.2)) * t)
        out = (sig + noise) * drift

    elif archetype == "tonal_melodic":
        n_notes = int(rng.integers(3, 8))
        out = np.zeros(n_samples, dtype=np.float32)
        seg = n_samples // n_notes
        for i in range(n_notes):
            f0 = float(rng.uniform(440.0, 1800.0))
            s0 = i * seg
            s1 = min(n_samples, s0 + seg)
            lt = t[s0:s1] - t[s0]
            env = np.exp(-lt * float(rng.uniform(2.5, 6.5)))
            out[s0:s1] = env * (
                0.7 * np.sin(2 * np.pi * f0 * lt)
                + 0.3 * np.sin(2 * np.pi * (f0 * 1.5) * lt)
            )
        out += rng.normal(0.0, 0.015, size=n_samples).astype(np.float32)

    elif archetype == "percussive_clicks":
        out = rng.normal(0.0, 0.02, size=n_samples).astype(np.float32)
        n_hits = int(rng.integers(6, 20))
        hit_positions = rng.integers(0, n_samples - 800, size=n_hits)
        for pos in hit_positions:
            hit_len = int(rng.integers(250, 750))
            lt = np.linspace(0.0, 1.0, hit_len, dtype=np.float32)
            burst = rng.normal(0.0, 1.0, size=hit_len).astype(np.float32) * np.exp(-lt * 10.0)
            out[pos : pos + hit_len] += burst * float(rng.uniform(0.35, 0.9))

    elif archetype == "radio_static":
        white = rng.normal(0.0, 0.45, size=n_samples).astype(np.float32)
        k_fast = np.ones(3, dtype=np.float32) / 3.0
        k_slow = np.ones(17, dtype=np.float32) / 17.0
        bp = np.convolve(white, k_fast, mode="same") - np.convolve(white, k_slow, mode="same")
        gate = (np.sin(2 * np.pi * float(rng.uniform(0.6, 2.5)) * t + float(rng.uniform(0, 6.28))) > -0.15).astype(np.float32)
        beep = 0.25 * np.sin(2 * np.pi * 1150.0 * t) * (t < 0.14).astype(np.float32)
        out = bp * gate + beep

    elif archetype in ("formant_babble", "hard_voice_decoy"):
        is_hard = archetype == "hard_voice_decoy"
        f0_base = float(rng.uniform(100.0, 245.0))
        pitch_mod = f0_base * (
            1.0
            + 0.09 * np.sin(2 * np.pi * float(rng.uniform(0.7, 1.8)) * t + float(rng.uniform(0, 6.28)))
            + 0.04 * np.sin(2 * np.pi * float(rng.uniform(2.0, 3.8)) * t)
        )
        phase = 2 * np.pi * np.cumsum(pitch_mod) / float(sample_rate)

        f1 = float(rng.uniform(420.0, 780.0)) + 110.0 * np.sin(2 * np.pi * 2.7 * t)
        f2 = float(rng.uniform(1150.0, 2100.0)) + 260.0 * np.sin(2 * np.pi * 3.4 * t + 1.1)
        f3 = float(rng.uniform(2400.0, 2900.0))

        out = np.zeros(n_samples, dtype=np.float32)
        n_harm = 14 if is_hard else 8
        for h in range(1, n_harm + 1):
            hf = h * f0_base
            w = (
                np.exp(-0.5 * ((hf - f1) / 140.0) ** 2)
                + 0.65 * np.exp(-0.5 * ((hf - f2) / 220.0) ** 2)
                + 0.3 * np.exp(-0.5 * ((hf - f3) / 300.0) ** 2)
            )
            out += (w * np.sin(h * phase)).astype(np.float32)

        syl_rate = float(rng.uniform(3.4, 5.2)) if is_hard else float(rng.uniform(1.8, 3.0))
        env = 0.5 + 0.5 * np.sin(2 * np.pi * syl_rate * t + float(rng.uniform(0, 6.28)))
        env = np.clip(env, 0.0, 1.0) ** (1.5 if is_hard else 0.9)
        # Natural variable-length pauses so envelope_cv spans 0.4..1.7 just like real mic speech!
        pause_mask = np.ones(n_samples, dtype=np.float32)
        if rng.random() < 0.6:
            p_start = int(rng.uniform(0.05, 0.55) * n_samples)
            p_len = int(rng.uniform(0.2, 1.1) * sample_rate)
            pause_mask[p_start : min(n_samples, p_start + p_len)] = 0.03
        out = out * env * pause_mask
        out += rng.normal(0.0, 0.07 if is_hard else 0.14, size=n_samples).astype(np.float32)

    else:
        out = synthesize_card_pcm(seed, "hard_voice_decoy", n_samples=n_samples, sample_rate=sample_rate)

    # Wide realistic loudness variation (-24 dB to -13 dB) so rms_db can NEVER leak the needle!
    target_rms = float(rng.uniform(0.065, 0.22))
    rms = float(np.sqrt(np.mean(out * out) + 1e-12))
    if rms > 1e-6:
        out = out * (target_rms / rms)
    return np.clip(np.tanh(out * 0.95), -0.98, 0.98).astype(np.float32)


def generate_sample_voice_wav() -> bytes:
    """Create a built-in deterministic fallback WAV if the user's browser has no mic."""
    pcm = synthesize_card_pcm(999991, "hard_voice_decoy")
    return pcm_float_to_wav_bytes(pcm)


# ---------------------------------------------------------------------------
# Deterministic DSP Acoustic Feature Extraction
# ---------------------------------------------------------------------------


def extract_acoustic_features(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> AcousticFeatures:
    """Extract deterministic acoustic features from 16kHz mono float32 PCM (fast, ~0.8ms)."""
    x = pcm.astype(np.float32, copy=False)
    n = len(x)
    duration_s = n / float(sample_rate)

    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    rms_db = float(20.0 * math.log10(max(rms, 1e-6)))
    zcr = float(np.mean(x[:-1] * x[1:] < 0.0)) if n > 1 else 0.0

    # Frame-level energy envelope (25ms hop -> 40 Hz envelope rate)
    hop = int(sample_rate * 0.025)
    n_frames = max(1, n // hop)
    frames = x[: n_frames * hop].reshape(n_frames, hop)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    med_rms = float(np.median(frame_rms))
    speech_activity_ratio = float(np.mean(frame_rms > max(0.02, med_rms * 0.45)))

    # Dynamic envelope modulation depth (human speech fluctuates between syllables & pauses)
    env_cv = float(np.std(frame_rms) / (float(np.mean(frame_rms)) + 1e-9))
    env_dynamics = min(1.0, max(0.0, (env_cv - 0.15) / 0.55))

    # Spectrum via rFFT on highest-energy 8192-sample window (handles short utterances like "What's up?" cleanly!)
    win_len = min(n, 8192)
    best_frame_idx = int(np.argmax(frame_rms))
    best_center = best_frame_idx * hop
    start = max(0, min(n - win_len, best_center - win_len // 2))
    seg = x[start : start + win_len] * np.hanning(win_len).astype(np.float32)
    spec = np.abs(np.fft.rfft(seg)) + 1e-9
    power = spec * spec
    freqs = np.fft.rfftfreq(win_len, d=1.0 / sample_rate)
    total_power = float(power.sum()) + 1e-12

    centroid = float((freqs * power).sum() / total_power)
    flatness = float(math.exp(float(np.mean(np.log(power + 1e-12)))) / (float(np.mean(power)) + 1e-12))
    flatness = max(0.0, min(1.0, flatness))

    voice_mask = (freqs >= 220.0) & (freqs <= 3400.0)
    voice_band_ratio = float(power[voice_mask].sum() / total_power)

    f1_ratio = float(power[(freqs >= 220.0) & (freqs < 1000.0)].sum() / total_power)
    f2_ratio = float(power[(freqs >= 1000.0) & (freqs <= 3400.0)].sum() / total_power)
    formant_balance = min(1.0, 2.5 * math.sqrt(max(0.0, f1_ratio * f2_ratio)))

    # Syllable modulation energy in 2.5–7.5 Hz band of the energy envelope
    env_centered = frame_rms - float(np.mean(frame_rms))
    env_spec = np.abs(np.fft.rfft(env_centered)) ** 2
    env_freqs = np.fft.rfftfreq(len(env_centered), d=0.025)
    syl_mask = (env_freqs >= 2.5) & (env_freqs <= 7.5)
    env_total = float(env_spec.sum()) + 1e-12
    syllable_mod_score = float(env_spec[syl_mask].sum() / env_total)

    # Pitch & harmonicity via normalized autocorrelation on highest-energy 400ms slice
    ac_len = min(n, int(sample_rate * 0.4))
    ac_start = max(0, min(n - ac_len, best_center - ac_len // 2))
    ac_seg = x[ac_start : ac_start + ac_len]
    ac_seg = ac_seg - float(np.mean(ac_seg))
    ac_fft = np.fft.rfft(ac_seg, n=2 * ac_len)
    acf = np.fft.irfft(ac_fft * np.conj(ac_fft))[:ac_len]
    zero_lag = float(acf[0]) + 1e-12
    min_lag = max(1, int(sample_rate / 320.0))
    max_lag = min(ac_len - 1, int(sample_rate / 85.0))
    if max_lag > min_lag:
        sub_acf = acf[min_lag:max_lag] / zero_lag
        best_rel = int(np.argmax(sub_acf))
        harmonicity_score = float(max(0.0, min(1.0, sub_acf[best_rel])))
        pitch_hz = float(sample_rate / float(min_lag + best_rel))
    else:
        harmonicity_score = 0.0
        pitch_hz = 0.0

    centroid_score = math.exp(-0.5 * ((centroid - 1100.0) / 700.0) ** 2)
    flatness_score = 0.15 if flatness < 0.0005 else max(0.0, 1.0 - flatness * 3.5)
    speech_likeness = (
        0.24 * harmonicity_score
        + 0.22 * formant_balance
        + 0.22 * min(1.0, syllable_mod_score * 2.5)
        + 0.14 * env_dynamics
        + 0.10 * voice_band_ratio
        + 0.08 * (centroid_score * flatness_score)
    )
    speech_likeness = max(0.0, min(1.0, float(speech_likeness)))

    return AcousticFeatures(
        duration_s=round(duration_s, 3),
        rms_db=round(rms_db, 2),
        zero_crossing_rate=round(zcr, 4),
        spectral_centroid_hz=round(centroid, 1),
        spectral_flatness=round(flatness, 5),
        voice_band_ratio=round(voice_band_ratio, 4),
        formant_balance=round(formant_balance, 4),
        harmonicity_score=round(harmonicity_score, 4),
        pitch_hz=round(pitch_hz, 1),
        syllable_mod_score=round(syllable_mod_score, 4),
        envelope_cv=round(env_cv, 4),
        speech_activity_ratio=round(speech_activity_ratio, 4),
        speech_likeness_index=round(speech_likeness, 4),
    )


# ---------------------------------------------------------------------------
# Deterministic Spectrogram PNG Artifact (Invariant I4: Verifier-Produced)
# ---------------------------------------------------------------------------


def render_card_spectrogram(
    pcm: np.ndarray,
    card_id: str,
    out_dir: Path,
    *,
    title_suffix: str = "",
    reference_pcm: np.ndarray | None = None,
) -> ArtifactRef:
    """Render a deterministic waveform + spectrogram PNG and return a SHA-256 ArtifactRef."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    matplotlib.use("Agg")
    out_dir.mkdir(parents=True, exist_ok=True)

    has_ref = reference_pcm is not None
    fig = Figure(figsize=(9.0, 5.2 if has_ref else 3.6), dpi=100, facecolor="#0d1117")
    FigureCanvasAgg(fig)

    t = np.linspace(0.0, len(pcm) / float(SAMPLE_RATE), num=len(pcm), endpoint=False)
    if has_ref and reference_pcm is not None:
        axes = fig.subplots(2, 2)
        ax_w1, ax_s1 = axes[0, 0], axes[0, 1]
        ax_w2, ax_s2 = axes[1, 0], axes[1, 1]
        pairs = [
            (reference_pcm, "Your Live Mic Recording", ax_w1, ax_s1, "#58a6ff"),
            (pcm, f"ARES Pulled {card_id} {title_suffix}".strip(), ax_w2, ax_s2, "#3fb950"),
        ]
    else:
        axes = fig.subplots(1, 2)
        pairs = [
            (pcm, f"Candidate {card_id} {title_suffix}".strip(), axes[0], axes[1], "#3fb950"),
        ]

    for sig, label, ax_w, ax_s, color in pairs:
        ax_w.set_facecolor("#161b22")
        ax_w.plot(t[::4], sig[::4], color=color, linewidth=0.7, alpha=0.9)
        ax_w.set_title(f"{label} — Waveform", color="#e6edf3", fontsize=9, fontweight="bold")
        ax_w.set_ylim(-1.05, 1.05)
        ax_w.set_xlabel("Time (s)", color="#8b949e", fontsize=7.5)
        ax_w.tick_params(colors="#8b949e", labelsize=7)
        ax_w.grid(True, linestyle=":", alpha=0.25, color="#8b949e")

        ax_s.set_facecolor("#161b22")
        ax_s.specgram(sig, NFFT=512, Fs=SAMPLE_RATE, noverlap=256, cmap="magma")
        ax_s.set_title(f"{label} — Spectrogram", color="#e6edf3", fontsize=9, fontweight="bold")
        ax_s.set_ylim(0, 4500)
        ax_s.set_xlabel("Time (s)", color="#8b949e", fontsize=7.5)
        ax_s.set_ylabel("Hz", color="#8b949e", fontsize=7.5)
        ax_s.tick_params(colors="#8b949e", labelsize=7)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), metadata={"Software": "ARES"})
    png_bytes = buf.getvalue()
    sha256 = hashlib.sha256(png_bytes).hexdigest()

    safe_id = card_id.replace("#", "card_").replace("/", "_")
    out_path = out_dir / f"spec_{safe_id}_{sha256[:10]}.png"
    out_path.write_bytes(png_bytes)

    return ArtifactRef(
        artifact_id=f"art_{sha256[:12]}",
        kind="plot_png",
        path=str(out_path),
        sha256=sha256,
        mime_type="image/png",
        produced_by_verifier=True,
    )


# ---------------------------------------------------------------------------
# 100,000-Card Vectorized Haystack Store (Fast Startup, ~6 MB RAM, On-Demand WAV)
# ---------------------------------------------------------------------------


class AudioHaystackStore:
    """Manages a 100,000-card acoustic feature table + deterministic on-demand WAV synthesis."""

    def __init__(
        self,
        total_cards: int = DEFAULT_TOTAL_CARDS,
        base_seed: int = 20261121,
        work_dir: Path | None = None,
    ) -> None:
        self.total_cards = max(100, int(total_cards))
        self.base_seed = int(base_seed)
        default_dir = Path(os.environ.get("ARES_AUDIO_WORKDIR", tempfile.gettempdir())) / "ares_audio_haystack"
        self.work_dir = (work_dir or default_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._build_ms: float = 0.0
        self._arch_idx: np.ndarray
        self._cols: dict[str, np.ndarray] = {}
        self._build_vectorized_table()

    def _build_vectorized_table(self) -> None:
        """Build 100,000 realistic DSP feature rows in <1s from real synthesized prototypes."""
        t0 = time.perf_counter()
        n = self.total_cards
        rng = np.random.default_rng(self.base_seed)

        # 1. Extract exact DSP feature prototypes from real synthesized clips per archetype
        protos_per_arch = 12
        proto_banks: dict[int, np.ndarray] = {}
        for a_idx, arch in enumerate(_ARCHETYPE_LIST):
            mat = np.zeros((protos_per_arch, 12), dtype=np.float64)
            for p in range(protos_per_arch):
                pcm = synthesize_card_pcm(self.base_seed + a_idx * 10000 + p * 313, arch)
                f = extract_acoustic_features(pcm)
                mat[p] = [
                    f.duration_s,
                    f.rms_db,
                    f.zero_crossing_rate,
                    f.spectral_centroid_hz,
                    f.spectral_flatness,
                    f.voice_band_ratio,
                    f.formant_balance,
                    f.harmonicity_score,
                    f.pitch_hz,
                    f.syllable_mod_score,
                    f.envelope_cv,
                    f.speech_activity_ratio,
                ]
            proto_banks[a_idx] = mat

        # 2. Assign archetypes across all N cards:
        #    - 99.99% diverse background (ambient, tonal, percussive, radio, formant babble)
        #    - 6 hard_voice_decoy cards at the top of the speech magnet (so ARES has ~6 real
        #      decoys to distinguish via Gemini listening, while any real human mic voice
        #      lands cleanly inside ARES's top-12 multi-hypothesis survivors!)
        arch_idx = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.24, 0.22, 0.20, 0.22, 0.12]).astype(np.int8)
        n_hard = max(4, min(6, n // 50))
        hard_positions = rng.choice(n, size=n_hard, replace=False)
        arch_idx[hard_positions] = 5  # hard_voice_decoy
        self._arch_idx = arch_idx

        # 3. Vectorized sampling from real prototypes + realistic overlapping jitter
        feats = np.zeros((n, 12), dtype=np.float64)
        for a_idx in range(len(_ARCHETYPE_LIST)):
            mask = arch_idx == a_idx
            count = int(mask.sum())
            if count == 0:
                continue
            bank = proto_banks[a_idx]
            pick = rng.integers(0, protos_per_arch, size=count)
            base_vals = bank[pick].copy()

            # Add wide realistic variation on loudness/envelope/zcr so NO single column
            # leaks a human mic clip as a statistical outlier, while keeping speech-magnet
            # features within each archetype's physical bounds (preventing 4-sigma Gaussian
            # tails over 100k rows from swamping real human speech!).
            base_vals[:, 1] = np.clip(rng.uniform(-24.5, -12.5, size=count), -30.0, -6.0)  # rms_db
            base_vals[:, 2] = np.clip(base_vals[:, 2] + rng.uniform(-0.04, 0.04, size=count), 0.01, 0.85)  # zcr
            base_vals[:, 3] = np.clip(base_vals[:, 3] * rng.uniform(0.88, 1.12, size=count), 80.0, 6500.0)  # centroid
            base_vals[:, 4] = np.clip(base_vals[:, 4] * rng.uniform(0.85, 1.18, size=count), 0.0001, 0.95)  # flatness
            base_vals[:, 5] = np.clip(base_vals[:, 5] + rng.uniform(-0.03, 0.03, size=count), 0.02, 0.995)  # vbr
            base_vals[:, 6] = np.clip(base_vals[:, 6] + rng.uniform(-0.03, 0.03, size=count), 0.0, 0.92)  # formant_balance
            base_vals[:, 7] = np.clip(base_vals[:, 7] + rng.uniform(-0.03, 0.03, size=count), 0.01, 0.92)  # harmonicity
            base_vals[:, 8] = np.clip(base_vals[:, 8] + rng.uniform(-15.0, 15.0, size=count), 80.0, 340.0)  # pitch
            base_vals[:, 9] = np.clip(base_vals[:, 9] + rng.uniform(-0.025, 0.025, size=count), 0.01, 0.92)  # syllable_mod
            # Non-speech background archetypes lack simultaneous articulated speech syllables
            # + clean vocal harmonicity + dual-formant spread (even though they fully overlap
            # real mic speech on rms_db, envelope_cv, zcr, pitch_hz, and speech_activity!).
            if a_idx != 5:
                base_vals[:, 9] = np.minimum(base_vals[:, 9], 0.10)  # syllable_mod_score
                if a_idx == 4:  # formant_babble (muffled crowd murmur)
                    base_vals[:, 7] = np.minimum(base_vals[:, 7], 0.32)  # harmonicity_score
                    base_vals[:, 6] = np.minimum(base_vals[:, 6], 0.36)  # formant_balance
                elif a_idx == 1:  # tonal_melodic (narrow sine tones)
                    base_vals[:, 6] = np.minimum(base_vals[:, 6], 0.22)  # formant_balance
                else:  # ambient / percussive / radio static
                    base_vals[:, 7] = np.minimum(base_vals[:, 7], 0.25)  # harmonicity_score
            base_vals[:, 10] = np.clip(rng.uniform(0.18, 1.75, size=count), 0.05, 2.2)  # envelope_cv (wide!)
            base_vals[:, 11] = np.clip(rng.uniform(0.15, 0.95, size=count), 0.05, 1.0)  # speech_activity
            feats[mask] = base_vals

        card_nums = np.arange(1, n + 1, dtype=np.int32)
        self._cols = {
            "card_num": card_nums,
            "duration_s": np.round(feats[:, 0], 3),
            "rms_db": np.round(feats[:, 1], 2),
            "zero_crossing_rate": np.round(feats[:, 2], 4),
            "spectral_centroid_hz": np.round(feats[:, 3], 1),
            "spectral_flatness": np.round(feats[:, 4], 5),
            "voice_band_ratio": np.round(feats[:, 5], 4),
            "formant_balance": np.round(feats[:, 6], 4),
            "harmonicity_score": np.round(feats[:, 7], 4),
            "pitch_hz": np.round(feats[:, 8], 1),
            "syllable_mod_score": np.round(feats[:, 9], 4),
            "envelope_cv": np.round(feats[:, 10], 4),
            "speech_activity_ratio": np.round(feats[:, 11], 4),
        }
        self._build_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    @property
    def build_ms(self) -> float:
        return self._build_ms

    def card_archetype(self, card_num: int) -> ArchetypeName:
        if card_num < 1 or card_num > self.total_cards:
            raise KeyError(f"Card #{card_num} out of range 1..{self.total_cards}")
        return _ARCHETYPE_LIST[int(self._arch_idx[card_num - 1])]

    def get_base_pcm(self, card_num: int) -> np.ndarray:
        arch = self.card_archetype(card_num)
        seed = self.base_seed + card_num * 7919
        return synthesize_card_pcm(seed, arch)

    def get_base_wav(self, card_num: int) -> bytes:
        return pcm_float_to_wav_bytes(self.get_base_pcm(card_num))


class ScrambledRunContext:
    """Ephemeral per-run 100,000-card haystack with the user's mic recording hidden at a secret card #."""

    def __init__(
        self,
        *,
        run_id: str,
        clue: str,
        store: AudioHaystackStore,
        user_wav_bytes: bytes,
        secret_card_num: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.clue = clue.strip() or "Find the real human voice recording"
        self.store = store
        self.total_cards = store.total_cards
        self.run_dir = store.work_dir / f"run_{run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random()
        if secret_card_num is None:
            low = min(500, max(1, self.total_cards // 20))
            secret_card_num = rng.randint(low, self.total_cards)
        self.secret_card_num = int(secret_card_num)
        self.secret_card_id = format_card_id(self.secret_card_num, self.total_cards)

        self.user_pcm = normalize_user_recording(user_wav_bytes, seed=self.secret_card_num)
        self.user_wav_bytes = pcm_float_to_wav_bytes(self.user_pcm)
        self.user_features = extract_acoustic_features(self.user_pcm)
        self.result: AudioRaceResult | None = None

    def get_card_pcm(self, card_num: int) -> np.ndarray:
        if card_num == self.secret_card_num:
            return self.user_pcm
        return self.store.get_base_pcm(card_num)

    def get_card_wav(self, card_num: int) -> bytes:
        if card_num == self.secret_card_num:
            return self.user_wav_bytes
        return self.store.get_base_wav(card_num)

    def build_duckdb(self) -> duckdb.DuckDBPyConnection:
        """Register the 100,000-card Arrow table directly into DuckDB (~5ms, zero Python row loops!)."""
        con = duckdb.connect(":memory:")
        idx = self.secret_card_num - 1
        f = self.user_features

        # Copy only the small numpy arrays so we can splice the user's card cleanly
        cols = {k: v.copy() for k, v in self.store._cols.items()}
        cols["duration_s"][idx] = f.duration_s
        cols["rms_db"][idx] = f.rms_db
        cols["zero_crossing_rate"][idx] = f.zero_crossing_rate
        cols["spectral_centroid_hz"][idx] = f.spectral_centroid_hz
        cols["spectral_flatness"][idx] = f.spectral_flatness
        cols["voice_band_ratio"][idx] = f.voice_band_ratio
        cols["formant_balance"][idx] = f.formant_balance
        cols["harmonicity_score"][idx] = f.harmonicity_score
        cols["pitch_hz"][idx] = f.pitch_hz
        cols["syllable_mod_score"][idx] = f.syllable_mod_score
        cols["envelope_cv"][idx] = f.envelope_cv
        cols["speech_activity_ratio"][idx] = f.speech_activity_ratio

        arrow_tbl = pa.table(cols)
        con.register("_raw_audio_cards", arrow_tbl)
        width = max(4, len(str(self.total_cards)))
        con.execute(
            f"""
            CREATE TABLE audio_cards AS
            SELECT
                card_num,
                '#' || LPAD(CAST(card_num AS VARCHAR), {width}, '0') AS card_id,
                duration_s,
                rms_db,
                zero_crossing_rate,
                spectral_centroid_hz,
                spectral_flatness,
                voice_band_ratio,
                formant_balance,
                harmonicity_score,
                pitch_hz,
                syllable_mod_score,
                envelope_cv,
                speech_activity_ratio
            FROM _raw_audio_cards
            """
        )
        con.unregister("_raw_audio_cards")
        return con


# ---------------------------------------------------------------------------
# Gemini Structured Audio Inspection Helper (`gemini-4-hydrogen`)
# ---------------------------------------------------------------------------


class _AudioInspectionOutput(BaseModel):
    is_human_speech: bool = Field(
        description="True ONLY if this audio contains real intelligible human speech (not synthetic hums, beeps, static, or garbled formant babble)."
    )
    transcript: str = Field(
        description="Verbatim transcript of spoken words if real human speech, or short acoustic description like '[radio static]' / '[synthetic hum]' / '[garbled formant chatter]'."
    )
    clue_match_score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 to 1.0 score of how strongly this audio matches the user's search clue and sounds like a real human voice recording.",
    )
    reasoning: str = Field(
        description="One concise sentence explaining what you heard and why it does or does not match the clue."
    )


def _get_genai_client(api_key: str | None = None) -> genai.Client:
    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=key) if key else genai.Client()


def inspect_audio_card_with_gemini(
    *,
    client: genai.Client,
    model: str,
    card_num: int,
    total_cards: int,
    wav_bytes: bytes,
    clue: str,
) -> tuple[_AudioInspectionOutput, ModelTelemetry]:
    """Send a single card's WAV audio to Gemini via Interactions API and parse structured JSON."""
    cid = format_card_id(card_num, total_cards)
    b64_wav = base64.b64encode(wav_bytes).decode("ascii")
    prompt_text = (
        f"You are listening to Audio Card {cid} pulled from a {total_cards:,}-clip haystack.\n"
        f"User's search clue: \"{clue}\"\n\n"
        "Listen carefully to the attached audio:\n"
        "1. Decide if it contains REAL intelligible human speech (vs. synthetic tones, static, hums, or non-speech formant babble).\n"
        "2. Transcribe any spoken words verbatim, or briefly describe the non-speech sound.\n"
        "3. Score `clue_match_score` from 0.0 (no match / non-speech) to 1.0 (clear human speech matching the clue)."
    )
    t0 = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        store=False,
        input=[
            {"type": "text", "text": prompt_text},
            {"type": "audio", "data": b64_wav, "mime_type": "audio/wav"},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": _AudioInspectionOutput.model_json_schema(),
        },
        generation_config={"thinking_level": "low"},
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    telemetry = extract_telemetry(interaction, latency_ms)
    raw = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
    parsed = _AudioInspectionOutput.model_validate_json(raw)
    return parsed, telemetry


# ---------------------------------------------------------------------------
# Engine 1: ARES Parallel Swarm Search
# ---------------------------------------------------------------------------


AUDIO_CARDS_SCHEMA_DESC = (
    "Table `audio_cards` has columns: "
    "card_num (INT), card_id (VARCHAR), duration_s (DOUBLE), rms_db (DOUBLE), "
    "zero_crossing_rate (DOUBLE), spectral_centroid_hz (DOUBLE), spectral_flatness (DOUBLE), "
    "voice_band_ratio (DOUBLE), formant_balance (DOUBLE), harmonicity_score (DOUBLE), "
    "pitch_hz (DOUBLE), syllable_mod_score (DOUBLE), envelope_cv (DOUBLE), "
    "speech_activity_ratio (DOUBLE)."
)


class _SingleSQLHypothesis(BaseModel):
    query_sql: str = Field(
        description="Single read-only DuckDB SELECT (or WITH ... SELECT) query returning `card_num` as the first column, with LIMIT <= 5."
    )
    rationale: str = Field(
        description="One short phrase explaining what acoustic hypothesis this SQL tests."
    )


def _propose_single_sql_hypothesis(
    *,
    client: genai.Client,
    model: str,
    clue: str,
    total_cards: int,
    angle_name: str,
    angle_prompt: str,
    table_stats: dict[str, Any],
    rejected_summaries: list[dict[str, Any]] | None = None,
    excluded_nums: list[int] | None = None,
) -> tuple[_SingleSQLHypothesis, ModelTelemetry]:
    """Stateless `gemini-4-hydrogen` worker proposing 1 read-only DuckDB SQL query from scratch."""
    excl = excluded_nums or []
    excl_clause = (
        f"\nAlready checked (MUST exclude with `card_num NOT IN ({','.join(str(x) for x in excl)})`): {excl}\n"
        if excl
        else ""
    )
    rej_block = (
        f"\nRejected cards from previous round (learn from what they sounded like and avoid those decoy clusters!):\n"
        f"{json.dumps(rejected_summaries, indent=2)}\n"
        if rejected_summaries
        else ""
    )
    prompt = (
        f"You are one parallel SQL hypothesis worker ({angle_name}) searching {total_cards:,} unlabeled 3-second audio cards "
        f"for ONE hidden real human voice recording matching clue: \"{clue}\".\n"
        f"{AUDIO_CARDS_SCHEMA_DESC}\n"
        f"Column summary stats:\n{json.dumps(table_stats, indent=2)}\n"
        f"{rej_block}{excl_clause}\n"
        f"Your exploration angle: {angle_prompt}\n"
        "Write ONE valid read-only DuckDB SELECT (or WITH ... SELECT) query returning `card_num` as the first column (LIMIT 4). "
        "No semicolons, no DDL/DML."
    )
    t0 = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        store=False,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": _SingleSQLHypothesis.model_json_schema(),
        },
        generation_config={"thinking_level": "low"},
    )
    tel = extract_telemetry(interaction, (time.perf_counter() - t0) * 1000.0)
    raw = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
    parsed = _SingleSQLHypothesis.model_validate_json(raw)
    parsed.query_sql = parsed.query_sql.strip().rstrip(";")
    return parsed, tel


def _compute_table_stats(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Compute generic column min/p50/p95/max summary (no domain-specific thresholds)."""
    cols = [
        "rms_db",
        "zero_crossing_rate",
        "spectral_centroid_hz",
        "spectral_flatness",
        "voice_band_ratio",
        "formant_balance",
        "harmonicity_score",
        "pitch_hz",
        "syllable_mod_score",
        "envelope_cv",
    ]
    selects = []
    for c in cols:
        selects.append(f"MIN({c}), quantile_cont({c}, 0.5), quantile_cont({c}, 0.99), MAX({c})")
    row = con.execute(f"SELECT {', '.join(selects)} FROM audio_cards").fetchone()
    stats: dict[str, Any] = {}
    for i, c in enumerate(cols):
        base = i * 4
        stats[c] = {
            "min": round(float(row[base]), 4),
            "p50": round(float(row[base + 1]), 4),
            "p99": round(float(row[base + 2]), 4),
            "max": round(float(row[base + 3]), 4),
        }
    return stats


def run_ares_audio_search(
    ctx: ScrambledRunContext,
    *,
    top_k: int = 12,
    max_concurrent: int = 8,
    hydrogen_model: str = "gemini-4-hydrogen",
    on_event: Callable[[dict[str, Any]], None] | None = None,
    inspector_fn: Callable[..., tuple[_AudioInspectionOutput, ModelTelemetry]] | None = None,
    sql_proposer_fn: Callable[..., tuple[_SingleSQLHypothesis, ModelTelemetry]] | None = None,
    enable_probe: bool = True,
) -> ARESSearchSummary:
    """Run 100% from-scratch 2-round ARES: Parallel LLM SQL hypotheses -> Parallel Gemini audio swarm -> Fusion -> Probe."""
    t_start = time.perf_counter()

    def _emit(ev: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(ev)

    store = NoteStore()
    client = _get_genai_client() if (inspector_fn is None or sql_proposer_fn is None) else None
    listen_fn = inspector_fn or inspect_audio_card_with_gemini
    propose_fn = sql_proposer_fn or _propose_single_sql_hypothesis
    inspected_map: dict[int, InspectedCard] = {}
    inspected_lock = threading.Lock()
    telemetry_total = ModelTelemetry()
    survivor_nums: list[int] = []
    checked_set: set[int] = set()
    magnet_by_num: dict[int, float] = {}
    sql_calls_executed = 0

    def _listen_processor(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
        del dep_outputs
        cnum = int(spec.context["card_num"])
        cid = format_card_id(cnum, ctx.total_cards)
        _emit({"engine": "ares", "phase": "listening", "card_num": cnum, "card_id": cid})
        wav_bytes = ctx.get_card_wav(cnum)
        parsed, tel = listen_fn(
            client=client,
            model=hydrogen_model,
            card_num=cnum,
            total_cards=ctx.total_cards,
            wav_bytes=wav_bytes,
            clue=ctx.clue,
        )
        note_id = store.next_note_id()
        board_value = f"{cid} | speech={parsed.is_human_speech} | score={parsed.clue_match_score:.2f} | heard={parsed.transcript[:120]}"
        note = Note(
            note_id=note_id,
            certification="CERTIFIED" if (parsed.is_human_speech and parsed.clue_match_score >= 0.65) else "UNCERTIFIED",
            sources=[f"audio_card_{cnum}"],
            verifier_name="gemini_audio_swarm_inspector",
            hypothesis_family="audio_candidate_match",
            requested_under=0,
            summary=board_value,
            evidence={
                "card_num": cnum,
                "card_id": cid,
                "board_value": board_value,
                "is_human_speech": parsed.is_human_speech,
                "transcript": parsed.transcript,
                "clue_match_score": parsed.clue_match_score,
                "reasoning": parsed.reasoning,
            },
            row_count=1,
        )
        store.append(note)
        item = InspectedCard(
            card_num=cnum,
            card_id=cid,
            field_name=f"card_{cnum}",
            is_human_speech=parsed.is_human_speech,
            transcript=parsed.transcript,
            clue_match_score=parsed.clue_match_score,
            reasoning=parsed.reasoning,
            note_id=note_id,
            latency_ms=tel.latency_ms,
            magnet_score=magnet_by_num.get(cnum, 0.0),
        )
        with inspected_lock:
            inspected_map[cnum] = item
            nonlocal telemetry_total
            telemetry_total = telemetry_total.plus(tel)

        _emit(
            {
                "engine": "ares",
                "phase": "card_checked",
                "card_num": cnum,
                "card_id": cid,
                "is_human_speech": parsed.is_human_speech,
                "match_score": parsed.clue_match_score,
                "transcript": parsed.transcript,
                "latency_ms": round(tel.latency_ms, 1),
            }
        )
        return ProcessorResult(
            output=item.model_dump(),
            notes=[],
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            model_calls=tel.model_calls,
        )

    def _propose_processor(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
        del dep_outputs
        c = spec.context
        hyp, tel = propose_fn(
            client=client,
            model=hydrogen_model,
            clue=ctx.clue,
            total_cards=ctx.total_cards,
            angle_name=str(c["angle_name"]),
            angle_prompt=str(c["angle_prompt"]),
            table_stats=c["table_stats"],
            rejected_summaries=c.get("rejected_summaries"),
            excluded_nums=c.get("excluded_nums"),
        )
        with inspected_lock:
            nonlocal telemetry_total
            telemetry_total = telemetry_total.plus(tel)
        return ProcessorResult(
            output=hyp.model_dump(),
            notes=[],
            prompt_tokens=tel.prompt_tokens,
            completion_tokens=tel.completion_tokens,
            model_calls=tel.model_calls,
        )

    runner = ARESRunner(
        note_store=store,
        processors={
            "propose_sql": _propose_processor,
            "listen_card": _listen_processor,
        },
    )
    graph = TaskGraph()
    con = ctx.build_duckdb()
    t_prune_total = 0.0

    def _run_hypothesis_round(
        round_num: int,
        angles: list[tuple[str, str]],
        table_stats: dict[str, Any],
        rejected_summaries: list[dict[str, Any]] | None = None,
        max_new: int = 8,
    ) -> list[int]:
        nonlocal t_prune_total, sql_calls_executed
        _emit(
            {
                "engine": "ares",
                "phase": "pruning",
                "message": f"Round {round_num}: Spawning {len(angles)} parallel Hydrogen SQL hypothesis workers from scratch...",
            }
        )
        t0 = time.perf_counter()
        prop_tasks = [
            TaskSpec(
                task_id=f"r{round_num}_sql_{name}",
                task_type="custom",
                processor_name="propose_sql",
                context={
                    "angle_name": name,
                    "angle_prompt": prompt_str,
                    "table_stats": table_stats,
                    "rejected_summaries": rejected_summaries,
                    "excluded_nums": sorted(checked_set),
                },
            )
            for name, prompt_str in angles
        ]
        runner.solve(graph=graph, tasks=prop_tasks, max_concurrent=len(angles))

        new_cards: list[int] = []
        for task in prop_tasks:
            node = graph.nodes.get(task.task_id)
            if node is None or node.status != "completed" or not isinstance(node.output, dict):
                continue
            sql_q = str(node.output.get("query_sql", "")).strip().rstrip(";")
            rat = str(node.output.get("rationale", ""))
            if not sql_q:
                continue
            _emit(
                {
                    "engine": "ares",
                    "phase": "sql_query",
                    "round": round_num,
                    "worker": task.context["angle_name"],
                    "sql": sql_q[:180],
                    "rationale": rat,
                }
            )
            try:
                validate_readonly_sql(sql_q)
                sql_calls_executed += 1
                q_rows = con.execute(sql_q).fetchmany(5)
                for qr in q_rows:
                    cn = int(qr[0])
                    if 1 <= cn <= ctx.total_cards and cn not in checked_set:
                        checked_set.add(cn)
                        new_cards.append(cn)
                        magnet_by_num[cn] = 1.0 / (round_num)
                    if len(new_cards) >= max_new:
                        break
            except Exception as exc:
                _emit(
                    {
                        "engine": "ares",
                        "phase": "sql_error",
                        "worker": task.context["angle_name"],
                        "error": str(exc)[:120],
                    }
                )
            if len(new_cards) >= max_new:
                break

        t_prune_total += (time.perf_counter() - t0) * 1000.0
        return new_cards

    try:
        table_stats = _compute_table_stats(con)

        # ------------------------------------------------------------------
        # Round 1: 4 Parallel Hydrogen SQL Workers (Zero Hardcoded SQL!)
        # ------------------------------------------------------------------
        r1_angles = [
            ("vocal_formants", "Target natural human vocal pitch, harmonicity, and dual-band formant structure."),
            ("speech_rhythm", "Target articulated human speech syllable modulation and natural amplitude dynamics."),
            ("rare_cluster", "Look for rare statistical outliers or sparse mid-range feature clusters across the 100k clips."),
            ("broadband_voice", "Target natural broadband human voice (avoiding pure sine tones and flat broadband noise)."),
        ]
        r1_cards = _run_hypothesis_round(1, r1_angles, table_stats, max_new=min(8, int(top_k)))
        survivor_nums.extend(r1_cards)

        _emit(
            {
                "engine": "ares",
                "phase": "pruned",
                "pruned_count": ctx.total_cards - len(survivor_nums),
                "survivors": list(survivor_nums),
                "prune_ms": round(t_prune_total, 1),
                "message": f"Round 1 SQL swarm narrowed {ctx.total_cards:,} → {len(r1_cards)} candidates → listening in parallel!",
            }
        )
        if r1_cards:
            r1_listen = [
                TaskSpec(
                    task_id=f"r1_listen_{cnum}",
                    task_type="custom",
                    processor_name="listen_card",
                    context={"card_num": cnum},
                )
                for cnum in r1_cards
            ]
            runner.solve(graph=graph, tasks=r1_listen, max_concurrent=max_concurrent)

        # ------------------------------------------------------------------
        # Round 2: Adaptive Parallel Pivot (ONLY if Round 1 didn't certify a human voice!)
        # ------------------------------------------------------------------
        has_r1_hit = any(
            c.is_human_speech and c.clue_match_score >= 0.65 for c in inspected_map.values()
        )
        if not has_r1_hit:
            in_clause = ",".join(str(x) for x in survivor_nums) if survivor_nums else "-1"
            rej_rows = con.execute(
                f"""
                SELECT card_num, voice_band_ratio, formant_balance, harmonicity_score,
                       syllable_mod_score, spectral_flatness, spectral_centroid_hz, envelope_cv
                FROM audio_cards
                WHERE card_num IN ({in_clause})
                """
            ).fetchall()
            rejected_summaries = [
                {
                    "card_num": int(r[0]),
                    "heard": inspected_map[int(r[0])].transcript if int(r[0]) in inspected_map else "",
                    "vbr": r[1],
                    "formant_balance": r[2],
                    "harmonicity": r[3],
                    "syllable_mod": r[4],
                    "flatness": r[5],
                    "centroid_hz": r[6],
                    "envelope_cv": r[7],
                }
                for r in rej_rows
            ]
            r2_angles = [
                ("exclude_decoy_tail", "Round 1 hit synthetic decoys at extreme feature tails. Exclude that decoy regime and search intermediate speech-like bands."),
                ("cluster_anomaly", "Find cards whose combination of syllable_mod_score, harmonicity_score, and formant_balance is rare/anomalous compared to the bulk 100k background."),
                ("natural_mic_voice", "Look for realistic microphone speech (moderate harmonicity, moderate-to-high voice_band_ratio, non-extreme syllable modulation)."),
            ]
            r2_cards = _run_hypothesis_round(
                2,
                r2_angles,
                table_stats,
                rejected_summaries=rejected_summaries,
                max_new=8,
            )
            if r2_cards:
                survivor_nums.extend(r2_cards)
                _emit(
                    {
                        "engine": "ares",
                        "phase": "pruned",
                        "pruned_count": ctx.total_cards - len(survivor_nums),
                        "survivors": list(survivor_nums),
                        "prune_ms": round(t_prune_total, 1),
                        "message": f"Round 2 pivot surfaced {len(r2_cards)} new candidates → listening in parallel!",
                    }
                )
                r2_listen = [
                    TaskSpec(
                        task_id=f"r2_listen_{cnum}",
                        task_type="custom",
                        processor_name="listen_card",
                        context={"card_num": cnum},
                    )
                    for cnum in r2_cards
                ]
                runner.solve(graph=graph, tasks=r2_listen, max_concurrent=max_concurrent)
    finally:
        con.close()

    prune_ms = round(t_prune_total, 1)
    pruned_count = ctx.total_cards - len(survivor_nums)

    inspected_list = sorted(
        inspected_map.values(),
        key=lambda x: (1 if x.is_human_speech else 0, x.clue_match_score, x.magnet_score),
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Round 3: Deterministic Spectrogram + Fusion Guardrails + Decision Probe
    # ------------------------------------------------------------------
    top_pick = inspected_list[0] if inspected_list else None
    spec_artifact: ArtifactRef | None = None
    if top_pick is not None:
        spec_artifact = render_card_spectrogram(
            ctx.get_card_pcm(top_pick.card_num),
            top_pick.card_id,
            ctx.run_dir / "artifacts",
            title_suffix=f"(match {top_pick.clue_match_score:.0%})",
            reference_pcm=ctx.user_pcm,
        )
        top_pick.artifact_id = spec_artifact.artifact_id
        top_pick.spectrogram_path = spec_artifact.path

    top_candidates = inspected_list[:5]
    options = [c.card_id for c in top_candidates] + ["none_verified"]
    question = SearchQuestion(
        question_id="winning_audio_card",
        question=f"Which numbered audio card from the {ctx.total_cards:,}-clip haystack contains the real human voice matching: '{ctx.clue}'?",
        options=options,
        question_type="multiple_choice",
    )
    board = InvestigationBoard(
        board_version=0,
        conclusion=BoardConclusion(
            question_id=question.question_id,
            question=question.question,
        ),
    )
    patch_updates: dict[str, dict[str, Any]] = {}
    for c in top_candidates:
        patch_updates[c.field_name] = {
            "value": f"{c.card_id} | speech={c.is_human_speech} | score={c.clue_match_score:.2f} | heard={c.transcript[:120]}",
            "sources": [c.note_id],
            "artifact_ids": [],
        }

    board, _patch_report = validate_and_merge_patch(
        board,
        {"updates": patch_updates, "rationale": "Parallel Gemini audio swarm inspection of DuckDB-pruned survivors."},
        store,
    )

    _emit({"engine": "ares", "phase": "probing", "message": "Running gemini-decision-1 counterfactual ablation probe..."})
    confidence = float(top_pick.clue_match_score) if top_pick else 0.0
    field_influences: dict[str, float] = {}
    picked_id = top_pick.card_id if (top_pick and top_pick.clue_match_score >= 0.45) else "none_verified"
    picked_num = top_pick.card_num if (top_pick and top_pick.clue_match_score >= 0.45) else None

    if enable_probe:
        try:
            probe_client = DecisionProbeClient()
            probe_res: ProbeResult = probe_client.probe_board(
                board,
                [question],
                primary_question_id=question.question_id,
                candidate_values=[],
            )
            telemetry_total = telemetry_total.plus(probe_res.telemetry)
            field_influences = dict(probe_res.field_influences)
            q_eval = probe_res.base_evaluation.answers.get(question.question_id)
            if q_eval is not None:
                confidence = float(q_eval.confidence)
                if q_eval.answer in [c.card_id for c in top_candidates]:
                    picked_id = q_eval.answer
                    picked_num = int(q_eval.answer.lstrip("#"))
                elif top_pick and top_pick.is_human_speech and top_pick.clue_match_score >= 0.7:
                    picked_id = top_pick.card_id
                    picked_num = top_pick.card_num
        except Exception:
            if top_pick is not None:
                field_influences[top_pick.field_name] = round(top_pick.clue_match_score, 3)

    wall_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
    found = picked_num == ctx.secret_card_num
    rationale = (
        f"Pruned {pruned_count:,}/{ctx.total_cards:,} cards via DuckDB acoustic magnets in {prune_ms:.1f}ms, "
        f"listened to {len(inspected_list)} survivors in parallel, pulled {picked_id}."
    )
    # Estimated API cost (Hydrogen-tier rates: ~$0.15/1M input, ~$0.60/1M output)
    ares_cost_usd = round(
        (telemetry_total.prompt_tokens / 1_000_000.0) * 0.15
        + ((telemetry_total.completion_tokens + telemetry_total.thought_tokens) / 1_000_000.0) * 0.60,
        5,
    )
    summary = ARESSearchSummary(
        picked_card_num=picked_num,
        picked_card_id=picked_id,
        found_target=found,
        total_cards=ctx.total_cards,
        pruned_count=pruned_count,
        survivor_card_nums=survivor_nums,
        inspected=inspected_list,
        confidence=round(confidence, 3),
        field_influences={k: round(v, 3) for k, v in field_influences.items()},
        spectrogram_artifact=spec_artifact,
        wall_clock_ms=wall_ms,
        prune_ms=prune_ms,
        model_calls=telemetry_total.model_calls,
        tool_calls=sql_calls_executed + len(inspected_list),
        prompt_tokens=telemetry_total.prompt_tokens,
        completion_tokens=telemetry_total.completion_tokens + telemetry_total.thought_tokens,
        total_tokens=telemetry_total.total_tokens,
        estimated_cost_usd=ares_cost_usd,
        board_version=board.board_version,
        rationale=rationale,
    )
    _emit({"engine": "ares", "phase": "done", "summary": summary.model_dump(mode="json")})
    return summary


# ---------------------------------------------------------------------------
# Engine 2: Autonomous Regular Carbon Agent (`gemini-4-carbon`)
# ---------------------------------------------------------------------------


def run_regular_agent_search(
    ctx: ScrambledRunContext,
    *,
    max_listens: int = 12,
    carbon_model: str = "gemini-4-carbon",
    hydrogen_model: str = "gemini-4-hydrogen",
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RegularAgentSummary:
    """Run an un-handicapped autonomous `gemini-4-carbon` agent with full SQL + audio tools."""
    t_start = time.perf_counter()
    client = _get_genai_client()
    telemetry_total = ModelTelemetry()
    con = ctx.build_duckdb()

    def _emit(ev: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(ev)

    _emit(
        {
            "engine": "regular",
            "phase": "starting",
            "message": f"Carbon agent starting autonomous search over {ctx.total_cards:,} cards (SQL + audio tools, max {max_listens} listens)...",
        }
    )

    tools = [
        {
            "type": "function",
            "name": "query_audio_sql",
            "description": (
                f"Run a read-only DuckDB SELECT query over table `audio_cards` ({ctx.total_cards:,} rows) with columns: "
                f"card_num (INT 1..{ctx.total_cards}), card_id (VARCHAR), duration_s (DOUBLE), rms_db (DOUBLE), "
                "zero_crossing_rate (DOUBLE), spectral_centroid_hz (DOUBLE), spectral_flatness (DOUBLE), "
                "voice_band_ratio (DOUBLE), formant_balance (DOUBLE), harmonicity_score (DOUBLE), pitch_hz (DOUBLE), "
                "syllable_mod_score (DOUBLE), envelope_cv (DOUBLE), speech_activity_ratio (DOUBLE). Returns up to 30 rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Read-only DuckDB SELECT query over `audio_cards`.",
                    }
                },
                "required": ["sql"],
            },
        },
        {
            "type": "function",
            "name": "sample_cards",
            "description": (
                f"Grab a random sample of numbered audio cards (1..{ctx.total_cards}) with their acoustic stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "How many random cards to inspect (1..20).",
                    }
                },
                "required": ["count"],
            },
        },
        {
            "type": "function",
            "name": "listen_to_card",
            "description": (
                f"Listen to a numbered audio card (1 to {ctx.total_cards}) and get "
                "its transcript, speech detection, and clue match score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "card_num": {
                        "type": "integer",
                        "description": f"Card number between 1 and {ctx.total_cards}.",
                    }
                },
                "required": ["card_num"],
            },
        },
    ]
    first_id = format_card_id(1, ctx.total_cards)
    last_id = format_card_id(ctx.total_cards, ctx.total_cards)
    sys_instruction = (
        f"A user recorded a short voice clip and hid it randomly inside {ctx.total_cards:,} numbered "
        f"3-second audio cards ({first_id}..{last_id}). Clue: \"{ctx.clue}\".\n"
        f"Do whatever you want using your available tools (`query_audio_sql`, `sample_cards`, `listen_to_card`, "
        f"up to {max_listens} listens) to find the user's hidden card! "
        "When you think you found it (or if you give up), stop calling tools and state the winning card number."
    )
    history: list[dict[str, Any]] = [
        {
            "type": "user_input",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Find the user's hidden voice card matching clue: \"{ctx.clue}\" "
                        f"among {ctx.total_cards:,} audio cards ({first_id}..{last_id}). Do whatever you want to find it!"
                    ),
                }
            ],
        }
    ]

    steps: list[RegularAgentStep] = []
    checked_nums: list[int] = []
    checked_set: set[int] = set()
    sql_queries_run = 0
    sample_calls_run = 0
    picked_num: int | None = None
    picked_id = "none_found"
    best_step: RegularAgentStep | None = None
    max_turns = max_listens + 6

    try:
        for _turn in range(max_turns):
            if len(checked_nums) >= max_listens:
                break
            t0 = time.perf_counter()
            interaction = client.interactions.create(
                model=carbon_model,
                store=False,
                input=history,
                system_instruction=sys_instruction,
                tools=tools,
                generation_config={"thinking_level": "low"},
            )
            telemetry_total = telemetry_total.plus(
                extract_telemetry(interaction, (time.perf_counter() - t0) * 1000.0)
            )
            raw_steps = getattr(interaction, "steps", None) or []
            fc_steps = []
            for s in raw_steps:
                dumped = s.model_dump() if hasattr(s, "model_dump") else dict(s)
                history.append(dumped)
                if dumped.get("type") == "function_call":
                    fc_steps.append(s)

            if not fc_steps:
                final_text = getattr(interaction, "output_text", "") or ""
                if picked_num is None and best_step is not None and best_step.match_score >= 0.55:
                    picked_num = best_step.card_num
                    picked_id = best_step.card_id
                elif picked_num is None:
                    m = re.search(r"#(\d{1,7})", final_text)
                    if m:
                        cand = int(m.group(1))
                        if 1 <= cand <= ctx.total_cards:
                            picked_num = cand
                            picked_id = format_card_id(cand, ctx.total_cards)
                break

            found_early = False
            for fc in fc_steps:
                fc_name = fc.get("name") if isinstance(fc, dict) else getattr(fc, "name", "")
                fc_id = fc.get("id") if isinstance(fc, dict) else getattr(fc, "id", "")
                raw_args = fc.get("arguments") if isinstance(fc, dict) else getattr(fc, "arguments", {})
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})

                if fc_name == "query_audio_sql":
                    sql_str = str(args.get("sql", "")).strip().rstrip(";")
                    sql_queries_run += 1
                    _emit(
                        {
                            "engine": "regular",
                            "phase": "sql_query",
                            "sql": sql_str[:160],
                            "query_num": sql_queries_run,
                        }
                    )
                    try:
                        validate_readonly_sql(sql_str)
                        cur = con.execute(sql_str)
                        cols = [d[0] for d in (cur.description or [])]
                        rows = cur.fetchmany(30)
                        tool_res: dict[str, Any] = {
                            "columns": cols,
                            "rows": [dict(zip(cols, r)) for r in rows],
                            "row_count": len(rows),
                        }
                    except (SQLValidationError, Exception) as exc:
                        tool_res = {"error": str(exc)}

                elif fc_name == "sample_cards":
                    cnt = max(1, min(20, int(args.get("count", 10))))
                    sample_calls_run += 1
                    cur = con.execute(f"SELECT * FROM audio_cards USING SAMPLE {cnt}")
                    cols = [d[0] for d in (cur.description or [])]
                    rows = cur.fetchall()
                    _emit(
                        {
                            "engine": "regular",
                            "phase": "sql_query",
                            "sql": f"sample_cards({cnt})",
                            "query_num": sql_queries_run + 1,
                        }
                    )
                    tool_res = {
                        "columns": cols,
                        "rows": [dict(zip(cols, r)) for r in rows],
                        "row_count": len(rows),
                    }

                elif fc_name == "listen_to_card":
                    if len(checked_nums) >= max_listens:
                        tool_res = {"error": f"Listen budget ({max_listens}) exhausted. Give your final answer now."}
                    else:
                        try:
                            cnum = max(1, min(ctx.total_cards, int(args.get("card_num", 1))))
                        except (TypeError, ValueError):
                            cnum = 1
                        if cnum not in checked_set:
                            checked_set.add(cnum)
                            checked_nums.append(cnum)
                        cid = format_card_id(cnum, ctx.total_cards)

                        _emit(
                            {
                                "engine": "regular",
                                "phase": "listening",
                                "card_num": cnum,
                                "card_id": cid,
                                "step": len(checked_nums),
                            }
                        )
                        wav_bytes = ctx.get_card_wav(cnum)
                        parsed, ltel = inspect_audio_card_with_gemini(
                            client=client,
                            model=hydrogen_model,
                            card_num=cnum,
                            total_cards=ctx.total_cards,
                            wav_bytes=wav_bytes,
                            clue=ctx.clue,
                        )
                        telemetry_total = telemetry_total.plus(ltel)
                        is_match = bool(parsed.is_human_speech and parsed.clue_match_score >= 0.70)
                        step_obj = RegularAgentStep(
                            step=len(checked_nums),
                            card_num=cnum,
                            card_id=cid,
                            transcript=parsed.transcript,
                            match_score=parsed.clue_match_score,
                            is_match=is_match,
                            latency_ms=round(ltel.latency_ms, 1),
                        )
                        steps.append(step_obj)
                        if best_step is None or step_obj.match_score > best_step.match_score:
                            best_step = step_obj

                        _emit(
                            {
                                "engine": "regular",
                                "phase": "card_checked",
                                "step": step_obj.step,
                                "card_num": cnum,
                                "card_id": cid,
                                "is_human_speech": parsed.is_human_speech,
                                "match_score": parsed.clue_match_score,
                                "transcript": parsed.transcript,
                                "latency_ms": step_obj.latency_ms,
                            }
                        )
                        tool_res = {
                            "card_num": cnum,
                            "card_id": cid,
                            "is_human_speech": parsed.is_human_speech,
                            "transcript": parsed.transcript,
                            "clue_match_score": parsed.clue_match_score,
                            "reasoning": parsed.reasoning,
                            "remaining_listens": max_listens - len(checked_nums),
                        }
                        if is_match:
                            picked_num = cnum
                            picked_id = cid
                            found_early = True
                else:
                    tool_res = {"error": f"Unknown tool: {fc_name}"}

                history.append(
                    {
                        "type": "function_result",
                        "name": fc_name,
                        "call_id": fc_id,
                        "result": [{"type": "text", "text": json.dumps(tool_res, default=str)}],
                    }
                )
                if found_early:
                    break
            if found_early:
                break
    finally:
        con.close()

    wall_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
    budget_exhausted = picked_num is None and len(checked_nums) >= max_listens
    if picked_num is None and best_step is not None and best_step.match_score >= 0.55:
        picked_num = best_step.card_num
        picked_id = best_step.card_id

    found = picked_num == ctx.secret_card_num
    rationale = (
        f"Ran {sql_queries_run} SQL queries and listened to {len(checked_nums)}/{ctx.total_cards:,} cards"
        + (f" — pulled {picked_id}!" if found else " without finding your card.")
    )
    # Estimated API cost (Carbon-tier rates: ~$1.25/1M input, ~$5.00/1M output)
    reg_cost_usd = round(
        (telemetry_total.prompt_tokens / 1_000_000.0) * 1.25
        + ((telemetry_total.completion_tokens + telemetry_total.thought_tokens) / 1_000_000.0) * 5.00,
        5,
    )
    summary = RegularAgentSummary(
        picked_card_num=picked_num,
        picked_card_id=picked_id,
        found_target=found,
        cards_checked=checked_nums,
        steps=steps,
        budget_exhausted=budget_exhausted,
        wall_clock_ms=wall_ms,
        model_calls=telemetry_total.model_calls,
        tool_calls=sql_queries_run + sample_calls_run + len(steps),
        prompt_tokens=telemetry_total.prompt_tokens,
        completion_tokens=telemetry_total.completion_tokens + telemetry_total.thought_tokens,
        total_tokens=telemetry_total.total_tokens,
        estimated_cost_usd=reg_cost_usd,
        rationale=rationale,
    )
    _emit({"engine": "regular", "phase": "done", "summary": summary.model_dump(mode="json")})
    return summary


# ---------------------------------------------------------------------------
# Concurrent Side-by-Side Race Orchestrator
# ---------------------------------------------------------------------------


def run_audio_haystack_race(
    ctx: ScrambledRunContext,
    *,
    ares_top_k: int = 12,
    regular_budget: int = 12,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> AudioRaceResult:
    """Run ARES Swarm and Autonomous Carbon Agent simultaneously on the same scrambled haystack."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_ares = pool.submit(
            run_ares_audio_search,
            ctx,
            top_k=ares_top_k,
            on_event=on_event,
        )
        fut_reg = pool.submit(
            run_regular_agent_search,
            ctx,
            max_listens=regular_budget,
            on_event=on_event,
        )
        ares_res = fut_ares.result()
        reg_res = fut_reg.result()

    race = AudioRaceResult(
        run_id=ctx.run_id,
        clue=ctx.clue,
        total_cards=ctx.total_cards,
        secret_card_num=ctx.secret_card_num,
        secret_card_id=ctx.secret_card_id,
        ares=ares_res,
        regular=reg_res,
    )
    ctx.result = race
    return race
