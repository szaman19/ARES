"""Tests for the 100,000-card ARES Audio Haystack ("Is This Your Card?") domain and server."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi.testclient import TestClient

from domain.audio_haystack import (
    AudioHaystackStore,
    ScrambledRunContext,
    _AudioInspectionOutput,
    _SingleSQLHypothesis,
    generate_sample_voice_wav,
    pcm_float_to_wav_bytes,
    render_card_spectrogram,
    run_ares_audio_search,
    synthesize_card_pcm,
)
from models.gemini_provider import ModelTelemetry
from ui.audio_haystack_app import app


def _mock_sql_proposer(*, angle_name: str, excluded_nums: list[int] | None = None, **_kwargs):
    excl = excluded_nums or []
    not_in = f"AND card_num NOT IN ({','.join(str(x) for x in excl)})" if excl else ""
    queries = {
        "vocal_formants": f"SELECT card_num FROM audio_cards WHERE voice_band_ratio >= 0.25 {not_in} ORDER BY (harmonicity_score + formant_balance) DESC LIMIT 4",
        "speech_rhythm": f"SELECT card_num FROM audio_cards WHERE voice_band_ratio >= 0.25 {not_in} ORDER BY syllable_mod_score DESC LIMIT 4",
        "rare_cluster": f"SELECT card_num FROM audio_cards WHERE syllable_mod_score > 0.105 {not_in} ORDER BY formant_balance DESC LIMIT 4",
        "broadband_voice": f"SELECT card_num FROM audio_cards WHERE spectral_flatness BETWEEN 0.001 AND 0.45 {not_in} ORDER BY (harmonicity_score + syllable_mod_score) DESC LIMIT 4",
    }
    sql = queries.get(
        angle_name,
        f"SELECT card_num FROM audio_cards WHERE syllable_mod_score > 0.105 {not_in} ORDER BY (harmonicity_score + formant_balance) DESC LIMIT 4",
    )
    return (
        _SingleSQLHypothesis(query_sql=sql, rationale=f"Mock {angle_name}"),
        ModelTelemetry(prompt_tokens=80, completion_tokens=30, total_tokens=110, model_calls=1, latency_ms=8.0),
    )


def test_100k_haystack_and_ares_parallel_swarm_end_to_end(tmp_path: Path) -> None:
    store = AudioHaystackStore(total_cards=100_000, base_seed=1234, work_dir=tmp_path)
    assert store.total_cards == 100_000
    assert store.build_ms > 0

    user_wav = generate_sample_voice_wav()
    ctx = ScrambledRunContext(
        run_id="test100k",
        clue="What's up?",
        store=store,
        user_wav_bytes=user_wav,
        secret_card_num=42137,
    )
    assert ctx.secret_card_id == "#042137"

    # Verify rms_db is NOT an outlier (hundreds of cards share the same loudness band!)
    con = ctx.build_duckdb()
    try:
        same_rms_count = con.execute(
            "SELECT COUNT(*) FROM audio_cards WHERE ABS(rms_db - ?) <= 0.5",
            [ctx.user_features.rms_db],
        ).fetchone()[0]
    finally:
        con.close()
    assert same_rms_count > 1000

    # Exercise the full ARESRunner parallel swarm + NoteStore + fusion/patch + spectrogram path!
    def _mock_inspector(*, card_num: int, **_kwargs):
        is_target = card_num == 42137
        return (
            _AudioInspectionOutput(
                is_human_speech=is_target,
                transcript="What's up?" if is_target else "[synthetic babble]",
                clue_match_score=0.98 if is_target else 0.05,
                reasoning="Clear human voice" if is_target else "Synthetic noise",
            ),
            ModelTelemetry(prompt_tokens=100, completion_tokens=25, total_tokens=125, model_calls=1, latency_ms=12.0),
        )

    summary = run_ares_audio_search(
        ctx,
        top_k=12,
        max_concurrent=6,
        inspector_fn=_mock_inspector,
        sql_proposer_fn=_mock_sql_proposer,
        enable_probe=False,
    )
    assert summary.total_cards == 100_000
    assert summary.pruned_count >= 100_000 - 12
    assert 42137 in summary.survivor_card_nums
    assert summary.found_target is True
    assert summary.picked_card_num == 42137
    assert summary.picked_card_id == "#042137"
    assert summary.spectrogram_artifact is not None
    assert Path(summary.spectrogram_artifact.path).is_file()


def test_short_noisy_laptop_mic_recording_lands_in_ares_top12(tmp_path: Path) -> None:
    """Verify even a short 1.4s noisy/breathy laptop mic utterance lands in ARES top-12 of 100k."""
    import numpy as np

    store = AudioHaystackStore(total_cards=100_000, base_seed=5678, work_dir=tmp_path)
    # Simulate a short 1.4s voice burst surrounded by room noise (just like a casual laptop mic clip)
    voice_burst = synthesize_card_pcm(9911, "hard_voice_decoy", n_samples=22400)
    rng = np.random.default_rng(77)
    noisy_clip = rng.normal(0.0, 0.015, size=48000).astype(np.float32)
    noisy_clip[12000 : 12000 + 22400] += voice_burst * 0.85 + rng.normal(0.0, 0.04, size=22400).astype(np.float32)

    ctx = ScrambledRunContext(
        run_id="shortmic",
        clue="hello",
        store=store,
        user_wav_bytes=pcm_float_to_wav_bytes(noisy_clip),
        secret_card_num=5545,
    )

    def _mock_inspector(*, card_num: int, **_kwargs):
        is_target = card_num == 5545
        return (
            _AudioInspectionOutput(
                is_human_speech=is_target,
                transcript="Hello" if is_target else "[babble]",
                clue_match_score=0.95 if is_target else 0.05,
                reasoning="match" if is_target else "no",
            ),
            ModelTelemetry(prompt_tokens=50, completion_tokens=20, total_tokens=70, model_calls=1, latency_ms=5.0),
        )

    res = run_ares_audio_search(
        ctx,
        top_k=12,
        max_concurrent=6,
        inspector_fn=_mock_inspector,
        sql_proposer_fn=_mock_sql_proposer,
        enable_probe=False,
    )
    assert 5545 in res.survivor_card_nums
    assert res.found_target is True


def test_deterministic_spectrogram_artifact(tmp_path: Path) -> None:
    pcm = synthesize_card_pcm(42, "hard_voice_decoy")
    art1 = render_card_spectrogram(pcm, "#000042", tmp_path / "a", reference_pcm=pcm)
    art2 = render_card_spectrogram(pcm, "#000042", tmp_path / "b", reference_pcm=pcm)
    assert art1.produced_by_verifier is True
    assert art1.sha256 == art2.sha256
    assert Path(art1.path).is_file()


def test_fastapi_endpoints_and_audio_playback() -> None:
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    assert h.json()["total_cards"] == 100_000

    card_audio = client.get("/api/haystack/cards/42137/audio")
    assert card_audio.status_code == 200
    assert card_audio.headers["content-type"] == "audio/wav"
    assert card_audio.content[:4] == b"RIFF"

    wav_bytes = pcm_float_to_wav_bytes(synthesize_card_pcm(777, "hard_voice_decoy"))
    resp = client.post(
        "/api/runs",
        json={
            "clue": "testing one two three",
            "use_sample": False,
            "audio_wav_b64": base64.b64encode(wav_bytes).decode("ascii"),
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["total_cards"] == 100_000
    assert payload["secret_card_id"].startswith("#")

    run_card = client.get(
        f"/api/haystack/cards/{payload['secret_card_num']}/audio",
        params={"run_id": payload["run_id"]},
    )
    assert run_card.status_code == 200
    assert run_card.content[:4] == b"RIFF"
