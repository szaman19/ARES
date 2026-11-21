"""Tests for safe multimodal dataset staging, safety guards, and ARES search integration."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ares.sql_guard import SQLValidationError
from ares.store import NoteStore
from domain.multimodal_datasets import (
    DATASET_CATALOG,
    MultimodalSearchDataset,
    SafetyViolationError,
    safe_extract_single_json_from_zip,
    sha256_bytes,
    verify_magic_bytes,
    verify_safe_https_url,
    verify_safe_output_path,
)


MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_url_and_extension_safety_guards(tmp_path: Path) -> None:
    # Trusted HTTPS hosts pass
    assert verify_safe_https_url(
        "https://huggingface.co/datasets/MultimodalUniverse/btsbot/resolve/main/0000.parquet"
    )
    assert verify_safe_https_url(
        "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/a.jpg"
    )

    # Plain HTTP rejected
    with pytest.raises(SafetyViolationError, match="https://"):
        verify_safe_https_url("http://huggingface.co/datasets/foo.parquet")

    # Untrusted host rejected
    with pytest.raises(SafetyViolationError, match="Untrusted host"):
        verify_safe_https_url("https://drive.google.com/uc?id=12345")

    # Unsafe extensions (.pkl, .pt, .sh) rejected even on trusted hosts
    for bad in ("model.pkl", "weights.pt", "install.sh", "payload.pickle"):
        with pytest.raises(SafetyViolationError, match="Forbidden unsafe"):
            verify_safe_https_url(f"https://huggingface.co/datasets/org/repo/{bad}")

    # Path traversal & unsafe output extension rejected
    with pytest.raises(SafetyViolationError, match="outside cache root"):
        verify_safe_output_path(tmp_path / ".." / "escape.parquet", tmp_path)

    with pytest.raises(SafetyViolationError, match="Forbidden unsafe"):
        verify_safe_output_path(tmp_path / "bad.pkl", tmp_path)


def test_zip_slip_and_non_json_archive_blocked() -> None:
    # Valid single JSON inside ZIP succeeds
    buf_ok = io.BytesIO()
    with zipfile.ZipFile(buf_ok, "w") as zf:
        zf.writestr("labels.json", '{"ok": true}')
    extracted = safe_extract_single_json_from_zip(buf_ok.getvalue())
    assert json.loads(extracted.decode("utf-8")) == {"ok": True}

    # Path traversal blocked
    buf_slip = io.BytesIO()
    with zipfile.ZipFile(buf_slip, "w") as zf:
        zf.writestr("../../evil.json", '{"bad": true}')
    with pytest.raises(SafetyViolationError, match="Zip-slip"):
        safe_extract_single_json_from_zip(buf_slip.getvalue())

    # Non-JSON file inside ZIP blocked
    buf_sh = io.BytesIO()
    with zipfile.ZipFile(buf_sh, "w") as zf:
        zf.writestr("run.sh", "#!/bin/sh\necho nope")
    with pytest.raises(SafetyViolationError, match="Non-JSON"):
        safe_extract_single_json_from_zip(buf_sh.getvalue())


def test_magic_byte_verification() -> None:
    verify_magic_bytes(MINIMAL_PNG, ".png")
    verify_magic_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\xff\xd9", ".jpg")
    verify_magic_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 32, ".wav")

    with pytest.raises(SafetyViolationError, match="PNG"):
        verify_magic_bytes(b"not a png payload", ".png")

    with pytest.raises(SafetyViolationError, match="JPEG"):
        verify_magic_bytes(b"PK\x03\x04fakejpeg", ".jpg")

    with pytest.raises(SafetyViolationError, match="RIFF/WAVE"):
        verify_magic_bytes(b"ID3notawavfile" + b"\x00" * 40, ".wav")


def test_ares_search_verifier_and_ground_truth_isolation(tmp_path: Path) -> None:
    assert set(DATASET_CATALOG.keys()) == {
        "btsbot",
        "caltech_camera_traps",
        "pyro_sdis",
        "urbansound8k",
    }

    ds_dir = tmp_path / "btsbot"
    obs_dir = ds_dir / "observations"
    media_dir = ds_dir / "media"
    obs_dir.mkdir(parents=True)
    media_dir.mkdir(parents=True)

    # Write verified PNG media stamps
    (media_dir / "sn_real_1.png").write_bytes(MINIMAL_PNG)
    (media_dir / "agn_decoy_2.png").write_bytes(MINIMAL_PNG)

    table = pa.Table.from_pydict(
        {
            "record_id": ["sn_real_1", "agn_decoy_2"],
            "object_id": ["ZTF21aaa", "ZTF21bbb"],
            "magpsf": [18.4, 18.2],
            "drb": [0.999, 0.998],
            "ndethist": [2, 145],
            "sgscore1": [0.05, 0.88],
            "media_relpath": ["media/sn_real_1.png", "media/agn_decoy_2.png"],
        }
    )
    pq.write_table(table, obs_dir / "part-0000.parquet")

    gt_payload = {
        "dataset_id": "btsbot",
        "record_count": 2,
        "records": {
            "sn_real_1": {"is_true_event": True, "is_decoy": False},
            "agn_decoy_2": {"is_true_event": False, "is_decoy": True},
        },
    }
    (ds_dir / "ground_truth.json").write_text(json.dumps(gt_payload), encoding="utf-8")

    ds = MultimodalSearchDataset.open("btsbot", cache_root=tmp_path)
    store = NoteStore()

    # 1. SQL search works immediately via ARES DuckDBTrustedRunner
    rows, sql_note = ds.search_sql(
        "SELECT record_id, magpsf, drb, ndethist, sgscore1 "
        "FROM observations WHERE drb > 0.99 ORDER BY ndethist ASC",
        note_store=store,
    )
    assert len(rows) == 2
    assert rows[0]["record_id"] == "sn_real_1"
    assert sql_note.certification == "UNCERTIFIED"

    # 2. Ground-truth labels are NOT in SQL columns and external file reads are blocked
    sample_rows, _ = ds.search_sql("SELECT * FROM observations LIMIT 1")
    exposed_cols = set(sample_rows[0].keys())
    assert "is_true_event" not in exposed_cols
    assert "is_SN" not in exposed_cols

    with pytest.raises(SQLValidationError):
        ds.search_sql(f"SELECT * FROM read_json_auto('{ds_dir / 'ground_truth.json'}')")

    # 3. Deterministic TRUSTED verifier emits CERTIFIED Note + verified PNG ArtifactRef
    ver_res, ver_note = ds.inspect_candidate("sn_real_1", note_store=store)
    assert ver_res.passed is True
    assert ver_note.certification == "CERTIFIED"
    assert len(ver_note.artifacts) == 1
    art = ver_note.artifacts[0]
    assert art.kind == "image"
    assert art.mime_type == "image/png"
    assert art.produced_by_verifier is True
    assert art.sha256 == sha256_bytes(MINIMAL_PNG)
    assert Path(art.path).is_file()

    # 4. Scoring computes accuracy & decoy rejection rate cleanly
    metrics = ds.score_predictions({"sn_real_1": True, "agn_decoy_2": False})
    assert metrics["accuracy"] == 1.0
    assert metrics["decoy_rejection_rate"] == 1.0


@pytest.mark.parametrize(
    "dataset_id",
    ["btsbot", "caltech_camera_traps", "pyro_sdis", "urbansound8k"],
)
def test_staged_real_datasets_if_cached(dataset_id: str) -> None:
    cache_root = Path(__file__).resolve().parents[1] / ".data_cache"
    if not (cache_root / dataset_id / "observations").is_dir():
        pytest.skip(f"{dataset_id} not staged locally in {cache_root}")

    ds = MultimodalSearchDataset.open(dataset_id, cache_root=cache_root)
    rows, sql_note = ds.search_sql(
        "SELECT record_id, media_relpath FROM observations WHERE media_relpath != '' LIMIT 3"
    )
    assert len(rows) >= 1
    assert sql_note.certification == "UNCERTIFIED"

    # Inspect real candidate media & verify ArtifactRef integrity
    first_id = rows[0]["record_id"]
    ver_res, ver_note = ds.inspect_candidate(first_id, fetch_on_demand=False)
    assert ver_res.passed is True
    assert ver_note.certification == "CERTIFIED"
    assert len(ver_note.artifacts) == 1
    art = ver_note.artifacts[0]
    assert Path(art.path).is_file()
    assert len(art.sha256) == 64

    # Verify ground truth is withheld from SQL and available for scoring
    gt = ds.load_ground_truth()
    assert first_id in gt
    score = ds.score_predictions({first_id: bool(gt[first_id]["is_true_event"])})
    assert score["accuracy"] == 1.0

