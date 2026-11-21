"""Tests for the ARES Multimodal Explorer FastAPI app & narrow-waist integration."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from multimodal_demo.app import create_demo_app

MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _seed_mini_btsbot(cache_root: Path) -> None:
    ds_dir = cache_root / "btsbot"
    (ds_dir / "observations").mkdir(parents=True)
    (ds_dir / "media").mkdir(parents=True)
    (ds_dir / "media" / "sn_1.png").write_bytes(MINIMAL_PNG)
    (ds_dir / "media" / "decoy_1.png").write_bytes(MINIMAL_PNG)

    tbl = pa.Table.from_pydict(
        {
            "record_id": ["sn_1", "decoy_1"],
            "object_id": ["ZTF21sn", "ZTF21agn"],
            "magpsf": [18.3, 18.1],
            "sigmapsf": [0.05, 0.04],
            "dsnrms": [24.0, 29.0],
            "drb": [0.9995, 0.9991],
            "ndethist": [2, 140],
            "sgscore1": [0.04, 0.91],
            "distpsnr1": [1.8, 0.1],
            "nbad": [0, 2],
            "media_relpath": ["media/sn_1.png", "media/decoy_1.png"],
        }
    )
    pq.write_table(tbl, ds_dir / "observations" / "part-0000.parquet")
    (ds_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "dataset_id": "btsbot",
                "record_count": 2,
                "records": {
                    "sn_1": {"is_true_event": True, "is_decoy": False},
                    "decoy_1": {"is_true_event": False, "is_decoy": True},
                },
            }
        ),
        encoding="utf-8",
    )


def test_demo_app_search_investigate_duel_and_guards(tmp_path: Path) -> None:
    _seed_mini_btsbot(tmp_path)
    client = TestClient(create_demo_app(cache_root=tmp_path))

    # 1. Index HTML & datasets catalog
    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "Multimodal Haystack Explorer" in r_html.text

    r_ds = client.get("/api/datasets")
    assert r_ds.status_code == 200
    bts = next(d for d in r_ds.json()["datasets"] if d["dataset_id"] == "btsbot")
    assert bts["staged"] is True

    # 2. Guarded SQL search works & blocks ground_truth leak
    r_search = client.post(
        "/api/search",
        json={
            "dataset_id": "btsbot",
            "sql": "SELECT record_id, magpsf, drb FROM observations ORDER BY drb DESC",
        },
    )
    assert r_search.status_code == 200
    assert r_search.json()["row_count"] == 2
    assert r_search.json()["note"]["certification"] == "UNCERTIFIED"

    r_leak = client.post(
        "/api/search",
        json={
            "dataset_id": "btsbot",
            "sql": "SELECT * FROM read_json_auto('ground_truth.json')",
        },
    )
    assert r_leak.status_code == 400
    assert "SQL Guard Blocked" in r_leak.json()["detail"]

    # 3. Investigate runs ares.solve(), 0.0 rebuild drift, and counterfactual ablation
    r_inv = client.post(
        "/api/investigate",
        json={
            "dataset_id": "btsbot",
            "record_id": "sn_1",
            "use_live_gemini": False,
            "ablate_fields": [],
        },
    )
    assert r_inv.status_code == 200
    inv = r_inv.json()
    assert inv["invariants"]["board_drift"] == 0.0
    assert inv["invariants"]["certified_notes"] == 3
    assert inv["probe"]["attached_image_count"] == 1
    assert inv["probe"]["primary_answer"] == "real_supernova"

    # Ablating visual_media_evidence strips the image & flips offline verdict to insufficient_evidence
    r_abl = client.post(
        "/api/investigate",
        json={
            "dataset_id": "btsbot",
            "record_id": "sn_1",
            "use_live_gemini": False,
            "ablate_fields": ["visual_media_evidence"],
        },
    )
    assert r_abl.status_code == 200
    assert r_abl.json()["probe"]["attached_image_count"] == 0
    assert r_abl.json()["probe"]["primary_answer"] == "insufficient_evidence"

    # 4. Duel runs both positive & decoy side-by-side
    r_duel = client.post(
        "/api/duel",
        json={"dataset_id": "btsbot", "use_live_gemini": False},
    )
    assert r_duel.status_code == 200
    assert r_duel.json()["summary"]["real_detected"] is True
    assert r_duel.json()["summary"]["decoy_rejected"] is True

    # 5. Media endpoint serves verified PNG and blocks traversal
    r_png = client.get("/api/media/btsbot/sn_1.png")
    assert r_png.status_code == 200
    assert r_png.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    "dataset_id",
    ["caltech_camera_traps", "btsbot", "pyro_sdis", "urbansound8k"],
)
def test_real_staged_datasets_end_to_end_in_demo(dataset_id: str) -> None:
    cache_root = Path(__file__).resolve().parents[1] / ".data_cache"
    if not (cache_root / dataset_id / "observations").is_dir():
        pytest.skip(f"{dataset_id} not staged in {cache_root}")

    client = TestClient(create_demo_app(cache_root=cache_root))
    r_duel = client.post(
        "/api/duel",
        json={"dataset_id": dataset_id, "use_live_gemini": False},
    )
    assert r_duel.status_code == 200
    body = r_duel.json()
    assert body["real_candidate"]["invariants"]["board_drift"] == 0.0
    assert body["decoy_candidate"]["invariants"]["board_drift"] == 0.0
    assert body["real_candidate"]["media"]["image_url"] is not None
