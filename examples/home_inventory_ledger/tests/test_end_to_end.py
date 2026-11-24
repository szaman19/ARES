"""End-to-end integration tests covering ingest, pipeline, follow-up CERTIFIED flip, FastAPI, and 5 Sigma eval."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.server import app
from eval.generate_dev_clips import ensure_dev_assets
from eval.run_eval import run_evaluation_suite, wilson_interval


def test_end_to_end_api_walkthrough_followup_cart_and_export(tmp_path: Path) -> None:
    ensure_dev_assets()
    client = TestClient(app)

    # 1. Config endpoint
    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    assert cfg.json()["models"]["understander"] == "gemini-4-carbon"
    assert cfg.json()["models"]["extractor"] == "gemini-4-hydrogen"
    assert cfg.json()["models"]["decision_probe"] == "gemini-decision-1"

    # 2. Sample Policy parse
    pol_res = client.post("/api/policy", data={"use_sample": "true", "replay": "true"})
    assert pol_res.status_code == 200
    assert pol_res.json()["verifier_passed"] is True
    assert pol_res.json()["policy"]["coverage_c_limit"] == 7500.0

    # 3. Walkthrough pipeline -> Verdict + Contradiction + VoI Ask + 0 Drift
    walk_res = client.post(
        "/api/demo/walkthrough",
        data={"coverage_limit": "7500", "replay": "true"},
    )
    assert walk_res.status_code == 200
    board_0 = walk_res.json()
    session_id = board_0["session_id"]
    assert board_0["decision"]["verdict"] == "likely_underinsured"

    assert board_0["telemetry"]["board_drift_count"] == 0
    items_0 = list(board_0["items"].values())
    # Deduplicated 7 raw observations (2 keyboard sightings) -> 6 physical items!
    assert len(items_0) == 6

    # Check narration conflict caught on Bose headphones ("Sony" spoken vs BOSE visual)
    hp = next(i for i in items_0 if i["category"] == "headphones")
    assert "narration_conflict" in hp["flags"]

    # Check laptop starts UNCERTIFIED (no invented unit_price!) and is top VoI ask
    laptop_0 = next(i for i in items_0 if i["category"] == "laptop")
    assert laptop_0["price_status"] == "UNCERTIFIED"
    assert laptop_0["unit_price"] is None
    assert board_0["requests"][0]["item_id"] == laptop_0["item_id"]

    # 4. Follow-up close-up clip (`Model A2992`) -> Laptop flips to CERTIFIED $1,999!
    fup_res = client.post(f"/api/session/{session_id}/demo-followup")
    assert fup_res.status_code == 200
    board_1 = fup_res.json()
    assert board_1["telemetry"]["board_drift_count"] == 0

    laptop_1 = board_1["items"][laptop_0["item_id"]]
    assert laptop_1["price_status"] == "CERTIFIED"
    assert laptop_1["unit_price"] == 1999.0
    assert laptop_1["matched_listing"]["sku"] == "A2992-M3PRO"
    assert board_1["decision"]["certified_total"] > board_0["decision"]["certified_total"]
    # With $7,374.99 certified + keyboard + Bose ranges > $8,000 limit -> underinsured!
    assert board_1["decision"]["verdict"] == "underinsured"

    # 5. Shopify Replacement Cart + CSV/HTML Schedule of Loss exports
    cart_res = client.post(f"/api/session/{session_id}/cart")
    assert cart_res.status_code == 200
    cart = cart_res.json()
    assert cart["line_count"] >= 5
    assert cart["requires_user_confirmation"] is True

    csv_res = client.get(f"/api/session/{session_id}/export.csv")
    assert csv_res.status_code == 200
    assert "Arithmetic Verified: True" in csv_res.text

    html_res = client.get(f"/api/session/{session_id}/export.html")
    assert html_res.status_code == 200
    assert "Pre-Loss Schedule of Contents" in html_res.text


def test_5_sigma_eval_and_stress_suite(tmp_path: Path) -> None:
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and 0.40 < hi < 0.50

    report = run_evaluation_suite(
        manifest_path=None,
        replay_mode=True,
        run_stress=True,
        out_dir=tmp_path / "results",
    )
    sys_map = report["systems"]
    # Ledger has 0 hallucinated prices, 0 duplicates, and catches the planted contradiction!
    assert sys_map["Ledger_0_asks"]["raw_counts"]["halluc_price"] == 0
    assert sys_map["Ledger_1_ask"]["raw_counts"]["halluc_price"] == 0
    assert sys_map["Ledger_1_ask"]["raw_counts"]["dup"] == 0
    assert sys_map["Ledger_1_ask"]["raw_counts"]["contra_caught"] == 1
    # Baseline B0 has hallucinated prices, 1 duplicate keyboard, and 0 contradictions caught
    assert sys_map["B0"]["raw_counts"]["halluc_price"] > 0
    assert sys_map["B0"]["raw_counts"]["dup"] == 1
    assert sys_map["B0"]["raw_counts"]["contra_caught"] == 0

    # Stress ladders executed and maintained 0 board drift + blocked blurry/dark false certifications
    stress_by_name = {s["transform"]: s for s in report["stress_ladders"]}
    assert stress_by_name["darken_severe"]["certified_items"] == 0
    assert stress_by_name["blur_severe"]["certified_items"] == 0
    assert all(s["board_drift"] == 0 for s in report["stress_ladders"])
