"""FastAPI backend for the locally runnable ARES Multimodal Explorer & Probe Demo.

Run locally:
    ./scripts/run_multimodal_demo.sh
Then open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from adapters.sql_guard import SQLValidationError
from contracts import ArtifactRef, DecisionQuestion, InvestigationBoard
from domain.multimodal_datasets import (
    CANONICAL_MULTIMODAL_FIELDS,
    DATASET_CATALOG,
    MultimodalSearchDataset,
    SafetyViolationError,
    extract_multimodal_note_patches,
    verify_safe_output_path,
)
from fusion.patch import (
    compute_board_drift,
    create_initial_board,
    deterministic_patch_from_notes,
    rebuild_board_from_notes,
    validate_and_merge_patch,
)
from models.decision_probe import (
    BoardEvaluation,
    DecisionProbeClient,
    QuestionEvaluation,
    ablate_board_fields,
    total_variation_distance,
)


def _filter_artifacts_for_board(
    base_board: InvestigationBoard,
    ablated_board: InvestigationBoard,
    artifact_refs: list[ArtifactRef] | None,
) -> list[ArtifactRef] | None:
    """Keep only `ArtifactRef`s still cited by non-unknown fields on `ablated_board`."""
    if not artifact_refs:
        return None
    has_any_cited = any(
        bool(f.artifact_ids) for f in base_board.fields.values() if f.status != "unknown"
    )
    if not has_any_cited:
        return None
    surviving_ids = {
        aid
        for f in ablated_board.fields.values()
        if f.status != "unknown"
        for aid in f.artifact_ids
    }
    return [a for a in artifact_refs if a.artifact_id in surviving_ids]

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load GEMINI_API_KEY from worktree .env or parent repo .env (git worktrees don't copy .env)
for candidate_env in (
    WORKTREE_ROOT / ".env",
    WORKTREE_ROOT.parent.parent / ".env",
):
    if candidate_env.is_file():
        load_dotenv(candidate_env, override=False)


class SearchRequest(BaseModel):
    dataset_id: str
    sql: str


class InvestigateRequest(BaseModel):
    dataset_id: str
    record_id: str
    use_live_gemini: bool = True
    ablate_fields: list[str] = Field(default_factory=list)
    reveal_ground_truth: bool = False


class DuelRequest(BaseModel):
    dataset_id: str
    use_live_gemini: bool = True


def _has_gemini_key() -> bool:
    return bool((os.environ.get("GEMINI_API_KEY") or "").strip())


def _media_urls_for_row(dataset_id: str, row: dict[str, Any], artifacts: list[ArtifactRef] | None = None) -> dict[str, str | None]:
    image_url: str | None = None
    rel = str(row.get("media_relpath") or "")
    if rel:
        image_url = f"/api/media/{dataset_id}/{Path(rel).name}"
    elif artifacts:
        image_url = f"/api/media/{dataset_id}/{Path(artifacts[0].path).name}"

    audio_url: str | None = None
    wav_rel = str(row.get("wav_relpath") or "")
    if wav_rel:
        audio_url = f"/api/audio/{dataset_id}/{Path(wav_rel).name}"

    return {"image_url": image_url, "audio_url": audio_url}


def _heuristic_fallback_eval(
    ds: MultimodalSearchDataset,
    board: InvestigationBoard,
    row: dict[str, Any],
    has_attached_image: bool,
) -> BoardEvaluation:
    """Transparent offline fallback when live Gemini API is disabled or unreachable."""
    questions = ds.spec.decision_questions
    primary_q = questions[0]
    opts = primary_q.options
    pos_opt = opts[0]
    decoy_opt = opts[1]
    abstain_opt = opts[2] if len(opts) > 2 else "insufficient_evidence"

    if not has_attached_image:
        # Without visual media, tabular triggers are genuinely ambiguous!
        probs = {pos_opt: 0.34, decoy_opt: 0.33, abstain_opt: 0.33}
        ans = abstain_opt
        conf = 0.34
        yn_p = 0.22
    else:
        # Offline structural proxy using contextual + media availability
        if ds.name == "btsbot":
            score_pos = 0.86 if (row.get("ndethist", 99) <= 15 and row.get("sgscore1", 1.0) < 0.45) else 0.16
        elif ds.name == "caltech_camera_traps":
            score_pos = 0.82 if (row.get("max_bbox_area_frac", 0.0) >= 0.015 and row.get("hour_local", 12) in (0,1,2,3,4,5,6,18,19,20,21,22,23)) else 0.30
        elif ds.name == "pyro_sdis":
            score_pos = 0.80 if row.get("warm_glare_fraction", 0.0) < 0.08 else 0.28
        else:
            score_pos = 0.81 if row.get("salience", 2) == 1 else 0.29
        probs = {
            pos_opt: round(score_pos, 4),
            decoy_opt: round(max(0.0, 0.94 - score_pos), 4),
            abstain_opt: 0.06,
        }
        ans = max(probs, key=lambda k: probs[k])
        conf = probs[ans]
        yn_p = score_pos

    evals: dict[str, QuestionEvaluation] = {
        primary_q.question_id: QuestionEvaluation(
            question_id=primary_q.question_id,
            answer=ans,
            confidence=conf,
            probabilities=probs,
        )
    }
    if len(questions) > 1:
        q2 = questions[1]
        evals[q2.question_id] = QuestionEvaluation(
            question_id=q2.question_id,
            answer="yes" if yn_p >= 0.5 else "no",
            confidence=max(yn_p, 1.0 - yn_p),
            probabilities={"yes": round(yn_p, 4), "no": round(1.0 - yn_p, 4)},
            raw_score=yn_p,
        )
    return BoardEvaluation(answers=evals)


def _evaluate_and_probe_multimodal_board(
    ds: MultimodalSearchDataset,
    base_board: InvestigationBoard,
    active_board: InvestigationBoard,
    artifacts: list[ArtifactRef],
    row: dict[str, Any],
    *,
    use_live_gemini: bool,
) -> dict[str, Any]:
    """Run Decision-1 evaluation + per-field counterfactual ablations with artifact filtering."""
    questions: list[DecisionQuestion] = list(ds.spec.decision_questions)
    primary_id = questions[0].question_id
    active_arts = _filter_artifacts_for_board(base_board, active_board, artifacts) or []

    mode_used = "live_gemini_decision_1"
    probe_error: str | None = None
    api_calls = 0

    if use_live_gemini and _has_gemini_key():
        try:
            client = DecisionProbeClient()
            active_eval = client.evaluate_board(
                active_board,
                questions,
                artifact_refs=active_arts or None,
            )
            api_calls += 1

            # Compute per-field counterfactual reliance (`influence`) on active non-unknown fields,
            # filtering `artifacts` per ablated board so `visual_media_evidence` isolates image impact!
            non_unknown = [
                fn for fn, f in active_board.fields.items() if f.status != "unknown"
            ]
            swings: dict[str, float] = {fn: 0.0 for fn in active_board.fields}
            counterfactual_snapshots: dict[str, dict[str, Any]] = {}
            base_probs = active_eval.answers[primary_id].probabilities

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                fut_map = {}
                for fn in non_unknown:
                    abl_b = ablate_board_fields(active_board, [fn])
                    abl_arts = _filter_artifacts_for_board(active_board, abl_b, active_arts)
                    fut_map[
                        pool.submit(
                            client.evaluate_board,
                            abl_b,
                            questions,
                            artifact_refs=abl_arts or None,
                        )
                    ] = fn

                for fut in concurrent.futures.as_completed(fut_map):
                    fn = fut_map[fut]
                    abl_ev = fut.result()
                    api_calls += 1
                    q_ans = abl_ev.answers[primary_id]
                    tvd = total_variation_distance(base_probs, q_ans.probabilities)
                    swings[fn] = round(tvd, 4)
                    counterfactual_snapshots[fn] = {
                        "answer_when_ablated": q_ans.answer,
                        "confidence_when_ablated": round(q_ans.confidence, 4),
                        "tvd_swing": round(tvd, 4),
                    }
        except Exception as exc:
            mode_used = "deterministic_fallback"
            probe_error = str(exc)
            active_eval = _heuristic_fallback_eval(ds, active_board, row, bool(active_arts))
            swings = {}
            counterfactual_snapshots = {}
            base_probs = active_eval.answers[primary_id].probabilities
            for fn, f in active_board.fields.items():
                if f.status == "unknown":
                    swings[fn] = 0.0
                    continue
                abl_b = ablate_board_fields(active_board, [fn])
                abl_arts = _filter_artifacts_for_board(active_board, abl_b, active_arts)
                abl_ev = _heuristic_fallback_eval(ds, abl_b, row, bool(abl_arts))
                q_ans = abl_ev.answers[primary_id]
                tvd = total_variation_distance(base_probs, q_ans.probabilities)
                swings[fn] = round(tvd, 4)
                counterfactual_snapshots[fn] = {
                    "answer_when_ablated": q_ans.answer,
                    "confidence_when_ablated": round(q_ans.confidence, 4),
                    "tvd_swing": round(tvd, 4),
                }
    else:
        mode_used = "deterministic_offline"
        active_eval = _heuristic_fallback_eval(ds, active_board, row, bool(active_arts))
        swings = {}
        counterfactual_snapshots = {}
        base_probs = active_eval.answers[primary_id].probabilities
        for fn, f in active_board.fields.items():
            if f.status == "unknown":
                swings[fn] = 0.0
                continue
            abl_b = ablate_board_fields(active_board, [fn])
            abl_arts = _filter_artifacts_for_board(active_board, abl_b, active_arts)
            abl_ev = _heuristic_fallback_eval(ds, abl_b, row, bool(abl_arts))
            q_ans = abl_ev.answers[primary_id]
            tvd = total_variation_distance(base_probs, q_ans.probabilities)
            swings[fn] = round(tvd, 4)
            counterfactual_snapshots[fn] = {
                "answer_when_ablated": q_ans.answer,
                "confidence_when_ablated": round(q_ans.confidence, 4),
                "tvd_swing": round(tvd, 4),
            }

    total_swing = sum(swings.values())
    influences = {
        fn: round(swings[fn] / total_swing, 4) if total_swing > 1e-9 else 0.0
        for fn in active_board.fields
    }
    primary_ans = active_eval.answers[primary_id]
    gate_ans = (
        active_eval.answers[questions[1].question_id] if len(questions) > 1 else None
    )

    return {
        "mode_used": mode_used,
        "probe_error": probe_error,
        "api_calls": api_calls,
        "attached_image_count": len(active_arts),
        "primary_question": questions[0].question,
        "primary_answer": primary_ans.answer,
        "primary_confidence": round(primary_ans.confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in primary_ans.probabilities.items()},
        "verified_gate_answer": gate_ans.answer if gate_ans else "unknown",
        "verified_gate_confidence": round(gate_ans.confidence, 4) if gate_ans else 0.0,
        "raw_tvd_swings": swings,
        "field_influences": influences,
        "counterfactual_snapshots": counterfactual_snapshots,
    }


def _investigate_single_candidate(
    cache_root: Path,
    req: InvestigateRequest,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    ds = MultimodalSearchDataset.open(req.dataset_id, cache_root=cache_root)
    row = ds.get_observation_row(req.record_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Record {req.record_id!r} not found in {req.dataset_id}.",
        )

    # 1. Execute parallel TRUSTED verifiers via ARES `solve()`
    note_store, artifacts = ds.run_ares_candidate_verifiers(
        req.record_id, fetch_on_demand=True
    )

    # 2. Fuse into InvestigationBoard & verify Invariants I1-I7 + I3 0.0 rebuild drift
    seed_board = create_initial_board(
        ds.spec.decision_questions,
        seed_fields=CANONICAL_MULTIMODAL_FIELDS,
    )
    patch = deterministic_patch_from_notes(
        note_store.snapshot(),
        note_extractor=extract_multimodal_note_patches,
    )
    base_board, patch_report = validate_and_merge_patch(
        seed_board,
        patch,
        note_store,
        protected_fields=CANONICAL_MULTIMODAL_FIELDS,
        verify_artifact_files=True,
    )
    rebuilt = rebuild_board_from_notes(
        note_store,
        ds.spec.decision_questions,
        seed_fields=CANONICAL_MULTIMODAL_FIELDS,
        protected_fields=CANONICAL_MULTIMODAL_FIELDS,
        note_extractor=extract_multimodal_note_patches,
        verify_artifact_files=True,
    )
    drift = compute_board_drift(base_board, rebuilt)

    # 3. Apply interactive user counterfactual ablations (e.g. hiding `visual_media_evidence`)
    active_board = (
        ablate_board_fields(base_board, req.ablate_fields)
        if req.ablate_fields
        else base_board.model_copy(deep=True)
    )

    # 4. Run Decision-1 evaluation & counterfactual reliance probe
    probe_summary = _evaluate_and_probe_multimodal_board(
        ds,
        base_board,
        active_board,
        artifacts,
        row,
        use_live_gemini=req.use_live_gemini,
    )

    for fname, fobj in active_board.fields.items():
        fobj.influence = probe_summary["field_influences"].get(fname, 0.0)

    urls = _media_urls_for_row(req.dataset_id, row, artifacts)
    gt_info = None
    if req.reveal_ground_truth:
        gt_info = ds.load_ground_truth().get(str(req.record_id))

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "dataset_id": req.dataset_id,
        "record_id": str(req.record_id),
        "observation": row,
        "media": urls,
        "board": active_board.model_dump(),
        "notes": [n.model_dump() for n in note_store.snapshot()],
        "artifacts": [a.model_dump() for a in artifacts],
        "invariants": {
            "board_drift": round(drift, 6),
            "applied_fields": patch_report.applied_fields,
            "certified_notes": sum(
                1 for n in note_store.snapshot() if n.certification == "CERTIFIED"
            ),
        },
        "probe": probe_summary,
        "ablated_fields": list(req.ablate_fields),
        "ground_truth": gt_info,
        "duration_ms": elapsed_ms,
    }


def create_demo_app(cache_root: str | Path | None = None) -> FastAPI:
    eff_cache_root = Path(cache_root or (WORKTREE_ROOT / ".data_cache")).resolve()

    app = FastAPI(
        title="ARES Multimodal Dataset Explorer & Counterfactual Probe",
        version="0.1.0",
    )
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/datasets")
    def list_datasets() -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for ds_id, spec in DATASET_CATALOG.items():
            ds_dir = eff_cache_root / ds_id
            staged = (ds_dir / "observations").is_dir() and any(
                (ds_dir / "observations").glob("*.parquet")
            )
            manifest: dict[str, Any] = {}
            if staged:
                manifest = MultimodalSearchDataset.open(
                    ds_id, cache_root=eff_cache_root
                ).load_manifest()
            items.append(
                {
                    "dataset_id": ds_id,
                    "short_label": spec.short_label,
                    "emoji": spec.emoji,
                    "title": spec.title,
                    "domain": spec.domain,
                    "source_url": spec.source_url,
                    "primary_media_kind": spec.primary_media_kind,
                    "positive_option": spec.positive_option,
                    "positive_label_description": spec.positive_label_description,
                    "decoy_description": spec.decoy_description,
                    "schema_summary": spec.schema_summary,
                    "preview_columns": list(spec.preview_columns),
                    "staged": staged,
                    "row_count": manifest.get("row_count", 0),
                    "positive_count": manifest.get("positive_count", 0),
                    "decoy_count": manifest.get("decoy_count", 0),
                    "staged_media_files": manifest.get("staged_media_files", 0),
                    "size_mb": round(
                        manifest.get("total_dataset_bytes", 0) / (1024 * 1024), 2
                    ),
                    "presets": [
                        {
                            "label": p.label,
                            "description": p.description,
                            "sql": p.sql,
                        }
                        for p in spec.preset_magnets
                    ],
                }
            )
        return {
            "gemini_key_present": _has_gemini_key(),
            "cache_root": str(eff_cache_root),
            "datasets": items,
        }

    @app.post("/api/search")
    def run_sql_search(req: SearchRequest) -> dict[str, Any]:
        try:
            ds = MultimodalSearchDataset.open(req.dataset_id, cache_root=eff_cache_root)
            qres, note = ds.search_sql(req.sql)
        except SQLValidationError as exc:
            raise HTTPException(status_code=400, detail=f"SQL Guard Blocked Query: {exc}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Query error: {exc}") from exc

        enriched_rows = []
        for r in qres.rows:
            urls = _media_urls_for_row(req.dataset_id, r)
            enriched_rows.append({**r, **urls})

        return {
            "dataset_id": req.dataset_id,
            "query": qres.query,
            "content_hash": qres.content_hash,
            "estimated_rows": qres.cost_estimate.estimated_units,
            "estimated_bytes": qres.cost_estimate.estimated_bytes,
            "duration_ms": round(qres.duration_ms, 2),
            "row_count": qres.row_count,
            "columns": qres.columns,
            "rows": enriched_rows,
            "note": note.model_dump(),
        }

    @app.post("/api/investigate")
    def investigate_candidate(req: InvestigateRequest) -> dict[str, Any]:
        try:
            return _investigate_single_candidate(eff_cache_root, req)
        except SafetyViolationError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/api/duel")
    def run_real_vs_decoy_duel(req: DuelRequest) -> dict[str, Any]:
        ds = MultimodalSearchDataset.open(req.dataset_id, cache_root=eff_cache_root)
        pos_id, decoy_id = ds.pick_duel_pair()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_pos = pool.submit(
                _investigate_single_candidate,
                eff_cache_root,
                InvestigateRequest(
                    dataset_id=req.dataset_id,
                    record_id=pos_id,
                    use_live_gemini=req.use_live_gemini,
                    reveal_ground_truth=True,
                ),
            )
            f_dec = pool.submit(
                _investigate_single_candidate,
                eff_cache_root,
                InvestigateRequest(
                    dataset_id=req.dataset_id,
                    record_id=decoy_id,
                    use_live_gemini=req.use_live_gemini,
                    reveal_ground_truth=True,
                ),
            )
            pos_res = f_pos.result()
            dec_res = f_dec.result()

        pos_opt = ds.spec.positive_option
        pos_correct = pos_res["probe"]["primary_answer"] == pos_opt
        decoy_rejected = dec_res["probe"]["primary_answer"] != pos_opt

        return {
            "dataset_id": req.dataset_id,
            "positive_option": pos_opt,
            "real_candidate": pos_res,
            "decoy_candidate": dec_res,
            "summary": {
                "real_detected": pos_correct,
                "decoy_rejected": decoy_rejected,
                "both_passed_naive_tabular_filter": True,
            },
        }

    @app.get("/api/ground_truth/{dataset_id}/{record_id}")
    def reveal_ground_truth(dataset_id: str, record_id: str) -> dict[str, Any]:
        ds = MultimodalSearchDataset.open(dataset_id, cache_root=eff_cache_root)
        gt = ds.load_ground_truth().get(str(record_id))
        if gt is None:
            raise HTTPException(status_code=404, detail="Ground truth record not found.")
        return {"dataset_id": dataset_id, "record_id": record_id, "ground_truth": gt}

    @app.get("/api/media/{dataset_id}/{filename}")
    def serve_media(dataset_id: str, filename: str) -> FileResponse:
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid media filename.")
        ds = MultimodalSearchDataset.open(dataset_id, cache_root=eff_cache_root)
        target = verify_safe_output_path(ds.media_dir / filename, ds.dataset_dir)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Media file not found.")
        mime = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(str(target), media_type=mime)

    @app.get("/api/audio/{dataset_id}/{filename}")
    def serve_audio(dataset_id: str, filename: str) -> FileResponse:
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(status_code=400, detail="Invalid audio filename.")
        ds = MultimodalSearchDataset.open(dataset_id, cache_root=eff_cache_root)
        target = verify_safe_output_path(ds.audio_dir / filename, ds.dataset_dir)
        if not target.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found.")
        return FileResponse(str(target), media_type="audio/wav")

    return app


app = create_demo_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARES Multimodal Dataset Demo Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-dir", default=str(WORKTREE_ROOT / ".data_cache"))
    args = parser.parse_args()

    demo_app = create_demo_app(cache_root=args.cache_dir)
    print(f"\n🚀 ARES Multimodal Explorer running at: http://{args.host}:{args.port}\n", flush=True)
    uvicorn.run(demo_app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
