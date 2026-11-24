"""Zero-dependency local HTTP server for the ARES NASA TESS Observatory Example.

Provides REST endpoints for:
- Listing real NASA TESS benchmark cases (`/api/cases`)
- Planting a selected case into an isolated live workspace (`/api/plant`)
- Running deterministic physics verifiers + 2x2 PNG diagnostic cards (`/api/verify_preview`)
- Running full end-to-end ARES (`gemini-4-hydrogen` + `gemini-4-carbon` + `DeterministicBoardEvaluator`)
  and optional single-agent `gemini-4-carbon` SQL baseline (`/api/investigate`)
- Revealing official NASA Exoplanet Archive ground-truth metadata (`/api/reveal`)
- Serving generated SHA-256 PNG diagnostic cards (`/api/artifact/<filename>`)

Launch:
  uv run python examples/exoplanet_triage/observatory_server.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import queue
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ares.context import (
    create_initial_board,
    deterministic_patch_from_notes,
    validate_and_merge_patch,
)
from ares.contracts import GroundTruth, InMemoryVerifierRegistry, NeedleSpec
from ares.loop import LoopConfig, run_investigation_loop
from ares.models.gemini import GeminiModelProvider
from ares.store import NoteStore
from examples.exoplanet_triage.artifacts import render_exoplanet_diagnostic_card
from examples.exoplanet_triage.domain import (
    ExoplanetTriageDomain,
    _query_all_candidate_metrics,
)
from examples.incident_rca.benchmark import build_baseline_board_for_scoring

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ObservatoryDemoState:
    """Thread-safe state manager for live stage runs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.domain = ExoplanetTriageDomain()
        self.work_root = Path(
            tempfile.mkdtemp(prefix="ares_tess_stage_", dir=str(REPO_ROOT / "data"))
        )
        self.base_dir = self.work_root / "base_sector"
        self.active_dir = self.work_root / "active_sector"
        self.artifacts_dir = self.work_root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        print("[Observatory] Staging base NASA TESS sector Parquet shards...", flush=True)
        self.domain.generate(str(self.base_dir), size_gb=0.05, seed=42)
        self.base_stats = self._count_haystack_stats(self.base_dir)
        self.active_case: NeedleSpec | None = None
        self.active_truth: GroundTruth | None = None
        self.exoplanet_benchmark = self._load_json_file(
            REPO_ROOT / "examples" / "exoplanet_triage" / "results" / "exoplanet_tess_stage1.json"
        )
        self.incident_benchmark = self._load_json_file(
            REPO_ROOT / "examples" / "incident_rca" / "results" / "micro_bench_stage1.json"
        )
        self.cached_bench_results = self.exoplanet_benchmark
        self.exoplanet_baseline_by_case: dict[str, dict[str, Any]] = {}
        if self.exoplanet_benchmark:
            for c in self.exoplanet_benchmark.get("cases", []):
                cid = c.get("case_id")
                base = c.get("baseline")
                if cid and isinstance(base, dict):
                    self.exoplanet_baseline_by_case[str(cid)] = base

    @staticmethod
    def _load_json_file(path: Path) -> dict[str, Any] | None:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except Exception:
                return None
        return None

    @staticmethod
    def _count_haystack_stats(sector_dir: Path) -> dict[str, Any]:
        con = duckdb.connect(":memory:")
        lc_glob = str(sector_dir / "lightcurves" / "*.parquet").replace("'", "''")
        cat_glob = str(sector_dir / "stellar_catalog" / "*.parquet").replace("'", "''")
        lc_cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{lc_glob}')").fetchone()[0]
        star_cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{cat_glob}')").fetchone()[0]
        tce_cnt = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{cat_glob}') WHERE candidate_period_days IS NOT NULL"
        ).fetchone()[0]
        con.close()
        total_bytes = sum(f.stat().st_size for f in sector_dir.rglob("*.parquet"))
        return {
            "cadence_rows": int(lc_cnt),
            "total_stars": int(star_cnt),
            "tce_candidates": int(tce_cnt),
            "parquet_mb": round(total_bytes / (1024 * 1024), 2),
        }

    def plant_case(self, case_id: str) -> dict[str, Any]:
        needles = {n.case_id: n for n in self.domain.needle_menu()}
        if case_id not in needles:
            raise KeyError(f"Unknown case_id: {case_id}")
        needle = needles[case_id]

        with self.lock:
            if self.active_dir.exists():
                shutil.rmtree(self.active_dir)
            shutil.copytree(self.base_dir, self.active_dir)
            t0 = time.perf_counter()
            truth = self.domain.plant(str(self.active_dir), needle)
            plant_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            self.active_case = needle
            self.active_truth = truth
            stats = self._count_haystack_stats(self.active_dir)

        return {
            "case_id": needle.case_id,
            "category": needle.category,
            "scenario_type": needle.scenario_type,
            "subtlety": needle.subtlety,
            "plant_ms": plant_ms,
            "sector_stats": stats,
        }

    def _render_ui_cards(
        self,
        case_id: str,
        case_art_dir: Path,
        existing_artifacts: list[Any],
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Ensure Stage UI displays full + masked CCD cards for both confirmed planets and pruned impostors."""
        artifact_urls: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for a in existing_artifacts:
            if a.path not in seen_paths and Path(a.path).is_file():
                seen_paths.add(a.path)
                artifact_urls.append(
                    {
                        "artifact_id": a.artifact_id,
                        "sha256": a.sha256,
                        "url": f"/api/artifact/{case_id}/{Path(a.path).name}",
                        "filename": Path(a.path).name,
                    }
                )

        masked_cards: dict[str, str] = {}
        con = duckdb.connect(":memory:")
        try:
            cands = _query_all_candidate_metrics(con, str(self.active_dir), tic_filter=None)
            top_snr = [
                c
                for c in cands
                if c["n_transits"] >= 2 and c["transit_snr"] >= 7.0 and c["depth_ppm"] >= 150.0
            ][:3]
            for cand in top_snr:
                tic_id = str(cand["tic_id"])
                full_art = render_exoplanet_diagnostic_card(
                    con,
                    str(self.active_dir),
                    tic_id,
                    str(case_art_dir),
                    mask_ccd_panel=False,
                )
                if full_art.path not in seen_paths:
                    seen_paths.add(full_art.path)
                    artifact_urls.append(
                        {
                            "artifact_id": full_art.artifact_id,
                            "sha256": full_art.sha256,
                            "url": f"/api/artifact/{case_id}/{Path(full_art.path).name}",
                            "filename": Path(full_art.path).name,
                        }
                    )
                m_art = render_exoplanet_diagnostic_card(
                    con,
                    str(self.active_dir),
                    tic_id,
                    str(case_art_dir),
                    mask_ccd_panel=True,
                )
                masked_cards[tic_id] = f"/api/artifact/{case_id}/{Path(m_art.path).name}"
        finally:
            con.close()

        return artifact_urls, masked_cards

    def run_deterministic_preview(self, case_id: str) -> dict[str, Any]:
        """Instant deterministic verifier & 2x2 PNG diagnostic run (zero network dependency)."""
        plant_info = self.plant_case(case_id)
        with self.lock:
            assert self.active_truth is not None
            backend = self.domain.create_backend(str(self.active_dir))
            registry = InMemoryVerifierRegistry()
            self.domain.register_verifiers(registry)
            store = NoteStore()

            case_art_dir = self.artifacts_dir / case_id
            case_art_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            notes = []
            for idx, vname in enumerate(registry.list_verifiers()):
                _vres, note = backend.execute_verifier(
                    registry,
                    vname,
                    params={"artifact_dir": str(case_art_dir)},
                    requested_under=0,
                    hypothesis_family=vname,
                    content_hash=f"preview_{case_id}_{idx}_{vname}",
                    note_store=store,
                )
                notes.append(note)

            board = create_initial_board(
                self.domain.search_questions(),
                seed_fields=self.domain.pivotal_fields(),
            )
            patch = deterministic_patch_from_notes(
                notes, note_extractor=self.domain.extract_field_patches_from_note
            )
            board, report = validate_and_merge_patch(
                board,
                patch,
                store,
                protected_fields=self.domain.pivotal_fields(),
                seed_fields=self.domain.pivotal_fields(),
                note_extractor=self.domain.extract_field_patches_from_note,
            )

            existing_arts = [a for n in notes for a in n.artifacts]
            artifact_urls, masked_cards = self._render_ui_cards(
                case_id, case_art_dir, existing_arts
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            return {
                "mode": "deterministic_verifiers",
                "plant": plant_info,
                "elapsed_ms": elapsed_ms,
                "board": board.model_dump(),
                "board_drift": report.board_drift,
                "notes": [n.model_dump() for n in notes],
                "artifacts": artifact_urls,
                "masked_ccd_artifacts": masked_cards,
            }

    def run_live_ares(
        self,
        case_id: str,
        include_baseline: bool = False,
        progress_cb: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Execute Gemini 4 + DeterministicBoardEvaluator ARES loop with optional SSE progress."""
        def _emit(payload: dict[str, Any]) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(payload)
                except Exception:
                    pass

        _emit({"stage": "plant", "message": f"Planting real NASA TESS shard for {case_id}..."})
        plant_info = self.plant_case(case_id)
        _emit(
            {
                "stage": "perceive",
                "message": (
                    f"Swarm active: gemini-4-hydrogen writing SQL magnets over "
                    f"{plant_info['sector_stats']['cadence_rows']:,} cadences..."
                ),
            }
        )

        with self.lock:
            assert self.active_truth is not None
            truth = self.active_truth
            case_art_dir = self.artifacts_dir / case_id
            case_art_dir.mkdir(parents=True, exist_ok=True)

            provider = GeminiModelProvider()
            cfg = LoopConfig(
                max_rounds=2,
                max_concurrent_workers=4,
                wall_clock_budget_s=55.0,
                artifact_dir=str(case_art_dir),
            )

            seen_stages: set[str] = set()

            def _on_ares_event(ev: Any) -> None:
                tid = str(getattr(ev, "task_id", ""))
                etype = str(getattr(ev, "event_type", ""))
                if "_propose_" in tid and "perceive" not in seen_stages:
                    seen_stages.add("perceive")
                    _emit(
                        {
                            "stage": "perceive",
                            "task_id": tid,
                            "message": "gemini-4-hydrogen branching parallel SQL hypotheses...",
                        }
                    )
                elif ("_verify_" in tid or "_query_" in tid) and "verify" not in seen_stages:
                    seen_stages.add("verify")
                    _emit(
                        {
                            "stage": "verify",
                            "task_id": tid,
                            "message": "Running TRUSTED DuckDB physics & 11x11 CCD verifiers...",
                        }
                    )
                elif etype == "task_completed" and "_verify_" in tid and "fuse" not in seen_stages:
                    seen_stages.add("fuse")
                    _emit(
                        {
                            "stage": "fuse",
                            "task_id": tid,
                            "message": "gemini-4-carbon updating context & enforcing Invariants I1-I7...",
                        }
                    )

            def _on_round(round_tel: Any, _board: Any) -> None:
                _emit(
                    {
                        "stage": "probe",
                        "round_index": getattr(round_tel, "round_index", 1),
                        "conclusion_answer": getattr(round_tel, "conclusion_answer", "unknown"),
                        "confidence": getattr(round_tel, "conclusion_confidence", 0.0),
                        "message": (
                            f"Round {getattr(round_tel, 'round_index', 1)} complete · "
                            f"Deterministic 4-signal board evaluation & priority scored."
                        ),
                    }
                )

            ares_res = run_investigation_loop(
                self.domain,
                str(self.active_dir),
                provider=provider,
                config=cfg,
                on_round=_on_round,
                on_ares_event=_on_ares_event,
            )
            ares_score = self.domain.score(ares_res.board.model_dump(), truth)

            baseline_payload: dict[str, Any] | None = None
            if include_baseline:
                backend = self.domain.create_backend(str(self.active_dir))

                def _qexec(q: str) -> dict[str, Any]:
                    qres, _ = backend.execute_query(q)
                    return qres.model_dump() if hasattr(qres, "model_dump") else dict(qres)

                t_b = time.perf_counter()
                b_raw = provider.run_single_agent_baseline(
                    environment_description=self.domain.environment_description(),
                    magnet_rules=self.domain.magnet_rules(),
                    search_questions=self.domain.search_questions(),
                    query_executor=_qexec,
                    max_queries=10,
                    timeout_seconds=120.0,
                    thinking_level="low",
                )
                b_ms = round((time.perf_counter() - t_b) * 1000.0, 2)
                b_board = build_baseline_board_for_scoring(
                    b_raw, self.domain.decision_questions()
                )
                b_score = self.domain.score(b_board, truth)
                baseline_payload = {
                    "answers": b_raw["answers"],
                    "confidence": b_raw["confidence"],
                    "rationale": b_raw["rationale"],
                    "query_count": b_raw["query_count"],
                    "wall_clock_ms": b_ms,
                    "score": b_score,
                    "pre_recorded": False,
                }
            elif case_id in self.exoplanet_baseline_by_case:
                cached_b = dict(self.exoplanet_baseline_by_case[case_id])
                cached_b["pre_recorded"] = True
                baseline_payload = cached_b

            artifact_urls, masked_cards = self._render_ui_cards(
                case_id, case_art_dir, ares_res.artifacts
            )

            return {
                "mode": "live_ares",
                "plant": plant_info,
                "termination_reason": ares_res.termination_reason,
                "wall_clock_ms": ares_res.wall_clock_ms,
                "board": ares_res.board.model_dump(),
                "board_drift": ares_res.final_board_drift,
                "ungrounded_fields": ares_res.ungrounded_fields_on_board,
                "illegal_cert_upgrades": ares_res.illegal_cert_upgrades_on_board,
                "queries_executed": ares_res.queries_executed,
                "verifiers_executed": ares_res.verifiers_executed,
                "evaluator_calls": ares_res.evaluator_calls,
                "ares_score": ares_score,
                "rounds": [r.model_dump() for r in ares_res.rounds],
                "notes": [n.model_dump() for n in ares_res.notes],
                "artifacts": artifact_urls,
                "masked_ccd_artifacts": masked_cards,
                "baseline": baseline_payload,
            }

    def reveal_case_truth(self, case_id: str) -> dict[str, Any]:
        needles = {n.case_id: n for n in self.domain.needle_menu()}
        if case_id not in needles:
            raise KeyError(f"Unknown case_id: {case_id}")
        needle = needles[case_id]
        target_meta = self.domain.reveal_nasa_archive_truth(needle.target_entity)
        decoy_meta = (
            self.domain.reveal_nasa_archive_truth(needle.decoy_entity)
            if needle.decoy_entity
            else None
        )
        return {
            "case_id": needle.case_id,
            "category": needle.category,
            "description": needle.description,
            "target_entity": needle.target_entity,
            "decoy_entity": needle.decoy_entity,
            "expected_decisions": needle.expected_decisions,
            "nasa_target_archive": target_meta,
            "nasa_decoy_archive": decoy_meta,
        }


def make_handler(state: ObservatoryDemoState):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                idx_file = STATIC_DIR / "index.html"
                content = idx_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            if path == "/api/cases":
                cases_out = []
                for n in state.domain.needle_menu():
                    cases_out.append(
                        {
                            "case_id": n.case_id,
                            "category": n.category,
                            "scenario_type": n.scenario_type,
                            "subtlety": n.subtlety,
                            "blind_title": n.metadata.get("blind_title", n.case_id),
                            "sector": n.metadata.get("sector", 1),
                        }
                    )
                self._send_json(
                    {
                        "domain": state.domain.name,
                        "is_real_nasa_data": state.domain.is_using_real_nasa_data,
                        "base_sector_stats": state.base_stats,
                        "cases": cases_out,
                        "cached_benchmark": state.cached_bench_results,
                        "exoplanet_benchmark": state.exoplanet_benchmark,
                        "incident_benchmark": state.incident_benchmark,
                    }
                )
                return

            if path.startswith("/api/stream_ares/"):
                case_id = urllib.parse.unquote(path[len("/api/stream_ares/") :])
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()

                ev_q: queue.Queue[dict[str, Any] | None] = queue.Queue()

                def _worker() -> None:
                    try:
                        res = state.run_live_ares(
                            case_id,
                            include_baseline=False,
                            progress_cb=lambda p: ev_q.put(p),
                        )
                        ev_q.put({"stage": "complete", "result": res})
                    except Exception as exc:
                        ev_q.put({"stage": "error", "error": str(exc)})
                    finally:
                        ev_q.put(None)

                t = threading.Thread(target=_worker, daemon=True)
                t.start()

                try:
                    while True:
                        try:
                            item = ev_q.get(timeout=1.0)
                        except queue.Empty:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        if item is None:
                            break
                        payload = json.dumps(item, default=str).encode("utf-8")
                        self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            if path.startswith("/api/reveal/"):
                case_id = urllib.parse.unquote(path[len("/api/reveal/") :])
                try:
                    self._send_json(state.reveal_case_truth(case_id))
                except KeyError as exc:
                    self._send_json({"error": str(exc)}, status=404)
                return

            if path.startswith("/api/artifact/"):
                rel = urllib.parse.unquote(path[len("/api/artifact/") :])
                target = (state.artifacts_dir / rel).resolve()
                if not str(target).startswith(str(state.artifacts_dir.resolve())) or not target.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                mime, _ = mimetypes.guess_type(str(target))
                raw = target.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mime or "image/png")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception:
                body = {}

            try:
                if parsed.path == "/api/verify_preview":
                    cid = str(body["case_id"])
                    self._send_json(state.run_deterministic_preview(cid))
                    return
                if parsed.path == "/api/investigate":
                    cid = str(body["case_id"])
                    inc_base = bool(body.get("include_baseline", False))
                    self._send_json(state.run_live_ares(cid, include_baseline=inc_base))
                    return
                self._send_json({"error": f"Unknown route {parsed.path}"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch ARES NASA TESS Observatory Stage UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    state = ObservatoryDemoState()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        f"\n[Observatory Live Demo Ready] -> http://{args.host}:{args.port}\n"
        f"  Real NASA TESS Data: {state.domain.is_using_real_nasa_data}\n"
        f"  Base Haystack: {state.base_stats['cadence_rows']:,} cadences across "
        f"{state.base_stats['total_stars']} real stars ({state.base_stats['parquet_mb']} MB Parquet)\n",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
