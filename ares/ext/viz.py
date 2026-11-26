"""Self-contained interactive HTML slide-deck and trace viewer for ARES."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Sequence

from ares.store import ArtifactStore, CheckpointStore
from ares.types import Event, Task, Usage

_MAX_INLINE_ARTIFACT_CHARS = 65536


def _safe_json_for_script(obj: Any) -> str:
    """Serialize ``obj`` to JSON safe for embedding inside an HTML ``<script>`` tag."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _total_usage(tasks: dict[str, Task]) -> Usage:
    total = Usage()
    for task in tasks.values():
        total = total.add(task.usage)
    return total


def _derive_status(tasks: dict[str, Task], root_task_id: str, events: list[Event]) -> tuple[str, float]:
    wall_clock_ms = 0.0
    status: str | None = None
    for event in reversed(events):
        if event.event_type == "run_finished":
            if status is None and "status" in event.payload:
                status = str(event.payload["status"])
            if wall_clock_ms == 0.0 and "wall_clock_ms" in event.payload:
                wall_clock_ms = float(event.payload["wall_clock_ms"])
            break
    if status is not None:
        return status, wall_clock_ms
    root = tasks.get(root_task_id)
    if root is not None and root.status in ("done", "failed"):
        return root.status, wall_clock_ms
    if any(t.status == "suspended" for t in tasks.values()):
        return "suspended", wall_clock_ms
    if any(t.status == "failed" for t in tasks.values()):
        return "failed", wall_clock_ms
    return "running", wall_clock_ms


def export_run_bundle(
    run_dir: str | Path,
    *,
    title: str | None = None,
) -> dict[str, Any]:
    """Read a checkpointed ARES ``run_dir`` into a self-contained JSON-serializable bundle."""
    run_path = Path(run_dir)
    checkpoint = CheckpointStore(run_path)
    snapshot = checkpoint.load_state()
    if snapshot is None:
        raise FileNotFoundError(f"No checkpoint state.json found in {run_path}")

    tasks = {
        tid: Task.model_validate(tdata)
        for tid, tdata in snapshot.get("tasks", {}).items()
    }
    events = checkpoint.load_events()
    root_task_id = str(snapshot.get("root_task_id", ""))
    run_id = str(snapshot.get("run_id", run_path.name))
    status, wall_clock_ms = _derive_status(tasks, root_task_id, events)
    usage = _total_usage(tasks)

    store = ArtifactStore(run_path / "artifacts")
    artifacts: dict[str, dict[str, Any]] = {}
    for ref in store.list_refs():
        raw = store.get_bytes(ref.ref_id)
        truncated = False
        is_binary = False
        try:
            text = raw.decode("utf-8")
            if len(text) > _MAX_INLINE_ARTIFACT_CHARS:
                text = text[:_MAX_INLINE_ARTIFACT_CHARS]
                truncated = True
        except UnicodeDecodeError:
            is_binary = True
            text = f"<binary payload: {len(raw)} bytes, sha256={ref.sha256}>"

        artifacts[ref.ref_id] = {
            "ref_id": ref.ref_id,
            "sha256": ref.sha256,
            "byte_size": ref.byte_size,
            "mime_type": ref.mime_type,
            "preview": ref.preview,
            "content": text,
            "truncated": truncated,
            "binary": is_binary,
        }

    return {
        "schema": "ares_viz_bundle_v1",
        "title": title or f"Real ARES Run ({run_id})",
        "run_id": run_id,
        "root_task_id": root_task_id,
        "status": status,
        "wall_clock_ms": round(wall_clock_ms, 2),
        "run_dir": str(run_path),
        "budget": snapshot.get("budget", {}),
        "usage": usage.model_dump(),
        "tasks": {tid: task.model_dump() for tid, task in tasks.items()},
        "events": [ev.model_dump() for ev in events],
        "artifacts": artifacts,
    }


def _story_physics_simulation() -> dict[str, Any]:
    """Story 1: Simple 5-step physics ODE discovery & verification story."""
    solver_diff = (
        "--- a/solver.py\n"
        "+++ b/solver.py\n"
        "@@ -4,9 +4,9 @@\n"
        " def simulate_oscillator(t_grid, y0, params):\n"
        "-    # Attempt 1: Explicit Euler (unstable at t=4.2s)\n"
        "-    for i in range(len(t_grid) - 1):\n"
        "-        dv = -alpha * v - beta * x - gamma * (x ** 3)\n"
        "-        x, v = x + dt * v, v + dt * dv\n"
        "+    # Attempt 2: Adaptive RK45 (energy stable)\n"
        "+    sol = solve_ivp(\n"
        "+        rhs, (t_grid[0], t_grid[-1]), y0,\n"
        "+        t_eval=t_grid, method='RK45', rtol=1e-7\n"
        "+    )\n"
        "+    return sol.y.T\n"
    )
    cert_receipt = {
        "certified": True,
        "model": "Duffing nonlinear damped oscillator",
        "checks": {
            "step_convergence": "PASS (delta 3.1e-6)",
            "energy_bound_dE_dt": "PASS (-0.004 <= 0)",
            "holdout_rmse": "PASS (0.0114 <= 0.025)",
        },
        "verifier_input": "ONLY (code_ref_id, dataset_ref_id) — zero LLM prose",
    }
    unverified_receipt = {
        "certified": False,
        "status": "unverified",
        "reason": "Damping-torque sensor channel missing; all 3 models failed energy conservation.",
    }

    artifacts = {
        "art_phys_data": {
            "ref_id": "art_phys_data",
            "sha256": "9f1a8c44d2e0b712a8c01499f08b1234",
            "byte_size": 2481920,
            "mime_type": "application/json",
            "preview": "12 sensor trajectories (2.48 MB stored on disk)",
            "content": json.dumps(
                {
                    "dataset": "duffing_sensor_bench",
                    "trajectories": 12,
                    "raw_bytes": 2481920,
                    "summary_passed_to_llm": "310 tokens (mean/std/periodicity only)",
                },
                indent=2,
            ),
        },
        "art_solver_diff": {
            "ref_id": "art_solver_diff",
            "sha256": "5a11bb22cc33dd44ee55ff6677889900",
            "byte_size": len(solver_diff.encode("utf-8")),
            "mime_type": "text/x-diff",
            "preview": "Euler -> RK45 repair diff",
            "content": solver_diff,
        },
        "art_cert_receipt": {
            "ref_id": "art_cert_receipt",
            "sha256": "f00d1234567890abcdef1234567890ab",
            "byte_size": len(json.dumps(cert_receipt).encode("utf-8")),
            "mime_type": "application/json",
            "preview": "Certified Duffing RK45 receipt",
            "content": json.dumps(cert_receipt, indent=2),
        },
        "art_unverified_receipt": {
            "ref_id": "art_unverified_receipt",
            "sha256": "0000deadbeef1234567890abcdef1234",
            "byte_size": len(json.dumps(unverified_receipt).encode("utf-8")),
            "mime_type": "application/json",
            "preview": "Honest unverified receipt",
            "content": json.dumps(unverified_receipt, indent=2),
        },
    }

    tasks: dict[str, dict[str, Any]] = {
        "data": {
            "id": "data",
            "state": "ingest_sensor_data",
            "parent_id": None,
            "depends_on": [],
            "depth": 0,
            "status": "done",
            "deterministic": True,
            "spilled_artifact_id": "art_phys_data",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "input": {"dataset": "duffing_sensor_bench", "size_mb": 2.48},
            "output": {"__ares_ref__": "art_phys_data", "summary_tokens": 310},
            "viz": {
                "roleBadge": "orchestrator",
                "label": "Sensor Data (2.5 MB)",
                "subtitle": "Stored on disk · tiny summary only",
                "summary": "Raw sensor arrays stay on disk in ArtifactStore. Downstream local models only receive a 310-token summary.",
                "snippetTitle": "What enters the model vs. what stays on disk",
                "snippet": "On disk (art_phys_data): 2,481,920 bytes of time-series arrays.\nPassed to model: {\"summary\": \"damped oscillation, frequency increases with amplitude\", \"ref\": \"art_phys_data\"}",
            },
        },
        "branch_a": {
            "id": "branch_a",
            "state": "propose_model",
            "parent_id": "data",
            "depends_on": [],
            "depth": 1,
            "status": "done",
            "deterministic": False,
            "usage": {"prompt_tokens": 520, "completion_tokens": 180, "total_tokens": 700, "llm_calls": 1},
            "input": {"branch": "A", "hypothesis": "Linear damped ODE"},
            "output": {"holdout_rmse": 0.418, "passed": False},
            "viz": {
                "roleBadge": "llm",
                "label": "A · Linear Model",
                "subtitle": "Testing hypothesis...",
                "summary": "Plausible linear decoy: fits low-amplitude data well, but fails high-amplitude holdout trajectories.",
                "snippetTitle": "Bounded prompt (520 tokens)",
                "snippet": "Summary: damped oscillation, decay ratio 0.82.\nPropose an ODE model JSON.\n\nModel output: x'' + 0.18 x' + 1.2 x = 0",
            },
        },
        "branch_b": {
            "id": "branch_b",
            "state": "propose_model",
            "parent_id": "data",
            "depends_on": [],
            "depth": 1,
            "status": "done",
            "deterministic": False,
            "usage": {"prompt_tokens": 540, "completion_tokens": 210, "total_tokens": 750, "llm_calls": 1},
            "input": {"branch": "B", "hypothesis": "Duffing nonlinear ODE"},
            "output": {"passed": False, "error": "Explicit Euler diverged at t=4.20s"},
            "viz": {
                "roleBadge": "llm",
                "label": "B · Duffing Nonlinear",
                "subtitle": "Testing hypothesis...",
                "summary": "Picks the right nonlinear x^3 physics, but Attempt 1 uses explicit Euler and blows up numerically.",
                "snippetTitle": "Bounded prompt (540 tokens) & <think>",
                "snippet": "Thinking: Frequency rises with amplitude -> test cubic x^3 term.\nAttempt 1 code: fixed-step explicit Euler loop.",
            },
        },
        "branch_c": {
            "id": "branch_c",
            "state": "propose_model",
            "parent_id": "data",
            "depends_on": [],
            "depth": 1,
            "status": "done",
            "deterministic": False,
            "usage": {"prompt_tokens": 530, "completion_tokens": 195, "total_tokens": 725, "llm_calls": 1},
            "input": {"branch": "C", "hypothesis": "Polynomial surrogate"},
            "output": {"passed": False, "energy_ok": False},
            "viz": {
                "roleBadge": "llm",
                "label": "C · Polynomial Fit",
                "subtitle": "Testing hypothesis...",
                "summary": "Degree-7 polynomial overfits training curves and violates physical energy conservation (dE/dt > 0).",
                "snippetTitle": "Bounded prompt (530 tokens)",
                "snippet": "Model output: degree-7 polynomial surrogate vector field.",
            },
        },
        "verify": {
            "id": "verify",
            "state": "verify_sim",
            "parent_id": None,
            "depends_on": ["branch_a", "branch_b", "branch_c"],
            "depth": 2,
            "status": "done",
            "deterministic": True,
            "spilled_artifact_id": "art_cert_receipt",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "input": {"inputs_allowed": ["code_ref_id", "dataset_ref_id"], "llm_prose_hidden": True},
            "output": {"a": "FAIL (RMSE 0.418)", "b": "FAIL (Euler NaN)", "c": "FAIL (dE/dt > 0)"},
            "viz": {
                "roleBadge": "verifier",
                "label": "Blind Physics Check",
                "subtitle": "Sees code + data only",
                "summary": "Isolated Python/SciPy sandbox. It never sees the LLM's explanations — only code and sensor data.",
                "snippetTitle": "3 deterministic checks",
                "snippet": "1. Step-refinement convergence (< 1e-4)\n2. Energy dissipation bound (dE/dt <= 0)\n3. High-amplitude holdout RMSE (<= 0.025)",
            },
        },
        "repair_b": {
            "id": "repair_b",
            "state": "refine_model",
            "parent_id": "branch_b",
            "depends_on": ["verify"],
            "depth": 1,
            "status": "done",
            "deterministic": False,
            "usage": {"prompt_tokens": 610, "completion_tokens": 220, "total_tokens": 830, "llm_calls": 1},
            "input": {"branch": "B", "failure_slice": "Explicit Euler diverged at t=4.20s (NaN)"},
            "output": {"certified": True, "holdout_rmse": 0.0114},
            "viz": {
                "roleBadge": "verifier",
                "label": "B · RK45 Refine",
                "subtitle": "✓ Passes all 3 checks",
                "diffArtifactRef": "art_solver_diff",
                "summary": "Branch B sees only a 340-char error snippet, refines Euler to RK45, and passes all 3 physics checks.",
                "snippetTitle": "Euler -> RK45 refinement (610-token prompt)",
                "snippet": solver_diff,
            },
        },
        "gate": {
            "id": "gate",
            "state": "scientist_review",
            "parent_id": None,
            "depends_on": ["repair_b"],
            "depth": 0,
            "status": "done",
            "deterministic": True,
            "spilled_artifact_id": "art_cert_receipt",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "input": {"resume_key": "hpc_signoff"},
            "output": {"approved": True, "model": "Duffing RK45"},
            "viz": {
                "roleBadge": "gate",
                "label": "Scientist Review",
                "subtitle": "Paused cleanly -> Approved",
                "summary": "ARES pauses before launching expensive 3D runs. A scientist inspects the diff/receipt and resumes without re-running anything.",
                "snippetTitle": "Verified receipt (art_cert_receipt)",
                "snippet": json.dumps(cert_receipt, indent=2),
            },
        },
    }

    certified_events = [
        {"seq": 1, "beat": 0, "event_type": "task_done", "task_id": "data", "payload": {"spilled": "art_phys_data"}},
        {"seq": 2, "beat": 1, "event_type": "task_spawned", "task_id": "data", "payload": {"branches": ["A", "B", "C"], "join": "any"}},
        {"seq": 3, "beat": 2, "event_type": "task_done", "task_id": "verify", "payload": {"a": "RMSE fail", "b": "Euler NaN", "c": "Energy fail"}},
        {"seq": 4, "beat": 3, "event_type": "task_done", "task_id": "repair_b", "payload": {"certified": True, "rmse": 0.0114}},
        {"seq": 5, "beat": 3, "event_type": "task_cancelled", "task_id": "branch_a", "payload": {"reason": "race_won_by:branch_b"}},
        {"seq": 6, "beat": 3, "event_type": "task_cancelled", "task_id": "branch_c", "payload": {"reason": "race_won_by:branch_b"}},
        {"seq": 7, "beat": 4, "event_type": "task_suspended", "task_id": "gate", "payload": {"resume_key": "hpc_signoff"}},
        {"seq": 8, "beat": 4, "event_type": "run_finished", "task_id": "gate", "payload": {"status": "done"}},
    ]

    certified_beats = [
        {
            "headline": "Keep big data out of the model",
            "narration": "Raw trajectories stay on disk; the local model only sees a tiny summary.",
            "visibleTaskIds": ["data"],
            "activeTaskIds": ["data"],
            "statusOverrides": {"data": "done"},
            "subtitleOverrides": {},
            "contextMetrics": {"aresPromptTokens": 310, "aresCapTokens": 4096, "baselinePromptTokens": 1850, "baselineCapTokens": 4096, "baselineStatus": "OK"},
        },
        {
            "headline": "Propose 3 physics models in parallel",
            "narration": "Each branch runs in its own small, isolated context window.",
            "visibleTaskIds": ["data", "branch_a", "branch_b", "branch_c"],
            "activeTaskIds": ["branch_a", "branch_b", "branch_c"],
            "statusOverrides": {"branch_a": "running", "branch_b": "running", "branch_c": "running"},
            "subtitleOverrides": {
                "branch_a": "Linear damped ODE...",
                "branch_b": "Duffing cubic ODE...",
                "branch_c": "Polynomial surrogate...",
            },
            "contextMetrics": {"aresPromptTokens": 540, "aresCapTokens": 4096, "baselinePromptTokens": 3620, "baselineCapTokens": 4096, "baselineStatus": "Near limit"},
        },
        {
            "headline": "Verify with a blind physics check",
            "narration": "The verifier never sees LLM prose — all 3 initial proposals fail.",
            "visibleTaskIds": ["data", "branch_a", "branch_b", "branch_c", "verify"],
            "activeTaskIds": ["verify"],
            "statusOverrides": {"branch_a": "failed", "branch_b": "failed", "branch_c": "failed", "verify": "done"},
            "subtitleOverrides": {
                "branch_a": "✗ High holdout error (0.42)",
                "branch_b": "✗ Euler blew up at t=4.2s",
                "branch_c": "✗ Violated energy bound",
            },
            "contextMetrics": {"aresPromptTokens": 540, "aresCapTokens": 4096, "baselinePromptTokens": 5180, "baselineCapTokens": 4096, "baselineStatus": "OVERFLOW"},
        },
        {
            "headline": "Refine one branch, cancel the rest",
            "narration": "Branch B refines Euler -> RK45 from a tiny error snippet; A & C cancel automatically.",
            "visibleTaskIds": ["data", "branch_a", "branch_b", "branch_c", "verify", "repair_b"],
            "activeTaskIds": ["repair_b"],
            "statusOverrides": {"branch_a": "cancelled", "branch_c": "cancelled", "branch_b": "done", "repair_b": "done"},
            "subtitleOverrides": {
                "branch_a": "Cancelled (race lost)",
                "branch_b": "Sent Euler error to refine",
                "branch_c": "Cancelled (race lost)",
                "repair_b": "✓ RK45 passes all 3 checks",
            },
            "contextMetrics": {"aresPromptTokens": 610, "aresCapTokens": 4096, "baselinePromptTokens": 7490, "baselineCapTokens": 4096, "baselineStatus": "OVERFLOW"},
        },
        {
            "headline": "Pause for scientist sign-off",
            "narration": "ARES checkpoints cleanly before expensive runs — click any box to inspect details.",
            "visibleTaskIds": ["data", "branch_a", "branch_b", "branch_c", "verify", "repair_b", "gate"],
            "activeTaskIds": ["gate"],
            "statusOverrides": {"branch_a": "cancelled", "branch_c": "cancelled", "gate": "done"},
            "subtitleOverrides": {
                "branch_a": "Cancelled (race lost)",
                "branch_c": "Cancelled (race lost)",
                "gate": "✓ Approved · Certified RK45",
            },
            "contextMetrics": {"aresPromptTokens": 610, "aresCapTokens": 4096, "baselinePromptTokens": 7490, "baselineCapTokens": 4096, "baselineStatus": "OVERFLOW"},
        },
    ]

    # Variant 2: Missing sensor -> honest abstention
    unverified_tasks = json.loads(json.dumps(tasks))
    del unverified_tasks["gate"]
    unverified_tasks["repair_b"]["status"] = "failed"
    unverified_tasks["repair_b"]["spilled_artifact_id"] = "art_unverified_receipt"
    unverified_tasks["repair_b"]["viz"]["roleBadge"] = "gate"
    unverified_tasks["repair_b"]["viz"]["label"] = "Result: Unverified"
    unverified_tasks["repair_b"]["viz"]["subtitle"] = "Missing sensor · no hallucination"
    unverified_tasks["repair_b"]["viz"]["summary"] = "Because the required damping-torque channel is missing, ARES refuses to guess and returns unverified."
    unverified_tasks["repair_b"]["viz"]["snippetTitle"] = "Honest abstention receipt"
    unverified_tasks["repair_b"]["viz"]["snippet"] = json.dumps(unverified_receipt, indent=2)

    unverified_beats = json.loads(json.dumps(certified_beats[:4]))
    unverified_beats[3]["headline"] = "When data is missing, ARES refuses to bluff"
    unverified_beats[3]["narration"] = "All branches fail the physical energy check, so ARES outputs unverified instead of picking a bad fit."
    unverified_beats[3]["statusOverrides"] = {"branch_a": "failed", "branch_b": "failed", "branch_c": "failed", "repair_b": "suspended"}
    unverified_beats[3]["subtitleOverrides"] = {
        "branch_a": "✗ Failed energy check",
        "branch_b": "✗ Failed energy check",
        "branch_c": "✗ Failed energy check",
        "repair_b": "Unverified (missing channel)",
    }

    return {
        "id": "physics_sim",
        "tabLabel": "Story 1: Physics Simulation",
        "variants": [
            {
                "id": "certified",
                "label": "Normal Run",
                "beats": certified_beats,
                "tasks": tasks,
                "events": certified_events,
                "artifacts": artifacts,
            },
            {
                "id": "unverified",
                "label": "Missing Sensor",
                "beats": unverified_beats,
                "tasks": unverified_tasks,
                "events": certified_events[:4],
                "artifacts": artifacts,
            },
        ],
    }


def _story_literature_data() -> dict[str, Any]:
    """Story 2: Simple 4-step literature + cohort hypothesis triangulation story."""
    paper_slice_m14 = (
        "[PMC8419204 Methods — bounded 400-char slice]\n"
        "Cohort B (n=412, randomized across 4 centers) measured Factor M-14.\n"
        "Adjusted for age & extraction batch (FDR=0.008): beta=+0.41.\n"
        "Replicated in Cohort R (p=0.0041, protocol PROT-44B)."
    )
    sql_check = (
        "-- Blind DuckDB + OLS confounder check\n"
        "SELECT candidate_id, raw_corr, adj_beta_after_batch_age, adj_p_value\n"
        "FROM cohort_measurements JOIN replication_assays USING (candidate_id);"
    )
    winner_receipt = {
        "certified": "Factor M-14",
        "signals": {"literature": True, "raw_cohort": True, "confounder_adjusted_replication": True},
        "adj_p_value": 0.0041,
        "rejected_decoy": "Factor X-99 (adj_p=0.61, confounded by extraction batch)",
    }
    underdet_receipt = {
        "status": "unverified",
        "candidate": "Factor M-14",
        "reason": "Replication cohort table is missing (NULL). Paused for collaborator data.",
    }

    artifacts = {
        "art_slice_m14": {
            "ref_id": "art_slice_m14",
            "sha256": "bb22cc33dd44ee55ff66778899001122",
            "byte_size": len(paper_slice_m14.encode("utf-8")),
            "mime_type": "text/plain",
            "preview": "PMC8419204 excerpt for Factor M-14",
            "content": paper_slice_m14,
        },
        "art_sql_query": {
            "ref_id": "art_sql_query",
            "sha256": "cc33dd44ee55ff667788990011223344",
            "byte_size": len(sql_check.encode("utf-8")),
            "mime_type": "text/plain",
            "preview": "Confounder-adjusted verification query",
            "content": sql_check,
        },
        "art_m14_receipt": {
            "ref_id": "art_m14_receipt",
            "sha256": "ee55ff66778899001122334455667788",
            "byte_size": len(json.dumps(winner_receipt).encode("utf-8")),
            "mime_type": "application/json",
            "preview": "Certified Factor M-14 receipt",
            "content": json.dumps(winner_receipt, indent=2),
        },
        "art_underdet_receipt": {
            "ref_id": "art_underdet_receipt",
            "sha256": "ff667788990011223344556677889900",
            "byte_size": len(json.dumps(underdet_receipt).encode("utf-8")),
            "mime_type": "application/json",
            "preview": "Paused unverified receipt",
            "content": json.dumps(underdet_receipt, indent=2),
        },
    }

    tasks: dict[str, dict[str, Any]] = {
        "corpus": {
            "id": "corpus",
            "state": "coarse_scan",
            "parent_id": None,
            "depends_on": [],
            "depth": 0,
            "status": "done",
            "deterministic": True,
            "spilled_artifact_id": "art_slice_m14",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "input": {"papers": 40, "cohort_tables": 6, "candidates": 32},
            "output": {"top_k": 3},
            "viz": {
                "roleBadge": "orchestrator",
                "label": "40 Papers + 6 Tables",
                "subtitle": "Read in small bounded slices",
                "summary": "40 papers and 6 experimental tables stay out-of-line. Each worker scans one small bucket.",
                "snippetTitle": "Bounded slice example",
                "snippet": paper_slice_m14,
            },
        },
        "cand_x": {
            "id": "cand_x",
            "state": "screen_candidate",
            "parent_id": "corpus",
            "depends_on": [],
            "depth": 1,
            "status": "done",
            "deterministic": False,
            "usage": {"prompt_tokens": 410, "completion_tokens": 95, "total_tokens": 505, "llm_calls": 1},
            "input": {"candidate": "Factor X-99", "paper_hits": 18},
            "output": {"coarse_rank": 1, "raw_fc": 4.8},
            "viz": {
                "roleBadge": "llm",
                "label": "Factor X-99",
                "subtitle": "#1 coarse buzz (18 papers)",
                "summary": "Loud decoy: lots of paper mentions and huge raw fold-change, but cases & controls were in separate batches.",
                "snippetTitle": "Why a naive agent picks X-99",
                "snippet": "18 citations, raw fold-change 4.8x — but extraction batch was confounded.",
            },
        },
        "cand_m": {
            "id": "cand_m",
            "state": "screen_candidate",
            "parent_id": "corpus",
            "depends_on": [],
            "depth": 1,
            "status": "done",
            "deterministic": False,
            "usage": {"prompt_tokens": 470, "completion_tokens": 105, "total_tokens": 575, "llm_calls": 1},
            "input": {"candidate": "Factor M-14", "paper_hits": 5},
            "output": {"coarse_rank": 2, "protocol": "PROT-44B"},
            "viz": {
                "roleBadge": "llm",
                "label": "Factor M-14",
                "subtitle": "#2 moderate buzz (5 papers)",
                "summary": "Subtle true mechanism: lower citation buzz, but consistent across multi-center protocols.",
                "snippetTitle": "Bounded prompt (470 tokens)",
                "snippet": paper_slice_m14,
            },
        },
        "cand_k": {
            "id": "cand_k",
            "state": "screen_candidate",
            "parent_id": "corpus",
            "depends_on": [],
            "depth": 1,
            "status": "done",
            "deterministic": False,
            "usage": {"prompt_tokens": 395, "completion_tokens": 88, "total_tokens": 483, "llm_calls": 1},
            "input": {"candidate": "Factor K-07", "paper_hits": 4},
            "output": {"coarse_rank": 3},
            "viz": {
                "roleBadge": "llm",
                "label": "Factor K-07",
                "subtitle": "#3 weak signal",
                "summary": "Third coarse survivor; eliminated cleanly by the statistical verifier.",
                "snippetTitle": "Candidate summary",
                "snippet": "4 citations, weak raw correlation.",
            },
        },
        "stats": {
            "id": "stats",
            "state": "verify_cohort_stats",
            "parent_id": None,
            "depends_on": ["cand_x", "cand_m", "cand_k"],
            "depth": 1,
            "status": "done",
            "deterministic": True,
            "spilled_artifact_id": "art_sql_query",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "input": {"adjust_for": ["extraction_batch", "age"]},
            "output": {"X-99": "p=0.61 (FAIL)", "M-14": "p=0.0041 (PASS)", "K-07": "p=0.38 (FAIL)"},
            "viz": {
                "roleBadge": "verifier",
                "label": "Blind Statistical Check",
                "subtitle": "Controls for batch & age",
                "summary": "Deterministic DuckDB + OLS check runs independently of LLM prose and exposes X-99 as a batch artifact.",
                "snippetTitle": "Deterministic SQL verification",
                "snippet": sql_check,
            },
        },
        "result": {
            "id": "result",
            "state": "finalize",
            "parent_id": None,
            "depends_on": ["stats"],
            "depth": 0,
            "status": "done",
            "deterministic": True,
            "spilled_artifact_id": "art_m14_receipt",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0},
            "input": {"require_signals": 3},
            "output": winner_receipt,
            "viz": {
                "roleBadge": "verifier",
                "label": "Certified: Factor M-14",
                "subtitle": "Linked to paper slice + SQL",
                "summary": "Only Factor M-14 passes literature, raw cohort, and confounder-adjusted replication.",
                "snippetTitle": "Audit receipt (art_m14_receipt)",
                "snippet": json.dumps(winner_receipt, indent=2),
            },
        },
    }

    events = [
        {"seq": 1, "beat": 0, "event_type": "task_done", "task_id": "corpus", "payload": {"candidates": 32}},
        {"seq": 2, "beat": 1, "event_type": "task_spawned", "task_id": "corpus", "payload": {"top_k": ["X-99", "M-14", "K-07"]}},
        {"seq": 3, "beat": 2, "event_type": "task_done", "task_id": "stats", "payload": {"rejected_decoy": "Factor X-99", "certified": "Factor M-14"}},
        {"seq": 4, "beat": 3, "event_type": "run_finished", "task_id": "result", "payload": {"status": "done", "target": "Factor M-14"}},
    ]

    beats = [
        {
            "headline": "Read large corpora in small slices",
            "narration": "Papers and tables stay on disk so small local models never overflow.",
            "visibleTaskIds": ["corpus"],
            "activeTaskIds": ["corpus"],
            "statusOverrides": {"corpus": "done"},
            "subtitleOverrides": {},
            "contextMetrics": {"aresPromptTokens": 380, "aresCapTokens": 4096, "baselinePromptTokens": 2100, "baselineCapTokens": 4096, "baselineStatus": "OK"},
        },
        {
            "headline": "Screen 32 candidates down to 3",
            "narration": "A loud candidate (X-99) leads coarse buzz ahead of subtle M-14.",
            "visibleTaskIds": ["corpus", "cand_x", "cand_m", "cand_k"],
            "activeTaskIds": ["cand_x", "cand_m", "cand_k"],
            "statusOverrides": {"cand_x": "done", "cand_m": "done", "cand_k": "done"},
            "subtitleOverrides": {},
            "contextMetrics": {"aresPromptTokens": 470, "aresCapTokens": 4096, "baselinePromptTokens": 4450, "baselineCapTokens": 4096, "baselineStatus": "OVERFLOW"},
        },
        {
            "headline": "Test blindly against confounders",
            "narration": "Independent stats expose X-99 as a batch decoy and confirm M-14.",
            "visibleTaskIds": ["corpus", "cand_x", "cand_m", "cand_k", "stats"],
            "activeTaskIds": ["stats"],
            "statusOverrides": {"cand_x": "failed", "cand_k": "failed", "cand_m": "done", "stats": "done"},
            "subtitleOverrides": {
                "cand_x": "✗ Batch decoy (p=0.61)",
                "cand_m": "✓ Replicated (p=0.004)",
                "cand_k": "✗ No effect (p=0.38)",
            },
            "contextMetrics": {"aresPromptTokens": 485, "aresCapTokens": 4096, "baselinePromptTokens": 8910, "baselineCapTokens": 4096, "baselineStatus": "OVERFLOW"},
        },
        {
            "headline": "Certify with full audit trail",
            "narration": "Click any box to inspect the exact paper slice, SQL query, and receipt.",
            "visibleTaskIds": ["corpus", "cand_x", "cand_m", "cand_k", "stats", "result"],
            "activeTaskIds": ["result"],
            "statusOverrides": {"cand_x": "failed", "cand_k": "failed", "result": "done"},
            "subtitleOverrides": {
                "cand_x": "✗ Batch decoy (p=0.61)",
                "cand_m": "✓ Replicated (p=0.004)",
                "cand_k": "✗ No effect (p=0.38)",
            },
            "contextMetrics": {"aresPromptTokens": 485, "aresCapTokens": 4096, "baselinePromptTokens": 8910, "baselineCapTokens": 4096, "baselineStatus": "OVERFLOW"},
        },
    ]

    underdet_tasks = json.loads(json.dumps(tasks))
    underdet_tasks["result"]["status"] = "suspended"
    underdet_tasks["result"]["spilled_artifact_id"] = "art_underdet_receipt"
    underdet_tasks["result"]["viz"]["roleBadge"] = "gate"
    underdet_tasks["result"]["viz"]["label"] = "Paused: Unverified"
    underdet_tasks["result"]["viz"]["subtitle"] = "Awaiting replication cohort"
    underdet_tasks["result"]["viz"]["summary"] = "Replication cohort data is missing. ARES pauses cleanly instead of extrapolating from 2-of-3 signals."
    underdet_tasks["result"]["viz"]["snippetTitle"] = "Honest abstention receipt"
    underdet_tasks["result"]["viz"]["snippet"] = json.dumps(underdet_receipt, indent=2)

    underdet_beats = json.loads(json.dumps(beats))
    underdet_beats[2]["headline"] = "Replication data is missing"
    underdet_beats[2]["narration"] = "X-99 is rejected as a decoy, while M-14 has only 2 of 3 required signals."
    underdet_beats[2]["subtitleOverrides"] = {
        "cand_x": "✗ Batch decoy (p=0.61)",
        "cand_m": "? Missing Cohort R",
        "cand_k": "✗ No effect (p=0.38)",
    }
    underdet_beats[3]["headline"] = "Pause honestly instead of guessing"
    underdet_beats[3]["narration"] = "ARES suspends with unverified status so collaborators can supply the table later."
    underdet_beats[3]["statusOverrides"] = {"cand_x": "failed", "cand_k": "failed", "result": "suspended"}
    underdet_beats[3]["subtitleOverrides"] = underdet_beats[2]["subtitleOverrides"]

    return {
        "id": "literature_data",
        "tabLabel": "Story 2: Literature + Data",
        "variants": [
            {
                "id": "certified",
                "label": "Normal Run",
                "beats": beats,
                "tasks": tasks,
                "events": events,
                "artifacts": artifacts,
            },
            {
                "id": "under_determined",
                "label": "Missing Cohort",
                "beats": underdet_beats,
                "tasks": underdet_tasks,
                "events": events[:3],
                "artifacts": artifacts,
            },
        ],
    }


def _story_overview_vision() -> dict[str, Any]:
    """Opening conceptual page: goals, suitability, methodology, and ChARGe -> ARES."""
    beats = [
        {
            "layout": "goals",
            "headline": "Make local models viable for scientific reasoning — with full visibility",
            "narration": "Chat loops break on long scientific tasks; ARES turns reasoning into a transparent state machine.",
        },
        {
            "layout": "methodology",
            "headline": "Move control flow out of prompts into a lightweight state machine",
            "narration": "Local models handle focused reasoning steps while ARES manages context, branching, and checks.",
        },
        {
            "layout": "charge",
            "headline": "From ChARGe (SC26) to a general local-model runtime",
            "narration": "ChARGe proved Propose → Verify → Refine + Context Switching work in chemistry; ARES generalizes them.",
        },
        {
            "layout": "walkthroughs",
            "headline": "Walk through Propose → Verify → Refine in action",
            "narration": "Choose a scientific scenario below, or inspect a real checkpoint run from this repo.",
        },
    ]
    return {
        "id": "overview",
        "kind": "overview",
        "tabLabel": "Overview & Vision",
        "variants": [
            {
                "id": "vision",
                "label": "Vision",
                "beats": beats,
                "tasks": {},
                "events": [],
                "artifacts": {},
            }
        ],
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f8f9fb;
    --surface: #ffffff;
    --surface-sub: #f3f5f8;
    --border: #dfe4ec;
    --text: #111827;
    --muted: #5b6472;
    --accent: #1d4ed8;
    --accent-soft: #eff6ff;
    --green: #15803d;
    --green-soft: #f0fdf4;
    --amber: #b45309;
    --amber-soft: #fffbeb;
    --red: #b91c1c;
    --red-soft: #fef2f2;
    --purple: #6d28d9;
    --purple-soft: #f5f3ff;
    --shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    --drawer-shadow: -8px 0 28px rgba(15, 23, 42, 0.12);
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-sans);
    line-height: 1.45;
  }
  .page {
    max-width: 1220px;
    min-height: 100vh;
    margin: 0 auto;
    padding: 18px 22px 20px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }
  .topbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .logo-pill {
    background: #1f2937;
    color: #ffffff;
    font-weight: 700;
    font-size: 12.5px;
    letter-spacing: 0.06em;
    padding: 5px 10px;
    border-radius: 6px;
  }
  .tabs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .tab-btn {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--muted);
    padding: 6px 12px;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
  }
  .tab-btn:hover { color: var(--text); border-color: #b7c2d0; }
  .tab-btn.active {
    background: var(--accent-soft);
    color: var(--accent);
    border-color: var(--accent);
  }
  .topbar-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .pill-group {
    display: inline-flex;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 2px;
    gap: 2px;
  }
  .pill-btn {
    border: none;
    background: transparent;
    color: var(--muted);
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .pill-btn.active {
    background: var(--surface-sub);
    color: var(--text);
  }
  .btn-subtle {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--muted);
    padding: 5px 10px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-subtle:hover { color: var(--text); border-color: #b7c2d0; }

  .slide-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: var(--shadow);
    padding: 22px 26px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .slide-header {
    text-align: center;
    max-width: 740px;
    margin: 0 auto;
  }
  .step-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 4px;
  }
  .slide-headline {
    font-size: 24px;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.01em;
  }
  .slide-sentence {
    font-size: 15px;
    color: var(--muted);
    margin: 0;
  }

  .stage-wrap {
    flex: 1;
    min-height: 410px;
    background: var(--surface-sub);
    border: 1px solid var(--border);
    border-radius: 10px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .stage-wrap.scrollable {
    overflow: auto;
    justify-content: flex-start;
  }
  svg#dag-svg {
    display: block;
    width: 100%;
    height: 410px;
  }
  .stage-wrap.scrollable svg#dag-svg {
    width: auto;
    min-width: 100%;
  }
  .stage-hint {
    position: absolute;
    right: 12px;
    bottom: 9px;
    font-size: 11.5px;
    color: var(--muted);
    background: rgba(255, 255, 255, 0.88);
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    pointer-events: none;
  }
  .node-group {
    cursor: pointer;
    transition: opacity 0.2s ease;
  }
  .node-group:hover rect.node-box {
    stroke-width: 2.2;
  }

  /* Overview & Vision conceptual slides */
  .overview-stage {
    width: 100%;
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 18px;
  }
  .two-col {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
  }
  .four-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
  }
  .jump-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
  }
  @media (max-width: 860px) {
    .two-col, .four-grid, .jump-grid { grid-template-columns: 1fr; }
  }
  .ov-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px 20px;
  }
  .ov-card.accent {
    border-color: #bfdbfe;
    background: #f8fbff;
  }
  .ov-card h3 {
    margin: 0 0 10px;
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
  }
  .ov-card ul {
    margin: 0;
    padding-left: 18px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 13.5px;
    color: var(--muted);
  }
  .ov-card li strong { color: var(--text); }
  .ov-card p {
    margin: 0;
    font-size: 13.5px;
    color: var(--muted);
  }
  .ov-num {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    color: var(--accent);
    background: var(--accent-soft);
    padding: 2px 7px;
    border-radius: 999px;
    margin-bottom: 6px;
  }
  .charge-flow {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
  }
  .jump-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    text-align: left;
    cursor: pointer;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }
  .jump-card:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
  }
  .jump-tag {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--accent);
    margin-bottom: 6px;
  }
  .jump-title {
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
  }
  .jump-desc {
    font-size: 13px;
    color: var(--muted);
  }

  .footer-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    padding-top: 4px;
    border-top: 1px solid var(--border);
  }
  .nav-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .btn {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    padding: 6px 13px;
    border-radius: 7px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
  }
  .btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary {
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
  }
  .btn-primary:hover:not(:disabled) { color: #ffffff; opacity: 0.92; }
  .dots {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0 6px;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    border: none;
    background: #cbd5e1;
    padding: 0;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .dot.active {
    width: 24px;
    background: var(--accent);
  }
  .dot.done {
    background: #86efac;
  }

  .context-strip {
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
  }
  .mini-meter {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
  }
  .mini-track {
    width: 110px;
    height: 8px;
    background: var(--surface-sub);
    border: 1px solid var(--border);
    border-radius: 999px;
    overflow: hidden;
  }
  .mini-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
  }
  .mono { font-family: var(--font-mono); }

  /* Slide-over detail drawer — hidden until a node is clicked */
  .drawer-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.18);
    z-index: 40;
  }
  .drawer-backdrop.open { display: block; }
  .drawer {
    display: none;
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(460px, 94vw);
    background: var(--surface);
    border-left: 1px solid var(--border);
    box-shadow: var(--drawer-shadow);
    z-index: 50;
    flex-direction: column;
  }
  .drawer.open { display: flex; }
  .drawer-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    padding: 16px 18px 12px;
    border-bottom: 1px solid var(--border);
  }
  .drawer-title {
    font-size: 16px;
    font-weight: 700;
  }
  .close-btn {
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--muted);
    width: 28px;
    height: 28px;
    border-radius: 6px;
    font-size: 16px;
    line-height: 1;
    cursor: pointer;
  }
  .close-btn:hover { color: var(--text); border-color: var(--text); }
  .drawer-body {
    padding: 16px 18px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .drawer-summary {
    font-size: 13.5px;
    color: var(--text);
    margin: 0;
  }
  .snippet-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
    margin-bottom: 6px;
  }
  pre.code-block {
    margin: 0;
    font-family: var(--font-mono);
    font-size: 11.5px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
    background: var(--surface-sub);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    max-height: 280px;
    overflow: auto;
  }
  .diff-add { color: var(--green); }
  .diff-del { color: var(--red); }
  .diff-hunk { color: var(--accent); font-weight: 600; }
  details.raw-details {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 11px;
    background: var(--surface);
  }
  details.raw-details summary {
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
  }
  details.raw-details[open] summary {
    margin-bottom: 8px;
    color: var(--text);
  }
</style>
</head>
<body>
<div class="page">
  <header class="topbar">
    <div class="topbar-left">
      <div class="logo-pill">ARES</div>
      <div class="tabs" id="scenario-tabs"></div>
    </div>
    <div class="topbar-right">
      <div class="pill-group" id="variant-controls"></div>
      <label class="btn-subtle" title="Load a JSON bundle exported by python -m ares.ext.viz">
        Load JSON
        <input type="file" id="load-json-input" accept=".json" style="display:none" />
      </label>
    </div>
  </header>

  <main class="slide-card">
    <div class="slide-header">
      <div class="step-eyebrow" id="step-eyebrow">STEP 1 OF 5</div>
      <h1 class="slide-headline" id="beat-headline"></h1>
      <p class="slide-sentence" id="beat-narration"></p>
    </div>

    <div class="stage-wrap" id="stage-wrap">
      <div class="overview-stage" id="overview-stage" style="display:none"></div>
      <svg id="dag-svg" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="stage-hint" id="stage-hint">Click any box for prompt or artifact</div>
    </div>

    <footer class="footer-bar">
      <div class="nav-controls">
        <button class="btn" id="prev-beat" type="button">&larr; Prev</button>
        <div class="dots" id="beat-dots"></div>
        <button class="btn btn-primary" id="next-beat" type="button">Next &rarr;</button>
        <button class="btn" id="play-toggle" type="button">Play</button>
      </div>

      <div class="context-strip">
        <div class="mini-meter">
          <span style="font-weight:600">ARES</span>
          <div class="mini-track"><div class="mini-fill" id="bar-ares" style="background:var(--green)"></div></div>
          <span class="mono" id="bar-ares-label"></span>
        </div>
        <div class="mini-meter">
          <span style="font-weight:600;color:var(--muted)">Standard Chat</span>
          <div class="mini-track"><div class="mini-fill" id="bar-baseline" style="background:var(--red)"></div></div>
          <span class="mono" id="bar-baseline-label"></span>
        </div>
      </div>
    </footer>
  </main>
</div>

<div class="drawer-backdrop" id="drawer-backdrop"></div>
<aside class="drawer" id="detail-drawer" aria-hidden="true">
  <div class="drawer-head">
    <div>
      <div class="drawer-title" id="drawer-title">Details</div>
      <div style="font-size:12px;color:var(--muted);margin-top:2px" id="drawer-subtitle"></div>
    </div>
    <button class="close-btn" id="drawer-close" type="button" title="Close (Esc)">&times;</button>
  </div>
  <div class="drawer-body" id="drawer-body"></div>
</aside>

<script>
const ARES_PAYLOAD = __PAYLOAD_JSON__;

(function() {
  const scenarios = [];
  if (Array.isArray(ARES_PAYLOAD.curatedStories)) {
    for (const story of ARES_PAYLOAD.curatedStories) {
      scenarios.push({ ...story, kind: story.kind || "story" });
    }
  }
  if (ARES_PAYLOAD.realRunBundle) {
    scenarios.push(buildRealRunScenario(ARES_PAYLOAD.realRunBundle));
  }

  let state = {
    scenarioIdx: 0,
    variantIdx: 0,
    beatIdx: 0,
    selectedTaskId: null,
    drawerOpen: false,
    playTimer: null,
  };

  function buildRealRunScenario(bundle) {
    const tasks = bundle.tasks || {};
    const events = bundle.events || [];
    const artifacts = bundle.artifacts || {};
    const taskIds = Object.keys(tasks);
    const totalTokens = (bundle.usage && bundle.usage.total_tokens) || 0;
    const llmCalls = (bundle.usage && bundle.usage.llm_calls) || 0;
    const cancelledCount = taskIds.filter((id) => tasks[id].status === "cancelled").length;

    const waves = [];
    const seenTasks = new Set();
    let currentWave = [];
    for (const ev of events) {
      if (ev.task_id && tasks[ev.task_id] && !seenTasks.has(ev.task_id)) {
        seenTasks.add(ev.task_id);
        currentWave.push(ev.task_id);
      }
      if (ev.event_type === "checkpoint_saved" || ev.event_type === "run_finished") {
        if (currentWave.length > 0 || waves.length === 0) {
          waves.push([...seenTasks]);
          currentWave = [];
        }
      }
    }
    if (waves.length === 0) waves.push(taskIds);

    const maxWaves = Math.min(waves.length, 5);
    const sampledWaves = [];
    for (let i = 0; i < maxWaves; i++) {
      const idx = Math.round((i * (waves.length - 1)) / Math.max(1, maxWaves - 1));
      sampledWaves.push(waves[idx]);
    }
    sampledWaves[sampledWaves.length - 1] = taskIds;

    const beats = sampledWaves.map((visibleIds, idx) => {
      const isFinal = idx === sampledWaves.length - 1;
      const avgTok = Math.min(900, Math.max(250, Math.round(totalTokens / Math.max(1, llmCalls || 1))));
      return {
        headline: isFinal ? `Real run finished (${bundle.status})` : `Live checkpoint wave ${idx + 1}`,
        narration: isFinal
          ? `Real ARES run (${taskIds.length} tasks, ${cancelledCount} cancelled early). Click any box to inspect artifacts.`
          : `Showing ${visibleIds.length} of ${taskIds.length} tasks materialized from state.json.`,
        visibleTaskIds: visibleIds,
        activeTaskIds: visibleIds.slice(-3),
        statusOverrides: {},
        subtitleOverrides: {},
        contextMetrics: {
          aresPromptTokens: avgTok,
          aresCapTokens: 4096,
          baselinePromptTokens: Math.max(1200, totalTokens),
          baselineCapTokens: 4096,
          baselineStatus: totalTokens > 4096 ? "OVERFLOW" : "OK",
        },
      };
    });

    const enrichedTasks = {};
    for (const [tid, t] of Object.entries(tasks)) {
      let roleBadge = t.deterministic ? "verifier" : "llm";
      if (t.status === "suspended" || t.suspend_reason) roleBadge = "gate";
      else if (t.continuation_id || (t.wait_for && t.wait_for.length > 0)) roleBadge = "orchestrator";
      enrichedTasks[tid] = {
        ...t,
        viz: {
          roleBadge,
          label: t.state,
          subtitle: `${t.id} · ${t.status}`,
          summary: `Real checkpointed task ${t.id} (${t.state}), status=${t.status}, duration=${Number(t.duration_ms || 0).toFixed(1)} ms.`,
          snippetTitle: "Task input & output",
          snippet: JSON.stringify({ input: t.input, output: t.output }, null, 2),
        },
      };
    }

    const taggedEvents = events.map((ev, i) => ({
      ...ev,
      beat: Math.min(beats.length - 1, Math.floor((i * beats.length) / Math.max(1, events.length))),
    }));

    return {
      kind: "real",
      id: "real_run",
      tabLabel: "Live Checkpoint",
      variants: [
        {
          id: "real",
          label: bundle.run_id,
          beats,
          tasks: enrichedTasks,
          events: taggedEvents,
          artifacts,
        },
      ],
    };
  }

  function currentScenario() { return scenarios[state.scenarioIdx] || scenarios[0]; }
  function currentVariant() { const sc = currentScenario(); return sc.variants[state.variantIdx] || sc.variants[0]; }
  function currentBeat() { const vr = currentVariant(); return vr.beats[state.beatIdx] || vr.beats[0]; }

  function stopPlay() {
    if (state.playTimer !== null) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
    document.getElementById("play-toggle").textContent = "Play";
  }

  function togglePlay() {
    if (state.playTimer !== null) { stopPlay(); return; }
    document.getElementById("play-toggle").textContent = "Pause";
    state.playTimer = setInterval(() => {
      const vr = currentVariant();
      if (state.beatIdx < vr.beats.length - 1) {
        state.beatIdx += 1;
        render();
      } else {
        stopPlay();
      }
    }, 2600);
  }

  function closeDrawer() {
    state.drawerOpen = false;
    document.getElementById("drawer-backdrop").classList.remove("open");
    const d = document.getElementById("detail-drawer");
    d.classList.remove("open");
    d.setAttribute("aria-hidden", "true");
  }

  function openDrawer(taskId) {
    state.selectedTaskId = taskId;
    state.drawerOpen = true;
    renderDrawer();
    document.getElementById("drawer-backdrop").classList.add("open");
    const d = document.getElementById("detail-drawer");
    d.classList.add("open");
    d.setAttribute("aria-hidden", "false");
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderDiffHtml(text) {
    return text
      .split("\\n")
      .map((line) => {
        const escaped = escHtml(line);
        if (line.startsWith("+") && !line.startsWith("+++")) return `<span class="diff-add">${escaped}</span>`;
        if (line.startsWith("-") && !line.startsWith("---")) return `<span class="diff-del">${escaped}</span>`;
        if (line.startsWith("@@") || line.startsWith("diff --git")) return `<span class="diff-hunk">${escaped}</span>`;
        return escaped;
      })
      .join("\\n");
  }

  function collectArtifactIdsForTask(task, artifacts) {
    const ids = new Set();
    if (task.spilled_artifact_id && artifacts[task.spilled_artifact_id]) ids.add(task.spilled_artifact_id);
    if (Array.isArray(task.artifact_refs)) {
      for (const r of task.artifact_refs) {
        if (r && r.ref_id && artifacts[r.ref_id]) ids.add(r.ref_id);
      }
    }
    if (task.viz && task.viz.diffArtifactRef && artifacts[task.viz.diffArtifactRef]) {
      ids.add(task.viz.diffArtifactRef);
    }
    function scanObj(val) {
      if (!val || typeof val !== "object") {
        if (typeof val === "string" && val.startsWith("art_") && artifacts[val]) ids.add(val);
        return;
      }
      if (typeof val.__ares_ref__ === "string" && artifacts[val.__ares_ref__]) ids.add(val.__ares_ref__);
      for (const v of Object.values(val)) scanObj(v);
    }
    scanObj(task.input);
    scanObj(task.output);
    return Array.from(ids);
  }

  function renderTabs() {
    const container = document.getElementById("scenario-tabs");
    container.innerHTML = "";
    scenarios.forEach((sc, idx) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tab-btn" + (idx === state.scenarioIdx ? " active" : "");
      btn.textContent = sc.tabLabel;
      btn.onclick = () => {
        stopPlay();
        closeDrawer();
        state.scenarioIdx = idx;
        state.variantIdx = 0;
        state.beatIdx = 0;
        state.selectedTaskId = null;
        render();
      };
      container.appendChild(btn);
    });
  }

  function renderHeaderAndFooter() {
    const sc = currentScenario();
    const vr = currentVariant();
    const beat = currentBeat();

    const varRow = document.getElementById("variant-controls");
    if (sc.variants.length <= 1) {
      varRow.style.display = "none";
    } else {
      varRow.style.display = "inline-flex";
      varRow.innerHTML = "";
      sc.variants.forEach((v, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "pill-btn" + (idx === state.variantIdx ? " active" : "");
        btn.textContent = v.label;
        btn.onclick = () => {
          stopPlay();
          closeDrawer();
          state.variantIdx = idx;
          state.beatIdx = 0;
          state.selectedTaskId = null;
          render();
        };
        varRow.appendChild(btn);
      });
    }

    document.getElementById("step-eyebrow").textContent = `Step ${state.beatIdx + 1} of ${vr.beats.length}`;
    document.getElementById("beat-headline").textContent = beat.headline;
    document.getElementById("beat-narration").textContent = beat.narration;

    document.getElementById("prev-beat").disabled = state.beatIdx <= 0;
    document.getElementById("next-beat").disabled = state.beatIdx >= vr.beats.length - 1;

    const dots = document.getElementById("beat-dots");
    dots.innerHTML = "";
    vr.beats.forEach((_, idx) => {
      const dot = document.createElement("button");
      dot.type = "button";
      dot.className = "dot" + (idx === state.beatIdx ? " active" : idx < state.beatIdx ? " done" : "");
      dot.title = `Step ${idx + 1}`;
      dot.onclick = () => {
        stopPlay();
        state.beatIdx = idx;
        render();
      };
      dots.appendChild(dot);
    });

    const contextStrip = document.querySelector(".context-strip");
    if (sc.kind === "overview") {
      contextStrip.style.display = "none";
      return;
    }
    contextStrip.style.display = "flex";
    const m = beat.contextMetrics || {};
    const aresTok = m.aresPromptTokens || 0;
    const aresCap = m.aresCapTokens || 4096;
    const baseTok = m.baselinePromptTokens || 0;
    const baseCap = m.baselineCapTokens || 4096;

    document.getElementById("bar-ares").style.width = `${Math.min(100, Math.round((aresTok / aresCap) * 100))}%`;
    document.getElementById("bar-ares-label").textContent = `${aresTok} tok`;
    document.getElementById("bar-baseline").style.width = `${Math.min(100, Math.round((baseTok / baseCap) * 100))}%`;
    const overflowTag = baseTok > baseCap ? " (Overflow)" : "";
    document.getElementById("bar-baseline-label").textContent = `${baseTok} tok${overflowTag}`;
    document.getElementById("bar-baseline-label").style.color = baseTok > baseCap ? "var(--red)" : "var(--muted)";
  }

  function renderOverviewStage(beat) {
    const stageWrap = document.getElementById("stage-wrap");
    const ov = document.getElementById("overview-stage");
    const svg = document.getElementById("dag-svg");
    const hint = document.getElementById("stage-hint");

    stageWrap.classList.remove("scrollable");
    svg.style.display = "none";
    hint.style.display = "none";
    ov.style.display = "flex";

    if (beat.layout === "goals") {
      ov.innerHTML = `<div class="two-col">
        <div class="ov-card">
          <h3>Why standard chat loops break</h3>
          <ul>
            <li><strong>Context rot:</strong> raw data, papers, and errors overflow 4K–16K local models.</li>
            <li><strong>Self-grading bias:</strong> one chat loop hides decoys and broken invariants.</li>
            <li><strong>Black-box fragility:</strong> crashes lose work; hard to audit or pause.</li>
          </ul>
        </div>
        <div class="ov-card accent">
          <h3>What ARES is suited for</h3>
          <ul>
            <li><strong>Propose → Verify → Refine loops:</strong> solvers, molecules, code.</li>
            <li><strong>Coarse-to-fine screening:</strong> triage papers and datasets, then verify blindly.</li>
            <li><strong>Long or human-gated pipelines:</strong> pause cleanly for data or scientist sign-off.</li>
          </ul>
        </div>
      </div>`;
    } else if (beat.layout === "methodology") {
      ov.innerHTML = `<div class="four-grid">
        <div class="ov-card">
          <span class="ov-num">1 · Context Switching</span>
          <h3>Isolated local frames</h3>
          <p>Each step sees only its own compact input, keeping prompt size flat O(1).</p>
        </div>
        <div class="ov-card">
          <span class="ov-num">2 · ArtifactStore</span>
          <h3>Out-of-line data &amp; slicing</h3>
          <p>Large datasets, papers, and logs stay on disk; models read bounded slices.</p>
        </div>
        <div class="ov-card">
          <span class="ov-num">3 · Blind Verification</span>
          <h3>Tools outside the LLM loop</h3>
          <p>Verifiers see only candidate + data — never LLM prose — or return unverified.</p>
        </div>
        <div class="ov-card">
          <span class="ov-num">4 · Parallel &amp; Auditable</span>
          <h3>Race branches + checkpoints</h3>
          <p>Cancel losing branches on first win, checkpoint waves, and trace every prompt.</p>
        </div>
      </div>`;
    } else if (beat.layout === "charge") {
      ov.innerHTML = `
        <div class="charge-flow">
          <svg viewBox="0 0 820 135" width="100%" height="125" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <marker id="ov-arr" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
                <path d="M0,1 L7,4 L0,7 Z" fill="#64748b" />
              </marker>
            </defs>
            <rect x="20" y="28" width="175" height="68" rx="10" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.6" />
            <text x="107" y="54" text-anchor="middle" font-size="13" font-weight="700" fill="#111827">1 · Propose</text>
            <text x="107" y="75" text-anchor="middle" font-size="11" fill="#4b5563">Local LLM (isolated frame)</text>

            <path d="M 195 62 L 265 62" stroke="#64748b" stroke-width="1.7" marker-end="url(#ov-arr)" />

            <rect x="267" y="28" width="195" height="68" rx="10" fill="#f0fdf4" stroke="#15803d" stroke-width="1.6" />
            <text x="364" y="54" text-anchor="middle" font-size="13" font-weight="700" fill="#111827">2 · Verify (Blind)</text>
            <text x="364" y="75" text-anchor="middle" font-size="11" fill="#4b5563">Deterministic check outside loop</text>

            <path d="M 462 62 L 532 62" stroke="#64748b" stroke-width="1.7" marker-end="url(#ov-arr)" />

            <rect x="534" y="28" width="130" height="68" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
            <text x="599" y="54" text-anchor="middle" font-size="13" font-weight="700" fill="#111827">Passes?</text>
            <text x="599" y="75" text-anchor="middle" font-size="11" fill="#15803d">Yes → Done / Gate</text>

            <path d="M 664 62 L 716 62" stroke="#15803d" stroke-width="1.7" marker-end="url(#ov-arr)" />
            <rect x="718" y="36" width="84" height="52" rx="9" fill="#f0fdf4" stroke="#15803d" stroke-width="1.5" />
            <text x="760" y="66" text-anchor="middle" font-size="12" font-weight="700" fill="#15803d">Certified</text>

            <path d="M 599 96 C 599 124, 107 124, 107 98" fill="none" stroke="#64748b" stroke-width="1.6" stroke-dasharray="5 4" marker-end="url(#ov-arr)" />
            <text x="355" y="122" text-anchor="middle" font-size="11" font-weight="600" fill="#5b6472">3 · Refine (No → Context-switched slice only)</text>
          </svg>
        </div>
        <div class="two-col">
          <div class="ov-card">
            <h3>ChARGe (SC26 Workshops)</h3>
            <p>Showed <strong>Propose → Verify → Refine</strong> + two-level <strong>Context Switching</strong> (GPT 5.4 + local Ollama) lifted molecule optimization (63% → 111%) and synthesis (12% → 40%).</p>
          </div>
          <div class="ov-card accent">
            <h3>ARES (This Runtime)</h3>
            <p>Moves Context Switching, deduplication, early stopping, and checkpoints out of prompts into the runtime — so <strong>small local models</strong> run reliably with full auditability.</p>
          </div>
        </div>`;
    } else {
      const cards = scenarios.slice(1).map((sc, offset) => {
        const targetIdx = offset + 1;
        const tag = sc.kind === "real" ? "Real Repo Run" : `Example ${offset + 1}`;
        const desc = sc.id === "physics_sim"
          ? "Parallel Duffing ODE proposals, blind physics check, and RK45 refinement."
          : sc.id === "literature_data"
          ? "Coarse-to-fine screening across 40 papers + 6 cohort tables; rejects a batch decoy."
          : "Real checkpointed run from this repository with spilled git diff artifacts.";
        return `<button type="button" class="jump-card" data-jump-idx="${targetIdx}">
          <div class="jump-tag">${escHtml(tag)}</div>
          <div class="jump-title">${escHtml(sc.tabLabel)}</div>
          <div class="jump-desc">${escHtml(desc)}</div>
        </button>`;
      }).join("");
      ov.innerHTML = `<div class="jump-grid">${cards}</div>`;
      ov.querySelectorAll(".jump-card").forEach((btn) => {
        btn.addEventListener("click", () => {
          stopPlay();
          closeDrawer();
          state.scenarioIdx = Number(btn.getAttribute("data-jump-idx"));
          state.variantIdx = 0;
          state.beatIdx = 0;
          state.selectedTaskId = null;
          render();
        });
      });
    }
  }

  function renderDag() {
    const sc = currentScenario();
    const beat = currentBeat();
    if (sc.kind === "overview") {
      renderOverviewStage(beat);
      return;
    }
    document.getElementById("overview-stage").style.display = "none";
    document.getElementById("dag-svg").style.display = "block";
    document.getElementById("stage-hint").style.display = "block";

    const vr = currentVariant();
    const stageWrap = document.getElementById("stage-wrap");
    const svg = document.getElementById("dag-svg");
    const visibleSet = new Set(beat.visibleTaskIds || Object.keys(vr.tasks));
    const activeSet = new Set(beat.activeTaskIds || []);
    const overrides = beat.statusOverrides || {};
    const subOverrides = beat.subtitleOverrides || {};

    const tasks = {};
    for (const id of visibleSet) {
      if (vr.tasks[id]) {
        tasks[id] = {
          ...vr.tasks[id],
          effectiveStatus: overrides[id] || vr.tasks[id].status,
          effectiveSubtitle: subOverrides[id] || (vr.tasks[id].viz && vr.tasks[id].viz.subtitle) || vr.tasks[id].state,
        };
      }
    }
    const ids = Object.keys(tasks);

    const col = {};
    ids.forEach((id) => { col[id] = 0; });
    for (let iter = 0; iter < ids.length + 2; iter++) {
      for (const id of ids) {
        const t = tasks[id];
        if (t.parent_id && tasks[t.parent_id]) col[id] = Math.max(col[id], col[t.parent_id] + 1);
        for (const dep of (t.depends_on || [])) {
          if (tasks[dep]) col[id] = Math.max(col[id], col[dep] + 1);
        }
        if (t.continuation_id && tasks[t.continuation_id]) {
          col[t.continuation_id] = Math.max(col[t.continuation_id], col[id] + 1);
        }
      }
    }

    const columns = {};
    for (const id of ids) {
      const c = col[id];
      if (!columns[c]) columns[c] = [];
      columns[c].push(id);
    }

    const colKeys = Object.keys(columns).map(Number).sort((a, b) => a - b);
    const isLargeRealDag = colKeys.length > 6 || ids.length > 14;
    if (isLargeRealDag) stageWrap.classList.add("scrollable");
    else stageWrap.classList.remove("scrollable");

    const nodeW = 220;
    const nodeH = 72;
    const colGap = 58;
    const rowGap = 28;
    const padX = 34;
    const padY = 28;

    const maxRows = Math.max(1, ...colKeys.map((c) => columns[c].length));
    const totalH = Math.max(390, padY * 2 + maxRows * nodeH + Math.max(0, maxRows - 1) * rowGap);
    const totalW = Math.max(760, padX * 2 + colKeys.length * nodeW + Math.max(0, colKeys.length - 1) * colGap);

    const pos = {};
    colKeys.forEach((cIdx, visualCol) => {
      const group = columns[cIdx];
      const x = padX + visualCol * (nodeW + colGap);
      const groupH = group.length * nodeH + Math.max(0, group.length - 1) * rowGap;
      const startY = Math.max(padY, (totalH - groupH) / 2);
      group.forEach((id, rIdx) => {
        pos[id] = { x, y: startY + rIdx * (nodeH + rowGap) };
      });
    });

    const statusMeta = {
      done: { label: "DONE", color: "var(--green)", fill: "var(--green-soft)" },
      running: { label: "RUNNING", color: "var(--accent)", fill: "var(--accent-soft)" },
      waiting: { label: "WAITING", color: "var(--purple)", fill: "var(--purple-soft)" },
      pending: { label: "PENDING", color: "var(--muted)", fill: "#ffffff" },
      suspended: { label: "PAUSED", color: "var(--amber)", fill: "var(--amber-soft)" },
      failed: { label: "FAILED", color: "var(--red)", fill: "var(--red-soft)" },
      cancelled: { label: "CANCELLED", color: "var(--red)", fill: "#ffffff" },
    };

    const parts = [];
    parts.push(`<defs>
      <marker id="arrow-dep" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,1 L7,4 L0,7 Z" fill="#94a3b8" />
      </marker>
    </defs>`);

    const drawnEdges = new Set();
    function addEdge(fromId, toId, dashed) {
      if (!pos[fromId] || !pos[toId]) return;
      const key = `${fromId}->${toId}`;
      if (drawnEdges.has(key)) return;
      drawnEdges.add(key);
      const p1 = pos[fromId];
      const p2 = pos[toId];
      const x1 = p1.x + nodeW;
      const y1 = p1.y + nodeH / 2;
      const x2 = p2.x;
      const y2 = p2.y + nodeH / 2;
      const dx = Math.max(28, Math.abs(x2 - x1) * 0.45);
      const d = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
      const isCancelled = tasks[toId].effectiveStatus === "cancelled";
      const opacity = isCancelled ? "0.35" : "0.85";
      const dashAttr = dashed ? 'stroke-dasharray="5 4"' : "";
      parts.push(`<path d="${d}" fill="none" stroke="#94a3b8" stroke-width="1.7" ${dashAttr} marker-end="url(#arrow-dep)" opacity="${opacity}" />`);
    }

    for (const id of ids) {
      const t = tasks[id];
      if (t.parent_id && tasks[t.parent_id]) addEdge(t.parent_id, id, true);
      for (const dep of (t.depends_on || [])) addEdge(dep, id, false);
      if (t.continuation_id && tasks[t.continuation_id]) addEdge(id, t.continuation_id, false);
    }

    for (const id of ids) {
      const t = tasks[id];
      const p = pos[id];
      const st = statusMeta[t.effectiveStatus] || statusMeta.pending;
      const isActive = activeSet.has(id);
      const isSelected = state.drawerOpen && state.selectedTaskId === id;
      const isCancelled = t.effectiveStatus === "cancelled";

      const fill = isActive ? st.fill : "#ffffff";
      const stroke = isSelected ? "var(--text)" : (isActive ? st.color : "#cbd5e1");
      const strokeW = isSelected ? 2.4 : (isActive ? 2.0 : 1.3);
      const opacity = isCancelled ? 0.48 : (isActive ? 1.0 : 0.82);

      const titleText = ((t.viz && t.viz.label) || t.state).slice(0, 25);
      const subText = String(t.effectiveSubtitle || "").slice(0, 32);

      parts.push(`<g class="node-group" data-task-id="${escHtml(id)}" opacity="${opacity}">
        <rect class="node-box" x="${p.x}" y="${p.y}" width="${nodeW}" height="${nodeH}" rx="10"
              fill="${fill}" stroke="${stroke}" stroke-width="${strokeW}"
              ${isCancelled ? 'stroke-dasharray="4 3"' : ''} />
        <text x="${p.x + 14}" y="${p.y + 27}" font-size="13" font-weight="700" fill="var(--text)" ${isCancelled ? 'text-decoration="line-through"' : ''}>${escHtml(titleText)}</text>
        <text x="${p.x + nodeW - 12}" y="${p.y + 26}" text-anchor="end" font-size="10" font-weight="700" fill="${st.color}">${st.label}</text>
        <text x="${p.x + 14}" y="${p.y + 49}" font-size="11.5" fill="${isActive && !isCancelled ? 'var(--text)' : 'var(--muted)'}">${escHtml(subText)}</text>
      </g>`);
    }

    svg.setAttribute("viewBox", `0 0 ${totalW} ${totalH}`);
    if (isLargeRealDag) {
      svg.setAttribute("width", String(totalW));
      svg.setAttribute("height", String(totalH));
    } else {
      svg.removeAttribute("width");
      svg.removeAttribute("height");
    }
    svg.innerHTML = parts.join("");

    svg.querySelectorAll(".node-group").forEach((el) => {
      el.addEventListener("click", () => {
        openDrawer(el.getAttribute("data-task-id"));
        renderDag();
      });
    });
  }

  function renderDrawer() {
    const vr = currentVariant();
    const beat = currentBeat();
    const task = state.selectedTaskId ? vr.tasks[state.selectedTaskId] : null;
    if (!task) return;

    const effStatus = (beat.statusOverrides && beat.statusOverrides[task.id]) || task.status;
    document.getElementById("drawer-title").textContent = (task.viz && task.viz.label) || task.state;
    document.getElementById("drawer-subtitle").textContent = `Status: ${effStatus.toUpperCase()} · Task ID: ${task.id}`;

    const blocks = [];
    const summary = (task.viz && task.viz.summary) || `Task ${task.id} (${task.state}).`;
    blocks.push(`<p class="drawer-summary">${escHtml(summary)}</p>`);

    const snippetTitle = (task.viz && task.viz.snippetTitle) || "Frame data";
    const snippet = (task.viz && task.viz.snippet) || JSON.stringify(task.output || task.input || {}, null, 2);
    const isDiff = snippet.includes("--- a/") || snippet.includes("diff --git");
    const renderedSnippet = isDiff ? renderDiffHtml(snippet) : escHtml(snippet);
    blocks.push(`<div><div class="snippet-label">${escHtml(snippetTitle)}</div><pre class="code-block">${renderedSnippet}</pre></div>`);

    const artIds = collectArtifactIdsForTask(task, vr.artifacts || {});
    if (artIds.length > 0) {
      const art = vr.artifacts[artIds[0]];
      const artDiff = art.mime_type === "text/x-diff" || String(art.content || "").includes("diff --git");
      const artHtml = artDiff ? renderDiffHtml(art.content || "") : escHtml(art.content || "");
      blocks.push(`<details class="raw-details"><summary>Artifact ${escHtml(art.ref_id)} (${art.byte_size} bytes)</summary><pre class="code-block">${artHtml}</pre></details>`);
    }

    const events = (vr.events || []).filter((ev) => ev.task_id === task.id);
    const rawDump = JSON.stringify({ task_input: task.input, task_output: task.output, events }, null, 2);
    blocks.push(`<details class="raw-details"><summary>Raw JSON &amp; event trace</summary><pre class="code-block">${escHtml(rawDump)}</pre></details>`);

    document.getElementById("drawer-body").innerHTML = blocks.join("");
  }

  function render() {
    renderTabs();
    renderHeaderAndFooter();
    renderDag();
    if (state.drawerOpen) renderDrawer();
  }

  document.getElementById("prev-beat").onclick = () => {
    stopPlay();
    if (state.beatIdx > 0) { state.beatIdx -= 1; render(); }
  };
  document.getElementById("next-beat").onclick = () => {
    stopPlay();
    const vr = currentVariant();
    if (state.beatIdx < vr.beats.length - 1) { state.beatIdx += 1; render(); }
  };
  document.getElementById("play-toggle").onclick = togglePlay;
  document.getElementById("drawer-close").onclick = () => { closeDrawer(); renderDag(); };
  document.getElementById("drawer-backdrop").onclick = () => { closeDrawer(); renderDag(); };

  document.getElementById("load-json-input").addEventListener("change", (ev) => {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const bundle = JSON.parse(String(reader.result));
        const sc = buildRealRunScenario(bundle);
        const existingIdx = scenarios.findIndex((s) => s.id === "real_run");
        if (existingIdx >= 0) scenarios[existingIdx] = sc;
        else scenarios.push(sc);
        stopPlay();
        closeDrawer();
        state.scenarioIdx = existingIdx >= 0 ? existingIdx : scenarios.length - 1;
        state.variantIdx = 0;
        state.beatIdx = sc.variants[0].beats.length - 1;
        state.selectedTaskId = null;
        render();
      } catch (err) {
        alert("Invalid ARES run bundle JSON: " + err);
      }
    };
    reader.readAsText(file);
  });

  document.addEventListener("keydown", (ev) => {
    if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA")) return;
    if (ev.key === "Escape" && state.drawerOpen) {
      closeDrawer();
      renderDag();
    } else if (ev.key === "ArrowRight") {
      stopPlay();
      const vr = currentVariant();
      if (state.beatIdx < vr.beats.length - 1) { state.beatIdx += 1; render(); }
    } else if (ev.key === "ArrowLeft") {
      stopPlay();
      if (state.beatIdx > 0) { state.beatIdx -= 1; render(); }
    } else if (ev.key === " ") {
      ev.preventDefault();
      togglePlay();
    } else if (/^[1-9]$/.test(ev.key)) {
      const idx = Number(ev.key) - 1;
      if (idx < scenarios.length) {
        stopPlay();
        closeDrawer();
        state.scenarioIdx = idx;
        state.variantIdx = 0;
        state.beatIdx = 0;
        state.selectedTaskId = null;
        render();
      }
    }
  });

  render();
})();
</script>
</body>
</html>
"""


def render_visualizer_html(
    *,
    real_run_bundle: dict[str, Any] | None = None,
    include_curated_stories: bool = True,
    title: str = "ARES v2 — Interactive Reasoning Pipeline Visualizer",
) -> str:
    """Render a self-contained single-file HTML slide deck and DAG/artifact visualizer."""
    curated = (
        [_story_overview_vision(), _story_physics_simulation(), _story_literature_data()]
        if include_curated_stories
        else []
    )
    payload = {
        "curatedStories": curated,
        "realRunBundle": real_run_bundle,
    }
    return (
        _HTML_TEMPLATE.replace("__TITLE__", html.escape(title, quote=True))
        .replace("__PAYLOAD_JSON__", _safe_json_for_script(payload))
    )


def write_visualizer_html(
    output_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    include_curated_stories: bool = True,
    title: str = "ARES v2 — Interactive Reasoning Pipeline Visualizer",
) -> Path:
    """Export an optional ``run_dir`` and write the standalone HTML visualizer to ``output_path``."""
    bundle = export_run_bundle(run_dir) if run_dir is not None else None
    rendered = render_visualizer_html(
        real_run_bundle=bundle,
        include_curated_stories=include_curated_stories,
        title=title,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for ``python -m ares.ext.viz``."""
    parser = argparse.ArgumentParser(
        prog="python -m ares.ext.viz",
        description="Export a checkpointed ARES run directory into an interactive HTML trace viewer.",
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Optional ARES checkpoint directory containing state.json, events.jsonl, and artifacts/.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="ares_visualizer.html",
        help="Destination HTML path (default: ares_visualizer.html).",
    )
    parser.add_argument(
        "--bundle-json",
        default=None,
        help="Optional path to also write the exported run bundle JSON.",
    )
    parser.add_argument(
        "--no-curated",
        action="store_true",
        help="Omit curated Story 1 / Story 2 walkthroughs and render only the provided run_dir.",
    )
    parser.add_argument(
        "--title",
        default="ARES v2 — Interactive Reasoning Pipeline Visualizer",
        help="HTML document title.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint writing an interactive HTML visualizer (and optional JSON bundle)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.no_curated and args.run_dir is None:
        parser.error("--no-curated requires a run_dir")

    bundle = export_run_bundle(args.run_dir) if args.run_dir is not None else None
    if args.bundle_json is not None:
        if bundle is None:
            parser.error("--bundle-json requires a run_dir")
        bundle_path = Path(args.bundle_json)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    html_text = render_visualizer_html(
        real_run_bundle=bundle,
        include_curated_stories=not args.no_curated,
        title=args.title,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"Wrote interactive ARES visualizer to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
