"""Offline tests proving ARES PDA keeps prompts O(1) bounded across deep chains."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from ares import ArtifactStore, CheckpointStore, RunBudget, TaskSpec
from ares.ext.llm import ContextLengthExceededError, LLMMessage, LLMResponse, OpenAICompatClient
from examples.long_horizon_small_context.machine import (
    HopExtraction,
    InstrumentedLLMClient,
    MockSmallContextLLM,
    build_machine,
    create_llm_client,
    generate_multihop_case,
    run_ares,
    run_benchmark,
    run_rolling_baseline,
)
from examples.long_horizon_small_context.run import main as cli_main


def test_ares_stays_o1_bounded_and_accurate_across_horizons_while_baseline_overflows() -> None:
    prompt_cap = 2000
    horizons = (4, 8, 12)
    peaks: list[int] = []

    for hops in horizons:
        summary = run_benchmark(
            hops=hops,
            seed=42,
            noise_lines=25,
            include_decoy=True,
            prompt_char_cap=prompt_cap,
            max_inline_bytes=600,
        )
        assert summary.ares_correct is True
        assert summary.ares_selected == summary.true_candidate
        assert summary.peak_prompt_chars < prompt_cap
        assert summary.peak_prompt_chars < 1000
        peaks.append(summary.peak_prompt_chars)

        if hops == 4:
            assert summary.baseline_overflowed is False
            assert summary.baseline_correct is True
            assert summary.baseline_peak_prompt_chars <= prompt_cap
        else:
            assert summary.baseline_overflowed is True
            assert summary.baseline_correct is False
            assert summary.baseline_selected is None
            assert summary.baseline_peak_prompt_chars > prompt_cap

    # O(1) invariance: peak prompt chars barely changes from H=4 to H=12.
    assert max(peaks) - min(peaks) < 100


def test_decoy_is_rejected_and_hallucinated_verification_fails_deterministically() -> None:
    case = generate_multihop_case(hops=8, seed=7, noise_lines=25, include_decoy=True)
    assert case.decoy_candidate is not None
    assert case.decoy_candidate != case.true_candidate

    class HallucinatingDecoyLLM:
        """Malicious mock LLM that falsely claims root_cause_verified on the decoy."""

        def complete(
            self,
            messages: Sequence[LLMMessage],
            *,
            system: str | None = None,
            response_model: type[BaseModel] | None = None,
            temperature: float = 0.0,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            _ = system, temperature, max_tokens
            latest = messages[-1].content
            if f"Candidate: {case.decoy_candidate}" in latest:
                extraction = HopExtraction(next_token=None, terminal="root_cause_verified")
                return LLMResponse(
                    text=extraction.model_dump_json(),
                    parsed=extraction if response_model is HopExtraction else None,
                )
            return MockSmallContextLLM().complete(
                messages, response_model=response_model
            )

    res = run_ares(case, prompt_char_cap=2000, llm=HallucinatingDecoyLLM())
    assert res.correct is True
    assert res.selected == case.true_candidate
    assert case.decoy_candidate not in res.verified_candidates


def test_large_hop_logs_spill_and_slice_within_o1_prompt_cap(tmp_path: Path) -> None:
    # 120 lines (~1900 chars per hop) exceeds max_inline_bytes=600 and approaches prompt_cap=2000,
    # yet ARES reads bounded ArtifactStore slices so prompts stay strictly < 2000 chars.
    case = generate_multihop_case(hops=8, seed=99, noise_lines=120, include_decoy=True)
    res = run_ares(
        case,
        prompt_char_cap=1600,
        max_inline_bytes=600,
        run_dir=tmp_path / "large_log_run",
    )
    assert res.correct is True
    assert res.peak_prompt_chars < 1600

    store = ArtifactStore(tmp_path / "large_log_run" / "artifacts")
    assert len(store.list_refs()) >= case.hops

    baseline = run_rolling_baseline(case, prompt_char_cap=1600)
    assert baseline.overflowed is True
    assert baseline.correct is False


def test_mock_enforces_context_length_exceeded_error() -> None:
    case = generate_multihop_case(hops=4, seed=42, noise_lines=25)
    client = InstrumentedLLMClient(MockSmallContextLLM(), prompt_char_cap=250)
    machine = build_machine(case, llm=client, prompt_char_cap=250)
    result = machine.run(TaskSpec(state="start"))
    assert result.status == "failed"
    assert any(
        t.error and "exceeds cap 250" in t.error
        for t in result.tasks.values()
        if t.status == "failed"
    )

    baseline = run_rolling_baseline(case, prompt_char_cap=250)
    assert baseline.overflowed is True
    assert baseline.error is not None
    assert "exceeds cap 250" in baseline.error


def test_mid_chain_crash_and_resume_does_not_rerun_done_hops(tmp_path: Path) -> None:
    case = generate_multihop_case(hops=8, seed=17, noise_lines=25, include_decoy=True)
    full_run_dir = tmp_path / "full_run"
    crash_dir = tmp_path / "crash_run"

    machine1 = build_machine(case, prompt_char_cap=2000, max_inline_bytes=600)
    snapshot_captured = False

    def on_event(event: Any) -> None:
        nonlocal snapshot_captured
        if snapshot_captured or event.event_type != "task_started":
            return
        state_file = full_run_dir / "state.json"
        if not state_file.exists():
            return
        snap = json.loads(state_file.read_text(encoding="utf-8"))
        # Early-failing candidates finish "done" while the true target has waiting hops 0..3
        # and a pending hop >= 4 in the saved checkpoint.
        progressed_true_hops = [
            t
            for t in snap["tasks"].values()
            if t["state"] == "hop"
            and t["input"].get("candidate") == case.true_candidate
            and t["status"] in ("waiting", "done")
        ]
        pending_true_hops = [
            t
            for t in snap["tasks"].values()
            if t["state"] == "hop"
            and t["input"].get("candidate") == case.true_candidate
            and t["status"] == "pending"
            and int(t["input"].get("hop", 0)) >= 4
        ]
        done_tasks = [t for t in snap["tasks"].values() if t["status"] == "done"]
        if len(progressed_true_hops) >= 4 and pending_true_hops and done_tasks:
            shutil.copytree(full_run_dir, crash_dir)
            # Simulate mid-wave SIGKILL where hop >= 4 was actively "running".
            crashed_snap = json.loads((crash_dir / "state.json").read_text(encoding="utf-8"))
            crashed_tid = pending_true_hops[0]["id"]
            crashed_snap["tasks"][crashed_tid]["status"] = "running"
            crashed_snap["tasks"][crashed_tid]["attempts"] = 1
            CheckpointStore(crash_dir).save_state(crashed_snap)
            snapshot_captured = True

    budget = RunBudget(max_tasks=500, max_depth=32, max_inline_bytes=600)
    first_result = machine1.run(
        TaskSpec(state="start"),
        run_dir=full_run_dir,
        max_concurrent=4,
        budget=budget,
        on_event=on_event,
    )
    assert first_result.status == "done"
    assert snapshot_captured is True

    # Count tasks already "done" or "waiting" in the crashed snapshot.
    pre_snap = CheckpointStore(crash_dir).load_state()
    assert pre_snap is not None
    pre_completed_attempts = {
        tid: t["attempts"]
        for tid, t in pre_snap["tasks"].items()
        if t["status"] in ("done", "waiting")
    }
    pre_done_ids = [tid for tid, t in pre_snap["tasks"].items() if t["status"] == "done"]
    assert len(pre_done_ids) >= 1
    assert len(pre_completed_attempts) >= 5

    resume_llm = InstrumentedLLMClient(MockSmallContextLLM(), prompt_char_cap=2000)
    machine2 = build_machine(case, llm=resume_llm, prompt_char_cap=2000, max_inline_bytes=600)
    resumed = machine2.resume(crash_dir, max_concurrent=4, budget=budget)

    assert resumed.status == "done"
    assert resumed.output["selected"] == case.true_candidate

    # Already-done and already-waiting hops were NOT re-executed on resume.
    for tid, prior_attempts in pre_completed_attempts.items():
        assert resumed.tasks[tid].attempts == prior_attempts
    # Remaining hops were fewer than a full fresh run (8 + decoy + early fails).
    full_calls = machine1.telemetry.call_count  # type: ignore[attr-defined]
    assert 1 <= resume_llm.telemetry.call_count < full_calls
    assert resume_llm.telemetry.peak_prompt_chars < 2000


def test_cli_entrypoint_smoke(capsys: Any) -> None:
    rc = cli_main(["--mock", "--horizons", "4,8,12", "--context-cap", "2500"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "ARES Long-Horizon Small-Context Benchmark" in captured.out
    assert "OVERFLOW" in captured.out


def test_openai_compat_client_wiring_offline() -> None:
    case = generate_multihop_case(hops=4, seed=42, noise_lines=25)
    mock_inner = MockSmallContextLLM()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        msgs = [LLMMessage(role=m["role"], content=m["content"]) for m in body["messages"]]
        resp = mock_inner.complete(msgs, response_model=HopExtraction)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": resp.text}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    client = OpenAICompatClient(
        base_url="http://mock.local/v1",
        model="small-model",
        transport=httpx.MockTransport(handler),
    )
    res = run_ares(case, prompt_char_cap=2000, llm=client)
    assert res.correct is True
    assert res.selected == case.true_candidate
    assert res.peak_prompt_chars < 2000

    live_wrapped = create_llm_client(mock=False, base_url="http://mock.local/v1", model="m")
    assert isinstance(live_wrapped.inner, OpenAICompatClient)

