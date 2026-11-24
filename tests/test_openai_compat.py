"""Offline unit and integration tests for `ares/models/openai_compat.py` and `ModelProvider`."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from ares.contracts import (
    ArtifactRef,
    BoardConclusion,
    InvestigationBoard,
    Note,
    SearchQuestion,
)
from ares.engine import TaskSpec, solve
from ares.loop import LoopConfig, run_investigation_loop
from ares.models import (
    ContextLengthExceededError,
    GeminiModelProvider,
    ModelProvider,
    OpenAICompatibleProvider,
)
from examples.codebase_audit.domain import CodebaseConcurrencyDomain

_TINY_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _chat_response(
    *,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 30,
    reasoning_tokens: int = 0,
) -> httpx.Response:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens + reasoning_tokens,
    }
    if reasoning_tokens:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": usage,
        },
    )


def test_provider_protocol_conformance() -> None:
    transport = httpx.MockTransport(lambda req: _chat_response(content="{}"))
    with OpenAICompatibleProvider(transport=transport) as oai:
        assert isinstance(oai, ModelProvider)
    gemini = GeminiModelProvider(client=object())
    assert isinstance(gemini, ModelProvider)


def test_openai_compat_propose_processor_and_auth_headers() -> None:
    requests_seen: list[tuple[dict[str, str], dict[str, Any], str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests_seen.append((dict(request.headers), body, str(request.url)))
        payload = json.dumps(
            {
                "hypotheses": [
                    {
                        "query": "inspect:reachable;",
                        "params_json": "{}",
                        "hypothesis_family": "entry_reachability",
                        "expected_signal": "Reachable modules",
                        "target_fields": ["public_entry_reachable"],
                        "verifier_name": "verify_reachability",
                    },
                    {
                        "query": "inspect:lock_cycle",
                        "params_json": "{}",
                        "hypothesis_family": "entry_reachability",
                        "expected_signal": "Invalid verifier stripped",
                        "target_fields": [],
                        "verifier_name": "unknown_verifier",
                    },
                ]
            }
        )
        return _chat_response(content=payload, prompt_tokens=145, completion_tokens=55)

    transport = httpx.MockTransport(_handler)
    with OpenAICompatibleProvider(
        model="qwen2.5:3b",
        base_url="http://localhost:11434/v1/",
        transport=transport,
    ) as provider:
        proc = provider.make_propose_processor(
            environment_description="CodeGraph environment",
            magnet_rules="Emit inspect:<prop>",
            pivotal_fields=["public_entry_reachable"],
            available_verifiers=["verify_reachability"],
        )
        report = solve(
            tasks=[
                TaskSpec(
                    task_id="p_reach",
                    task_type="propose",
                    requested_under=2,
                    context={"hypothesis_family": "entry_reachability", "max_queries": 2},
                )
            ],
            processors={"propose": proc},
        )

    assert len(requests_seen) == 1
    headers, req_body, req_url = requests_seen[0]
    assert req_url == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in {k.lower() for k in headers}
    assert req_body["model"] == "qwen2.5:3b"
    assert req_body["response_format"] == {"type": "json_object"}

    assert report.errors == {}
    assert report.usage.model_calls == 1
    assert report.usage.total_prompt_tokens == 145
    assert report.usage.peak_prompt_tokens == 145

    specs = report.graph.nodes["p_reach"].output
    assert len(specs) == 2
    assert specs[0].query == "inspect:reachable"
    assert specs[0].verifier_name == "verify_reachability"
    assert specs[0].requested_under == 2
    assert specs[1].verifier_name is None

    # Verify Bearer header is sent only when api_key is provided
    with OpenAICompatibleProvider(api_key="sk-local-test", transport=transport) as auth_provider:
        auth_provider.propose_hypotheses(
            environment_description="env",
            hypothesis_family="entry_reachability",
        )
    assert requests_seen[-1][0].get("authorization") == "Bearer sk-local-test"


def test_openai_compat_board_patch_text_vs_vision_and_filtering(tmp_path: Path) -> None:
    png_path = tmp_path / "slice.png"
    png_path.write_bytes(_TINY_PNG_BYTES)
    art = ArtifactRef(
        artifact_id="art-png-1",
        kind="plot_png",
        path=str(png_path),
        sha256="abc123",
        mime_type="image/png",
        produced_by_verifier=True,
    )
    note = Note(
        note_id="n_0001",
        certification="CERTIFIED",
        verifier_name="verify_reachability",
        summary="SessionPool reachable",
        evidence={"candidates": ["SessionPool"]},
        artifacts=[art],
        row_count=1,
    )
    board = InvestigationBoard(
        board_version=0,
        conclusion=BoardConclusion(question_id="buggy_module", question="Which module?"),
    )
    seen_bodies: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        seen_bodies.append(json.loads(request.content.decode("utf-8")))
        patch_payload = json.dumps(
            {
                "updates": [
                    {
                        "field_name": "public_entry_reachable",
                        "value_json": json.dumps("SessionPool"),
                        "sources": ["n_0001", "bogus_note"],
                        "artifact_ids": ["art-png-1", "bogus_art"],
                    },
                    {
                        "field_name": "invented_field",
                        "value_json": "bad",
                        "sources": ["bogus_only"],
                        "artifact_ids": [],
                    },
                ],
                "rationale": "Grounded SessionPool from n_0001",
            }
        )
        return _chat_response(content=patch_payload, prompt_tokens=220, completion_tokens=50)

    transport = httpx.MockTransport(_handler)
    with OpenAICompatibleProvider(support_vision=False, transport=transport) as text_only:
        patch_txt, tel_txt = text_only.propose_board_patch(board=board, new_notes=[note])
    assert isinstance(seen_bodies[0]["messages"][1]["content"], str)
    assert patch_txt["updates"] == {
        "public_entry_reachable": {
            "value": "SessionPool",
            "sources": ["n_0001"],
            "artifact_ids": ["art-png-1"],
        }
    }
    assert tel_txt.peak_prompt_tokens == 220

    with OpenAICompatibleProvider(support_vision=True, transport=transport) as vision_on:
        vision_on.propose_board_patch(board=board, new_notes=[note], include_images=True)
    vision_user_content = seen_bodies[1]["messages"][1]["content"]
    assert isinstance(vision_user_content, list)
    assert vision_user_content[0]["type"] == "text"
    expected_url = f"data:image/png;base64,{base64.b64encode(_TINY_PNG_BYTES).decode('ascii')}"
    assert vision_user_content[1] == {
        "type": "image_url",
        "image_url": {"url": expected_url},
    }


def test_openai_compat_followup_renders_attempted_queries() -> None:
    captured_prompts: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured_prompts.append(str(body["messages"][1]["content"]))
        return _chat_response(
            content=json.dumps(
                {
                    "hypotheses": [
                        {
                            "query": "inspect:race_repro",
                            "params_json": "{}",
                            "hypothesis_family": "stress_repro",
                            "expected_signal": "Deterministic race repro",
                            "target_fields": ["deterministic_race_repro"],
                            "verifier_name": None,
                        }
                    ]
                }
            ),
            prompt_tokens=175,
        )

    board = InvestigationBoard(
        board_version=2,
        conclusion=BoardConclusion(question_id="buggy_module", question="Which module?"),
    )
    with OpenAICompatibleProvider(transport=httpx.MockTransport(_handler)) as provider:
        specs, tel = provider.propose_followup_tasks(
            board=board,
            target_gap_fields=["deterministic_race_repro"],
            environment_description="CodeGraph",
            attempted_queries=["inspect:reachable", "inspect:lock_cycle"],
        )

    assert len(specs) == 1
    assert specs[0].requested_under == 2
    assert tel.peak_prompt_tokens == 175
    assert "### Already Attempted Queries (Do Not Repeat)" in captured_prompts[0]
    assert "inspect:reachable" in captured_prompts[0]
    assert "inspect:lock_cycle" in captured_prompts[0]


def test_openai_compat_json_repair_and_schema_validation_retry() -> None:
    calls: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            # Bad schema / non-JSON on first attempt
            return _chat_response(
                content="I think you should run {\"wrong_key\": [1, 2]}",
                prompt_tokens=120,
                completion_tokens=20,
            )
        # Second attempt has prose wrapper, markdown fence, and trailing commas
        repaired_content = (
            "Sure, here is the JSON:\n"
            "```json\n"
            "{\n"
            '  "hypotheses": [\n'
            "    {\n"
            '      "query": "inspect:race_repro;",\n'
            '      "params_json": "{}",\n'
            '      "hypothesis_family": "stress_repro",\n'
            '      "expected_signal": "check race",\n'
            '      "target_fields": ["deterministic_race_repro",],\n'
            '      "verifier_name": null,\n'
            "    },\n"
            "  ],\n"
            "}\n"
            "```\n"
            "Let me know if you need more!"
        )
        return _chat_response(content=repaired_content, prompt_tokens=190, completion_tokens=45)

    with OpenAICompatibleProvider(
        max_retries=2,
        transport=httpx.MockTransport(_handler),
    ) as provider:
        specs, tel = provider.propose_hypotheses(
            environment_description="CodeGraph",
            hypothesis_family="stress_repro",
        )

    assert len(calls) == 2
    assert "failed JSON schema validation" in calls[1]["messages"][-1]["content"]
    assert len(specs) == 1
    assert specs[0].query == "inspect:race_repro"
    assert tel.model_calls == 2
    assert tel.prompt_tokens == 310
    assert tel.peak_prompt_tokens == 190


def test_openai_compat_response_format_fallback_to_prompt_only() -> None:
    seen_formats: list[Any] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_formats.append(body.get("response_format"))
        if body.get("response_format") is not None:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format type json_schema is unsupported"}},
            )
        return _chat_response(
            content=json.dumps(
                {
                    "hypotheses": [
                        {
                            "query": "inspect:reachable",
                            "params_json": "{}",
                            "hypothesis_family": "entry_reachability",
                            "expected_signal": "check reachable",
                        }
                    ]
                }
            ),
            prompt_tokens=95,
        )

    with OpenAICompatibleProvider(
        structured_output_mode="json_schema",
        transport=httpx.MockTransport(_handler),
    ) as provider:
        specs, tel = provider.propose_hypotheses(
            environment_description="CodeGraph",
            hypothesis_family="entry_reachability",
        )

    assert len(seen_formats) == 2
    assert seen_formats[0]["type"] == "json_schema"
    assert seen_formats[1] is None
    assert len(specs) == 1
    assert tel.model_calls == 1
    assert tel.peak_prompt_tokens == 95


def test_openai_compat_context_length_error_fails_loud_without_retry() -> None:
    call_count = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        del request
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            400,
            json={"error": {"message": "This model's maximum context length is 4096 tokens."}},
        )

    with OpenAICompatibleProvider(
        max_retries=3,
        transport=httpx.MockTransport(_handler),
    ) as provider:
        with pytest.raises(ContextLengthExceededError, match="Context window exceeded"):
            provider.propose_hypotheses(
                environment_description="CodeGraph",
                hypothesis_family="entry_reachability",
            )

    assert call_count == 1


def test_openai_compat_single_agent_baseline_sliding_window_and_overflow() -> None:
    seen_histories: list[list[dict[str, Any]]] = []
    turn_idx = 0

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal turn_idx
        turn_idx += 1
        body = json.loads(request.content.decode("utf-8"))
        msgs = body["messages"]
        seen_histories.append(msgs)
        # Guard against orphaned `role == "tool"` messages after trimming
        if len(msgs) > 2:
            assert msgs[2]["role"] != "tool"

        if turn_idx <= 3:
            return _chat_response(
                tool_calls=[
                    {
                        "id": f"call_{turn_idx}",
                        "type": "function",
                        "function": {
                            "name": "execute_readonly_query",
                            "arguments": json.dumps({"query": f"inspect:hop_{turn_idx}"}),
                        },
                    }
                ],
                prompt_tokens=110 + turn_idx * 15,
                completion_tokens=20,
            )

        return _chat_response(
            content=json.dumps(
                {
                    "answers": [
                        {"question_id": "buggy_module", "answer": "SessionPool"},
                        {"question_id": "sufficiently_verified", "answer": "yes"},
                    ],
                    "confidence": 0.91,
                    "rationale": "Checked 3 hops.",
                }
            ),
            prompt_tokens=160,
            completion_tokens=30,
        )

    questions = CodebaseConcurrencyDomain().search_questions()
    queries_run: list[str] = []

    def _exec(q: str) -> dict[str, Any]:
        queries_run.append(q)
        return {"query": q, "hits": ["SessionPool"]}

    with OpenAICompatibleProvider(transport=httpx.MockTransport(_handler)) as provider:
        res = provider.run_single_agent_baseline(
            environment_description="CodeGraph",
            search_questions=questions,
            query_executor=_exec,
            max_queries=5,
            max_history_messages=4,
        )

    assert len(queries_run) == 3
    assert res["context_overflow"] is False
    assert res["dropped_history_messages"] > 0
    assert res["answers"]["buggy_module"] == "SessionPool"
    assert res["answers"]["sufficiently_verified"] == "yes"
    assert res["confidence"] == pytest.approx(0.91)
    assert res["peak_prompt_tokens"] == 160
    assert res["telemetry"].model_calls == 4

    # Context overflow during baseline returns graceful report instead of crashing
    def _overflow_handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(413, json={"error": {"message": "prompt is too long for context"}})

    with OpenAICompatibleProvider(transport=httpx.MockTransport(_overflow_handler)) as ov_provider:
        ov_res = ov_provider.run_single_agent_baseline(
            environment_description="CodeGraph",
            search_questions=questions,
            query_executor=_exec,
            max_queries=4,
        )

    assert ov_res["context_overflow"] is True
    assert ov_res["confidence"] == 0.0
    assert "ContextLengthExceededError" in ov_res["rationale"]


def test_openai_compat_runs_full_ares_loop_on_codebase_audit(tmp_path: Path) -> None:
    domain = CodebaseConcurrencyDomain()
    ws = tmp_path / "code_oai_ws"
    domain.generate(str(ws))
    truth = domain.plant(str(ws), domain.needle_menu()[0])

    prop_map = {
        "entry_reachability": ("inspect:reachable", "verify_reachability"),
        "lock_graph_cycles": ("inspect:lock_cycle", "verify_lock_inversion"),
        "stress_repro": ("inspect:race_repro", "verify_race_repro"),
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        user_text = str(body["messages"][1]["content"])

        if "### Assigned Hypothesis Family" in user_text:
            m = re.search(r"### Assigned Hypothesis Family\n([^\n]+)", user_text)
            fam = m.group(1).strip() if m else "entry_reachability"
            q_str, v_name = prop_map.get(fam, ("inspect:reachable", "verify_reachability"))
            return _chat_response(
                content=json.dumps(
                    {
                        "hypotheses": [
                            {
                                "query": q_str,
                                "params_json": "{}",
                                "hypothesis_family": fam,
                                "expected_signal": f"Verify {fam}",
                                "target_fields": [],
                                "verifier_name": v_name,
                            }
                        ]
                    }
                ),
                prompt_tokens=180,
                completion_tokens=40,
            )

        if "### New Immutable Evidence Notes" in user_text:
            # Ground an exploratory hint strictly in a real note ID present in the prompt
            note_matches = re.findall(r'"note_id":\s*"([^"]+)"', user_text)
            updates = []
            if note_matches and "SessionPool" in user_text:
                updates.append(
                    {
                        "field_name": "exploratory_candidate",
                        "value_json": json.dumps("SessionPool"),
                        "sources": [note_matches[-1]],
                        "artifact_ids": [],
                    }
                )
            return _chat_response(
                content=json.dumps(
                    {"updates": updates, "rationale": "Grounded patch from OpenAI-compat model."}
                ),
                prompt_tokens=260,
                completion_tokens=45,
            )

        return _chat_response(
            content=json.dumps({"hypotheses": []}),
            prompt_tokens=150,
            completion_tokens=20,
        )

    with OpenAICompatibleProvider(
        model="qwen2.5:3b",
        transport=httpx.MockTransport(_handler),
    ) as provider:
        res = run_investigation_loop(
            domain,
            str(ws),
            provider=provider,
            config=LoopConfig(max_rounds=3, max_concurrent_workers=3),
        )

    assert res.termination_reason == "certified_success"
    assert res.board.conclusion.answer == "SessionPool"
    assert res.final_board_drift == 0.0
    assert res.ungrounded_fields_on_board == 0
    assert res.peak_prompt_tokens == 260
    assert res.rounds[0].peak_prompt_tokens == 260
    assert domain.score(res.board.model_dump(), truth)["correct"] is True
