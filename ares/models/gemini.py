"""Backend- and domain-agnostic Gemini 4 Interactions API provider (`gemini-4-hydrogen` and `gemini-4-carbon`).

Implements `ares.models.base.ModelProvider`:
- Parallel hypothesis magnet proposal (`propose_hypotheses` + stateless ARES `make_propose_processor`)
- Multimodal context patch synthesis (`propose_board_patch`, attaching up to 3 verifier PNG slices)
- Targeted follow-up task proposal (`propose_followup_tasks`) for unresolved board gaps
- Single-agent `gemini-4-carbon` read-only baseline (`run_single_agent_baseline`)
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from ares.contracts import (
    HypothesisSpec,
    InvestigationBoard,
    Note,
    SearchQuestion,
)
from ares.engine import ProcessorFn
from ares.models.base import (
    BaselineAnswerEntry,
    BaselineFinalEnvelope,
    BoardPatchEnvelope,
    ContextLengthExceededError,
    FieldUpdateItem,
    HypothesisProposalEnvelope,
    ModelTelemetry,
    ProposedHypothesisItem,
    build_board_patch_prompts,
    build_followup_prompts,
    build_propose_prompts,
    default_make_propose_processor,
    filter_board_patch_updates,
    parse_hypothesis_items,
    sanitize_query,
    strip_json_fences,
)

VALID_THINKING_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})

# Backward-compatible private aliases (e.g. `examples/nyc_cameras/domain.py`)
_strip_json_fences = strip_json_fences
_ProposedHypothesisItem = ProposedHypothesisItem
_HypothesisProposalEnvelope = HypothesisProposalEnvelope
_FieldUpdateItem = FieldUpdateItem
_BoardPatchEnvelope = BoardPatchEnvelope
_BaselineAnswerEntry = BaselineAnswerEntry
_BaselineFinalEnvelope = BaselineFinalEnvelope


def _normalize_thinking_level(level: str) -> str:
    """Normalize thinking_level for Gemini 4 ('low', 'medium', 'high'; never 'minimal')."""
    cleaned = (level or "low").strip().lower()
    if cleaned == "minimal" or cleaned not in VALID_THINKING_LEVELS:
        return "low"
    return cleaned


def _get_usage_int(usage: Any, *attr_names: str) -> int:
    if usage is None:
        return 0
    for name in attr_names:
        val = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if val is not None:
            try:
                return max(0, int(val))
            except (TypeError, ValueError):
                continue
    return 0


def extract_telemetry(resp: Any, latency_ms: float, model_calls: int = 1) -> ModelTelemetry:
    """Extract defensive ModelTelemetry from an Interactions API response."""
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    prompt_toks = _get_usage_int(
        usage, "total_input_tokens", "prompt_token_count", "input_tokens"
    )
    completion_toks = _get_usage_int(
        usage, "total_output_tokens", "candidates_token_count", "output_tokens"
    )
    thought_toks = _get_usage_int(
        usage, "total_thought_tokens", "thoughts_token_count", "thought_tokens"
    )
    total_toks = _get_usage_int(usage, "total_tokens", "total_token_count")
    if total_toks == 0 and (prompt_toks + completion_toks + thought_toks) > 0:
        total_toks = prompt_toks + completion_toks + thought_toks
    return ModelTelemetry(
        prompt_tokens=prompt_toks,
        completion_tokens=completion_toks,
        thought_tokens=thought_toks,
        total_tokens=total_toks,
        peak_prompt_tokens=prompt_toks,
        model_calls=model_calls,
        latency_ms=round(max(0.0, latency_ms), 3),
    )


class GeminiModelProvider:
    """Interactions API (`v1`) wrapper for `gemini-4-hydrogen` and `gemini-4-carbon`."""

    def __init__(
        self,
        *,
        hydrogen_model: str = "gemini-4-hydrogen",
        carbon_model: str = "gemini-4-carbon",
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        self.hydrogen_model = hydrogen_model
        self.carbon_model = carbon_model
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            from google import genai

            key = self._api_key or os.environ.get("GEMINI_API_KEY")
            self._client = genai.Client(api_key=key) if key else genai.Client()
        return self._client

    def propose_hypotheses(
        self,
        *,
        environment_description: str,
        hypothesis_family: str,
        magnet_rules: str = "",
        pivotal_fields: Sequence[str] = (),
        board: InvestigationBoard | None = None,
        available_verifiers: Sequence[str] | None = None,
        requested_under: int = 0,
        max_queries: int = 2,
        thinking_level: str = "low",
        board_summary: str | None = None,
    ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
        """Use `gemini-4-hydrogen` to propose structured HypothesisSpecs for a family."""
        cap = max(1, int(max_queries))
        allowed_verifiers = list(available_verifiers or ())
        effective_requested_under = (
            board.board_version if (board is not None and requested_under == 0) else requested_under
        )
        sys_instruction, prompt = build_propose_prompts(
            environment_description=environment_description,
            hypothesis_family=hypothesis_family,
            magnet_rules=magnet_rules,
            pivotal_fields=pivotal_fields,
            board=board,
            available_verifiers=allowed_verifiers,
            max_queries=cap,
            board_summary=board_summary,
            role_label="parallel hypothesis worker (`gemini-4-hydrogen`)",
        )

        client = self._get_client()
        t0 = time.perf_counter()
        interaction = client.interactions.create(
            model=self.hydrogen_model,
            store=False,
            input=prompt,
            system_instruction=sys_instruction,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": HypothesisProposalEnvelope.model_json_schema(),
            },
            generation_config={"thinking_level": _normalize_thinking_level(thinking_level)},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = extract_telemetry(interaction, latency_ms)

        raw_text = strip_json_fences(
            getattr(interaction, "output_text", "") or '{"hypotheses": []}'
        )
        envelope = HypothesisProposalEnvelope.model_validate_json(raw_text)
        specs = parse_hypothesis_items(
            envelope.hypotheses,
            cap=cap,
            default_family=hypothesis_family,
            default_targets=(),
            allowed_verifiers=allowed_verifiers,
            requested_under=effective_requested_under,
        )
        return specs, telemetry

    def make_propose_processor(
        self,
        *,
        environment_description: str,
        magnet_rules: str = "",
        pivotal_fields: Sequence[str] = (),
        available_verifiers: Sequence[str] | None = None,
    ) -> ProcessorFn:
        """Build a stateless ARES ProcessorFn wrapping `propose_hypotheses`."""
        return default_make_propose_processor(
            self,
            environment_description=environment_description,
            magnet_rules=magnet_rules,
            pivotal_fields=pivotal_fields,
            available_verifiers=available_verifiers,
        )

    def propose_board_patch(
        self,
        *,
        board: InvestigationBoard,
        new_notes: list[Note],
        environment_description: str = "",
        pivotal_fields: Sequence[str] = (),
        include_images: bool = True,
        thinking_level: str = "medium",
    ) -> tuple[dict[str, Any], ModelTelemetry]:
        """Use `gemini-4-carbon` (multimodal) to synthesize a grounded board patch dict."""
        sys_instruction, prompt_str, valid_note_ids, valid_artifact_ids, png_artifacts = (
            build_board_patch_prompts(
                board=board,
                new_notes=new_notes,
                environment_description=environment_description,
                pivotal_fields=pivotal_fields,
                include_images=include_images,
                role_label="ARES Context Synthesis Engine (`gemini-4-carbon`)",
            )
        )

        if png_artifacts:
            multimodal_input: Any = [{"type": "text", "text": prompt_str}]
            for art in png_artifacts:
                b64_data = base64.b64encode(Path(art.path).read_bytes()).decode("ascii")
                multimodal_input.append(
                    {
                        "type": "image",
                        "data": b64_data,
                        "mime_type": art.mime_type or "image/png",
                    }
                )
        else:
            multimodal_input = prompt_str

        client = self._get_client()
        t0 = time.perf_counter()
        interaction = client.interactions.create(
            model=self.carbon_model,
            store=False,
            input=multimodal_input,
            system_instruction=sys_instruction,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": BoardPatchEnvelope.model_json_schema(),
            },
            generation_config={"thinking_level": _normalize_thinking_level(thinking_level)},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = extract_telemetry(interaction, latency_ms)

        raw_text = strip_json_fences(
            getattr(interaction, "output_text", "") or '{"updates": []}'
        )
        envelope = BoardPatchEnvelope.model_validate_json(raw_text)
        patch = filter_board_patch_updates(envelope, valid_note_ids, valid_artifact_ids)
        return patch, telemetry

    def propose_followup_tasks(
        self,
        *,
        board: InvestigationBoard,
        target_gap_fields: list[str],
        environment_description: str,
        magnet_rules: str = "",
        available_verifiers: Sequence[str] | None = None,
        max_queries: int = 3,
        thinking_level: str = "low",
        attempted_queries: Sequence[str] = (),
    ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
        """Use `gemini-4-carbon` to propose targeted follow-up magnets/verifiers for unresolved fields."""
        cap = max(1, int(max_queries))
        allowed_verifiers = list(available_verifiers or ())
        sys_instruction, prompt = build_followup_prompts(
            board=board,
            target_gap_fields=target_gap_fields,
            environment_description=environment_description,
            magnet_rules=magnet_rules,
            available_verifiers=allowed_verifiers,
            max_queries=cap,
            attempted_queries=attempted_queries,
            role_label="ARES Follow-Up Search Planner (`gemini-4-carbon`)",
        )

        client = self._get_client()
        t0 = time.perf_counter()
        interaction = client.interactions.create(
            model=self.carbon_model,
            store=False,
            input=prompt,
            system_instruction=sys_instruction,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": HypothesisProposalEnvelope.model_json_schema(),
            },
            generation_config={"thinking_level": _normalize_thinking_level(thinking_level)},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = extract_telemetry(interaction, latency_ms)

        raw_text = strip_json_fences(
            getattr(interaction, "output_text", "") or '{"hypotheses": []}'
        )
        envelope = HypothesisProposalEnvelope.model_validate_json(raw_text)
        specs = parse_hypothesis_items(
            envelope.hypotheses,
            cap=cap,
            default_family="followup_gap_resolution",
            default_targets=target_gap_fields,
            allowed_verifiers=allowed_verifiers,
            requested_under=board.board_version,
        )
        return specs, telemetry

    def run_single_agent_baseline(
        self,
        *,
        environment_description: str,
        search_questions: Sequence[SearchQuestion],
        query_executor: Callable[[str], dict[str, Any]],
        magnet_rules: str = "",
        tool_name: str = "execute_readonly_query",
        tool_description: str = (
            "Execute a read-only magnet query against the domain environment and return structured results."
        ),
        max_queries: int = 12,
        timeout_seconds: float = 180.0,
        thinking_level: str = "medium",
    ) -> dict[str, Any]:
        """Run an honest single-agent `gemini-4-carbon` baseline using identical read-only query tools."""
        query_tool = {
            "type": "function",
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Single read-only query/expression obeying the domain magnet rules.",
                    }
                },
                "required": ["query"],
            },
        }

        questions_desc = [q.model_dump() for q in search_questions]
        rules_block = f"\n{magnet_rules}\n" if magnet_rules else ""
        sys_instruction = (
            f"You are an expert investigator with access to `{tool_name}`.{rules_block}"
            f"You may execute at most {max_queries} queries. "
            "Verify all required independent signals before selecting a confident target; if any "
            "required corroborating signal is missing, select the abstention/unverified option. "
            "When finished, respond with a JSON object matching `{answers: [{question_id, answer}], "
            "confidence, rationale}`."
        )
        initial_prompt = (
            f"### Domain Environment\n{environment_description}\n\n"
            f"### Search Questions\n{json.dumps(questions_desc, indent=2)}\n\n"
            f"Investigate using `{tool_name}` and output your final JSON answer."
        )

        history: list[dict[str, Any]] = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": initial_prompt}],
            }
        ]
        queries_executed: list[str] = []
        total_telemetry = ModelTelemetry()
        client = self._get_client()
        norm_thinking = _normalize_thinking_level(thinking_level)
        deadline = time.perf_counter() + float(timeout_seconds)
        final_text = ""

        max_turns = max(2, int(max_queries) + 3)
        for _turn in range(max_turns):
            if time.perf_counter() >= deadline:
                break

            t0 = time.perf_counter()
            interaction = client.interactions.create(
                model=self.carbon_model,
                store=False,
                input=history,
                system_instruction=sys_instruction,
                tools=[query_tool],
                generation_config={"thinking_level": norm_thinking},
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            total_telemetry = total_telemetry.plus(extract_telemetry(interaction, latency_ms))

            steps = getattr(interaction, "steps", None) or []
            fc_steps: list[Any] = []
            for step in steps:
                dumped = step.model_dump() if hasattr(step, "model_dump") else dict(step)
                history.append(dumped)
                if dumped.get("type") == "function_call":
                    fc_steps.append(step)

            if not fc_steps:
                final_text = getattr(interaction, "output_text", "") or ""
                break

            budget_exhausted = False
            for fc in fc_steps:
                fc_name = fc.get("name") if isinstance(fc, dict) else getattr(fc, "name", "")
                fc_id = fc.get("id") if isinstance(fc, dict) else getattr(fc, "id", "")
                raw_args = (
                    fc.get("arguments") if isinstance(fc, dict) else getattr(fc, "arguments", {})
                )
                if isinstance(raw_args, str):
                    try:
                        args_dict = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args_dict = {}
                else:
                    args_dict = dict(raw_args or {})

                if fc_name != tool_name:
                    res_payload: dict[str, Any] = {"error": f"Unsupported tool: {fc_name}"}
                elif len(queries_executed) >= max_queries or time.perf_counter() >= deadline:
                    budget_exhausted = True
                    res_payload = {
                        "error": (
                            f"Query limit ({max_queries}) or timeout reached. "
                            "Do not call any more tools; output your final JSON answer now."
                        )
                    }
                else:
                    q_str = sanitize_query(
                        str(args_dict.get("query", args_dict.get("sql", "")))
                    )
                    queries_executed.append(q_str)
                    try:
                        res_payload = query_executor(q_str)
                    except Exception as exc:
                        res_payload = {"error": str(exc)}

                history.append(
                    {
                        "type": "function_result",
                        "name": fc_name,
                        "call_id": fc_id,
                        "result": [
                            {"type": "text", "text": json.dumps(res_payload, default=str)}
                        ],
                    }
                )

            if budget_exhausted or len(queries_executed) >= max_queries:
                history.append(
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Query budget is exhausted. Provide your final JSON "
                                    "conclusion now without calling tools."
                                ),
                            }
                        ],
                    }
                )
                t0 = time.perf_counter()
                synth = client.interactions.create(
                    model=self.carbon_model,
                    store=False,
                    input=history,
                    system_instruction=sys_instruction,
                    tools=[query_tool],
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": BaselineFinalEnvelope.model_json_schema(),
                    },
                    generation_config={"thinking_level": norm_thinking},
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                total_telemetry = total_telemetry.plus(extract_telemetry(synth, latency_ms))
                final_text = getattr(synth, "output_text", "") or ""
                break

        answers_by_qid: dict[str, str] = {
            q.question_id: (q.options[-1] if q.options else "unknown")
            for q in search_questions
        }
        conf_val = 0.0
        rationale_val = ""

        cleaned_final = strip_json_fences(final_text)
        if cleaned_final:
            try:
                raw_json = json.loads(cleaned_final)
                if isinstance(raw_json, dict):
                    if isinstance(raw_json.get("answers"), list):
                        env = BaselineFinalEnvelope.model_validate(raw_json)
                        for entry in env.answers:
                            if entry.question_id in answers_by_qid:
                                answers_by_qid[entry.question_id] = entry.answer
                        conf_val = float(env.confidence)
                        rationale_val = env.rationale
                    else:
                        for q in search_questions:
                            if q.question_id in raw_json:
                                answers_by_qid[q.question_id] = str(raw_json[q.question_id])
                        conf_val = float(raw_json.get("confidence", 0.0))
                        rationale_val = str(raw_json.get("rationale", ""))
            except Exception:
                rationale_val = cleaned_final[:500]

        return {
            "answers": answers_by_qid,
            "confidence": max(0.0, min(1.0, conf_val)),
            "rationale": rationale_val,
            "query_count": len(queries_executed),
            "queries_executed": queries_executed,
            "telemetry": total_telemetry,
            "peak_prompt_tokens": total_telemetry.peak_prompt_tokens,
            "context_overflow": False,
        }


__all__ = [
    "ContextLengthExceededError",
    "GeminiModelProvider",
    "ModelTelemetry",
    "extract_telemetry",
]
