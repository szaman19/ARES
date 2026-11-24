"""OpenAI-compatible ModelProvider (Ollama, vLLM, llama.cpp, LM Studio, or compatible endpoints).

Uses `httpx` directly against `/v1/chat/completions` with zero additional dependencies.
"""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ares.contracts import (
    HypothesisSpec,
    InvestigationBoard,
    Note,
    SearchQuestion,
)
from ares.engine import ProcessorFn
from ares.models.base import (
    BaselineFinalEnvelope,
    BoardPatchEnvelope,
    ContextLengthExceededError,
    HypothesisProposalEnvelope,
    ModelTelemetry,
    build_board_patch_prompts,
    build_followup_prompts,
    build_propose_prompts,
    default_make_propose_processor,
    extract_json_object,
    filter_board_patch_updates,
    parse_hypothesis_items,
    sanitize_query,
)

TModel = TypeVar("TModel", bound=BaseModel)

_CONTEXT_ERROR_MARKERS: tuple[str, ...] = (
    "context length",
    "maximum context",
    "too many tokens",
    "token limit",
    "prompt is too long",
    "exceeds the available context",
)

_RESPONSE_FORMAT_ERROR_MARKERS: tuple[str, ...] = (
    "response_format",
    "json_schema",
    "unsupported",
)


def _extract_error_message(resp: httpx.Response) -> str:
    raw = resp.text or ""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])
            if err:
                return str(err)
            if data.get("detail"):
                return str(data["detail"])
    except Exception:
        pass
    return raw


def _is_context_length_error(status_code: int, message: str) -> bool:
    if status_code not in (400, 413, 422):
        return False
    lower = message.lower()
    return any(marker in lower for marker in _CONTEXT_ERROR_MARKERS)


def _is_response_format_error(status_code: int, message: str) -> bool:
    if status_code not in (400, 422):
        return False
    lower = message.lower()
    return any(marker in lower for marker in _RESPONSE_FORMAT_ERROR_MARKERS)


def _extract_chat_telemetry(payload: dict[str, Any], latency_ms: float) -> ModelTelemetry:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    prompt_toks = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    completion_toks = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    details = usage.get("completion_tokens_details")
    thought_toks = (
        int(details.get("reasoning_tokens", 0) or 0)
        if isinstance(details, dict)
        else 0
    )
    total_toks = int(usage.get("total_tokens", 0) or 0)
    if total_toks == 0 and (prompt_toks + completion_toks + thought_toks) > 0:
        total_toks = prompt_toks + completion_toks + thought_toks
    return ModelTelemetry(
        prompt_tokens=max(0, prompt_toks),
        completion_tokens=max(0, completion_toks),
        thought_tokens=max(0, thought_toks),
        total_tokens=max(0, total_toks),
        peak_prompt_tokens=max(0, prompt_toks),
        model_calls=1,
        latency_ms=round(max(0.0, latency_ms), 3),
    )


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _trim_chat_history(
    messages: list[dict[str, Any]],
    max_history_messages: int,
) -> tuple[list[dict[str, Any]], int]:
    """Slide out oldest intermediate messages while preserving [system, initial_user]."""
    if max_history_messages < 2 or len(messages) <= max_history_messages:
        return list(messages), 0

    head = list(messages[:2])
    tail_budget = max(0, max_history_messages - 2)
    rest = list(messages[2:])
    if tail_budget == 0:
        return head, len(rest)

    kept_start = max(0, len(rest) - tail_budget)
    # Avoid leaving an orphaned `role="tool"` result without its preceding assistant `tool_calls`.
    while kept_start < len(rest) and rest[kept_start].get("role") == "tool":
        kept_start += 1

    trimmed = head + rest[kept_start:]
    return trimmed, kept_start


class OpenAICompatibleProvider:
    """OpenAI-compatible `/v1/chat/completions` provider for small local or hosted models."""

    def __init__(
        self,
        *,
        model: str = "qwen2.5:3b",
        worker_model: str | None = None,
        synthesizer_model: str | None = None,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        timeout_s: float = 120.0,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        structured_output_mode: Literal["json_schema", "json_object", "prompt_only"] = "json_object",
        max_retries: int = 2,
        support_vision: bool = False,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.worker_model = worker_model or model
        self.synthesizer_model = synthesizer_model or model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured_output_mode = structured_output_mode
        self.max_retries = max(0, int(max_retries))
        self.support_vision = support_vision

        self._owns_client = http_client is None
        if http_client is not None:
            self._client = http_client
        else:
            client_kwargs: dict[str, Any] = {"timeout": timeout_s}
            if transport is not None:
                client_kwargs["transport"] = transport
            self._client = httpx.Client(**client_kwargs)

    def close(self) -> None:
        """Close the internal HTTP client if owned by this provider."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleProvider:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb
        self.close()

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat_completion(
        self,
        *,
        model_name: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], ModelTelemetry]:
        url = f"{self.base_url}/chat/completions"

        def _send(fmt: dict[str, Any] | None) -> tuple[dict[str, Any], ModelTelemetry]:
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
            }
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens
            if fmt is not None:
                payload["response_format"] = fmt
            if tools is not None:
                payload["tools"] = tools

            t0 = time.perf_counter()
            resp = self._client.post(url, json=payload, headers=self._build_headers())
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code >= 400:
                err_msg = _extract_error_message(resp)
                if _is_context_length_error(resp.status_code, err_msg):
                    raise ContextLengthExceededError(
                        f"Context window exceeded for model '{model_name}' (HTTP {resp.status_code}): {err_msg}"
                    )
                if fmt is not None and _is_response_format_error(resp.status_code, err_msg):
                    raise _UnsupportedResponseFormatError(err_msg)
                resp.raise_for_status()

            body = resp.json()
            if not isinstance(body, dict):
                raise ValueError(f"Expected JSON object response from {url}, got {type(body).__name__}.")
            return body, _extract_chat_telemetry(body, latency_ms)

        try:
            return _send(response_format)
        except _UnsupportedResponseFormatError:
            return _send(None)

    def _build_response_format(self, schema_cls: type[TModel]) -> dict[str, Any] | None:
        if self.structured_output_mode == "json_schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "strict": True,
                    "schema": schema_cls.model_json_schema(),
                },
            }
        if self.structured_output_mode == "json_object":
            return {"type": "json_object"}
        return None

    def _structured_call(
        self,
        *,
        schema_cls: type[TModel],
        model_name: str,
        sys_instruction: str,
        user_content: Any,
    ) -> tuple[TModel, ModelTelemetry]:
        schema_json = json.dumps(schema_cls.model_json_schema(), separators=(",", ":"))
        eff_sys = (
            f"{sys_instruction}\n\n"
            "Respond ONLY with a single valid JSON object conforming strictly to this JSON Schema:\n"
            f"{schema_json}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": eff_sys},
            {"role": "user", "content": user_content},
        ]
        fmt = self._build_response_format(schema_cls)
        total_tel = ModelTelemetry()

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            body, tel = self._chat_completion(
                model_name=model_name,
                messages=messages,
                response_format=fmt,
            )
            total_tel = total_tel.plus(tel)

            choices = body.get("choices") or []
            if not choices:
                last_error = ValueError("Chat completion returned no choices.")
                raw_text = ""
            else:
                msg = choices[0].get("message") or {}
                raw_text = _extract_message_text(msg)
                json_candidate = extract_json_object(raw_text)
                try:
                    parsed = schema_cls.model_validate_json(json_candidate)
                    return parsed, total_tel
                except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                    last_error = exc

            if attempt >= self.max_retries:
                break

            messages.append({"role": "assistant", "content": raw_text or "{}"})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed JSON schema validation: {last_error}. "
                        "Reply ONLY with a single valid JSON object matching the required schema."
                    ),
                }
            )

        raise ValueError(
            f"Failed to parse valid {schema_cls.__name__} after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error

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
        """Propose structured HypothesisSpecs for one family using the configured worker model."""
        del thinking_level
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
            role_label=f"parallel hypothesis worker ({self.worker_model})",
        )
        envelope, telemetry = self._structured_call(
            schema_cls=HypothesisProposalEnvelope,
            model_name=self.worker_model,
            sys_instruction=sys_instruction,
            user_content=prompt,
        )
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
        thinking_level: str = "low",
    ) -> tuple[dict[str, Any], ModelTelemetry]:
        """Synthesize a lossless, strictly grounded patch for `InvestigationBoard`."""
        del thinking_level
        sys_instruction, prompt_str, valid_note_ids, valid_artifact_ids, png_artifacts = (
            build_board_patch_prompts(
                board=board,
                new_notes=new_notes,
                environment_description=environment_description,
                pivotal_fields=pivotal_fields,
                include_images=include_images and self.support_vision,
                role_label=f"ARES Context Synthesis Engine ({self.synthesizer_model})",
            )
        )

        if png_artifacts and self.support_vision:
            user_content: Any = [{"type": "text", "text": prompt_str}]
            for art in png_artifacts:
                b64_data = base64.b64encode(Path(art.path).read_bytes()).decode("ascii")
                mime = art.mime_type or "image/png"
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_data}"},
                    }
                )
        else:
            user_content = prompt_str

        envelope, telemetry = self._structured_call(
            schema_cls=BoardPatchEnvelope,
            model_name=self.synthesizer_model,
            sys_instruction=sys_instruction,
            user_content=user_content,
        )
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
        """Propose targeted follow-up HypothesisSpecs for unresolved board fields."""
        del thinking_level
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
            role_label=f"ARES Follow-Up Search Planner ({self.worker_model})",
        )
        envelope, telemetry = self._structured_call(
            schema_cls=HypothesisProposalEnvelope,
            model_name=self.worker_model,
            sys_instruction=sys_instruction,
            user_content=prompt,
        )
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
        max_history_messages: int | None = None,
    ) -> dict[str, Any]:
        """Run a rolling-context single-agent baseline over OpenAI-compatible `tools`."""
        del thinking_level
        query_tool = {
            "type": "function",
            "function": {
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
            {"role": "system", "content": sys_instruction},
            {"role": "user", "content": initial_prompt},
        ]
        queries_executed: list[str] = []
        total_telemetry = ModelTelemetry()
        deadline = time.perf_counter() + float(timeout_seconds)
        final_text = ""
        dropped_history_messages = 0
        context_overflow = False

        max_turns = max(2, int(max_queries) + 3)
        try:
            for _turn in range(max_turns):
                if time.perf_counter() >= deadline:
                    break

                if max_history_messages is not None:
                    trimmed, dropped = _trim_chat_history(history, max_history_messages)
                    history = trimmed
                    dropped_history_messages += dropped

                body, tel = self._chat_completion(
                    model_name=self.synthesizer_model,
                    messages=history,
                    tools=[query_tool],
                )
                total_telemetry = total_telemetry.plus(tel)

                choices = body.get("choices") or []
                if not choices:
                    break
                msg = choices[0].get("message") or {}
                tool_calls = msg.get("tool_calls") or []

                if not tool_calls:
                    final_text = _extract_message_text(msg)
                    break

                history.append(
                    {
                        "role": "assistant",
                        "content": _extract_message_text(msg),
                        "tool_calls": tool_calls,
                    }
                )

                budget_exhausted = False
                for idx, tc in enumerate(tool_calls):
                    tc_id = str(tc.get("id", f"call_{idx}"))
                    fn_block = tc.get("function") or {}
                    fn_name = str(fn_block.get("name", ""))
                    raw_args = fn_block.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            args_dict = json.loads(raw_args)
                        except json.JSONDecodeError:
                            args_dict = {}
                    else:
                        args_dict = dict(raw_args or {})

                    if fn_name != tool_name:
                        res_payload: dict[str, Any] = {"error": f"Unsupported tool: {fn_name}"}
                    elif len(queries_executed) >= max_queries or time.perf_counter() >= deadline:
                        budget_exhausted = True
                        res_payload = {
                            "error": (
                                f"Query limit ({max_queries}) or timeout reached. "
                                "Do not call any more tools; output your final JSON answer now."
                            )
                        }
                    else:
                        q_str = sanitize_query(str(args_dict.get("query", "")))
                        queries_executed.append(q_str)
                        try:
                            res_payload = query_executor(q_str)
                        except Exception as exc:
                            res_payload = {"error": str(exc)}

                    history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "name": fn_name,
                            "content": json.dumps(res_payload, default=str),
                        }
                    )

                if budget_exhausted or len(queries_executed) >= max_queries:
                    history.append(
                        {
                            "role": "user",
                            "content": (
                                "Query budget is exhausted. Provide your final JSON "
                                "conclusion now without calling tools."
                            ),
                        }
                    )
                    if max_history_messages is not None:
                        trimmed, dropped = _trim_chat_history(history, max_history_messages)
                        history = trimmed
                        dropped_history_messages += dropped

                    synth_body, synth_tel = self._chat_completion(
                        model_name=self.synthesizer_model,
                        messages=history,
                        response_format=self._build_response_format(BaselineFinalEnvelope),
                    )
                    total_telemetry = total_telemetry.plus(synth_tel)
                    synth_choices = synth_body.get("choices") or []
                    if synth_choices:
                        final_text = _extract_message_text(synth_choices[0].get("message") or {})
                    break
        except ContextLengthExceededError as exc:
            context_overflow = True
            final_text = f"ContextLengthExceededError: {exc}"

        answers_by_qid: dict[str, str] = {
            q.question_id: (q.options[-1] if q.options else "unknown")
            for q in search_questions
        }
        conf_val = 0.0
        rationale_val = final_text if context_overflow else ""

        if not context_overflow and final_text:
            cleaned_final = extract_json_object(final_text)
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
                rationale_val = final_text[:500]

        return {
            "answers": answers_by_qid,
            "confidence": 0.0 if context_overflow else max(0.0, min(1.0, conf_val)),
            "rationale": rationale_val,
            "query_count": len(queries_executed),
            "queries_executed": queries_executed,
            "telemetry": total_telemetry,
            "peak_prompt_tokens": total_telemetry.peak_prompt_tokens,
            "context_overflow": context_overflow,
            "dropped_history_messages": dropped_history_messages,
        }


class _UnsupportedResponseFormatError(RuntimeError):
    """Internal signal to fall back once to prompt-only structured generation."""


__all__ = ["OpenAICompatibleProvider"]
