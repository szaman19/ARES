"""Backend- and domain-agnostic Gemini 4 Interactions API provider (`gemini-4-hydrogen` and `gemini-4-carbon`).

Implements:
- Parallel hypothesis magnet proposal (`propose_hypotheses` + stateless ARES `make_propose_processor`)
- Multimodal fusion board patch synthesis (`propose_board_patch`, attaching up to 3 verifier PNG slices)
- High-VoI follow-up task proposal (`propose_followup_tasks`)
- Honest single-agent `gemini-4-carbon` read-only baseline (`run_single_agent_baseline`)
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from ares.engine import ProcessorFn, ProcessorResult, TaskSpec
from contracts import (
    ArtifactRef,
    DecisionQuestion,
    HypothesisSpec,
    InvestigationBoard,
    Note,
)

VALID_THINKING_LEVELS: frozenset[str] = frozenset({"low", "medium", "high"})


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


class ModelTelemetry(BaseModel):
    """Token, call count, and latency telemetry for Gemini model calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    latency_ms: float = 0.0

    def plus(self, other: ModelTelemetry) -> ModelTelemetry:
        return ModelTelemetry(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            thought_tokens=self.thought_tokens + other.thought_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            model_calls=self.model_calls + other.model_calls,
            latency_ms=round(self.latency_ms + other.latency_ms, 3),
        )


def extract_telemetry(resp: Any, latency_ms: float, model_calls: int = 1) -> ModelTelemetry:
    """Extract defensive ModelTelemetry from an Interactions or Decisions API response."""
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
        model_calls=model_calls,
        latency_ms=round(max(0.0, latency_ms), 3),
    )


class _ProposedHypothesisItem(BaseModel):
    query: str = Field(
        default="",
        description=(
            "Single read-only magnet query/expression obeying the domain's magnet rules "
            "(or empty string if strictly invoking a deterministic verifier via `params_json`)."
        ),
    )
    params_json: str = Field(
        default="{}",
        description="Optional JSON object string of structured parameters for the magnet or verifier.",
    )
    hypothesis_family: str = Field(description="Hypothesis family identifier.")
    expected_signal: str = Field(
        description="Concise description of the empirical signal or invariant tested."
    )
    target_fields: list[str] = Field(
        default_factory=list,
        description="Target InvestigationBoard fields informed by this hypothesis.",
    )
    verifier_name: str | None = Field(
        default=None,
        description=(
            "Exact registered verifier name from `available_verifiers` when deterministic "
            "certification applies, or null for exploratory probing."
        ),
    )


class _HypothesisProposalEnvelope(BaseModel):
    hypotheses: list[_ProposedHypothesisItem] = Field(default_factory=list)


class _FieldUpdateItem(BaseModel):
    field_name: str = Field(
        description="Board field name to create or update (prefer canonical pivotal field names)."
    )
    value_json: str = Field(
        description=(
            "Exact value grounded directly in the cited notes' evidence or summary, encoded as a "
            "plain string or JSON literal."
        )
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Exact `note_id` values from the provided notes supporting this field.",
    )
    artifact_ids: list[str] = Field(
        default_factory=list,
        description="Exact `artifact_id` values from the cited notes relevant to this field.",
    )


class _BoardPatchEnvelope(BaseModel):
    updates: list[_FieldUpdateItem] = Field(default_factory=list)
    rationale: str = Field(
        default="",
        description="Concise synthesis rationale referencing cited note_ids.",
    )


class _BaselineAnswerEntry(BaseModel):
    question_id: str = Field(description="Exact `question_id` from DecisionQuestions.")
    answer: str = Field(description="Selected option from the question's `options`.")


class _BaselineFinalEnvelope(BaseModel):
    answers: list[_BaselineAnswerEntry] = Field(default_factory=list)
    confidence: float = Field(description="Calibrated overall confidence in [0.0, 1.0].")
    rationale: str = Field(default="", description="Brief empirical summary of findings.")


def _strip_json_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_value_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return "unknown"
    try:
        parsed = json.loads(text)
        if parsed is None:
            return "unknown"
        return parsed
    except (json.JSONDecodeError, TypeError):
        return text


def _parse_params_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return dict(parsed) if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _sanitize_query(query: str) -> str:
    cleaned = (query or "").strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _summarize_board_for_prompt(board: InvestigationBoard | str | dict[str, Any] | None) -> str:
    if board is None:
        return "No active board state yet (initial exploration round)."
    if isinstance(board, str):
        return board
    if isinstance(board, dict):
        return json.dumps(board, indent=2, default=str)
    fields_summary = {
        fname: {
            "value": f.value,
            "status": f.status,
            "influence": f.influence,
            "voi": f.voi,
        }
        for fname, f in board.fields.items()
    }
    return json.dumps(
        {
            "board_version": board.board_version,
            "decision": board.decision.model_dump(),
            "fields": fields_summary,
        },
        indent=2,
        default=str,
    )


def _collect_png_artifacts(notes: list[Note], max_images: int = 3) -> list[ArtifactRef]:
    seen_ids: set[str] = set()
    candidates: list[ArtifactRef] = []
    for note in notes:
        for art in note.artifacts:
            if art.artifact_id in seen_ids:
                continue
            is_png = art.kind in ("plot_png", "image") or art.mime_type == "image/png"
            if not is_png:
                continue
            if not Path(art.path).is_file():
                continue
            seen_ids.add(art.artifact_id)
            candidates.append(art)
    candidates.sort(key=lambda a: (0 if a.produced_by_verifier else 1, a.artifact_id))
    return candidates[:max_images]


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

    def _parse_hypothesis_items(
        self,
        items: Sequence[_ProposedHypothesisItem],
        *,
        cap: int,
        default_family: str,
        default_targets: Sequence[str],
        allowed_verifiers: Sequence[str],
        requested_under: int,
    ) -> list[HypothesisSpec]:
        allowed_set = set(allowed_verifiers)
        specs: list[HypothesisSpec] = []
        for item in items[:cap]:
            q_str = _sanitize_query(item.query)
            params = _parse_params_json(item.params_json)
            vname = item.verifier_name
            if vname in ("", "null", "none", "None"):
                vname = None
            if vname is not None and allowed_verifiers and vname not in allowed_set:
                vname = None
            if not q_str and vname is None and not params:
                continue
            specs.append(
                HypothesisSpec(
                    query=q_str,
                    params=params,
                    hypothesis_family=item.hypothesis_family or default_family,
                    expected_signal=item.expected_signal,
                    target_fields=list(item.target_fields or default_targets),
                    verifier_name=vname,
                    requested_under=requested_under,
                )
            )
        return specs

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
        board_text = (
            board_summary
            if board_summary is not None
            else _summarize_board_for_prompt(board)
        )

        rules_block = f"\nCRITICAL MAGNET RULES:\n{magnet_rules}\n" if magnet_rules else ""
        pivotal_block = (
            f"\nTarget canonical pivotal board fields where applicable: {list(pivotal_fields)}."
            if pivotal_fields
            else ""
        )
        sys_instruction = (
            "You are an ARES parallel hypothesis worker (`gemini-4-hydrogen`). "
            "Emit concise, high-signal, read-only magnet queries and/or deterministic verifier "
            "bindings that test the assigned hypothesis family without reading raw haystacks."
            f"{rules_block}"
            "When a registered deterministic verifier in `available_verifiers` directly tests "
            "the target signal, bind its exact name in `verifier_name`; otherwise set `verifier_name` to null."
            f"{pivotal_block}"
        )
        prompt = (
            f"### Domain Environment\n{environment_description}\n\n"
            f"### Assigned Hypothesis Family\n{hypothesis_family}\n\n"
            f"### Available Trusted Verifiers\n{json.dumps(allowed_verifiers)}\n\n"
            f"### Current Investigation Board\n{board_text}\n\n"
            f"Propose up to {cap} distinct hypothesis magnets for family '{hypothesis_family}'."
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
                "schema": _HypothesisProposalEnvelope.model_json_schema(),
            },
            generation_config={"thinking_level": _normalize_thinking_level(thinking_level)},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = extract_telemetry(interaction, latency_ms)

        raw_text = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
        envelope = _HypothesisProposalEnvelope.model_validate_json(raw_text)
        specs = self._parse_hypothesis_items(
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

        def _processor(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
            del dep_outputs
            ctx = spec.context or {}
            family = str(ctx.get("hypothesis_family", "general_exploration"))
            max_queries = int(ctx.get("max_queries", 2))
            thinking_level = str(ctx.get("thinking_level", "low"))
            env_desc = str(ctx.get("environment_description", environment_description))
            rules = str(ctx.get("magnet_rules", magnet_rules))
            piv_fields = ctx.get("pivotal_fields", pivotal_fields)
            verifiers = ctx.get("available_verifiers", available_verifiers)

            raw_board = ctx.get("board")
            board_obj: InvestigationBoard | None = None
            board_summary: str | None = ctx.get("board_summary")
            if isinstance(raw_board, InvestigationBoard):
                board_obj = raw_board
            elif isinstance(raw_board, dict) and "decision" in raw_board:
                board_obj = InvestigationBoard.model_validate(raw_board)
            elif isinstance(raw_board, str) and board_summary is None:
                board_summary = raw_board

            specs, telemetry = self.propose_hypotheses(
                environment_description=env_desc,
                hypothesis_family=family,
                magnet_rules=rules,
                pivotal_fields=piv_fields,
                board=board_obj,
                available_verifiers=verifiers,
                requested_under=spec.requested_under,
                max_queries=max_queries,
                thinking_level=thinking_level,
                board_summary=board_summary,
            )
            return ProcessorResult(
                output=specs,
                prompt_tokens=telemetry.prompt_tokens,
                completion_tokens=telemetry.completion_tokens,
                model_calls=telemetry.model_calls,
            )

        return _processor

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
        notes_payload = [
            {
                "note_id": n.note_id,
                "certification": n.certification,
                "verifier_name": n.verifier_name,
                "hypothesis_family": n.hypothesis_family,
                "summary": n.summary,
                "row_count": n.row_count,
                "evidence": n.evidence,
                "artifact_ids": [a.artifact_id for a in n.artifacts],
            }
            for n in new_notes
        ]

        valid_note_ids: set[str] = {n.note_id for n in new_notes}
        valid_artifact_ids: set[str] = {
            a.artifact_id for n in new_notes for a in n.artifacts
        }
        for existing_field in board.fields.values():
            valid_note_ids.update(existing_field.sources)
            valid_artifact_ids.update(existing_field.artifact_ids)

        pivotal_hint = (
            f"1. Populate canonical pivotal fields {list(pivotal_fields)} whenever supported by note evidence "
            "(if a verifier returned 0 candidates / UNCERTIFIED, do NOT claim a candidate passed).\n"
            if pivotal_fields
            else "1. Populate concise, informative board field names grounded in note evidence.\n"
        )
        sys_instruction = (
            "You are the ARES Fusion Synthesis Engine (`gemini-4-carbon`). "
            "Synthesize new immutable evidence notes (and any attached verifier PNG artifacts) "
            "into a strictly grounded patch for the bounded InvestigationBoard.\n"
            "CRITICAL INVARIANTS:\n"
            f"{pivotal_hint}"
            "2. Prefer CERTIFIED verifier notes over UNCERTIFIED exploratory notes when citing `sources`, "
            "because citing any UNCERTIFIED note taints a field as uncertified.\n"
            "3. ONLY cite exact `note_id` strings from the provided notes/board, and ONLY attach "
            "exact `artifact_id` strings from those notes.\n"
            "4. NEVER invent entities, counts, or values—every `value_json` must be grounded "
            "strictly in the cited notes' `evidence` or `summary`."
        )

        png_artifacts = _collect_png_artifacts(new_notes, max_images=3) if include_images else []
        artifact_manifest = [
            {"artifact_id": a.artifact_id, "kind": a.kind, "mime_type": a.mime_type}
            for a in png_artifacts
        ]
        prompt_str = (
            f"### Domain Environment\n{environment_description or '(not provided)'}\n\n"
            f"### Current InvestigationBoard\n{_summarize_board_for_prompt(board)}\n\n"
            f"### New Immutable Evidence Notes\n{json.dumps(notes_payload, indent=2, default=str)}\n\n"
            f"### Attached Verifier PNG Artifacts\n{json.dumps(artifact_manifest)}\n\n"
            "Produce the grounded board patch updates and rationale."
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
                "schema": _BoardPatchEnvelope.model_json_schema(),
            },
            generation_config={"thinking_level": _normalize_thinking_level(thinking_level)},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = extract_telemetry(interaction, latency_ms)

        raw_text = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
        envelope = _BoardPatchEnvelope.model_validate_json(raw_text)

        updates_dict: dict[str, dict[str, Any]] = {}
        for item in envelope.updates:
            fname = item.field_name.strip()
            if not fname:
                continue
            filtered_sources = [s for s in item.sources if s in valid_note_ids]
            if not filtered_sources:
                continue
            filtered_artifacts = [a for a in item.artifact_ids if a in valid_artifact_ids]
            updates_dict[fname] = {
                "value": _parse_value_json(item.value_json),
                "sources": filtered_sources,
                "artifact_ids": filtered_artifacts,
            }

        return {
            "updates": updates_dict,
            "rationale": envelope.rationale,
        }, telemetry

    def propose_followup_tasks(
        self,
        *,
        board: InvestigationBoard,
        high_voi_fields: list[str],
        environment_description: str,
        magnet_rules: str = "",
        available_verifiers: Sequence[str] | None = None,
        max_queries: int = 3,
        thinking_level: str = "low",
    ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
        """Use `gemini-4-carbon` to propose targeted follow-up magnets/verifiers for high-VoI gaps."""
        cap = max(1, int(max_queries))
        allowed_verifiers = list(available_verifiers or ())
        rules_block = f"\nCRITICAL MAGNET RULES:\n{magnet_rules}\n" if magnet_rules else ""

        sys_instruction = (
            "You are the ARES Follow-Up Acquisition Planner (`gemini-4-carbon`). "
            "Inspect the current InvestigationBoard and propose targeted read-only magnet queries "
            "and deterministic verifier bindings from `available_verifiers` that resolve high-VoI "
            "unknown or uncertified fields."
            f"{rules_block}"
        )
        prompt = (
            f"### Domain Environment\n{environment_description}\n\n"
            f"### Available Trusted Verifiers\n{json.dumps(allowed_verifiers)}\n\n"
            f"### High-VoI / Uncertified Target Fields\n{json.dumps(high_voi_fields)}\n\n"
            f"### Current InvestigationBoard\n{_summarize_board_for_prompt(board)}\n\n"
            f"Propose up to {cap} targeted follow-up HypothesisSpecs."
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
                "schema": _HypothesisProposalEnvelope.model_json_schema(),
            },
            generation_config={"thinking_level": _normalize_thinking_level(thinking_level)},
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = extract_telemetry(interaction, latency_ms)

        raw_text = _strip_json_fences(getattr(interaction, "output_text", "") or "{}")
        envelope = _HypothesisProposalEnvelope.model_validate_json(raw_text)
        specs = self._parse_hypothesis_items(
            envelope.hypotheses,
            cap=cap,
            default_family="followup_voi_acquisition",
            default_targets=high_voi_fields,
            allowed_verifiers=allowed_verifiers,
            requested_under=board.board_version,
        )
        return specs, telemetry

    def run_single_agent_baseline(
        self,
        *,
        environment_description: str,
        decision_questions: Sequence[DecisionQuestion],
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

        questions_desc = [q.model_dump() for q in decision_questions]
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
            f"### Decision Questions\n{json.dumps(questions_desc, indent=2)}\n\n"
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
                    q_str = _sanitize_query(
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
                                    "decision now without calling tools."
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
                        "schema": _BaselineFinalEnvelope.model_json_schema(),
                    },
                    generation_config={"thinking_level": norm_thinking},
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                total_telemetry = total_telemetry.plus(extract_telemetry(synth, latency_ms))
                final_text = getattr(synth, "output_text", "") or ""
                break

        answers_by_qid: dict[str, str] = {
            q.question_id: (q.options[-1] if q.options else "unknown")
            for q in decision_questions
        }
        conf_val = 0.0
        rationale_val = ""

        cleaned_final = _strip_json_fences(final_text)
        if cleaned_final:
            try:
                raw_json = json.loads(cleaned_final)
                if isinstance(raw_json, dict):
                    if isinstance(raw_json.get("answers"), list):
                        env = _BaselineFinalEnvelope.model_validate(raw_json)
                        for entry in env.answers:
                            if entry.question_id in answers_by_qid:
                                answers_by_qid[entry.question_id] = entry.answer
                        conf_val = float(env.confidence)
                        rationale_val = env.rationale
                    else:
                        for q in decision_questions:
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
        }


__all__ = [
    "GeminiModelProvider",
    "ModelTelemetry",
    "extract_telemetry",
]
