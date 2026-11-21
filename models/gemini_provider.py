"""Gemini 4 Interactions API provider (`gemini-4-hydrogen` and `gemini-4-carbon`).

Implements:
- Parallel SQL hypothesis proposal (`propose_hypotheses` + stateless ARES `make_propose_processor`)
- Multimodal fusion board patch synthesis (`propose_board_patch`, attaching up to 3 verifier PNG plots)
- High-VoI follow-up task proposal (`propose_followup_tasks`)
- Honest single-agent `gemini-4-carbon` read-only SQL baseline (`run_carbon_sql_baseline`)
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from ares.engine import ProcessorFn, ProcessorResult, TaskSpec
from contracts import (
    ArtifactRef,
    DecisionQuestion,
    HypothesisQuerySpec,
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
        """Return a new ModelTelemetry summing self and other."""
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


# ----------------------------------------------------------------------
# Strict Structured Output Schemas (no untyped dict[str, Any])
# ----------------------------------------------------------------------


class _ProposedHypothesisItem(BaseModel):
    query_sql: str = Field(
        description=(
            "Single read-only DuckDB SELECT or WITH...SELECT SQL query referencing "
            "registered table view names directly (e.g. service_logs, metrics, traces). "
            "Never use read_parquet(...) or <data_dir> placeholders."
        )
    )
    hypothesis_family: str = Field(description="Hypothesis family identifier.")
    expected_signal: str = Field(
        description="Concise description of the empirical signal or anomaly tested."
    )
    target_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Target board fields (e.g. logs_retry_spike, metrics_zscore_breach, "
            "traces_error_cluster, three_way_root_cause)."
        ),
    )
    verifier_name: str | None = Field(
        default=None,
        description=(
            "Exact registered verifier name from available_verifiers when deterministic "
            "certification applies, or null for exploratory SQL."
        ),
    )


class _HypothesisProposalEnvelope(BaseModel):
    queries: list[_ProposedHypothesisItem] = Field(default_factory=list)


class _FieldUpdateItem(BaseModel):
    field_name: str = Field(
        description=(
            "Board field name to create or update (prefer canonical names: "
            "logs_retry_spike, metrics_zscore_breach, traces_error_cluster, "
            "three_way_root_cause, fault_type, candidate_remediation)."
        )
    )
    value_json: str = Field(
        description=(
            "Exact value grounded in note evidence/summary, encoded as a plain string "
            "(e.g. 'payments-db') or JSON literal grounded directly in the cited notes."
        )
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Exact note_id values from the provided notes supporting this field.",
    )
    artifact_ids: list[str] = Field(
        default_factory=list,
        description="Exact artifact_id values from the cited notes relevant to this field.",
    )


class _BoardPatchEnvelope(BaseModel):
    updates: list[_FieldUpdateItem] = Field(default_factory=list)
    rationale: str = Field(
        default="",
        description="Concise synthesis rationale referencing cited note_ids.",
    )


class _BaselineFinalAnswer(BaseModel):
    root_cause: str = Field(
        description="Predicted verified root cause component name, or 'none_verified'."
    )
    recommended_action: str = Field(
        description="Recommended remediation action (e.g. rollback, restart, scale_pool, escalate)."
    )
    sufficiently_verified: str = Field(
        description="'yes' if all three required signals are verified on one component, else 'no'."
    )
    confidence: float = Field(
        description="Calibrated confidence in [0.0, 1.0]."
    )
    rationale: str = Field(
        default="",
        description="Brief empirical summary of SQL findings.",
    )


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


def _sanitize_sql(sql: str) -> str:
    cleaned = (sql or "").strip()
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
    """Collect up to `max_images` existing PNG ArtifactRefs from notes (verifier-produced first)."""
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
    # Prioritize deterministic verifier-produced artifacts first (Multimodal Addendum §1-§2)
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
            load_dotenv()
            key = self._api_key or os.environ.get("GEMINI_API_KEY")
            self._client = genai.Client(api_key=key) if key else genai.Client()
        return self._client

    def propose_hypotheses(
        self,
        *,
        schema_description: str,
        hypothesis_family: str,
        board: InvestigationBoard | None = None,
        available_verifiers: list[str] | None = None,
        requested_under: int = 0,
        max_queries: int = 2,
        thinking_level: str = "low",
        board_summary: str | None = None,
    ) -> tuple[list[HypothesisQuerySpec], ModelTelemetry]:
        """Use `gemini-4-hydrogen` to propose read-only DuckDB SQL HypothesisQuerySpecs."""
        cap = max(1, int(max_queries))
        allowed_verifiers = list(available_verifiers or [])
        effective_requested_under = (
            board.board_version if (board is not None and requested_under == 0) else requested_under
        )
        board_text = (
            board_summary
            if board_summary is not None
            else _summarize_board_for_prompt(board)
        )

        sys_instruction = (
            "You are an ARES parallel SQL hypothesis worker. "
            "Emit concise, high-signal, read-only DuckDB SELECT (or WITH ... SELECT) queries "
            "that test the assigned hypothesis family. "
            "CRITICAL SQL RULES:\n"
            "1. Reference table view names directly (e.g. `service_logs`, `metrics`, `traces`). "
            "NEVER call `read_parquet(...)` and NEVER include `<data_dir>` placeholders.\n"
            "2. Emit single read-only SELECT/WITH queries only (no semicolons, no DDL/DML).\n"
            "3. Bound result sets with GROUP BY / ORDER BY / LIMIT <= 50.\n"
            "4. When a registered deterministic verifier in `available_verifiers` directly tests "
            "the target signal, bind its exact name in `verifier_name`; otherwise set `verifier_name` to null.\n"
            "5. Target canonical board fields where applicable: `logs_retry_spike`, "
            "`metrics_zscore_breach`, `traces_error_cluster`, `three_way_root_cause`."
        )
        prompt = (
            f"### Schema Description\n{schema_description}\n\n"
            f"### Assigned Hypothesis Family\n{hypothesis_family}\n\n"
            f"### Available Trusted Verifiers\n{json.dumps(allowed_verifiers)}\n\n"
            f"### Current Investigation Board\n{board_text}\n\n"
            f"Propose up to {cap} distinct read-only DuckDB SQL hypothesis queries for family "
            f"'{hypothesis_family}'."
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

        allowed_set = set(allowed_verifiers)
        specs: list[HypothesisQuerySpec] = []
        for item in envelope.queries[:cap]:
            sql = _sanitize_sql(item.query_sql)
            if not sql:
                continue
            vname = item.verifier_name
            if vname in ("", "null", "none", "None"):
                vname = None
            if vname is not None and allowed_verifiers and vname not in allowed_set:
                vname = None
            specs.append(
                HypothesisQuerySpec(
                    query_sql=sql,
                    hypothesis_family=item.hypothesis_family or hypothesis_family,
                    expected_signal=item.expected_signal,
                    target_fields=list(item.target_fields),
                    verifier_name=vname,
                    requested_under=effective_requested_under,
                )
            )
        return specs, telemetry

    def make_propose_processor(
        self,
        *,
        schema_description: str,
        available_verifiers: list[str] | None = None,
    ) -> ProcessorFn:
        """Build a stateless ARES ProcessorFn wrapping `propose_hypotheses`."""

        def _processor(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
            del dep_outputs  # Stateless proposal from TaskSpec.context
            ctx = spec.context or {}
            family = str(ctx.get("hypothesis_family", "general_exploration"))
            max_queries = int(ctx.get("max_queries", 2))
            thinking_level = str(ctx.get("thinking_level", "low"))
            schema = str(ctx.get("schema_description", schema_description))
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
                schema_description=schema,
                hypothesis_family=family,
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
        schema_description: str = "",
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

        sys_instruction = (
            "You are the ARES Fusion Synthesis Engine (`gemini-4-carbon`). "
            "Synthesize new immutable evidence notes (and any attached verifier PNG plots) "
            "into a strictly grounded patch for the bounded InvestigationBoard.\n"
            "CRITICAL INVARIANTS:\n"
            "1. Populate canonical pivotal fields whenever supported by note evidence:\n"
            "   - `logs_retry_spike`: service name (e.g. 'payments-db') with CERTIFIED/observed "
            "upstream ERROR retry spike (`verify_upstream_retry_spike` or `service_logs` notes). "
            "If a verifier returned 0 candidates (`UNCERTIFIED`), do NOT claim a service passed.\n"
            "   - `metrics_zscore_breach`: service name with metric z-score >= 4.0 breach "
            "(`verify_metric_zscore_breach` or `metrics` notes).\n"
            "   - `traces_error_cluster`: service name with >= 12 ERROR trace spans >= 800ms "
            "(`verify_trace_error_cluster` or `traces` notes).\n"
            "   - `three_way_root_cause`: verified root cause service from `verify_three_way_root_cause` "
            "when `certification == 'CERTIFIED'` and `verified_root_causes` is non-empty "
            "(or 'none_verified' if `verify_three_way_root_cause` ran and found 0 candidates).\n"
            "   - `fault_type`: grounded fault signature when supported by notes.\n"
            "   - `candidate_remediation`: grounded remediation ('rollback', 'restart', 'scale_pool', "
            "or 'escalate') when supported by notes.\n"
            "2. Prefer CERTIFIED verifier notes over UNCERTIFIED exploratory notes when citing `sources`, "
            "because citing any UNCERTIFIED note taints a field as uncertified.\n"
            "3. ONLY cite exact `note_id` strings from the provided notes/board, and ONLY attach "
            "exact `artifact_id` strings from those notes.\n"
            "4. NEVER invent service names, counts, or values—every `value_json` must be grounded "
            "strictly in the cited notes' `evidence` or `summary`."
        )

        png_artifacts = _collect_png_artifacts(new_notes, max_images=3) if include_images else []
        artifact_manifest = [
            {"artifact_id": a.artifact_id, "kind": a.kind, "mime_type": a.mime_type}
            for a in png_artifacts
        ]
        prompt_str = (
            f"### Domain Schema\n{schema_description or '(not provided)'}\n\n"
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

        patch = {
            "updates": updates_dict,
            "rationale": envelope.rationale,
        }
        return patch, telemetry

    def propose_followup_tasks(
        self,
        *,
        board: InvestigationBoard,
        high_voi_fields: list[str],
        schema_description: str,
        available_verifiers: list[str] | None = None,
        max_queries: int = 3,
        thinking_level: str = "low",
    ) -> tuple[list[HypothesisQuerySpec], ModelTelemetry]:
        """Use `gemini-4-carbon` to propose targeted follow-up SQL/verifier tasks for high-VoI gaps."""
        cap = max(1, int(max_queries))
        allowed_verifiers = list(available_verifiers or [])

        sys_instruction = (
            "You are the ARES Follow-Up Acquisition Planner (`gemini-4-carbon`). "
            "Inspect the current InvestigationBoard and propose targeted read-only DuckDB SQL "
            "queries (and deterministic verifier bindings from `available_verifiers`) that resolve "
            "high-VoI unknown or uncertified fields.\n"
            "CRITICAL RULES:\n"
            "1. Reference table view names directly (`service_logs`, `metrics`, `traces`) — "
            "NEVER use `read_parquet(...)` or `<data_dir>`.\n"
            "2. Bind `verifier_name` to an exact entry from `available_verifiers` whenever certifying "
            "`logs_retry_spike` (`verify_upstream_retry_spike`), `metrics_zscore_breach` "
            "(`verify_metric_zscore_breach`), `traces_error_cluster` (`verify_trace_error_cluster`), "
            "or `three_way_root_cause` (`verify_three_way_root_cause`).\n"
            "3. Emit single read-only SELECT/WITH queries without semicolons."
        )
        prompt = (
            f"### Schema Description\n{schema_description}\n\n"
            f"### Available Trusted Verifiers\n{json.dumps(allowed_verifiers)}\n\n"
            f"### High-VoI / Uncertified Target Fields\n{json.dumps(high_voi_fields)}\n\n"
            f"### Current InvestigationBoard\n{_summarize_board_for_prompt(board)}\n\n"
            f"Propose up to {cap} targeted follow-up HypothesisQuerySpecs."
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

        allowed_set = set(allowed_verifiers)
        specs: list[HypothesisQuerySpec] = []
        for item in envelope.queries[:cap]:
            sql = _sanitize_sql(item.query_sql)
            if not sql:
                continue
            vname = item.verifier_name
            if vname in ("", "null", "none", "None"):
                vname = None
            if vname is not None and allowed_verifiers and vname not in allowed_set:
                vname = None
            specs.append(
                HypothesisQuerySpec(
                    query_sql=sql,
                    hypothesis_family=item.hypothesis_family or "followup_voi_acquisition",
                    expected_signal=item.expected_signal,
                    target_fields=list(item.target_fields or high_voi_fields),
                    verifier_name=vname,
                    requested_under=board.board_version,
                )
            )
        return specs, telemetry

    def run_carbon_sql_baseline(
        self,
        *,
        schema_description: str,
        decision_questions: list[DecisionQuestion],
        sql_executor: Callable[[str], dict[str, Any]],
        max_queries: int = 12,
        timeout_seconds: float = 180.0,
        thinking_level: str = "medium",
    ) -> dict[str, Any]:
        """Run an honest single-agent `gemini-4-carbon` read-only SQL baseline with stateless tools."""
        sql_tool = {
            "type": "function",
            "name": "execute_readonly_sql",
            "description": (
                "Execute a read-only DuckDB SELECT/WITH query over table views "
                "(service_logs, metrics, traces) and return columns, rows, and row_count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": (
                            "Single read-only DuckDB SELECT/WITH query referencing view names "
                            "directly (never read_parquet or <data_dir>)."
                        ),
                    }
                },
                "required": ["sql"],
            },
        }

        questions_desc = [q.model_dump() for q in decision_questions]
        sys_instruction = (
            "You are a senior SRE incident investigator with access to `execute_readonly_sql`. "
            "Query the DuckDB views (`service_logs`, `metrics`, `traces`) directly—never use "
            "`read_parquet(...)` or `<data_dir>`. "
            f"You may execute at most {max_queries} SQL queries. "
            "Check all three required signals (upstream ERROR retries, metric z-score >= 4.0 breaches, "
            "and correlated ERROR trace spans >= 800ms on shared trace_ids) before concluding. "
            "If any signal is missing for all candidate services, conclude root_cause='none_verified', "
            "recommended_action='escalate', sufficiently_verified='no'. "
            "When finished, respond with a JSON object containing keys: "
            "`root_cause`, `recommended_action`, `sufficiently_verified` ('yes'|'no'), "
            "`confidence` (0.0..1.0), and `rationale`."
        )
        initial_prompt = (
            f"### Schema Description\n{schema_description}\n\n"
            f"### Decision Questions\n{json.dumps(questions_desc, indent=2)}\n\n"
            "Investigate the incident using `execute_readonly_sql` and output your final JSON answer."
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
                tools=[sql_tool],
                generation_config={"thinking_level": norm_thinking},
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            total_telemetry = total_telemetry.plus(extract_telemetry(interaction, latency_ms))

            steps = getattr(interaction, "steps", None) or []
            fc_steps: list[Any] = []
            for step in steps:
                dumped = step.model_dump() if hasattr(step, "model_dump") else dict(step)
                history.append(dumped)
                stype = dumped.get("type")
                if stype == "function_call":
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

                if fc_name != "execute_readonly_sql":
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
                    sql_str = _sanitize_sql(str(args_dict.get("sql", "")))
                    queries_executed.append(sql_str)
                    try:
                        res_payload = sql_executor(sql_str)
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
                                    "SQL query budget is exhausted. Provide your final JSON "
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
                    tools=[sql_tool],
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": _BaselineFinalAnswer.model_json_schema(),
                    },
                    generation_config={"thinking_level": norm_thinking},
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                total_telemetry = total_telemetry.plus(extract_telemetry(synth, latency_ms))
                final_text = getattr(synth, "output_text", "") or ""
                break

        parsed_answer = self._parse_baseline_answer(final_text)
        return {
            "root_cause": parsed_answer.root_cause,
            "recommended_action": parsed_answer.recommended_action,
            "sufficiently_verified": parsed_answer.sufficiently_verified,
            "confidence": max(0.0, min(1.0, float(parsed_answer.confidence))),
            "rationale": parsed_answer.rationale,
            "answers": {
                "root_cause": parsed_answer.root_cause,
                "recommended_action": parsed_answer.recommended_action,
                "sufficiently_verified": parsed_answer.sufficiently_verified,
            },
            "queries_executed": queries_executed,
            "query_count": len(queries_executed),
            "telemetry": total_telemetry,
        }

    @staticmethod
    def _parse_baseline_answer(raw_text: str) -> _BaselineFinalAnswer:
        cleaned = _strip_json_fences(raw_text)
        if cleaned:
            # Try direct JSON parse, or extract first {...} JSON object from prose
            candidates = [cleaned]
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match and match.group(0) != cleaned:
                candidates.append(match.group(0))
            for cand in candidates:
                try:
                    data = json.loads(cand)
                    if isinstance(data, dict):
                        ver_raw = data.get("sufficiently_verified", "no")
                        if isinstance(ver_raw, bool):
                            ver_str = "yes" if ver_raw else "no"
                        else:
                            ver_str = "yes" if str(ver_raw).strip().lower() in ("yes", "true") else "no"
                        return _BaselineFinalAnswer(
                            root_cause=str(data.get("root_cause", "none_verified")),
                            recommended_action=str(data.get("recommended_action", "escalate")),
                            sufficiently_verified=ver_str,
                            confidence=float(data.get("confidence", 0.0)),
                            rationale=str(data.get("rationale", "")),
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return _BaselineFinalAnswer(
            root_cause="none_verified",
            recommended_action="escalate",
            sufficiently_verified="no",
            confidence=0.0,
            rationale=raw_text or "Timed out or unparseable baseline output.",
        )


__all__ = [
    "GeminiModelProvider",
    "ModelTelemetry",
    "extract_telemetry",
]
