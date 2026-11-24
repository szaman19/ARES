"""Shared `ModelProvider` protocol, telemetry, schemas, and lossless prompt/JSON helpers for ARES."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from ares.contracts import (
    ArtifactRef,
    HypothesisSpec,
    InvestigationBoard,
    Note,
)
from ares.engine import ProcessorFn, ProcessorResult, TaskSpec


class ContextLengthExceededError(RuntimeError):
    """Raised explicitly when a model backend reports context-window exhaustion."""


class ModelTelemetry(BaseModel):
    """Token, peak-prompt, call count, and latency telemetry for model provider calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thought_tokens: int = 0
    total_tokens: int = 0
    peak_prompt_tokens: int = 0
    model_calls: int = 0
    latency_ms: float = 0.0

    def model_post_init(self, __context: Any) -> None:
        if self.peak_prompt_tokens == 0 and self.model_calls == 1 and self.prompt_tokens > 0:
            self.peak_prompt_tokens = self.prompt_tokens

    def plus(self, other: ModelTelemetry) -> ModelTelemetry:
        return ModelTelemetry(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            thought_tokens=self.thought_tokens + other.thought_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            peak_prompt_tokens=max(self.peak_prompt_tokens, other.peak_prompt_tokens),
            model_calls=self.model_calls + other.model_calls,
            latency_ms=round(self.latency_ms + other.latency_ms, 3),
        )


@runtime_checkable
class ModelProvider(Protocol):
    """Pluggable LLM provider protocol for ARES parallel workers and context synthesis."""

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
        """Propose structured HypothesisSpecs for one hypothesis family."""
        ...

    def make_propose_processor(
        self,
        *,
        environment_description: str,
        magnet_rules: str = "",
        pivotal_fields: Sequence[str] = (),
        available_verifiers: Sequence[str] | None = None,
    ) -> ProcessorFn:
        """Build a stateless ARES ProcessorFn wrapping `propose_hypotheses`."""
        ...

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
        """Synthesize a grounded patch dict for `InvestigationBoard` from `new_notes`."""
        ...

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
    ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
        """Propose targeted follow-up HypothesisSpecs for unresolved board gap fields."""
        ...


class ProposedHypothesisItem(BaseModel):
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


class HypothesisProposalEnvelope(BaseModel):
    hypotheses: list[ProposedHypothesisItem]


class FieldUpdateItem(BaseModel):
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


class BoardPatchEnvelope(BaseModel):
    updates: list[FieldUpdateItem]
    rationale: str = Field(
        default="",
        description="Concise synthesis rationale referencing cited note_ids.",
    )


class BaselineAnswerEntry(BaseModel):
    question_id: str = Field(description="Exact `question_id` from SearchQuestions.")
    answer: str = Field(description="Selected option from the question's `options`.")


class BaselineFinalEnvelope(BaseModel):
    answers: list[BaselineAnswerEntry]
    confidence: float = Field(description="Calibrated overall confidence in [0.0, 1.0].")
    rationale: str = Field(default="", description="Brief empirical summary of findings.")


def strip_json_fences(raw_text: str) -> str:
    """Strip outer markdown code fences if present."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before `}` or `]` outside JSON string literals."""
    out: list[str] = []
    in_string = False
    escape = False
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue

        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in ("}", "]"):
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)


def extract_json_object(raw_text: str) -> str:
    """Extract a balanced JSON object from raw model output, tolerating fences, prose, and trailing commas."""
    text = strip_json_fences(raw_text)
    if not text:
        return "{}"

    # If model wrapped fenced JSON inside prose (e.g. "Here is the JSON:\n```json ... ```"),
    # prefer the fenced block if it contains an object.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text or "", flags=re.IGNORECASE)
    if fence_match:
        inner = fence_match.group(1).strip()
        if "{" in inner:
            text = inner

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    end_idx: int | None = None

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if end_idx is None:
        return _remove_trailing_commas(text[start:])
    return _remove_trailing_commas(text[start : end_idx + 1])


def parse_value_json(raw: str) -> Any:
    """Parse a JSON-encoded field value string, falling back to plain text."""
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


def parse_params_json(raw: str) -> dict[str, Any]:
    """Parse a JSON object string of structured parameters."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return dict(parsed) if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def sanitize_query(query: str) -> str:
    """Trim whitespace and trailing semicolons from a magnet query."""
    cleaned = (query or "").strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def summarize_board_for_prompt(board: InvestigationBoard | str | dict[str, Any] | None) -> str:
    """Render a compact lossless JSON summary of `InvestigationBoard` for model prompts."""
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
            "priority": f.priority,
            "uncertainty": f.uncertainty,
        }
        for fname, f in board.fields.items()
    }
    return json.dumps(
        {
            "board_version": board.board_version,
            "conclusion": board.conclusion.model_dump(),
            "fields": fields_summary,
        },
        indent=2,
        default=str,
    )


def collect_png_artifacts(notes: list[Note], max_images: int = 3) -> list[ArtifactRef]:
    """Collect up to `max_images` existing PNG ArtifactRefs from `notes`, prioritizing verifiers."""
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


def parse_hypothesis_items(
    items: Sequence[ProposedHypothesisItem],
    *,
    cap: int,
    default_family: str,
    default_targets: Sequence[str],
    allowed_verifiers: Sequence[str],
    requested_under: int,
) -> list[HypothesisSpec]:
    """Convert raw structured hypothesis items into validated `HypothesisSpec` objects."""
    allowed_set = set(allowed_verifiers)
    specs: list[HypothesisSpec] = []
    for item in items[:cap]:
        q_str = sanitize_query(item.query)
        params = parse_params_json(item.params_json)
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


def filter_board_patch_updates(
    envelope: BoardPatchEnvelope,
    valid_note_ids: set[str],
    valid_artifact_ids: set[str],
) -> dict[str, Any]:
    """Filter parsed `BoardPatchEnvelope` updates to existing cited note and artifact IDs."""
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
            "value": parse_value_json(item.value_json),
            "sources": filtered_sources,
            "artifact_ids": filtered_artifacts,
        }

    return {
        "updates": updates_dict,
        "rationale": envelope.rationale,
    }


def build_propose_prompts(
    *,
    environment_description: str,
    hypothesis_family: str,
    magnet_rules: str = "",
    pivotal_fields: Sequence[str] = (),
    board: InvestigationBoard | None = None,
    available_verifiers: Sequence[str] | None = None,
    max_queries: int = 2,
    board_summary: str | None = None,
    role_label: str = "parallel hypothesis worker",
) -> tuple[str, str]:
    """Build `(system_instruction, user_prompt)` for initial hypothesis proposal."""
    cap = max(1, int(max_queries))
    allowed_verifiers = list(available_verifiers or ())
    board_text = (
        board_summary
        if board_summary is not None
        else summarize_board_for_prompt(board)
    )
    rules_block = f"\nCRITICAL MAGNET RULES:\n{magnet_rules}\n" if magnet_rules else ""
    pivotal_block = (
        f"\nTarget canonical pivotal board fields where applicable: {list(pivotal_fields)}."
        if pivotal_fields
        else ""
    )
    sys_instruction = (
        f"You are an ARES {role_label}. "
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
    return sys_instruction, prompt


def build_board_patch_prompts(
    *,
    board: InvestigationBoard,
    new_notes: list[Note],
    environment_description: str = "",
    pivotal_fields: Sequence[str] = (),
    include_images: bool = True,
    role_label: str = "ARES Context Synthesis Engine",
) -> tuple[str, str, set[str], set[str], list[ArtifactRef]]:
    """Build lossless synthesis prompts plus valid citation sets and PNG artifacts."""
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
        f"You are the {role_label}. "
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

    png_artifacts = collect_png_artifacts(new_notes, max_images=3) if include_images else []
    artifact_manifest = [
        {"artifact_id": a.artifact_id, "kind": a.kind, "mime_type": a.mime_type}
        for a in png_artifacts
    ]
    prompt_str = (
        f"### Domain Environment\n{environment_description or '(not provided)'}\n\n"
        f"### Current InvestigationBoard\n{summarize_board_for_prompt(board)}\n\n"
        f"### New Immutable Evidence Notes\n{json.dumps(notes_payload, indent=2, default=str)}\n\n"
        f"### Attached Verifier PNG Artifacts\n{json.dumps(artifact_manifest)}\n\n"
        "Produce the grounded board patch updates and rationale."
    )
    return sys_instruction, prompt_str, valid_note_ids, valid_artifact_ids, png_artifacts


def build_followup_prompts(
    *,
    board: InvestigationBoard,
    target_gap_fields: list[str],
    environment_description: str,
    magnet_rules: str = "",
    available_verifiers: Sequence[str] | None = None,
    max_queries: int = 3,
    attempted_queries: Sequence[str] = (),
    role_label: str = "ARES Follow-Up Search Planner",
) -> tuple[str, str]:
    """Build `(system_instruction, user_prompt)` for follow-up task generation."""
    cap = max(1, int(max_queries))
    allowed_verifiers = list(available_verifiers or ())
    rules_block = f"\nCRITICAL MAGNET RULES:\n{magnet_rules}\n" if magnet_rules else ""

    sys_instruction = (
        f"You are the {role_label}. "
        "Inspect the current InvestigationBoard and propose targeted read-only magnet queries "
        "and deterministic verifier bindings from `available_verifiers` that resolve "
        "uncertain, unknown, or uncertified target fields."
        f"{rules_block}"
    )
    attempted_block = (
        f"### Already Attempted Queries (Do Not Repeat)\n{json.dumps(list(attempted_queries), indent=2)}\n\n"
        if attempted_queries
        else ""
    )
    prompt = (
        f"### Domain Environment\n{environment_description}\n\n"
        f"### Available Trusted Verifiers\n{json.dumps(allowed_verifiers)}\n\n"
        f"### Unresolved / Uncertified Target Fields\n{json.dumps(target_gap_fields)}\n\n"
        f"{attempted_block}"
        f"### Current InvestigationBoard\n{summarize_board_for_prompt(board)}\n\n"
        f"Propose up to {cap} targeted follow-up HypothesisSpecs."
    )
    return sys_instruction, prompt


def default_make_propose_processor(
    provider: ModelProvider,
    *,
    environment_description: str,
    magnet_rules: str = "",
    pivotal_fields: Sequence[str] = (),
    available_verifiers: Sequence[str] | None = None,
) -> ProcessorFn:
    """Build a stateless ARES ProcessorFn wrapping any `ModelProvider.propose_hypotheses`."""

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
        elif isinstance(raw_board, dict) and "conclusion" in raw_board:
            board_obj = InvestigationBoard.model_validate(raw_board)
        elif isinstance(raw_board, str) and board_summary is None:
            board_summary = raw_board

        specs, telemetry = provider.propose_hypotheses(
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
            peak_prompt_tokens=telemetry.peak_prompt_tokens,
            model_calls=telemetry.model_calls,
        )

    return _processor


__all__ = [
    "BaselineAnswerEntry",
    "BaselineFinalEnvelope",
    "BoardPatchEnvelope",
    "ContextLengthExceededError",
    "FieldUpdateItem",
    "HypothesisProposalEnvelope",
    "ModelProvider",
    "ModelTelemetry",
    "ProposedHypothesisItem",
    "build_board_patch_prompts",
    "build_followup_prompts",
    "build_propose_prompts",
    "collect_png_artifacts",
    "default_make_propose_processor",
    "extract_json_object",
    "filter_board_patch_updates",
    "parse_hypothesis_items",
    "parse_params_json",
    "parse_value_json",
    "sanitize_query",
    "strip_json_fences",
    "summarize_board_for_prompt",
]
