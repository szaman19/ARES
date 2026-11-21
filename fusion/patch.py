"""Deterministic InvestigationBoard fusion, patch validation, and invariant enforcement.

100% modality-, backend-, and domain-agnostic. Enforces all hard board invariants
(ARES Plan §3.3, Invariants I1–I7):
- I1: Strict provenance, certified value groundedness, and multimodal artifact verification.
- I2: Monotonic / weakest-link certification and alias field deduplication.
- I3: Immutable NoteStore source-of-truth rebuild (`rebuild_board_from_notes`) and `board_drift`.
- I5: Staleness tracking (`requested_under` vs. `board_version`).
- I7: Bounded active board (`<= 12` fields; archive `influence < 0.02` after `>= 2` stale rounds).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ares.store import NoteStore
from contracts import (
    ArtifactRef,
    BoardDecision,
    BoardField,
    DecisionQuestion,
    FieldStatus,
    InvestigationBoard,
    Note,
)


_EMPTY_RESULT_SENTINELS: frozenset[str] = frozenset(
    {"none", "none_verified", "no_match", "absent", "false", "insufficient_evidence"}
)
_STATUS_RANK: dict[FieldStatus, int] = {
    "unknown": 0,
    "uncertified": 1,
    "certified": 2,
}
_METADATA_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "truncated",
        "estimated_units",
        "estimated_rows",
        "estimated_bytes",
        "duration_ms",
        "columns",
    }
)


NoteExtractorFn = Callable[[Note], dict[str, Any] | None]


class PatchValidationError(ValueError):
    """Raised when a board patch violates a deterministic fusion invariant (I1–I7)."""

    def __init__(
        self,
        invariant: str,
        field_name: str | None = None,
        message: str = "",
    ) -> None:
        if not message and field_name is not None and (" " in field_name or len(field_name) > 64):
            message = field_name
            field_name = None
        self.invariant: str = invariant
        self.field_name: str | None = field_name
        self.message: str = message
        loc = f"{invariant}:{field_name}" if field_name else invariant
        super().__init__(f"[{loc}] {message}" if message else f"[{loc}]")


class FieldPatch(BaseModel):
    """Proposed update to a single field on the InvestigationBoard."""

    value: Any = "unknown"
    sources: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    status: FieldStatus | None = None
    requested_under: int | None = None


class BoardPatch(BaseModel):
    """Proposed batch of field updates emitted by Fusion or deterministic replay."""

    updates: dict[str, FieldPatch] = Field(default_factory=dict)
    rationale: str = ""


class PatchReport(BaseModel):
    """Telemetry and audit report returned by `validate_and_merge_patch`."""

    board_version: int
    applied_fields: list[str] = Field(default_factory=list)
    deduplicated_fields: list[str] = Field(default_factory=list)
    stale_fields: list[str] = Field(default_factory=list)
    archived_fields: list[str] = Field(default_factory=list)
    rejected_fields: dict[str, str] = Field(default_factory=dict)
    board_drift: float = 0.0


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_patch_value(value: Any) -> Any:
    if value is None:
        return "unknown"
    if isinstance(value, str) and value.strip().lower() == "unknown":
        return "unknown"
    return value


def _canonical_json_key(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value.strip())
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def _values_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        if (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and not isinstance(a, bool)
            and not isinstance(b, bool)
        ):
            return math.isclose(float(a), float(b), abs_tol=1e-6)
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a.strip() == b.strip()
    return _canonical_json_key(a) == _canonical_json_key(b)


def _extract_leaves(
    obj: Any,
    *,
    str_leaves: set[str],
    num_leaves: list[float],
    bool_leaves: set[bool],
) -> None:
    if obj is None:
        return
    if isinstance(obj, bool):
        bool_leaves.add(obj)
        return
    if isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            num_leaves.append(float(obj))
        return
    if isinstance(obj, str):
        str_leaves.add(obj.lower())
        str_leaves.add(obj.strip().lower())
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _extract_leaves(
                v, str_leaves=str_leaves, num_leaves=num_leaves, bool_leaves=bool_leaves
            )
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _extract_leaves(
                item, str_leaves=str_leaves, num_leaves=num_leaves, bool_leaves=bool_leaves
            )


def _note_has_zero_candidates(note: Note) -> bool:
    if note.row_count == 0:
        return True
    ev = note.evidence
    if "candidates" in ev and isinstance(ev["candidates"], list) and len(ev["candidates"]) == 0:
        if "rows" not in ev or not ev.get("rows"):
            return True
    return False


def create_initial_board(
    decision_questions: Sequence[DecisionQuestion],
    *,
    seed_fields: Sequence[str] = (),
) -> InvestigationBoard:
    """Create a fresh version-0 InvestigationBoard from domain DecisionQuestions."""
    decisions: dict[str, BoardDecision] = {}
    for q in decision_questions:
        decisions[q.question_id] = BoardDecision(
            question_id=q.question_id,
            question=q.question,
            answer="unknown",
        )

    if decision_questions:
        first_qid = decision_questions[0].question_id
        primary = decisions[first_qid].model_copy(deep=True)
    else:
        primary = BoardDecision(
            question_id="primary_decision",
            question="What is the verified conclusion?",
            answer="unknown",
        )
        decisions[primary.question_id] = primary.model_copy(deep=True)

    fields: dict[str, BoardField] = {}
    for fname in _dedup_preserve_order(seed_fields):
        fields[fname] = BoardField(
            value="unknown",
            status="unknown",
            sources=[],
            artifact_ids=[],
            influence=0.0,
            voi=0.0,
            requested_under=0,
            last_updated_version=0,
            stale_rounds=0,
        )

    return InvestigationBoard(
        board_version=0,
        decision=primary,
        decisions=decisions,
        fields=fields,
        archived_fields=[],
    )


def compute_weakest_status(
    sources: Sequence[str],
    note_store: NoteStore,
    *,
    field_name: str | None = None,
) -> FieldStatus:
    """Compute weakest-link certification status across cited Note IDs (Invariant I2)."""
    if not sources:
        return "unknown"

    all_certified = True
    for nid in sources:
        if not note_store.exists(nid):
            raise PatchValidationError(
                "I1",
                field_name,
                f"Cited note_id '{nid}' does not exist in NoteStore.",
            )
        note = note_store.get(nid)
        if note.certification != "CERTIFIED":
            all_certified = False

    return "certified" if all_certified else "uncertified"


def value_is_grounded_in_notes(value: Any, cited_notes: Sequence[Note]) -> bool:
    """Deterministically verify that `value` is grounded in `cited_notes` (Invariant I1)."""
    if not cited_notes:
        return False

    str_leaves: set[str] = set()
    num_leaves: list[float] = []
    bool_leaves: set[bool] = set()
    summaries_lower: list[str] = []
    serialized_blobs_lower: list[str] = []

    for note in cited_notes:
        _extract_leaves(
            note.evidence,
            str_leaves=str_leaves,
            num_leaves=num_leaves,
            bool_leaves=bool_leaves,
        )
        num_leaves.append(float(note.row_count))
        if note.summary:
            summaries_lower.append(note.summary.lower())
        try:
            serialized_blobs_lower.append(
                json.dumps(note.evidence, sort_keys=True, default=str).lower()
            )
        except TypeError:
            serialized_blobs_lower.append(repr(note.evidence).lower())

    all_empty = all(_note_has_zero_candidates(n) for n in cited_notes)
    any_non_empty = any(n.row_count > 0 for n in cited_notes)

    def _scalar_grounded(val: Any) -> bool:
        if val is None:
            return all_empty
        if isinstance(val, bool):
            if val in bool_leaves:
                return True
            return val is any_non_empty
        if isinstance(val, (int, float)):
            fval = float(val)
            if not math.isfinite(fval):
                return False
            return any(math.isclose(fval, leaf, abs_tol=1e-2) for leaf in num_leaves)
        if isinstance(val, str):
            raw_lower = val.lower()
            stripped_lower = val.strip().lower()
            if not stripped_lower:
                return False
            if all_empty and stripped_lower in _EMPTY_RESULT_SENTINELS:
                return True
            if raw_lower in str_leaves or stripped_lower in str_leaves:
                return True
            if len(stripped_lower) >= 2:
                if any(stripped_lower in s for s in summaries_lower):
                    return True
                if any(stripped_lower in b for b in serialized_blobs_lower):
                    return True
            return False
        return False

    def _check(node: Any) -> bool:
        if isinstance(node, (str, int, float, bool)) or node is None:
            return _scalar_grounded(node)
        if isinstance(node, list):
            if not node:
                return all_empty
            return all(_check(item) for item in node)
        if isinstance(node, dict):
            if not node:
                return all_empty
            return all(_check(v) for v in node.values())
        return False

    return _check(value)


def _validate_and_collect_artifacts(
    explicit_artifact_ids: Sequence[str],
    cited_notes: Sequence[Note],
    note_store: NoteStore,
    *,
    field_name: str,
    verify_artifact_files: bool,
) -> list[str]:
    """Enforce I1 multimodal artifact provenance and auto-collect valid verifier artifacts."""
    cited_artifacts: dict[str, ArtifactRef] = {
        art.artifact_id: art for note in cited_notes for art in note.artifacts
    }

    validated: list[str] = []
    for aid in explicit_artifact_ids:
        if not isinstance(aid, str) or not aid.strip():
            raise PatchValidationError(
                "I1", field_name, "Artifact ID must be a non-empty string."
            )
        art = note_store.get_artifact(aid)
        if art is None:
            raise PatchValidationError(
                "I1", field_name, f"Artifact '{aid}' does not exist in NoteStore."
            )
        if aid not in cited_artifacts:
            raise PatchValidationError(
                "I1",
                field_name,
                f"Artifact '{aid}' does not belong to any cited Note in sources.",
            )
        if not art.produced_by_verifier:
            raise PatchValidationError(
                "I1",
                field_name,
                f"Artifact '{aid}' was not produced by a trusted verifier.",
            )
        if not art.sha256 or not art.sha256.strip():
            raise PatchValidationError(
                "I1",
                field_name,
                f"Artifact '{aid}' is missing a valid sha256 digest.",
            )
        if verify_artifact_files and not Path(art.path).is_file():
            raise PatchValidationError(
                "I1",
                field_name,
                f"Artifact '{aid}' file does not exist on disk: {art.path}",
            )
        validated.append(aid)

    for note in cited_notes:
        for art in note.artifacts:
            if (
                art.produced_by_verifier
                and bool(art.sha256 and art.sha256.strip())
                and (not verify_artifact_files or Path(art.path).is_file())
            ):
                validated.append(art.artifact_id)

    return _dedup_preserve_order(validated)


def deduplicate_fields(
    fields: dict[str, BoardField],
    note_store: NoteStore,
    *,
    candidate_names: Collection[str] | None = None,
    protected_fields: Collection[str] = (),
) -> tuple[dict[str, BoardField], list[str], dict[str, str]]:
    """Merge non-protected duplicate alias fields using union of sources and weakest status (I2)."""
    protected_set = frozenset(protected_fields)
    surviving: dict[str, BoardField] = {}
    deduplicated: list[str] = []
    alias_to_target: dict[str, str] = {}

    for fname, field in fields.items():
        can_be_alias = (
            fname not in protected_set
            and field.status != "unknown"
            and (candidate_names is None or fname in candidate_names)
        )
        if not can_be_alias:
            surviving[fname] = field.model_copy(deep=True)
            continue

        matched_target: str | None = None
        for target_name, target_field in surviving.items():
            if target_field.status == "unknown":
                continue
            if not _values_equal(field.value, target_field.value):
                continue

            if (
                target_name in protected_set
                and target_field.status == "certified"
                and field.status != "certified"
            ):
                continue

            same_sources = frozenset(field.sources) == frozenset(target_field.sources)
            identical_complex = isinstance(field.value, (dict, list)) and len(field.value) > 0
            shared_sources_non_protected = (
                target_name not in protected_set
                and bool(set(field.sources) & set(target_field.sources))
            )

            if same_sources or identical_complex or shared_sources_non_protected:
                matched_target = target_name
                break

        if matched_target is None:
            surviving[fname] = field.model_copy(deep=True)
            continue

        target = surviving[matched_target]
        merged_sources = _dedup_preserve_order([*target.sources, *field.sources])
        merged_artifacts = _dedup_preserve_order([*target.artifact_ids, *field.artifact_ids])
        merged_status = compute_weakest_status(
            merged_sources, note_store, field_name=matched_target
        )
        surviving[matched_target] = target.model_copy(
            update={
                "sources": merged_sources,
                "artifact_ids": merged_artifacts,
                "status": merged_status,
                "influence": max(target.influence, field.influence),
                "voi": max(target.voi, field.voi),
                "requested_under": min(target.requested_under, field.requested_under),
                "last_updated_version": max(
                    target.last_updated_version, field.last_updated_version
                ),
                "stale_rounds": 0,
            }
        )
        deduplicated.append(fname)
        alias_to_target[fname] = matched_target

    return surviving, deduplicated, alias_to_target


def archive_low_influence_fields(
    board: InvestigationBoard,
    *,
    updated_fields: Collection[str] = (),
    protected_fields: Collection[str] = (),
    advance_round: bool = True,
    archive_influence_threshold: float = 0.02,
    archive_min_stale_rounds: int = 2,
    max_active_fields: int = 12,
) -> tuple[InvestigationBoard, list[str]]:
    """Enforce Invariant I7: archive stale low-influence fields and cap active board size."""
    protected_set = frozenset(protected_fields)
    updated_set = frozenset(updated_fields)
    new_fields: dict[str, BoardField] = {
        k: v.model_copy(deep=True) for k, v in board.fields.items()
    }
    archived_all = list(board.archived_fields)
    newly_archived: list[str] = []

    if advance_round:
        for fname, fld in list(new_fields.items()):
            if fld.status == "unknown":
                continue
            if fname in updated_set:
                new_fields[fname] = fld.model_copy(update={"stale_rounds": 0})
            elif fld.influence < archive_influence_threshold:
                new_fields[fname] = fld.model_copy(
                    update={"stale_rounds": fld.stale_rounds + 1}
                )
            else:
                new_fields[fname] = fld.model_copy(update={"stale_rounds": 0})

    for fname, fld in list(new_fields.items()):
        if fld.status == "unknown":
            continue
        if fname in protected_set:
            continue
        if (
            fld.influence < archive_influence_threshold
            and fld.stale_rounds >= archive_min_stale_rounds
        ):
            del new_fields[fname]
            newly_archived.append(fname)
            if fname not in archived_all:
                archived_all.append(fname)

    while len(new_fields) > max_active_fields:
        non_unknown_candidates = [
            (fname, fld)
            for fname, fld in new_fields.items()
            if fld.status != "unknown"
        ]
        if non_unknown_candidates:
            non_unknown_candidates.sort(
                key=lambda item: (
                    item[0] in protected_set,
                    item[1].influence,
                    item[1].voi,
                    item[1].last_updated_version,
                    item[0],
                )
            )
            victim_name = non_unknown_candidates[0][0]
        else:
            evictable_unknown = [
                (fname, fld)
                for fname, fld in new_fields.items()
                if fname not in protected_set
                and fld.voi < archive_influence_threshold
            ]
            if not evictable_unknown:
                break
            evictable_unknown.sort(
                key=lambda item: (
                    item[1].voi,
                    item[1].last_updated_version,
                    item[0],
                )
            )
            victim_name = evictable_unknown[0][0]

        del new_fields[victim_name]
        newly_archived.append(victim_name)
        if victim_name not in archived_all:
            archived_all.append(victim_name)

    updated_board = board.model_copy(
        update={
            "fields": new_fields,
            "archived_fields": archived_all,
        }
    )
    return updated_board, newly_archived


def _extract_raw_updates(patch: BoardPatch | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(patch, BoardPatch):
        return dict(patch.updates), patch.rationale
    if not isinstance(patch, dict):
        raise PatchValidationError(
            "I1", None, f"Patch must be a BoardPatch or dict, got {type(patch).__name__}."
        )
    if "updates" in patch and isinstance(patch["updates"], dict):
        return dict(patch["updates"]), str(patch.get("rationale", ""))
    if "fields" in patch and isinstance(patch["fields"], dict):
        return dict(patch["fields"]), str(patch.get("rationale", ""))

    updates = {
        k: v
        for k, v in patch.items()
        if not (k == "rationale" and isinstance(v, str))
    }
    rationale = str(patch.get("rationale", "")) if isinstance(patch.get("rationale"), str) else ""
    return updates, rationale


def _coerce_field_patch(field_name: str, raw: Any) -> FieldPatch:
    if isinstance(raw, FieldPatch):
        return raw
    if not isinstance(raw, dict):
        raise PatchValidationError(
            "I1",
            field_name,
            f"Field patch for '{field_name}' must be a FieldPatch or dict.",
        )

    data = dict(raw)
    if "sources" in data:
        if not isinstance(data["sources"], list) or any(
            not isinstance(s, str) for s in data["sources"]
        ):
            raise PatchValidationError(
                "I1",
                field_name,
                "FieldPatch.sources must be a list of note_id strings.",
            )
    if "artifact_ids" in data:
        if not isinstance(data["artifact_ids"], list) or any(
            not isinstance(a, str) for a in data["artifact_ids"]
        ):
            raise PatchValidationError(
                "I1",
                field_name,
                "FieldPatch.artifact_ids must be a list of artifact_id strings.",
            )
    if "status" in data and data["status"] is not None:
        if not isinstance(data["status"], str):
            raise PatchValidationError(
                "I2", field_name, f"Invalid certification status: {data['status']!r}"
            )
        norm_status = data["status"].strip().lower()
        if norm_status not in _STATUS_RANK:
            raise PatchValidationError(
                "I2", field_name, f"Unrecognized certification status: {data['status']!r}"
            )
        data["status"] = norm_status

    try:
        return FieldPatch.model_validate(data)
    except ValidationError as exc:
        raise PatchValidationError("I1", field_name, f"Invalid FieldPatch schema: {exc}") from exc


def validate_and_merge_patch(
    board: InvestigationBoard,
    patch: BoardPatch | dict[str, Any],
    note_store: NoteStore,
    *,
    protected_fields: Collection[str] = (),
    seed_fields: Sequence[str] = (),
    note_extractor: NoteExtractorFn | None = None,
    strict: bool = True,
    advance_round: bool = True,
    archive_influence_threshold: float = 0.02,
    archive_min_stale_rounds: int = 2,
    max_active_fields: int = 12,
    verify_artifact_files: bool = True,
    compute_drift: bool = True,
) -> tuple[InvestigationBoard, PatchReport]:
    """Validate and merge a BoardPatch into InvestigationBoard enforcing Invariants I1–I7."""
    raw_updates, _ = _extract_raw_updates(patch)
    eff_protected = frozenset(protected_fields) | frozenset(seed_fields)

    working_fields: dict[str, BoardField] = {
        k: v.model_copy(deep=True) for k, v in board.fields.items()
    }
    pre_existing_names = set(working_fields.keys())
    working_archived = list(board.archived_fields)

    applied_fields: list[str] = []
    stale_fields: list[str] = []
    rejected_fields: dict[str, str] = {}
    newly_added_names: set[str] = set()

    provisional_version = board.board_version + 1

    for field_name, raw_fp in raw_updates.items():
        try:
            if not isinstance(field_name, str) or not field_name.strip():
                raise PatchValidationError("I1", str(field_name), "Field name must be non-empty.")

            fp = _coerce_field_patch(field_name, raw_fp)
            norm_val = _normalize_patch_value(fp.value)

            if any(not s.strip() for s in fp.sources):
                raise PatchValidationError(
                    "I1", field_name, "Empty string note_id in sources is not allowed."
                )
            patch_sources = _dedup_preserve_order(fp.sources)
            existing = working_fields.get(field_name)

            if norm_val == "unknown":
                if patch_sources or fp.artifact_ids:
                    raise PatchValidationError(
                        "I1",
                        field_name,
                        "Field with value='unknown' must have empty sources and artifact_ids.",
                    )
                if fp.status is not None and fp.status != "unknown":
                    raise PatchValidationError(
                        "I2",
                        field_name,
                        f"Illegal status '{fp.status}' for 'unknown' field with empty sources.",
                    )
                if existing is not None and existing.status == "certified":
                    raise PatchValidationError(
                        "I1",
                        field_name,
                        "Cannot overwrite an already 'certified' field back to 'unknown'.",
                    )

                eff_req = (
                    fp.requested_under
                    if fp.requested_under is not None
                    else board.board_version
                )
                if eff_req < 0:
                    raise PatchValidationError(
                        "I5", field_name, "requested_under must be >= 0."
                    )
                if board.board_version > eff_req:
                    stale_fields.append(field_name)

                working_fields[field_name] = BoardField(
                    value="unknown",
                    status="unknown",
                    sources=[],
                    artifact_ids=[],
                    influence=0.0,
                    voi=existing.voi if existing is not None else 0.0,
                    requested_under=eff_req,
                    last_updated_version=provisional_version,
                    stale_rounds=0,
                )
                if field_name in working_archived:
                    working_archived.remove(field_name)
                applied_fields.append(field_name)
                continue

            if not patch_sources:
                raise PatchValidationError(
                    "I1",
                    field_name,
                    "Non-unknown field must cite at least one note_id in sources.",
                )

            for nid in patch_sources:
                if not note_store.exists(nid):
                    raise PatchValidationError(
                        "I1",
                        field_name,
                        f"Cited note_id '{nid}' does not exist in NoteStore.",
                    )

            cited_notes = [note_store.get(nid) for nid in patch_sources]

            patch_status = compute_weakest_status(
                patch_sources, note_store, field_name=field_name
            )
            if fp.status is not None and _STATUS_RANK[fp.status] > _STATUS_RANK[patch_status]:
                raise PatchValidationError(
                    "I2",
                    field_name,
                    f"Illegal certification upgrade: requested '{fp.status}' "
                    f"but cited sources only support '{patch_status}'.",
                )

            validated_artifacts = _validate_and_collect_artifacts(
                fp.artifact_ids,
                cited_notes,
                note_store,
                field_name=field_name,
                verify_artifact_files=verify_artifact_files,
            )

            if patch_status == "certified":
                if not value_is_grounded_in_notes(norm_val, cited_notes):
                    raise PatchValidationError(
                        "I1",
                        field_name,
                        f"Ungrounded value {norm_val!r} for certified note(s) {patch_sources}.",
                    )

            if fp.requested_under is not None:
                if fp.requested_under < 0:
                    raise PatchValidationError(
                        "I5", field_name, "requested_under must be >= 0."
                    )
                eff_req = fp.requested_under
            else:
                eff_req = min(n.requested_under for n in cited_notes)

            if board.board_version > eff_req:
                stale_fields.append(field_name)

            if existing is None or existing.status == "unknown":
                merged_sources = patch_sources
                merged_artifacts = validated_artifacts
                merged_status = patch_status
                merged_influence = 0.0
                merged_voi = 0.0 if merged_status == "certified" else existing.voi if existing else 0.0
            elif _values_equal(norm_val, existing.value):
                merged_sources = _dedup_preserve_order([*existing.sources, *patch_sources])
                merged_artifacts = _dedup_preserve_order(
                    [*existing.artifact_ids, *validated_artifacts]
                )
                merged_status = compute_weakest_status(
                    merged_sources, note_store, field_name=field_name
                )
                merged_influence = existing.influence
                merged_voi = 0.0 if merged_status == "certified" else existing.voi
            elif existing.status == "certified":
                if patch_status != "certified":
                    raise PatchValidationError(
                        "I2",
                        field_name,
                        "Cannot overwrite a 'certified' field with a conflicting value "
                        "backed by uncertified sources.",
                    )
                merged_sources = _dedup_preserve_order([*existing.sources, *patch_sources])
                merged_artifacts = _dedup_preserve_order(
                    [*existing.artifact_ids, *validated_artifacts]
                )
                merged_status = compute_weakest_status(
                    merged_sources, note_store, field_name=field_name
                )
                merged_influence = existing.influence
                merged_voi = 0.0
            else:
                if patch_status == "certified":
                    merged_sources = patch_sources
                    merged_artifacts = validated_artifacts
                    merged_status = "certified"
                    merged_influence = existing.influence
                    merged_voi = 0.0
                else:
                    merged_sources = _dedup_preserve_order([*existing.sources, *patch_sources])
                    merged_artifacts = _dedup_preserve_order(
                        [*existing.artifact_ids, *validated_artifacts]
                    )
                    merged_status = compute_weakest_status(
                        merged_sources, note_store, field_name=field_name
                    )
                    merged_influence = existing.influence
                    merged_voi = existing.voi

            working_fields[field_name] = BoardField(
                value=norm_val,
                status=merged_status,
                sources=merged_sources,
                artifact_ids=merged_artifacts,
                influence=merged_influence,
                voi=merged_voi,
                requested_under=eff_req,
                last_updated_version=provisional_version,
                stale_rounds=0,
            )
            if field_name not in pre_existing_names:
                newly_added_names.add(field_name)
            if field_name in working_archived:
                working_archived.remove(field_name)
            applied_fields.append(field_name)

        except PatchValidationError as exc:
            if strict:
                raise
            rejected_fields[str(field_name)] = str(exc)

    deduped_fields, deduplicated_aliases, alias_to_target = deduplicate_fields(
        working_fields,
        note_store,
        candidate_names=newly_added_names,
        protected_fields=eff_protected,
    )

    updated_targets: set[str] = {
        alias_to_target.get(fname, fname) for fname in applied_fields
    }

    intermediate_board = board.model_copy(
        update={
            "fields": deduped_fields,
            "archived_fields": working_archived,
        }
    )

    final_board, archived_now = archive_low_influence_fields(
        intermediate_board,
        updated_fields=updated_targets,
        protected_fields=eff_protected,
        advance_round=advance_round,
        archive_influence_threshold=archive_influence_threshold,
        archive_min_stale_rounds=archive_min_stale_rounds,
        max_active_fields=max_active_fields,
    )

    mutated = bool(applied_fields or deduplicated_aliases or archived_now)
    final_version = board.board_version + 1 if mutated else board.board_version
    final_board = final_board.model_copy(update={"board_version": final_version})

    drift = 0.0
    if compute_drift:
        rebuilt = rebuild_board_from_notes(
            note_store,
            seed_fields=seed_fields,
            protected_fields=eff_protected,
            note_extractor=note_extractor,
            verify_artifact_files=verify_artifact_files,
        )
        drift = compute_board_drift(final_board, rebuilt)

    report = PatchReport(
        board_version=final_version,
        applied_fields=applied_fields,
        deduplicated_fields=deduplicated_aliases,
        stale_fields=_dedup_preserve_order(stale_fields),
        archived_fields=archived_now,
        rejected_fields=rejected_fields,
        board_drift=drift,
    )
    return final_board, report


def _extract_generic_note_value(note: Note) -> Any:
    if note.row_count == 0:
        return "none"

    ev = note.evidence
    rows = ev.get("rows")
    if isinstance(rows, list) and rows:
        first_row = rows[0]
        if isinstance(first_row, dict):
            for val in first_row.values():
                if isinstance(val, (str, int, float, bool)) and val is not None:
                    return val
        elif isinstance(first_row, (list, tuple)) and first_row:
            for val in first_row:
                if isinstance(val, (str, int, float, bool)) and val is not None:
                    return val
        elif isinstance(first_row, (str, int, float, bool)):
            return first_row

    candidates = ev.get("candidates")
    if isinstance(candidates, list) and candidates:
        first_cand = candidates[0]
        if isinstance(first_cand, dict):
            for val in first_cand.values():
                if isinstance(val, (str, int, float, bool)) and val is not None:
                    return val

    for key, val in ev.items():
        if key in _METADATA_EVIDENCE_KEYS:
            continue
        if isinstance(val, (str, int, float, bool)) and val is not None:
            return val
        if isinstance(val, list) and val and isinstance(val[0], (str, int, float, bool)):
            return val[0] if len(val) == 1 else val

    return note.summary if note.summary else note.row_count


def _merge_note_field_update(
    updates: dict[str, FieldPatch],
    field_cert: dict[str, bool],
    *,
    field_name: str,
    value: Any,
    note: Note,
    valid_art_ids: list[str],
) -> None:
    is_cert = note.certification == "CERTIFIED"
    prev = updates.get(field_name)
    if prev is None:
        updates[field_name] = FieldPatch(
            value=value,
            sources=[note.note_id],
            artifact_ids=valid_art_ids,
            requested_under=note.requested_under,
        )
        field_cert[field_name] = is_cert
        return

    prev_cert = field_cert[field_name]
    if is_cert and not prev_cert:
        updates[field_name] = FieldPatch(
            value=value,
            sources=[note.note_id],
            artifact_ids=valid_art_ids,
            requested_under=note.requested_under,
        )
        field_cert[field_name] = True
    elif is_cert == prev_cert and _values_equal(value, prev.value):
        updates[field_name] = FieldPatch(
            value=value,
            sources=_dedup_preserve_order([*prev.sources, note.note_id]),
            artifact_ids=_dedup_preserve_order([*prev.artifact_ids, *valid_art_ids]),
            requested_under=min(
                prev.requested_under
                if prev.requested_under is not None
                else note.requested_under,
                note.requested_under,
            ),
        )
    elif is_cert and prev_cert:
        updates[field_name] = FieldPatch(
            value=value,
            sources=_dedup_preserve_order([*prev.sources, note.note_id]),
            artifact_ids=_dedup_preserve_order([*prev.artifact_ids, *valid_art_ids]),
            requested_under=note.requested_under,
        )


def deterministic_patch_from_notes(
    notes: Sequence[Note],
    *,
    note_extractor: NoteExtractorFn | None = None,
) -> BoardPatch:
    """Deterministically map immutable Notes into FieldPatches (zero LLM calls)."""
    updates: dict[str, FieldPatch] = {}
    field_cert: dict[str, bool] = {}

    for note in notes:
        valid_art_ids = [
            art.artifact_id
            for art in note.artifacts
            if art.produced_by_verifier and bool(art.sha256 and art.sha256.strip())
        ]
        extracted = note_extractor(note) if note_extractor is not None else None
        if extracted:
            for fname, val in extracted.items():
                _merge_note_field_update(
                    updates,
                    field_cert,
                    field_name=fname,
                    value=val,
                    note=note,
                    valid_art_ids=valid_art_ids,
                )
        else:
            base_name = f"note_{note.hypothesis_family or note.verifier_name or note.note_id}"
            val = _extract_generic_note_value(note)
            fname = base_name
            if fname in updates and not _values_equal(val, updates[fname].value):
                fname = f"note_{note.note_id}"
            _merge_note_field_update(
                updates,
                field_cert,
                field_name=fname,
                value=val,
                note=note,
                valid_art_ids=valid_art_ids,
            )

    return BoardPatch(
        updates=updates,
        rationale="Deterministic replay from NoteStore",
    )


def rebuild_board_from_notes(
    note_store: NoteStore,
    decision_questions: Sequence[DecisionQuestion] | None = None,
    *,
    seed_fields: Sequence[str] = (),
    protected_fields: Collection[str] = (),
    note_extractor: NoteExtractorFn | None = None,
    verify_artifact_files: bool = False,
) -> InvestigationBoard:
    """Reconstruct the canonical InvestigationBoard from immutable NoteStore records alone (I3)."""
    board = create_initial_board(decision_questions or (), seed_fields=seed_fields)
    notes = note_store.snapshot()
    if not notes:
        return board

    patch = deterministic_patch_from_notes(notes, note_extractor=note_extractor)
    if not patch.updates:
        return board

    rebuilt, _ = validate_and_merge_patch(
        board,
        patch,
        note_store,
        protected_fields=frozenset(protected_fields) | frozenset(seed_fields),
        seed_fields=seed_fields,
        note_extractor=note_extractor,
        strict=False,
        advance_round=False,
        verify_artifact_files=verify_artifact_files,
        compute_drift=False,
    )
    return rebuilt


def _values_compatible(patched_val: Any, rebuilt_val: Any) -> bool:
    if _values_equal(patched_val, rebuilt_val):
        return True

    patched_strs: set[str] = set()
    patched_nums: list[float] = []
    patched_bools: set[bool] = set()
    _extract_leaves(
        patched_val,
        str_leaves=patched_strs,
        num_leaves=patched_nums,
        bool_leaves=patched_bools,
    )

    if isinstance(rebuilt_val, str):
        r_low = rebuilt_val.strip().lower()
        return r_low in patched_strs

    if isinstance(rebuilt_val, list) and rebuilt_val:
        return all(
            isinstance(item, str) and item.strip().lower() in patched_strs
            for item in rebuilt_val
        )

    return False


def compute_board_drift(
    patched_board: InvestigationBoard,
    rebuilt_board: InvestigationBoard,
) -> float:
    """Compute normalized disagreement in `[0.0, 1.0]` between patched and rebuilt boards (I3)."""
    rebuilt_active: dict[str, BoardField] = {
        k: f for k, f in rebuilt_board.fields.items() if f.status != "unknown"
    }

    rebuilt_note_status: dict[str, FieldStatus] = {}
    for rf in rebuilt_active.values():
        for nid in rf.sources:
            prev = rebuilt_note_status.get(nid)
            if prev is None or _STATUS_RANK[rf.status] > _STATUS_RANK[prev]:
                rebuilt_note_status[nid] = rf.status

    active_patched: dict[str, BoardField] = {
        k: f
        for k, f in patched_board.fields.items()
        if f.status != "unknown" or f.sources or _normalize_patch_value(f.value) != "unknown"
    }

    if not active_patched:
        if not rebuilt_active or patched_board.archived_fields:
            return 0.0
        return 1.0

    disagreements: list[float] = []
    for fname, pf in active_patched.items():
        if pf.status == "unknown" or _normalize_patch_value(pf.value) == "unknown":
            disagreements.append(1.0)
            continue
        if not pf.sources:
            disagreements.append(1.0)
            continue
        if any(nid not in rebuilt_note_status for nid in pf.sources):
            if not rebuilt_board.archived_fields:
                disagreements.append(1.0)
                continue
            known_sources = [nid for nid in pf.sources if nid in rebuilt_note_status]
            if not known_sources:
                disagreements.append(0.0)
                continue
            expected_status: FieldStatus = (
                "certified"
                if all(rebuilt_note_status[nid] == "certified" for nid in known_sources)
                else "uncertified"
            )
        else:
            expected_status = (
                "certified"
                if all(rebuilt_note_status[nid] == "certified" for nid in pf.sources)
                else "uncertified"
            )

        if pf.status != expected_status:
            disagreements.append(1.0)
            continue

        if fname in rebuilt_active:
            rf = rebuilt_active[fname]
            if set(pf.sources) == set(rf.sources) and not _values_compatible(pf.value, rf.value):
                disagreements.append(1.0)
                continue
            if (
                pf.status == "certified"
                and rf.status == "certified"
                and not _values_compatible(pf.value, rf.value)
            ):
                if set(pf.sources).issuperset(set(rf.sources)):
                    disagreements.append(1.0)
                    continue

        disagreements.append(0.0)

    if not disagreements:
        return 0.0
    drift = sum(disagreements) / len(disagreements)
    return round(min(1.0, max(0.0, drift)), 6)


__all__ = [
    "BoardPatch",
    "FieldPatch",
    "NoteExtractorFn",
    "PatchReport",
    "PatchValidationError",
    "archive_low_influence_fields",
    "compute_board_drift",
    "compute_weakest_status",
    "create_initial_board",
    "deduplicate_fields",
    "deterministic_patch_from_notes",
    "rebuild_board_from_notes",
    "validate_and_merge_patch",
    "value_is_grounded_in_notes",
]
