"""Backend- and domain-agnostic Fusion layer for ARES InvestigationBoard state & loop."""

from fusion.loop import (
    LoopConfig,
    LoopRunResult,
    RoundTelemetry,
    TerminationReason,
    run_investigation_loop,
)
from fusion.patch import (
    BoardPatch,
    FieldPatch,
    NoteExtractorFn,
    PatchReport,
    PatchValidationError,
    archive_low_influence_fields,
    compute_board_drift,
    compute_weakest_status,
    create_initial_board,
    deduplicate_fields,
    deterministic_patch_from_notes,
    rebuild_board_from_notes,
    validate_and_merge_patch,
    value_is_grounded_in_notes,
)

__all__ = [
    "BoardPatch",
    "FieldPatch",
    "LoopConfig",
    "LoopRunResult",
    "NoteExtractorFn",
    "PatchReport",
    "PatchValidationError",
    "RoundTelemetry",
    "TerminationReason",
    "archive_low_influence_fields",
    "compute_board_drift",
    "compute_weakest_status",
    "create_initial_board",
    "deduplicate_fields",
    "deterministic_patch_from_notes",
    "rebuild_board_from_notes",
    "run_investigation_loop",
    "validate_and_merge_patch",
    "value_is_grounded_in_notes",
]
