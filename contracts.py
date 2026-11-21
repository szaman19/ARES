"""Shared data contracts for the ARES Hackathon harness.

Defines the narrow-waist data types shared between:
- Domain plug-ins (`domain/`)
- ARES verifiers and execution runner
- Model providers (`gemini-4-hydrogen`, `gemini-4-carbon`, `gemini-decision-1`)
- Fusion board state & invariant enforcement (`fusion/patch.py`)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


CertificationStatus = Literal["CERTIFIED", "UNCERTIFIED"]
FieldStatus = Literal["certified", "uncertified", "unknown"]
CaseCategory = Literal["multi_clue_join", "look_alike_decoy", "under_determined"]
RecommendedAction = Literal["rollback", "restart", "scale_pool", "escalate"]
SubtletyLevel = Literal["low", "medium", "high"]
QuestionType = Literal["multiple_choice", "yes_no", "bounded_score"]
ArtifactKind = Literal["plot_png", "image", "table_slice"]


class ArtifactRef(BaseModel):
    """Reference to a deterministic visual or tabular artifact (Multimodal Addendum §1)."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    kind: ArtifactKind
    path: str
    sha256: str
    mime_type: str
    produced_by_verifier: bool


class BoardDecision(BaseModel):
    """Active closed-set decision state on the investigation board."""

    question_id: str
    question: str
    answer: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    certified_influence_share: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertified_influence_share: float = Field(default=0.0, ge=0.0, le=1.0)
    unknown_voi_max: float = Field(default=0.0, ge=0.0, le=1.0)


class BoardField(BaseModel):
    """Single tracked evidence field on the bounded investigation board."""

    value: Any = "unknown"
    status: FieldStatus = "unknown"
    sources: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    influence: float = Field(default=0.0, ge=0.0, le=1.0)
    voi: float = Field(default=0.0, ge=0.0, le=1.0)
    requested_under: int = Field(default=0, ge=0)
    last_updated_version: int = Field(default=0, ge=0)
    stale_rounds: int = Field(default=0, ge=0)


class InvestigationBoard(BaseModel):
    """Bounded versioned JSON investigation board (ARES Plan §5.1)."""

    board_version: int = Field(default=0, ge=0)
    decision: BoardDecision
    decisions: dict[str, BoardDecision] = Field(default_factory=dict)
    fields: dict[str, BoardField] = Field(default_factory=dict)
    archived_fields: list[str] = Field(default_factory=list)


class Note(BaseModel):
    """Immutable, append-only evidence record produced by ARES verifiers."""

    model_config = ConfigDict(frozen=True)

    note_id: str
    certification: CertificationStatus
    sources: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    verifier_name: str | None = None
    hypothesis_family: str = ""
    query_sql: str = ""
    query_hash: str = ""
    requested_under: int = Field(default=0, ge=0)
    summary: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    row_count: int = Field(default=0, ge=0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HypothesisQuerySpec(BaseModel):
    """Structured SQL hypothesis emitted by parallel swarm workers."""

    query_sql: str
    hypothesis_family: str
    expected_signal: str
    target_fields: list[str] = Field(default_factory=list)
    verifier_name: str | None = None
    requested_under: int = Field(default=0, ge=0)


class DecisionQuestion(BaseModel):
    """Closed-set question evaluated by gemini-decision-1."""

    question_id: str
    question: str
    options: list[str]
    question_type: QuestionType = "multiple_choice"


class NeedleSpec(BaseModel):
    """Specification for a planted incident needle (benchmark & live judge menu)."""

    case_id: str
    category: CaseCategory
    fault_type: str
    service: str
    upstream_service: str
    onset_ts: str
    window_seconds: int = Field(default=120, gt=0)
    subtlety: SubtletyLevel = "medium"
    description: str = ""
    recommended_action: RecommendedAction = "escalate"
    decoy_service: str | None = None
    decoy_upstream_service: str | None = None
    missing_clue: Literal["logs", "metrics", "traces"] | None = None
    seed: int = 0


class GroundTruth(BaseModel):
    """Ground-truth label and metadata for scoring a planted needle."""

    case_id: str
    category: CaseCategory
    root_cause_component: str | None
    recommended_action: RecommendedAction
    is_sufficiently_verified: bool
    expected_status: CertificationStatus
    pivotal_clues: list[str] = Field(default_factory=list)
    decoy_component: str | None = None
    onset_ts: str
    end_ts: str
    shard_id: str
    shared_trace_ids: list[str] = Field(default_factory=list)
    planted_files: list[str] = Field(default_factory=list)


class VerifierResult(BaseModel):
    """Deterministic output returned by a TRUSTED verifier."""

    verifier_name: str
    passed: bool
    certification: CertificationStatus
    evidence: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    row_count: int = Field(default=0, ge=0)
    summary: str = ""


@runtime_checkable
class VerifierRegistry(Protocol):
    """Registry interface for deterministic TRUSTED verifiers."""

    def register(
        self,
        name: str,
        fn: Callable[..., VerifierResult],
        *,
        description: str = "",
        trusted: bool = True,
    ) -> None:
        """Register a deterministic verifier function by name."""
        ...

    def get(self, name: str) -> Callable[..., VerifierResult]:
        """Retrieve a registered verifier by name."""
        ...

    def list_verifiers(self) -> list[str]:
        """List all registered verifier names."""
        ...


class InMemoryVerifierRegistry:
    """Simple concrete VerifierRegistry for standalone domain use and testing."""

    def __init__(self) -> None:
        self._verifiers: dict[str, Callable[..., VerifierResult]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., VerifierResult],
        *,
        description: str = "",
        trusted: bool = True,
    ) -> None:
        self._verifiers[name] = fn
        self._metadata[name] = {"description": description, "trusted": trusted}

    def get(self, name: str) -> Callable[..., VerifierResult]:
        if name not in self._verifiers:
            raise KeyError(f"Verifier not registered: {name}")
        return self._verifiers[name]

    def list_verifiers(self) -> list[str]:
        return sorted(self._verifiers.keys())

    def metadata(self, name: str) -> dict[str, Any]:
        return dict(self._metadata[name])


@runtime_checkable
class Domain(Protocol):
    """Domain-agnostic plug-in protocol (ARES Plan §6)."""

    name: str

    def generate(self, out_dir: str, size_gb: float, seed: int) -> None:
        """Write background haystack as partitioned Parquet shards."""
        ...

    def plant(self, out_dir: str, needle: NeedleSpec) -> GroundTruth:
        """Append a fresh Parquet shard containing the needle (used in benchmark & live demo)."""
        ...

    def needle_menu(self) -> list[NeedleSpec]:
        """Return selectable needle definitions for the live judge menu."""
        ...

    def schema_description(self) -> str:
        """Tables, columns, join keys, and semantics for Hydrogen/Carbon prompts."""
        ...

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        """Register deterministic TRUSTED DuckDB verifiers (no LLM calls)."""
        ...

    def decision_questions(self) -> list[DecisionQuestion]:
        """Closed-set questions the board must answer."""
        ...

    def hypothesis_families(self) -> list[str]:
        """Initial failure/signal families for parallel swarm exploration."""
        ...

    def score(self, board: dict[str, Any], truth: GroundTruth) -> dict[str, Any]:
        """Evaluate final board answer, certification accuracy, and candidate narrowing."""
        ...


__all__ = [
    "ArtifactKind",
    "ArtifactRef",
    "BoardDecision",
    "BoardField",
    "CaseCategory",
    "CertificationStatus",
    "DecisionQuestion",
    "Domain",
    "FieldStatus",
    "GroundTruth",
    "HypothesisQuerySpec",
    "InMemoryVerifierRegistry",
    "InvestigationBoard",
    "NeedleSpec",
    "Note",
    "QuestionType",
    "RecommendedAction",
    "SubtletyLevel",
    "VerifierRegistry",
    "VerifierResult",
]
