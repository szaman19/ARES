"""Shared narrow-waist data contracts and pluggable protocols for ARES.

Strictly modality-, backend-, and domain-agnostic:
- Core ARES primitives (`Note`, `ArtifactRef`, `HypothesisSpec`, `CostEstimate`,
  `VerifierResult`, `VerifierRegistry`, `ExecutionBackend`)
- Bounded InvestigationBoard contracts (`InvestigationBoard`, `BoardField`, `BoardDecision`)
- Pluggable `Domain` interface (`environment_description`, `create_backend`,
  `register_verifiers`, `decision_questions`, `pivotal_fields`, `required_signal_fields`)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


CertificationStatus = Literal["CERTIFIED", "UNCERTIFIED"]
FieldStatus = Literal["certified", "uncertified", "unknown"]
CaseCategory = Literal["multi_clue_join", "look_alike_decoy", "under_determined"]
SubtletyLevel = Literal["low", "medium", "high"]
QuestionType = Literal["multiple_choice", "yes_no", "bounded_score"]
ArtifactKind = Literal["plot_png", "image", "table_slice"]


class GuardValidationError(ValueError):
    """Raised when an automated magnet/query violates sandbox or cost root guards (Invariant I6)."""


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
    """Immutable, append-only evidence record produced by ARES queries or verifiers."""

    model_config = ConfigDict(frozen=True)

    note_id: str
    certification: CertificationStatus
    sources: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    verifier_name: str | None = None
    hypothesis_family: str = ""
    query: str = ""
    content_hash: str = ""
    requested_under: int = Field(default=0, ge=0)
    summary: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    row_count: int = Field(default=0, ge=0)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HypothesisSpec(BaseModel):
    """Backend-agnostic structured hypothesis magnet emitted by parallel swarm workers."""

    query: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    hypothesis_family: str = ""
    expected_signal: str = ""
    target_fields: list[str] = Field(default_factory=list)
    verifier_name: str | None = None
    requested_under: int = Field(default=0, ge=0)


class CostEstimate(BaseModel):
    """Backend-agnostic pre-execution cost estimate (rows, AST nodes, solver steps, bytes)."""

    estimated_units: int = Field(default=0, ge=0)
    estimated_bytes: int = Field(default=0, ge=0)
    plan_summary: str = ""
    within_budget: bool = True
    rejection_reason: str | None = None


class DecisionQuestion(BaseModel):
    """Closed-set question evaluated by `gemini-decision-1`."""

    question_id: str
    question: str
    options: list[str]
    question_type: QuestionType = "multiple_choice"
    instructions: str = ""
    option_criteria: dict[str, str] = Field(default_factory=dict)


class NeedleSpec(BaseModel):
    """Modality-agnostic specification for a planted benchmark needle."""

    case_id: str
    category: CaseCategory
    scenario_type: str
    target_entity: str
    decoy_entity: str | None = None
    missing_signal: str | None = None
    subtlety: SubtletyLevel = "medium"
    description: str = ""
    expected_decisions: dict[str, str] = Field(default_factory=dict)
    seed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundTruth(BaseModel):
    """Modality-agnostic ground-truth label and metadata for scoring a benchmark case."""

    case_id: str
    category: CaseCategory
    primary_target: str | None
    expected_decisions: dict[str, str] = Field(default_factory=dict)
    is_sufficiently_verified: bool
    expected_status: CertificationStatus
    pivotal_fields: list[str] = Field(default_factory=list)
    decoy_target: str | None = None
    planted_artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    """Concrete VerifierRegistry for standalone domain use and testing."""

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
class ExecutionBackend(Protocol):
    """Pluggable magnet & verifier execution adapter (tabular, AST, simulator, solver)."""

    def validate_query(self, query: str) -> str:
        """Validate read-only/sandbox safety under Invariant I6; raise GuardValidationError on failure."""
        ...

    def compute_content_hash(
        self,
        query: str,
        *,
        hypothesis_family: str = "",
        verifier_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Return a canonical SHA-256 content hash for deduplication (Invariant I6)."""
        ...

    def estimate_cost(self, query: str) -> CostEstimate:
        """Estimate execution cost before dispatch for budget enforcement & cheapest-first sorting."""
        ...

    def execute_query(
        self,
        query: str,
        *,
        params: dict[str, Any] | None = None,
        requested_under: int = 0,
        hypothesis_family: str = "",
        content_hash: str = "",
        note_store: Any | None = None,
    ) -> tuple[Any, Note]:
        """Execute an exploratory read-only magnet query, always emitting an UNCERTIFIED Note."""
        ...

    def execute_verifier(
        self,
        registry: VerifierRegistry,
        verifier_name: str,
        *,
        params: dict[str, Any] | None = None,
        requested_under: int = 0,
        hypothesis_family: str = "",
        query: str = "",
        content_hash: str = "",
        note_store: Any | None = None,
    ) -> tuple[VerifierResult, Note]:
        """Execute a deterministic verifier enforcing Invariant I4 and I2 taint discipline."""
        ...


@runtime_checkable
class Domain(Protocol):
    """Modality- and backend-agnostic domain plug-in protocol."""

    name: str

    def generate(self, out_dir: str, size_gb: float, seed: int) -> None:
        """Materialize a deterministic background haystack inside `out_dir`."""
        ...

    def plant(self, out_dir: str, needle: NeedleSpec) -> GroundTruth:
        """Plant a specific benchmark case into `out_dir` and return GroundTruth."""
        ...

    def needle_menu(self) -> list[NeedleSpec]:
        """Return selectable benchmark / live-demo needle definitions."""
        ...

    def environment_description(self) -> str:
        """Describe haystack structure, schemas, entities, and semantics for worker prompts."""
        ...

    def magnet_rules(self) -> str:
        """Return backend-specific syntax and safety rules for worker magnet authoring."""
        ...

    def create_backend(self, workspace_dir: str) -> ExecutionBackend | None:
        """Instantiate the domain's ExecutionBackend for `workspace_dir` (or None for pure-Python)."""
        ...

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        """Register deterministic TRUSTED verifiers (no LLM calls)."""
        ...

    def decision_questions(self) -> list[DecisionQuestion]:
        """Closed-set questions the investigation board must answer."""
        ...

    def hypothesis_families(self) -> list[str]:
        """Initial hypothesis/signal families for parallel swarm exploration."""
        ...

    def pivotal_fields(self) -> list[str]:
        """Canonical board field names seeded at version 0 and protected from eviction."""
        ...

    def required_signal_fields(self) -> tuple[str, ...]:
        """Independent signal fields that must all be certified to confirm a hypothesis."""
        ...

    def extract_field_patches_from_note(self, note: Note) -> dict[str, Any] | None:
        """Optional deterministic mapping from a verifier Note to `{field_name: value}`."""
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
    "CostEstimate",
    "DecisionQuestion",
    "Domain",
    "ExecutionBackend",
    "FieldStatus",
    "GroundTruth",
    "GuardValidationError",
    "HypothesisSpec",
    "InMemoryVerifierRegistry",
    "InvestigationBoard",
    "NeedleSpec",
    "Note",
    "QuestionType",
    "SubtletyLevel",
    "VerifierRegistry",
    "VerifierResult",
]
