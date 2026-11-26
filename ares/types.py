"""Core data contracts for the ARES v2 Pushdown Automaton & DAG engine."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

JoinMode = Literal["all", "any"]
TaskStatus = Literal[
    "pending",
    "running",
    "waiting",
    "done",
    "failed",
    "suspended",
    "cancelled",
]


class Usage(BaseModel):
    """Cumulative LLM token, call, and cost accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> Usage:
        """Return a new Usage summing ``self`` and ``other``."""
        prompt = self.prompt_tokens + other.prompt_tokens
        completion = self.completion_tokens + other.completion_tokens
        self_total = self.total_tokens or (self.prompt_tokens + self.completion_tokens)
        other_total = other.total_tokens or (other.prompt_tokens + other.completion_tokens)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=self_total + other_total,
            llm_calls=self.llm_calls + other.llm_calls,
            cost_usd=round(self.cost_usd + other.cost_usd, 10),
        )

    def __add__(self, other: Usage) -> Usage:
        return self.add(other)


class ArtifactRef(BaseModel, frozen=True):
    """Immutable reference to a content-addressed artifact stored in ArtifactStore."""

    ref_id: str
    sha256: str
    byte_size: int
    mime_type: str = "application/octet-stream"
    preview: str = ""
    path: str | None = None


class TaskSpec(BaseModel):
    """Specification used to enqueue a root, child, or continuation task."""

    id: str | None = None
    state: str
    input: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    dedup_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Runtime execution frame for a task in the PDA + Dependency DAG."""

    id: str
    state: str
    input: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    depth: int = 0
    status: TaskStatus = "pending"
    output: Any = None
    error: str | None = None
    attempts: int = 0
    dedup_key: str | None = None
    join_mode: JoinMode = "all"
    continuation_id: str | None = None
    wait_for: list[str] = Field(default_factory=list)
    suspend_reason: str | None = None
    resume_key: str | None = None
    complete_on_resume: bool = False
    resume_data: Any = None
    spilled_artifact_id: str | None = None
    deterministic: bool = False
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class Done(BaseModel):
    """Transition indicating the current task completed and popped a result."""

    kind: Literal["done"] = "done"
    output: Any = None
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    usage: Usage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Spawn(BaseModel):
    """Transition pushing child tasks and an optional continuation onto the PDA."""

    kind: Literal["spawn"] = "spawn"
    tasks: list[TaskSpec] = Field(default_factory=list)
    then: TaskSpec | None = None
    join: JoinMode = "all"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Suspend(BaseModel):
    """Transition pausing the current task until external resume input arrives."""

    kind: Literal["suspend"] = "suspend"
    reason: str
    resume_key: str | None = None
    complete_on_resume: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Fail(BaseModel):
    """Transition indicating task failure, optionally retryable."""

    kind: Literal["fail"] = "fail"
    error: str
    retryable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


Transition = Annotated[Done | Spawn | Suspend | Fail, Field(discriminator="kind")]


class RunBudget(BaseModel):
    """Execution limits for a single StateMachine run or resumed session."""

    max_tasks: int = 500
    max_depth: int = 32
    max_wall_clock_s: float = 300.0
    max_inline_bytes: int | None = 4096
    max_tokens: int | None = None
    max_cost_usd: float | None = None


EventType = Literal[
    "task_enqueued",
    "task_started",
    "task_spawned",
    "task_done",
    "task_deduped",
    "task_retried",
    "task_suspended",
    "task_failed",
    "task_cancelled",
    "checkpoint_saved",
    "run_finished",
    "trace",
]


class Event(BaseModel):
    """Structured lifecycle event emitted during state machine execution."""

    seq: int
    timestamp: str
    event_type: EventType
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Final outcome of a StateMachine run or resume invocation."""

    run_id: str
    status: Literal["done", "suspended", "failed", "budget_exceeded"]
    root_task_id: str
    output: Any = None
    tasks: dict[str, Task]
    suspended_task_ids: list[str] = Field(default_factory=list)
    failed_task_ids: list[str] = Field(default_factory=list)
    cancelled_task_ids: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    run_dir: str | None = None
    wall_clock_ms: float = 0.0
