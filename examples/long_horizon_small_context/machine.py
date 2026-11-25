"""ARES PDA state machine vs rolling-context baseline on deep multi-hop chains."""

from __future__ import annotations

import json
import random
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ares import ArtifactStore, Done, RunBudget, Spawn, StateMachine, Task, TaskContext, TaskSpec
from ares.ext.llm import (
    ContextLengthExceededError,
    LLMClient,
    LLMMessage,
    LLMResponse,
    OpenAICompatClient,
)

DEFAULT_CANDIDATES: tuple[str, ...] = (
    "svc-alpha",
    "svc-beta",
    "svc-gamma",
    "svc-delta",
)

HOP_SYSTEM_PROMPT = (
    "Extract next_token or terminal status from the current log slice. "
    "Return JSON with next_token and terminal."
)

BASELINE_SYSTEM_PROMPT = (
    "Extract next_token or terminal status from the latest hop log. "
    "Return JSON with next_token and terminal."
)

_NEXT_TOKEN_RE = re.compile(r"\bnext=([A-Za-z0-9_-]+)")
_TERMINAL_RE = re.compile(r"\bstatus=(root_cause_verified|TERMINAL_BROKEN)\b")


class HopRecord(BaseModel):
    """Single deterministic hop observation in a generated multi-hop case."""

    candidate: str
    hop: int
    token: str
    log_text: str
    next_token: str | None = None
    terminal: str | None = None


class MultiHopCase(BaseModel):
    """Generated multi-hop causal chain benchmark dataset."""

    hops: int
    seed: int
    noise_lines: int
    include_decoy: bool
    candidates: list[str]
    true_candidate: str
    decoy_candidate: str | None = None
    entry_tokens: dict[str, str]
    records: dict[str, HopRecord]

    @staticmethod
    def _record_key(candidate: str, hop: int, token: str) -> str:
        return f"{candidate}:{hop}:{token}"

    def get_record(self, candidate: str, hop: int, token: str) -> HopRecord | None:
        """Look up the ground-truth record for ``(candidate, hop, token)``."""
        return self.records.get(self._record_key(candidate, hop, token))

    def get_log(self, candidate: str, hop: int, token: str) -> str | None:
        """Return the raw noisy log snippet for ``(candidate, hop, token)``."""
        rec = self.get_record(candidate, hop, token)
        return rec.log_text if rec is not None else None

    def verify_trail(self, candidate: str, trail: list[str]) -> tuple[bool, int]:
        """Deterministically verify start-to-end token linkage and final invariant."""
        if candidate not in self.candidates or not trail:
            return False, 0
        if trail[0] != self.entry_tokens.get(candidate):
            return False, 0

        verified_hops = 0
        for hop_idx, tok in enumerate(trail):
            rec = self.get_record(candidate, hop_idx, tok)
            if rec is None:
                return False, verified_hops
            if hop_idx < self.hops - 1:
                if hop_idx + 1 >= len(trail):
                    return False, verified_hops
                if rec.next_token != trail[hop_idx + 1]:
                    return False, verified_hops
                verified_hops += 1
            else:
                if rec.terminal == "root_cause_verified" and rec.next_token is None:
                    verified_hops += 1
                    return len(trail) == self.hops, verified_hops
                return False, verified_hops

        return False, verified_hops


def _make_token(candidate: str, hop: int, rng: random.Random) -> str:
    prefix = candidate.split("-", 1)[-1][:4].upper()
    return f"TOK-{prefix}-{hop:02d}-{rng.randrange(0x10000):04x}"


def _build_noisy_log(
    rng: random.Random,
    *,
    hop: int,
    noise_lines: int,
    next_token: str | None = None,
    terminal: str | None = None,
) -> str:
    noise_count = max(0, noise_lines - 1)
    lines = [f"n{idx:02d} ok id={rng.randrange(0x10000):04x}" for idx in range(noise_count)]
    if next_token is not None:
        signal = f"TRACE hop={hop} next={next_token}"
    else:
        signal = f"TRACE hop={hop} status={terminal or 'TERMINAL_BROKEN'}"
    insert_at = rng.randrange(len(lines) + 1)
    lines.insert(insert_at, signal)
    return "\n".join(lines)


def generate_multihop_case(
    hops: int,
    seed: int = 42,
    noise_lines: int = 25,
    include_decoy: bool = True,
) -> MultiHopCase:
    """Deterministically generate a multi-hop candidate investigation dataset.

    One true target candidate has a complete ``hops``-step causal chain ending in
    ``root_cause_verified``. When ``include_decoy`` is True, one look-alike decoy
    candidate chains through ``hops // 2`` (or ``hops - 1`` for short chains)
    before hitting ``TERMINAL_BROKEN``. Remaining candidates fail at hop 0 or 1.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if noise_lines < 1:
        raise ValueError("noise_lines must be >= 1")

    rng = random.Random(seed)
    candidates = list(DEFAULT_CANDIDATES)
    true_candidate = rng.choice(candidates)
    remaining = [c for c in candidates if c != true_candidate]
    decoy_candidate = rng.choice(remaining) if include_decoy else None
    early_fail_candidates = [c for c in remaining if c != decoy_candidate]

    entry_tokens: dict[str, str] = {}
    records: dict[str, HopRecord] = {}

    def add_record(
        candidate: str,
        hop: int,
        token: str,
        *,
        next_token: str | None = None,
        terminal: str | None = None,
    ) -> None:
        log_text = _build_noisy_log(
            rng,
            hop=hop,
            noise_lines=noise_lines,
            next_token=next_token,
            terminal=terminal,
        )
        key = MultiHopCase._record_key(candidate, hop, token)
        records[key] = HopRecord(
            candidate=candidate,
            hop=hop,
            token=token,
            log_text=log_text,
            next_token=next_token,
            terminal=terminal,
        )

    # True target candidate: complete H-hop chain.
    true_tokens = [_make_token(true_candidate, h, rng) for h in range(hops)]
    entry_tokens[true_candidate] = true_tokens[0]
    for h in range(hops):
        add_record(
            true_candidate,
            h,
            true_tokens[h],
            next_token=true_tokens[h + 1] if h < hops - 1 else None,
            terminal="root_cause_verified" if h == hops - 1 else None,
        )

    # Look-alike decoy candidate: partial chain ending with TERMINAL_BROKEN.
    if decoy_candidate is not None:
        decoy_stop = 0 if hops == 1 else (hops - 1 if hops <= 3 else max(1, hops // 2))
        decoy_tokens = [_make_token(decoy_candidate, h, rng) for h in range(decoy_stop + 1)]
        entry_tokens[decoy_candidate] = decoy_tokens[0]
        for h in range(decoy_stop + 1):
            is_stop = h == decoy_stop
            add_record(
                decoy_candidate,
                h,
                decoy_tokens[h],
                next_token=None if is_stop else decoy_tokens[h + 1],
                terminal="TERMINAL_BROKEN" if is_stop else None,
            )

    # Remaining candidates fail immediately at hop 0 or hop 1.
    for idx, candidate in enumerate(early_fail_candidates):
        fail_hop = 1 if (idx % 2 == 1 and hops >= 2) else 0
        toks = [_make_token(candidate, h, rng) for h in range(fail_hop + 1)]
        entry_tokens[candidate] = toks[0]
        for h in range(fail_hop + 1):
            is_stop = h == fail_hop
            add_record(
                candidate,
                h,
                toks[h],
                next_token=None if is_stop else toks[h + 1],
                terminal="TERMINAL_BROKEN" if is_stop else None,
            )

    return MultiHopCase(
        hops=hops,
        seed=seed,
        noise_lines=noise_lines,
        include_decoy=include_decoy,
        candidates=candidates,
        true_candidate=true_candidate,
        decoy_candidate=decoy_candidate,
        entry_tokens=entry_tokens,
        records=records,
    )


class HopExtraction(BaseModel):
    """Structured extraction output returned by the hop LLM call."""

    next_token: str | None = None
    terminal: str | None = None


class PromptTelemetry:
    """Thread-safe recorder of prompt character lengths across LLM calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lengths: list[int] = []

    def record(self, char_count: int) -> None:
        """Append a single prompt character length sample."""
        with self._lock:
            self._lengths.append(char_count)

    @property
    def lengths(self) -> list[int]:
        """Return a snapshot copy of recorded prompt lengths."""
        with self._lock:
            return list(self._lengths)

    @property
    def peak_prompt_chars(self) -> int:
        """Return the maximum recorded prompt character length (0 if unused)."""
        with self._lock:
            return max(self._lengths, default=0)

    @property
    def call_count(self) -> int:
        """Return total number of recorded LLM calls."""
        with self._lock:
            return len(self._lengths)


def _compute_prompt_chars(messages: Sequence[LLMMessage], system: str | None) -> int:
    return (len(system) if system else 0) + sum(len(msg.content) for msg in messages)


def _parse_extraction(response: LLMResponse) -> HopExtraction:
    if isinstance(response.parsed, HopExtraction):
        return response.parsed
    if isinstance(response.parsed, dict):
        return HopExtraction.model_validate(response.parsed)
    text = response.text.strip()
    if text:
        try:
            return HopExtraction.model_validate(json.loads(text))
        except Exception:
            pass
    next_match = _NEXT_TOKEN_RE.search(text)
    term_match = _TERMINAL_RE.search(text)
    return HopExtraction(
        next_token=next_match.group(1) if next_match else None,
        terminal=term_match.group(1) if term_match else None,
    )


class MockSmallContextLLM:
    """Deterministic offline LLMClient simulating a small-context extractor."""

    def __init__(self, *, prompt_char_cap: int | None = None) -> None:
        self.prompt_char_cap = prompt_char_cap
        self.telemetry = PromptTelemetry()

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Parse the latest user message's log slice and enforce optional cap."""
        _ = temperature, max_tokens
        prompt_chars = _compute_prompt_chars(messages, system)
        self.telemetry.record(prompt_chars)
        if self.prompt_char_cap is not None and prompt_chars > self.prompt_char_cap:
            raise ContextLengthExceededError(
                f"Prompt length {prompt_chars} chars exceeds cap {self.prompt_char_cap}"
            )

        latest = messages[-1].content if messages else ""
        next_match = _NEXT_TOKEN_RE.search(latest)
        term_match = _TERMINAL_RE.search(latest)
        extraction = HopExtraction(
            next_token=next_match.group(1) if next_match else None,
            terminal=term_match.group(1) if term_match else None,
        )
        text = extraction.model_dump_json()
        parsed = extraction if response_model is HopExtraction else None
        return LLMResponse(text=text, parsed=parsed)


class InstrumentedLLMClient:
    """Thread-safe LLMClient wrapper enforcing ``prompt_char_cap`` and recording sizes."""

    def __init__(self, inner: LLMClient, *, prompt_char_cap: int = 2000) -> None:
        self.inner = inner
        self.prompt_char_cap = prompt_char_cap
        self.telemetry = PromptTelemetry()

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        system: str | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Record prompt size, fail with ContextLengthExceededError on overflow, else delegate."""
        prompt_chars = _compute_prompt_chars(messages, system)
        self.telemetry.record(prompt_chars)
        if prompt_chars > self.prompt_char_cap:
            raise ContextLengthExceededError(
                f"Prompt length {prompt_chars} chars exceeds cap {self.prompt_char_cap}"
            )
        return self.inner.complete(
            messages,
            system=system,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def create_llm_client(
    *,
    mock: bool = True,
    base_url: str | None = None,
    model: str = "qwen2.5:3b",
    api_key: str | None = None,
    prompt_char_cap: int = 2000,
) -> InstrumentedLLMClient:
    """Build either a deterministic mock client or an ``OpenAICompatClient``."""
    if mock:
        return InstrumentedLLMClient(MockSmallContextLLM(), prompt_char_cap=prompt_char_cap)
    live = OpenAICompatClient(
        base_url=base_url or "http://localhost:11434/v1",
        model=model,
        api_key=api_key,
    )
    return InstrumentedLLMClient(live, prompt_char_cap=prompt_char_cap)


def _wrap_llm(llm: LLMClient | None, prompt_char_cap: int) -> InstrumentedLLMClient:
    if llm is None:
        return InstrumentedLLMClient(MockSmallContextLLM(), prompt_char_cap=prompt_char_cap)
    if isinstance(llm, InstrumentedLLMClient):
        llm.prompt_char_cap = prompt_char_cap
        return llm
    return InstrumentedLLMClient(llm, prompt_char_cap=prompt_char_cap)


def build_machine(
    case: MultiHopCase,
    llm: LLMClient | None = None,
    prompt_char_cap: int = 2000,
    max_inline_bytes: int = 600,
) -> StateMachine:
    """Build an ARES PDA ``StateMachine`` that keeps hop prompts O(1) bounded."""
    client = _wrap_llm(llm, prompt_char_cap)
    machine = StateMachine()
    machine.llm_client = client  # type: ignore[attr-defined]
    machine.telemetry = client.telemetry  # type: ignore[attr-defined]
    machine.max_inline_bytes = max_inline_bytes  # type: ignore[attr-defined]

    slice_limit = max(200, min(500, prompt_char_cap // 3))
    slice_step = max(100, slice_limit - 60)

    @machine.state("start")
    def start(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn:
        _ = task, deps, ctx
        return Spawn(
            tasks=[
                TaskSpec(
                    state="hop",
                    input={
                        "candidate": candidate,
                        "hop": 0,
                        "token": case.entry_tokens[candidate],
                        "trail": [case.entry_tokens[candidate]],
                    },
                )
                for candidate in case.candidates
            ],
            then=TaskSpec(state="select_verified"),
        )

    @machine.state("hop")
    def hop_state(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn:
        _ = deps
        candidate = str(task.input["candidate"])
        hop_idx = int(task.input["hop"])
        token = str(task.input["token"])
        trail = list(task.input.get("trail", [token]))

        record = case.get_record(candidate, hop_idx, token)
        if record is None:
            return Spawn(
                tasks=[],
                then=TaskSpec(
                    state="verify_chain",
                    input={
                        "candidate": candidate,
                        "trail": trail,
                        "claimed_terminal": "MISSING_RECORD",
                    },
                ),
            )

        # Store raw log out-of-line in ArtifactStore and read only bounded slices.
        ref = ctx.store.put_text(record.log_text, name=f"{candidate}_hop_{hop_idx}.log")
        total_len = len(record.log_text)
        extraction = HopExtraction()
        offset = 0
        while offset < total_len:
            log_slice = ctx.store.read_slice(ref.ref_id, offset=offset, limit=slice_limit)
            prompt = (
                f"Candidate: {candidate}\n"
                f"Hop: {hop_idx}/{case.hops - 1}\n"
                f"Token: {token}\n"
                f"Log slice:\n{log_slice}"
            )
            response = client.complete(
                [LLMMessage(role="user", content=prompt)],
                system=HOP_SYSTEM_PROMPT,
                response_model=HopExtraction,
                temperature=0.0,
                max_tokens=128,
            )
            extraction = _parse_extraction(response)
            if extraction.next_token or extraction.terminal:
                break
            if offset + slice_limit >= total_len:
                break
            offset += slice_step

        if extraction.terminal is not None or hop_idx >= case.hops - 1:
            return Spawn(
                tasks=[],
                then=TaskSpec(
                    state="verify_chain",
                    input={
                        "candidate": candidate,
                        "trail": trail,
                        "claimed_terminal": extraction.terminal or "END_OF_CHAIN",
                    },
                ),
            )

        if extraction.next_token:
            next_token = extraction.next_token.strip()
            return Spawn(
                tasks=[],
                then=TaskSpec(
                    state="hop",
                    input={
                        "candidate": candidate,
                        "hop": hop_idx + 1,
                        "token": next_token,
                        "trail": [*trail, next_token],
                    },
                ),
            )

        return Spawn(
            tasks=[],
            then=TaskSpec(
                state="verify_chain",
                input={
                    "candidate": candidate,
                    "trail": trail,
                    "claimed_terminal": "NO_TOKEN_FOUND",
                },
            ),
        )

    @machine.state("verify_chain", deterministic=True)
    def verify_chain(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = deps, ctx
        candidate = str(task.input["candidate"])
        trail = [str(tok) for tok in task.input.get("trail", [])]
        verified, hops_verified = case.verify_trail(candidate, trail)
        return Done(
            output={
                "candidate": candidate,
                "verified": verified,
                "hops_verified": hops_verified,
                "trail_len": len(trail),
            }
        )

    @machine.state("select_verified", deterministic=True)
    def select_verified(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = task
        results_by_candidate: dict[str, dict[str, Any]] = {}
        for dep_val in deps.values():
            ref_id = ArtifactStore.extract_ref_id(dep_val)
            payload = ctx.store.get_json(ref_id) if ref_id is not None else dep_val
            if isinstance(payload, dict) and "candidate" in payload:
                results_by_candidate[str(payload["candidate"])] = payload

        verified_candidates = [
            cand for cand, info in results_by_candidate.items() if info.get("verified") is True
        ]
        selected = verified_candidates[0] if len(verified_candidates) == 1 else "unverified"
        return Done(
            output={
                "selected": selected,
                "verified_candidates": verified_candidates,
                "results": results_by_candidate,
            }
        )

    return machine


class AresRunResult(BaseModel):
    """Structured outcome from running the ARES PDA state machine on a case."""

    selected: str
    correct: bool
    peak_prompt_chars: int
    prompt_lengths: list[int] = Field(default_factory=list)
    llm_calls: int
    total_tasks: int
    wall_clock_ms: float
    status: str
    verified_candidates: list[str] = Field(default_factory=list)


class BaselineRunResult(BaseModel):
    """Structured outcome from running the rolling-context baseline on a case."""

    selected: str | None = None
    correct: bool
    overflowed: bool
    peak_prompt_chars: int
    prompt_lengths: list[int] = Field(default_factory=list)
    llm_calls: int
    error: str | None = None


class BenchmarkSummary(BaseModel):
    """Side-by-side comparison metrics for ARES PDA vs Rolling Baseline."""

    hops: int
    seed: int
    prompt_char_cap: int
    true_candidate: str
    ares_selected: str
    ares_correct: bool
    peak_prompt_chars: int
    ares_llm_calls: int
    total_tasks: int
    ares_wall_clock_ms: float
    baseline_selected: str | None = None
    baseline_correct: bool
    baseline_overflowed: bool
    baseline_peak_prompt_chars: int
    baseline_error: str | None = None


def run_ares(
    case: MultiHopCase,
    *,
    prompt_char_cap: int = 2000,
    max_inline_bytes: int = 600,
    llm: LLMClient | None = None,
    run_dir: str | Path | None = None,
    max_concurrent: int = 4,
) -> AresRunResult:
    """Execute the ARES PDA state machine on ``case`` and collect prompt telemetry."""
    machine = build_machine(
        case,
        llm=llm,
        prompt_char_cap=prompt_char_cap,
        max_inline_bytes=max_inline_bytes,
    )
    budget = RunBudget(
        max_tasks=max(500, case.hops * 25),
        max_depth=max(32, case.hops + 10),
        max_inline_bytes=max_inline_bytes,
    )
    res = machine.run(
        TaskSpec(state="start"),
        run_dir=run_dir,
        max_concurrent=max_concurrent,
        budget=budget,
    )
    telemetry: PromptTelemetry = machine.telemetry  # type: ignore[attr-defined]
    out = res.output if isinstance(res.output, dict) else {}
    selected = str(out.get("selected", "unverified"))
    verified_candidates = [str(c) for c in out.get("verified_candidates", [])]
    return AresRunResult(
        selected=selected,
        correct=(res.status == "done" and selected == case.true_candidate),
        peak_prompt_chars=telemetry.peak_prompt_chars,
        prompt_lengths=telemetry.lengths,
        llm_calls=telemetry.call_count,
        total_tasks=len(res.tasks),
        wall_clock_ms=res.wall_clock_ms,
        status=res.status,
        verified_candidates=verified_candidates,
    )


def run_rolling_baseline(
    case: MultiHopCase,
    *,
    prompt_char_cap: int = 2000,
    llm: LLMClient | None = None,
) -> BaselineRunResult:
    """Simulate a rolling-transcript agent appending every hop's log observation.

    For each candidate, all prior hop logs and assistant turns stay in the same
    growing conversation, causing ``ContextLengthExceededError`` once the chain
    horizon exceeds ``prompt_char_cap``.
    """
    client = _wrap_llm(llm, prompt_char_cap)
    verified_candidates: list[str] = []

    for candidate in case.candidates:
        token = case.entry_tokens[candidate]
        trail = [token]
        messages: list[LLMMessage] = []

        for hop_idx in range(case.hops):
            record = case.get_record(candidate, hop_idx, token)
            if record is None:
                break

            messages.append(
                LLMMessage(
                    role="user",
                    content=f"[{candidate} hop={hop_idx} token={token}]\n{record.log_text}",
                )
            )
            try:
                response = client.complete(
                    messages,
                    system=BASELINE_SYSTEM_PROMPT,
                    response_model=HopExtraction,
                    temperature=0.0,
                    max_tokens=128,
                )
            except ContextLengthExceededError as exc:
                return BaselineRunResult(
                    selected=None,
                    correct=False,
                    overflowed=True,
                    peak_prompt_chars=client.telemetry.peak_prompt_chars,
                    prompt_lengths=client.telemetry.lengths,
                    llm_calls=client.telemetry.call_count,
                    error=str(exc),
                )

            messages.append(LLMMessage(role="assistant", content=response.text))
            extraction = _parse_extraction(response)
            if extraction.terminal is not None or hop_idx >= case.hops - 1:
                break
            if not extraction.next_token:
                break
            token = extraction.next_token.strip()
            trail.append(token)

        verified, _ = case.verify_trail(candidate, trail)
        if verified:
            verified_candidates.append(candidate)

    selected = verified_candidates[0] if len(verified_candidates) == 1 else "unverified"
    return BaselineRunResult(
        selected=selected,
        correct=(selected == case.true_candidate),
        overflowed=False,
        peak_prompt_chars=client.telemetry.peak_prompt_chars,
        prompt_lengths=client.telemetry.lengths,
        llm_calls=client.telemetry.call_count,
    )


def run_benchmark(
    hops: int,
    *,
    seed: int = 42,
    noise_lines: int = 25,
    include_decoy: bool = True,
    prompt_char_cap: int = 2000,
    max_inline_bytes: int = 600,
    llm_factory: Any = None,
) -> BenchmarkSummary:
    """Run both ARES PDA and the rolling-context baseline on the same case."""
    case = generate_multihop_case(
        hops=hops,
        seed=seed,
        noise_lines=noise_lines,
        include_decoy=include_decoy,
    )
    ares_llm = llm_factory() if llm_factory is not None else None
    baseline_llm = llm_factory() if llm_factory is not None else None

    ares_res = run_ares(
        case,
        prompt_char_cap=prompt_char_cap,
        max_inline_bytes=max_inline_bytes,
        llm=ares_llm,
    )
    baseline_res = run_rolling_baseline(
        case,
        prompt_char_cap=prompt_char_cap,
        llm=baseline_llm,
    )
    return BenchmarkSummary(
        hops=hops,
        seed=seed,
        prompt_char_cap=prompt_char_cap,
        true_candidate=case.true_candidate,
        ares_selected=ares_res.selected,
        ares_correct=ares_res.correct,
        peak_prompt_chars=ares_res.peak_prompt_chars,
        ares_llm_calls=ares_res.llm_calls,
        total_tasks=ares_res.total_tasks,
        ares_wall_clock_ms=ares_res.wall_clock_ms,
        baseline_selected=baseline_res.selected,
        baseline_correct=baseline_res.correct,
        baseline_overflowed=baseline_res.overflowed,
        baseline_peak_prompt_chars=baseline_res.peak_prompt_chars,
        baseline_error=baseline_res.error,
    )
