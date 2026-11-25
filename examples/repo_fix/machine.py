"""Bounded-context multi-file repo fix PDA StateMachine using GitWorktreeSandbox."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ares import (
    ArtifactStore,
    Done,
    Fail,
    Spawn,
    StateMachine,
    Task,
    TaskContext,
    TaskSpec,
)
from ares.llm import LLMMessage, OpenAICompatClient
from ares.sandbox import GitWorktreeSandbox

_FAILURE_INLINE_LIMIT = 500
_FAILURE_SLICE_LIMIT = 1000


class FileRelevance(BaseModel):
    """Structured relevance classification for one repository file."""

    relevant: bool
    reason: str = ""


class FileEdit(BaseModel):
    """Full-file replacement proposed for one repository path."""

    path: str
    content: str


class PatchProposal(BaseModel):
    """Structured multi-file patch emitted by a live LLM branch."""

    edits: list[FileEdit] = Field(default_factory=list)
    rationale: str = ""


def _unwrap_dep(val: Any, store: ArtifactStore) -> Any:
    if ArtifactStore.is_artifact_handle(val):
        ref_id = ArtifactStore.extract_ref_id(val)
        if ref_id is not None:
            return store.get_json(ref_id)
    return val


def _summarize_failure(stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    return combined[:_FAILURE_SLICE_LIMIT] if combined else "pytest failed with empty output"


def _offline_inspect(rel_path: str, file_slice: str, failure_slice: str) -> tuple[bool, str]:
    """Deterministic small-slice file classifier matching failing modules/symbols."""
    stem = Path(rel_path).stem
    if stem == "__init__":
        return False, "Package re-export only"
    lowered_fail = failure_slice.lower()
    if rel_path.lower() in lowered_fail or f"{stem}.py" in lowered_fail:
        return True, f"Directly referenced in failing pytest trace ({rel_path})"
    for line in file_slice.splitlines():
        if line.startswith("def "):
            fn_name = line[4:].split("(", 1)[0].strip()
            if fn_name and fn_name in failure_slice:
                return True, f"Defines failing symbol {fn_name}()"
    return False, "Unreferenced distractor or helper module"


def _offline_edits(branch_id: str, attempt: int, failure_summary: str) -> dict[str, str]:
    """Simulate realistic multi-branch repo repairs driven by pytest feedback."""
    fixed_invoice = '''"""Invoice total calculation combining line items, discounts, and tax."""

from __future__ import annotations

from billing.currency import round_cents
from billing.discounts import apply_discount


def compute_invoice_total(
    line_items: list[tuple[int, int]],
    *,
    discount_percent: float = 0.0,
    tax_rate: float = 0.0,
) -> int:
    """Compute final invoice total in cents from ``(unit_cents, quantity)`` pairs.

    Discount applies to the subtotal BEFORE sales tax is added.
    """
    if tax_rate < 0:
        raise ValueError("tax_rate must be non-negative")
    subtotal = sum(unit_cents * qty for unit_cents, qty in line_items)
    discounted = apply_discount(subtotal, discount_percent)
    if discounted == 0:
        return 0
    tax_cents = round_cents(discounted * tax_rate)
    return discounted + tax_cents
'''
    fixed_discounts = '''"""Discount calculation helpers."""

from __future__ import annotations

from billing.currency import round_cents


def apply_discount(amount_cents: int, discount_percent: float) -> int:
    """Apply a percentage discount in [0, 100] to ``amount_cents``.

    Negative percentages must raise ValueError; percentages over 100 clamp to 100%.
    """
    if discount_percent < 0:
        raise ValueError("discount_percent must be non-negative")
    clamped = min(float(discount_percent), 100.0)
    discount = round_cents(amount_cents * (clamped / 100.0))
    return max(0, amount_cents - discount)
'''
    still_broken_discounts = '''"""Discount calculation helpers."""

from __future__ import annotations

from billing.currency import round_cents


def apply_discount(amount_cents: int, discount_percent: float) -> int:
    """Incomplete attempt."""
    return amount_cents - round_cents(amount_cents * (discount_percent / 100.0))
'''

    if branch_id == "branch_0":
        if attempt == 1:
            # Attempt 1 fixes only invoice.py; pytest catches discounts.py bugs.
            return {"billing/invoice.py": fixed_invoice}
        if "apply_discount" in failure_summary or "ValueError" in failure_summary or attempt >= 2:
            return {
                "billing/invoice.py": fixed_invoice,
                "billing/discounts.py": fixed_discounts,
            }
        return {"billing/invoice.py": fixed_invoice}

    if branch_id == "branch_2":
        if attempt < 3:
            return {"billing/discounts.py": still_broken_discounts}
        return {
            "billing/invoice.py": fixed_invoice,
            "billing/discounts.py": fixed_discounts,
        }

    # branch_1: persistently wrong patch
    return {"billing/discounts.py": still_broken_discounts}


def build_machine(
    llm: OpenAICompatClient | None = None,
    *,
    max_attempts: int = 3,
    race: bool = True,
    max_file_slice_chars: int = 1200,
) -> StateMachine:
    """Construct the bounded-context multi-file ``repo_fix`` StateMachine.

    Callable with zero arguments for ``ares run`` CLI compatibility.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    machine = StateMachine()

    @machine.state("locate", deterministic=True)
    def locate(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done | Spawn | Fail:
        _ = deps
        repo_dir = str(task.input["repo_dir"])
        resolved_attempts = int(task.input.get("max_attempts", max_attempts))

        with GitWorktreeSandbox(repo_dir) as wt:
            baseline = wt.run_command([sys.executable, "-m", "pytest", "-q"])
            all_py = wt.list_files(extensions=[".py"])

        source_files = [p for p in all_py if not p.startswith("tests/")]
        if baseline.passed:
            return Done(
                output={
                    "certified": True,
                    "already_passing": True,
                    "patch_ref_id": None,
                    "source_files": source_files,
                }
            )

        failure_text = _summarize_failure(baseline.stdout, baseline.stderr)
        failure_ref = ctx.store.put_text(
            failure_text,
            mime_type="text/plain",
            name="baseline_pytest.log",
        )
        return Spawn(
            tasks=[
                TaskSpec(
                    state="inspect_file",
                    input={
                        "repo_dir": repo_dir,
                        "rel_path": rel_path,
                        "failure_ref_id": failure_ref.ref_id,
                    },
                )
                for rel_path in source_files
            ],
            then=TaskSpec(
                state="plan_branches",
                input={
                    "repo_dir": repo_dir,
                    "failure_ref_id": failure_ref.ref_id,
                    "max_attempts": resolved_attempts,
                },
            ),
        )

    @machine.state("inspect_file")
    def inspect_file(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = deps
        repo_dir = str(task.input["repo_dir"])
        rel_path = str(task.input["rel_path"])
        failure_ref_id = str(task.input["failure_ref_id"])

        with GitWorktreeSandbox(repo_dir) as wt:
            full_text = wt.read_file(rel_path)

        file_ref = ctx.store.put_text(
            full_text,
            mime_type="text/x-python",
            name=Path(rel_path).name,
        )
        file_slice = ctx.store.read_slice(file_ref.ref_id, offset=0, limit=max_file_slice_chars)
        failure_slice = ctx.store.read_slice(failure_ref_id, offset=0, limit=_FAILURE_SLICE_LIMIT)

        if llm is None:
            relevant, reason = _offline_inspect(rel_path, file_slice, failure_slice)
            ctx.record_usage(
                prompt_tokens=max(1, len(file_slice) // 4),
                completion_tokens=16,
                llm_calls=1,
            )
        else:
            prompt = (
                f"Failing pytest excerpt:\n{failure_slice}\n\n"
                f"Candidate file {rel_path} (slice):\n{file_slice}\n\n"
                "Is this file directly involved in the failing tests?"
            )
            resp = llm.complete(
                [LLMMessage(role="user", content=prompt)],
                system="Classify whether a single file slice is relevant to a test failure.",
                response_model=FileRelevance,
            )
            ctx.record_usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                llm_calls=1,
            )
            rel_obj = resp.parsed
            relevant = bool(rel_obj.relevant)
            reason = str(rel_obj.reason)

        return Done(
            output={
                "rel_path": rel_path,
                "relevant": relevant,
                "reason": reason,
                "file_ref_id": file_ref.ref_id,
            }
        )

    @machine.state("plan_branches")
    def plan_branches(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn | Fail:
        repo_dir = str(task.input["repo_dir"])
        failure_ref_id = str(task.input["failure_ref_id"])
        resolved_attempts = int(task.input.get("max_attempts", max_attempts))

        inspected = [_unwrap_dep(v, ctx.store) for v in deps.values()]
        inspected.sort(key=lambda item: str(item["rel_path"]))
        relevant_files = [
            {"rel_path": item["rel_path"], "file_ref_id": item["file_ref_id"]}
            for item in inspected
            if item.get("relevant")
        ]
        ignored_files = [
            item["rel_path"] for item in inspected if not item.get("relevant")
        ]
        if not relevant_files:
            relevant_files = [
                {"rel_path": item["rel_path"], "file_ref_id": item["file_ref_id"]}
                for item in inspected
            ]

        initial_failure = ctx.store.read_slice(failure_ref_id, offset=0, limit=_FAILURE_SLICE_LIMIT)
        failure_payload, _ = ctx.store.maybe_spill(initial_failure, _FAILURE_INLINE_LIMIT)

        branches = [
            TaskSpec(
                state="propose_patch",
                input={
                    "repo_dir": repo_dir,
                    "branch_id": f"branch_{idx}",
                    "attempt": 1,
                    "max_attempts": resolved_attempts,
                    "relevant_files": relevant_files,
                    "failure_summary": failure_payload,
                },
            )
            for idx in range(3)
        ]
        return Spawn(
            tasks=branches,
            then=TaskSpec(
                state="select_patch",
                input={
                    "repo_dir": repo_dir,
                    "relevant_files": [f["rel_path"] for f in relevant_files],
                    "ignored_files": ignored_files,
                },
            ),
            join="any" if race else "all",
        )

    @machine.state("propose_patch")
    def propose_patch(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn | Fail:
        _ = deps
        repo_dir = str(task.input["repo_dir"])
        branch_id = str(task.input["branch_id"])
        attempt = int(task.input["attempt"])
        branch_max_attempts = int(task.input.get("max_attempts", max_attempts))
        relevant_files: list[dict[str, str]] = list(task.input["relevant_files"])

        raw_failure = task.input.get("failure_summary")
        if ArtifactStore.is_artifact_handle(raw_failure):
            f_ref = ArtifactStore.extract_ref_id(raw_failure)
            assert f_ref is not None
            failure_summary = ctx.store.read_slice(f_ref, offset=0, limit=_FAILURE_SLICE_LIMIT)
        else:
            failure_summary = str(raw_failure or "")

        if llm is None:
            edits_map = _offline_edits(branch_id, attempt, failure_summary)
            prompt_chars = sum(
                len(ctx.store.read_slice(item["file_ref_id"], 0, max_file_slice_chars))
                for item in relevant_files
            ) + len(failure_summary)
            ctx.record_usage(
                prompt_tokens=max(1, prompt_chars // 4),
                completion_tokens=max(1, sum(len(v) for v in edits_map.values()) // 4),
                llm_calls=1,
            )
        else:
            file_blocks = "\n\n".join(
                f"--- {item['rel_path']} ---\n"
                f"{ctx.store.read_slice(item['file_ref_id'], 0, max_file_slice_chars)}"
                for item in relevant_files
            )
            prompt = (
                f"Branch: {branch_id} (attempt {attempt}/{branch_max_attempts})\n"
                f"Pytest failure:\n{failure_summary}\n\n"
                f"Relevant source files:\n{file_blocks}\n\n"
                "Return JSON edits fixing the failing tests."
            )
            resp = llm.complete(
                [LLMMessage(role="user", content=prompt)],
                system="You are a precise Python software engineer. Return minimal file edits.",
                response_model=PatchProposal,
            )
            ctx.record_usage(
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                llm_calls=1,
            )
            proposal: PatchProposal = resp.parsed
            edits_map = {edit.path: edit.content for edit in proposal.edits}

        with GitWorktreeSandbox(repo_dir) as wt:
            for rel_path, content in edits_map.items():
                wt.write_file(rel_path, content)
            diff_text = wt.diff()

        patch_ref = ctx.store.put_text(
            diff_text,
            mime_type="text/x-diff",
            name=f"{branch_id}_attempt_{attempt}.patch",
        )
        return Spawn(
            tasks=[
                TaskSpec(
                    state="verify_worktree",
                    # Invariant I4: verifier sees ONLY repo_dir + patch_ref_id.
                    input={"repo_dir": repo_dir, "patch_ref_id": patch_ref.ref_id},
                )
            ],
            then=TaskSpec(
                state="decide_repair",
                input={
                    "repo_dir": repo_dir,
                    "branch_id": branch_id,
                    "attempt": attempt,
                    "max_attempts": branch_max_attempts,
                    "patch_ref_id": patch_ref.ref_id,
                    "relevant_files": relevant_files,
                },
            ),
        )

    @machine.state("verify_worktree", deterministic=True)
    def verify_worktree(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        _ = deps
        repo_dir = str(task.input["repo_dir"])
        patch_ref_id = str(task.input["patch_ref_id"])
        patch_text = ctx.store.get_text(patch_ref_id)

        with GitWorktreeSandbox(repo_dir) as wt:
            if patch_text.strip():
                wt.apply_patch(patch_text)
            res = wt.run_command([sys.executable, "-m", "pytest", "-q"])

        full_log = f"returncode={res.returncode}\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"
        log_ref = ctx.store.put_text(
            full_log,
            mime_type="text/plain",
            name=f"verify_{patch_ref_id}.log",
        )
        return Done(
            output={
                "passed": res.passed,
                "returncode": res.returncode,
                "failure_summary": "" if res.passed else _summarize_failure(res.stdout, res.stderr),
                "patch_ref_id": patch_ref_id,
                "log_ref_id": log_ref.ref_id,
            }
        )

    @machine.state("decide_repair")
    def decide_repair(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done | Spawn | Fail:
        repo_dir = str(task.input["repo_dir"])
        branch_id = str(task.input["branch_id"])
        attempt = int(task.input["attempt"])
        branch_max_attempts = int(task.input.get("max_attempts", max_attempts))
        patch_ref_id = str(task.input["patch_ref_id"])
        relevant_files = list(task.input["relevant_files"])

        verify_out = _unwrap_dep(next(iter(deps.values())), ctx.store)
        if verify_out.get("passed"):
            return Done(
                output={
                    "certified": True,
                    "branch_id": branch_id,
                    "attempt": attempt,
                    "patch_ref_id": patch_ref_id,
                    "log_ref_id": verify_out.get("log_ref_id"),
                }
            )

        failure_summary = str(verify_out.get("failure_summary", ""))
        if attempt < branch_max_attempts:
            payload, _ = ctx.store.maybe_spill(failure_summary, _FAILURE_INLINE_LIMIT)
            return Spawn(
                tasks=[],
                then=TaskSpec(
                    state="propose_patch",
                    input={
                        "repo_dir": repo_dir,
                        "branch_id": branch_id,
                        "attempt": attempt + 1,
                        "max_attempts": branch_max_attempts,
                        "relevant_files": relevant_files,
                        "failure_summary": payload,
                    },
                ),
            )

        if race:
            return Fail(
                error=f"Branch {branch_id} exhausted {branch_max_attempts} attempts: {failure_summary[:200]}"
            )
        return Done(
            output={
                "certified": False,
                "branch_id": branch_id,
                "attempt": attempt,
                "patch_ref_id": patch_ref_id,
                "failure_summary": failure_summary,
                "log_ref_id": verify_out.get("log_ref_id"),
            }
        )

    @machine.state("select_patch")
    def select_patch(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        relevant_files = list(task.input.get("relevant_files", []))
        ignored_files = list(task.input.get("ignored_files", []))
        branch_results = [_unwrap_dep(v, ctx.store) for v in deps.values()]
        branch_results.sort(key=lambda item: str(item.get("branch_id", "")))

        certified = [b for b in branch_results if b.get("certified")]
        if not certified:
            return Done(
                output={
                    "certified": False,
                    "selected_branch": None,
                    "patch_ref_id": None,
                    "relevant_files": relevant_files,
                    "ignored_files": ignored_files,
                    "branches": branch_results,
                }
            )

        best = min(
            certified,
            key=lambda item: (int(item.get("attempt", 10**9)), str(item.get("branch_id", ""))),
        )
        return Done(
            output={
                "certified": True,
                "selected_branch": best["branch_id"],
                "attempts": best["attempt"],
                "patch_ref_id": best["patch_ref_id"],
                "relevant_files": relevant_files,
                "ignored_files": ignored_files,
                "branches": branch_results,
            }
        )

    return machine
