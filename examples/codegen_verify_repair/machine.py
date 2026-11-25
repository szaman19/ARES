"""ARES PDA + DAG example: parallel codegen, isolated pytest verification, and bounded repair."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ares import ArtifactStore, Done, Fail, Spawn, StateMachine, Task, TaskContext, TaskSpec
from ares.ext.llm import LLMMessage, OpenAICompatClient

_PYTEST_TIMEOUT_S = 15
_FAILURE_SLICE_LIMIT = 2048
_FAILURE_INLINE_LIMIT = 1024
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CodingTask:
    """Problem specification and trusted verification suite for a coding task."""

    name: str
    spec_text: str
    trusted_tests: str


TASKS: dict[str, CodingTask] = {
    "slugify": CodingTask(
        name="slugify",
        spec_text=(
            "Implement `slugify(text: str) -> str`:\n"
            "- Normalize Unicode using NFKD and drop non-ASCII combining characters.\n"
            "- Lowercase the text.\n"
            "- Replace any run of non-ASCII-alphanumeric characters (`[^a-z0-9]+`) with `-`.\n"
            "- Strip leading and trailing `-` characters.\n"
            "- Return `\"\"` when no ASCII alphanumeric characters remain."
        ),
        trusted_tests='''\
from solution import slugify


def test_basic_words() -> None:
    assert slugify("Hello World") == "hello-world"


def test_punctuation_and_edges() -> None:
    assert slugify("  Hello, World!!  ") == "hello-world"
    assert slugify("foo---bar__baz") == "foo-bar-baz"


def test_unicode_accents() -> None:
    assert slugify("Café & Résumé") == "cafe-resume"
    assert slugify("naïve façade") == "naive-facade"


def test_digits_and_empty() -> None:
    assert slugify("ARES v2 Release 2026") == "ares-v2-release-2026"
    assert slugify("") == ""
    assert slugify("---!!!") == ""
''',
    ),
    "rle_encode": CodingTask(
        name="rle_encode",
        spec_text=(
            "Implement `rle_encode(text: str) -> list[tuple[str, int]]`:\n"
            "- Perform lossless run-length encoding over consecutive identical characters.\n"
            "- Return `(char, count)` tuples in left-to-right order.\n"
            "- Return `[]` for the empty string `\"\"`."
        ),
        trusted_tests='''\
from solution import rle_encode


def test_empty_and_single() -> None:
    assert rle_encode("") == []
    assert rle_encode("x") == [("x", 1)]


def test_consecutive_runs() -> None:
    assert rle_encode("aaabbc") == [("a", 3), ("b", 2), ("c", 1)]
    assert rle_encode("aba") == [("a", 1), ("b", 1), ("a", 1)]


def test_digits_and_spaces() -> None:
    assert rle_encode("111  2") == [("1", 3), (" ", 2), ("2", 1)]
''',
    ),
}


def _strip_python_fences(text: str) -> str:
    """Extract Python source from optional markdown code fences."""
    stripped = text.strip()
    match = _CODE_FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip() + "\n"
    return stripped + "\n"


def _offline_generate(
    task_name: str,
    branch_id: str,
    attempt: int,
    failure_summary: str,
) -> str:
    """Deterministic offline coder simulating 3 competing branch trajectories."""
    if task_name == "slugify":
        if branch_id == "branch_0":
            if attempt == 1:
                # Subtle edge-case bug: forgets NFKD transliteration and .strip("-").
                return (
                    "import re\n\n\n"
                    "def slugify(text: str) -> str:\n"
                    '    return re.sub(r"[^a-z0-9]+", "-", text.lower())\n'
                )
            # Repair attempt >= 2 guided by pytest failure summary.
            if failure_summary:
                return (
                    "import re\n"
                    "import unicodedata\n\n\n"
                    "def slugify(text: str) -> str:\n"
                    '    normalized = unicodedata.normalize("NFKD", text)\n'
                    '    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()\n'
                    '    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")\n'
                )
        if branch_id == "branch_1":
            # Persistently flawed across repair attempts (proves bounded failure).
            if attempt == 1:
                return (
                    "def slugify(text: str) -> str:\n"
                    '    return text.strip().lower().replace(" ", "-")\n'
                )
            if attempt == 2:
                return (
                    "def slugify(text: str) -> str:\n"
                    '    cleaned = text.lower().replace(",", "").replace("!", "")\n'
                    '    return "-".join(cleaned.split())\n'
                )
            return (
                "import re\n\n\n"
                "def slugify(text: str) -> str:\n"
                '    return re.sub(r"[^a-z0-9]+", "-", text.lower()).replace("_", "-")\n'
            )
        # branch_2: takes 3 attempts to converge (so branch_0 wins on attempt=2).
        if attempt == 1:
            return (
                "def slugify(text: str) -> str:\n"
                '    return text.lower().strip("- ")\n'
            )
        if attempt == 2:
            return (
                "import re\n\n\n"
                "def slugify(text: str) -> str:\n"
                '    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n'
            )
        return (
            "import re\n"
            "import unicodedata\n\n\n"
            "def slugify(text: str) -> str:\n"
            '    normalized = unicodedata.normalize("NFKD", text)\n'
            '    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()\n'
            '    return re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")\n'
        )

    if task_name == "rle_encode":
        if branch_id == "branch_0":
            if attempt == 1:
                # Buggy attempt 1: crashes on empty input and drops final run.
                return (
                    "def rle_encode(text: str) -> list[tuple[str, int]]:\n"
                    "    runs: list[tuple[str, int]] = []\n"
                    "    cur = text[0]\n"
                    "    count = 1\n"
                    "    for ch in text[1:]:\n"
                    "        if ch == cur:\n"
                    "            count += 1\n"
                    "        else:\n"
                    "            runs.append((cur, count))\n"
                    "            cur = ch\n"
                    "            count = 1\n"
                    "    return runs\n"
                )
            if failure_summary:
                return (
                    "def rle_encode(text: str) -> list[tuple[str, int]]:\n"
                    "    if not text:\n"
                    "        return []\n"
                    "    runs: list[tuple[str, int]] = []\n"
                    "    cur = text[0]\n"
                    "    count = 1\n"
                    "    for ch in text[1:]:\n"
                    "        if ch == cur:\n"
                    "            count += 1\n"
                    "        else:\n"
                    "            runs.append((cur, count))\n"
                    "            cur = ch\n"
                    "            count = 1\n"
                    "    runs.append((cur, count))\n"
                    "    return runs\n"
                )
        if branch_id == "branch_1":
            # Persistently wrong: counts total occurrences rather than consecutive runs.
            return (
                "from collections import Counter\n\n\n"
                "def rle_encode(text: str) -> list[tuple[str, int]]:\n"
                "    return list(Counter(text).items())\n"
            )
        # branch_2: takes 3 attempts to converge.
        if attempt == 1:
            return (
                "def rle_encode(text: str) -> list[tuple[str, int]]:\n"
                "    return [(ch, 1) for ch in text]\n"
            )
        if attempt == 2:
            return (
                "def rle_encode(text: str) -> list[tuple[str, int]]:\n"
                "    if not text:\n"
                "        return []\n"
                "    return [(text[0], len(text))]\n"
            )
        return (
            "def rle_encode(text: str) -> list[tuple[str, int]]:\n"
            "    if not text:\n"
            "        return []\n"
            "    runs: list[tuple[str, int]] = []\n"
            "    cur = text[0]\n"
            "    count = 1\n"
            "    for ch in text[1:]:\n"
            "        if ch == cur:\n"
            "            count += 1\n"
            "        else:\n"
            "            runs.append((cur, count))\n"
            "            cur = ch\n"
            "            count = 1\n"
            "    runs.append((cur, count))\n"
            "    return runs\n"
        )

    raise KeyError(f"Unknown coding task: {task_name!r}")


def _llm_generate(
    llm: OpenAICompatClient,
    coding_task: CodingTask,
    branch_id: str,
    attempt: int,
    failure_summary: str,
    prev_code: str | None,
) -> str:
    """Generate or repair Python module source via an OpenAICompatClient."""
    branch_temps = {"branch_0": 0.0, "branch_1": 0.2, "branch_2": 0.4}
    temperature = branch_temps.get(branch_id, 0.2)
    system = (
        "You are a Python 3.11+ engineer. Respond ONLY with valid Python source code "
        "for `solution.py`. Do not include tests or commentary."
    )
    if attempt == 1 or not failure_summary:
        prompt = f"Strategy branch: {branch_id}\n\nSpecification:\n{coding_task.spec_text}"
    else:
        prior_block = f"\n\nPrevious code:\n```python\n{prev_code}\n```" if prev_code else ""
        prompt = (
            f"Strategy branch: {branch_id} (repair attempt {attempt})\n\n"
            f"Specification:\n{coding_task.spec_text}"
            f"{prior_block}\n\n"
            f"Pytest failure output:\n{failure_summary}\n\n"
            "Fix all defects and return the complete corrected `solution.py`."
        )
    resp = llm.complete(
        [LLMMessage(role="user", content=prompt)],
        system=system,
        temperature=temperature,
    )
    return _strip_python_fences(resp.text)


def _unwrap_dep(value: Any, store: ArtifactStore) -> Any:
    """Resolve an ArtifactStore spill handle if a dependency output was spilled."""
    if ArtifactStore.is_artifact_handle(value):
        ref_id = ArtifactStore.extract_ref_id(value)
        if ref_id is not None:
            return store.get_json(ref_id)
    return value


def _summarize_pytest_failure(stdout: str, stderr: str) -> str:
    """Build a bounded failure summary from pytest stdout and stderr."""
    combined = "\n".join(part for part in (stdout.strip(), stderr.strip()) if part)
    if not combined:
        return "pytest failed with empty output"
    return combined[:_FAILURE_SLICE_LIMIT]


def build_machine(
    llm: OpenAICompatClient | None = None,
    max_attempts: int = 3,
    race: bool = False,
) -> StateMachine:
    """Construct the codegen-verify-repair ARES StateMachine.

    Callable with zero arguments for ``ares run`` CLI compatibility.
    When ``race=True``, uses ``Spawn(..., join="any")`` so the first branch to pass
    ``verify_pytest`` cancels remaining competing branches immediately.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    machine = StateMachine()

    @machine.state("plan")
    def plan(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn | Fail:
        _ = deps, ctx
        task_name = str(task.input.get("task", "slugify"))
        if task_name not in TASKS:
            return Fail(error=f"Unknown task {task_name!r}; expected one of {sorted(TASKS)}")
        resolved_max_attempts = int(task.input.get("max_attempts", max_attempts))
        if resolved_max_attempts < 1:
            return Fail(error=f"max_attempts must be >= 1, got {resolved_max_attempts}")

        branches = [
            TaskSpec(
                state="generate",
                input={
                    "task": task_name,
                    "branch_id": f"branch_{idx}",
                    "attempt": 1,
                    "max_attempts": resolved_max_attempts,
                },
            )
            for idx in range(3)
        ]
        return Spawn(
            tasks=branches,
            then=TaskSpec(state="select_best", input={"task": task_name}),
            join="any" if race else "all",
        )

    @machine.state("generate")
    def generate(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Spawn | Fail:
        _ = deps
        task_name = str(task.input["task"])
        if task_name not in TASKS:
            return Fail(error=f"Unknown task {task_name!r}")
        branch_id = str(task.input["branch_id"])
        attempt = int(task.input["attempt"])
        branch_max_attempts = int(task.input.get("max_attempts", max_attempts))

        raw_failure = task.input.get("failure_summary")
        if ArtifactStore.is_artifact_handle(raw_failure):
            failure_ref_id = ArtifactStore.extract_ref_id(raw_failure)
            assert failure_ref_id is not None
            failure_summary = ctx.store.read_slice(failure_ref_id, 0, _FAILURE_SLICE_LIMIT)
        elif raw_failure is None:
            failure_summary = ""
        else:
            failure_summary = str(raw_failure)

        prev_code_ref_id = task.input.get("prev_code_ref_id")
        prev_code = ctx.store.get_text(str(prev_code_ref_id)) if prev_code_ref_id else None

        if llm is None:
            source = _offline_generate(task_name, branch_id, attempt, failure_summary)
        else:
            source = _llm_generate(
                llm,
                TASKS[task_name],
                branch_id,
                attempt,
                failure_summary,
                prev_code,
            )

        code_ref = ctx.store.put_text(
            source,
            mime_type="text/x-python",
            name=f"{branch_id}_attempt_{attempt}.py",
        )
        return Spawn(
            tasks=[
                TaskSpec(
                    state="verify_pytest",
                    # Invariant I4: verifier receives ONLY task identifier and code ref.
                    input={"task": task_name, "code_ref_id": code_ref.ref_id},
                )
            ],
            then=TaskSpec(
                state="decide_repair",
                input={
                    "task": task_name,
                    "branch_id": branch_id,
                    "attempt": attempt,
                    "max_attempts": branch_max_attempts,
                    "code_ref_id": code_ref.ref_id,
                },
            ),
        )

    @machine.state("verify_pytest", deterministic=True)
    def verify_pytest(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done | Fail:
        _ = deps
        task_name = str(task.input["task"])
        if task_name not in TASKS:
            return Fail(error=f"Unknown task {task_name!r}")
        code_ref_id = str(task.input["code_ref_id"])
        candidate_source = ctx.store.get_text(code_ref_id)
        trusted_tests = TASKS[task_name].trusted_tests

        with tempfile.TemporaryDirectory(prefix="ares_verify_") as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "solution.py").write_text(candidate_source, encoding="utf-8")
            (tmp_path / "test_solution.py").write_text(trusted_tests, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", "test_solution.py"],
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=_PYTEST_TIMEOUT_S,
                )
                returncode = proc.returncode
                stdout = proc.stdout
                stderr = proc.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = -1
                stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = f"pytest timed out after {_PYTEST_TIMEOUT_S}s"

        full_log = f"returncode={returncode}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        log_ref = ctx.store.put_text(
            full_log,
            mime_type="text/plain",
            name=f"pytest_{code_ref_id}.log",
        )
        passed = returncode == 0
        failure_summary = "" if passed else _summarize_pytest_failure(stdout, stderr)
        return Done(
            output={
                "passed": passed,
                "returncode": returncode,
                "failure_summary": failure_summary,
                "code_ref_id": code_ref_id,
                "log_ref_id": log_ref.ref_id,
            }
        )

    @machine.state("decide_repair")
    def decide_repair(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done | Spawn | Fail:
        task_name = str(task.input["task"])
        branch_id = str(task.input["branch_id"])
        attempt = int(task.input["attempt"])
        branch_max_attempts = int(task.input.get("max_attempts", max_attempts))
        code_ref_id = str(task.input["code_ref_id"])

        verify_out = _unwrap_dep(next(iter(deps.values())), ctx.store)
        passed = bool(verify_out.get("passed", False))
        log_ref_id = verify_out.get("log_ref_id")

        if passed:
            return Done(
                output={
                    "certified": True,
                    "branch_id": branch_id,
                    "attempt": attempt,
                    "code_ref_id": code_ref_id,
                    "log_ref_id": log_ref_id,
                }
            )

        failure_summary = str(verify_out.get("failure_summary", ""))
        if attempt < branch_max_attempts:
            summary_payload, _ = ctx.store.maybe_spill(failure_summary, _FAILURE_INLINE_LIMIT)
            return Spawn(
                tasks=[],
                then=TaskSpec(
                    state="generate",
                    input={
                        "task": task_name,
                        "branch_id": branch_id,
                        "attempt": attempt + 1,
                        "max_attempts": branch_max_attempts,
                        "failure_summary": summary_payload,
                        "prev_code_ref_id": code_ref_id,
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
                "code_ref_id": code_ref_id,
                "failure_summary": failure_summary,
                "log_ref_id": log_ref_id,
            }
        )

    @machine.state("select_best")
    def select_best(task: Task, deps: dict[str, Any], ctx: TaskContext) -> Done:
        task_name = str(task.input.get("task", "slugify"))
        branch_results = [
            _unwrap_dep(dep_val, ctx.store)
            for dep_val in deps.values()
        ]
        branch_results.sort(key=lambda item: str(item.get("branch_id", "")))

        certified = [item for item in branch_results if item.get("certified")]
        if not certified:
            return Done(
                output={
                    "certified": False,
                    "task": task_name,
                    "selected_branch": None,
                    "code_ref_id": None,
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
                "task": task_name,
                "selected_branch": best["branch_id"],
                "attempts": best["attempt"],
                "code_ref_id": best["code_ref_id"],
                "branches": branch_results,
            }
        )

    return machine
