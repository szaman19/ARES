"""Offline end-to-end tests proving core ARES & Context Management run with zero SQL or DuckDB.

Tests:
1. Standalone pure-Python `task_type="verify"` execution in `ares.solve(..., backend=None)`
   enforcing Invariant I4 (context isolation), Invariant I2 (taint guard), and Invariant I6
   (content-hash deduplication across rounds).
2. Complete non-SQL `Domain` (`CodebaseConcurrencyDomain` over AST/call & lock graphs from
   `examples.codebase_audit`) running end-to-end through `run_investigation_loop()` with ARES's
   built-in `DeterministicBoardEvaluator` on decoy and under-determined cases without DuckDB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ares import (
    Domain,
    ExecutionBackend,
    GuardValidationError,
    InMemoryVerifierRegistry,
    LoopConfig,
    NoteStore,
    TaskGraph,
    TaskSpec,
    VerifierResult,
    run_investigation_loop,
    solve,
)
from examples.codebase_audit.domain import ASTGraphBackend, CodebaseConcurrencyDomain
from examples.codebase_audit.run import _OfflineSwarmProposer


def test_standalone_python_verifiers_run_without_any_backend() -> None:
    """`task_type='verify'` works natively with `backend=None` (zero SQL/DuckDB)."""
    seen_inputs: list[dict[str, Any]] = []

    def _trusted_check(params: dict[str, Any]) -> VerifierResult:
        seen_inputs.append(dict(params))
        ok = params.get("smiles") == "c1ccccc1"
        return VerifierResult(
            verifier_name="check_aromatic_ring",
            passed=ok,
            certification="CERTIFIED" if ok else "UNCERTIFIED",
            evidence={"smiles": params.get("smiles"), "aromatic": ok},
            row_count=1 if ok else 0,
            summary="Aromatic ring verified." if ok else "Not benzene.",
        )

    def _untrusted_heuristic(params: dict[str, Any]) -> VerifierResult:
        return VerifierResult(
            verifier_name="llm_heuristic",
            passed=True,
            certification="CERTIFIED",  # Must be downgraded to UNCERTIFIED by ARES taint guard
            evidence={"smiles": params.get("smiles")},
            row_count=1,
            summary="Heuristic guess",
        )

    reg = InMemoryVerifierRegistry()
    reg.register("check_aromatic_ring", _trusted_check, trusted=True)
    reg.register("llm_heuristic", _untrusted_heuristic, trusted=False)

    store = NoteStore()
    graph = TaskGraph()

    r1 = solve(
        tasks=[
            TaskSpec(
                task_id="t_trusted",
                task_type="verify",
                verifier_name="check_aromatic_ring",
                verifier_params={"smiles": "c1ccccc1"},
                context={"board_secret": "must_never_leak_into_verifier"},
                requested_under=0,
            ),
            TaskSpec(
                task_id="t_untrusted",
                task_type="verify",
                verifier_name="llm_heuristic",
                verifier_params={"smiles": "c1ccccc1"},
                requested_under=0,
            ),
        ],
        graph=graph,
        backend=None,
        registry=reg,
        note_store=store,
    )

    assert r1.errors == {}
    assert r1.usage.completed_tasks == 2
    assert r1.usage.verifiers_executed == 2
    assert r1.usage.certified_notes == 1
    assert r1.usage.uncertified_notes == 1
    assert seen_inputs == [{"smiles": "c1ccccc1"}]  # Invariant I4 verified

    # Round 2: Re-submitting identical verifier+params deduplicates automatically
    r2 = solve(
        tasks=[
            TaskSpec(
                task_id="t_trusted_dup",
                task_type="verify",
                verifier_name="check_aromatic_ring",
                verifier_params={"smiles": "c1ccccc1"},
                requested_under=1,
            )
        ],
        graph=graph,
        backend=None,
        registry=reg,
        note_store=store,
    )
    assert r2.graph.nodes["t_trusted_dup"].status == "skipped_duplicate"
    assert r2.usage.skipped_duplicate_tasks == 1
    assert len(store) == 2


def test_ast_graph_backend_enforces_root_guards(tmp_path: Path) -> None:
    backend = ASTGraphBackend(str(tmp_path))
    assert backend.validate_query("  inspect:reachable ") == "inspect:reachable"
    with pytest.raises(GuardValidationError):
        backend.validate_query("SELECT * FROM code_graph")
    with pytest.raises(GuardValidationError):
        backend.validate_query("inspect:../etc/passwd")
    with pytest.raises(GuardValidationError):
        backend.validate_query("inspect:reachable; drop")


def test_non_sql_domain_runs_full_investigation_loop_end_to_end(tmp_path: Path) -> None:
    domain = CodebaseConcurrencyDomain()
    assert isinstance(domain, Domain)
    assert isinstance(domain.create_backend(str(tmp_path)), ExecutionBackend)

    # Case 1: Decoy disambiguation (SessionPool has all 3 signals; TokenCache decoy has 2/3)
    decoy_dir = tmp_path / "code_decoy"
    domain.generate(str(decoy_dir))
    truth_decoy = domain.plant(str(decoy_dir), domain.needle_menu()[0])

    res_decoy = run_investigation_loop(
        domain,
        str(decoy_dir),
        provider=_OfflineSwarmProposer(),  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3),
    )

    assert res_decoy.termination_reason == "certified_success"
    assert res_decoy.board.conclusion.answer == "SessionPool"
    assert res_decoy.board.conclusion.verified is True
    assert res_decoy.final_board_drift == 0.0
    assert res_decoy.ungrounded_fields_on_board == 0
    assert res_decoy.queries_executed >= 1
    assert domain.score(res_decoy.board.model_dump(), truth_decoy) == {
        "correct": True,
        "decoy_avoided": True,
    }

    # Case 2: Under-determined (missing deterministic race repro -> honest_uncertified)
    ud_dir = tmp_path / "code_ud"
    domain.generate(str(ud_dir))
    truth_ud = domain.plant(str(ud_dir), domain.needle_menu()[1])

    res_ud = run_investigation_loop(
        domain,
        str(ud_dir),
        provider=_OfflineSwarmProposer(),  # type: ignore[arg-type]
        config=LoopConfig(max_rounds=3),
    )
    assert res_ud.termination_reason == "honest_uncertified"
    assert res_ud.board.conclusion.answer == "none_verified"
    assert res_ud.board.conclusion.verified is False
    assert "deterministic_race_repro" in res_ud.blocking_fields
    assert res_ud.final_board_drift == 0.0
    assert domain.score(res_ud.board.model_dump(), truth_ud)["correct"] is True
