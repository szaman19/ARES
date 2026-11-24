"""Tests for the optional external `BoardEvaluator` example (`examples/external_evaluator/`)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ares import LoopConfig, run_investigation_loop
from ares.contracts import (
    BoardConclusion,
    BoardEvaluator,
    BoardField,
    InvestigationBoard,
    SearchQuestion,
)
from examples.codebase_audit.domain import CODE_PIVOTAL_FIELDS, CodebaseConcurrencyDomain
from examples.codebase_audit.run import _OfflineSwarmProposer
from examples.external_evaluator.decision_probe import (
    DecisionProbeClient,
    DecisionProbeEvaluator,
    _build_questions_payload,
    ablate_board_fields,
    perturb_board_field,
    serialize_board_for_decision,
)
from examples.incident_rca.domain import MicroBenchDomain


def _sample_questions() -> list[SearchQuestion]:
    return MicroBenchDomain().search_questions()


def _make_empty_board() -> InvestigationBoard:
    return InvestigationBoard(
        board_version=1,
        conclusion=BoardConclusion(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
    )


class _FakeDecisionClient:
    """Deterministic fake `v1beta` client recording every `decisions.create` invocation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.decisions = SimpleNamespace(create=self._create)

    def _create(
        self,
        *,
        model: str,
        input: Any,
        questions: dict[str, dict[str, Any]],
    ) -> Any:
        self.calls.append({"model": model, "input": input, "questions": questions})
        raw_text = input[0]["text"] if isinstance(input, list) else input
        parsed = json.loads(raw_text)

        assert set(parsed.keys()) == {"board_version", "fields"}
        fields = parsed["fields"]

        required_clues = ("logs_retry_spike", "metrics_zscore_breach", "traces_error_cluster")
        all_payments_certified = all(
            fields.get(c, {}).get("status") == "certified"
            and fields.get(c, {}).get("value") == "payments-db"
            for c in required_clues
        )

        if all_payments_certified:
            rc_probs = {
                "payments-db": 0.94,
                "none_verified": 0.03,
                "checkout-svc": 0.01,
                "auth-svc": 0.01,
                "cache-redis": 0.01,
            }
            rc_choice = "payments-db"
            rc_conf = 0.94
            act_choice = "rollback"
            act_conf = 0.92
            act_probs = {"rollback": 0.92, "restart": 0.03, "scale_pool": 0.02, "escalate": 0.03}
            noul_val = 0.95
        else:
            rc_probs = {
                "none_verified": 0.91,
                "payments-db": 0.06,
                "checkout-svc": 0.01,
                "auth-svc": 0.01,
                "cache-redis": 0.01,
            }
            rc_choice = "none_verified"
            rc_conf = 0.91
            act_choice = "escalate"
            act_conf = 0.93
            act_probs = {"escalate": 0.93, "rollback": 0.04, "restart": 0.02, "scale_pool": 0.01}
            noul_val = 0.06

        answers: dict[str, Any] = {}
        for qid, qspec in questions.items():
            qtype = qspec["type"]
            if qtype == "choice":
                if qid == "root_cause":
                    full_probs = {opt: rc_probs.get(opt, 0.0) for opt in qspec["criteria"]}
                    answers[qid] = SimpleNamespace(
                        choice=rc_choice,
                        confidence=rc_conf,
                        probabilities=full_probs,
                    )
                else:
                    full_probs = {opt: act_probs.get(opt, 0.0) for opt in qspec["criteria"]}
                    answers[qid] = SimpleNamespace(
                        choice=act_choice,
                        confidence=act_conf,
                        probabilities=full_probs,
                    )
            elif qtype == "noul":
                answers[qid] = SimpleNamespace(
                    noul=noul_val,
                    confidence=max(noul_val, 1.0 - noul_val),
                )
            elif qtype == "score":
                labels = list(qspec["criteria"])
                answers[qid] = SimpleNamespace(
                    score=1.0,
                    confidence=0.88,
                    probabilities={
                        lbl: (0.88 if idx == 0 else 0.12) for idx, lbl in enumerate(labels)
                    },
                )

        usage = SimpleNamespace(
            total_input_tokens=120,
            total_output_tokens=0,
            total_thought_tokens=0,
            total_tokens=120,
        )
        return SimpleNamespace(answers=answers, usage=usage)


class _MockGenericDecisionClient:
    def __init__(self, required_fields: Sequence[str]) -> None:
        self._req = tuple(required_fields)
        self.decisions = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: Any, questions: dict[str, dict[str, Any]]) -> Any:
        assert model == "gemini-decision-1"
        raw = input[0]["text"] if isinstance(input, list) else input
        fields = json.loads(raw)["fields"]

        primary_qid = next(qid for qid, q in questions.items() if q["type"] == "choice")
        opts = [o for o in questions[primary_qid]["criteria"] if o != "none_verified"]

        winner = None
        if all(
            fields.get(r, {}).get("status") == "certified"
            and fields.get(r, {}).get("value") not in ("unknown", "none")
            for r in self._req
        ):
            for cand in opts:
                if all(
                    fields[r]["value"] == cand
                    or (isinstance(fields[r]["value"], list) and cand in fields[r]["value"])
                    for r in self._req
                ):
                    winner = cand
                    break

        probs = {o: 0.0 for o in questions[primary_qid]["criteria"]}
        if winner:
            probs[winner] = 0.96
            probs["none_verified"] = 0.04
            choice = winner
            noul = 0.97
        else:
            probs["none_verified"] = 0.95
            if opts:
                probs[opts[0]] = 0.05
            choice = "none_verified"
            noul = 0.03

        return SimpleNamespace(
            answers={
                primary_qid: SimpleNamespace(choice=choice, confidence=0.95, probabilities=probs),
                "sufficiently_verified": SimpleNamespace(noul=noul, confidence=0.96),
            },
            usage=SimpleNamespace(total_input_tokens=80, total_tokens=80),
        )


def test_ablate_board_fields_keeps_keys_and_sets_explicit_unknown() -> None:
    board = InvestigationBoard(
        board_version=3,
        conclusion=BoardConclusion(
            question_id="root_cause",
            question="Which component is the verified root cause?",
            answer="payments-db",
            confidence=0.95,
            verified=True,
        ),
        fields={
            "logs_retry_spike": BoardField(
                value="payments-db",
                status="certified",
                sources=["note-1"],
                artifact_ids=["art-1"],
                priority=0.34,
            ),
            "metrics_zscore_breach": BoardField(
                value="payments-db",
                status="certified",
                sources=["note-2"],
            ),
        },
    )

    ablated = ablate_board_fields(board, ["logs_retry_spike"])

    assert set(ablated.fields.keys()) == {"logs_retry_spike", "metrics_zscore_breach"}
    assert board.fields["logs_retry_spike"].value == "payments-db"

    assert ablated.fields["logs_retry_spike"].value == "unknown"
    assert ablated.fields["logs_retry_spike"].status == "unknown"
    assert ablated.fields["logs_retry_spike"].sources == []
    assert ablated.fields["logs_retry_spike"].artifact_ids == []

    serialized = serialize_board_for_decision(ablated)
    parsed = json.loads(serialized)
    assert "conclusion" not in parsed
    assert "conclusions" not in parsed
    assert parsed["fields"]["logs_retry_spike"] == {"value": "unknown", "status": "unknown"}
    assert parsed["fields"]["metrics_zscore_breach"] == {
        "value": "payments-db",
        "status": "certified",
    }


def test_serialize_board_shuffle_seed_permutes_field_order() -> None:
    board = _make_empty_board()
    for idx in range(6):
        board.fields[f"field_{idx}"] = BoardField(value=f"val_{idx}", status="certified")

    orders = {
        tuple(json.loads(serialize_board_for_decision(board, shuffle_seed=s))["fields"].keys())
        for s in range(5)
    }
    assert len(orders) > 1
    perturbed = perturb_board_field(board, "field_0", "override", status="uncertified")
    assert perturbed.fields["field_0"].value == "override"
    assert perturbed.fields["field_0"].status == "uncertified"


def test_decision_probe_rubrics_use_domain_supplied_criteria_and_generic_fallbacks() -> None:
    domain = MicroBenchDomain()
    questions = domain.search_questions()
    payload = _build_questions_payload(
        questions, required_signal_fields=domain.required_signal_fields()
    )

    sv_inst = payload["sufficiently_verified"]["instructions"]
    assert "logs_retry_spike" in sv_inst
    assert "Answer true" in sv_inst

    rc_spec = payload["root_cause"]
    assert "logs_retry_spike" in rc_spec["criteria"]["none_verified"]
    assert "ALL THREE required signal fields" in rc_spec["criteria"]["payments-db"]

    generic_qs = [
        SearchQuestion(
            question_id="lead_molecule",
            question="Which molecule passes all filters?",
            options=["mol_01", "mol_02", "none_verified"],
        ),
        SearchQuestion(
            question_id="sufficiently_verified",
            question="Are all chemistry filters verified?",
            options=["yes", "no"],
            question_type="yes_no",
        ),
    ]
    generic_payload = _build_questions_payload(
        generic_qs,
        required_signal_fields=("pharmacophore_match", "pains_clean", "low_strain_3d"),
    )
    assert "pharmacophore_match" in generic_payload["lead_molecule"]["instructions"]
    assert "low_strain_3d" in generic_payload["sufficiently_verified"]["instructions"]
    assert "noul >= 0.5" in generic_payload["sufficiently_verified"]["instructions"]


def test_probe_board_one_board_per_call_and_pivotal_vs_distractor_influence() -> None:
    fake_client = _FakeDecisionClient()
    probe = DecisionProbeClient(client=fake_client)
    questions = _sample_questions()

    board = InvestigationBoard(
        board_version=2,
        conclusion=BoardConclusion(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
        fields={
            "logs_retry_spike": BoardField(
                value="payments-db", status="certified", sources=["n1"]
            ),
            "metrics_zscore_breach": BoardField(
                value="payments-db", status="certified", sources=["n2"]
            ),
            "traces_error_cluster": BoardField(
                value="payments-db", status="certified", sources=["n3"]
            ),
            "distractor_cache_noise": BoardField(
                value="cache-redis", status="uncertified", sources=["n4"]
            ),
            "auxiliary_host_note": BoardField(value="unknown", status="unknown"),
        },
    )

    candidate_vals = ["payments-db", "cache-redis"]
    result = probe.probe_board(
        board,
        questions,
        candidate_values=candidate_vals,
    )

    expected_ablations = 4
    expected_voi_variants = 1 * len(candidate_vals)
    assert len(fake_client.calls) == 1 + expected_ablations + expected_voi_variants
    assert result.api_calls == 1 + expected_ablations + expected_voi_variants
    assert result.telemetry.model_calls == 7
    assert result.telemetry.prompt_tokens == 7 * 120

    assert result.base_evaluation.answers["root_cause"].answer == "payments-db"
    assert result.updated_board.conclusion.answer == "payments-db"
    assert result.updated_board.conclusions["recommended_action"].answer == "rollback"
    assert result.updated_board.conclusions["sufficiently_verified"].answer == "yes"

    for clue in ("logs_retry_spike", "metrics_zscore_breach", "traces_error_cluster"):
        assert result.field_influences[clue] == pytest.approx(1.0 / 3.0, abs=1e-4)
        assert result.updated_board.fields[clue].priority == pytest.approx(1.0 / 3.0, abs=1e-4)

    assert result.field_influences["distractor_cache_noise"] == pytest.approx(0.0, abs=1e-6)
    assert result.certified_influence_share == pytest.approx(1.0, abs=1e-5)
    assert result.uncertified_influence_share == pytest.approx(0.0, abs=1e-5)
    assert result.unknown_voi_max == pytest.approx(0.0, abs=1e-5)


def test_probe_board_computes_high_unknown_voi_when_missing_third_clue() -> None:
    fake_client = _FakeDecisionClient()
    probe = DecisionProbeClient(client=fake_client)
    questions = _sample_questions()

    board = InvestigationBoard(
        board_version=1,
        conclusion=BoardConclusion(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
        fields={
            "logs_retry_spike": BoardField(
                value="payments-db", status="certified", sources=["n1"]
            ),
            "metrics_zscore_breach": BoardField(
                value="payments-db", status="certified", sources=["n2"]
            ),
            "traces_error_cluster": BoardField(value="unknown", status="unknown"),
        },
    )

    cands = ["payments-db", "auth-svc"]
    result = probe.probe_board(board, questions, candidate_values=cands)

    assert len(fake_client.calls) == 1 + 2 + 2
    assert result.api_calls == 5
    assert result.updated_board.conclusion.answer == "none_verified"
    assert result.updated_board.conclusions["sufficiently_verified"].answer == "no"

    assert result.field_voi["traces_error_cluster"] > 0.80
    assert result.unknown_voi_max > 0.80
    assert result.updated_board.conclusion.metadata["unknown_voi_max"] > 0.80


def test_probe_board_contested_confidence_triggers_shuffled_repeat_and_bounded_score() -> None:
    calls: list[str] = []

    def _contested_create(*, model: str, input: Any, questions: dict[str, dict[str, Any]]) -> Any:
        assert model == "gemini-decision-1"
        calls.append(input)
        answers: dict[str, Any] = {}
        for qid, qspec in questions.items():
            if qspec["type"] == "choice":
                probs = {opt: 0.0 for opt in qspec["criteria"]}
                first_opt = next(iter(qspec["criteria"]))
                second_opt = list(qspec["criteria"])[1]
                probs[first_opt] = 0.60
                probs[second_opt] = 0.40
                answers[qid] = SimpleNamespace(
                    choice=first_opt, confidence=0.60, probabilities=probs
                )
            elif qspec["type"] == "score":
                labels = list(qspec["criteria"])
                answers[qid] = SimpleNamespace(
                    score=2.0,
                    confidence=0.75,
                    probabilities={labels[0]: 0.25, labels[1]: 0.75},
                )
        return SimpleNamespace(
            answers=answers,
            usage=SimpleNamespace(total_input_tokens=50, total_tokens=50),
        )

    fake_client = SimpleNamespace(decisions=SimpleNamespace(create=_contested_create))
    probe = DecisionProbeClient(client=fake_client)

    board = InvestigationBoard(
        board_version=1,
        conclusion=BoardConclusion(question_id="root_cause", question="Root cause?"),
        fields={
            "f1": BoardField(value="a", status="unknown"),
            "f2": BoardField(value="b", status="unknown"),
        },
    )
    questions = [
        SearchQuestion(
            question_id="root_cause",
            question="Root cause?",
            options=["cand_a", "none_verified"],
            question_type="multiple_choice",
        ),
        SearchQuestion(
            question_id="severity_rubric",
            question="Severity level?",
            options=["low", "high"],
            question_type="bounded_score",
        ),
    ]

    res = probe.probe_board(board, questions, candidate_values=[])
    assert res.api_calls == 2
    assert len(calls) == 2
    assert res.base_evaluation.answers["severity_rubric"].answer == "high"
    assert res.base_evaluation.answers["severity_rubric"].raw_score == pytest.approx(2.0)


def test_decision_probe_evaluator_plugs_into_run_investigation_loop(tmp_path: Path) -> None:
    domain = CodebaseConcurrencyDomain()
    evaluator = DecisionProbeEvaluator(
        DecisionProbeClient(client=_MockGenericDecisionClient(CODE_PIVOTAL_FIELDS))
    )
    assert isinstance(evaluator, BoardEvaluator)

    decoy_dir = tmp_path / "ext_decoy"
    domain.generate(str(decoy_dir))
    truth = domain.plant(str(decoy_dir), domain.needle_menu()[0])

    res = run_investigation_loop(
        domain,
        str(decoy_dir),
        provider=_OfflineSwarmProposer(),  # type: ignore[arg-type]
        evaluator=evaluator,
        config=LoopConfig(max_rounds=3),
    )

    assert res.termination_reason == "certified_success"
    assert res.board.conclusion.answer == "SessionPool"
    assert res.evaluator_calls > 0
    assert res.final_board_drift == 0.0
    assert domain.score(res.board.model_dump(), truth)["correct"] is True


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("ARES_RUN_LIVE_TESTS") == "1"),
    reason="Live Gemini Decision API tests are opt-in (set GEMINI_API_KEY and ARES_RUN_LIVE_TESTS=1)",
)
def test_live_decision_probe_smoke() -> None:
    domain = MicroBenchDomain()
    board = InvestigationBoard(
        board_version=1,
        conclusion=BoardConclusion(
            question_id="root_cause",
            question="Which component is the verified root cause?",
        ),
        fields={
            "logs_retry_spike": BoardField(value="payments-db", status="certified", sources=["n1"]),
            "metrics_zscore_breach": BoardField(value="payments-db", status="certified", sources=["n2"]),
            "traces_error_cluster": BoardField(value="unknown", status="unknown"),
        },
    )
    probe = DecisionProbeClient()
    res = probe.probe_board(
        board,
        domain.search_questions(),
        required_signal_fields=domain.required_signal_fields(),
        candidate_values=["payments-db", "auth-svc"],
    )
    assert res.api_calls >= 3
    assert res.unknown_voi_max > 0.0
