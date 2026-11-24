"""Runnable example showing how to plug an external `BoardEvaluator` into ARES.

Demonstrates attaching `DecisionProbeEvaluator` (wrapping `DecisionProbeClient`)
to `run_investigation_loop(...)` without putting decision probing inside core `ares`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ares import BoardEvaluator, LoopConfig, run_investigation_loop
from ares.models import GeminiModelProvider
from examples.codebase_audit.domain import CODE_PIVOTAL_FIELDS, CodebaseConcurrencyDomain
from examples.codebase_audit.run import _OfflineSwarmProposer
from examples.external_evaluator.decision_probe import (
    DecisionProbeClient,
    DecisionProbeEvaluator,
)


class _MockDecisionClient:
    """Offline mock implementing the `client.decisions.create` interface."""

    def __init__(self, required_fields: tuple[str, ...]) -> None:
        self._req = required_fields
        self.decisions = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: Any, questions: dict[str, dict[str, Any]]) -> Any:
        del model
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARES External Evaluator Example.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline with deterministic mocks (no Gemini API access required).",
    )
    args = parser.parse_args()

    domain = CodebaseConcurrencyDomain()
    if args.mock:
        provider: Any = _OfflineSwarmProposer()
        probe_client = DecisionProbeClient(client=_MockDecisionClient(tuple(CODE_PIVOTAL_FIELDS)))
    else:
        provider = GeminiModelProvider()
        probe_client = DecisionProbeClient()

    evaluator = DecisionProbeEvaluator(probe_client)
    assert isinstance(evaluator, BoardEvaluator)

    with tempfile.TemporaryDirectory(prefix="ares_ext_eval_") as tmp_str:
        ws = Path(tmp_str) / "decoy_case"
        domain.generate(str(ws))
        needle = domain.needle_menu()[0]
        truth = domain.plant(str(ws), needle)

        res = run_investigation_loop(
            domain,
            str(ws),
            provider=provider,
            evaluator=evaluator,
            config=LoopConfig(max_rounds=3),
        )
        scored = domain.score(res.board.model_dump(), truth)
        print(
            f"External Evaluator Example ({'mock' if args.mock else 'live'}):\n"
            f"  Termination: {res.termination_reason}\n"
            f"  Conclusion:  {res.board.conclusion.answer} "
            f"(confidence={res.board.conclusion.confidence:.2f}, verified={res.board.conclusion.verified})\n"
            f"  Eval Calls:  {res.evaluator_calls}\n"
            f"  Metadata:    {res.board.conclusion.metadata}\n"
            f"  Score:       {scored}"
        )


if __name__ == "__main__":
    main()
