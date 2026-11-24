"""Runnable demo for the non-SQL codebase concurrency audit example on ARES.

Runs completely offline by default using a deterministic swarm magnet proposer
and ARES's built-in `DeterministicBoardEvaluator` (pass `--live` to use Gemini 4).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ares import (
    HypothesisSpec,
    InvestigationBoard,
    LoopConfig,
    Note,
    ProcessorResult,
    TaskSpec,
    run_investigation_loop,
)
from ares.models import GeminiModelProvider, ModelTelemetry
from examples.codebase_audit.domain import CodebaseConcurrencyDomain


class _OfflineSwarmProposer:
    """Deterministic local proposer simulating parallel `gemini-4-hydrogen` / `carbon` roles."""

    def make_propose_processor(self, **kwargs: Any) -> Any:
        del kwargs

        def _proc(spec: TaskSpec, dep_outputs: dict[str, Any]) -> ProcessorResult:
            del dep_outputs
            fam = (spec.context or {}).get("hypothesis_family", "entry_reachability")
            prop_map = {
                "entry_reachability": "inspect:reachable",
                "lock_graph_cycles": "inspect:lock_cycle",
                "stress_repro": "inspect:race_repro",
            }
            return ProcessorResult(
                output=[
                    HypothesisSpec(
                        query=prop_map.get(fam, "inspect:reachable"),
                        hypothesis_family=fam,
                        expected_signal=f"Check {fam}",
                        requested_under=spec.requested_under,
                    )
                ],
                prompt_tokens=40,
                completion_tokens=16,
                model_calls=1,
            )

        return _proc

    def propose_board_patch(
        self, *, new_notes: list[Note], **kwargs: Any
    ) -> tuple[dict[str, Any], ModelTelemetry]:
        del kwargs
        updates: dict[str, Any] = {}
        q_notes = [n for n in new_notes if n.verifier_name is None and n.row_count > 0]
        if q_notes:
            mods = q_notes[0].evidence.get("matched_modules", [])
            if mods:
                updates["exploratory_hint"] = {
                    "value": mods[0],
                    "sources": [q_notes[0].note_id],
                    "artifact_ids": [],
                }
        return {"updates": updates, "rationale": "Offline context synthesis"}, ModelTelemetry(
            model_calls=1
        )

    def propose_followup_tasks(
        self, *, board: InvestigationBoard, **kwargs: Any
    ) -> tuple[list[HypothesisSpec], ModelTelemetry]:
        del kwargs
        return (
            [
                HypothesisSpec(
                    query="inspect:race_repro",
                    hypothesis_family="followup_gap_resolution",
                    expected_signal="Recheck deterministic race repro",
                    requested_under=board.board_version,
                )
            ],
            ModelTelemetry(model_calls=1),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARES Codebase Concurrency Audit Example.")
    parser.add_argument("--live", action="store_true", help="Use live GeminiModelProvider (requires GEMINI_API_KEY).")
    args = parser.parse_args()

    domain = CodebaseConcurrencyDomain()
    provider: Any = GeminiModelProvider() if args.live else _OfflineSwarmProposer()

    with tempfile.TemporaryDirectory(prefix="ares_codebase_audit_") as tmp_str:
        tmp_root = Path(tmp_str)
        print("=== ARES Non-SQL Codebase Concurrency Audit ===")
        for needle in domain.needle_menu():
            ws = tmp_root / needle.case_id
            domain.generate(str(ws))
            truth = domain.plant(str(ws), needle)

            result = run_investigation_loop(
                domain,
                str(ws),
                provider=provider,
                config=LoopConfig(max_rounds=3, max_concurrent_workers=3),
            )
            scored = domain.score(result.board.model_dump(), truth)
            print(
                f"\nCase: {needle.case_id} ({needle.category})\n"
                f"  Termination: {result.termination_reason} | "
                f"Conclusion: {result.board.conclusion.answer} "
                f"(verified={result.board.conclusion.verified}, "
                f"confidence={result.board.conclusion.confidence:.2f})\n"
                f"  Blocking fields: {result.blocking_fields}\n"
                f"  Board drift: {result.final_board_drift} | "
                f"Ungrounded: {result.ungrounded_fields_on_board} | "
                f"Score: {scored}"
            )


if __name__ == "__main__":
    main()
