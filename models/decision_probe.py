"""Calibrated decision probe, leave-one-out 'unknown' ablation, and VoI client (`gemini-decision-1`).

Enforces ARES Plan §4 hard invariants:
1. Strictly ONE board state per `decisions.create` API call (never multi-board prompts).
2. Explicit `"unknown"` ablation (`ablate_board_fields` sets `value="unknown"`, `status="unknown"`
   and NEVER deletes keys from `board.fields`).
3. Excludes prior `decision` answers/influences from serialized input to prevent self-confirming bias.
4. Symmetric choice rubrics for `root_cause` (`none_verified` vs components) and `recommended_action`.
"""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

from contracts import (
    ArtifactRef,
    BoardDecision,
    BoardField,
    DecisionQuestion,
    FieldStatus,
    InvestigationBoard,
)
from models.gemini_provider import ModelTelemetry, extract_telemetry


class QuestionEvaluation(BaseModel):
    """Calibrated single-question evaluation returned by `gemini-decision-1`."""

    question_id: str
    answer: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(default_factory=dict)
    raw_score: float | None = None


class BoardEvaluation(BaseModel):
    """Evaluation of all closed-set DecisionQuestions on a single InvestigationBoard."""

    answers: dict[str, QuestionEvaluation] = Field(default_factory=dict)
    telemetry: ModelTelemetry = Field(default_factory=ModelTelemetry)


class ProbeResult(BaseModel):
    """Full decision probe, counterfactual ablation influence, and VoI result."""

    base_evaluation: BoardEvaluation
    field_influences: dict[str, float] = Field(default_factory=dict)
    raw_swings: dict[str, float] = Field(default_factory=dict)
    certified_influence_share: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertified_influence_share: float = Field(default=0.0, ge=0.0, le=1.0)
    field_voi: dict[str, float] = Field(default_factory=dict)
    unknown_voi_max: float = Field(default=0.0, ge=0.0, le=1.0)
    updated_board: InvestigationBoard
    api_calls: int = 0
    telemetry: ModelTelemetry = Field(default_factory=ModelTelemetry)


# ----------------------------------------------------------------------
# Pure Board Transformation & Serialization Helpers (ARES Plan §4)
# ----------------------------------------------------------------------


def ablate_board_fields(
    board: InvestigationBoard,
    field_names: list[str],
) -> InvestigationBoard:
    """Set target fields to explicit `value='unknown'`, `status='unknown'` WITHOUT deleting keys."""
    ablated = board.model_copy(deep=True)
    for fname in field_names:
        if fname in ablated.fields:
            target = ablated.fields[fname]
            target.value = "unknown"
            target.status = "unknown"
            target.sources = []
            target.artifact_ids = []
    return ablated


def perturb_board_field(
    board: InvestigationBoard,
    field_name: str,
    candidate_value: Any,
    status: FieldStatus = "certified",
) -> InvestigationBoard:
    """Set a single board field to `candidate_value` for counterfactual VoI simulation."""
    perturbed = board.model_copy(deep=True)
    if field_name in perturbed.fields:
        perturbed.fields[field_name].value = candidate_value
        perturbed.fields[field_name].status = status
    else:
        perturbed.fields[field_name] = BoardField(
            value=candidate_value,
            status=status,
        )
    return perturbed


def serialize_board_for_decision(
    board: InvestigationBoard,
    *,
    shuffle_seed: int | None = None,
) -> str:
    """Serialize ONLY ONE board's active field evidence for `gemini-decision-1`.

    Excludes prior `decision`, `decisions`, `influence`, and `voi` metadata so
    `gemini-decision-1` evaluates strictly from empirical field values and statuses
    without self-confirming bias. Preserves shuffled key order when `shuffle_seed` is set.
    """
    keys = list(board.fields.keys())
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(keys)

    fields_payload: dict[str, dict[str, Any]] = {}
    for k in keys:
        f = board.fields[k]
        fields_payload[k] = {
            "value": f.value,
            "status": f.status,
        }

    single_board_payload = {
        "board_version": board.board_version,
        "fields": fields_payload,
    }
    # Do NOT pass sort_keys=True so optional shuffle_seed ordering is preserved.
    return json.dumps(single_board_payload, indent=2, default=str)


def total_variation_distance(
    probs_a: dict[str, float],
    probs_b: dict[str, float],
) -> float:
    """Compute Total Variation Distance `0.5 * sum_x |P(x) - Q(x)|` in `[0.0, 1.0]`."""
    keys = set(probs_a.keys()) | set(probs_b.keys())
    if not keys:
        return 0.0
    tvd = 0.5 * sum(abs(probs_a.get(k, 0.0) - probs_b.get(k, 0.0)) for k in keys)
    return max(0.0, min(1.0, tvd))


# ----------------------------------------------------------------------
# Symmetric Calibration Rubrics for `gemini-decision-1`
# ----------------------------------------------------------------------


def _build_choice_question_spec(q: DecisionQuestion) -> dict[str, Any]:
    options = list(q.options)
    criteria: dict[str, str] = {}

    if "none_verified" in options:
        instructions = (
            f"{q.question} Evaluate strictly from the board's `fields` (`value` and `status`). "
            "The three required deterministic signal fields are `logs_retry_spike`, "
            "`metrics_zscore_breach`, and `traces_error_cluster` (a signal field may hold either "
            "a single service string or a list of candidate services when a partial 2-of-3 decoy "
            "co-exists with a 3-of-3 root cause). "
            "A component is verified if and only if ALL THREE required signal fields "
            "(`logs_retry_spike`, `metrics_zscore_breach`, `traces_error_cluster`) have "
            "`status='certified'` and that component appears in all three (even if unrelated "
            "exploratory fields on the board have `status='uncertified'`)."
        )
        for opt in options:
            if opt == "none_verified":
                criteria[opt] = (
                    "Select if ANY of the three required signal fields (`logs_retry_spike`, "
                    "`metrics_zscore_breach`, `traces_error_cluster`) is missing, has "
                    "status != 'certified' (i.e. 'unknown' or 'uncertified'), or has value in "
                    "('unknown', 'none'), OR if no single component appears in all three."
                )
            else:
                criteria[opt] = (
                    f"Select if all three required signal fields (`logs_retry_spike`, "
                    f"`metrics_zscore_breach`, `traces_error_cluster`) have status='certified' "
                    f"(with none 'unknown', 'uncertified', or 'none') and '{opt}' is present in "
                    f"ALL THREE required signal fields (as the string value or inside the candidate list)."
                )
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }

    if "escalate" in options:
        instructions = (
            f"{q.question} Select a concrete remediation ONLY when a single root cause component "
            "appears across all three `status='certified'` required signal fields "
            "(`logs_retry_spike`, `metrics_zscore_breach`, `traces_error_cluster`); otherwise "
            "select 'escalate'. Inspect `metric_anomaly_type` (for that verified 3-of-3 component) "
            "to distinguish pool exhaustion vs lock/latency regressions."
        )
        action_rubrics = {
            "escalate": (
                "Select when ANY of `logs_retry_spike`, `metrics_zscore_breach`, or "
                "`traces_error_cluster` is missing, has status != 'certified', or has value in "
                "('unknown', 'none'), or when no single component appears in all three."
            ),
            "rollback": (
                "Select ONLY when a component is certified across all three required signals and "
                "certified evidence (`metric_anomaly_type`, `fault_type`, or `candidate_remediation`) "
                "for that verified component indicates a deployment/token regression or payments-db "
                "lock regression warranting rollback (e.g. auth-svc with p99_latency_ms / "
                "auth_token_drift, payments-db with lock_wait_ms, or candidate_remediation='rollback')."
            ),
            "restart": (
                "Select ONLY when a component is certified across all three required signals and "
                "certified evidence (`metric_anomaly_type`, `fault_type`, or `candidate_remediation`) "
                "for that verified component indicates a transient lock/resource wedge warranting "
                "restart (e.g. inventory-db with lock_wait_ms or candidate_remediation='restart')."
            ),
            "scale_pool": (
                "Select ONLY when a component is certified across all three required signals and "
                "certified evidence (`metric_anomaly_type`, `fault_type`, or `candidate_remediation`) "
                "for that verified component indicates connection pool exhaustion "
                "(pool_utilization_pct, POOL_EXHAUSTED, pool_exhaustion, or "
                "candidate_remediation='scale_pool')."
            ),
        }
        for opt in options:
            criteria[opt] = action_rubrics.get(
                opt,
                f"Select ONLY when certified 3-signal root-cause evidence specifically supports '{opt}'.",
            )
        return {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }

    for opt in options:
        if opt.lower() in ("unknown", "abstain", "undetermined"):
            criteria[opt] = (
                "Select when the board fields are unknown, uncertified, or insufficient to "
                "verify any specific option."
            )
        else:
            criteria[opt] = (
                f"Select when status='certified' board fields directly verify '{opt}'."
            )
    return {
        "type": "choice",
        "instructions": q.question,
        "criteria": criteria,
    }


def _build_questions_payload(questions: list[DecisionQuestion]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for q in questions:
        if q.question_type == "multiple_choice":
            payload[q.question_id] = _build_choice_question_spec(q)
        elif q.question_type == "yes_no":
            if q.question_id == "sufficiently_verified" or "verified" in q.question.lower():
                inst = (
                    f"{q.question} Evaluate strictly from the three required deterministic signal "
                    "fields (`logs_retry_spike`, `metrics_zscore_breach`, `traces_error_cluster`). "
                    "Answer true (noul >= 0.5) when at least one component appears across ALL THREE "
                    "required signal fields (`logs_retry_spike`, `metrics_zscore_breach`, "
                    "`traces_error_cluster`) where each of those three required fields has "
                    "status='certified' (even if a signal list also contains a partial 2-of-3 decoy "
                    "component, or unrelated exploratory fields on the board have status='uncertified'). "
                    "Answer false (noul < 0.5) whenever ANY of `logs_retry_spike`, "
                    "`metrics_zscore_breach`, or `traces_error_cluster` is missing, has "
                    "status != 'certified', or has value in ('none', 'unknown'), or when no single "
                    "component appears in all three."
                )
            else:
                inst = q.question
            payload[q.question_id] = {
                "type": "noul",
                "instructions": inst,
            }
        elif q.question_type == "bounded_score":
            payload[q.question_id] = {
                "type": "score",
                "instructions": q.question,
                "criteria": list(q.options),
            }
        else:
            raise ValueError(f"Unsupported DecisionQuestion.question_type: {q.question_type}")
    return payload


def _infer_default_candidates(
    board: InvestigationBoard,
    primary_question: DecisionQuestion | None,
    max_candidates: int = 2,
) -> list[str]:
    """Infer 2 high-signal candidate values for VoI simulation when not explicitly provided."""
    excluded = {"none_verified", "unknown", "escalate", "yes", "no", "none"}
    allowed_options = [
        opt for opt in (primary_question.options if primary_question else []) if opt not in excluded
    ]
    allowed_set = set(allowed_options)

    observed: list[str] = []
    for f in board.fields.values():
        if f.status == "unknown":
            continue
        if isinstance(f.value, str) and f.value in allowed_set and f.value not in observed:
            observed.append(f.value)
        elif isinstance(f.value, dict):
            svc = f.value.get("service")
            if isinstance(svc, str) and svc in allowed_set and svc not in observed:
                observed.append(svc)

    for opt in allowed_options:
        if len(observed) >= max_candidates:
            break
        if opt not in observed:
            observed.append(opt)

    return observed[:max_candidates]


# ----------------------------------------------------------------------
# DecisionProbeClient (`v1beta` `gemini-decision-1`)
# ----------------------------------------------------------------------


class DecisionProbeClient:
    """Calibrated Decisions API (`v1beta`) client for `gemini-decision-1`."""

    def __init__(
        self,
        *,
        decision_model: str = "gemini-decision-1",
        client: Any | None = None,
        api_key: str | None = None,
    ) -> None:
        self.decision_model = decision_model
        self._client = client
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            load_dotenv()
            key = self._api_key or os.environ.get("GEMINI_API_KEY")
            if key:
                self._client = genai.Client(
                    api_key=key,
                    http_options={"api_version": "v1beta"},
                )
            else:
                self._client = genai.Client(http_options={"api_version": "v1beta"})
        return self._client

    def evaluate_board(
        self,
        board: InvestigationBoard,
        questions: list[DecisionQuestion],
        *,
        shuffle_seed: int | None = None,
        artifact_refs: list[ArtifactRef] | None = None,
    ) -> BoardEvaluation:
        """Evaluate ONE InvestigationBoard state via a single `decisions.create` call."""
        board_str = serialize_board_for_decision(board, shuffle_seed=shuffle_seed)
        questions_payload = _build_questions_payload(questions)

        valid_pngs: list[ArtifactRef] = []
        if artifact_refs:
            seen: set[str] = set()
            for art in artifact_refs:
                if art.artifact_id in seen:
                    continue
                is_png = art.kind in ("plot_png", "image") or art.mime_type == "image/png"
                if is_png and Path(art.path).is_file():
                    seen.add(art.artifact_id)
                    valid_pngs.append(art)
                if len(valid_pngs) >= 3:
                    break

        if valid_pngs:
            decision_input: Any = [{"type": "text", "text": board_str}]
            for art in valid_pngs:
                b64_data = base64.b64encode(Path(art.path).read_bytes()).decode("ascii")
                decision_input.append(
                    {
                        "type": "image",
                        "data": b64_data,
                        "mime_type": art.mime_type or "image/png",
                    }
                )
        else:
            decision_input = board_str

        client = self._get_client()
        t0 = time.perf_counter()
        resp = client.decisions.create(
            model=self.decision_model,
            input=decision_input,
            questions=questions_payload,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = extract_telemetry(resp, latency_ms)

        raw_answers = getattr(resp, "answers", None) or {}
        evaluations: dict[str, QuestionEvaluation] = {}

        for q in questions:
            ans = (
                raw_answers.get(q.question_id)
                if isinstance(raw_answers, dict)
                else getattr(raw_answers, q.question_id, None)
            )
            if ans is None:
                evaluations[q.question_id] = QuestionEvaluation(
                    question_id=q.question_id,
                    answer="unknown",
                    confidence=0.0,
                    probabilities={},
                )
                continue

            if q.question_type == "multiple_choice":
                choice_val = str(
                    ans.get("choice", "") if isinstance(ans, dict) else getattr(ans, "choice", "")
                )
                probs_raw = (
                    ans.get("probabilities")
                    if isinstance(ans, dict)
                    else getattr(ans, "probabilities", None)
                ) or {}
                probs = {opt: 0.0 for opt in q.options}
                for k, v in dict(probs_raw).items():
                    probs[str(k)] = float(v)
                conf_raw = (
                    ans.get("confidence")
                    if isinstance(ans, dict)
                    else getattr(ans, "confidence", None)
                )
                conf = float(conf_raw) if conf_raw is not None else probs.get(choice_val, 0.0)
                evaluations[q.question_id] = QuestionEvaluation(
                    question_id=q.question_id,
                    answer=choice_val,
                    confidence=max(0.0, min(1.0, conf)),
                    probabilities=probs,
                )

            elif q.question_type == "yes_no":
                noul_raw = (
                    ans.get("noul", 0.5) if isinstance(ans, dict) else getattr(ans, "noul", 0.5)
                )
                noul_val = max(0.0, min(1.0, float(noul_raw)))
                conf_raw = (
                    ans.get("confidence")
                    if isinstance(ans, dict)
                    else getattr(ans, "confidence", None)
                )
                conf = (
                    float(conf_raw)
                    if conf_raw is not None
                    else max(noul_val, 1.0 - noul_val)
                )
                ans_label = "yes" if noul_val >= 0.5 else "no"
                evaluations[q.question_id] = QuestionEvaluation(
                    question_id=q.question_id,
                    answer=ans_label,
                    confidence=max(0.0, min(1.0, conf)),
                    probabilities={"yes": noul_val, "no": 1.0 - noul_val},
                    raw_score=noul_val,
                )

            elif q.question_type == "bounded_score":
                score_raw = (
                    ans.get("score", 0.0)
                    if isinstance(ans, dict)
                    else getattr(ans, "score", 0.0)
                )
                score_val = float(score_raw)
                probs_raw = (
                    ans.get("probabilities")
                    if isinstance(ans, dict)
                    else getattr(ans, "probabilities", None)
                ) or {}
                probs = {str(k): float(v) for k, v in dict(probs_raw).items()}
                conf_raw = (
                    ans.get("confidence")
                    if isinstance(ans, dict)
                    else getattr(ans, "confidence", None)
                )
                conf = float(conf_raw) if conf_raw is not None else 0.0
                best_label = max(probs, key=lambda k: probs[k]) if probs else str(score_val)
                evaluations[q.question_id] = QuestionEvaluation(
                    question_id=q.question_id,
                    answer=best_label,
                    confidence=max(0.0, min(1.0, conf)),
                    probabilities=probs,
                    raw_score=score_val,
                )

        return BoardEvaluation(answers=evaluations, telemetry=telemetry)

    @staticmethod
    def _average_evaluations(
        eval_a: BoardEvaluation,
        eval_b: BoardEvaluation,
        questions: list[DecisionQuestion],
    ) -> BoardEvaluation:
        """Average two BoardEvaluations (e.g. canonical + shuffled field-order repeat)."""
        merged_answers: dict[str, QuestionEvaluation] = {}
        for q in questions:
            qa = eval_a.answers.get(q.question_id)
            qb = eval_b.answers.get(q.question_id)
            if qa is None and qb is None:
                continue
            if qa is None:
                assert qb is not None
                merged_answers[q.question_id] = qb
                continue
            if qb is None:
                merged_answers[q.question_id] = qa
                continue

            all_keys = set(qa.probabilities.keys()) | set(qb.probabilities.keys())
            avg_probs = {
                k: 0.5 * (qa.probabilities.get(k, 0.0) + qb.probabilities.get(k, 0.0))
                for k in all_keys
            }
            if q.question_type == "yes_no":
                p_yes = avg_probs.get("yes", 0.5)
                ans_str = "yes" if p_yes >= 0.5 else "no"
                conf = max(p_yes, 1.0 - p_yes)
                raw_score: float | None = p_yes
            elif q.question_type == "multiple_choice":
                ans_str = (
                    max(avg_probs, key=lambda k: avg_probs[k]) if avg_probs else qa.answer
                )
                conf = avg_probs.get(ans_str, 0.5 * (qa.confidence + qb.confidence))
                raw_score = None
            else:
                raw_score = 0.5 * ((qa.raw_score or 0.0) + (qb.raw_score or 0.0))
                ans_str = (
                    max(avg_probs, key=lambda k: avg_probs[k]) if avg_probs else qa.answer
                )
                conf = 0.5 * (qa.confidence + qb.confidence)

            merged_answers[q.question_id] = QuestionEvaluation(
                question_id=q.question_id,
                answer=ans_str,
                confidence=max(0.0, min(1.0, conf)),
                probabilities=avg_probs,
                raw_score=raw_score,
            )

        return BoardEvaluation(
            answers=merged_answers,
            telemetry=eval_a.telemetry.plus(eval_b.telemetry),
        )

    def probe_board(
        self,
        board: InvestigationBoard,
        questions: list[DecisionQuestion],
        *,
        primary_question_id: str = "root_cause",
        candidate_values: list[str] | None = None,
        run_group_ablations: bool = True,
        contested_confidence_band: tuple[float, float] = (0.45, 0.85),
        max_concurrent: int = 6,
        artifact_refs: list[ArtifactRef] | None = None,
    ) -> ProbeResult:
        """Evaluate active board, run parallel single-field 'unknown' ablations, and compute VoI."""
        if not questions:
            raise ValueError("probe_board requires at least one DecisionQuestion.")

        primary_q = next(
            (q for q in questions if q.question_id == primary_question_id),
            questions[0],
        )
        eff_primary_id = primary_q.question_id
        workers = max(1, int(max_concurrent))

        # 1. Base board evaluation (1 API call; +1 shuffled repeat if inside contested band)
        base_eval = self.evaluate_board(board, questions, artifact_refs=artifact_refs)
        total_telemetry = base_eval.telemetry
        api_calls = 1

        primary_base = base_eval.answers[eff_primary_id]
        band_low, band_high = contested_confidence_band
        if (
            len(board.fields) > 1
            and band_low < band_high
            and band_low <= primary_base.confidence <= band_high
        ):
            shuffled_eval = self.evaluate_board(
                board,
                questions,
                shuffle_seed=17,
                artifact_refs=artifact_refs,
            )
            api_calls += 1
            base_eval = self._average_evaluations(base_eval, shuffled_eval, questions)
            total_telemetry = base_eval.telemetry
            primary_base = base_eval.answers[eff_primary_id]

        base_probs = primary_base.probabilities

        # 2. Parallel single-field "unknown" ablations for every field with status != "unknown"
        ablation_fields = [
            fname for fname, f in board.fields.items() if f.status != "unknown"
        ]
        raw_swings: dict[str, float] = {fname: 0.0 for fname in board.fields}

        if ablation_fields:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_field = {
                    pool.submit(
                        self.evaluate_board,
                        ablate_board_fields(board, [fname]),
                        questions,
                    ): fname
                    for fname in ablation_fields
                }
                for fut in concurrent.futures.as_completed(fut_to_field):
                    fname = fut_to_field[fut]
                    abl_eval = fut.result()
                    api_calls += 1
                    total_telemetry = total_telemetry.plus(abl_eval.telemetry)
                    abl_probs = abl_eval.answers[eff_primary_id].probabilities
                    raw_swings[fname] = total_variation_distance(base_probs, abl_probs)

        # Optional group ablation fallback ONLY when redundant backup fields mask single-field swings
        non_abstaining = primary_base.answer not in (
            "none_verified",
            "unknown",
            "no",
            "escalate",
        )
        max_single_swing = max((raw_swings[f] for f in ablation_fields), default=0.0)
        if (
            run_group_ablations
            and len(ablation_fields) >= 2
            and max_single_swing < 0.05
            and non_abstaining
        ):
            certified_group = [
                fn for fn in ablation_fields if board.fields[fn].status == "certified"
            ]
            uncertified_group = [
                fn for fn in ablation_fields if board.fields[fn].status == "uncertified"
            ]
            groups_to_test = [g for g in (certified_group, uncertified_group) if g]
            for grp in groups_to_test:
                grp_eval = self.evaluate_board(
                    ablate_board_fields(board, grp),
                    questions,
                )
                api_calls += 1
                total_telemetry = total_telemetry.plus(grp_eval.telemetry)
                grp_swing = total_variation_distance(
                    base_probs, grp_eval.answers[eff_primary_id].probabilities
                )
                if grp_swing > 0.0:
                    share = grp_swing / len(grp)
                    for fn in grp:
                        raw_swings[fn] = max(raw_swings[fn], share)

        sum_swings = sum(raw_swings[fn] for fn in ablation_fields)
        field_influences: dict[str, float] = {fname: 0.0 for fname in board.fields}
        if sum_swings > 0.0:
            for fn in ablation_fields:
                field_influences[fn] = raw_swings[fn] / sum_swings
            cert_share = sum(
                field_influences[fn]
                for fn in ablation_fields
                if board.fields[fn].status == "certified"
            )
            uncert_share = sum(
                field_influences[fn]
                for fn in ablation_fields
                if board.fields[fn].status == "uncertified"
            )
        else:
            cert_share = 0.0
            has_certified = any(board.fields[fn].status == "certified" for fn in ablation_fields)
            if not ablation_fields and non_abstaining:
                uncert_share = 1.0
            elif ablation_fields and not has_certified and non_abstaining:
                uncert_share = 1.0
            else:
                uncert_share = 0.0

        cert_share = max(0.0, min(1.0, cert_share))
        uncert_share = max(0.0, min(1.0, uncert_share))

        # 3. Parallel VoI simulation across 2-3 candidate values (1 board per API call)
        unknown_fields = [
            fname for fname, f in board.fields.items() if f.status == "unknown"
        ]
        if unknown_fields:
            voi_targets = unknown_fields
        elif not non_abstaining:
            # When no 'unknown' fields remain yet decision is still unverified, probe 'uncertified' fields
            voi_targets = [
                fname for fname, f in board.fields.items() if f.status == "uncertified"
            ]
        else:
            voi_targets = []

        if candidate_values is not None:
            cands = list(candidate_values)[:3]
        elif voi_targets:
            cands = _infer_default_candidates(board, primary_q, max_candidates=2)
        else:
            cands = []

        field_voi: dict[str, float] = {fname: 0.0 for fname in board.fields}
        if voi_targets and cands:
            voi_jobs = [(fname, cval) for fname in voi_targets for cval in cands]
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_job = {
                    pool.submit(
                        self.evaluate_board,
                        perturb_board_field(board, fname, cval, status="certified"),
                        questions,
                    ): (fname, cval)
                    for fname, cval in voi_jobs
                }
                for fut in concurrent.futures.as_completed(fut_to_job):
                    fname, _cval = fut_to_job[fut]
                    p_eval = fut.result()
                    api_calls += 1
                    total_telemetry = total_telemetry.plus(p_eval.telemetry)
                    swing = total_variation_distance(
                        base_probs, p_eval.answers[eff_primary_id].probabilities
                    )
                    if swing > field_voi[fname]:
                        field_voi[fname] = swing

        unknown_voi_max = max((field_voi[fn] for fn in voi_targets), default=0.0)
        unknown_voi_max = max(0.0, min(1.0, unknown_voi_max))

        # 4. Populate updated_board copy cleanly
        updated = board.model_copy(deep=True)
        for fname, fobj in updated.fields.items():
            fobj.influence = round(max(0.0, min(1.0, field_influences.get(fname, 0.0))), 6)
            fobj.voi = round(max(0.0, min(1.0, field_voi.get(fname, 0.0))), 6)

        for q in questions:
            q_eval = base_eval.answers[q.question_id]
            updated.decisions[q.question_id] = BoardDecision(
                question_id=q.question_id,
                question=q.question,
                answer=q_eval.answer,
                confidence=round(max(0.0, min(1.0, q_eval.confidence)), 6),
                certified_influence_share=round(cert_share, 6),
                uncertified_influence_share=round(uncert_share, 6),
                unknown_voi_max=round(unknown_voi_max, 6),
            )

        if eff_primary_id in updated.decisions:
            updated.decision = updated.decisions[eff_primary_id].model_copy(deep=True)

        return ProbeResult(
            base_evaluation=base_eval,
            field_influences=field_influences,
            raw_swings=raw_swings,
            certified_influence_share=round(cert_share, 6),
            uncertified_influence_share=round(uncert_share, 6),
            field_voi=field_voi,
            unknown_voi_max=round(unknown_voi_max, 6),
            updated_board=updated,
            api_calls=api_calls,
            telemetry=total_telemetry,
        )


__all__ = [
    "BoardEvaluation",
    "DecisionProbeClient",
    "ProbeResult",
    "QuestionEvaluation",
    "ablate_board_fields",
    "perturb_board_field",
    "serialize_board_for_decision",
    "total_variation_distance",
]
