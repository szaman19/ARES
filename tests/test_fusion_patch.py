"""Comprehensive unit tests for `fusion/patch.py` covering Invariants I1–I7."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from adapters import DuckDBBackend
from ares.store import NoteStore
from contracts import (
    ArtifactRef,
    BoardField,
    DecisionQuestion,
    InMemoryVerifierRegistry,
    Note,
)
from domain.micro_bench import (
    INCIDENT_PIVOTAL_FIELDS,
    MicroBenchDomain,
    extract_incident_note_fields,
)
from fusion import (
    BoardPatch,
    FieldPatch,
    PatchReport,
    PatchValidationError,
    archive_low_influence_fields,
    compute_board_drift,
    compute_weakest_status,
    create_initial_board,
    deduplicate_fields,
    deterministic_patch_from_notes,
    rebuild_board_from_notes,
    validate_and_merge_patch,
    value_is_grounded_in_notes,
)


def _sample_questions() -> list[DecisionQuestion]:
    return [
        DecisionQuestion(
            question_id="root_cause",
            question="Which component is the verified root cause?",
            options=["payments-db", "auth-service", "inventory-db", "none_verified"],
        ),
        DecisionQuestion(
            question_id="recommended_action",
            question="What is the recommended remediation action?",
            options=["rollback", "restart", "scale_pool", "escalate"],
        ),
        DecisionQuestion(
            question_id="sufficiently_verified",
            question="Is the root cause sufficiently verified?",
            options=["yes", "no"],
            question_type="yes_no",
        ),
    ]


def _make_png_artifact(
    tmp_path: Path,
    artifact_id: str = "art_plot_01",
    *,
    produced_by_verifier: bool = True,
    sha256: str | None = None,
    create_file: bool = True,
) -> ArtifactRef:
    file_path = tmp_path / f"{artifact_id}.png"
    payload = b"\x89PNG\r\n\x1a\ndeterministic-slice-bytes"
    if create_file:
        file_path.write_bytes(payload)
    digest = sha256 if sha256 is not None else hashlib.sha256(payload).hexdigest()
    return ArtifactRef(
        artifact_id=artifact_id,
        kind="plot_png",
        path=str(file_path),
        sha256=digest,
        mime_type="image/png",
        produced_by_verifier=produced_by_verifier,
    )


def _seed_store(tmp_path: Path) -> tuple[NoteStore, ArtifactRef]:
    store = NoteStore()
    png_art = _make_png_artifact(tmp_path, "art_payments_3way")

    store.append(
        Note(
            note_id="n_0001",
            certification="CERTIFIED",
            verifier_name="verify_upstream_retry_spike",
            hypothesis_family="upstream_retry_spike",
            requested_under=0,
            row_count=1,
            summary="Found 1 (service, shard) retry spike(s).",
            evidence={
                "candidates": [
                    {
                        "service": "payments-db",
                        "shard_id": "shard_01",
                        "retry_log_count": 18,
                        "distinct_traces": 15,
                        "first_ts": "2026-11-21T14:03:07Z",
                        "last_ts": "2026-11-21T14:04:12Z",
                    }
                ]
            },
        )
    )
    store.append(
        Note(
            note_id="n_0002",
            certification="CERTIFIED",
            verifier_name="verify_metric_zscore_breach",
            hypothesis_family="metric_baseline_zscore_breach",
            requested_under=0,
            row_count=1,
            summary="Found 1 metric z-score breach(es).",
            evidence={
                "candidates": [
                    {
                        "service": "payments-db",
                        "shard_id": "shard_01",
                        "metric_name": "pool_utilization_pct",
                        "breach_buckets": 6,
                        "max_zscore": 6.425,
                    }
                ]
            },
        )
    )
    store.append(
        Note(
            note_id="n_0003",
            certification="UNCERTIFIED",
            verifier_name=None,
            hypothesis_family="ad_hoc_pool_check",
            requested_under=0,
            row_count=2,
            summary="Ad-hoc read-only query returned 2 row(s) in 3.2ms.",
            evidence={
                "columns": ["service", "msg"],
                "rows": [["payments-db", "pool saturation warning"], ["auth-service", "retry"]],
            },
        )
    )
    store.append(
        Note(
            note_id="n_0004",
            certification="CERTIFIED",
            verifier_name="verify_three_way_root_cause",
            hypothesis_family="cross_service_three_way_join",
            artifacts=[png_art],
            requested_under=0,
            row_count=1,
            summary="CERTIFIED 3-way root cause(s): ['payments-db']",
            evidence={
                "verified_root_causes": ["payments-db"],
                "candidates": [
                    {
                        "service": "payments-db",
                        "shard_id": "shard_01",
                        "correlated_traces": 15,
                        "aligned_metric_buckets": 6,
                        "max_zscore": 6.425,
                    }
                ],
            },
        )
    )
    return store, png_art


def _new_board():
    return create_initial_board(_sample_questions(), seed_fields=INCIDENT_PIVOTAL_FIELDS)


def _merge(board, patch, store, **kwargs):
    kwargs.setdefault("seed_fields", INCIDENT_PIVOTAL_FIELDS)
    kwargs.setdefault("protected_fields", INCIDENT_PIVOTAL_FIELDS)
    kwargs.setdefault("note_extractor", extract_incident_note_fields)
    return validate_and_merge_patch(board, patch, store, **kwargs)


def test_create_initial_board_defaults_and_custom_seeds() -> None:
    questions = _sample_questions()
    board = create_initial_board(questions, seed_fields=INCIDENT_PIVOTAL_FIELDS)

    assert board.board_version == 0
    assert board.decision.question_id == "root_cause"
    assert board.decision.answer == "unknown"
    assert set(board.decisions.keys()) == {
        "root_cause",
        "recommended_action",
        "sufficiently_verified",
    }
    assert list(board.fields.keys()) == INCIDENT_PIVOTAL_FIELDS
    for fld in board.fields.values():
        assert fld.value == "unknown"
        assert fld.status == "unknown"
        assert fld.sources == []
        assert fld.artifact_ids == []

    empty_seed_board = create_initial_board(questions)
    assert empty_seed_board.fields == {}

    fallback_board = create_initial_board([], seed_fields=["custom_clue"])
    assert fallback_board.decision.question_id == "primary_decision"
    assert "custom_clue" in fallback_board.fields


def test_i1_rejects_non_unknown_field_with_empty_sources(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    patch = BoardPatch(
        updates={
            "logs_retry_spike": FieldPatch(value="payments-db", sources=[]),
        }
    )
    with pytest.raises(PatchValidationError) as exc_info:
        _merge(board, patch, store)

    assert exc_info.value.invariant == "I1"
    assert exc_info.value.field_name == "logs_retry_spike"


def test_i1_rejects_missing_note_id_in_sources(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    patch = BoardPatch(
        updates={
            "logs_retry_spike": FieldPatch(value="payments-db", sources=["n_9999"]),
        }
    )
    with pytest.raises(PatchValidationError) as exc_info:
        _merge(board, patch, store)

    assert exc_info.value.invariant == "I1"
    assert exc_info.value.field_name == "logs_retry_spike"
    assert "n_9999" in str(exc_info.value)


def test_i1_unknown_value_rules_and_blocks_certified_reset(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    with pytest.raises(PatchValidationError) as exc_info:
        _merge(
            board,
            {"logs_retry_spike": {"value": "unknown", "sources": ["n_0001"]}},
            store,
        )
    assert exc_info.value.invariant == "I1"

    board, _ = _merge(
        board,
        {"logs_retry_spike": {"value": "payments-db", "sources": ["n_0001"]}},
        store,
    )
    assert board.fields["logs_retry_spike"].status == "certified"

    with pytest.raises(PatchValidationError) as exc_info2:
        _merge(
            board,
            {"logs_retry_spike": {"value": None, "sources": []}},
            store,
        )
    assert exc_info2.value.invariant == "I1"
    assert exc_info2.value.field_name == "logs_retry_spike"


def test_i1_certified_value_groundedness_checks(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    with pytest.raises(PatchValidationError) as exc_info:
        _merge(
            board,
            {"logs_retry_spike": {"value": "inventory-db", "sources": ["n_0001"]}},
            store,
        )
    assert exc_info.value.invariant == "I1"
    assert exc_info.value.field_name == "logs_retry_spike"

    with pytest.raises(PatchValidationError) as exc_num:
        _merge(
            board,
            {"max_z": {"value": 42.5, "sources": ["n_0002"]}},
            store,
        )
    assert exc_num.value.invariant == "I1"

    n1 = store.get("n_0001")
    n2 = store.get("n_0002")
    assert not value_is_grounded_in_notes(["payments-db", "inventory-db"], [n1])
    assert not value_is_grounded_in_notes({"service": "inventory-db"}, [n1])
    assert not value_is_grounded_in_notes([], [n1])
    assert not value_is_grounded_in_notes({}, [n1])

    assert value_is_grounded_in_notes("payments-db", [n1])
    assert value_is_grounded_in_notes(6.42, [n2])
    assert value_is_grounded_in_notes(True, [n1])
    assert value_is_grounded_in_notes(
        {"service": "payments-db", "shard": "shard_01", "traces": 15}, [n1]
    )

    empty_note = Note(
        note_id="n_empty",
        certification="CERTIFIED",
        verifier_name="verify_trace_error_cluster",
        row_count=0,
        evidence={"candidates": []},
        summary="No downstream trace error cluster met threshold.",
    )
    assert value_is_grounded_in_notes("none", [empty_note])
    assert value_is_grounded_in_notes("none_verified", [empty_note])
    assert value_is_grounded_in_notes([], [empty_note])


def test_i1_multimodal_artifact_provenance_enforcement(tmp_path: Path) -> None:
    store, valid_art = _seed_store(tmp_path)
    board = _new_board()

    with pytest.raises(PatchValidationError) as exc_missing:
        _merge(
            board,
            {
                "three_way_root_cause": {
                    "value": "payments-db",
                    "sources": ["n_0004"],
                    "artifact_ids": ["art_does_not_exist"],
                }
            },
            store,
        )
    assert exc_missing.value.invariant == "I1"

    with pytest.raises(PatchValidationError) as exc_unlinked:
        _merge(
            board,
            {
                "logs_retry_spike": {
                    "value": "payments-db",
                    "sources": ["n_0001"],
                    "artifact_ids": [valid_art.artifact_id],
                }
            },
            store,
        )
    assert exc_unlinked.value.invariant == "I1"

    untrusted_art = _make_png_artifact(
        tmp_path, "art_untrusted", produced_by_verifier=False
    )
    store.append(
        Note(
            note_id="n_untrusted_art",
            certification="UNCERTIFIED",
            artifacts=[untrusted_art],
            row_count=1,
            evidence={"candidates": [{"service": "payments-db"}]},
        )
    )
    with pytest.raises(PatchValidationError) as exc_untrusted:
        _merge(
            board,
            {
                "plot_check": {
                    "value": "payments-db",
                    "sources": ["n_untrusted_art"],
                    "artifact_ids": ["art_untrusted"],
                }
            },
            store,
        )
    assert exc_untrusted.value.invariant == "I1"

    no_sha_art = _make_png_artifact(tmp_path, "art_no_sha", sha256="")
    store.append(
        Note(
            note_id="n_no_sha",
            certification="UNCERTIFIED",
            artifacts=[no_sha_art],
            row_count=1,
            evidence={"candidates": [{"service": "payments-db"}]},
        )
    )
    with pytest.raises(PatchValidationError) as exc_nosha:
        _merge(
            board,
            {
                "sha_check": {
                    "value": "payments-db",
                    "sources": ["n_no_sha"],
                    "artifact_ids": ["art_no_sha"],
                }
            },
            store,
        )
    assert exc_nosha.value.invariant == "I1"

    ghost_art = _make_png_artifact(tmp_path, "art_ghost", create_file=False)
    store.append(
        Note(
            note_id="n_ghost",
            certification="CERTIFIED",
            artifacts=[ghost_art],
            row_count=1,
            evidence={"candidates": [{"service": "payments-db"}]},
        )
    )
    with pytest.raises(PatchValidationError) as exc_ghost:
        _merge(
            board,
            {
                "ghost_check": {
                    "value": "payments-db",
                    "sources": ["n_ghost"],
                    "artifact_ids": ["art_ghost"],
                }
            },
            store,
            verify_artifact_files=True,
        )
    assert exc_ghost.value.invariant == "I1"

    board_ghost_ok, _ = _merge(
        board,
        {
            "ghost_check": {
                "value": "payments-db",
                "sources": ["n_ghost"],
                "artifact_ids": ["art_ghost"],
            }
        },
        store,
        verify_artifact_files=False,
    )
    assert board_ghost_ok.fields["ghost_check"].artifact_ids == ["art_ghost"]

    board_ok, _ = _merge(
        board,
        {"three_way_root_cause": {"value": "payments-db", "sources": ["n_0004"]}},
        store,
        verify_artifact_files=True,
    )
    assert board_ok.fields["three_way_root_cause"].status == "certified"
    assert board_ok.fields["three_way_root_cause"].artifact_ids == [valid_art.artifact_id]


def test_i2_rejects_illegal_certification_upgrade(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    with pytest.raises(PatchValidationError) as exc_info:
        _merge(
            board,
            BoardPatch(
                updates={
                    "pool_saturation": FieldPatch(
                        value="payments-db",
                        sources=["n_0003"],
                        status="certified",
                    )
                }
            ),
            store,
        )
    assert exc_info.value.invariant == "I2"
    assert exc_info.value.field_name == "pool_saturation"

    with pytest.raises(PatchValidationError) as exc_mixed:
        _merge(
            board,
            {
                "updates": {
                    "pool_saturation": {
                        "value": "payments-db",
                        "sources": ["n_0001", "n_0003"],
                        "status": "CERTIFIED",
                    }
                }
            },
            store,
        )
    assert exc_mixed.value.invariant == "I2"


def test_i2_weakest_link_and_same_field_merge_semantics(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    assert compute_weakest_status([], store) == "unknown"
    assert compute_weakest_status(["n_0001", "n_0002"], store) == "certified"
    assert compute_weakest_status(["n_0001", "n_0003"], store) == "uncertified"

    board, _ = _merge(
        board,
        {"mixed_signal": {"value": "payments-db", "sources": ["n_0001", "n_0003"]}},
        store,
    )
    assert board.fields["mixed_signal"].status == "uncertified"

    board, _ = _merge(
        board,
        {"logs_retry_spike": {"value": "payments-db", "sources": ["n_0001"]}},
        store,
    )
    assert board.fields["logs_retry_spike"].status == "certified"

    with pytest.raises(PatchValidationError) as exc_conflict:
        _merge(
            board,
            {"logs_retry_spike": {"value": "auth-service", "sources": ["n_0003"]}},
            store,
        )
    assert exc_conflict.value.invariant == "I2"

    board, _ = _merge(
        board,
        {"logs_retry_spike": {"value": "payments-db", "sources": ["n_0003"]}},
        store,
    )
    assert board.fields["logs_retry_spike"].sources == ["n_0001", "n_0003"]
    assert board.fields["logs_retry_spike"].status == "uncertified"

    board, _ = _merge(
        board,
        {"hunch_service": {"value": "auth-service", "sources": ["n_0003"]}},
        store,
    )
    assert board.fields["hunch_service"].status == "uncertified"
    board, _ = _merge(
        board,
        {"hunch_service": {"value": "payments-db", "sources": ["n_0001"]}},
        store,
    )
    assert board.fields["hunch_service"].value == "payments-db"
    assert board.fields["hunch_service"].sources == ["n_0001"]
    assert board.fields["hunch_service"].status == "certified"


def test_i2_deduplicates_alias_fields_with_union_sources_and_weakest_cert(
    tmp_path: Path,
) -> None:
    store, png_art = _seed_store(tmp_path)
    board = _new_board()

    board, report1 = _merge(
        board,
        {
            "logs_retry_spike": {"value": "payments-db", "sources": ["n_0001"]},
            "metrics_zscore_breach": {"value": "payments-db", "sources": ["n_0002"]},
            "culprit_cluster": {
                "value": {"service": "payments-db", "shard_id": "shard_01"},
                "sources": ["n_0004"],
            },
        },
        store,
    )
    assert report1.deduplicated_fields == []
    assert board.fields["culprit_cluster"].status == "certified"
    assert board.fields["culprit_cluster"].artifact_ids == [png_art.artifact_id]

    board2, report2 = _merge(
        board,
        {
            "retry_spike_alias": {"value": "payments-db", "sources": ["n_0001"]},
            "culprit_cluster_dup": {
                "value": {"service": "payments-db", "shard_id": "shard_01"},
                "sources": ["n_0003"],
            },
        },
        store,
    )

    assert set(report2.deduplicated_fields) == {"retry_spike_alias", "culprit_cluster_dup"}
    assert "retry_spike_alias" not in board2.fields
    assert "culprit_cluster_dup" not in board2.fields
    assert "logs_retry_spike" in board2.fields
    assert "metrics_zscore_breach" in board2.fields
    merged_cluster = board2.fields["culprit_cluster"]
    assert merged_cluster.sources == ["n_0004", "n_0003"]
    assert merged_cluster.artifact_ids == [png_art.artifact_id]
    assert merged_cluster.status == "uncertified"

    direct_fields = {
        "db_finding": BoardField(
            value="payments-db", status="certified", sources=["n_0001"]
        ),
        "db_finding_alias": BoardField(
            value="payments-db", status="uncertified", sources=["n_0001", "n_0003"]
        ),
    }
    deduped, aliases, mapping = deduplicate_fields(direct_fields, store)
    assert aliases == ["db_finding_alias"]
    assert mapping == {"db_finding_alias": "db_finding"}
    assert deduped["db_finding"].sources == ["n_0001", "n_0003"]
    assert deduped["db_finding"].status == "uncertified"


def test_i3_notestore_immutability_and_deterministic_rebuild_drift(tmp_path: Path) -> None:
    store, png_art = _seed_store(tmp_path)
    note1 = store.get("n_0001")

    with pytest.raises(ValidationError):
        note1.certification = "UNCERTIFIED"  # type: ignore[misc]

    with pytest.raises(ValueError, match="already exists"):
        store.append(note1)

    det_patch = deterministic_patch_from_notes(
        store.snapshot(), note_extractor=extract_incident_note_fields
    )
    assert det_patch.updates["logs_retry_spike"].value == "payments-db"
    assert det_patch.updates["metrics_zscore_breach"].value == "payments-db"
    assert det_patch.updates["three_way_root_cause"].value == "payments-db"
    assert det_patch.updates["three_way_root_cause"].artifact_ids == [png_art.artifact_id]
    assert "note_ad_hoc_pool_check" in det_patch.updates

    multi_and_empty = [
        Note(
            note_id="n_multi",
            certification="CERTIFIED",
            verifier_name="verify_trace_error_cluster",
            row_count=2,
            evidence={"candidates": [{"service": "auth-service"}, {"service": "payments-db"}]},
        ),
        Note(
            note_id="n_zero",
            certification="UNCERTIFIED",
            verifier_name="verify_upstream_retry_spike",
            row_count=0,
            evidence={"candidates": []},
        ),
    ]
    det_me = deterministic_patch_from_notes(
        multi_and_empty, note_extractor=extract_incident_note_fields
    )
    assert det_me.updates["traces_error_cluster"].value == ["auth-service", "payments-db"]
    assert det_me.updates["logs_retry_spike"].value == "none"

    rebuilt_a = rebuild_board_from_notes(
        store,
        _sample_questions(),
        seed_fields=INCIDENT_PIVOTAL_FIELDS,
        protected_fields=INCIDENT_PIVOTAL_FIELDS,
        note_extractor=extract_incident_note_fields,
    )
    rebuilt_b = rebuild_board_from_notes(
        store,
        _sample_questions(),
        seed_fields=INCIDENT_PIVOTAL_FIELDS,
        protected_fields=INCIDENT_PIVOTAL_FIELDS,
        note_extractor=extract_incident_note_fields,
    )
    assert rebuilt_a.model_dump() == rebuilt_b.model_dump()

    board = _new_board()
    board, report = _merge(
        board,
        {
            "logs_retry_spike": {"value": "payments-db", "sources": ["n_0001"]},
            "first_conn_error_ts": {
                "value": "2026-11-21T14:03:07Z",
                "sources": ["n_0001"],
            },
            "pool_hunch": {"value": "pool saturation warning", "sources": ["n_0003"]},
        },
        store,
    )
    assert report.board_drift == 0.0
    assert compute_board_drift(board, rebuilt_a) == 0.0

    tampered_status = board.model_copy(deep=True)
    tampered_status.fields["pool_hunch"].status = "certified"
    assert compute_board_drift(tampered_status, rebuilt_a) > 0.0

    tampered_val = board.model_copy(deep=True)
    tampered_val.fields["logs_retry_spike"].value = "inventory-db"
    assert compute_board_drift(tampered_val, rebuilt_a) > 0.0

    tampered_src = board.model_copy(deep=True)
    tampered_src.fields["logs_retry_spike"].sources = ["n_phantom"]
    assert compute_board_drift(tampered_src, rebuilt_a) > 0.0


def test_i3_end_to_end_with_micro_bench_verifiers(tmp_path: Path) -> None:
    data_dir = tmp_path / "haystack"
    art_dir = tmp_path / "artifacts"
    domain = MicroBenchDomain()
    domain.generate(str(data_dir), size_gb=0.001, seed=7)
    needle = domain.needle_menu()[0]
    truth = domain.plant(str(data_dir), needle)

    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    backend = DuckDBBackend(str(data_dir))
    store = NoteStore()

    for vname in (
        "verify_upstream_retry_spike",
        "verify_metric_zscore_breach",
        "verify_trace_error_cluster",
        "verify_three_way_root_cause",
    ):
        backend.execute_verifier(
            registry,
            vname,
            params={"artifact_dir": str(art_dir)},
            requested_under=0,
            note_store=store,
        )

    board = create_initial_board(
        domain.decision_questions(), seed_fields=domain.pivotal_fields()
    )
    patch = deterministic_patch_from_notes(
        store.snapshot(), note_extractor=domain.extract_field_patches_from_note
    )
    fused_board, report = validate_and_merge_patch(
        board,
        patch,
        store,
        seed_fields=domain.pivotal_fields(),
        protected_fields=domain.pivotal_fields(),
        note_extractor=domain.extract_field_patches_from_note,
    )

    assert report.board_drift == 0.0
    assert fused_board.fields["three_way_root_cause"].status == "certified"
    assert fused_board.fields["three_way_root_cause"].value == truth.primary_target
    assert len(fused_board.fields["three_way_root_cause"].artifact_ids) >= 1


def test_i2_i3_under_determined_and_decoy_cases_preserve_honest_certification(
    tmp_path: Path,
) -> None:
    domain = MicroBenchDomain()
    menu = domain.needle_menu()
    decoy_needle = next(n for n in menu if n.category == "look_alike_decoy")
    under_needle = next(n for n in menu if n.category == "under_determined")

    decoy_dir = tmp_path / "decoy_case"
    domain.generate(str(decoy_dir), size_gb=0.001, seed=11)
    decoy_truth = domain.plant(str(decoy_dir), decoy_needle)
    reg = InMemoryVerifierRegistry()
    domain.register_verifiers(reg)
    decoy_backend = DuckDBBackend(str(decoy_dir))
    decoy_store = NoteStore()
    for vname in reg.list_verifiers():
        decoy_backend.execute_verifier(reg, vname, note_store=decoy_store)

    decoy_board, decoy_rep = validate_and_merge_patch(
        create_initial_board(
            domain.decision_questions(), seed_fields=domain.pivotal_fields()
        ),
        deterministic_patch_from_notes(
            decoy_store.snapshot(), note_extractor=domain.extract_field_patches_from_note
        ),
        decoy_store,
        seed_fields=domain.pivotal_fields(),
        protected_fields=domain.pivotal_fields(),
        note_extractor=domain.extract_field_patches_from_note,
    )
    assert decoy_rep.board_drift == 0.0
    assert decoy_board.fields["three_way_root_cause"].status == "certified"
    assert decoy_board.fields["three_way_root_cause"].value == decoy_truth.primary_target
    assert decoy_board.fields["three_way_root_cause"].value != decoy_truth.decoy_target

    under_dir = tmp_path / "under_case"
    domain.generate(str(under_dir), size_gb=0.001, seed=19)
    domain.plant(str(under_dir), under_needle)
    under_backend = DuckDBBackend(str(under_dir))
    under_store = NoteStore()
    for vname in reg.list_verifiers():
        under_backend.execute_verifier(reg, vname, note_store=under_store)

    under_board, under_rep = validate_and_merge_patch(
        create_initial_board(
            domain.decision_questions(), seed_fields=domain.pivotal_fields()
        ),
        deterministic_patch_from_notes(
            under_store.snapshot(), note_extractor=domain.extract_field_patches_from_note
        ),
        under_store,
        seed_fields=domain.pivotal_fields(),
        protected_fields=domain.pivotal_fields(),
        note_extractor=domain.extract_field_patches_from_note,
    )
    assert under_rep.board_drift == 0.0
    assert under_board.fields["three_way_root_cause"].status == "uncertified"
    assert under_board.fields["three_way_root_cause"].value == "none"


def test_i5_tracks_stale_note_arrivals_and_versions(tmp_path: Path) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    board, r1 = _merge(
        board,
        {"logs_retry_spike": {"value": "payments-db", "sources": ["n_0001"]}},
        store,
    )
    assert r1.board_version == 1
    assert r1.stale_fields == []

    fresh_note = Note(
        note_id="n_fresh_v1",
        certification="CERTIFIED",
        verifier_name="verify_trace_error_cluster",
        requested_under=1,
        row_count=1,
        evidence={"candidates": [{"service": "payments-db"}]},
    )
    store.append(fresh_note)
    board, r2 = _merge(
        board,
        {"traces_error_cluster": {"value": "payments-db", "sources": ["n_fresh_v1"]}},
        store,
    )
    assert r2.board_version == 2
    assert r2.stale_fields == []

    board, r3 = _merge(
        board,
        {"metrics_zscore_breach": {"value": "payments-db", "sources": ["n_0002"]}},
        store,
    )
    assert r3.board_version == 3
    assert r3.stale_fields == ["metrics_zscore_breach"]
    fld = board.fields["metrics_zscore_breach"]
    assert fld.requested_under == 0
    assert fld.last_updated_version == 3


def test_i7_archives_low_influence_after_two_stale_rounds_and_caps_at_12(
    tmp_path: Path,
) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()
    board.fields["traces_error_cluster"].voi = 0.45

    board, r1 = _merge(
        board,
        {
            "logs_retry_spike": {"value": "payments-db", "sources": ["n_0001"]},
            "low_inf_distractor": {"value": "shard_01", "sources": ["n_0001"]},
            "high_inf_clue": {"value": "2026-11-21T14:03:07Z", "sources": ["n_0001"]},
        },
        store,
    )
    assert r1.archived_fields == []
    board.fields["logs_retry_spike"].influence = 0.005
    board.fields["low_inf_distractor"].influence = 0.01
    board.fields["high_inf_clue"].influence = 0.35

    board, r2 = _merge(
        board,
        {"metrics_zscore_breach": {"value": "payments-db", "sources": ["n_0002"]}},
        store,
    )
    assert r2.archived_fields == []
    assert board.fields["low_inf_distractor"].stale_rounds == 1

    board, r3 = _merge(
        board,
        {"three_way_root_cause": {"value": "payments-db", "sources": ["n_0004"]}},
        store,
    )
    assert r3.archived_fields == ["low_inf_distractor"]
    assert "low_inf_distractor" not in board.fields
    assert "low_inf_distractor" in board.archived_fields
    assert "logs_retry_spike" in board.fields
    assert "traces_error_cluster" in board.fields
    assert "high_inf_clue" in board.fields
    assert store.exists("n_0001")

    many_updates: dict[str, dict[str, object]] = {}
    for idx in range(12):
        nid = f"n_extra_{idx:02d}"
        store.append(
            Note(
                note_id=nid,
                certification="UNCERTIFIED",
                row_count=1,
                evidence={"val": f"extra_val_{idx}"},
            )
        )
        many_updates[f"extra_field_{idx:02d}"] = {
            "value": f"extra_val_{idx}",
            "sources": [nid],
        }

    capped_board, r_cap = _merge(board, many_updates, store)
    assert len(capped_board.fields) <= 12
    assert len(r_cap.archived_fields) >= 5
    assert r_cap.board_drift == 0.0
    for piv in INCIDENT_PIVOTAL_FIELDS:
        assert piv in capped_board.fields
    assert "high_inf_clue" in capped_board.fields

    after_helper, helper_archived = archive_low_influence_fields(
        capped_board,
        protected_fields=INCIDENT_PIVOTAL_FIELDS,
        max_active_fields=10,
    )
    assert len(after_helper.fields) <= 10
    assert len(helper_archived) == 2


def test_non_strict_mode_skips_invalid_fields_and_merges_valid_ones(
    tmp_path: Path,
) -> None:
    store, _ = _seed_store(tmp_path)
    board = _new_board()

    raw_patch = {
        "fields": {
            "bad_no_sources": {"value": "payments-db", "sources": []},
            "bad_missing_note": {"value": "payments-db", "sources": ["n_missing"]},
            "bad_ungrounded": {"value": "inventory-db", "sources": ["n_0001"]},
            "bad_illegal_upgrade": {
                "value": "payments-db",
                "sources": ["n_0003"],
                "status": "certified",
            },
            "logs_retry_spike": {"value": "payments-db", "sources": ["n_0001"]},
            "pool_note": {"value": "pool saturation warning", "sources": ["n_0003"]},
        },
        "rationale": "Mixed batch with hallucinated and valid fields",
    }

    merged_board, report = _merge(board, raw_patch, store, strict=False)

    assert isinstance(report, PatchReport)
    assert report.board_version == 1
    assert set(report.applied_fields) == {"logs_retry_spike", "pool_note"}
    assert set(report.rejected_fields.keys()) == {
        "bad_no_sources",
        "bad_missing_note",
        "bad_ungrounded",
        "bad_illegal_upgrade",
    }
    assert "[I1:bad_no_sources]" in report.rejected_fields["bad_no_sources"]
    assert "[I1:bad_ungrounded]" in report.rejected_fields["bad_ungrounded"]
    assert "[I2:bad_illegal_upgrade]" in report.rejected_fields["bad_illegal_upgrade"]

    assert merged_board.fields["logs_retry_spike"].status == "certified"
    assert merged_board.fields["pool_note"].status == "uncertified"
    assert report.board_drift == 0.0
