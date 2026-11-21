"""Unit and integration tests for Agent A2: ARES TaskGraph engine & DuckDB TRUSTED runner."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import duckdb
import pytest

from ares import (
    DuckDBTrustedRunner,
    NoteStore,
    ProcessorResult,
    SQLGuardConfig,
    SQLValidationError,
    SolveEvent,
    TaskGraph,
    TaskSpec,
    normalize_sql,
    query_content_hash,
    solve,
    validate_readonly_sql,
)
from contracts import (
    HypothesisQuerySpec,
    InMemoryVerifierRegistry,
    Note,
    VerifierResult,
)
from domain.micro_bench import MicroBenchDomain


@pytest.fixture(scope="module")
def planted_micro_bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Generate a micro-bench dataset with one planted 3-way multi-clue needle."""
    data_dir = tmp_path_factory.mktemp("ares_a2_data")
    artifact_dir = tmp_path_factory.mktemp("ares_a2_artifacts")
    domain = MicroBenchDomain()
    domain.generate(str(data_dir), size_gb=0.02, seed=42)
    needle = domain.needle_menu()[0]  # mc_01_payments_db_lock (multi_clue_join)
    truth = domain.plant(str(data_dir), needle)
    registry = InMemoryVerifierRegistry()
    domain.register_verifiers(registry)
    return {
        "data_dir": str(data_dir),
        "artifact_dir": str(artifact_dir),
        "domain": domain,
        "needle": needle,
        "truth": truth,
        "registry": registry,
    }


# ----------------------------------------------------------------------
# 1. Read-Only SQL Validation & Content-Hash Dedup (Invariant I6)
# ----------------------------------------------------------------------


def test_validate_readonly_sql_allows_select_and_with() -> None:
    sql1 = "-- comment\nSELECT service, COUNT(*) AS n FROM service_logs WHERE level = 'ERROR' GROUP BY service;"
    norm = validate_readonly_sql(sql1)
    assert norm.startswith("SELECT service, COUNT(*) AS n FROM service_logs")
    assert not norm.endswith(";")

    cte_sql = """
    /* block comment */
    WITH errs AS (
        SELECT trace_id, service FROM traces WHERE status = 'ERROR'
    )
    SELECT service, COUNT(*) FROM errs GROUP BY service
    """
    norm_cte = validate_readonly_sql(cte_sql)
    assert norm_cte.startswith("WITH errs AS")


def test_validate_readonly_sql_ignores_keywords_inside_literals() -> None:
    sql = "SELECT * FROM service_logs WHERE message = 'DELETE failed; DROP not executed'"
    norm = validate_readonly_sql(sql)
    assert "'DELETE failed; DROP not executed'" in norm


@pytest.mark.parametrize(
    "bad_sql",
    [
        "",
        "   -- only a comment",
        "DROP TABLE service_logs",
        "DELETE FROM service_logs WHERE level = 'INFO'",
        "INSERT INTO service_logs VALUES (1)",
        "UPDATE service_logs SET level = 'INFO'",
        "COPY (SELECT 1) TO '/tmp/evil.parquet'",
        "PRAGMA enable_profiling",
        "SET threads = 16",
        "ATTACH '/tmp/other.db' AS other",
        "SELECT 1; DROP TABLE service_logs",
        "SELECT getenv('API_KEY')",
        "SELECT * FROM duckdb_secrets()",
        "SELECT * FROM read_parquet('https://evil.example.com/data.parquet')",
        "SELECT * FROM read_parquet('s3://bucket/file.parquet')",
    ],
)
def test_validate_readonly_sql_rejects_unsafe_queries(bad_sql: str) -> None:
    with pytest.raises(SQLValidationError):
        validate_readonly_sql(bad_sql)


def test_validate_readonly_sql_enforces_data_dir_sandbox(tmp_path: Path) -> None:
    cfg = SQLGuardConfig(allowed_data_dir=str(tmp_path))
    safe = f"SELECT * FROM read_parquet('{tmp_path}/service_logs/*.parquet')"
    assert validate_readonly_sql(safe, cfg).startswith("SELECT")

    escaped = f"SELECT * FROM read_parquet('{tmp_path}/../etc/passwd')"
    with pytest.raises(SQLValidationError, match="escapes allowed data directory"):
        validate_readonly_sql(escaped, cfg)


def test_normalize_and_content_hash_deduplication() -> None:
    q1 = "select service, count(*) from service_logs where level = 'ERROR' group by service;"
    q2 = """
        -- identical query with different formatting
        SELECT   service,   COUNT(*)
        FROM service_logs
        WHERE level = 'ERROR'
        GROUP BY service
    """
    assert normalize_sql(q1) == normalize_sql(q2)
    assert query_content_hash(q1, hypothesis_family="retries") == query_content_hash(
        q2, hypothesis_family="retries"
    )
    # Different hypothesis family produces a distinct hash
    assert query_content_hash(q1, hypothesis_family="retries") != query_content_hash(
        q1, hypothesis_family="latency"
    )


# ----------------------------------------------------------------------
# 2. EXPLAIN Cost Estimation & Budget Caps
# ----------------------------------------------------------------------


def test_explain_cost_estimation_and_budget_cap(planted_micro_bench: dict[str, object]) -> None:
    runner = DuckDBTrustedRunner(str(planted_micro_bench["data_dir"]))
    cost = runner.estimate_cost(
        "SELECT service, COUNT(*) AS cnt FROM service_logs WHERE level = 'ERROR' GROUP BY service"
    )
    assert cost.within_budget is True
    assert cost.estimated_rows > 0
    assert cost.estimated_bytes > 0
    assert cost.plan_summary != ""

    # Tight cap triggers rejection before execution
    tight_cfg = SQLGuardConfig(
        max_estimated_rows=10,
        allowed_data_dir=str(planted_micro_bench["data_dir"]),
    )
    tight_cost = runner.estimate_cost(
        "SELECT * FROM service_logs",
        config=tight_cfg,
    )
    assert tight_cost.within_budget is False
    assert tight_cost.rejection_reason is not None

    with pytest.raises(SQLValidationError, match="exceed guard cap"):
        runner.execute_readonly_query("SELECT * FROM service_logs", config=tight_cfg)


# ----------------------------------------------------------------------
# 3. Immutable NoteStore (Invariant I3)
# ----------------------------------------------------------------------


def test_note_store_append_only_and_lookups() -> None:
    store = NoteStore()
    n1 = Note(
        note_id=store.next_note_id(),
        certification="CERTIFIED",
        query_hash="abc123",
        summary="verified",
    )
    store.append(n1)
    assert store.exists("n_0001")
    assert store.get("n_0001").summary == "verified"
    assert store.has_query_hash("abc123")
    assert store.get_by_query_hash("abc123") == n1

    # Duplicate note_id must be rejected
    with pytest.raises(ValueError, match="already exists"):
        store.append(n1)


# ----------------------------------------------------------------------
# 4. DuckDBTrustedRunner: Ad-hoc SQL vs TRUSTED Verifiers & Taint Guard
# ----------------------------------------------------------------------


def test_adhoc_sql_emits_uncertified_note_with_row_cap(
    planted_micro_bench: dict[str, object],
) -> None:
    cfg = SQLGuardConfig(
        max_result_rows=5,
        allowed_data_dir=str(planted_micro_bench["data_dir"]),
    )
    runner = DuckDBTrustedRunner(str(planted_micro_bench["data_dir"]), guard_config=cfg)
    store = NoteStore()

    res, note = runner.execute_readonly_query(
        "SELECT service, level, retry_count FROM service_logs",
        requested_under=3,
        hypothesis_family="upstream_retry_spike",
        note_store=store,
    )
    assert res.row_count == 5
    assert res.truncated is True
    assert note.certification == "UNCERTIFIED"
    assert note.requested_under == 3
    assert note.note_id == "n_0001"
    assert store.exists(note.note_id)


def test_trusted_verifier_emits_certified_note_with_artifacts(
    planted_micro_bench: dict[str, object],
) -> None:
    runner = DuckDBTrustedRunner(str(planted_micro_bench["data_dir"]))
    registry = planted_micro_bench["registry"]
    assert isinstance(registry, InMemoryVerifierRegistry)
    store = NoteStore()

    vres, note = runner.execute_verifier(
        registry,
        "verify_three_way_root_cause",
        params={"artifact_dir": str(planted_micro_bench["artifact_dir"])},
        requested_under=7,
        hypothesis_family="cross_service_three_way_join",
        note_store=store,
    )
    assert vres.passed is True
    assert note.certification == "CERTIFIED"
    assert note.requested_under == 7
    assert note.verifier_name == "verify_three_way_root_cause"
    assert len(note.artifacts) >= 1
    art = note.artifacts[0]
    assert art.produced_by_verifier is True
    assert Path(art.path).is_file()
    assert store.has_artifact(art.artifact_id)


def test_untrusted_verifier_is_tainted_to_uncertified(
    planted_micro_bench: dict[str, object],
) -> None:
    runner = DuckDBTrustedRunner(str(planted_micro_bench["data_dir"]))
    reg = InMemoryVerifierRegistry()

    def _fake_untrusted(
        con: duckdb.DuckDBPyConnection,
        data_dir: str,
        params: dict[str, object] | None = None,
    ) -> VerifierResult:
        # Even if an untrusted verifier tries to claim CERTIFIED, ARES must taint it.
        return VerifierResult(
            verifier_name="untrusted_llm_guess",
            passed=True,
            certification="CERTIFIED",
            evidence={"guess": "payments-db"},
            row_count=1,
            summary="Untrusted claim",
        )

    reg.register("untrusted_llm_guess", _fake_untrusted, trusted=False)
    store = NoteStore()
    vres, note = runner.execute_verifier(reg, "untrusted_llm_guess", note_store=store)
    assert vres.certification == "UNCERTIFIED"
    assert note.certification == "UNCERTIFIED"


def test_batch_dedup_guard_and_cheapest_first_order(
    planted_micro_bench: dict[str, object],
) -> None:
    runner = DuckDBTrustedRunner(str(planted_micro_bench["data_dir"]))
    store = NoteStore()

    specs = [
        # Heavy scan (service_logs has ~110K rows)
        HypothesisQuerySpec(
            query_sql="SELECT service, COUNT(*) FROM service_logs GROUP BY service",
            hypothesis_family="logs_scan",
            expected_signal="counts",
            requested_under=1,
        ),
        # Duplicate of first query with different formatting
        HypothesisQuerySpec(
            query_sql="  select   service, count(*)  from service_logs  group by service; ",
            hypothesis_family="logs_scan",
            expected_signal="counts",
            requested_under=1,
        ),
        # Lighter scan (metrics has fewer rows than service_logs)
        HypothesisQuerySpec(
            query_sql="SELECT metric_name, COUNT(*) FROM metrics GROUP BY metric_name",
            hypothesis_family="metrics_scan",
            expected_signal="counts",
            requested_under=1,
        ),
        # Unsafe SQL that must be rejected by root guard
        HypothesisQuerySpec(
            query_sql="DROP TABLE service_logs",
            hypothesis_family="evil",
            expected_signal="none",
            requested_under=1,
        ),
    ]

    notes, stats = runner.plan_and_execute_batch(specs, note_store=store, max_concurrent=2)
    assert stats["submitted"] == 4
    assert stats["executed"] == 2
    assert stats["skipped_duplicates"] == 1
    assert stats["rejected_guard"] == 1
    assert len(notes) == 2

    # Verify cheapest-first ordering: metrics scan hash comes before service_logs scan hash
    metrics_hash = query_content_hash(specs[2].query_sql, hypothesis_family="metrics_scan")
    logs_hash = query_content_hash(specs[0].query_sql, hypothesis_family="logs_scan")
    assert stats["execution_order_hashes"] == [metrics_hash, logs_hash]


# ----------------------------------------------------------------------
# 5. ARES solve(graph=...) Engine: Parallelism, Graph Extension, I4, Callbacks
# ----------------------------------------------------------------------


def test_solve_parallel_max_concurrent_and_callbacks() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()
    events: list[SolveEvent] = []

    def _slow_worker(spec: TaskSpec, deps: dict[str, object]) -> ProcessorResult:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.06)
        with lock:
            active -= 1
        return ProcessorResult(
            output={"task": spec.task_id},
            prompt_tokens=100,
            completion_tokens=25,
            model_calls=1,
        )

    tasks = [
        TaskSpec(task_id=f"w_{i}", task_type="propose", processor_name="worker")
        for i in range(6)
    ]

    report = solve(
        tasks=tasks,
        processors={"worker": _slow_worker},
        max_concurrent=3,
        on_event=events.append,
    )

    assert report.usage.completed_tasks == 6
    assert report.usage.failed_tasks == 0
    assert report.usage.model_calls == 6
    assert report.usage.total_prompt_tokens == 600
    assert report.usage.total_completion_tokens == 150
    assert 2 <= peak <= 3
    assert any(e.event_type == "task_started" for e in events)
    assert sum(1 for e in events if e.event_type == "task_completed") == 6


def test_solve_extends_existing_graph_and_deduplicates_across_rounds(
    planted_micro_bench: dict[str, object],
) -> None:
    runner = DuckDBTrustedRunner(str(planted_micro_bench["data_dir"]))
    registry = planted_micro_bench["registry"]
    assert isinstance(registry, InMemoryVerifierRegistry)
    store = NoteStore()

    # Round 1: Run two trusted verifiers + one ad-hoc SQL query
    round1_tasks = [
        TaskSpec(
            task_id="r1_retry",
            task_type="verify",
            verifier_name="verify_upstream_retry_spike",
            requested_under=1,
            context={"board_claim": "should_never_reach_verifier"},
        ),
        TaskSpec(
            task_id="r1_sql",
            task_type="sql_query",
            requested_under=1,
            hypothesis_spec=HypothesisQuerySpec(
                query_sql="SELECT service, COUNT(*) AS n FROM traces WHERE status = 'ERROR' GROUP BY service",
                hypothesis_family="downstream_trace_timeout_cluster",
                expected_signal="error_counts",
                requested_under=1,
            ),
        ),
    ]
    report1 = solve(
        tasks=round1_tasks,
        duckdb_runner=runner,
        registry=registry,
        note_store=store,
        max_concurrent=2,
    )
    assert report1.usage.completed_tasks == 2
    assert report1.usage.certified_notes == 1
    assert report1.usage.uncertified_notes == 1
    assert len(store) == 2

    # Round 2: Extend the SAME graph with:
    # - a duplicate of r1_sql (must be skipped_duplicate)
    # - an unsafe SQL query (must be rejected_guard)
    # - a dependent 3-way verifier producing PNG artifacts
    round2_tasks = [
        TaskSpec(
            task_id="r2_dup_sql",
            task_type="sql_query",
            requested_under=2,
            hypothesis_spec=HypothesisQuerySpec(
                query_sql="  select service, count(*) as n from traces where status = 'ERROR' group by service; ",
                hypothesis_family="downstream_trace_timeout_cluster",
                expected_signal="error_counts",
                requested_under=2,
            ),
        ),
        TaskSpec(
            task_id="r2_bad_sql",
            task_type="sql_query",
            requested_under=2,
            hypothesis_spec=HypothesisQuerySpec(
                query_sql="DELETE FROM traces",
                hypothesis_family="invalid",
                expected_signal="none",
                requested_under=2,
            ),
        ),
        TaskSpec(
            task_id="r2_three_way",
            task_type="verify",
            depends_on=["r1_retry"],
            verifier_name="verify_three_way_root_cause",
            verifier_params={"artifact_dir": str(planted_micro_bench["artifact_dir"])},
            requested_under=2,
        ),
    ]

    report2 = solve(
        tasks=round2_tasks,
        graph=report1.graph,
        duckdb_runner=runner,
        registry=registry,
        note_store=store,
        max_concurrent=2,
    )
    assert report2.graph is report1.graph
    assert report2.graph.nodes["r2_dup_sql"].status == "skipped_duplicate"
    assert report2.graph.nodes["r2_bad_sql"].status == "rejected_guard"
    assert report2.graph.nodes["r2_three_way"].status == "completed"
    assert report2.usage.skipped_duplicate_tasks == 1
    assert report2.usage.rejected_guard_tasks == 1
    assert report2.usage.completed_tasks == 1
    assert report2.usage.certified_notes == 1
    assert report2.usage.artifacts_produced >= 1
    assert len(store) == 3  # Only r2_three_way appended a new note


def test_invariant_i4_context_never_passed_to_verifier(
    planted_micro_bench: dict[str, object],
) -> None:
    """Verify TaskSpec.context is strictly isolated from TRUSTED verifier inputs (I4)."""
    received_params: dict[str, object] = {}

    def _spy_verifier(
        con: duckdb.DuckDBPyConnection,
        data_dir: str,
        params: dict[str, object] | None = None,
    ) -> VerifierResult:
        received_params.update(params or {})
        return VerifierResult(
            verifier_name="spy_verifier",
            passed=True,
            certification="CERTIFIED",
            evidence={"ok": True},
            row_count=1,
            summary="spy ok",
        )

    reg = InMemoryVerifierRegistry()
    reg.register("spy_verifier", _spy_verifier, trusted=True)
    runner = DuckDBTrustedRunner(str(planted_micro_bench["data_dir"]))

    report = solve(
        tasks=[
            TaskSpec(
                task_id="i4_check",
                task_type="verify",
                verifier_name="spy_verifier",
                verifier_params={"threshold": 42},
                context={"secret_board_claim": "payments-db"},
            )
        ],
        duckdb_runner=runner,
        registry=reg,
    )
    assert report.usage.completed_tasks == 1
    assert received_params == {"threshold": 42}
    assert "secret_board_claim" not in received_params


def test_task_graph_rejects_cycles_and_missing_deps() -> None:
    g = TaskGraph()
    g.add_task(TaskSpec(task_id="a", depends_on=["b"]))
    with pytest.raises(ValueError, match="cycle"):
        g.add_task(TaskSpec(task_id="b", depends_on=["a"]))

    g2 = TaskGraph([TaskSpec(task_id="x", depends_on=["nonexistent"])])
    with pytest.raises(ValueError, match="unknown task"):
        g2.validate_dependencies()


def test_ares_library_narrow_waist_boundary() -> None:
    """ARES library must never import or reference InvestigationBoard or fusion."""
    ares_dir = Path(__file__).resolve().parent.parent / "ares"
    forbidden_tokens = ("InvestigationBoard", "BoardField", "validate_and_merge_patch", "fusion.")
    for py_file in ares_dir.glob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in src, f"Narrow-waist violation: {token!r} found in {py_file.name}"
