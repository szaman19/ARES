"""Unit and integration tests for core ARES TaskGraph engine, boundary hygiene, and DuckDBBackend."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import duckdb
import pytest

from ares import (
    HypothesisSpec,
    InMemoryVerifierRegistry,
    Note,
    NoteStore,
    ProcessorResult,
    SolveEvent,
    TaskGraph,
    TaskSpec,
    VerifierResult,
    solve,
)
from ares.adapters import (
    DuckDBBackend,
    SQLGuardConfig,
    SQLValidationError,
    normalize_sql,
    sql_content_hash,
    validate_readonly_sql,
)
from examples.incident_rca.domain import MicroBenchDomain


@pytest.fixture(scope="module")
def planted_micro_bench(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Generate a micro-bench dataset with one planted 3-way multi-clue needle."""
    data_dir = tmp_path_factory.mktemp("ares_a2_data")
    artifact_dir = tmp_path_factory.mktemp("ares_a2_artifacts")
    domain = MicroBenchDomain()
    domain.generate(str(data_dir), size_gb=0.02, seed=42)
    needle = domain.needle_menu()[0]
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
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM read_json_auto('/tmp/data.json')",
        "SELECT * FROM parquet_metadata('/tmp/file.parquet')",
        "SELECT * FROM '/etc/passwd.parquet'",
        "SELECT * FROM service_logs JOIN '/tmp/other.parquet' ON true",
        "SELECT $$unparsed dollar string$$",
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

    quoted_view = 'SELECT EXTRACT(EPOCH FROM ts), TRIM(\' \' FROM service) FROM "service_logs"'
    assert validate_readonly_sql(quoted_view, cfg).startswith("SELECT")

    escaped = f"SELECT * FROM read_parquet('{tmp_path}/../etc/passwd')"
    with pytest.raises(SQLValidationError, match="escapes allowed data directory"):
        validate_readonly_sql(escaped, cfg)

    escaped_list = (
        f"SELECT * FROM read_parquet(['{tmp_path}/service_logs/a.parquet', "
        f"'{tmp_path}/../etc/passwd'])"
    )
    with pytest.raises(SQLValidationError, match="escapes allowed data directory"):
        validate_readonly_sql(escaped_list, cfg)


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
    assert sql_content_hash(q1, hypothesis_family="retries") == sql_content_hash(
        q2, hypothesis_family="retries"
    )
    assert sql_content_hash(q1, hypothesis_family="retries") != sql_content_hash(
        q1, hypothesis_family="latency"
    )


def test_explain_cost_estimation_and_budget_cap(planted_micro_bench: dict[str, object]) -> None:
    backend = DuckDBBackend(str(planted_micro_bench["data_dir"]))
    cost = backend.estimate_cost(
        "SELECT service, COUNT(*) AS cnt FROM service_logs WHERE level = 'ERROR' GROUP BY service"
    )
    assert cost.within_budget is True
    assert cost.estimated_units > 0
    assert cost.estimated_bytes > 0
    assert cost.plan_summary != ""

    tight_cfg = SQLGuardConfig(
        max_estimated_rows=10,
        allowed_data_dir=str(planted_micro_bench["data_dir"]),
    )
    tight_cost = backend.estimate_cost(
        "SELECT * FROM service_logs",
        config=tight_cfg,
    )
    assert tight_cost.within_budget is False
    assert tight_cost.rejection_reason is not None

    with pytest.raises(SQLValidationError, match="exceed guard cap"):
        backend.execute_query("SELECT * FROM service_logs", config=tight_cfg)


def test_note_store_append_only_and_lookups() -> None:
    store = NoteStore()
    n1 = Note(
        note_id=store.next_note_id(),
        certification="CERTIFIED",
        content_hash="abc123",
        summary="verified",
    )
    store.append(n1)
    assert store.exists("n_0001")
    assert store.get("n_0001").summary == "verified"
    assert store.has_content_hash("abc123")
    assert store.get_by_content_hash("abc123") == n1

    with pytest.raises(ValueError, match="already exists"):
        store.append(n1)


def test_note_store_search_ranking_and_serialization() -> None:
    store = NoteStore()
    n1 = store.append(
        Note(
            note_id=store.next_note_id(),
            certification="UNCERTIFIED",
            hypothesis_family="logs_scan",
            query="inspect:cache",
            summary="Cache pool timeout warning observed",
            evidence={"module": "TokenCache", "count": 2},
            row_count=1,
        )
    )
    n2 = store.append(
        Note(
            note_id=store.next_note_id(),
            certification="CERTIFIED",
            verifier_name="verify_lock_inversion",
            hypothesis_family="lock_graph",
            query="inspect:lock_cycle",
            summary="Deadlock lock inversion confirmed in SessionPool",
            evidence={"module": "SessionPool", "cycle": ["L1", "L2"]},
            row_count=1,
        )
    )
    n3 = store.append(
        Note(
            note_id=store.next_note_id(),
            certification="CERTIFIED",
            verifier_name="verify_race_repro",
            hypothesis_family="stress_repro",
            query="inspect:race_repro",
            summary="Deterministic race repro passed for SessionPool deadlock",
            evidence={"module": "SessionPool", "repro": True},
            row_count=1,
        )
    )

    hits = store.search("SessionPool deadlock", limit=5)
    assert [n.note_id for n in hits] == [n3.note_id, n2.note_id]

    cert_hits = store.search("SessionPool", certification="CERTIFIED", verifier_name="verify_lock_inversion")
    assert [n.note_id for n in cert_hits] == [n2.note_id]

    recent = store.search("", limit=2)
    assert [n.note_id for n in recent] == [n3.note_id, n2.note_id]
    assert n1.note_id not in {n.note_id for n in recent}

    restored = NoteStore.from_dicts(store.to_dicts())
    assert len(restored) == 3
    assert restored.get(n2.note_id).summary == n2.summary
    assert restored.next_note_id() == "n_0004"


def test_adhoc_sql_emits_uncertified_note_with_row_cap(
    planted_micro_bench: dict[str, object],
) -> None:
    cfg = SQLGuardConfig(
        max_result_rows=5,
        allowed_data_dir=str(planted_micro_bench["data_dir"]),
    )
    backend = DuckDBBackend(str(planted_micro_bench["data_dir"]), guard_config=cfg)
    store = NoteStore()

    res, note = backend.execute_query(
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
    backend = DuckDBBackend(str(planted_micro_bench["data_dir"]))
    registry = planted_micro_bench["registry"]
    assert isinstance(registry, InMemoryVerifierRegistry)
    store = NoteStore()

    vres, note = backend.execute_verifier(
        registry,
        "verify_three_way_root_cause",
        params={"artifact_dir": str(planted_micro_bench["artifact_dir"])},
        note_store=store,
    )
    assert vres.passed is True
    assert note.certification == "CERTIFIED"
    assert len(note.artifacts) >= 1

    # Untrusted verifier taint enforcement
    reg = InMemoryVerifierRegistry()

    def _fake_untrusted(
        con: duckdb.DuckDBPyConnection,
        data_dir: str,
        params: dict[str, object] | None = None,
    ) -> VerifierResult:
        del con, data_dir, params
        return VerifierResult(
            verifier_name="untrusted_llm_guess",
            passed=True,
            certification="CERTIFIED",
            evidence={"guess": "payments-db"},
            row_count=1,
            summary="Untrusted claim",
        )

    reg.register("untrusted_llm_guess", _fake_untrusted, trusted=False)
    vres_u, note_u = backend.execute_verifier(reg, "untrusted_llm_guess", note_store=store)
    assert vres_u.certification == "UNCERTIFIED"
    assert note_u.certification == "UNCERTIFIED"


def test_batch_dedup_guard_and_cheapest_first_order(
    planted_micro_bench: dict[str, object],
) -> None:
    backend = DuckDBBackend(str(planted_micro_bench["data_dir"]))
    store = NoteStore()

    specs = [
        HypothesisSpec(
            query="SELECT service, COUNT(*) FROM service_logs GROUP BY service",
            hypothesis_family="logs_scan",
            expected_signal="counts",
            requested_under=1,
        ),
        HypothesisSpec(
            query="  select   service, count(*)  from service_logs  group by service; ",
            hypothesis_family="logs_scan",
            expected_signal="counts",
            requested_under=1,
        ),
        HypothesisSpec(
            query="SELECT metric_name, COUNT(*) FROM metrics GROUP BY metric_name",
            hypothesis_family="metrics_scan",
            expected_signal="counts",
            requested_under=1,
        ),
        HypothesisSpec(
            query="DROP TABLE service_logs",
            hypothesis_family="evil",
            expected_signal="none",
            requested_under=1,
        ),
    ]

    notes, stats = backend.plan_and_execute_batch(specs, note_store=store, max_concurrent=2)
    assert stats["submitted"] == 4
    assert stats["executed"] == 2
    assert stats["skipped_duplicates"] == 1
    assert stats["rejected_guard"] == 1
    assert len(notes) == 2

    metrics_hash = sql_content_hash(specs[2].query, hypothesis_family="metrics_scan")
    logs_hash = sql_content_hash(specs[0].query, hypothesis_family="logs_scan")
    assert stats["execution_order_hashes"] == [metrics_hash, logs_hash]


def test_solve_parallel_max_concurrent_and_callbacks() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()
    events: list[SolveEvent] = []

    def _slow_worker(spec: TaskSpec, deps: dict[str, object]) -> ProcessorResult:
        del deps
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
    assert report.usage.peak_prompt_tokens == 100
    assert 2 <= peak <= 3
    assert any(e.event_type == "task_started" for e in events)
    assert sum(1 for e in events if e.event_type == "task_completed") == 6


def test_solve_extends_existing_graph_and_deduplicates_across_rounds(
    planted_micro_bench: dict[str, object],
) -> None:
    backend = DuckDBBackend(str(planted_micro_bench["data_dir"]))
    registry = planted_micro_bench["registry"]
    assert isinstance(registry, InMemoryVerifierRegistry)
    store = NoteStore()

    round1_tasks = [
        TaskSpec(
            task_id="r1_retry",
            task_type="verify",
            verifier_name="verify_upstream_retry_spike",
            requested_under=1,
            context={"board_claim": "should_never_reach_verifier"},
        ),
        TaskSpec(
            task_id="r1_query",
            task_type="query",
            requested_under=1,
            hypothesis_spec=HypothesisSpec(
                query="SELECT service, COUNT(*) AS n FROM traces WHERE status = 'ERROR' GROUP BY service",
                hypothesis_family="downstream_trace_timeout_cluster",
                expected_signal="error_counts",
                requested_under=1,
            ),
        ),
    ]
    report1 = solve(
        tasks=round1_tasks,
        backend=backend,
        registry=registry,
        note_store=store,
        max_concurrent=2,
    )
    assert report1.usage.completed_tasks == 2
    assert report1.usage.certified_notes == 1
    assert report1.usage.uncertified_notes == 1
    assert len(store) == 2

    round2_tasks = [
        TaskSpec(
            task_id="r2_dup_query",
            task_type="query",
            requested_under=2,
            hypothesis_spec=HypothesisSpec(
                query="  select service, count(*) as n from traces where status = 'ERROR' group by service; ",
                hypothesis_family="downstream_trace_timeout_cluster",
                expected_signal="error_counts",
                requested_under=2,
            ),
        ),
        TaskSpec(
            task_id="r2_bad_query",
            task_type="query",
            requested_under=2,
            hypothesis_spec=HypothesisSpec(
                query="DELETE FROM traces",
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
        backend=backend,
        registry=registry,
        note_store=store,
        max_concurrent=2,
    )
    assert report2.graph is report1.graph
    assert report2.graph.nodes["r2_dup_query"].status == "skipped_duplicate"
    assert report2.graph.nodes["r2_bad_query"].status == "rejected_guard"
    assert report2.graph.nodes["r2_three_way"].status == "completed"
    assert report2.usage.skipped_duplicate_tasks == 1
    assert report2.usage.rejected_guard_tasks == 1
    assert report2.usage.completed_tasks == 1
    assert report2.usage.certified_notes == 1
    assert report2.usage.artifacts_produced >= 1
    assert len(store) == 3


def test_invariant_i4_context_never_passed_to_verifier(
    planted_micro_bench: dict[str, object],
) -> None:
    received_params: dict[str, object] = {}

    def _spy_verifier(
        con: duckdb.DuckDBPyConnection,
        data_dir: str,
        params: dict[str, object] | None = None,
    ) -> VerifierResult:
        del con, data_dir
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
    backend = DuckDBBackend(str(planted_micro_bench["data_dir"]))

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
        backend=backend,
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


def test_strict_library_and_implementation_boundaries() -> None:
    """Enforce zero DuckDB/SQL leaks, zero domain leaks, and zero decision-module tokens in `ares/`."""
    repo_root = Path(__file__).resolve().parent.parent

    domain_leaks = (
        "service_logs",
        "logs_retry_spike",
        "metrics_zscore_breach",
        "traces_error_cluster",
        "three_way_root_cause",
        "payments-db",
        "inventory-db",
        "scale_pool",
    )
    decision_module_tokens = (
        "decision_probe",
        "DecisionProbeClient",
        "gemini-decision-1",
        "decisions.create",
        "DecisionQuestion",
        "BoardDecision",
        "uncertified_influence_share",
        "unknown_voi_max",
    )

    # 1. Task DAG engine (`ares/engine.py`) and provenance store (`ares/store.py`)
    #    must know nothing about SQL/DuckDB/Parquet, InvestigationBoard, domains, or decision modules.
    engine_forbidden = (
        "duckdb",
        "parquet",
        "sql",
        "InvestigationBoard",
        "BoardField",
        "validate_and_merge_patch",
        "fusion",
        *domain_leaks,
        *decision_module_tokens,
    )
    for fname in ("engine.py", "store.py"):
        py_file = repo_root / "ares" / fname
        src_lower = py_file.read_text(encoding="utf-8").lower()
        for token in engine_forbidden:
            assert token.lower() not in src_lower, (
                f"Narrow-waist violation: {token!r} found in ares/{fname}"
            )

    # 2. Contracts, Context, Loop, and Models must be free of SQL/DuckDB/Parquet,
    #    domain strings, and decision-module tokens.
    non_adapter_core = [
        repo_root / "ares" / "contracts.py",
        repo_root / "ares" / "context.py",
        repo_root / "ares" / "loop.py",
        *list((repo_root / "ares" / "models").glob("*.py")),
    ]
    for py_file in non_adapter_core:
        src_lower = py_file.read_text(encoding="utf-8").lower()
        forbidden = (
            "duckdb",
            "parquet",
            "read_parquet",
            *domain_leaks,
            *decision_module_tokens,
        )
        if py_file.parent.name != "models":
            forbidden = (*forbidden, "sql")
        for token in forbidden:
            assert token.lower() not in src_lower, (
                f"Implementation leak: {token!r} found in {py_file.relative_to(repo_root)}"
            )

    # 3. Adapters (`ares/adapters/`) must be free of domain strings and decision-module tokens.
    for py_file in (repo_root / "ares" / "adapters").glob("*.py"):
        src_lower = py_file.read_text(encoding="utf-8").lower()
        for token in (*domain_leaks, *decision_module_tokens):
            assert token.lower() not in src_lower, (
                f"Adapter leak: {token!r} found in {py_file.relative_to(repo_root)}"
            )

    # 4. Importing top-level `ares` must NOT eagerly import `duckdb` or `ares.adapters`.
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, ares; "
                "assert 'duckdb' not in sys.modules, 'duckdb imported eagerly'; "
                "assert 'ares.adapters' not in sys.modules, 'ares.adapters imported eagerly'"
            ),
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Eager adapter import check failed: {proc.stderr}"
