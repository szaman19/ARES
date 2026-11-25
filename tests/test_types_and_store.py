"""Unit tests for ARES v2 core data contracts, ArtifactStore, and CheckpointStore."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from ares import (
    ArtifactRef,
    ArtifactStore,
    CheckpointStore,
    Done,
    Event,
    Fail,
    RunBudget,
    RunResult,
    Spawn,
    Suspend,
    Task,
    TaskSpec,
    Transition,
)


def test_types_contracts_and_discriminated_union() -> None:
    ref = ArtifactRef(ref_id="art_1234", sha256="abcd", byte_size=10)
    with pytest.raises(ValidationError):
        ref.byte_size = 20  # type: ignore[misc]

    adapter = TypeAdapter(Transition)
    parsed_done = adapter.validate_python({"kind": "done", "output": {"ok": True}})
    assert isinstance(parsed_done, Done)

    parsed_spawn = adapter.validate_python(
        {"kind": "spawn", "tasks": [{"state": "worker"}], "then": {"state": "join"}}
    )
    assert isinstance(parsed_spawn, Spawn)
    assert len(parsed_spawn.tasks) == 1
    assert parsed_spawn.then is not None

    parsed_suspend = adapter.validate_python({"kind": "suspend", "reason": "need approval"})
    assert isinstance(parsed_suspend, Suspend)

    parsed_fail = adapter.validate_python({"kind": "fail", "error": "boom", "retryable": True})
    assert isinstance(parsed_fail, Fail)

    spec = TaskSpec(state="start")
    task = Task(id="t1", state=spec.state)
    assert task.status == "pending"
    budget = RunBudget()
    assert budget.max_inline_bytes == 4096

    res = RunResult(
        run_id="run_1",
        status="done",
        root_task_id="t1",
        output="ok",
        tasks={"t1": task},
    )
    roundtrip = RunResult.model_validate_json(res.model_dump_json())
    assert roundtrip == res


@pytest.mark.parametrize("use_disk", [False, True])
def test_artifact_store_roundtrip_dedup_and_slice(tmp_path: Path, use_disk: bool) -> None:
    root = tmp_path / "artifacts" if use_disk else None
    store = ArtifactStore(root)

    raw = b"hello ares v2"
    expected_sha = hashlib.sha256(raw).hexdigest()
    ref1 = store.put_bytes(raw, mime_type="application/octet-stream", name="blob.bin")
    ref2 = store.put_bytes(raw)
    assert ref1.ref_id == f"art_{expected_sha[:16]}"
    assert ref1.ref_id == ref2.ref_id
    assert store.get_bytes(ref1.ref_id) == raw

    txt = "line0\nline1\nline2\n"
    tref = store.put_text(txt)
    assert store.get_text(tref.ref_id) == txt
    assert store.read_slice(tref.ref_id, offset=6, limit=5) == "line1"

    obj = {"b": 2, "a": [1, 2, 3]}
    jref = store.put_json(obj)
    assert store.get_json(jref.ref_id) == obj
    assert len(store.list_refs()) == 3
    assert store.get_ref(jref.ref_id) == jref

    if use_disk:
        assert (root / "index.json").exists()  # type: ignore[operator]
        reloaded = ArtifactStore(root)
        assert reloaded.get_json(jref.ref_id) == obj
        assert reloaded.get_text(tref.ref_id) == txt
        assert len(reloaded.list_refs()) == 3

    with pytest.raises(KeyError):
        store.get_bytes("art_does_not_exist")
    with pytest.raises(ValueError):
        store.read_slice(tref.ref_id, offset=-1, limit=5)


def test_artifact_store_maybe_spill_and_no_double_spill(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    small_val, small_ref = store.maybe_spill("tiny", max_inline_bytes=32)
    assert small_val == "tiny"
    assert small_ref is None

    big_text = "x" * 100
    handle, ref = store.maybe_spill(big_text, max_inline_bytes=32)
    assert ref is not None
    assert ArtifactStore.is_artifact_handle(handle)
    assert ArtifactStore.extract_ref_id(handle) == ref.ref_id
    assert store.get_text(ref.ref_id) == big_text

    # Passing the handle back to maybe_spill must not double-spill.
    handle2, ref2 = store.maybe_spill(handle, max_inline_bytes=8)
    assert handle2 == handle
    assert ref2 is None

    big_obj = {"items": list(range(50))}
    obj_handle, obj_ref = store.maybe_spill(big_obj, max_inline_bytes=32)
    assert obj_ref is not None
    assert ArtifactStore.is_artifact_handle(obj_handle)
    assert store.get_json(obj_ref.ref_id) == big_obj


def test_artifact_store_thread_safety(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    def worker(idx: int) -> str:
        ref = store.put_json({"worker": idx, "payload": "z" * 20})
        return ref.ref_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        ref_ids = list(pool.map(worker, range(40)))

    assert len(set(ref_ids)) == 40
    assert len(store.list_refs()) == 40


def test_checkpoint_store_atomic_state_and_events(tmp_path: Path) -> None:
    cp = CheckpointStore(tmp_path / "run_001")
    assert cp.load_state() is None
    assert cp.load_events() == []

    snapshot = {"version": 2, "run_id": "run_001", "tasks": {"t1": {"status": "done"}}}
    state_path = cp.save_state(snapshot)
    assert state_path.exists()
    assert cp.load_state() == snapshot
    assert not any(p.name.endswith(".tmp") for p in cp.run_dir.iterdir())

    ev1 = Event(seq=1, timestamp="2026-01-01T00:00:00Z", event_type="task_enqueued", task_id="t1")
    ev2 = Event(seq=2, timestamp="2026-01-01T00:00:01Z", event_type="run_finished", task_id="t1")
    cp.append_event(ev1)
    cp.append_event(ev2)
    assert cp.load_events() == [ev1, ev2]
