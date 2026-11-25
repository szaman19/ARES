"""Content-addressed artifact storage and atomic checkpoint persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from ares.types import ArtifactRef, Event

_HANDLE_KEY = "__ares_ref__"
_PREVIEW_CHARS = 120


def _make_preview(data: bytes) -> str:
    text = data[: _PREVIEW_CHARS * 4].decode("utf-8", errors="replace")
    collapsed = " ".join(text.split())
    if len(collapsed) > _PREVIEW_CHARS:
        return collapsed[: _PREVIEW_CHARS - 3] + "..."
    return collapsed


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


class ArtifactStore:
    """Thread-safe content-addressed store for large or binary task outputs.

    Supports both in-memory operation (``root_dir=None``) and disk-backed
    persistence (``<run_dir>/artifacts/`` with ``index.json`` for reload on resume).
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._root_dir = Path(root_dir) if root_dir is not None else None
        self._index_path = self._root_dir / "index.json" if self._root_dir is not None else None
        self._refs: dict[str, ArtifactRef] = {}
        self._blobs: dict[str, bytes] = {}
        if self._root_dir is not None:
            self._root_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    def _load_index(self) -> None:
        if self._index_path is None or not self._index_path.exists():
            return
        raw = json.loads(self._index_path.read_text(encoding="utf-8"))
        for ref_id, ref_data in raw.items():
            self._refs[ref_id] = ArtifactRef.model_validate(ref_data)

    def _save_index(self) -> None:
        if self._index_path is None:
            return
        payload = {ref_id: ref.model_dump() for ref_id, ref in self._refs.items()}
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        _atomic_write_bytes(self._index_path, encoded)

    @staticmethod
    def is_artifact_handle(value: Any) -> bool:
        """Return True when ``value`` is a compact ArtifactStore spill handle."""
        return (
            isinstance(value, dict)
            and _HANDLE_KEY in value
            and isinstance(value[_HANDLE_KEY], str)
        )

    @staticmethod
    def extract_ref_id(value: Any) -> str | None:
        """Extract ``ref_id`` from a spill handle, or return None."""
        if ArtifactStore.is_artifact_handle(value):
            return value[_HANDLE_KEY]
        return None

    @staticmethod
    def make_handle(ref: ArtifactRef) -> dict[str, Any]:
        """Build the compact JSON handle stored inline in place of spilled data."""
        return {
            _HANDLE_KEY: ref.ref_id,
            "byte_size": ref.byte_size,
            "mime_type": ref.mime_type,
            "preview": ref.preview,
        }

    def put_bytes(
        self,
        data: bytes,
        *,
        mime_type: str = "application/octet-stream",
        name: str | None = None,
        preview: str | None = None,
    ) -> ArtifactRef:
        """Store raw bytes content-addressed by SHA-256 and return an ArtifactRef."""
        _ = name  # Reserved for caller clarity; addressing is strictly content-based.
        digest = hashlib.sha256(data).hexdigest()
        ref_id = f"art_{digest[:16]}"

        with self._lock:
            existing = self._refs.get(ref_id)
            if existing is not None:
                if self._root_dir is None or (
                    existing.path is not None and Path(existing.path).exists()
                ):
                    return existing

            resolved_preview = preview if preview is not None else _make_preview(data)
            blob_path_str: str | None = None
            if self._root_dir is not None:
                blob_path = self._root_dir / ref_id
                _atomic_write_bytes(blob_path, data)
                blob_path_str = str(blob_path)
            else:
                self._blobs[ref_id] = bytes(data)

            ref = ArtifactRef(
                ref_id=ref_id,
                sha256=digest,
                byte_size=len(data),
                mime_type=mime_type,
                preview=resolved_preview,
                path=blob_path_str,
            )
            self._refs[ref_id] = ref
            self._save_index()
            return ref

    def put_text(
        self,
        text: str,
        *,
        mime_type: str = "text/plain",
        name: str | None = None,
    ) -> ArtifactRef:
        """Encode text as UTF-8 and store it in the ArtifactStore."""
        return self.put_bytes(text.encode("utf-8"), mime_type=mime_type, name=name)

    def put_json(self, obj: Any, *, name: str | None = None) -> ArtifactRef:
        """Serialize ``obj`` as deterministic UTF-8 JSON and store it."""
        serialized = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return self.put_bytes(
            serialized.encode("utf-8"),
            mime_type="application/json",
            name=name,
        )

    def get_bytes(self, ref_id: str) -> bytes:
        """Retrieve raw bytes for ``ref_id`` or raise KeyError if missing."""
        with self._lock:
            ref = self._refs.get(ref_id)
            if ref is None:
                raise KeyError(f"Unknown artifact ref: {ref_id}")
            if self._root_dir is None:
                return self._blobs[ref_id]
            blob_path = Path(ref.path) if ref.path is not None else (self._root_dir / ref_id)
        if not blob_path.exists():
            raise FileNotFoundError(f"Missing artifact blob for {ref_id}: {blob_path}")
        return blob_path.read_bytes()

    def get_text(self, ref_id: str) -> str:
        """Retrieve UTF-8 decoded text for ``ref_id``."""
        return self.get_bytes(ref_id).decode("utf-8")

    def get_json(self, ref_id: str) -> Any:
        """Retrieve and parse JSON stored under ``ref_id``."""
        return json.loads(self.get_text(ref_id))

    def read_slice(self, ref_id: str, offset: int = 0, limit: int = 1024) -> str:
        """Read a character slice ``[offset : offset + limit]`` from a text/JSON artifact."""
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        text = self.get_text(ref_id)
        return text[offset : offset + limit]

    def maybe_spill(
        self,
        value: Any,
        max_inline_bytes: int | None,
    ) -> tuple[Any, ArtifactRef | None]:
        """Spill ``value`` to the store if its serialized UTF-8 size exceeds ``max_inline_bytes``."""
        if value is None or max_inline_bytes is None or self.is_artifact_handle(value):
            return value, None

        if isinstance(value, str):
            raw_bytes = value.encode("utf-8")
            if len(raw_bytes) <= max_inline_bytes:
                return value, None
            ref = self.put_text(value)
            return self.make_handle(ref), ref

        if isinstance(value, bytes):
            if len(value) <= max_inline_bytes:
                return value, None
            ref = self.put_bytes(value)
            return self.make_handle(ref), ref

        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if len(serialized.encode("utf-8")) <= max_inline_bytes:
            return value, None
        ref = self.put_json(value)
        return self.make_handle(ref), ref

    def list_refs(self) -> list[ArtifactRef]:
        """Return all registered ArtifactRef objects sorted by ref_id."""
        with self._lock:
            return [self._refs[k] for k in sorted(self._refs)]

    def get_ref(self, ref_id: str) -> ArtifactRef | None:
        """Return the ArtifactRef for ``ref_id``, or None if absent."""
        with self._lock:
            return self._refs.get(ref_id)


class CheckpointStore:
    """Atomic state snapshot and append-only event log persistence in ``<run_dir>``."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self._lock = threading.Lock()

    def save_state(self, snapshot: dict[str, Any]) -> Path:
        """Atomically persist ``snapshot`` to ``<run_dir>/state.json``."""
        encoded = json.dumps(snapshot, indent=2, ensure_ascii=False).encode("utf-8")
        with self._lock:
            _atomic_write_bytes(self.state_path, encoded)
        return self.state_path

    def load_state(self) -> dict[str, Any] | None:
        """Load ``<run_dir>/state.json`` if it exists, otherwise return None."""
        if not self.state_path.exists():
            return None
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def append_event(self, event: Event) -> None:
        """Append a single Event as a JSON line to ``<run_dir>/events.jsonl``."""
        line = event.model_dump_json() + "\n"
        with self._lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def load_events(self) -> list[Event]:
        """Load all recorded Events from ``<run_dir>/events.jsonl``."""
        if not self.events_path.exists():
            return []
        events: list[Event] = []
        for raw_line in self.events_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped:
                events.append(Event.model_validate_json(stripped))
        return events
