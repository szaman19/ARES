"""Immutable, append-only NoteStore for ARES (Invariant I3).

100% modality- and backend-agnostic: provides thread-safe note appending, sequential
note ID generation, content-hash deduplication lookups, artifact provenance lookups,
bounded recall search, and JSON serialization for long-horizon checkpointing.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from typing import Any, Iterator

from ares.contracts import ArtifactRef, CertificationStatus, Note

_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9_-]+")
_NOTE_ID_NUM_RE = re.compile(r"^n_(\d+)$")


class NoteStore:
    """Thread-safe, append-only store of immutable Note records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._notes: list[Note] = []
        self._by_id: dict[str, Note] = {}
        self._by_content_hash: dict[str, Note] = {}
        self._artifacts: dict[str, ArtifactRef] = {}
        self._counter: int = 0

    def next_note_id(self) -> str:
        """Generate the next sequential note ID (e.g., n_0001)."""
        with self._lock:
            self._counter += 1
            while f"n_{self._counter:04d}" in self._by_id:
                self._counter += 1
            return f"n_{self._counter:04d}"

    def append(self, note: Note) -> Note:
        """Append an immutable Note. Raises ValueError if note_id already exists."""
        with self._lock:
            if note.note_id in self._by_id:
                raise ValueError(f"Note ID already exists (append-only store): {note.note_id}")
            self._notes.append(note)
            self._by_id[note.note_id] = note
            if note.content_hash and note.content_hash not in self._by_content_hash:
                self._by_content_hash[note.content_hash] = note
            for art in note.artifacts:
                self._artifacts[art.artifact_id] = art
            m = _NOTE_ID_NUM_RE.match(note.note_id)
            if m:
                self._counter = max(self._counter, int(m.group(1)))
            return note

    def get(self, note_id: str) -> Note:
        """Retrieve a Note by ID; raises KeyError if missing."""
        with self._lock:
            return self._by_id[note_id]

    def exists(self, note_id: str) -> bool:
        """Return True if a Note with note_id exists in the store."""
        with self._lock:
            return note_id in self._by_id

    def has_content_hash(self, content_hash: str) -> bool:
        """Return True if a Note with the given content_hash is already stored."""
        if not content_hash:
            return False
        with self._lock:
            return content_hash in self._by_content_hash

    def get_by_content_hash(self, content_hash: str) -> Note | None:
        """Return the first Note matching content_hash, or None."""
        if not content_hash:
            return None
        with self._lock:
            return self._by_content_hash.get(content_hash)

    def has_artifact(self, artifact_id: str) -> bool:
        """Return True if an ArtifactRef with artifact_id is attached to any stored Note."""
        with self._lock:
            return artifact_id in self._artifacts

    def get_artifact(self, artifact_id: str) -> ArtifactRef | None:
        """Return the ArtifactRef for artifact_id, or None if not found."""
        with self._lock:
            return self._artifacts.get(artifact_id)

    def list_notes(
        self,
        *,
        certification: CertificationStatus | None = None,
        hypothesis_family: str | None = None,
        verifier_name: str | None = None,
    ) -> list[Note]:
        """Return a snapshot list of stored notes, optionally filtered."""
        with self._lock:
            result = list(self._notes)
        if certification is not None:
            result = [n for n in result if n.certification == certification]
        if hypothesis_family is not None:
            result = [n for n in result if n.hypothesis_family == hypothesis_family]
        if verifier_name is not None:
            result = [n for n in result if n.verifier_name == verifier_name]
        return result

    def search(
        self,
        query: str = "",
        *,
        certification: CertificationStatus | None = None,
        verifier_name: str | None = None,
        hypothesis_family: str | None = None,
        limit: int = 10,
    ) -> list[Note]:
        """Search stored notes by metadata filters and keyword overlap for long-horizon recall."""
        if limit <= 0:
            return []

        with self._lock:
            indexed = list(enumerate(self._notes))

        filtered = [
            (idx, n)
            for idx, n in indexed
            if (certification is None or n.certification == certification)
            and (verifier_name is None or n.verifier_name == verifier_name)
            and (hypothesis_family is None or n.hypothesis_family == hypothesis_family)
        ]

        cleaned = (query or "").strip().lower()
        if not cleaned:
            return [n for _, n in reversed(filtered[-limit:])]

        tokens = list(dict.fromkeys(_SEARCH_TOKEN_RE.findall(cleaned)))
        if not tokens:
            return [n for _, n in reversed(filtered[-limit:])]

        scored: list[tuple[int, int, Note]] = []
        for idx, note in filtered:
            try:
                evidence_blob = json.dumps(note.evidence, sort_keys=True, default=str)
            except TypeError:
                evidence_blob = repr(note.evidence)
            haystack = " ".join(
                [
                    note.summary or "",
                    note.verifier_name or "",
                    note.hypothesis_family or "",
                    note.query or "",
                    evidence_blob,
                ]
            ).lower()
            score = sum(1 for tok in tokens if tok in haystack)
            if score > 0:
                scored.append((score, idx, note))

        scored.sort(key=lambda item: (-item[0], -item[1]))
        return [item[2] for item in scored[:limit]]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialize all stored notes into JSON-safe dicts in append order."""
        with self._lock:
            return [n.model_dump(mode="json") for n in self._notes]

    @classmethod
    def from_dicts(cls, items: Sequence[dict[str, Any]]) -> NoteStore:
        """Restore a NoteStore from serialized Note dictionaries."""
        store = cls()
        max_suffix = 0
        for item in items:
            note = Note.model_validate(item)
            store.append(note)
            m = _NOTE_ID_NUM_RE.match(note.note_id)
            if m:
                max_suffix = max(max_suffix, int(m.group(1)))
        with store._lock:
            store._counter = max(store._counter, max_suffix, len(store._notes))
        return store

    def snapshot(self) -> list[Note]:
        """Return an ordered shallow copy of all notes in append order."""
        with self._lock:
            return list(self._notes)

    def __len__(self) -> int:
        with self._lock:
            return len(self._notes)

    def __iter__(self) -> Iterator[Note]:
        return iter(self.snapshot())


__all__ = ["NoteStore"]
