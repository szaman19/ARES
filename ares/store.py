"""Immutable, append-only NoteStore for ARES (Invariant I3).

Provides thread-safe note appending, sequential note ID generation, content-hash
deduplication lookups, and artifact index lookups for multimodal provenance checks.
"""

from __future__ import annotations

import threading
from typing import Iterator

from contracts import ArtifactRef, CertificationStatus, Note


class NoteStore:
    """Thread-safe, append-only store of immutable Note records."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._notes: list[Note] = []
        self._by_id: dict[str, Note] = {}
        self._by_query_hash: dict[str, Note] = {}
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
            if note.query_hash and note.query_hash not in self._by_query_hash:
                self._by_query_hash[note.query_hash] = note
            for art in note.artifacts:
                self._artifacts[art.artifact_id] = art
            return note

    def get(self, note_id: str) -> Note:
        """Retrieve a Note by ID; raises KeyError if missing."""
        with self._lock:
            return self._by_id[note_id]

    def exists(self, note_id: str) -> bool:
        """Return True if a Note with note_id exists in the store."""
        with self._lock:
            return note_id in self._by_id

    def has_query_hash(self, query_hash: str) -> bool:
        """Return True if a Note with the given normalized query_hash is already stored."""
        if not query_hash:
            return False
        with self._lock:
            return query_hash in self._by_query_hash

    def get_by_query_hash(self, query_hash: str) -> Note | None:
        """Return the first Note matching query_hash, or None."""
        if not query_hash:
            return None
        with self._lock:
            return self._by_query_hash.get(query_hash)

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
