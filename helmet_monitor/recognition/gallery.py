"""In-memory gallery of registered worker faces, matched by cosine similarity."""

from __future__ import annotations

import threading
from collections.abc import Iterable

import numpy as np

from helmet_monitor.domain import Identity

UNKNOWN = "Unknown"


def normalize(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length so a dot product equals cosine similarity."""
    vector = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("Cannot normalise a zero vector")
    return vector / norm


class FaceGallery:
    """Thread-safe store of (worker id, name, normalised embedding)."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._lock = threading.Lock()
        self._ids: list[int] = []
        self._names: list[str] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        with self._lock:
            return len(self._ids)

    def load(self, entries: Iterable[tuple[int, str, np.ndarray]]) -> None:
        ids, names, vectors = [], [], []
        for worker_id, name, embedding in entries:
            ids.append(worker_id)
            names.append(name)
            vectors.append(normalize(embedding))
        with self._lock:
            self._ids, self._names = ids, names
            self._matrix = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)

    def add(self, worker_id: int, name: str, embedding: np.ndarray) -> None:
        vector = normalize(embedding)[np.newaxis, :]
        with self._lock:
            self._ids.append(worker_id)
            self._names.append(name)
            self._matrix = vector if self._matrix.size == 0 else np.vstack([self._matrix, vector])

    def remove(self, worker_id: int) -> bool:
        with self._lock:
            if worker_id not in self._ids:
                return False
            index = self._ids.index(worker_id)
            del self._ids[index]
            del self._names[index]
            self._matrix = np.delete(self._matrix, index, axis=0)
            if not self._ids:
                self._matrix = np.empty((0, 0), dtype=np.float32)
            return True

    def match(self, embedding: np.ndarray) -> Identity:
        """Best-matching worker, or ``Unknown`` if nobody clears the threshold."""
        query = normalize(embedding)
        with self._lock:
            if not self._ids:
                return Identity(None, UNKNOWN, 0.0)
            similarities = self._matrix @ query
            best = int(np.argmax(similarities))
            similarity = float(similarities[best])
            if similarity < self.threshold:
                return Identity(None, UNKNOWN, similarity)
            return Identity(self._ids[best], self._names[best], similarity)
