"""Turns noisy per-frame results into confirmed, rate-limited violation events.

A violation is logged only after it is seen in ``confirm_frames`` consecutive
frames (filters single-frame false positives), and then at most once per
``cooldown_seconds`` per worker, so one person standing bare-headed for a
minute produces one record instead of hundreds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from helmet_monitor.domain import FrameResult, PersonResult

UNKNOWN_KEY = "unknown"


@dataclass(frozen=True)
class ViolationEvent:
    camera_id: str
    worker_id: int | None
    name: str
    similarity: float | None
    confidence: float
    person: PersonResult


def _key(person: PersonResult) -> str:
    # Unidentified violators share one key: they can't be told apart reliably.
    if person.identity and person.identity.is_known:
        return f"worker:{person.identity.worker_id}"
    return UNKNOWN_KEY


class ViolationTracker:
    def __init__(self, camera_id: str, confirm_frames: int = 3, cooldown_seconds: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.camera_id = camera_id
        self.confirm_frames = max(1, confirm_frames)
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._streaks: dict[str, int] = {}
        self._last_logged: dict[str, float] = {}

    def update(self, result: FrameResult) -> list[ViolationEvent]:
        now = self._clock()
        seen: dict[str, PersonResult] = {}
        for person in result.violations:
            key = _key(person)
            # Keep the best-identified sighting when one key appears twice.
            current = seen.get(key)
            if current is None or person.head.confidence > current.head.confidence:
                seen[key] = person

        self._streaks = {key: self._streaks.get(key, 0) + 1 for key in seen}

        events = []
        for key, person in seen.items():
            if self._streaks[key] < self.confirm_frames:
                continue
            last = self._last_logged.get(key)
            if last is not None and now - last < self.cooldown_seconds:
                continue
            self._last_logged[key] = now
            identity = person.identity
            events.append(ViolationEvent(
                camera_id=self.camera_id,
                worker_id=identity.worker_id if identity else None,
                name=identity.name if identity else "Unknown",
                similarity=identity.similarity if identity and identity.is_known else None,
                confidence=person.head.confidence,
                person=person,
            ))
        return events
