"""Turns noisy per-frame results into confirmed, rate-limited violation events.

A violation is recorded only after a person has been without a hard hat for
``confirm_seconds`` (default 10 s) without interruption. Short detection
dropouts (up to ``grace_seconds``) don't reset the timer, but putting a hard
hat on, or leaving the frame for longer, does. After recording, the same
worker isn't recorded again for ``cooldown_seconds``, so one person standing
bare-headed for a minute produces one record instead of hundreds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class ViolationTimer:
    """Countdown shown above a bare-headed person in the live view."""

    elapsed: float
    remaining: float
    recorded: bool

    def to_dict(self) -> dict:
        return {"elapsed": round(self.elapsed, 1), "remaining": round(self.remaining, 1),
                "recorded": self.recorded}


@dataclass(frozen=True)
class TrackerUpdate:
    events: list[ViolationEvent] = field(default_factory=list)
    # One entry per person in the frame, in order; None for people wearing a hard hat.
    timers: list[ViolationTimer | None] = field(default_factory=list)


@dataclass
class _Streak:
    started: float
    last_seen: float


def _key(person: PersonResult) -> str:
    # Unidentified violators share one key: they can't be told apart reliably.
    if person.identity and person.identity.is_known:
        return f"worker:{person.identity.worker_id}"
    return UNKNOWN_KEY


class ViolationTracker:
    def __init__(self, camera_id: str, confirm_seconds: float = 10.0, cooldown_seconds: float = 60.0,
                 grace_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self.camera_id = camera_id
        self.confirm_seconds = max(0.0, confirm_seconds)
        self.cooldown_seconds = cooldown_seconds
        self.grace_seconds = grace_seconds
        self._clock = clock
        self._streaks: dict[str, _Streak] = {}
        self._last_recorded: dict[str, float] = {}

    def update(self, result: FrameResult) -> TrackerUpdate:
        now = self._clock()
        seen: dict[str, PersonResult] = {}
        for person in result.violations:
            key = _key(person)
            current = seen.get(key)
            if current is None or person.head.confidence > current.head.confidence:
                seen[key] = person

        # Forget streaks that have been out of sight longer than the grace period.
        for key in [k for k, s in self._streaks.items()
                    if k not in seen and now - s.last_seen > self.grace_seconds]:
            del self._streaks[key]

        events = []
        for key, person in seen.items():
            streak = self._streaks.setdefault(key, _Streak(started=now, last_seen=now))
            streak.last_seen = now
            if now - streak.started >= self.confirm_seconds and not self._in_cooldown(key, now):
                self._last_recorded[key] = now
                identity = person.identity
                events.append(ViolationEvent(
                    camera_id=self.camera_id,
                    worker_id=identity.worker_id if identity else None,
                    name=identity.name if identity else "Unknown",
                    similarity=identity.similarity if identity and identity.is_known else None,
                    confidence=person.head.confidence,
                    person=person,
                ))

        timers = []
        for person in result.people:
            if not person.is_violation:
                timers.append(None)
                continue
            key = _key(person)
            elapsed = now - self._streaks[key].started
            timers.append(ViolationTimer(elapsed=elapsed,
                                         remaining=max(0.0, self.confirm_seconds - elapsed),
                                         recorded=self._in_cooldown(key, now)))
        return TrackerUpdate(events=events, timers=timers)

    def _in_cooldown(self, key: str, now: float) -> bool:
        last = self._last_recorded.get(key)
        return last is not None and now - last < self.cooldown_seconds
