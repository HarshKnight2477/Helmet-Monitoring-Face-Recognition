"""Core data types shared by detection, recognition and compliance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Axis-aligned bounding box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = ix * iy
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def expand(self, left: float, top: float, right: float, bottom: float) -> Box:
        """Grow the box by fractions of its own width/height on each side."""
        return Box(self.x1 - left * self.width, self.y1 - top * self.height,
                   self.x2 + right * self.width, self.y2 + bottom * self.height)

    def to_list(self) -> list[int]:
        return [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]


@dataclass(frozen=True)
class Detection:
    """One object found by the helmet detector."""

    box: Box
    label: str
    confidence: float


@dataclass(frozen=True, eq=False)
class Face:
    """A detected face with its L2-normalised embedding."""

    box: Box
    embedding: np.ndarray
    score: float


@dataclass(frozen=True)
class Identity:
    """Result of matching a face against registered workers."""

    worker_id: int | None
    name: str
    similarity: float

    @property
    def is_known(self) -> bool:
        return self.worker_id is not None


@dataclass(frozen=True)
class HeadState:
    """A head region and whether a helmet was detected on it."""

    box: Box
    wearing_helmet: bool
    confidence: float


@dataclass(frozen=True)
class PersonResult:
    head: HeadState
    face_box: Box | None = None
    identity: Identity | None = None

    @property
    def is_violation(self) -> bool:
        return not self.head.wearing_helmet

    def to_dict(self) -> dict:
        return {
            "head_box": self.head.box.to_list(),
            "wearing_helmet": self.head.wearing_helmet,
            "confidence": round(self.head.confidence, 3),
            "face_box": self.face_box.to_list() if self.face_box else None,
            "worker_id": self.identity.worker_id if self.identity else None,
            "name": self.identity.name if self.identity else None,
            "similarity": round(self.identity.similarity, 3) if self.identity else None,
        }


@dataclass(frozen=True)
class FrameResult:
    people: list[PersonResult] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def violations(self) -> list[PersonResult]:
        return [p for p in self.people if p.is_violation]

    def to_dict(self) -> dict:
        return {
            "people": [p.to_dict() for p in self.people],
            "total_people": len(self.people),
            "with_helmet": len(self.people) - len(self.violations),
            "without_helmet": len(self.violations),
            "inference_ms": round(self.inference_ms, 1),
        }
