"""Bounding-box spatial verification.

Two geometric steps turn raw detections into per-person compliance:

1. ``resolve_heads`` - the detector reports ``helmet`` boxes (head wearing a
   helmet) and ``head`` boxes (bare head). When both fire on the same person
   (high IoU), only the more confident one is kept, so a person is never
   counted as compliant and violating at once.
2. ``assign_faces`` - each face is attached to at most one head, and only if
   the face centre lies inside that head's region. The region is extended
   downwards because a helmet box often covers the helmet more than the face.
"""

from __future__ import annotations

from collections.abc import Iterable

from helmet_monitor.domain import Box, Detection, Face, HeadState

# Region around a head box that a face centre must fall inside, as fractions
# of the head box size (left, top, right, bottom).
FACE_REGION_EXPANSION = (0.2, 0.1, 0.2, 0.8)
DUPLICATE_IOU = 0.5


def resolve_heads(
    detections: Iterable[Detection],
    helmet_labels: frozenset[str],
    no_helmet_labels: frozenset[str],
    duplicate_iou: float = DUPLICATE_IOU,
) -> list[HeadState]:
    candidates = []
    for det in detections:
        label = det.label.lower()
        if label in helmet_labels:
            candidates.append(HeadState(det.box, wearing_helmet=True, confidence=det.confidence))
        elif label in no_helmet_labels:
            candidates.append(HeadState(det.box, wearing_helmet=False, confidence=det.confidence))

    # Greedy suppression across classes: most confident box claims the person.
    kept: list[HeadState] = []
    for head in sorted(candidates, key=lambda h: h.confidence, reverse=True):
        if all(head.box.iou(other.box) < duplicate_iou for other in kept):
            kept.append(head)
    return kept


def face_region(head: Box) -> Box:
    return head.expand(*FACE_REGION_EXPANSION)


def assign_faces(heads: list[Box], faces: list[Face]) -> dict[int, int]:
    """Map head index -> face index, one-to-one, closest face centre first."""
    pairs = []
    for h, head in enumerate(heads):
        region = face_region(head)
        hx, hy = head.center
        scale = max(head.width, head.height, 1.0)
        for f, face in enumerate(faces):
            fx, fy = face.box.center
            if region.contains(fx, fy):
                distance = ((fx - hx) ** 2 + (fy - hy) ** 2) ** 0.5 / scale
                pairs.append((distance, h, f))

    assignment: dict[int, int] = {}
    used_faces: set[int] = set()
    for _, h, f in sorted(pairs):
        if h not in assignment and f not in used_faces:
            assignment[h] = f
            used_faces.add(f)
    return assignment
