"""Face detection + embedding with InsightFace (RetinaFace detector, ArcFace embeddings)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from helmet_monitor.domain import Box, Face
from helmet_monitor.recognition.gallery import normalize


class FaceEmbedder(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        """Find faces in a BGR image and return them with normalised embeddings."""
        ...


class InsightFaceEmbedder:
    """Wraps ``insightface.app.FaceAnalysis``.

    InsightFace expects BGR images (OpenCV's default), so frames are passed
    through unchanged both when registering workers and when monitoring.
    The model pack (~300 MB for buffalo_l) downloads on first use.
    """

    def __init__(self, model_name: str = "buffalo_l", detection_size: int = 640,
                 providers: list[str] | None = None):
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers or ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(detection_size, detection_size))

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        return [
            Face(box=Box(*map(float, face.bbox)), embedding=normalize(face.embedding),
                 score=float(face.det_score))
            for face in self._app.get(image_bgr)
        ]


def largest_face(faces: list[Face]) -> Face | None:
    return max(faces, key=lambda f: f.box.area, default=None)
