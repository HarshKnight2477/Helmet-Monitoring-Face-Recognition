"""Per-frame pipeline: detect helmets -> verify per head -> identify faces."""

from __future__ import annotations

import threading
import time

import numpy as np

from helmet_monitor.compliance.spatial import assign_faces, resolve_heads
from helmet_monitor.detection.detector import HelmetDetector
from helmet_monitor.domain import FrameResult, PersonResult
from helmet_monitor.recognition.embedder import FaceEmbedder
from helmet_monitor.recognition.gallery import FaceGallery


class FrameAnalyzer:
    """Shared by all camera sessions; a lock serialises model inference."""

    def __init__(
        self,
        detector: HelmetDetector,
        gallery: FaceGallery,
        embedder: FaceEmbedder | None,
        helmet_labels: frozenset[str],
        no_helmet_labels: frozenset[str],
    ):
        self.detector = detector
        self.gallery = gallery
        self.embedder = embedder
        self.helmet_labels = helmet_labels
        self.no_helmet_labels = no_helmet_labels
        self._lock = threading.Lock()

    def analyze(self, image_bgr: np.ndarray) -> FrameResult:
        start = time.perf_counter()
        with self._lock:
            heads = resolve_heads(self.detector.detect(image_bgr),
                                  self.helmet_labels, self.no_helmet_labels)
            # Face models are the slow part: skip them when nobody is in view.
            faces = self.embedder.detect(image_bgr) if heads and self.embedder else []

        assignment = assign_faces([h.box for h in heads], faces)
        people = []
        for index, head in enumerate(heads):
            face = faces[assignment[index]] if index in assignment else None
            people.append(PersonResult(
                head=head,
                face_box=face.box if face else None,
                identity=self.gallery.match(face.embedding) if face else None,
            ))
        return FrameResult(people=people, inference_ms=(time.perf_counter() - start) * 1000)
