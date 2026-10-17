"""Builds and holds the app's long-lived objects (models, database, gallery)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from helmet_monitor.compliance.analyzer import FrameAnalyzer
from helmet_monitor.compliance.tracker import ViolationEvent, ViolationTracker
from helmet_monitor.config import Settings
from helmet_monitor.detection.detector import HelmetDetector, YoloHelmetDetector
from helmet_monitor.recognition.embedder import FaceEmbedder, InsightFaceEmbedder
from helmet_monitor.recognition.gallery import FaceGallery
from helmet_monitor.storage.database import Database
from helmet_monitor.streaming.codec import encode_jpeg
from helmet_monitor.streaming.session import CameraSession

logger = logging.getLogger("helmet_monitor")

RED = (40, 40, 220)
GREEN = (60, 180, 75)


@dataclass
class Services:
    settings: Settings
    database: Database
    gallery: FaceGallery
    detector: HelmetDetector | None = None
    embedder: FaceEmbedder | None = None
    errors: dict[str, str] = field(default_factory=dict)
    sessions: dict[str, CameraSession] = field(default_factory=dict)
    _analyzer: FrameAnalyzer | None = field(default=None, init=False, repr=False)

    @property
    def analyzer(self) -> FrameAnalyzer | None:
        if self.detector is None:
            return None
        if self._analyzer is None:
            self._analyzer = FrameAnalyzer(self.detector, self.gallery, self.embedder,
                                           self.settings.helmet_labels,
                                           self.settings.no_helmet_labels)
        return self._analyzer

    def new_session(self, camera_id: str) -> CameraSession:
        tracker = ViolationTracker(camera_id, self.settings.violation_confirm_seconds,
                                   self.settings.violation_cooldown_seconds,
                                   self.settings.violation_grace_seconds)
        return CameraSession(camera_id, self.analyzer, tracker, self.record_violation,
                             self.settings.frame_queue_size)

    def record_violation(self, event: ViolationEvent, frame: np.ndarray) -> dict:
        """Save an annotated snapshot and a database row; return it for the UI."""
        snapshot = self._save_snapshot(event, frame)
        record = self.database.add_violation(event.camera_id, event.worker_id, event.name,
                                             event.similarity, event.confidence, snapshot)
        logger.info("Violation: %s on camera %s", event.name, event.camera_id)
        return {"id": record.id, "worker_id": record.worker_id, "name": record.worker_name,
                "similarity": record.similarity, "camera_id": record.camera_id,
                "snapshot": f"/snapshots/{snapshot}" if snapshot else None,
                "created_at": record.created_at}

    def _save_snapshot(self, event: ViolationEvent, frame: np.ndarray) -> str | None:
        try:
            image = frame.copy()
            x1, y1, x2, y2 = event.person.head.box.to_list()
            cv2.rectangle(image, (x1, y1), (x2, y2), RED, 3)
            cv2.putText(image, f"NO HARD HAT: {event.name}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{event.camera_id}.jpg"
            self.settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.snapshot_dir / name).write_bytes(encode_jpeg(image))
            return name
        except (OSError, ValueError):
            logger.exception("Could not save violation snapshot")
            return None


def build_services(settings: Settings) -> Services:
    """Load everything; a missing model is reported in /api/health rather than crashing."""
    database = Database(settings.database_path)
    gallery = FaceGallery(settings.face_match_threshold)
    gallery.load(database.worker_embeddings())
    services = Services(settings=settings, database=database, gallery=gallery)

    try:
        services.detector = YoloHelmetDetector(settings.model_path, settings.detection_confidence,
                                               settings.detection_image_size)
    except Exception as error:  # missing weights, missing ultralytics, corrupt file...
        services.errors["detector"] = str(error)
        logger.warning("Helmet detector not loaded: %s", error)

    try:
        services.embedder = InsightFaceEmbedder(settings.face_model_name,
                                                settings.face_detection_size,
                                                settings.face_device)
    except Exception as error:
        services.errors["face_model"] = str(error)
        logger.warning("Face model not loaded: %s", error)

    return services
