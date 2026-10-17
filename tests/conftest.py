"""Fakes for the two ML models so the suite runs fast with no downloads."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from helmet_monitor.config import Settings
from helmet_monitor.domain import Box, Detection, Face
from helmet_monitor.recognition.gallery import FaceGallery, normalize
from helmet_monitor.services import Services
from helmet_monitor.storage.database import Database

EMBEDDING_DIM = 512


def embedding(seed: int) -> np.ndarray:
    return normalize(np.random.default_rng(seed).standard_normal(EMBEDDING_DIM))


class FakeDetector:
    def __init__(self, detections: list[Detection] | None = None):
        self.detections = detections or []
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return list(self.detections)


class FakeEmbedder:
    def __init__(self, faces: list[Face] | None = None):
        self.faces = faces or []
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return list(self.faces)


def jpeg_bytes(width: int = 320, height: int = 240) -> bytes:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def bare_head(x1=100, y1=50, x2=160, y2=120, confidence=0.9) -> Detection:
    return Detection(Box(x1, y1, x2, y2), "head", confidence)


def helmet(x1=100, y1=50, x2=160, y2=120, confidence=0.9) -> Detection:
    return Detection(Box(x1, y1, x2, y2), "helmet", confidence)


def face_at(x1=110, y1=70, x2=150, y2=115, seed=1) -> Face:
    return Face(Box(x1, y1, x2, y2), embedding(seed), 0.99)


def add_worker(services: Services, name: str, seed: int) -> int:
    """Register a worker in both the database and the gallery, like the API does."""
    worker = services.database.add_worker(name, None, embedding(seed))
    services.gallery.add(worker.id, name, embedding(seed))
    return worker.id


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        model_path=tmp_path / "missing.pt",
        database_path=tmp_path / "test.db",
        snapshot_dir=tmp_path / "snapshots",
        violation_confirm_frames=3,
        violation_cooldown_seconds=60,
        face_match_threshold=0.45,
    )


@pytest.fixture
def make_services(settings):
    created = []

    def make(detector=None, embedder=None) -> Services:
        services = Services(settings=settings, database=Database(settings.database_path),
                            gallery=FaceGallery(settings.face_match_threshold),
                            detector=detector, embedder=embedder)
        created.append(services)
        return services

    yield make
    for services in created:
        services.database.close()
