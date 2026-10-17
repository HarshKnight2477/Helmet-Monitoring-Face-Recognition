"""Settings, read from environment variables (or a .env file) with sensible defaults."""

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env support: KEY=VALUE lines, existing variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _labels(value: str) -> frozenset[str]:
    return frozenset(label.strip().lower() for label in value.split(",") if label.strip())


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


@dataclass(frozen=True)
class Settings:
    # Helmet detector (YOLOv8)
    model_path: Path = field(default_factory=lambda: _path(_env("MODEL_PATH", "models/helmet_yolov8.pt")))
    detection_confidence: float = field(default_factory=lambda: float(_env("DETECTION_CONFIDENCE", "0.40")))
    detection_image_size: int = field(default_factory=lambda: int(_env("DETECTION_IMAGE_SIZE", "640")))
    helmet_labels: frozenset[str] = field(default_factory=lambda: _labels(_env("HELMET_LABELS", "helmet")))
    no_helmet_labels: frozenset[str] = field(default_factory=lambda: _labels(_env("NO_HELMET_LABELS", "head,no_helmet")))

    # Face recognition (InsightFace)
    face_model_name: str = field(default_factory=lambda: _env("FACE_MODEL_NAME", "buffalo_l"))
    face_detection_size: int = field(default_factory=lambda: int(_env("FACE_DETECTION_SIZE", "640")))
    face_match_threshold: float = field(default_factory=lambda: float(_env("FACE_MATCH_THRESHOLD", "0.45")))

    # Violation logging
    violation_confirm_frames: int = field(default_factory=lambda: int(_env("VIOLATION_CONFIRM_FRAMES", "3")))
    violation_cooldown_seconds: float = field(default_factory=lambda: float(_env("VIOLATION_COOLDOWN_SECONDS", "60")))

    # Streaming
    frame_queue_size: int = field(default_factory=lambda: int(_env("FRAME_QUEUE_SIZE", "2")))

    # Storage
    database_path: Path = field(default_factory=lambda: _path(_env("DATABASE_PATH", "data/helmet_monitor.db")))
    snapshot_dir: Path = field(default_factory=lambda: _path(_env("SNAPSHOT_DIR", "data/snapshots")))
    max_upload_mb: float = field(default_factory=lambda: float(_env("MAX_UPLOAD_MB", "5")))


def load_settings() -> Settings:
    _load_dotenv(PROJECT_DIR / ".env")
    return Settings()
