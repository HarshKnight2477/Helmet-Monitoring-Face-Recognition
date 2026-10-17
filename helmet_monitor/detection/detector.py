"""Helmet detection with a YOLOv8 model trained on helmet / head classes."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from helmet_monitor.domain import Box, Detection


class HelmetDetector(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        ...


class YoloHelmetDetector:
    def __init__(self, weights: Path, confidence: float = 0.4, image_size: int = 640):
        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"Helmet model not found at {weights}. Train one with "
                "training/helmet_yolov8_kaggle.ipynb and copy it there."
            )
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self.confidence = confidence
        self.image_size = image_size

    @property
    def labels(self) -> list[str]:
        return list(self._model.names.values())

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        result = self._model.predict(image_bgr, conf=self.confidence, imgsz=self.image_size,
                                     verbose=False)[0]
        boxes = result.boxes
        coordinates = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        return [
            Detection(Box(*map(float, xyxy)), result.names[cls], float(score))
            for xyxy, cls, score in zip(coordinates, classes, scores)
        ]
