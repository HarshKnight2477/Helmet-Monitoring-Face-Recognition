"""Face detection + embedding with InsightFace (RetinaFace detector, ArcFace embeddings)."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from helmet_monitor.domain import Box, Face
from helmet_monitor.recognition.gallery import normalize

logger = logging.getLogger("helmet_monitor")


class FaceEmbedder(Protocol):
    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        """Find faces in a BGR image and return them with normalised embeddings."""
        ...


def select_providers(device: str = "auto") -> list[str]:
    """ONNX Runtime providers for ``device`` = auto | cuda | cpu.

    ``auto`` uses the GPU when onnxruntime-gpu and CUDA are available, else the CPU.
    """
    import onnxruntime as ort

    device = device.lower()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        if hasattr(ort, "preload_dlls"):
            # Load CUDA/cuDNN from the nvidia-* pip packages (or PyTorch) before
            # creating sessions, so no system-wide CUDA install is needed.
            ort.preload_dlls()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        raise RuntimeError("FACE_DEVICE=cuda but onnxruntime-gpu with CUDA is not available; "
                           "install requirements-gpu.txt or set FACE_DEVICE=cpu")
    return ["CPUExecutionProvider"]


class InsightFaceEmbedder:
    """Wraps ``insightface.app.FaceAnalysis``.

    InsightFace expects BGR images (OpenCV's default), so frames are passed
    through unchanged both when registering workers and when monitoring.
    The model pack (~300 MB for buffalo_l) downloads on first use.
    """

    def __init__(self, model_name: str = "buffalo_l", detection_size: int = 640,
                 device: str = "auto", providers: list[str] | None = None):
        from insightface.app import FaceAnalysis

        providers = providers or select_providers(device)
        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        on_gpu = providers[0] == "CUDAExecutionProvider"
        self._app.prepare(ctx_id=0 if on_gpu else -1, det_size=(detection_size, detection_size))
        self.device = self._active_device()
        logger.info("Face model running on %s", self.device)

    def _active_device(self) -> str:
        sessions = [model.session for model in self._app.models.values()]
        return "cuda" if all(s.get_providers()[0] == "CUDAExecutionProvider" for s in sessions) else "cpu"

    def detect(self, image_bgr: np.ndarray) -> list[Face]:
        return [
            Face(box=Box(*map(float, face.bbox)), embedding=normalize(face.embedding),
                 score=float(face.det_score))
            for face in self._app.get(image_bgr)
        ]


def largest_face(faces: list[Face]) -> Face | None:
    return max(faces, key=lambda f: f.box.area, default=None)
