"""Image encoding helpers for frames sent over the WebSocket and uploads."""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG bytes into a BGR image, or None if the bytes aren't an image."""
    if not data:
        return None
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def decode_base64_image(text: str) -> np.ndarray | None:
    """Accepts plain base64 or a ``data:image/jpeg;base64,...`` URL."""
    if "," in text:
        text = text.split(",", 1)[1]
    try:
        return decode_image(base64.b64decode(text, validate=True))
    except (binascii.Error, ValueError):
        return None


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode image as JPEG")
    return buffer.tobytes()
