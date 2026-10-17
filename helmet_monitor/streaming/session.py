"""One CameraSession per WebSocket connection.

Receiving and processing run as separate tasks joined by a small queue.
When inference is slower than the camera, the oldest frame is dropped, so
results always describe what the camera sees *now* instead of falling
further and further behind.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import numpy as np

from helmet_monitor.compliance.analyzer import FrameAnalyzer
from helmet_monitor.compliance.tracker import ViolationEvent, ViolationTracker

logger = logging.getLogger("helmet_monitor")

SendJson = Callable[[dict], Awaitable[None]]
ViolationHandler = Callable[[ViolationEvent, np.ndarray], dict]


class CameraSession:
    def __init__(self, camera_id: str, analyzer: FrameAnalyzer, tracker: ViolationTracker,
                 on_violation: ViolationHandler, queue_size: int = 2):
        self.camera_id = camera_id
        self.analyzer = analyzer
        self.tracker = tracker
        self.on_violation = on_violation
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=max(1, queue_size))
        self.connected_at = datetime.now(timezone.utc)
        self.frames_received = 0
        self.frames_processed = 0
        self.frames_dropped = 0

    def submit(self, frame: np.ndarray) -> None:
        """Queue a frame, discarding the oldest one if the queue is full."""
        self.frames_received += 1
        if self.queue.full():
            self.queue.get_nowait()
            self.frames_dropped += 1
        self.queue.put_nowait(frame)

    async def run(self, send: SendJson) -> None:
        """Process queued frames forever, sending one result message per frame."""
        while True:
            frame = await self.queue.get()
            try:
                result = await asyncio.to_thread(self.analyzer.analyze, frame)
                events = self.tracker.update(result)
                logged = [await asyncio.to_thread(self.on_violation, event, frame)
                          for event in events]
            except Exception:  # one bad frame must not end the stream
                logger.exception("Frame processing failed on camera %s", self.camera_id)
                await send({"type": "error", "message": "Frame processing failed."})
                continue
            self.frames_processed += 1
            await send({
                "type": "result",
                "frame": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
                **result.to_dict(),
                "new_violations": logged,
            })

    def status(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "connected_at": self.connected_at.isoformat(timespec="seconds"),
            "frames_received": self.frames_received,
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
        }
