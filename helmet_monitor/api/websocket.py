"""WebSocket endpoint for live camera monitoring.

Client -> server: JPEG frames as binary messages (or JSON {"image": "<base64>"}).
Server -> client: one JSON result per processed frame (boxes, identities,
compliance counts, newly logged violations).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from helmet_monitor.services import Services
from helmet_monitor.streaming.codec import decode_base64_image, decode_image

router = APIRouter()
logger = logging.getLogger("helmet_monitor")

CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


@router.websocket("/ws/monitor/{camera_id}")
async def monitor(ws: WebSocket, camera_id: str):
    services: Services = ws.app.state.services
    await ws.accept()

    if not CAMERA_ID.match(camera_id):
        await _reject(ws, "Camera id may only contain letters, numbers, '-' and '_'.")
        return
    if services.analyzer is None:
        await _reject(ws, "Helmet detector is not loaded; see /api/health.")
        return
    if camera_id in services.sessions:
        await _reject(ws, f"Camera '{camera_id}' is already streaming.")
        return

    session = services.new_session(camera_id)
    services.sessions[camera_id] = session
    processor = asyncio.create_task(session.run(ws.send_json))
    logger.info("Camera %s connected", camera_id)

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            frame = _frame_from(message)
            if frame is None:
                await ws.send_json({"type": "error", "message": "Could not decode frame."})
                continue
            session.submit(frame)
    except WebSocketDisconnect:
        pass
    finally:
        processor.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await processor
        services.sessions.pop(camera_id, None)
        logger.info("Camera %s disconnected", camera_id)


def _frame_from(message: dict):
    if message.get("bytes"):
        return decode_image(message["bytes"])
    text = message.get("text")
    if text:
        try:
            text = json.loads(text).get("image", "")
        except (json.JSONDecodeError, AttributeError):
            pass
        return decode_base64_image(text) if isinstance(text, str) else None
    return None


async def _reject(ws: WebSocket, reason: str) -> None:
    await ws.send_json({"type": "error", "message": reason})
    await ws.close(code=1008)
