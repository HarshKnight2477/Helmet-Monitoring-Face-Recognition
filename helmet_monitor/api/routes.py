"""REST endpoints: health, workers, violations."""

from __future__ import annotations

import asyncio
import sqlite3

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from helmet_monitor.api.schemas import HealthOut, ViolationOut, WorkerCreated, WorkerOut
from helmet_monitor.recognition.embedder import largest_face
from helmet_monitor.services import Services
from helmet_monitor.streaming.codec import decode_image

router = APIRouter(prefix="/api")


def get_services(request: Request) -> Services:
    return request.app.state.services


@router.get("/health", response_model=HealthOut)
def health(services: Services = Depends(get_services)):
    ready = services.detector is not None and services.embedder is not None
    return HealthOut(
        status="ok" if ready else "degraded",
        detector_loaded=services.detector is not None,
        face_model_loaded=services.embedder is not None,
        registered_workers=len(services.gallery),
        active_cameras=len(services.sessions),
        errors=services.errors,
    )


@router.get("/cameras")
def cameras(services: Services = Depends(get_services)):
    return [session.status() for session in services.sessions.values()]


# Workers ---------------------------------------------------------------------

@router.get("/workers", response_model=list[WorkerOut])
def list_workers(services: Services = Depends(get_services)):
    return [WorkerOut(**vars(worker)) for worker in services.database.list_workers()]


@router.post("/workers", response_model=WorkerCreated, status_code=201)
async def register_worker(
    name: str = Form(..., min_length=1, max_length=100),
    email: str | None = Form(None, max_length=200),
    photo: UploadFile = File(...),
    services: Services = Depends(get_services),
):
    if services.embedder is None:
        raise HTTPException(503, "Face model is not loaded; see /api/health.")

    data = await photo.read()
    if len(data) > services.settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Photo is larger than {services.settings.max_upload_mb:g} MB.")
    image = decode_image(data)
    if image is None:
        raise HTTPException(422, "The upload is not a readable image.")

    # Off the event loop, so live camera streams keep flowing during registration.
    faces = await asyncio.to_thread(services.embedder.detect, image)
    face = largest_face(faces)
    if face is None:
        raise HTTPException(422, "No face found in the photo. Use a clear, front-facing photo.")

    try:
        worker = services.database.add_worker(name.strip(), (email or "").strip() or None,
                                              face.embedding)
    except sqlite3.IntegrityError:
        raise HTTPException(409, "A worker with this email is already registered.")
    services.gallery.add(worker.id, worker.name, face.embedding)
    return WorkerCreated(**vars(worker), faces_in_photo=len(faces))


@router.delete("/workers/{worker_id}", status_code=204)
def delete_worker(worker_id: int, services: Services = Depends(get_services)):
    # Gallery first, so a frame processed meanwhile can't match a deleted worker.
    services.gallery.remove(worker_id)
    if not services.database.delete_worker(worker_id):
        raise HTTPException(404, "Worker not found.")


# Violations ------------------------------------------------------------------

@router.get("/violations", response_model=list[ViolationOut])
def list_violations(limit: int = 50, worker_id: int | None = None,
                    services: Services = Depends(get_services)):
    records = services.database.list_violations(min(max(limit, 1), 500), worker_id)
    return [ViolationOut(**{**vars(r), "snapshot": f"/snapshots/{r.snapshot}" if r.snapshot else None})
            for r in records]


@router.get("/violations/summary")
def violation_summary(services: Services = Depends(get_services)):
    return services.database.violation_summary()
