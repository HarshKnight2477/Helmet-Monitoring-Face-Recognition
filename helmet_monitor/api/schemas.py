from pydantic import BaseModel


class WorkerOut(BaseModel):
    id: int
    name: str
    email: str | None
    created_at: str


class WorkerCreated(WorkerOut):
    faces_in_photo: int


class ViolationOut(BaseModel):
    id: int
    camera_id: str
    worker_id: int | None
    worker_name: str
    similarity: float | None
    confidence: float
    snapshot: str | None
    created_at: str


class HealthOut(BaseModel):
    status: str
    detector_loaded: bool
    face_model_loaded: bool
    face_device: str | None
    registered_workers: int
    active_cameras: int
    errors: dict[str, str]
