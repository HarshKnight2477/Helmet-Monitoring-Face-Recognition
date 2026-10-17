import numpy as np
import pytest
from fastapi.testclient import TestClient

from helmet_monitor.main import create_app

from conftest import FakeDetector, FakeEmbedder, add_worker, bare_head, embedding, face_at, jpeg_bytes


@pytest.fixture
def client_for(make_services):
    def make(detector=None, embedder=None):
        services = make_services(detector=detector, embedder=embedder)
        return TestClient(create_app(services)), services
    return make


def register(client, name="Asha", email=None, photo=None):
    data = {"name": name}
    if email:
        data["email"] = email
    return client.post("/api/workers", data=data,
                       files={"photo": ("face.jpg", photo or jpeg_bytes(), "image/jpeg")})


def test_health_reports_missing_models(client_for):
    client, _ = client_for()
    with client:
        health = client.get("/api/health").json()
    assert health["status"] == "degraded"
    assert not health["detector_loaded"] and not health["face_model_loaded"]


def test_health_ok_when_models_loaded(client_for):
    client, _ = client_for(FakeDetector(), FakeEmbedder())
    with client:
        assert client.get("/api/health").json()["status"] == "ok"


def test_index_page_is_served(client_for):
    client, _ = client_for()
    with client:
        response = client.get("/")
    assert response.status_code == 200 and "Helmet Monitor" in response.text


def test_register_list_and_delete_worker(client_for):
    client, services = client_for(embedder=FakeEmbedder([face_at(seed=3)]))
    with client:
        response = register(client, "Asha", "asha@example.com")
        assert response.status_code == 201
        worker = response.json()
        assert worker["name"] == "Asha" and worker["faces_in_photo"] == 1
        assert services.gallery.match(embedding(3)).name == "Asha"

        assert [w["name"] for w in client.get("/api/workers").json()] == ["Asha"]
        assert client.delete(f"/api/workers/{worker['id']}").status_code == 204
        assert client.delete(f"/api/workers/{worker['id']}").status_code == 404
        assert len(services.gallery) == 0


def test_register_rejects_photo_without_face(client_for):
    client, _ = client_for(embedder=FakeEmbedder([]))
    with client:
        response = register(client)
    assert response.status_code == 422 and "No face" in response.json()["detail"]


def test_register_rejects_non_image(client_for):
    client, _ = client_for(embedder=FakeEmbedder([face_at()]))
    with client:
        assert register(client, photo=b"definitely not a jpeg").status_code == 422


def test_register_rejects_duplicate_email(client_for):
    client, _ = client_for(embedder=FakeEmbedder([face_at()]))
    with client:
        register(client, "Asha", "asha@example.com")
        assert register(client, "Asha 2", "asha@example.com").status_code == 409


def test_register_needs_face_model(client_for):
    client, _ = client_for()
    with client:
        assert register(client).status_code == 503


def test_violation_log_and_snapshot(client_for):
    detector = FakeDetector([bare_head()])
    client, services = client_for(detector, FakeEmbedder([face_at(seed=3)]))
    add_worker(services, "Asha", seed=3)
    with client:
        session = services.new_session("cam-1")
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = services.analyzer.analyze(frame)
        events = [e for _ in range(3) for e in session.tracker.update(result)]  # confirm_frames=3
        assert len(events) == 1
        services.record_violation(events[0], frame)

        [record] = client.get("/api/violations").json()
        assert record["worker_name"] == "Asha" and record["camera_id"] == "cam-1"
        snapshot = client.get(record["snapshot"])
        assert snapshot.status_code == 200 and snapshot.headers["content-type"] == "image/jpeg"
        assert client.get("/api/violations/summary").json()["total"] == 1


@pytest.mark.parametrize("name", ["../secret.jpg", "x.png", "missing.jpg"])
def test_snapshot_route_rejects_bad_names(client_for, name):
    client, _ = client_for()
    with client:
        assert client.get(f"/snapshots/{name}").status_code == 404
