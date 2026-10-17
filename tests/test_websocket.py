import base64
import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from helmet_monitor.main import create_app

from conftest import FakeDetector, FakeEmbedder, add_worker, bare_head, face_at, jpeg_bytes


@pytest.fixture
def monitored(make_services):
    services = make_services(FakeDetector([bare_head()]), FakeEmbedder([face_at(seed=4)]))
    add_worker(services, "Asha", seed=4)
    return TestClient(create_app(services)), services


def test_streams_results_and_logs_confirmed_violation(monitored):
    client, services = monitored
    with client:
        with client.websocket_connect("/ws/monitor/cam-1") as ws:
            results = []
            for _ in range(3):
                ws.send_bytes(jpeg_bytes())
                results.append(ws.receive_json())
            assert "cam-1" in services.sessions
        [record] = services.database.list_violations()

    assert [r["type"] for r in results] == ["result"] * 3
    first = results[0]
    assert first["frame"] == {"width": 320, "height": 240}
    assert first["without_helmet"] == 1
    assert first["people"][0]["name"] == "Asha"
    # Recorded once (confirm time is 0 in tests), then held back by the cooldown.
    assert [len(r["new_violations"]) for r in results] == [1, 0, 0]
    assert results[0]["new_violations"][0]["name"] == "Asha"
    assert first["people"][0]["timer"] == {"elapsed": 0.0, "remaining": 0.0, "recorded": True}
    assert first["confirm_seconds"] == 0

    assert record.worker_name == "Asha" and record.snapshot
    assert services.sessions == {}  # session removed on disconnect


def test_accepts_base64_json_frames(monitored):
    client, _ = monitored
    payload = json.dumps({"image": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes()).decode()})
    with client, client.websocket_connect("/ws/monitor/cam-2") as ws:
        ws.send_text(payload)
        assert ws.receive_json()["type"] == "result"


def test_undecodable_frame_returns_error_and_keeps_stream_open(monitored):
    client, _ = monitored
    with client, client.websocket_connect("/ws/monitor/cam-3") as ws:
        ws.send_bytes(b"garbage")
        assert ws.receive_json() == {"type": "error", "message": "Could not decode frame."}
        ws.send_bytes(jpeg_bytes())
        assert ws.receive_json()["type"] == "result"


def test_rejects_when_detector_missing(make_services):
    client = TestClient(create_app(make_services()))
    with client, client.websocket_connect("/ws/monitor/cam-1") as ws:
        assert "not loaded" in ws.receive_json()["message"]
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_rejects_invalid_camera_id(monitored):
    client, _ = monitored
    with client, client.websocket_connect("/ws/monitor/bad id!") as ws:
        assert "Camera id" in ws.receive_json()["message"]


def test_rejects_duplicate_camera(monitored):
    client, _ = monitored
    with client, client.websocket_connect("/ws/monitor/cam-1") as first:
        first.send_bytes(jpeg_bytes())
        first.receive_json()
        with client.websocket_connect("/ws/monitor/cam-1") as second:
            assert "already streaming" in second.receive_json()["message"]
