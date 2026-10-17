import base64

import numpy as np

from helmet_monitor.storage.database import Database
from helmet_monitor.streaming.codec import decode_base64_image, decode_image, encode_jpeg

from conftest import embedding, jpeg_bytes


def test_worker_embedding_round_trip(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    worker = db.add_worker("Asha", "asha@example.com", embedding(1))
    [(worker_id, name, stored)] = db.worker_embeddings()
    assert (worker_id, name) == (worker.id, "Asha")
    np.testing.assert_allclose(stored, embedding(1))
    db.close()


def test_deleting_worker_keeps_violation_history(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    worker = db.add_worker("Asha", None, embedding(1))
    db.add_violation("cam-1", worker.id, "Asha", 0.8, 0.9, None)
    assert db.delete_worker(worker.id)
    assert not db.delete_worker(worker.id)
    [record] = db.list_violations()
    assert record.worker_id is None and record.worker_name == "Asha"
    db.close()


def test_violation_filters_and_summary():
    db = Database(":memory:")
    asha = db.add_worker("Asha", None, embedding(1))
    db.add_violation("cam-1", asha.id, "Asha", 0.8, 0.9, "a.jpg")
    db.add_violation("cam-1", asha.id, "Asha", 0.8, 0.9, None)
    db.add_violation("cam-2", None, "Unknown", None, 0.7, None)
    assert len(db.list_violations(worker_id=asha.id)) == 2
    assert db.list_violations(limit=1)[0].worker_name == "Unknown"  # newest first
    summary = db.violation_summary()
    assert summary["total"] == 3 and summary["today"] == 3
    assert summary["by_worker"][0] == {"worker_name": "Asha", "count": 2}
    db.close()


def test_jpeg_round_trip():
    image = np.full((40, 60, 3), 200, dtype=np.uint8)
    decoded = decode_image(encode_jpeg(image))
    assert decoded.shape == (40, 60, 3)


def test_base64_data_url():
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes()).decode()
    assert decode_base64_image(data_url).shape == (240, 320, 3)


def test_garbage_is_rejected():
    assert decode_image(b"not an image") is None
    assert decode_image(b"") is None
    assert decode_base64_image("@@@not base64@@@") is None
