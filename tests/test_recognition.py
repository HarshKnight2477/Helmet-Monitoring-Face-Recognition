import numpy as np
import pytest

from helmet_monitor.recognition.embedder import largest_face
from helmet_monitor.recognition.gallery import FaceGallery, normalize

from conftest import embedding, face_at


def test_normalize_gives_unit_length():
    assert np.linalg.norm(normalize(np.array([3.0, 4.0]))) == pytest.approx(1.0)


def test_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        normalize(np.zeros(4))


def test_empty_gallery_returns_unknown():
    identity = FaceGallery(0.45).match(embedding(1))
    assert identity.worker_id is None and identity.name == "Unknown"


def test_matches_registered_worker_by_cosine_similarity():
    gallery = FaceGallery(0.45)
    gallery.load([(1, "Asha", embedding(1)), (2, "Ravi", embedding(2))])
    # A slightly noisy version of Ravi's face should still match Ravi.
    noisy = embedding(2) + 0.02 * np.random.default_rng(9).standard_normal(512)
    identity = gallery.match(noisy)
    assert (identity.worker_id, identity.name) == (2, "Ravi")
    assert identity.similarity > 0.9


def test_stranger_below_threshold_is_unknown():
    gallery = FaceGallery(0.45)
    gallery.add(1, "Asha", embedding(1))
    identity = gallery.match(embedding(99))  # random vectors are ~orthogonal
    assert not identity.is_known
    assert identity.similarity < 0.45


def test_similarity_is_scale_invariant():
    gallery = FaceGallery(0.45)
    gallery.add(1, "Asha", embedding(1) * 7.5)
    assert gallery.match(embedding(1) * 0.1).similarity == pytest.approx(1.0, abs=1e-5)


def test_remove_worker():
    gallery = FaceGallery(0.45)
    gallery.add(1, "Asha", embedding(1))
    gallery.add(2, "Ravi", embedding(2))
    assert gallery.remove(1)
    assert not gallery.remove(1)
    assert len(gallery) == 1
    assert gallery.match(embedding(1)).name == "Unknown"
    assert gallery.match(embedding(2)).name == "Ravi"
    gallery.remove(2)
    assert gallery.match(embedding(2)).name == "Unknown"


def test_largest_face():
    small, big = face_at(0, 0, 10, 10), face_at(0, 0, 50, 50)
    assert largest_face([small, big]) is big
    assert largest_face([]) is None
