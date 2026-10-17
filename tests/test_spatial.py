import pytest

from helmet_monitor.compliance.spatial import assign_faces, resolve_heads
from helmet_monitor.domain import Box, Detection

from conftest import bare_head, face_at, helmet

HELMET = frozenset({"helmet"})
NO_HELMET = frozenset({"head", "no_helmet"})


class TestBox:
    def test_iou_of_identical_boxes_is_one(self):
        assert Box(0, 0, 10, 10).iou(Box(0, 0, 10, 10)) == 1.0

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert Box(0, 0, 10, 10).iou(Box(20, 20, 30, 30)) == 0.0

    def test_iou_of_half_overlap(self):
        assert Box(0, 0, 10, 10).iou(Box(5, 0, 15, 10)) == pytest.approx(50 / 150)

    def test_expand_grows_by_fraction_of_size(self):
        assert Box(10, 10, 20, 30).expand(0.5, 0, 0.5, 1.0) == Box(5, 10, 25, 50)


class TestResolveHeads:
    def test_classifies_helmet_and_bare_heads(self):
        heads = resolve_heads([helmet(0, 0, 50, 50), bare_head(200, 0, 250, 50)], HELMET, NO_HELMET)
        assert sorted(h.wearing_helmet for h in heads) == [False, True]

    def test_conflicting_boxes_on_same_head_keep_most_confident(self):
        heads = resolve_heads([helmet(confidence=0.6), bare_head(confidence=0.8)], HELMET, NO_HELMET)
        assert len(heads) == 1
        assert heads[0].wearing_helmet is False
        assert heads[0].confidence == 0.8

    def test_ignores_other_classes_and_is_case_insensitive(self):
        detections = [Detection(Box(0, 0, 50, 50), "person", 0.9),
                      Detection(Box(100, 0, 150, 50), "Helmet", 0.9)]
        heads = resolve_heads(detections, HELMET, NO_HELMET)
        assert [h.wearing_helmet for h in heads] == [True]

    def test_nearby_but_separate_people_are_both_kept(self):
        heads = resolve_heads([bare_head(0, 0, 50, 50), bare_head(40, 0, 90, 50)], HELMET, NO_HELMET)
        assert len(heads) == 2


class TestAssignFaces:
    def test_face_inside_head_is_assigned(self):
        assert assign_faces([Box(100, 50, 160, 120)], [face_at()]) == {0: 0}

    def test_face_outside_every_head_is_ignored(self):
        assert assign_faces([Box(100, 50, 160, 120)], [face_at(400, 300, 440, 340)]) == {}

    def test_face_below_a_helmet_box_still_matches(self):
        # Helmet box covers mostly the helmet; the face centre sits just below it.
        helmet_box = Box(100, 40, 160, 90)
        assert assign_faces([helmet_box], [face_at(110, 80, 150, 125)]) == {0: 0}

    def test_each_face_goes_to_one_head_closest_first(self):
        heads = [Box(100, 50, 160, 120), Box(150, 50, 210, 120)]
        faces = [face_at(165, 70, 195, 110, seed=1), face_at(115, 70, 145, 110, seed=2)]
        assert assign_faces(heads, faces) == {0: 1, 1: 0}

    def test_no_faces(self):
        assert assign_faces([Box(0, 0, 10, 10)], []) == {}
