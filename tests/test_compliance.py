import numpy as np

from helmet_monitor.compliance.analyzer import FrameAnalyzer
from helmet_monitor.compliance.tracker import ViolationTracker
from helmet_monitor.domain import Box, FrameResult, HeadState, Identity, PersonResult
from helmet_monitor.recognition.gallery import FaceGallery

from conftest import FakeDetector, FakeEmbedder, bare_head, embedding, face_at, helmet

FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def make_analyzer(detections, faces, workers=()):
    gallery = FaceGallery(0.45)
    gallery.load(workers)
    detector, embedder = FakeDetector(detections), FakeEmbedder(faces)
    analyzer = FrameAnalyzer(detector, gallery, embedder, frozenset({"helmet"}),
                             frozenset({"head"}))
    return analyzer, detector, embedder


class TestAnalyzer:
    def test_identifies_violator(self):
        analyzer, _, _ = make_analyzer([bare_head()], [face_at(seed=5)], [(7, "Asha", embedding(5))])
        result = analyzer.analyze(FRAME)
        assert len(result.violations) == 1
        person = result.violations[0]
        assert (person.identity.worker_id, person.identity.name) == (7, "Asha")
        assert person.face_box is not None

    def test_compliant_worker_is_identified_but_not_a_violation(self):
        analyzer, _, _ = make_analyzer([helmet()], [face_at(seed=5)], [(7, "Asha", embedding(5))])
        result = analyzer.analyze(FRAME)
        assert result.violations == []
        assert result.people[0].identity.name == "Asha"

    def test_unregistered_face_is_unknown(self):
        analyzer, _, _ = make_analyzer([bare_head()], [face_at(seed=5)], [(7, "Asha", embedding(6))])
        assert analyzer.analyze(FRAME).violations[0].identity.name == "Unknown"

    def test_head_without_visible_face_has_no_identity(self):
        analyzer, _, _ = make_analyzer([bare_head()], [])
        person = analyzer.analyze(FRAME).people[0]
        assert person.identity is None and person.face_box is None

    def test_face_model_is_skipped_when_nobody_is_in_view(self):
        analyzer, detector, embedder = make_analyzer([], [face_at()])
        assert analyzer.analyze(FRAME).people == []
        assert (detector.calls, embedder.calls) == (1, 0)

    def test_result_serialises_for_the_websocket(self):
        analyzer, _, _ = make_analyzer([bare_head(), helmet(200, 50, 260, 120)], [])
        data = analyzer.analyze(FRAME).to_dict()
        assert (data["total_people"], data["with_helmet"], data["without_helmet"]) == (2, 1, 1)
        assert data["people"][0]["head_box"] == [100, 50, 160, 120]


def violation(worker_id=None, name="Unknown", confidence=0.9) -> FrameResult:
    identity = Identity(worker_id, name, 0.8 if worker_id else 0.1)
    head = HeadState(Box(0, 0, 10, 10), wearing_helmet=False, confidence=confidence)
    return FrameResult(people=[PersonResult(head=head, identity=identity)])


def compliant() -> FrameResult:
    return FrameResult(people=[PersonResult(head=HeadState(Box(0, 0, 10, 10), True, 0.9))])


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TestTracker:
    def test_needs_consecutive_frames_before_logging(self):
        tracker = ViolationTracker("cam", confirm_frames=3, clock=FakeClock())
        assert tracker.update(violation(1, "Asha")) == []
        assert tracker.update(violation(1, "Asha")) == []
        events = tracker.update(violation(1, "Asha"))
        assert [(e.worker_id, e.name, e.camera_id) for e in events] == [(1, "Asha", "cam")]

    def test_streak_resets_when_violation_disappears(self):
        tracker = ViolationTracker("cam", confirm_frames=3, clock=FakeClock())
        tracker.update(violation(1, "Asha"))
        tracker.update(violation(1, "Asha"))
        tracker.update(compliant())
        assert tracker.update(violation(1, "Asha")) == []

    def test_cooldown_limits_repeat_logging(self):
        clock = FakeClock()
        tracker = ViolationTracker("cam", confirm_frames=1, cooldown_seconds=60, clock=clock)
        assert len(tracker.update(violation(1, "Asha"))) == 1
        clock.now += 30
        assert tracker.update(violation(1, "Asha")) == []
        clock.now += 31
        assert len(tracker.update(violation(1, "Asha"))) == 1

    def test_cooldown_is_per_worker(self):
        tracker = ViolationTracker("cam", confirm_frames=1, clock=FakeClock())
        tracker.update(violation(1, "Asha"))
        assert [e.name for e in tracker.update(violation(2, "Ravi"))] == ["Ravi"]

    def test_unknown_violators_are_logged_without_identity(self):
        tracker = ViolationTracker("cam", confirm_frames=1, clock=FakeClock())
        event = tracker.update(violation())[0]
        assert event.worker_id is None and event.name == "Unknown" and event.similarity is None

    def test_compliant_people_never_produce_events(self):
        tracker = ViolationTracker("cam", confirm_frames=1, clock=FakeClock())
        assert tracker.update(compliant()) == []
