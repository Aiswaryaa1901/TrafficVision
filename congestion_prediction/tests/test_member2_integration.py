"""Integration checks against Member 2's REAL code (vehicle_counting/).

The detector is STUBBED (no YOLO model needed) and the video is a SYNTHETIC
generated clip, so these verify the counting / file-format / Member 3 hand-over
logic - not detection quality.
"""

import json
import math

import pytest

from congestion_prediction import build_traffic_report
from vehicle_counting.tracker import VehicleTracker


def det(cls, cx, cy, size=40):
    return {"class": cls, "class_id": 2, "confidence": 0.9,
            "bbox": {"x1": cx - size, "y1": cy - size, "x2": cx + size, "y2": cy + size}}


def drive(tracker, cls, start_x, y, t0, speed=40, fps=10):
    """Move one vehicle right->left across the counting line, 1 detection/frame."""
    x, t = start_x, t0
    while x > 700:
        tracker.update([det(cls, x, y)], t)
        x -= speed
        t += 1 / fps


def test_default_tracker_behaviour_unchanged():
    t = VehicleTracker()
    assert (t.counting_line_x, t.lane_boundaries, t.max_distance, t.max_missed) == \
           (960, [300, 459, 647], 50, 10)
    assert t.get_lane(100) == "lane_1" and t.get_lane(500) == "lane_3" and t.get_lane(900) == "lane_4"


def test_crossing_is_counted_once_per_lane_and_class():
    t = VehicleTracker()
    drive(t, "car", 1200, 100, 5.0)          # lane_1, minute 0
    drive(t, "truck", 1200, 400, 8.0)        # lane_2
    drive(t, "motorcycle", 1200, 700, 12.0)  # lane_4
    assert t.vehicle_count == 3
    m0 = t.minute_counts[0]
    assert m0["lane_1"]["cars"] == 1 and m0["lane_2"]["trucks"] == 1
    assert m0["lane_4"]["two_wheelers"] == 1


def test_minutes_without_crossings_become_explicit_zeros():
    t = VehicleTracker()
    drive(t, "car", 1200, 100, 5.0)          # minute 0
    drive(t, "bus", 1200, 100, 130.0)        # minute 2 (nothing in minute 1)
    assert sorted(t.minute_counts) == [0, 2]          # original behaviour: gap
    t.fill_missing_minutes(2)
    exported = t.export_minute_counts()
    assert [e["minute"] for e in exported] == [0, 1, 2]
    assert all(v == 0 for lane in exported[1]["lanes"].values() for v in lane.values())
    assert exported[2]["lanes"]["lane_1"]["buses"] == 1


def test_tracker_does_not_crash_on_bad_input():
    t = VehicleTracker()
    for stamp in (None, "x", -5, float("nan")):
        t.update([det("car", 1000, 100), {"class": "car"}, {"bbox": {}}], stamp)
    assert t.update(None, 1.0) == []
    t.update([det("ambulance", 1000, 100)], 2.0)
    t.update([det("ambulance", 960, 100)], 2.1)
    assert t.vehicle_count == 1 and t.unmapped_count == 1      # counted, no category


def test_custom_geometry_arguments():
    t = VehicleTracker(counting_line_x=100, lane_boundaries=[10, 20, 30], max_distance=5)
    assert (t.counting_line_x, t.lane_boundaries, t.max_distance) == (100, [10, 20, 30], 5)


# ------------------ video_tracker.py end to end (stub detector) ---------------------
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


@pytest.fixture()
def vt(monkeypatch):
    from vehicle_counting import video_tracker
    monkeypatch.setattr(video_tracker, "load_model", lambda *a, **k: object())
    return video_tracker


def make_video(path, frames=60, fps=10, size=(320, 240)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write test video here")
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def moving_car_detector(width=320):
    """Stub of detect_frame: one car crossing right->left every 20 frames."""
    def fake(frame, frame_id=None, timestamp=None, model=None, **kw):
        phase = (frame_id - 1) % 20
        cx = width - 15 * phase
        return {"frame_id": frame_id, "timestamp": timestamp,
                "detections": [det("car", cx, 60, size=10)]}
    return fake


def test_video_tracker_runs_headless_and_feeds_member3(vt, monkeypatch, tmp_path):
    video = tmp_path / "clip.avi"
    make_video(video, frames=60, fps=10)                 # 6 s, 320x240: the 960 px line is off-frame
    monkeypatch.setattr(vt, "detect_frame", moving_car_detector())
    out = tmp_path / "vehicle_counts.json"
    assert vt.run(video, out, show=False) == 0

    data = json.loads(out.read_text())
    assert data and data[0]["minute"] == 0 and set(data[0]["lanes"]) == {"lane_1", "lane_2", "lane_3", "lane_4"}
    meta = json.loads((tmp_path / "vehicle_counts_meta.json").read_text())
    assert math.isclose(meta["duration_seconds"], 6.0, abs_tol=0.2) and meta["counting_line_x"] < 320
    total = sum(v for e in data for lane in e["lanes"].values() for v in lane.values())
    assert total == meta["total_crossings"] > 0

    report = build_traffic_report(out)                   # Member 3 reads Member 2's real output
    assert report["status"] in ("ok", "ok_with_warnings")
    assert report["latest"]["vehicle_flow"]["total"] == total
    assert report["data_quality"]["last_interval_completeness"] == "partial"   # 6 s < 1 min
    assert report["latest"]["is_partial"] is True


def test_video_tracker_handles_missing_video_and_zero_fps(vt, monkeypatch, tmp_path):
    assert vt.run(tmp_path / "nope.mp4", tmp_path / "o.json", show=False) == 2
    assert not (tmp_path / "o.json").exists()

    class FakeCap:
        def __init__(self, *a): self.n = 0
        def isOpened(self): return True
        def get(self, prop): return 0.0 if prop == cv2.CAP_PROP_FPS else 320 if prop == cv2.CAP_PROP_FRAME_WIDTH else 240
        def read(self):
            self.n += 1
            return (self.n <= 5), np.zeros((240, 320, 3), dtype=np.uint8)
        def release(self): pass

    video = tmp_path / "x.avi"
    video.write_bytes(b"")
    monkeypatch.setattr(vt.cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(vt, "detect_frame", moving_car_detector())
    out = tmp_path / "zero_fps.json"
    video.touch()
    assert vt.run(video, out, show=False) == 0
    meta = json.loads((tmp_path / "vehicle_counts_meta.json").read_text())
    assert meta["fps_assumed"] is True and meta["fps"] == 25.0

def test_invalid_timestamp_does_not_create_minute_zero_count():
    t = VehicleTracker()

    # First valid detection establishes the vehicle track.
    t.update([det("car", 990, 100)], 59.9)

    # Invalid timestamp: tracking/counting must not update.
    t.update([det("car", 980, 100)], None)

    assert t.vehicle_count == 0
    assert t.minute_counts == {}

    # Next valid detection crosses the line in minute 1.
    t.update([det("car", 960, 100)], 60.1)

    assert t.vehicle_count == 1
    assert 0 not in t.minute_counts
    assert t.minute_counts[1]["lane_1"]["cars"] == 1
