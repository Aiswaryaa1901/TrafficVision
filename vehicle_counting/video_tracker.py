"""Member 2 - video vehicle tracking and counting runner.

Run from the TrafficVision root folder:

    python vehicle_counting/video_tracker.py                       # sample_data/Traffic1.mp4
    python vehicle_counting/video_tracker.py path/to/traffic.mp4
    python vehicle_counting/video_tracker.py traffic.mp4 --no-show   # headless / no window

MEMBER 3 INTEGRATION CHANGES (behaviour of the tracking and counting itself is
unchanged):
  * wrapped in main() so importing this file no longer runs the whole video;
  * the video path can be given on the command line (default is still
    sample_data/Traffic1.mp4, now resolved from the project root);
  * the output JSON path no longer depends on the current working directory;
  * a missing/zero FPS no longer crashes the run (25 FPS is assumed);
  * a machine without a display no longer crashes on the preview window;
  * minutes with no crossings are written as explicit zeros, and the JSON is
    also saved if the run stops early (press q);
  * a small sidecar file vehicle_counts_meta.json records the video duration so
    Member 3 can tell a complete last minute from a partial one.
The counting line / lanes keep their original values unless they would fall
outside the frame (e.g. a smaller video), in which case they are rescaled and a
warning is printed.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import cv2

from vehicle_detection import detect_frame, load_model
from vehicle_counting.tracker import VehicleTracker

DEFAULT_VIDEO = PROJECT_ROOT / "sample_data" / "Traffic1.mp4"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "vehicle_counts.json"
META_NAME = "vehicle_counts_meta.json"
DEFAULT_FPS = 25.0

# The original hard-coded values were chosen for a 1920x1080 video.
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080


def fit_counting_geometry(tracker, frame_width, frame_height):
    """Rescale line/lanes only if the original values do not fit the frame."""
    if tracker.counting_line_x >= frame_width:
        new_x = frame_width // 2
        print(
            f"WARNING: counting line x={tracker.counting_line_x} is outside the "
            f"{frame_width}px-wide frame; using x={new_x}."
        )
        tracker.counting_line_x = new_x

    if frame_height and max(tracker.lane_boundaries) >= frame_height:
        scale = frame_height / REFERENCE_HEIGHT
        scaled = [int(y * scale) for y in tracker.lane_boundaries]
        print(
            f"WARNING: lane boundaries {tracker.lane_boundaries} do not fit a "
            f"{frame_height}px-high frame; using {scaled}."
        )
        tracker.lane_boundaries = scaled


def write_outputs(tracker, output_path, video_path, fps, fps_assumed,
                  frames_processed, completed):
    """Write vehicle_counts.json (unchanged format) and the metadata sidecar."""
    duration = frames_processed / fps if fps else 0.0
    last_minute = int(max(duration - 1e-9, 0.0) // 60)
    if tracker.minute_counts:
        last_minute = max(last_minute, max(tracker.minute_counts))
    tracker.fill_missing_minutes(last_minute)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(tracker.export_minute_counts(), file, indent=4)

    meta = {
        "video_file": Path(video_path).name,
        "fps": fps,
        "fps_assumed": fps_assumed,
        "frames_processed": frames_processed,
        "duration_seconds": round(duration, 3),
        "last_minute_index": last_minute,
        "run_completed": completed,
        "counting_line_x": tracker.counting_line_x,
        "lane_boundaries": tracker.lane_boundaries,
        "total_crossings": tracker.vehicle_count,
        "unmapped_class_crossings": tracker.unmapped_count,
    }
    with open(output_path.with_name(META_NAME), "w") as file:
        json.dump(meta, file, indent=4)
    return output_path


def run(video_path, output_path=DEFAULT_OUTPUT, show=True, max_frames=None):
    """Process a video. Returns 0 on success, non-zero error code otherwise."""
    video_path = Path(video_path)
    if not video_path.is_file():
        print(f"Error: video not found: {video_path}")
        return 2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not open video")
        return 3

    model = load_model()
    tracker = VehicleTracker()

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_assumed = not (fps and fps > 0 and fps == fps)  # catches 0 and NaN
    if fps_assumed:
        fps = DEFAULT_FPS
        print(f"WARNING: video reports no FPS; assuming {DEFAULT_FPS:g}.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video width:", frame_width)
    print("Video height:", frame_height)
    if frame_width > 0:
        fit_counting_geometry(tracker, frame_width, frame_height)

    frame_id = 0
    completed = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % 100 == 0:
            print("Processed frames:", frame_id)

        timestamp = (frame_id - 1) / fps

        result = detect_frame(
            frame, frame_id=frame_id, timestamp=timestamp, model=model
        )
        detections = result["detections"]
        tracked_vehicles = tracker.update(detections, timestamp)

        if show:
            line_x = tracker.counting_line_x
            cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 0, 255), 3)
            cv2.putText(
                frame, f"Vehicle Count: {tracker.vehicle_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2,
            )
            for vehicle in tracked_vehicles:
                bbox = vehicle["bbox"]
                x1, y1 = int(bbox["x1"]), int(bbox["y1"])
                x2, y2 = int(bbox["x2"]), int(bbox["y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{vehicle['class']} ID: {vehicle['id']}"
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2,
                )
            try:
                cv2.imshow("TrafficVision - Vehicle Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    completed = False
                    break
            except cv2.error:
                print("WARNING: no display available; continuing without preview.")
                show = False

        if max_frames is not None and frame_id >= max_frames:
            completed = False
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    print("\nMINUTE-WISE, LANE-WISE VEHICLE COUNT")
    for minute, lanes in sorted(tracker.minute_counts.items()):
        print(f"\nMinute {minute}")
        for lane, counts in lanes.items():
            print(f"{lane}: {counts}")

    out = write_outputs(
        tracker, output_path, video_path, fps, fps_assumed, frame_id, completed
    )
    print("\nLane-wise JSON file saved successfully:", out)
    print("Total vehicles:", tracker.vehicle_count)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="TrafficVision vehicle counting")
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO),
                        help="path to the traffic video")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="where to write vehicle_counts.json")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open the preview window")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many frames (quick test)")
    args = parser.parse_args(argv)
    return run(args.video, args.output, show=not args.no_show,
               max_frames=args.max_frames)


if __name__ == "__main__":
    sys.exit(main())
