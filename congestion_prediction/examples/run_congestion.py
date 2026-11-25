r"""Run Member 3 on Member 2's output and print a summary.

From the TrafficVision root folder (PowerShell):

    python congestion_prediction\examples\run_congestion.py
    python congestion_prediction\examples\run_congestion.py path\to\vehicle_counts.json --start-time 2026-11-25T10:00:00
    python congestion_prediction\examples\run_congestion.py --demo-synthetic      # SYNTHETIC data, not real traffic
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from congestion_prediction import build_traffic_report, save_report  # noqa: E402
from congestion_prediction.synthetic import DATA_LABEL, generate_synthetic_counts  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="TrafficVision Member 3 demo")
    parser.add_argument("input", nargs="?", default=None,
                        help="vehicle_counts.json or .csv (default: Member 2's output)")
    parser.add_argument("--interval", type=int, default=None, help="interval length in minutes")
    parser.add_argument("--start-time", default=None, help="ISO time of video minute 0")
    parser.add_argument("--cumulative", action="store_true", help="input holds running totals")
    parser.add_argument("--method", default=None, help="auto | weighted_moving_average | "
                        "exponential_smoothing | damped_trend")
    parser.add_argument("--output", default=None, help="where to save the JSON report")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--demo-synthetic", action="store_true",
                        help="use generated SYNTHETIC data (clearly labelled)")
    args = parser.parse_args(argv)

    source, label = args.input, None
    if args.demo_synthetic:
        source, label = generate_synthetic_counts(minutes=60), DATA_LABEL
        print("*** SYNTHETIC DEMO DATA - not real traffic ***")

    report = build_traffic_report(
        source, interval_minutes=args.interval, cumulative=args.cumulative,
        start_time=args.start_time, method=args.method, data_label=label)

    print(f"status: {report['status']}  data: {report['data_source']['label']}")
    if report["error"]:
        print("error:", report["error"])
    latest = report["latest"]
    if latest:
        flow = latest["vehicle_flow"]
        print(f"latest interval ({latest['video_time']}): {flow['total']} vehicles, "
              f"{latest['flow_per_minute']} /min -> {latest['congestion']['level']}"
              + ("  [partial]" if latest["is_partial"] else ""))
    prediction = report["prediction"]
    print(f"prediction: {prediction['status']} ({prediction['message']})")
    if prediction["method"]:
        print("method:", prediction["method"])
    for name, item in prediction["horizons"].items():
        print(f"  {name}: {item['predicted_vehicle_flow']} veh/min -> {item['predicted_level']}")
    evaluation = report["evaluation"]
    print("evaluation:", evaluation["status"], evaluation.get("reason", ""))
    for warning in report["data_quality"].get("warnings", [])[:5]:
        print("  warning:", warning)

    if not args.no_save:
        print("saved:", save_report(report, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
