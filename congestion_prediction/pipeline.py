"""One-call integration point for Member 4 (dashboard): build_traffic_report().

    from congestion_prediction import build_traffic_report
    report = build_traffic_report()                 # Member 2's vehicle_counts.json
    report["latest"]["congestion"]["level"]         # FREE / MODERATE / HEAVY / UNKNOWN
    report["prediction"]["horizons"]["next_10_minutes"]["predicted_level"]

The function never raises on bad data: problems are reported in
``report["status"]``, ``report["error"]`` and ``report["data_quality"]``.
It only READS Member 2's file.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from . import config
from .congestion_classifier import (
    InvalidTrafficDataError, classify_flow, effective_flow, get_thresholds,
)
from .data_processor import DataFormatError, ProcessedTraffic, process_traffic_data
from .traffic_predictor import evaluate_forecast, predict_traffic


def _describe_interval(interval: Dict[str, Any], lanes: Optional[int],
                       overrides: Optional[Dict[str, float]]) -> Dict[str, Any]:
    """Public per-interval record (flow, level, per-lane levels)."""
    entry: Dict[str, Any] = {
        "index": interval["index"],
        "video_time": interval["video_time"],
        "timestamp": interval["timestamp"],
        "observed": interval["observed"],
        "complete": interval["complete"],
        "vehicle_flow": interval["counts"],
        "flow_per_minute": interval["flow_per_minute"],
        "congestion": {"level": config.LEVEL_UNKNOWN, "flow_per_minute": None},
        "lanes": {},
    }
    if not interval["observed"] or interval["counts"] is None:
        return entry
    minutes = interval["observed_minutes"]
    flow = effective_flow(interval["counts"], minutes)
    entry["congestion"] = {"level": classify_flow(flow, lanes, overrides),
                           "flow_per_minute": round(flow, 4)}
    for lane, counts in interval["lanes"].items():
        lane_flow = effective_flow(counts, minutes)
        entry["lanes"][lane] = {
            "vehicle_flow": counts,
            "flow_per_minute": round(lane_flow, 4),
            "level": classify_flow(lane_flow, 1, overrides),
        }
    return entry


def _empty_report(status: str, error: Optional[str], label: str,
                  horizons: Sequence[int], interval: int) -> Dict[str, Any]:
    prediction = predict_traffic([], horizons=horizons, interval_minutes=interval)
    return {
        "schema_version": config.SCHEMA_VERSION,
        "status": status, "error": error,
        "data_source": {"label": label},
        "interval_minutes": interval,
        "history": [], "latest": None,
        "prediction": prediction,
        "evaluation": {"status": "not_evaluated", "reason": "no data"},
        "data_quality": {"status": status, "warnings": [error] if error else []},
        "disclaimer": config.DISCLAIMER,
    }


def build_traffic_report(
    source: Union[None, str, Path, List[Any], Dict[str, Any]] = None,
    *,
    interval_minutes: Optional[int] = None,
    cumulative: bool = False,
    start_time: Union[None, str, datetime] = None,
    meta: Optional[Dict[str, Any]] = None,
    horizons: Optional[Sequence[int]] = None,
    method: Optional[str] = None,
    lanes: Optional[int] = None,
    thresholds: Optional[Dict[str, float]] = None,
    evaluate: bool = True,
    data_label: Optional[str] = None,
    missing_policy: Optional[str] = None,
    duplicate_policy: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Process -> classify -> forecast -> (evaluate) and return one JSON-ready dict.

    ``source``: path to Member 2's vehicle_counts.json (default), a CSV, or a list
    of records. ``data_label`` names the data in the report (use "SYNTHETIC_DEMO"
    for generated data). ``strict=True`` re-raises read errors instead of
    returning status "error".
    """
    horizons = tuple(horizons or config.FORECAST_HORIZONS_MINUTES)
    interval = int(interval_minutes or config.INTERVAL_MINUTES)
    label = data_label or "member2"

    try:
        processed: ProcessedTraffic = process_traffic_data(
            source, interval_minutes=interval, cumulative=cumulative,
            start_time=start_time, meta=meta, missing_policy=missing_policy,
            duplicate_policy=duplicate_policy)
    except DataFormatError as exc:
        if strict:
            raise
        return _empty_report("error", str(exc), label, horizons, interval)

    if not processed.intervals:
        report = _empty_report("no_data", "no usable traffic records", label, horizons, interval)
        report["data_quality"] = dict(processed.quality, warnings=processed.warnings)
        return report

    lane_count = lanes or (len(processed.lane_names) if processed.lane_names
                           and processed.lane_names != ["all"] else None)
    limits = get_thresholds(lane_count, thresholds)

    history: List[Dict[str, Any]] = []
    series: List[Optional[float]] = []
    for interval_data in processed.intervals:
        history.append(_describe_interval(interval_data, lane_count, thresholds))
        usable = interval_data["usable_for_forecast"] and interval_data["counts"] is not None
        series.append(effective_flow(interval_data["counts"], interval_data["observed_minutes"])
                      if usable else None)

    # "latest" = most recent complete interval (falls back to last observed).
    latest_index = next((i for i in range(len(history) - 1, -1, -1)
                         if history[i]["observed"] and history[i]["complete"]), None)
    if latest_index is None:
        latest_index = next((i for i in range(len(history) - 1, -1, -1)
                             if history[i]["observed"]), len(history) - 1)
    latest = dict(history[latest_index])
    latest["is_partial"] = not latest["complete"]

    try:
        prediction = predict_traffic(series, horizons=horizons, interval_minutes=interval,
                                     lanes=lane_count, method=method, thresholds=thresholds)
    except (InvalidTrafficDataError, ValueError) as exc:
        if strict:
            raise
        prediction = predict_traffic([], horizons=horizons, interval_minutes=interval)
        prediction["message"] = f"prediction failed: {exc}"
    prediction["based_on"] = {
        "last_history_interval_index": history[-1]["index"],
        "forecast_starts_after_video_time": processed.intervals[-1]["video_time"],
        "note": "history = measured counts; prediction = estimates for the future",
    }

    evaluation = (evaluate_forecast(series, horizons=horizons, interval_minutes=interval)
                  if evaluate else {"status": "skipped", "reason": "evaluate=False"})

    quality = dict(processed.quality)
    quality["warnings"] = processed.warnings
    quality["lanes_assumed"] = lane_count is None
    status = "ok"
    if prediction["status"] != "ok" or quality.get("status") == "warnings":
        status = "ok_with_warnings"

    return {
        "schema_version": config.SCHEMA_VERSION,
        "status": status,
        "error": None,
        "data_source": {"label": label, "file": processed.source,
                        "counts_are_cumulative": cumulative},
        "interval_minutes": interval,
        "classification": {
            "method": "estimated traffic-flow based (vehicles per minute)",
            "uses_pcu_weights": config.USE_PCU,
            "lanes": limits["lanes"],
            "thresholds_vehicles_per_minute": {"FREE_up_to": limits["free_max"],
                                               "MODERATE_up_to": limits["moderate_max"]},
            "note": "thresholds are configurable assumptions, not calibrated on labelled data",
        },
        "history": history,
        "latest": latest,
        "prediction": prediction,
        "evaluation": evaluation,
        "data_quality": quality,
        "disclaimer": config.DISCLAIMER,
    }


def save_report(report: Dict[str, Any], path: Union[None, str, Path] = None) -> Path:
    """Write the report as JSON (default: congestion_prediction/outputs/).

    Refuses to write over Member 2's vehicle_counts.json.
    """
    target = Path(path) if path else config.DEFAULT_REPORT_PATH
    protected = config.DEFAULT_INPUT_PATH.resolve()
    if target.resolve() == protected or target.name.startswith("vehicle_counts"):
        raise ValueError(f"refusing to overwrite Member 2's data file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return target
