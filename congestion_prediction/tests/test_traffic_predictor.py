"""Tests for forecasting. All series are SYNTHETIC (generated or hand-made)."""

import math

import pytest

from congestion_prediction.synthetic import generate_synthetic_counts
from congestion_prediction.traffic_predictor import (
    METHOD_NAMES, clean_series, evaluate_forecast, predict_traffic,
)


def synthetic_flow(minutes=60):
    """Per-minute total flow from the SYNTHETIC generator."""
    return [sum(sum(c.values()) for c in rec["lanes"].values())
            for rec in generate_synthetic_counts(minutes)]


def test_sufficient_history_gives_forecasts():
    p = predict_traffic(synthetic_flow(60))
    assert p["kind"] == "forecast" and p["status"] == "ok"
    assert p["method"] in METHOD_NAMES and p["method_selection"]
    assert all(v["predicted_vehicle_flow"] is not None for v in p["horizons"].values())


def test_insufficient_history_is_handled():
    for series in ([], [30], [30, 40]):
        p = predict_traffic(series)
        assert p["status"] == "insufficient_data"
        for h in p["horizons"].values():
            assert h["predicted_vehicle_flow"] is None and h["predicted_level"] == "UNKNOWN"


def test_short_history_is_flagged_limited():
    p = predict_traffic([30, 32, 35, 31, 33])
    assert p["status"] == "limited_data" and p["method"] == "weighted_moving_average"


def test_missing_observations_interpolated_and_flagged():
    series = [30, 31, None, 33, 34, None, None, 36, 37, 38, 39, 40]
    p = predict_traffic(series)
    assert p["interpolated_points"] == 3 and p["status"] in ("ok", "limited_data")
    assert "interpolated" in p["message"]


def test_too_many_missing_refuses_to_forecast():
    p = predict_traffic([30, None, None, 31, None, None, 33, None, 34, None])
    assert p["status"] == "too_many_missing"
    assert p["horizons"]["next_5_minutes"]["predicted_vehicle_flow"] is None


def test_invalid_values_treated_as_missing():
    p = predict_traffic([30, 31, -5, float("nan"), "x", 33, 34, 35, 36, 37, 38, 39, 40, 41])
    assert p["data_cleaning"]["invalid_or_missing"] == 3


def test_long_gap_not_bridged():
    cleaned, flags, info = clean_series([1, 2, 3] + [None] * 8 + [10, 11, 12], max_gap=5)
    assert cleaned == [10, 11, 12] and info["dropped_before_long_gap"] == 3 and not any(flags)


def test_multiple_horizons_and_custom_horizons():
    p = predict_traffic(synthetic_flow(40), horizons=[5, 10, 15, 20])
    assert list(p["horizons"]) == ["next_5_minutes", "next_10_minutes",
                                   "next_15_minutes", "next_20_minutes"]
    assert [v["minutes"] for v in p["horizons"].values()] == [5, 10, 15, 20]


def test_output_structure():
    p = predict_traffic(synthetic_flow(30))
    for key in ("kind", "status", "message", "method", "interval_minutes",
                "history_points_used", "parameters", "horizons"):
        assert key in p
    for item in p["horizons"].values():
        assert set(item) == {"minutes", "predicted_vehicle_flow",
                             "predicted_vehicles_in_window", "predicted_level"}
        assert item["predicted_level"] in ("FREE", "MODERATE", "HEAVY")
        assert math.isclose(item["predicted_vehicles_in_window"],
                            item["predicted_vehicle_flow"] * item["minutes"], rel_tol=0.01)


def test_predictions_never_negative():
    falling = [60, 50, 40, 30, 20, 10, 5, 2, 1, 0, 0, 0]
    for method in METHOD_NAMES:
        p = predict_traffic(falling, method=method)
        for item in p["horizons"].values():
            assert item["predicted_vehicle_flow"] >= 0


def test_level_follows_prediction():
    heavy = predict_traffic([100] * 10)
    assert all(h["predicted_level"] == "HEAVY" for h in heavy["horizons"].values())
    quiet = predict_traffic([2] * 10)
    assert all(h["predicted_level"] == "FREE" for h in quiet["horizons"].values())


def test_constant_series_forecasts_constant():
    p = predict_traffic([25.0] * 15, method="weighted_moving_average")
    assert all(h["predicted_vehicle_flow"] == 25.0 for h in p["horizons"].values())


def test_larger_interval_uses_fewer_steps():
    p = predict_traffic([100] * 10, interval_minutes=5)
    assert p["horizons"]["next_5_minutes"]["predicted_vehicle_flow"] == 100


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        predict_traffic([1, 2, 3, 4], method="magic")
    with pytest.raises(ValueError):
        predict_traffic([1, 2, 3, 4], horizons=[0])
    with pytest.raises(ValueError):
        predict_traffic([1, 2, 3, 4], params={"ses_alpha": 5})
    with pytest.raises(ValueError):
        predict_traffic([1, 2, 3, 4], params={"bogus": 1})


def test_is_deterministic():
    s = synthetic_flow(60)
    assert predict_traffic(s) == predict_traffic(s)
    assert generate_synthetic_counts(10) == generate_synthetic_counts(10)


# ----------------------------- evaluation -------------------------------------
def test_evaluation_on_enough_data_reports_metrics():
    ev = evaluate_forecast(synthetic_flow(60))
    assert ev["status"] == "evaluated"
    assert ev["train_range"][1] + 1 == ev["test_range"][0]          # chronological, no overlap
    done = [h for h in ev["per_horizon"].values() if h["status"] == "evaluated"]
    assert done
    for h in done:
        for name, m in h["results"].items():
            assert m["MAE"] >= 0 and m["RMSE"] >= m["MAE"] - 1e-9
        assert "last_value" in h["results"]


def test_evaluation_with_too_little_data_is_not_evaluated():
    ev = evaluate_forecast([30, 31, 33, 35])
    assert ev["status"] == "not_evaluated" and "Not enough data" in ev["reason"]
    assert evaluate_forecast([])["status"] == "not_evaluated"


def test_evaluation_never_uses_future_data():
    """Series jumps exactly at the first test point; a forecaster that saw the
    future would have ~0 error, a correct one must be wrong about the jump."""
    series = [10.0] * 12 + [50.0] * 8
    ev = evaluate_forecast(series, horizons=[1], test_fraction=0.4)
    assert ev["per_horizon"]["next_1_minutes"]["results"]["last_value"]["MAE"] > 0
    assert ev["test_range"][0] == 12


def test_interpolated_targets_are_not_scored():
    series = [10.0] * 10 + [None, None] + [10.0] * 10
    ev = evaluate_forecast(series, horizons=[1])
    assert ev["status"] == "evaluated"
    assert ev["per_horizon"]["next_1_minutes"]["origins"] <= len(series) - 12 - 1
