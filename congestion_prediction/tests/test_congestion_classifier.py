"""Tests for congestion classification. All data below is SYNTHETIC / hand-made."""

import pytest

from congestion_prediction import config
from congestion_prediction.congestion_classifier import (
    InvalidTrafficDataError, classify_congestion, classify_flow, get_thresholds,
)

# 4 lanes by default -> FREE <= 40, MODERATE <= 80 vehicles per minute.


def test_free_traffic():
    assert classify_congestion({"cars": 10, "buses": 0, "trucks": 1, "two_wheelers": 4})["level"] == "FREE"


def test_moderate_traffic():
    assert classify_congestion({"cars": 50, "buses": 2, "trucks": 3, "two_wheelers": 5})["level"] == "MODERATE"


def test_heavy_traffic():
    assert classify_congestion({"cars": 80, "buses": 5, "trucks": 10, "two_wheelers": 10})["level"] == "HEAVY"


def test_empty_input_is_unknown():
    for empty in (None, {}):
        out = classify_congestion(empty)
        assert out["level"] == config.LEVEL_UNKNOWN and out["flow_per_minute"] is None
    assert classify_flow(None) == "UNKNOWN"


@pytest.mark.parametrize("bad", [
    {"cars": -1}, {"cars": "many"}, {"cars": float("nan")}, {"cars": float("inf")},
    {"cars": True}, {"unrelated": 5}, "text", [1, 2],
])
def test_invalid_input_raises(bad):
    with pytest.raises((InvalidTrafficDataError, TypeError)):
        classify_congestion(bad)


def test_invalid_flow_values():
    for bad in (-1, float("nan"), float("inf"), "5", True):
        with pytest.raises(InvalidTrafficDataError):
            classify_flow(bad)


def test_missing_categories_count_as_zero():
    a = classify_congestion({"cars": 30})
    b = classify_congestion({"cars": 30, "buses": 0, "trucks": 0, "two_wheelers": 0})
    assert a["flow_per_minute"] == b["flow_per_minute"] == 30


def test_total_key_is_ignored_not_double_counted():
    assert classify_congestion({"cars": 10, "total": 999})["flow_per_minute"] == 10


def test_different_categories_each_counted():
    for category in config.CATEGORIES:
        assert classify_congestion({category: 41})["level"] == "MODERATE"


def test_pcu_weighting_changes_level():
    counts = {"buses": 15}                       # 15 vehicles, 45 PCU
    assert classify_congestion(counts, use_pcu=False)["level"] == "FREE"
    assert classify_congestion(counts, use_pcu=True)["level"] == "MODERATE"


def test_boundaries_are_inclusive_upper_limits():
    limits = get_thresholds(4)
    assert limits["free_max"] == 40 and limits["moderate_max"] == 80
    assert classify_flow(40) == "FREE"
    assert classify_flow(40.01) == "MODERATE"
    assert classify_flow(80) == "MODERATE"
    assert classify_flow(80.01) == "HEAVY"
    assert classify_flow(0) == "FREE"


def test_lane_count_scales_thresholds():
    assert classify_flow(15, lanes=1) == "MODERATE"
    assert classify_flow(25, lanes=1) == "HEAVY"
    assert classify_flow(15, lanes=4) == "FREE"


def test_interval_length_converts_to_per_minute():
    assert classify_congestion({"cars": 100}, interval_minutes=5)["flow_per_minute"] == 20
    with pytest.raises(InvalidTrafficDataError):
        classify_congestion({"cars": 10}, interval_minutes=0)


def test_threshold_overrides_and_validation():
    assert classify_flow(15, lanes=1, overrides={"free_max_per_lane": 20, "moderate_max_per_lane": 30}) == "FREE"
    with pytest.raises(ValueError):
        get_thresholds(1, {"free_max_per_lane": 30, "moderate_max_per_lane": 10})
