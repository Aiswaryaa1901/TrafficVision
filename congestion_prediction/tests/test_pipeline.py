"""Tests for the Member 4 facing report. Data is SYNTHETIC unless noted."""

import json

import pytest

from congestion_prediction import build_traffic_report, config, save_report
from congestion_prediction.synthetic import DATA_LABEL, generate_synthetic_counts


def test_report_structure_for_dashboard():
    r = build_traffic_report(generate_synthetic_counts(60), data_label=DATA_LABEL,
                             start_time="2026-11-25T10:00:00")
    for key in ("schema_version", "status", "data_source", "classification", "history",
                "latest", "prediction", "evaluation", "data_quality", "disclaimer"):
        assert key in r
    assert r["data_source"]["label"] == "SYNTHETIC_DEMO"
    latest = r["latest"]
    assert set(latest["vehicle_flow"]) == {"cars", "buses", "trucks", "two_wheelers", "total"}
    assert latest["congestion"]["level"] in ("FREE", "MODERATE", "HEAVY")
    assert latest["timestamp"] == "2026-11-25T10:59:00"
    assert set(r["prediction"]["horizons"]) == {"next_5_minutes", "next_10_minutes",
                                                "next_15_minutes"}
    assert r["prediction"]["kind"] == "forecast" and len(r["history"]) == 60
    json.dumps(r)                                              # must be JSON-serialisable


def test_history_and_prediction_are_separate():
    r = build_traffic_report(generate_synthetic_counts(30))
    assert all("predicted_level" not in h for h in r["history"])
    assert r["prediction"]["based_on"]["last_history_interval_index"] == 29


def test_lane_levels_present():
    r = build_traffic_report(generate_synthetic_counts(10))
    lanes = r["latest"]["lanes"]
    assert set(lanes) == {"lane_1", "lane_2", "lane_3", "lane_4"}
    assert all(v["level"] in ("FREE", "MODERATE", "HEAVY") for v in lanes.values())


def test_real_sample_gives_honest_insufficient_forecast():
    """[real data] Member 2's 2-minute sample cannot support a forecast."""
    if not config.DEFAULT_INPUT_PATH.is_file():
        pytest.skip("Member 2 sample output not present")
    r = build_traffic_report()
    assert r["latest"]["congestion"]["level"] in ("FREE", "MODERATE", "HEAVY")
    assert r["prediction"]["status"] == "insufficient_data"
    assert r["evaluation"]["status"] == "not_evaluated"


@pytest.mark.parametrize("source", [[], None])
def test_empty_source_never_crashes(source, tmp_path):
    r = build_traffic_report(source if source is not None else tmp_path / "missing.json")
    assert r["status"] in ("no_data", "error")
    assert r["latest"] is None and r["prediction"]["status"] == "insufficient_data"


def test_broken_file_returns_error_report_and_strict_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    r = build_traffic_report(bad)
    assert r["status"] == "error" and r["error"]
    with pytest.raises(Exception):
        build_traffic_report(bad, strict=True)


def test_partial_final_minute_is_not_used_as_latest_or_for_forecast():
    data = generate_synthetic_counts(20)
    r = build_traffic_report(data, meta={"duration_seconds": 19 * 60 + 20})
    assert r["history"][-1]["complete"] is False
    assert r["latest"]["index"] == 18 and r["latest"]["is_partial"] is False
    assert r["prediction"]["history_points_used"] == 19


def test_save_report_writes_json_and_protects_member2_file(tmp_path):
    r = build_traffic_report(generate_synthetic_counts(15))
    out = save_report(r, tmp_path / "report.json")
    assert json.loads(out.read_text())["schema_version"] == config.SCHEMA_VERSION
    with pytest.raises(ValueError):
        save_report(r, config.DEFAULT_INPUT_PATH)
    with pytest.raises(ValueError):
        save_report(r, tmp_path / "vehicle_counts.json")


def test_threshold_override_changes_levels():
    data = generate_synthetic_counts(20)
    strict = build_traffic_report(data, thresholds={"free_max_per_lane": 1, "moderate_max_per_lane": 2})
    assert strict["latest"]["congestion"]["level"] == "HEAVY"
