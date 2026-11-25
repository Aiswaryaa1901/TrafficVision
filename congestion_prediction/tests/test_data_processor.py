"""Tests for data processing. Fixtures are SYNTHETIC (hand-written), except where
noted: the real Member 2 sample file is read read-only."""

import csv
import json

import pytest

from congestion_prediction import config
from congestion_prediction.data_processor import DataFormatError, process_traffic_data


def lane(c=0, b=0, t=0, w=0):
    return {"cars": c, "buses": b, "trucks": t, "two_wheelers": w}


def member2(minute, **lanes):
    return {"minute": minute, "lanes": lanes}


def write_json(tmp_path, data, name="counts.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_real_member2_sample_file_is_read_and_not_modified():
    """[real data] Member 2's actual vehicle_counts.json."""
    path = config.DEFAULT_INPUT_PATH
    if not path.is_file():
        pytest.skip("Member 2 sample output not present")
    before = path.read_bytes()
    result = process_traffic_data(path)
    assert path.read_bytes() == before
    assert len(result.intervals) == 2
    assert result.intervals[0]["counts"]["total"] == 68      # 28 + 10 + 18 + 12
    assert result.intervals[1]["counts"]["total"] == 33
    assert result.lane_names == ["lane_1", "lane_2", "lane_3", "lane_4"]


def test_valid_json_input(tmp_path):
    path = write_json(tmp_path, [member2(0, lane_1=lane(c=5, t=1), lane_2=lane(w=2)),
                                 member2(1, lane_1=lane(c=3), lane_2=lane(b=1))])
    r = process_traffic_data(path)
    assert [i["counts"]["total"] for i in r.intervals] == [8, 4]
    assert r.intervals[0]["counts"]["two_wheelers"] == 2
    assert r.intervals[0]["flow_per_minute"] == 8
    assert r.quality["missing_intervals"] == 0


def test_valid_csv_input(tmp_path):
    path = tmp_path / "counts.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["minute", "lane", "car", "bus", "truck", "motorcycle"])
        writer.writerow([0, "lane_1", 5, 0, 1, 2])
        writer.writerow([0, "lane_2", 4, 1, 0, 0])
        writer.writerow([1, "lane_1", 6, 0, 0, 1])
        writer.writerow([1, "lane_2", 2, 0, 0, 0])
    r = process_traffic_data(path)
    assert [i["counts"]["total"] for i in r.intervals] == [13, 9]
    assert set(r.intervals[0]["lanes"]) == {"lane_1", "lane_2"}


def test_flat_records_and_wrapper(tmp_path):
    data = {"records": [{"minute": 0, "car": 3, "motorbike": 2}, {"minute": 1, "cars": 4}]}
    r = process_traffic_data(write_json(tmp_path, data))
    assert [i["counts"]["total"] for i in r.intervals] == [5, 4]
    assert r.quality["missing_category_entries"] == 2


def test_missing_timestamps_are_skipped_and_reported():
    r = process_traffic_data([{"lanes": {"lane_1": lane(c=5)}}, member2(0, lane_1=lane(c=2))])
    assert len(r.intervals) == 1 and r.intervals[0]["counts"]["total"] == 2
    assert r.quality["entries_rejected"] == 1
    assert any("no minute" in w for w in r.warnings)


def test_missing_interval_in_series_is_not_invented():
    r = process_traffic_data([member2(0, a=lane(c=5)), member2(3, a=lane(c=7))])
    assert [i["observed"] for i in r.intervals] == [True, False, False, True]
    assert r.intervals[1]["counts"] is None and r.intervals[1]["flow_per_minute"] is None
    assert r.quality["missing_intervals"] == 2


def test_missing_interval_policy_zero():
    r = process_traffic_data([member2(0, a=lane(c=5)), member2(2, a=lane(c=7))],
                             missing_policy="zero")
    assert r.intervals[1]["counts"]["total"] == 0 and r.intervals[1]["observed"]


def test_duplicate_timestamps_last_wins_by_default():
    r = process_traffic_data([member2(0, a=lane(c=5)), member2(0, a=lane(c=9))])
    assert len(r.intervals) == 1 and r.intervals[0]["counts"]["total"] == 9
    assert r.quality["duplicate_entries"] == 1


def test_duplicate_timestamps_sum_policy():
    r = process_traffic_data([member2(0, a=lane(c=5)), member2(0, a=lane(c=9))],
                             duplicate_policy="sum")
    assert r.intervals[0]["counts"]["total"] == 14


def test_cumulative_counts_become_differences():
    data = [member2(0, a=lane(c=10)), member2(1, a=lane(c=25)), member2(2, a=lane(c=30))]
    r = process_traffic_data(data, cumulative=True)
    assert [i["counts"]["total"] for i in r.intervals] == [10, 15, 5]


def test_cumulative_counter_reset_is_handled():
    data = [member2(0, a=lane(c=10)), member2(1, a=lane(c=30)), member2(2, a=lane(c=4))]
    r = process_traffic_data(data, cumulative=True)
    assert [i["counts"]["total"] for i in r.intervals] == [10, 20, 4]    # never negative
    assert r.quality["counter_resets"] == 1


def test_cumulative_gap_is_spread_and_flagged():
    data = [member2(0, a=lane(c=10)), member2(3, a=lane(c=40))]
    r = process_traffic_data(data, cumulative=True)
    assert [i["counts"]["total"] for i in r.intervals] == [10, 10, 10, 10]
    assert r.quality["gap_spread_events"] == 1


@pytest.mark.parametrize("bad", [-3, "abc", None, float("nan"), float("inf"), 2.5, True, [1]])
def test_invalid_numbers_reject_that_lane_only(bad):
    r = process_traffic_data([member2(0, good=lane(c=5), broken={"cars": bad, "buses": 0})])
    assert r.intervals[0]["counts"]["total"] == 5            # nothing guessed for 'broken'
    assert r.quality["invalid_value_entries"] == 1


def test_lane_dropped_after_invalid_value_marks_interval_partial():
    r = process_traffic_data([member2(0, a=lane(c=5), b=lane(c=5)),
                              member2(1, a=lane(c=5), b={"cars": "x"})])
    assert r.intervals[0]["complete"] is True
    assert r.intervals[1]["complete"] is False and r.intervals[1]["lanes_missing"] == ["b"]
    assert r.intervals[1]["usable_for_forecast"] is False


def test_float_counts_that_are_whole_numbers_are_accepted():
    r = process_traffic_data([member2(0, a={"cars": 5.0})])
    assert r.intervals[0]["counts"]["cars"] == 5


def test_empty_and_garbage_input():
    for empty in ([], {}, [None, 5, "x"]):
        r = process_traffic_data(empty)
        assert r.intervals == [] and r.quality["status"] == "no_data"


def test_unreadable_files_raise_clear_error(tmp_path):
    with pytest.raises(DataFormatError):
        process_traffic_data(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataFormatError):
        process_traffic_data(bad)


def test_interval_aggregation_and_partial_last_interval():
    data = [member2(m, a=lane(c=10)) for m in range(7)]      # 7 minutes
    r = process_traffic_data(data, interval_minutes=5)
    assert len(r.intervals) == 2
    assert r.intervals[0]["counts"]["total"] == 50 and r.intervals[0]["flow_per_minute"] == 10
    assert r.intervals[1]["complete"] is False               # only 2 of 5 minutes
    assert r.intervals[1]["flow_per_minute"] == 10           # rate over the minutes seen


def test_metadata_marks_partial_final_minute():
    data = [member2(0, a=lane(c=10)), member2(1, a=lane(c=10)), member2(2, a=lane(c=3))]
    r = process_traffic_data(data, meta={"duration_seconds": 130})
    assert [i["complete"] for i in r.intervals] == [True, True, False]
    assert r.quality["last_interval_completeness"] == "partial"
    r2 = process_traffic_data(data, meta={"duration_seconds": 180})
    assert r2.intervals[-1]["complete"] is True
    assert r2.quality["last_interval_completeness"] == "confirmed"


def test_metadata_sidecar_file_is_picked_up(tmp_path):
    path = write_json(tmp_path, [member2(0, a=lane(c=1)), member2(1, a=lane(c=1))])
    (tmp_path / config.META_FILENAME).write_text(json.dumps({"duration_seconds": 90}))
    r = process_traffic_data(path)
    assert r.intervals[-1]["complete"] is False


def test_start_time_gives_iso_timestamps():
    r = process_traffic_data([member2(0, a=lane(c=1)), member2(1, a=lane(c=1))],
                             start_time="2026-11-25T10:00:00")
    assert r.intervals[1]["timestamp"] == "2026-11-25T10:01:00"
    assert r.intervals[1]["video_time"] == "01:00"


def test_iso_timestamps_in_input_are_converted():
    data = [{"timestamp": "2026-11-25T10:00:30", "cars": 4},
            {"timestamp": "2026-11-25T10:02:10", "cars": 6}]
    r = process_traffic_data(data)
    assert [i["start_minute"] for i in r.intervals] == [0, 1, 2]
    assert r.intervals[0]["timestamp"] == "2026-11-25T10:00:00"


def test_input_object_is_not_modified():
    data = [member2(0, a=lane(c=5))]
    snapshot = json.dumps(data)
    process_traffic_data(data)
    assert json.dumps(data) == snapshot
