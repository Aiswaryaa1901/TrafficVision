"""Read Member 2's vehicle counts and turn them into clean per-interval data.

Member 2's actual output (vehicle_counting/vehicle_counts.json) is a list of
*interval* counts - vehicles that crossed the counting line during each video
minute, per lane and category - NOT cumulative totals:

    [{"minute": 0, "lanes": {"lane_1": {"cars": 19, "buses": 0,
                                         "trucks": 9, "two_wheelers": 0}, ...}}, ...]

"minute" is the minute index from the start of the video (there is no wall-clock
time), so real timestamps are only produced if the caller supplies start_time.

Besides that format this module also accepts flat JSON records, an optional
{"records": [...]} wrapper and CSV files, and (with cumulative=True) cumulative
line-crossing counters, which are converted to per-interval differences.

Nothing here invents data: unusable values are rejected and reported in
``ProcessedTraffic.quality`` / ``.warnings``. The input file is only read.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from . import config

Source = Union[str, Path, List[Any], Dict[str, Any], None]

MAX_WARNINGS = 200


class DataFormatError(ValueError):
    """The input cannot be read at all (missing file, broken JSON, wrong type)."""


@dataclass
class ProcessedTraffic:
    """Clean, structured output of :func:`process_traffic_data`."""

    intervals: List[Dict[str, Any]]
    quality: Dict[str, Any]
    warnings: List[str]
    interval_minutes: int
    start_time: Optional[datetime]
    source: str
    lane_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "interval_minutes": self.interval_minutes,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "intervals": self.intervals,
            "quality": self.quality,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
class _Log:
    """Collects warnings (capped) and quality counters."""

    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.suppressed = 0
        self.counters: Dict[str, int] = {
            "records_read": 0, "entries_accepted": 0, "entries_rejected": 0,
            "invalid_value_entries": 0, "duplicate_entries": 0,
            "counter_resets": 0, "gap_spread_events": 0,
            "missing_category_entries": 0,
        }

    def warn(self, message: str) -> None:
        if len(self.warnings) < MAX_WARNINGS:
            self.warnings.append(message)
        else:
            self.suppressed += 1

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] += n


def to_count(value: Any) -> int:
    """Convert one raw count to a non-negative int, or raise ValueError."""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"not a count: {value!r}")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            raise ValueError("empty count")
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"not a number: {value!r}") from None
    if not isinstance(value, (int, float)):
        raise ValueError(f"not a number: {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"not finite: {value!r}")
        if value != int(value):
            raise ValueError(f"not a whole number: {value!r}")
        value = int(value)
    if value < 0:
        raise ValueError(f"negative count: {value}")
    return int(value)


def normalize_category(name: Any) -> Optional[str]:
    """Map any accepted spelling (car, motorbike, ...) to the canonical name."""
    if not isinstance(name, str):
        return None
    return config.CATEGORY_ALIASES.get(name.strip().lower().replace(" ", "_"))


def _extract_counts(mapping: Dict[str, Any]) -> Dict[str, int]:
    """Pull category counts out of a dict; raises ValueError on a bad value.

    Categories that are absent are simply not returned (caller fills 0 and
    records it as a missing-category entry).
    """
    counts: Dict[str, int] = {}
    for key, raw in mapping.items():
        category = normalize_category(key)
        if category is None:
            continue
        counts[category] = counts.get(category, 0) + to_count(raw)
    return counts


def _parse_time(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[datetime]]:
    """Return (minute_offset, absolute_datetime). Exactly one is not None."""
    if record.get("minute") is not None:
        minute = to_count(record["minute"])
        return float(minute), None
    for key in ("time_seconds", "seconds"):
        if record.get(key) is not None:
            return to_count(record[key]) / 60.0, None
    if record.get("timestamp") is not None:
        raw = record["timestamp"]
        if isinstance(raw, datetime):
            return None, raw
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if not math.isfinite(raw) or raw < 0:
                raise ValueError(f"bad timestamp: {raw!r}")
            return float(raw) / 60.0, None     # seconds since video start
        if isinstance(raw, str):
            text = raw.strip()
            try:
                return None, datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                try:
                    return float(text) / 60.0, None
                except ValueError:
                    raise ValueError(f"unreadable timestamp: {raw!r}") from None
    raise ValueError("record has no minute / timestamp")


# --------------------------------------------------------------------------- #
# Reading raw records
# --------------------------------------------------------------------------- #
def _read_source(source: Source, file_format: Optional[str]) -> Tuple[List[Any], str]:
    """Return (list of raw records, source label)."""
    if source is None:
        source = config.DEFAULT_INPUT_PATH
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise DataFormatError(f"input file not found: {path}")
        fmt = (file_format or path.suffix.lstrip(".")).lower()
        try:
            if fmt == "csv":
                with open(path, newline="", encoding="utf-8-sig") as handle:
                    return list(csv.DictReader(handle)), path.name
            with open(path, encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
            raise DataFormatError(f"cannot read {path.name}: {exc}") from exc
        return _unwrap(data), path.name
    return _unwrap(source), "in-memory"


def _unwrap(data: Any) -> List[Any]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("records", "minutes", "intervals", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        if any(normalize_category(k) for k in data) or "lanes" in data:
            return [data]                      # a single record
        return []
    raise DataFormatError(f"unsupported input type: {type(data).__name__}")


def _load_meta(source: Source, meta: Optional[Dict[str, Any]], log: _Log) -> Dict[str, Any]:
    """Optional sidecar from the updated video_tracker.py (video duration)."""
    reads_a_file = source is None or isinstance(source, (str, Path))
    if meta is None and reads_a_file:
        base = Path(source) if source is not None else config.DEFAULT_INPUT_PATH
        candidate = base.with_name(config.META_FILENAME)
        if candidate.is_file():
            try:
                with open(candidate, encoding="utf-8-sig") as handle:
                    meta = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                log.warn(f"ignored unreadable {candidate.name}: {exc}")
                return {}
    if not isinstance(meta, dict):
        return {}
    duration = meta.get("duration_seconds")
    if duration is not None:
        try:
            duration = float(duration)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError
        except (TypeError, ValueError):
            log.warn("ignored invalid duration_seconds in metadata")
            meta = {k: v for k, v in meta.items() if k != "duration_seconds"}
    return meta


# --------------------------------------------------------------------------- #
# Records -> entries  (one entry = one lane in one minute)
# --------------------------------------------------------------------------- #
Entry = Tuple[float, str, Dict[str, int]]    # (minute_offset, lane, counts)


def _records_to_entries(records: Iterable[Any], log: _Log,
                        start_time: Optional[datetime]) -> Tuple[List[Entry], Optional[datetime]]:
    parsed: List[Tuple[Optional[float], Optional[datetime], str, Dict[str, Any]]] = []
    for index, record in enumerate(records):
        log.count("records_read")
        if not isinstance(record, dict):
            log.count("entries_rejected")
            log.warn(f"record {index}: not an object, skipped")
            continue
        try:
            minute, absolute = _parse_time(record)
        except ValueError as exc:
            log.count("entries_rejected")
            log.warn(f"record {index}: {exc}; skipped")
            continue

        lanes = record.get("lanes")
        if isinstance(lanes, dict):
            for lane_name, lane_counts in lanes.items():
                if isinstance(lane_counts, dict):
                    parsed.append((minute, absolute, str(lane_name), lane_counts))
                else:
                    log.count("entries_rejected")
                    log.warn(f"record {index} {lane_name}: not an object, skipped")
        elif "lanes" in record:
            log.count("entries_rejected")
            log.warn(f"record {index}: 'lanes' is not an object, skipped")
        else:
            lane = record.get("lane")
            lane_name = str(lane).strip() if lane not in (None, "") else "all"
            parsed.append((minute, absolute, lane_name, record))

    # Resolve absolute (ISO) timestamps to minute offsets from one reference.
    absolutes = [a for _, a, _, _ in parsed if a is not None]
    reference = start_time
    if absolutes:
        naive = [a.replace(tzinfo=None) if a.tzinfo else a for a in absolutes]
        if reference is None:
            first = min(naive)
            reference = first.replace(second=0, microsecond=0)

    entries: List[Entry] = []
    for minute, absolute, lane_name, counts_map in parsed:
        if absolute is not None:
            stamp = absolute.replace(tzinfo=None) if absolute.tzinfo else absolute
            offset = (stamp - reference).total_seconds() / 60.0
            if offset < 0:
                log.count("entries_rejected")
                log.warn(f"{lane_name}: timestamp before start_time, skipped")
                continue
            minute = float(math.floor(offset))
        try:
            counts = _extract_counts(counts_map)
        except ValueError as exc:
            log.count("entries_rejected")
            log.count("invalid_value_entries")
            log.warn(f"minute {int(minute)} {lane_name}: {exc}; entry skipped")
            continue
        if not counts:
            log.count("entries_rejected")
            log.warn(f"minute {int(minute)} {lane_name}: no vehicle categories; entry skipped")
            continue
        missing = [c for c in config.CATEGORIES if c not in counts]
        if missing:
            log.count("missing_category_entries")
        full = {c: counts.get(c, 0) for c in config.CATEGORIES}   # absent = 0
        log.count("entries_accepted")
        entries.append((float(int(minute)), lane_name, full))
    return entries, reference if absolutes else start_time


def _cumulative_to_interval(entries: List[Entry], baseline: int, log: _Log) -> List[Entry]:
    """Turn cumulative counter readings into per-minute counts (per lane)."""
    # Two readings for the same lane and minute: keep the last one.
    latest: Dict[Tuple[str, float], Entry] = {}
    for entry in entries:
        key = (entry[1], entry[0])
        if key in latest:
            log.count("duplicate_entries")
            log.warn(f"duplicate cumulative reading for minute {int(entry[0])} "
                     f"{entry[1]}; keeping the last")
        latest[key] = entry
    by_lane: Dict[str, List[Entry]] = {}
    for entry in latest.values():
        by_lane.setdefault(entry[1], []).append(entry)

    out: List[Entry] = []
    for lane, lane_entries in by_lane.items():
        lane_entries.sort(key=lambda e: e[0])
        previous = {c: baseline for c in config.CATEGORIES}
        previous_minute = lane_entries[0][0] - 1
        for minute, _, reading in lane_entries:
            delta = {}
            for category in config.CATEGORIES:
                diff = reading[category] - previous[category]
                if diff < 0:                       # counter restarted
                    log.count("counter_resets")
                    log.warn(f"{lane} minute {int(minute)}: {category} counter reset "
                             f"({previous[category]} -> {reading[category]}); "
                             f"using the new reading as the count since the reset")
                    diff = reading[category]
                delta[category] = diff
            span = int(minute - previous_minute)
            if span > 1:                           # readings were skipped
                log.count("gap_spread_events")
                log.warn(f"{lane}: no reading for {span - 1} minute(s) before minute "
                         f"{int(minute)}; difference spread evenly over the gap")
            for step in range(span):
                piece = {}
                for category, total in delta.items():
                    base, rest = divmod(total, span)
                    piece[category] = base + (rest if step == span - 1 else 0)
                out.append((previous_minute + 1 + step, lane, piece))
            previous = dict(reading)
            previous_minute = minute
    return out


def _merge_duplicates(entries: List[Entry], policy: str, log: _Log) -> Dict[int, Dict[str, Dict[str, int]]]:
    """Group entries into {minute: {lane: counts}}, resolving duplicates."""
    grouped: Dict[int, Dict[str, Dict[str, int]]] = {}
    for minute, lane, counts in entries:
        slot = grouped.setdefault(int(minute), {})
        if lane in slot:
            log.count("duplicate_entries")
            log.warn(f"duplicate entry for minute {int(minute)} {lane}; policy={policy}")
            if policy == "sum":
                slot[lane] = {c: slot[lane][c] + counts[c] for c in config.CATEGORIES}
                continue
        slot[lane] = dict(counts)
    return grouped


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #
def _video_time(minutes: float) -> str:
    seconds = int(round(minutes * 60))
    hours, rest = divmod(seconds, 3600)
    mins, secs = divmod(rest, 60)
    return f"{hours:d}:{mins:02d}:{secs:02d}" if hours else f"{mins:02d}:{secs:02d}"


def _sum_counts(parts: Iterable[Dict[str, int]]) -> Dict[str, int]:
    total = {c: 0 for c in config.CATEGORIES}
    for part in parts:
        for category in config.CATEGORIES:
            total[category] += part[category]
    total["total"] = sum(total[c] for c in config.CATEGORIES)
    return total


def process_traffic_data(
    source: Source = None,
    *,
    interval_minutes: Optional[int] = None,
    cumulative: bool = False,
    start_time: Union[None, str, datetime] = None,
    meta: Optional[Dict[str, Any]] = None,
    missing_policy: Optional[str] = None,
    duplicate_policy: Optional[str] = None,
    initial_baseline: Optional[int] = None,
    file_format: Optional[str] = None,
) -> ProcessedTraffic:
    """Read Member 2 counts and return clean per-interval traffic data.

    source          path to vehicle_counts.json / .csv, a list of records, or None
                    (= Member 2's default output file).
    interval_minutes  analysis interval (default config.INTERVAL_MINUTES).
    cumulative      True if the counts are running totals (differences are taken;
                    counter resets are handled). Member 2's file is NOT cumulative.
    start_time      ISO string/datetime of video minute 0. Without it the
                    ``timestamp`` field is None and only ``video_time`` is given.
    meta            optional dict like vehicle_counts_meta.json (duration_seconds).
    missing_policy  "missing" (default) or "zero" - see config.
    duplicate_policy "last" (default) or "sum".

    Raises DataFormatError only if the input cannot be read at all. Empty input
    returns an empty result with quality["status"] == "no_data".
    """
    interval = int(interval_minutes or config.INTERVAL_MINUTES)
    if interval < 1:
        raise ValueError("interval_minutes must be >= 1")
    missing_policy = missing_policy or config.MISSING_INTERVAL_POLICY
    duplicate_policy = duplicate_policy or config.DUPLICATE_POLICY
    if missing_policy not in ("missing", "zero"):
        raise ValueError("missing_policy must be 'missing' or 'zero'")
    if duplicate_policy not in ("last", "sum"):
        raise ValueError("duplicate_policy must be 'last' or 'sum'")
    baseline = config.CUMULATIVE_INITIAL_BASELINE if initial_baseline is None else int(initial_baseline)

    log = _Log()
    if isinstance(start_time, str):
        try:
            start_time = datetime.fromisoformat(start_time.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"start_time is not an ISO date-time: {exc}") from exc
    if isinstance(start_time, datetime) and start_time.tzinfo:
        start_time = start_time.replace(tzinfo=None)

    records, label = _read_source(source, file_format)
    meta_info = _load_meta(source, meta, log)

    entries, reference = _records_to_entries(records, log, start_time)
    if cumulative:
        entries = _cumulative_to_interval(entries, baseline, log)
    minutes_map = _merge_duplicates(entries, "last" if cumulative else duplicate_policy, log)

    quality: Dict[str, Any] = dict(log.counters)
    quality["assumptions"] = {
        "counts_are_cumulative": cumulative,
        "missing_interval_policy": missing_policy,
        "duplicate_policy": duplicate_policy,
        "absent_category_means_zero": True,
    }
    quality["lanes_seen"] = []

    if not minutes_map:
        quality.update(status="no_data", interval_count=0, observed_intervals=0,
                       missing_intervals=0, partial_intervals=0,
                       last_interval_completeness="unknown")
        return ProcessedTraffic([], quality, _final_warnings(log, "no usable traffic records"),
                                interval, reference, label, [])

    lane_names = sorted({lane for lanes in minutes_map.values() for lane in lanes})
    real_lanes = [lane for lane in lane_names if lane != "all"]

    first_minute, last_minute = min(minutes_map), max(minutes_map)
    duration = meta_info.get("duration_seconds")
    if duration is not None:
        last_minute = max(last_minute, int(max(duration - 1e-9, 0.0) // 60))

    first_bin, last_bin = first_minute // interval, last_minute // interval
    intervals: List[Dict[str, Any]] = []
    for bin_index in range(first_bin, last_bin + 1):
        start_min = bin_index * interval
        observed_minutes = 0
        lane_parts: Dict[str, List[Dict[str, int]]] = {}
        lane_minutes: Dict[str, int] = {}
        for minute in range(start_min, start_min + interval):
            if minute in minutes_map:
                observed_minutes += 1
                for lane, counts in minutes_map[minute].items():
                    lane_parts.setdefault(lane, []).append(counts)
                    lane_minutes[lane] = lane_minutes.get(lane, 0) + 1
            elif missing_policy == "zero" and first_minute <= minute <= last_minute:
                observed_minutes += 1
                for lane in lane_names:
                    zero = {c: 0 for c in config.CATEGORIES}
                    lane_parts.setdefault(lane, []).append(zero)
                    lane_minutes[lane] = lane_minutes.get(lane, 0) + 1

        record: Dict[str, Any] = {
            "index": bin_index - first_bin,
            "start_minute": start_min,
            "end_minute": start_min + interval,
            "video_time": _video_time(start_min),
            "timestamp": (reference + timedelta(minutes=start_min)).isoformat() if reference else None,
            "observed": observed_minutes > 0,
            "observed_minutes": observed_minutes,
        }

        if observed_minutes == 0:
            record.update(complete=False, lanes_missing=[], counts=None,
                          flow_per_minute=None, lanes={}, usable_for_forecast=False)
            intervals.append(record)
            continue

        lanes_out = {}
        for lane in sorted(lane_parts):
            lanes_out[lane] = _sum_counts(lane_parts[lane])
        counts = _sum_counts(lanes_out.values()) if lanes_out else None
        lanes_missing = [lane for lane in real_lanes
                         if lane_minutes.get(lane, 0) < observed_minutes]
        complete = observed_minutes == interval and not lanes_missing
        if duration is not None and duration < (bin_index + 1) * interval * 60 - 1.0:
            complete = False                   # video ended inside this interval
        record.update(
            complete=complete,
            lanes_missing=lanes_missing,
            counts=counts,
            flow_per_minute=round(counts["total"] / observed_minutes, 4),
            lanes=lanes_out,
            usable_for_forecast=complete,
        )
        intervals.append(record)

    observed_count = sum(1 for r in intervals if r["observed"])
    missing_count = len(intervals) - observed_count
    partial_count = sum(1 for r in intervals if r["observed"] and not r["complete"])
    if duration is not None:
        completeness = "confirmed" if intervals[-1]["complete"] else "partial"
    else:
        completeness = "partial" if (intervals[-1]["observed"] and not intervals[-1]["complete"]) else "unknown"

    if missing_count:
        log.warn(f"{missing_count} interval(s) have no data (policy: {missing_policy})")
    if completeness == "unknown":
        log.warn("last interval may be partial: video duration unknown "
                 "(vehicle_counts_meta.json not found)")

    quality.update(
        lanes_seen=real_lanes or lane_names,
        interval_count=len(intervals),
        observed_intervals=observed_count,
        missing_intervals=missing_count,
        partial_intervals=partial_count,
        last_interval_completeness=completeness,
        video_duration_seconds=duration,
    )
    quality["status"] = "ok" if not (log.warnings or log.suppressed) else "warnings"
    return ProcessedTraffic(intervals, quality, _final_warnings(log), interval,
                            reference, label, real_lanes or lane_names)


def _final_warnings(log: _Log, extra: Optional[str] = None) -> List[str]:
    warnings = list(log.warnings)
    if extra:
        warnings.append(extra)
    if log.suppressed:
        warnings.append(f"... and {log.suppressed} more warning(s) not shown")
    return warnings
