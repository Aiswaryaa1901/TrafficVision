"""Flow-based congestion classification (FREE / MODERATE / HEAVY).

IMPORTANT - what this is and is not
-----------------------------------
Member 2 counts vehicles crossing a line. From that we can derive traffic FLOW
(vehicles per minute). We do not have speed, road occupancy or road geometry, so
the result is an *estimated traffic-flow-based congestion classification*, not a
measurement of road density. A fully jammed road can show LOW flow, so a FREE
label means "low flow", not necessarily "smooth traffic".

Rule (thresholds in config.py, vehicles per minute per lane, upper limits inclusive):
    flow <= FREE_MAX_PER_LANE * lanes      -> FREE
    flow <= MODERATE_MAX_PER_LANE * lanes  -> MODERATE
    otherwise                              -> HEAVY
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Union

from . import config


class InvalidTrafficDataError(ValueError):
    """A flow value or count is negative, non-numeric, NaN or infinite."""


def get_thresholds(lanes: Optional[int] = None,
                   overrides: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    """Absolute flow limits (vehicles/minute) for ``lanes`` lanes."""
    per_lane_free = config.FREE_MAX_PER_LANE
    per_lane_moderate = config.MODERATE_MAX_PER_LANE
    if overrides:
        per_lane_free = overrides.get("free_max_per_lane", per_lane_free)
        per_lane_moderate = overrides.get("moderate_max_per_lane", per_lane_moderate)
    if per_lane_free < 0 or per_lane_moderate < per_lane_free:
        raise ValueError("thresholds must satisfy 0 <= free_max <= moderate_max")
    lane_count = lanes if lanes and lanes > 0 else config.DEFAULT_LANE_COUNT
    return {
        "lanes": lane_count,
        "free_max": per_lane_free * lane_count,
        "moderate_max": per_lane_moderate * lane_count,
    }


def _valid_flow(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidTrafficDataError(f"flow must be a number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise InvalidTrafficDataError(f"flow must be finite and >= 0, got {value!r}")
    return float(value)


def classify_flow(flow_per_minute: Optional[float], lanes: Optional[int] = None,
                  overrides: Optional[Mapping[str, float]] = None) -> str:
    """Classify a flow in vehicles per minute.

    None means "no data" and returns UNKNOWN. Negative, NaN, infinite or
    non-numeric values raise InvalidTrafficDataError.
    """
    if flow_per_minute is None:
        return config.LEVEL_UNKNOWN
    flow = _valid_flow(flow_per_minute)
    limits = get_thresholds(lanes, overrides)
    if flow <= limits["free_max"]:
        return config.LEVEL_FREE
    if flow <= limits["moderate_max"]:
        return config.LEVEL_MODERATE
    return config.LEVEL_HEAVY


def effective_flow(counts: Mapping[str, Any], interval_minutes: float = 1.0,
                   use_pcu: Optional[bool] = None) -> float:
    """Vehicles per minute from category counts (optionally PCU-weighted)."""
    if interval_minutes is None or interval_minutes <= 0:
        raise InvalidTrafficDataError("interval_minutes must be > 0")
    use_pcu = config.USE_PCU if use_pcu is None else use_pcu
    total = 0.0
    for category in config.CATEGORIES:
        raw = counts.get(category, 0)
        if raw is None:
            raw = 0
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
                or not math.isfinite(raw) or raw < 0:
            raise InvalidTrafficDataError(f"invalid {category} count: {raw!r}")
        total += raw * (config.PCU_WEIGHTS[category] if use_pcu else 1.0)
    return total / interval_minutes


def classify_congestion(data: Union[None, int, float, Mapping[str, Any]],
                        interval_minutes: float = 1.0,
                        lanes: Optional[int] = None,
                        use_pcu: Optional[bool] = None,
                        overrides: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """Classify one interval.

    ``data`` is either a number (vehicles in the interval) or a dict of category
    counts such as {"cars": 20, "buses": 1, "trucks": 3, "two_wheelers": 5}
    (a "total" key is ignored; missing categories count as 0). Empty / None
    input returns level UNKNOWN. Invalid values raise InvalidTrafficDataError.

    Returns {"level", "flow_per_minute", "basis", "thresholds"}.
    """
    if data is None or (isinstance(data, Mapping) and not data):
        return {"level": config.LEVEL_UNKNOWN, "flow_per_minute": None,
                "basis": "no data", "thresholds": get_thresholds(lanes, overrides)}
    if isinstance(data, Mapping):
        if not any(k in data for k in config.CATEGORIES):
            raise InvalidTrafficDataError(
                f"no vehicle categories in input (expected some of {config.CATEGORIES})")
        flow = effective_flow(data, interval_minutes, use_pcu)
    else:
        if interval_minutes is None or interval_minutes <= 0:
            raise InvalidTrafficDataError("interval_minutes must be > 0")
        flow = _valid_flow(data) / interval_minutes
    pcu = config.USE_PCU if use_pcu is None else use_pcu
    return {
        "level": classify_flow(flow, lanes, overrides),
        "flow_per_minute": round(flow, 4),
        "basis": "estimated flow (vehicles/minute" + (", PCU-weighted" if pcu else "") + ")",
        "thresholds": get_thresholds(lanes, overrides),
    }
