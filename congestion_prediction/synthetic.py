"""SYNTHETIC demo/test data in Member 2's vehicle_counts.json format.

*** This is NOT real traffic. *** It exists only so the forecasting code and the
tests have a long-enough series (Member 2's real sample has just 2 minutes).
Never present results from it as real detections or real forecast accuracy.
Deterministic: a fixed seed (config.SYNTHETIC_SEED) is used.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from . import config

DATA_LABEL = "SYNTHETIC_DEMO"


def generate_synthetic_counts(minutes: int = 60, lanes: int = 4,
                              seed: Optional[int] = None,
                              base_per_lane: float = 8.0,
                              swing_per_lane: float = 6.0) -> List[Dict[str, Any]]:
    """Per-minute, per-lane counts with a slow rise/fall plus noise (SYNTHETIC)."""
    rng = random.Random(config.SYNTHETIC_SEED if seed is None else seed)
    records = []
    for minute in range(minutes):
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * minute / max(minutes, 2) - math.pi / 2)
        lane_data = {}
        for lane in range(1, lanes + 1):
            expected = base_per_lane + swing_per_lane * wave
            total = max(0, int(round(rng.gauss(expected, 1.5))))
            buses = 1 if rng.random() < 0.05 else 0
            trucks = int(total * 0.1)
            bikes = int(total * 0.15)
            cars = max(0, total - buses - trucks - bikes)
            lane_data[f"lane_{lane}"] = {"cars": cars, "buses": buses,
                                         "trucks": trucks, "two_wheelers": bikes}
        records.append({"minute": minute, "lanes": lane_data})
    return records
