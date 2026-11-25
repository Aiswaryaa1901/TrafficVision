"""All tunable settings for the Member 3 module (congestion + prediction).

Nothing in the other files is hard-coded: change values here, or pass an
override dict/argument to the public functions.
"""

from __future__ import annotations

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

# --- Input / output locations (derived from the project structure) -----------
# Member 2's output. It is only ever READ; this module never writes to it.
DEFAULT_INPUT_PATH = PROJECT_ROOT / "vehicle_counting" / "vehicle_counts.json"
# Optional sidecar written by the updated video_tracker.py (video duration).
META_FILENAME = "vehicle_counts_meta.json"
OUTPUT_DIR = MODULE_DIR / "outputs"          # git-ignored
DEFAULT_REPORT_PATH = OUTPUT_DIR / "traffic_report.json"

SCHEMA_VERSION = "1.0"

# --- Vehicle categories --------------------------------------------------------
# Canonical names are the ones Member 2 already uses.
CATEGORIES = ("cars", "buses", "trucks", "two_wheelers")

# Accepted spellings on input -> canonical name (case-insensitive).
CATEGORY_ALIASES = {
    "car": "cars", "cars": "cars",
    "bus": "buses", "buses": "buses",
    "truck": "trucks", "trucks": "trucks",
    "motorcycle": "two_wheelers", "motorcycles": "two_wheelers",
    "motorbike": "two_wheelers", "motorbikes": "two_wheelers",
    "two_wheeler": "two_wheelers", "two_wheelers": "two_wheelers",
    "twowheeler": "two_wheelers", "twowheelers": "two_wheelers",
}

# --- Data processing -----------------------------------------------------------
# Length of one analysis interval in minutes. Member 2 reports 1-minute counts.
INTERVAL_MINUTES = 1
# What an absent minute inside the observed range means:
#   "missing" -> unknown (flow=None, interpolated for forecasting, flagged)
#   "zero"    -> no vehicle crossed (correct for Member 2's ORIGINAL output, which
#                only created a minute when something crossed; the updated
#                video_tracker.py now writes explicit zeros).
MISSING_INTERVAL_POLICY = "missing"
# Same (minute, lane) reported twice: "last" keeps the later row, "sum" adds them.
DUPLICATE_POLICY = "last"
# Cumulative counters are assumed to start from this value before the first reading.
CUMULATIVE_INITIAL_BASELINE = 0

# --- Congestion classification (flow based) ---------------------------------------
# Upper limits, in vehicles per minute PER LANE (inclusive):
#   flow <= FREE_MAX      -> FREE
#   flow <= MODERATE_MAX  -> MODERATE
#   otherwise             -> HEAVY
# 10 and 20 veh/min/lane = 600 and 1200 veh/h/lane, rule-of-thumb figures for
# light and busy urban lanes (a lane saturates at roughly 1800-2000 veh/h).
# THEY ARE ASSUMPTIONS, NOT CALIBRATED against labelled TrafficVision footage.
FREE_MAX_PER_LANE = 10.0
MODERATE_MAX_PER_LANE = 20.0
# Used when the number of lanes is not known from the data (Member 2 uses 4).
DEFAULT_LANE_COUNT = 4

# Optional passenger-car-unit weighting (heavier vehicles take more road space).
# Off by default so the result is a plain vehicle count.
USE_PCU = False
PCU_WEIGHTS = {"cars": 1.0, "two_wheelers": 0.5, "buses": 3.0, "trucks": 3.0}

LEVEL_FREE = "FREE"
LEVEL_MODERATE = "MODERATE"
LEVEL_HEAVY = "HEAVY"
LEVEL_UNKNOWN = "UNKNOWN"    # no usable data

# --- Forecasting -----------------------------------------------------------------
FORECAST_HORIZONS_MINUTES = (5, 10, 15)

# "auto" | "weighted_moving_average" | "exponential_smoothing" | "damped_trend"
FORECAST_METHOD = "auto"

WMA_WINDOW = 5                 # last N intervals, linearly increasing weights
SES_ALPHA = 0.4                # exponential smoothing level weight
HOLT_ALPHA = 0.4               # damped trend: level weight
HOLT_BETA = 0.1                # damped trend: trend weight
HOLT_PHI = 0.9                 # damped trend: damping (<1 so trends fade out)

MIN_HISTORY_POINTS = 3         # fewer valid intervals -> no forecast at all
MIN_POINTS_FOR_SELECTION = 12  # fewer -> use the WMA default, status "limited_data"
MAX_MISSING_RATIO = 0.3        # more interpolated/missing intervals -> no forecast
MAX_INTERPOLATED_GAP = 5       # a longer run of missing intervals is not bridged

# --- Evaluation ----------------------------------------------------------------------
TEST_FRACTION = 0.4            # last 40 % of the series is held out (chronological)
MIN_TRAIN_POINTS = 8
MIN_EVAL_ORIGINS = 5           # fewer test forecasts -> "not_evaluated"

# Forecasting and classification use no randomness. Only synthetic.py (demo/test
# data, never real detections) uses a fixed seed:
SYNTHETIC_SEED = 42

DISCLAIMER = (
    "Estimated traffic-flow-based congestion classification and forecast. It is "
    "decision-support information only: speed, occupancy and road geometry are "
    "not measured, and it must not be used for automatic traffic or signal "
    "control. Human operators remain responsible for traffic-management decisions."
)
