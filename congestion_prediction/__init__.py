"""TrafficVision - Member 3: congestion classification and traffic prediction.

    from congestion_prediction import build_traffic_report
    report = build_traffic_report()          # reads Member 2's vehicle_counts.json

Importing this package needs only the Python standard library.
"""

from .config import (
    LEVEL_FREE, LEVEL_HEAVY, LEVEL_MODERATE, LEVEL_UNKNOWN,
)
from .congestion_classifier import (
    InvalidTrafficDataError, classify_congestion, classify_flow,
)
from .data_processor import (
    DataFormatError, ProcessedTraffic, process_traffic_data,
)
from .traffic_predictor import evaluate_forecast, predict_traffic
from .pipeline import build_traffic_report, save_report

__all__ = [
    "LEVEL_FREE", "LEVEL_MODERATE", "LEVEL_HEAVY", "LEVEL_UNKNOWN",
    "process_traffic_data", "classify_congestion", "classify_flow",
    "predict_traffic", "evaluate_forecast", "build_traffic_report", "save_report",
    "ProcessedTraffic", "DataFormatError", "InvalidTrafficDataError",
]
