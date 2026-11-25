"""Short-term traffic-flow forecasting (next 5 / 10 / 15 minutes) + evaluation.

Why simple methods, not ARIMA/Prophet
-------------------------------------
The only real data available is a few minutes of one-minute counts. ARIMA or
Prophet need far longer histories and extra dependencies, and would be
over-fitted on this little data. Three transparent, dependency-free baselines
are implemented instead:

  weighted_moving_average  last N intervals, newer ones weigh more      (flat forecast)
  exponential_smoothing    simple exponential smoothing                 (flat forecast)
  damped_trend             Holt's linear trend with damping (phi < 1)   (trend fades out)

method="auto" compares the three on the last part of the available history
(one-step-ahead, chronological) when there are enough points, otherwise it uses
the weighted moving average and reports status "limited_data".

Forecast quantity
-----------------
Flow is vehicles per minute. For a horizon of H minutes,
``predicted_vehicle_flow`` is the expected AVERAGE flow per minute over the next
H minutes and ``predicted_vehicles_in_window`` is that times H. The level comes
from the same thresholds as the current-congestion classification.

No randomness is used, so results are reproducible and no seeds are needed.
These are statistical estimates, not guarantees.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import config
from .congestion_classifier import classify_flow

Series = Sequence[Optional[float]]

METHOD_NAMES = ("weighted_moving_average", "exponential_smoothing", "damped_trend")


# --------------------------------------------------------------------------- #
# Forecasters: history (list of floats, len >= 1) -> list of `steps` floats
# --------------------------------------------------------------------------- #
def _wma(history: List[float], steps: int, window: int) -> List[float]:
    recent = history[-window:]
    weights = list(range(1, len(recent) + 1))
    value = sum(w * x for w, x in zip(weights, recent)) / sum(weights)
    return [value] * steps


def _ses(history: List[float], steps: int, alpha: float) -> List[float]:
    level = history[0]
    for x in history[1:]:
        level = alpha * x + (1 - alpha) * level
    return [level] * steps


def _damped(history: List[float], steps: int, alpha: float, beta: float,
            phi: float) -> List[float]:
    if len(history) < 2:
        return [history[-1]] * steps
    level, trend = history[0], history[1] - history[0]
    for x in history[1:]:
        previous = level
        level = alpha * x + (1 - alpha) * (previous + phi * trend)
        trend = beta * (level - previous) + (1 - beta) * phi * trend
    out, damp = [], 0.0
    for h in range(1, steps + 1):
        damp += phi ** h
        out.append(level + damp * trend)
    return out


def _forecaster(name: str, params: Dict[str, float]) -> Callable[[List[float], int], List[float]]:
    if name == "weighted_moving_average":
        return lambda h, s: _wma(h, s, int(params["wma_window"]))
    if name == "exponential_smoothing":
        return lambda h, s: _ses(h, s, params["ses_alpha"])
    if name == "damped_trend":
        return lambda h, s: _damped(h, s, params["holt_alpha"], params["holt_beta"],
                                    params["holt_phi"])
    raise ValueError(f"unknown forecasting method: {name!r}")


def _params(overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
    params = {
        "wma_window": config.WMA_WINDOW, "ses_alpha": config.SES_ALPHA,
        "holt_alpha": config.HOLT_ALPHA, "holt_beta": config.HOLT_BETA,
        "holt_phi": config.HOLT_PHI,
    }
    if overrides:
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(f"unknown forecast parameter(s): {sorted(unknown)}")
        params.update(overrides)
    if params["wma_window"] < 1:
        raise ValueError("wma_window must be >= 1")
    for key in ("ses_alpha", "holt_alpha", "holt_beta"):
        if not 0 < params[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 < params["holt_phi"] <= 1:
        raise ValueError("holt_phi must be in (0, 1]")
    return params


# --------------------------------------------------------------------------- #
# Series cleaning
# --------------------------------------------------------------------------- #
def _is_good(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def clean_series(values: Series, max_gap: Optional[int] = None
                 ) -> Tuple[List[float], List[bool], Dict[str, Any]]:
    """Prepare a flow series for forecasting.

    - invalid / None entries count as missing;
    - leading and trailing missing entries are dropped;
    - interior gaps of up to ``max_gap`` intervals are bridged by linear
      interpolation (flagged in the returned ``interpolated`` list);
    - a longer gap is NOT bridged: only the data after the last such gap is kept.

    Returns (cleaned_values, interpolated_flags, info).
    """
    max_gap = config.MAX_INTERPOLATED_GAP if max_gap is None else max_gap
    items: List[Optional[float]] = [float(v) if _is_good(v) else None for v in values]
    info: Dict[str, Any] = {"input_points": len(items), "invalid_or_missing": 0,
                            "interpolated": 0, "dropped_before_long_gap": 0}
    info["invalid_or_missing"] = sum(1 for v in items if v is None)

    first = next((i for i, v in enumerate(items) if v is not None), None)
    if first is None:
        return [], [], info
    last = max(i for i, v in enumerate(items) if v is not None)
    items = items[first:last + 1]

    cleaned: List[float] = []
    flags: List[bool] = []
    i = 0
    while i < len(items):
        if items[i] is not None:
            cleaned.append(items[i])           # type: ignore[arg-type]
            flags.append(False)
            i += 1
            continue
        j = i
        while items[j] is None:
            j += 1
        gap = j - i
        if gap > max_gap:                      # do not bridge: restart after the gap
            info["dropped_before_long_gap"] += len(cleaned)
            cleaned, flags = [], []
        else:
            left, right = cleaned[-1], items[j]
            for k in range(1, gap + 1):
                cleaned.append(left + (right - left) * k / (gap + 1))  # type: ignore[operator]
                flags.append(True)
                info["interpolated"] += 1
        i = j
    return cleaned, flags, info


# --------------------------------------------------------------------------- #
# Method selection
# --------------------------------------------------------------------------- #
def _select_method(history: List[float], params: Dict[str, float]) -> Tuple[str, str]:
    """Pick a method using ONLY ``history`` (one-step-ahead, chronological)."""
    n = len(history)
    if n < config.MIN_POINTS_FOR_SELECTION:
        return ("weighted_moving_average",
                f"only {n} points (< {config.MIN_POINTS_FOR_SELECTION}): too few to "
                f"compare methods, using the weighted moving average default")
    holdout = max(3, n // 4)
    errors: Dict[str, float] = {}
    for name in METHOD_NAMES:
        fn = _forecaster(name, params)
        total = 0.0
        for t in range(n - holdout, n):
            total += abs(fn(history[:t], 1)[0] - history[t])
        errors[name] = total / holdout
    best = min(METHOD_NAMES, key=lambda m: (errors[m], METHOD_NAMES.index(m)))
    detail = ", ".join(f"{m} MAE={errors[m]:.2f}" for m in METHOD_NAMES)
    return best, f"chosen on the last {holdout} of {n} points (one-step-ahead): {detail}"


def _horizon_key(minutes: int) -> str:
    return f"next_{minutes}_minutes"


def _window_mean(step_values: List[float], horizon_minutes: int, interval_minutes: int) -> float:
    steps = max(1, math.ceil(horizon_minutes / interval_minutes))
    window = step_values[:steps]
    return max(0.0, sum(window) / len(window))      # never negative


# --------------------------------------------------------------------------- #
# Public: predict
# --------------------------------------------------------------------------- #
def predict_traffic(
    flow_series: Series,
    *,
    horizons: Optional[Sequence[int]] = None,
    interval_minutes: Optional[int] = None,
    lanes: Optional[int] = None,
    method: Optional[str] = None,
    params: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Forecast future flow and congestion level.

    flow_series   historical flow per interval in vehicles PER MINUTE, oldest
                  first; None marks a missing interval.
    horizons      minutes ahead (default config: 5, 10, 15).
    interval_minutes  length of one history interval.
    lanes         number of lanes for the thresholds (None -> config default).
    method        "auto" (default) or one of METHOD_NAMES.

    The result separates the future from the past: it contains forecasts only
    (``"kind": "forecast"``); the history stays in the caller's data. With too
    little data the values are None, level UNKNOWN and ``status`` says why.
    """
    horizons = tuple(int(h) for h in (horizons or config.FORECAST_HORIZONS_MINUTES))
    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must be positive minutes")
    interval = int(interval_minutes or config.INTERVAL_MINUTES)
    if interval < 1:
        raise ValueError("interval_minutes must be >= 1")
    method = method or config.FORECAST_METHOD
    if method != "auto" and method not in METHOD_NAMES:
        raise ValueError(f"unknown forecasting method: {method!r}")
    parameters = _params(params)

    history, interpolated, info = clean_series(flow_series)
    n_real = len(history) - sum(interpolated)
    total_slots = len(flow_series) if flow_series is not None else 0
    missing_ratio = (info["invalid_or_missing"] / total_slots) if total_slots else 1.0

    result: Dict[str, Any] = {
        "kind": "forecast",
        "status": "insufficient_data",
        "message": "",
        "method": None,
        "method_selection": None,
        "interval_minutes": interval,
        "history_points_used": len(history),
        "observed_points_used": n_real,
        "interpolated_points": int(sum(interpolated)),
        "data_cleaning": info,
        "parameters": parameters,
        "horizons": {},
    }

    def empty(message: str, status: str) -> Dict[str, Any]:
        result["status"] = status
        result["message"] = message
        for h in horizons:
            result["horizons"][_horizon_key(h)] = {
                "minutes": h, "predicted_vehicle_flow": None,
                "predicted_vehicles_in_window": None,
                "predicted_level": config.LEVEL_UNKNOWN,
            }
        return result

    if n_real < config.MIN_HISTORY_POINTS:
        return empty(f"need at least {config.MIN_HISTORY_POINTS} observed intervals to "
                     f"forecast, have {n_real}", "insufficient_data")
    if missing_ratio > config.MAX_MISSING_RATIO:
        return empty(f"{missing_ratio:.0%} of the history is missing or unusable "
                     f"(limit {config.MAX_MISSING_RATIO:.0%})", "too_many_missing")

    if method == "auto":
        chosen, reason = _select_method(history, parameters)
    else:
        chosen, reason = method, "method set by the caller"
    steps = max(math.ceil(h / interval) for h in horizons)
    step_values = _forecaster(chosen, parameters)(history, steps)

    for h in horizons:
        per_minute = _window_mean(step_values, h, interval)
        result["horizons"][_horizon_key(h)] = {
            "minutes": h,
            "predicted_vehicle_flow": round(per_minute, 2),
            "predicted_vehicles_in_window": round(per_minute * h, 1),
            "predicted_level": classify_flow(per_minute, lanes, thresholds),
        }

    limited = len(history) < config.MIN_POINTS_FOR_SELECTION
    result.update(
        status="limited_data" if limited else "ok",
        method=chosen,
        method_selection=reason,
        message=("short history: forecast is a rough estimate"
                 if limited else "forecast from the weighted history"),
    )
    if result["interpolated_points"]:
        result["message"] += f"; {result['interpolated_points']} missing interval(s) interpolated"
    return result


# --------------------------------------------------------------------------- #
# Public: evaluate
# --------------------------------------------------------------------------- #
def _metrics(errors: List[float]) -> Dict[str, float]:
    n = len(errors)
    return {"MAE": round(sum(abs(e) for e in errors) / n, 3),
            "RMSE": round(math.sqrt(sum(e * e for e in errors) / n), 3)}


def evaluate_forecast(
    flow_series: Series,
    *,
    horizons: Optional[Sequence[int]] = None,
    interval_minutes: Optional[int] = None,
    test_fraction: Optional[float] = None,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Chronological hold-out evaluation (no random split).

    The first (1 - test_fraction) of the series is the training period; the rest
    is the test period. For every forecast origin in the test period the model
    sees only data before the origin, forecasts the next H minutes (average
    flow per minute) and is compared with the actual average over those minutes.
    Compared: weighted_moving_average, exponential_smoothing, damped_trend, auto
    (selected from past data only) and the baseline ``last_value`` (repeat the
    last observation). Targets that were interpolated are never scored.

    Reports "not_evaluated" (with the reason) when the series is too short.
    Metrics are in vehicles per minute.
    """
    horizons = tuple(int(h) for h in (horizons or config.FORECAST_HORIZONS_MINUTES))
    interval = int(interval_minutes or config.INTERVAL_MINUTES)
    fraction = config.TEST_FRACTION if test_fraction is None else test_fraction
    if not 0 < fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    parameters = _params(params)

    series, interpolated, _ = clean_series(flow_series)
    n = len(series)
    out: Dict[str, Any] = {
        "status": "not_evaluated", "reason": "", "points": n, "interval_minutes": interval,
        "methodology": "chronological hold-out, forecasts from past data only",
        "unit": "vehicles per minute (average over the horizon window)",
        "train_range": None, "test_range": None, "per_horizon": {},
    }
    split = max(config.MIN_TRAIN_POINTS, int(n * (1 - fraction)))
    if n < config.MIN_TRAIN_POINTS + config.MIN_EVAL_ORIGINS:
        out["reason"] = (f"only {n} usable intervals; need at least "
                         f"{config.MIN_TRAIN_POINTS + config.MIN_EVAL_ORIGINS} for a "
                         f"train/test evaluation. Not enough data to evaluate reliably.")
        return out

    out["train_range"] = [0, split - 1]
    out["test_range"] = [split, n - 1]
    any_done = False
    for h in horizons:
        steps = max(1, math.ceil(h / interval))
        names = list(METHOD_NAMES) + ["auto", "last_value"]
        errors: Dict[str, List[float]] = {name: [] for name in names}
        for origin in range(split, n - steps + 1):
            if any(interpolated[origin:origin + steps]):
                continue                                    # never score guessed values
            actual = sum(series[origin:origin + steps]) / steps
            history = series[:origin]
            for name in METHOD_NAMES:
                step_values = _forecaster(name, parameters)(history, steps)
                errors[name].append(_window_mean(step_values, h, interval) - actual)
            chosen, _ = _select_method(history, parameters)
            step_values = _forecaster(chosen, parameters)(history, steps)
            errors["auto"].append(_window_mean(step_values, h, interval) - actual)
            errors["last_value"].append(history[-1] - actual)

        key = _horizon_key(h)
        origins = len(errors["last_value"])
        if origins < config.MIN_EVAL_ORIGINS:
            out["per_horizon"][key] = {
                "status": "not_evaluated", "origins": origins,
                "reason": f"only {origins} test forecasts (need {config.MIN_EVAL_ORIGINS})"}
            continue
        any_done = True
        results = {name: _metrics(errors[name]) for name in names}
        best = min(METHOD_NAMES + ("auto",), key=lambda m: results[m]["MAE"])
        out["per_horizon"][key] = {
            "status": "evaluated", "origins": origins, "results": results,
            "best_method_by_MAE": best,
            "best_beats_last_value_baseline": results[best]["MAE"] < results["last_value"]["MAE"],
        }
    if any_done:
        out["status"] = "evaluated"
        out["reason"] = ""
    else:
        out["reason"] = "no horizon had enough test forecasts"
    return out
