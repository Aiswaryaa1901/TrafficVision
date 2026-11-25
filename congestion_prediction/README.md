# TrafficVision – Congestion Classification & Traffic Prediction (Member 3)

Turns Member 2's vehicle counts into (a) a **FREE / MODERATE / HEAVY** congestion
level per interval and (b) a **next 5 / 10 / 15 minute** flow forecast with the
predicted congestion level, in one JSON-friendly report for Member 4's dashboard.

> **What the congestion level means.** Member 2 counts vehicles crossing a line.
> From that we get traffic **flow** (vehicles per minute). Speed, road occupancy
> and road geometry are *not* measured, so the result is an **estimated
> traffic-flow-based congestion classification**, not road density. A jammed road
> can show *low* flow, so FREE means "low flow", not guaranteed smooth traffic.
> Output is decision-support only – **never for automatic signal/traffic control**;
> human operators stay responsible.

---

## 1. Architecture and folder structure

```
congestion_prediction/
  config.py                 all settings (thresholds, horizons, parameters, paths)
  data_processor.py         Member 2 output -> clean per-interval data + data-quality info
  congestion_classifier.py  flow -> FREE / MODERATE / HEAVY (classify_congestion, classify_flow)
  traffic_predictor.py      forecasting (predict_traffic) + chronological evaluation (evaluate_forecast)
  pipeline.py               build_traffic_report() / save_report()  <- Member 4 entry point
  synthetic.py              SYNTHETIC demo/test data generator (never real traffic)
  examples/run_congestion.py   command-line demo
  tests/                    88 automated tests (pytest)
  requirements.txt  .gitignore  README.md
```

Flow: `vehicle_counts.json` -> `process_traffic_data` -> `classify_congestion` per interval
-> `predict_traffic` on the flow series -> `build_traffic_report` (one dict) -> dashboard.
Only the Python standard library is needed at run time. The input file is only ever
**read**; `save_report` refuses to overwrite `vehicle_counts*` files.

## 2. Installation (Windows PowerShell, from the TrafficVision root)

```powershell
python --version                      # 3.9 - 3.12 expected
python -m venv congestion_prediction\.venv
congestion_prediction\.venv\Scripts\Activate.ps1
# If PowerShell blocks the script:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install -r congestion_prediction\requirements.txt     # only pytest (tests)
```

Recorded environment of the development run: Python 3.12.3, pytest 9.2.0 (tests only); the
Member 2 integration tests also use opencv 5.0.0 / numpy 2.5.4 (Member 1's dependencies; those
tests skip themselves if cv2 is missing). **Verified on Windows with Python 3.14.7 and pytest 9.2.0; other Python versions and platforms have not been verified (written to
be compatible: `pathlib`, no absolute paths).

## 3. Run

```powershell
python congestion_prediction\examples\run_congestion.py                      # Member 2's vehicle_counts.json
python congestion_prediction\examples\run_congestion.py path\to\vehicle_counts.json --start-time 2026-11-25T10:00:00
python congestion_prediction\examples\run_congestion.py --demo-synthetic     # SYNTHETIC data, clearly labelled
python -m pytest congestion_prediction\tests -v
```

The report is saved to `congestion_prediction\outputs\traffic_report.json` (git-ignored).

## 4. Input expected from Member 2 (actual format, not assumed)

`vehicle_counting/vehicle_counts.json`, written by `video_tracker.py`:

```json
[{"minute": 0, "lanes": {"lane_1": {"cars": 19, "buses": 0, "trucks": 9, "two_wheelers": 0}, "lane_2": {}}}]
```

* **Interval counts, not cumulative**: vehicles that crossed the line *during that video minute*.
  (`cumulative=True` is supported for other sources and takes differences.)
* `minute` is the minute index from the start of the video. There is **no wall-clock time**, so
  `timestamp` is `null` unless you pass `start_time`; `video_time` (`MM:SS`) is always given.
* Categories: `cars, buses, trucks, two_wheelers` (also accepted: car, bus, truck, motorcycle, motorbike, two_wheeler).
* Also read: flat JSON records, `{"records": [...]}`, CSV (`minute,lane,car,bus,truck,motorcycle`), ISO timestamps.
* Optional sidecar `vehicle_counts_meta.json` (written by the updated `video_tracker.py`) gives the video
  duration so a partial last minute is recognised.

## 5. Data-processing workflow

1. Read JSON/CSV (clear `DataFormatError` if unreadable; empty input -> `status: no_data`, no crash).
2. Validate every (minute, lane) entry. Counts must be whole numbers >= 0; a bad value (negative, text, NaN, inf,
   fractional, bool, empty) **rejects that lane entry** and is reported - nothing is guessed.
3. Absent categories in an entry count as 0 (logged). Unknown fields (e.g. `total`) are ignored and recomputed.
4. Duplicates for the same (minute, lane): `last` wins (default) or `sum`; always logged.
5. Cumulative input: per-lane differences; a counter **reset** (value goes down) uses the new reading as the count since
   the reset; skipped minutes are spread evenly and flagged.
6. Group into intervals (`INTERVAL_MINUTES`, default 1). A minute with no entry is **missing** (`flow=None`, not 0)
   unless `missing_policy="zero"`. The updated Member 2 now writes explicit zeros.
7. Each interval is flagged `observed`, `complete`, `lanes_missing`, `usable_for_forecast`. A last minute cut short
   (known from the metadata) is `complete: false` and is neither shown as "latest" nor used for forecasting.
8. `data_quality` summarises rejected/duplicate/reset/missing counts and lists warnings.

## 6. Congestion classification

Flow = vehicles/minute (optionally PCU-weighted, `USE_PCU`, off by default).

| Level | Rule (per lane, upper limit inclusive) | 4 lanes (Member 2's layout) |
|---|---|---|
| FREE | <= 10 veh/min/lane | <= 40 veh/min |
| MODERATE | <= 20 veh/min/lane | <= 80 veh/min |
| HEAVY | above | > 80 veh/min |
| UNKNOWN | no usable data | - |

**How thresholds were chosen / limits.** 10 and 20 veh/min/lane (600 and 1200 veh/h/lane) are rule-of-thumb figures
for light and busy urban lanes; a lane saturates at roughly 1800-2000 veh/h. They are **assumptions, not calibrated**
on labelled TrafficVision footage (none exists). Adjust them in `config.py` (`FREE_MAX_PER_LANE`,
`MODERATE_MAX_PER_LANE`, `DEFAULT_LANE_COUNT`) or per call with
`thresholds={"free_max_per_lane": .., "moderate_max_per_lane": ..}`. The lane count comes from the data (4 for
Member 2). Each lane also gets its own level (1-lane thresholds).

## 7. Forecasting

* Horizons: 5, 10, 15 minutes (`FORECAST_HORIZONS_MINUTES`). For horizon H, `predicted_vehicle_flow` is the expected
  **average vehicles per minute over the next H minutes**; `predicted_vehicles_in_window` = that x H.
* Methods (no extra dependencies, no randomness, parameters in `config.py`): weighted moving average (window 5),
  exponential smoothing (alpha 0.4), damped-trend Holt (alpha 0.4, beta 0.1, phi 0.9). `auto` compares them one-step-ahead
  on the last quarter of the history when >= 12 points exist; with fewer it uses the moving average and reports `limited_data`.
* **Why not ARIMA/Prophet:** only minutes of real data exist; such models would over-fit and need heavier dependencies.
* Safeguards: < 3 observed intervals -> `insufficient_data` (values `null`, level UNKNOWN); > 30 % missing ->
  `too_many_missing`; short gaps (<= 5) are linearly interpolated and flagged, longer gaps are not bridged;
  predictions are clipped at 0.

## 8. Evaluation

`evaluate_forecast` uses a **chronological** hold-out (first 60 % train, last 40 % test; no shuffling). Every test origin
sees only earlier data; the forecast of the next H-minute average is scored against the actual average (MAE, RMSE in
vehicles/min) and compared with the baseline `last_value`. Interpolated values are never scored. With too little data it
reports `not_evaluated`.

* **Real data:** Member 2's sample contains **2 minutes** -> forecasting and evaluation **cannot be done reliably**.
  No accuracy is claimed for real traffic.
* **Mechanics check on SYNTHETIC data only** (`generate_synthetic_counts(60)`, seed 42; 60 one-minute points; train
  indices 0-35, test 36-59; sine-wave flow + noise, so it says nothing about real accuracy). MAE / RMSE in vehicles/min:

| Horizon | origins | weighted MA | exp. smoothing | damped trend | auto | last_value baseline |
|---|---|---|---|---|---|---|
| 5 min | 20 | 4.29 / 4.58 | 4.47 / 4.78 | **2.99 / 3.41** | 3.34 / 3.77 | 3.55 / 4.43 |
| 10 min | 15 | 7.16 / 7.31 | 7.28 / 7.43 | **5.49 / 5.66** | 6.19 / 6.42 | 5.90 / 6.29 |
| 15 min | 10 | 9.77 / 9.89 | 9.86 / 9.97 | **8.25 / 8.35** | 9.54 / 9.65 | 8.70 / 8.99 |

Replace this with results on a real, longer recording (30-60 minutes or more) before reporting accuracy.

## 9. Output for Member 4

```python
from congestion_prediction import build_traffic_report
report = build_traffic_report()          # or build_traffic_report(path, start_time="2026-11-25T10:00:00")
```

Never raises on bad data (`strict=True` to raise). Main keys:

```text
status            ok | ok_with_warnings | no_data | error        (+ error message)
data_source       {label, file, counts_are_cumulative}           label "SYNTHETIC_DEMO" for generated data
classification    {method, lanes, thresholds_vehicles_per_minute, note}
history[]         measured intervals: index, video_time, timestamp, observed, complete,
                  vehicle_flow{cars,buses,trucks,two_wheelers,total}, flow_per_minute,
                  congestion{level, flow_per_minute}, lanes{lane_n: {vehicle_flow, flow_per_minute, level}}
latest            most recent COMPLETE interval (same fields + is_partial)
prediction        kind:"forecast", status (ok|limited_data|insufficient_data|too_many_missing), message, method,
                  method_selection, horizons{ next_5_minutes|next_10_minutes|next_15_minutes:
                  {minutes, predicted_vehicle_flow, predicted_vehicles_in_window, predicted_level} }, based_on
evaluation        status evaluated|not_evaluated|skipped + metrics (section 8)
data_quality      counters, warnings[], last_interval_completeness
disclaimer
```

Example (abridged; from a **synthetic** run):

```json
{"latest": {"video_time": "59:00",
            "vehicle_flow": {"cars": 26, "buses": 0, "trucks": 0, "two_wheelers": 3, "total": 29},
            "congestion": {"level": "FREE", "flow_per_minute": 29.0}},
 "prediction": {"status": "ok", "method": "damped_trend",
   "horizons": {"next_5_minutes": {"minutes": 5, "predicted_vehicle_flow": 28.82,
                                   "predicted_vehicles_in_window": 144.1, "predicted_level": "FREE"}}}}
```

Category names follow Member 2 (`cars, buses, trucks, two_wheelers`), not the illustrative `car/motorcycle` names.
Dashboard tips: plot `history[*].flow_per_minute` as actual values and `prediction.horizons` as a visibly different
forecast; show `prediction.status`/`message` when it is not `ok`; show `disclaimer`. Forecasts are for the whole road
(not per lane). Smaller API: `process_traffic_data()`, `classify_congestion()`, `predict_traffic()`, `evaluate_forecast()`.

## 10. Integration with Member 2 - changes made (authorised by the project owner)

Only `vehicle_counting/tracker.py` and `vehicle_counting/video_tracker.py` were changed. Counting logic, defaults
(line x=960, lanes 300/459/647, max_distance 50, max_missed 10) and the output format are unchanged.

* `tracker.py`: optional constructor arguments (defaults identical); `fill_missing_minutes()` + `export_minute_counts()`
  so minutes without crossings are written as explicit zeros; malformed detections/timestamps no longer raise;
  crossings of classes with no category are counted in `unmapped_count` (still in `vehicle_count`, as before).
* `video_tracker.py`: wrapped in `main()` (importing no longer runs the video); video path as argument (default
  `sample_data/Traffic1.mp4` from the project root); output path independent of the working directory; FPS 0 -> 25 assumed;
  headless-safe preview (`--no-show`); line/lane geometry rescaled **only** if it would fall outside the frame; JSON also
  written if stopped early; new sidecar `vehicle_counts_meta.json` (duration, fps, totals).
* The existing sample `vehicle_counts.json` was not touched.
* Limitations of Member 2 that remain: only right-to-left crossings are counted; centroid matching with a fixed 50 px
  radius can swap IDs in dense traffic; the counting line/lanes are fixed for one camera view.

Run Member 2 as before from the root: `python vehicle_counting\video_tracker.py path\to\traffic.mp4 [--no-show]`.
Verified here only with a **stubbed detector and a synthetic clip** (no real YOLO run, no real traffic video available).

## 11. Tests

`python -m pytest congestion_prediction\tests -v` - 88 tests: classification (free/moderate/heavy, empty, invalid,
missing and different categories, boundaries, PCU, lanes), data processing (JSON, CSV, missing/duplicate timestamps,
cumulative differences, counter resets, invalid numbers, metadata), forecasting (enough/insufficient/missing data,
horizons, structure, non-negative, determinism, chronological evaluation), the report, and Member 2 integration.
Fixtures are synthetic and labelled; two tests read Member 2's real 2-minute sample read-only.

## 12. Known limitations

* Flow-based estimate only (no speed/occupancy); thresholds uncalibrated; stop-and-go jams can read as low flow.
* Only 2 minutes of real data exist -> no reliable real forecast or accuracy figure yet.
* Vehicle classes beyond car/bus/truck/motorcycle (auto-rickshaw, van) depend on Member 1's COCO model.
* Without `start_time` there is no clock time. Last-minute completeness is "unknown" without the metadata sidecar.
* Forecasts are statistical estimates from short histories, not guarantees.

## 13. Responsible AI, reproducibility, AI assistance

* No personal data, plates or faces; only counts are used; no paid APIs. Synthetic data comes from `synthetic.py`
  (fixed seed 42, labelled `SYNTHETIC_DEMO`); no external datasets, so no dataset licences apply.
* Forecasting and classification are deterministic (no seeds). The configuration used is recorded in each report
  (`prediction.parameters`, `classification`).
* **AI assistance:** this module and the Member 2 fixes were written with Claude (Anthropic) from the project owner's
  "Master Development Prompt - Member 3" (build congestion classification and prediction inside the existing project,
  with tests and documentation). Review all code before submission and keep this note in the project report.
