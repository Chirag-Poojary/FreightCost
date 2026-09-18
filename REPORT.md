# Real-Data Rebuild — Results Report

Rebuilt the Indian B2B road-freight dataset on **real weather** (Open-Meteo Archive)
and **real diesel prices** (PPAC/IOC), then retrained both models and added a
SARIMA forecaster for the fuel and weather series. OSRM real distances were
skipped by choice (haversine×circuity route cache retained).

## 1. Environment — verified
venv (Python 3.14) + pandas, numpy, scikit-learn, xgboost, requests, scipy, faker,
statsmodels. Smoke test `--orders 2000 --seed 7`: OOF MAE ₹649 (2.1%), anomaly
rate 6.2%. Full pipeline (build + train_model2) runs clean.

## 2. Real weather — Open-Meteo Archive (no key)
`run_openmeteo.py` rewritten: resumable, and fetches each dest hub's **full
2023-01-01…2026-01-31 daily series in one call (18 calls, ~70s)** instead of
~2,800 per-window calls — same underlying real observations, complete coverage,
and it now also stores `temperature_2m_max` (the original fetched but never wrote
it, silently zeroing heat-day scoring).
Output: `real_cache/weather_cache.csv` — 2,916 hub-week rows, 18 hubs, 20,286 real
hub-days. Sanity: Mumbai Jul-2024 30–83 mm/day (real monsoon).

## 3. Real diesel — PPAC/IOC (no key)
Source: PPAC daily Retail Selling Prices (DashRSP mirror of the PPAC bulletin),
diesel, 4 metros, 2023-01-01…2025-06-20, real band (Delhi ₹87.6, Mumbai ₹90.0).
`load_ppac_diesel.py` extended: 4 metros → 4 states **direct-real**; other 9 hub
states **derived from the real national daily backbone + documented VAT offsets**
(labeled `PPAC-derived` in `source`); forward-filled daily to 2025-12-31 (diesel
was administratively price-static in this window, so carry-forward is faithful).
Output: `real_cache/fuel_prices.csv` — 14,248 (date,state) rows, 13 states, real
interstate spread (Gujarat ₹86.9 → Andhra ₹95.1). 4,384 direct-real, 9,864 derived.

## 4. OSRM — skipped (by choice)
Route cache remains haversine×Indian-circuity. Docker + India Geofabrik extract +
preprocessing was judged too heavy for a 153-route cache.

## 5. Pipeline refactor so the build CONSUMES real caches
`build_dataset.py` + `scripts/weather_fuel.py` previously **regenerated** weather
and fuel internally. Refactored (backward-compatible):
- `weather_fuel.RealWeather` explodes the cache into a (dest_hub, date) daily
  lookup; `make_weather_source()` / `load_fuel_prices()` load real files when
  `--weather-cache` / `--fuel-cache` are passed, else fall back to modeled.
- Real weather cache is passed through to the output unchanged.
- Intentional data-quality residue stays in the **build**, not the raw caches.
Full run: `build_dataset.py --orders 60000 --seed 42 --weather-cache … --fuel-cache …`
→ 0 fuel-fallback rows (full real coverage), 60k orders / 60,120 invoices.

## 6. Model results (real-data 60k, seed 42)

### Model 1 — Freight Cost (XGBoost, 5-fold OOF)
- **OOF MAE = ₹510.4**, **MAPE = 1.66%** (mean cost ₹31,238).
- All 60k rows carry an out-of-fold prediction (folds 0–4).

### Model 2 — Invoice Risk (XGBoost, 25% holdout)
- **PR-AUC 0.840 · Recall 0.887 · Precision 0.494 · F1 0.634** (threshold 0.5).
- Per-anomaly-type recall (as intended, detention easiest → weight hardest):
  - detention_padding **1.000** (296/296)
  - toll_inflation **0.903** (252/279)
  - weight_discrepancy **0.773** (255/330)
- Matches the documented reference (PR-AUC ≈0.87, recall ≈0.89, precision ≈0.50).

### Leakage checks — all pass
- `anomaly_type` exists **only** in `_ground_truth_audit.csv` (not in orders.csv,
  invoices.csv, or the Model 2 feature list).
- `model_a_predicted_cost` is genuine OOF: MAE ₹510 (>0), only 0.145% of rows have
  |resid|<₹1 (in-fold memorisation would be ~100%).
- `vendor_historical_risk_score` present, finite, computed causally (past-only).

## 7. SARIMA forecaster (statsmodels) — `forecast_series.py`
Honest metrics: sMAPE (bounded) + skill vs a naive baseline (persistence for fuel,
seasonal-naive/"same week last year" for weather). Backtest = held-out tail, then
refit on full series for the forward forecast.

- **Fuel** (per-state daily, ARIMA(1,1,1), 60-day holdout, 30-day forecast):
  MAE ≈ **0.00**, sMAPE **0.00%** — because diesel was **perfectly flat** in the
  holdout (test SD = 0). This is real (held prices), not overfitting: ARIMA
  correctly forecasts a flat continuation. Forecast → `forecasts/fuel_forecast.csv`.
- **Weather temperature** (per-hub weekly-mean, SARIMA(1,0,1)(1,1,0,52), 26-week
  holdout/forecast): median **MAE 1.40 °C, sMAPE 5.1%, skill +0.08** vs
  seasonal-naive — SARIMA modestly beats "same week last year".
- **Weather precipitation** (per-hub weekly-total): median **MAE 19.4 mm/wk,
  RMSE 31.7 mm/wk, skill +0.07**. High sMAPE (~141%) is inherent to intermittent
  monsoon rainfall (many near-zero weeks); a few hubs underperform seasonal-naive.
  Forecasts → `forecasts/weather_forecast.csv`; per-key metrics →
  `forecasts/backtest_metrics.csv`.

## Files
- Real caches: `real_cache/weather_cache.csv`, `real_cache/fuel_prices.csv`
  (+ raw `ppac_rsp_raw.csv`).
- Rebuilt dataset: `code/output/` (orders, invoices, fuel_prices, weather_cache,
  _ground_truth_audit, reference tables).
- Forecasts: `real_cache/forecasts/{fuel_forecast,weather_forecast,backtest_metrics}.csv`.
- Changed code: `extraction_scripts/run_openmeteo.py`,
  `extraction_scripts/load_ppac_diesel.py`, `code/build_dataset.py`,
  `code/scripts/weather_fuel.py`, new `code/forecast_series.py`.
