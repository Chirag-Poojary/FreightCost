"""
Time-series forecasters (SARIMA / statsmodels) on the REAL historical series that
feed the freight pipeline:

  FUEL    -- per-state daily diesel retail price  (real_cache/fuel_prices.csv)
             Diesel in 2023-25 is a near-random-walk (OMCs held prices), so a
             non-seasonal ARIMA(1,1,1) is the right model; we forecast the next
             `--fuel-horizon` days and backtest on a held-out tail.

  WEATHER -- per-hub weekly-aggregated series      (real_cache/weather_cache.csv)
             Strong annual seasonality, so SARIMA with seasonal period m=52.
             We model weekly-total precipitation and weekly-mean tmax, backtest
             on the last `--wx-test-weeks` weeks, and forecast `--wx-horizon`.

For every series we report a walk-forward-style holdout backtest (train on all
but the tail, forecast the tail, score MAE / MAPE / RMSE against the real tail),
then refit on the full series to emit a genuine forward forecast.

Usage:
  python forecast_series.py --fuel ../real_cache/fuel_prices.csv \
     --weather ../real_cache/weather_cache.csv --outdir ../real_cache/forecasts
"""
import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _scores(actual, pred, baseline=None):
    """MAE / RMSE / sMAPE, plus skill vs a naive baseline when supplied.

    sMAPE (symmetric) is bounded 0-200% and, unlike MAPE, does not explode on
    intermittent series with near-zero actuals (e.g. weekly precipitation).
    skill = 1 - MAE_model / MAE_baseline  (>0 means the model beats the baseline;
    baseline is persistence for fuel, seasonal-naive/'same week last year' for
    weather).
    """
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    m = ~(np.isnan(a) | np.isnan(p))
    a, p = a[m], p[m]
    if len(a) == 0:
        return dict(mae=np.nan, smape=np.nan, rmse=np.nan, skill=np.nan, n=0)
    err = p - a
    smape = np.mean(2 * np.abs(err) / np.clip(np.abs(a) + np.abs(p), 1e-6, None)) * 100
    out = dict(mae=float(np.mean(np.abs(err))), smape=float(smape),
               rmse=float(np.sqrt(np.mean(err ** 2))), n=int(len(a)))
    if baseline is not None:
        b = np.asarray(baseline, float)[m]
        bmae = np.mean(np.abs(b - a))
        out["skill"] = float(1 - out["mae"] / bmae) if bmae > 1e-9 else np.nan
    else:
        out["skill"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# FUEL: per-state daily diesel  ->  ARIMA(1,1,1)
# --------------------------------------------------------------------------- #
def forecast_fuel(fuel_csv, outdir, horizon, test_days, order=(1, 1, 1)):
    df = pd.read_csv(fuel_csv, parse_dates=["price_date"])
    rows_metrics, rows_fc = [], []
    for state, g in df.groupby("state"):
        s = (g.set_index("price_date")["diesel_price_per_litre"]
             .asfreq("D").ffill())
        if len(s) < test_days + 30:
            continue
        train, test = s.iloc[:-test_days], s.iloc[-test_days:]
        # persistence baseline: hold the last training value across the horizon
        base = np.full(test_days, float(train.iloc[-1]))
        try:
            bt = ARIMA(train, order=order).fit()
            pred = bt.forecast(steps=test_days)
            sc = _scores(test.values, pred.values, baseline=base)
            full = ARIMA(s, order=order).fit()
            fc = full.forecast(steps=horizon)
        except Exception as e:
            print(f"  [fuel] {state}: FAILED ({e})")
            continue
        sc.update(series="fuel_diesel", key=state, order=str(order),
                  last_price=round(float(s.iloc[-1]), 2),
                  test_var=round(float(test.std()), 3))
        rows_metrics.append(sc)
        for d, v in fc.items():
            rows_fc.append(dict(series="fuel_diesel", key=state,
                                date=str(pd.Timestamp(d).date()),
                                forecast=round(float(v), 3)))
        print(f"  [fuel] {state:16s} MAE={sc['mae']:.3f}  sMAPE={sc['smape']:.2f}%  "
              f"last={sc['last_price']:.2f}  test_sd={sc['test_var']:.3f}")
    pd.DataFrame(rows_fc).to_csv(f"{outdir}/fuel_forecast.csv", index=False)
    return rows_metrics


# --------------------------------------------------------------------------- #
# WEATHER: per-hub weekly series  ->  SARIMA seasonal (m=52)
# --------------------------------------------------------------------------- #
def _hub_weekly(weather_csv):
    df = pd.read_csv(weather_csv)
    recs = []
    has_temp = "temperature_max_json" in df.columns
    for row in df.itertuples():
        hub = row.route_key.split("-")[-1]
        dates = pd.date_range(row.window_start, row.window_end, freq="D")
        precip = json.loads(row.precipitation_sum_json)
        temp = json.loads(row.temperature_max_json) if has_temp else [None]*len(precip)
        for i, d in enumerate(dates):
            if i < len(precip):
                recs.append((hub, d, precip[i],
                             temp[i] if i < len(temp) else None))
    daily = pd.DataFrame(recs, columns=["hub", "date", "precip", "temp"])
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.drop_duplicates(["hub", "date"]).set_index("date")
    out = {}
    for hub, g in daily.groupby("hub"):
        wk = g.resample("W").agg(precip=("precip", "sum"), temp=("temp", "mean"))
        out[hub] = wk
    return out


def forecast_weather(weather_csv, outdir, horizon, test_weeks,
                     order=(1, 0, 1), seasonal=(1, 1, 0, 52)):
    weekly = _hub_weekly(weather_csv)
    rows_metrics, rows_fc = [], []
    for hub, wk in weekly.items():
        for var in ["precip", "temp"]:
            s = wk[var].astype(float).interpolate().dropna()
            if len(s) < 2 * seasonal[3] * 0 + test_weeks + 60:
                # need > ~1 season of history; skip if too short
                if len(s) < test_weeks + 60:
                    continue
            train, test = s.iloc[:-test_weeks], s.iloc[-test_weeks:]
            # seasonal-naive baseline: value from the same week one year earlier
            m = seasonal[3]
            base = (s.shift(m).iloc[-test_weeks:].values
                    if len(s) > m + test_weeks else None)
            try:
                bt = SARIMAX(train, order=order, seasonal_order=seasonal,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)
                pred = bt.forecast(steps=test_weeks)
                sc = _scores(test.values, pred.values, baseline=base)
                full = SARIMAX(s, order=order, seasonal_order=seasonal,
                               enforce_stationarity=False,
                               enforce_invertibility=False).fit(disp=False)
                fc = full.forecast(steps=horizon)
            except Exception as e:
                print(f"  [wx] {hub}/{var}: FAILED ({e})")
                continue
            sc.update(series=f"weather_{var}", key=hub,
                      order=f"{order}x{seasonal}")
            rows_metrics.append(sc)
            for d, v in fc.items():
                rows_fc.append(dict(series=f"weather_{var}", key=hub,
                                    date=str(pd.Timestamp(d).date()),
                                    forecast=round(float(v), 3)))
            skill = sc.get("skill", float("nan"))
            print(f"  [wx] {hub:6s} {var:6s} MAE={sc['mae']:6.2f}  "
                  f"RMSE={sc['rmse']:6.2f}  sMAPE={sc['smape']:6.1f}%  "
                  f"skill_vs_seasonal_naive={skill:+.2f}")
    pd.DataFrame(rows_fc).to_csv(f"{outdir}/weather_forecast.csv", index=False)
    return rows_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuel", default="../real_cache/fuel_prices.csv")
    ap.add_argument("--weather", default="../real_cache/weather_cache.csv")
    ap.add_argument("--outdir", default="../real_cache/forecasts")
    ap.add_argument("--fuel-horizon", type=int, default=30)
    ap.add_argument("--fuel-test-days", type=int, default=60)
    ap.add_argument("--wx-horizon", type=int, default=26)
    ap.add_argument("--wx-test-weeks", type=int, default=26)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    print("=== FUEL (per-state daily diesel, ARIMA(1,1,1)) ===")
    fm = forecast_fuel(a.fuel, a.outdir, a.fuel_horizon, a.fuel_test_days)
    print("\n=== WEATHER (per-hub weekly, SARIMA (1,0,1)x(1,1,0,52)) ===")
    wm = forecast_weather(a.weather, a.outdir, a.wx_horizon, a.wx_test_weeks)

    metrics = pd.DataFrame(fm + wm)
    metrics.to_csv(f"{a.outdir}/backtest_metrics.csv", index=False)

    print("\n=== BACKTEST SUMMARY (median across keys) ===")
    for series, g in metrics.groupby("series"):
        skill = g["skill"].median() if "skill" in g else float("nan")
        print(f"  {series:16s} n_keys={len(g):3d}  "
              f"median MAE={g['mae'].median():7.3f}  "
              f"median RMSE={g['rmse'].median():7.3f}  "
              f"median sMAPE={g['smape'].median():6.2f}%  "
              f"median skill={skill:+.2f}")
    print(f"\nWrote forecasts + backtest_metrics.csv to {a.outdir}")


if __name__ == "__main__":
    main()
