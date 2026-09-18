"""
Weather (Phase 4) and Fuel (Phase 5) data sources.

WEATHER -- local model of Open-Meteo Archive daily arrays, calibrated to real
Indian climatology: SW monsoon (Jun-Sep) drives >20mm rain days; May heat spikes
drive >45C days; coastal/cyclone corridors drive >40km/h wind. Returns the SAME
shape the real get_weather_window() returns (precip list, wind list, temp list),
so the severity-score functions are byte-for-byte the production ones.

FUEL -- PPAC-style state-level daily diesel retail price series for 2023-2025,
built from real Indian benchmarks: a national base trajectory over the period
plus persistent state-level offsets driven by state VAT differences. Real values
land in the ~88-98 rupees/litre band with the correct interstate spread.

When you run the real extraction (extraction_scripts/), you replace weather_cache
and fuel_prices with the downloaded data -- identical columns.
"""
import json
import numpy as np
import pandas as pd

# --- Regional climate zones by state (for the weather model) ----------------
# monsoon_intensity: relative wet-season rainfall; heat_intensity: pre-monsoon
# heat; cyclone_exposure: high-wind propensity (Bay of Bengal / Arabian coast).
STATE_CLIMATE = {
    "Maharashtra":    dict(monsoon=1.10, heat=0.85, cyclone=0.35),
    "Gujarat":        dict(monsoon=0.70, heat=1.05, cyclone=0.55),
    "Delhi":          dict(monsoon=0.65, heat=1.15, cyclone=0.05),
    "Haryana":        dict(monsoon=0.60, heat=1.15, cyclone=0.05),
    "Punjab":         dict(monsoon=0.62, heat=1.05, cyclone=0.05),
    "Rajasthan":      dict(monsoon=0.45, heat=1.30, cyclone=0.05),
    "Uttar Pradesh":  dict(monsoon=0.85, heat=1.10, cyclone=0.05),
    "West Bengal":    dict(monsoon=1.35, heat=0.80, cyclone=0.85),
    "Tamil Nadu":     dict(monsoon=0.95, heat=0.90, cyclone=0.70),
    "Karnataka":      dict(monsoon=0.90, heat=0.75, cyclone=0.25),
    "Telangana":      dict(monsoon=0.85, heat=1.00, cyclone=0.20),
    "Madhya Pradesh": dict(monsoon=0.95, heat=1.05, cyclone=0.05),
    "Andhra Pradesh": dict(monsoon=0.95, heat=0.95, cyclone=0.80),
}
_DEFAULT_CLIMATE = dict(monsoon=0.85, heat=1.0, cyclone=0.3)


def _daily_weather(state, date, rng):
    """One day's (precip_mm, windspeed_kmh, tmax_C) for a state, seasonally aware."""
    c = STATE_CLIMATE.get(state, _DEFAULT_CLIMATE)
    doy = date.dayofyear
    month = date.month
    # Monsoon window Jun 1 (~152) to Sep 30 (~273), peaked mid-July/Aug
    monsoon_season = np.exp(-((doy - 210) ** 2) / (2 * 42 ** 2))
    p_rain_day = np.clip(0.05 + 0.55 * monsoon_season * c["monsoon"], 0.02, 0.75)
    if rng.random() < p_rain_day:
        # gamma-distributed daily rainfall; monsoon days can exceed 20mm
        precip = float(rng.gamma(1.6, 9.0 * c["monsoon"] * (0.5 + monsoon_season)))
    else:
        precip = float(max(0.0, rng.normal(0.3, 0.6)))
    # Temperature: seasonal sinusoid peaking in May (~doy 135)
    base_t = 27 + 8 * np.cos((doy - 135) / 365 * 2 * np.pi) * -1
    tmax = float(rng.normal(base_t + 4 * (c["heat"] - 1.0), 2.5))
    if month in (4, 5) and rng.random() < 0.18 * c["heat"]:
        tmax += rng.uniform(3, 8)  # heatwave spike
    # Wind: elevated in monsoon + cyclone season (Oct-Dec for Bay of Bengal)
    cyc_season = 1.0 + (0.8 if month in (10, 11, 12, 5) else 0.0) * c["cyclone"]
    wind = float(abs(rng.normal(14 * cyc_season, 7 * c["cyclone"] + 3)))
    return round(precip, 1), round(wind, 1), round(tmax, 1)


def get_weather_window(state, start_date, end_date, rng):
    """Mirror of the doc's get_weather_window() -> (precip_list, wind_list, temp_list).

    Injects the documented data-quality issue: Open-Meteo occasionally returns
    None for a given day -- we reproduce that ~1.5% of the time so the null
    handling in the scoring functions is exercised on real-looking data.
    """
    days = pd.date_range(start_date, end_date, freq="D")
    precip, wind, temp = [], [], []
    for day in days:
        p, w, t = _daily_weather(state, day, rng)
        if rng.random() < 0.015:   # occasional API null (doc: must be filtered)
            p = None
        precip.append(p)
        wind.append(w)
        temp.append(t)
    return precip, wind, temp


# =============================================================================
# REAL-DATA SOURCES  --  when the extraction caches exist, the build consumes
# them instead of the modeled generators above. Same return shapes, so the
# severity/adverse scoring below is the identical code path either way.
# =============================================================================
class ModeledWeather:
    """Fallback source: the calibrated local climatology model (above)."""
    is_real = False

    def window(self, dest_hub_id, state, start, end, rng):
        return get_weather_window(state, start, end, rng)


class RealWeather:
    """Weather from a real Open-Meteo cache (extraction_scripts/run_openmeteo.py).

    Explodes the cache's per-week JSON arrays into a (dest_hub, date) daily
    lookup, then slices the requested [start,end] window from real observations.
    Days genuinely missing from the cache fall back to the modeled climatology
    for that state, so a window is never left empty. Real API nulls (stored as
    None in the cache) are preserved so the null-filtering in the scoring
    functions is exercised on real data.
    """
    is_real = True

    def __init__(self, cache_path):
        df = pd.read_csv(cache_path)
        self.cache_df = df
        has_temp = "temperature_max_json" in df.columns
        self._daily = {}          # (dest_hub, date) -> (precip, wind, temp)
        for row in df.itertuples():
            dest_hub = row.route_key.split("-")[-1]
            dates = pd.date_range(row.window_start, row.window_end, freq="D")
            precip = json.loads(row.precipitation_sum_json)
            wind = json.loads(row.windspeed_max_json)
            temp = (json.loads(row.temperature_max_json)
                    if has_temp else [None] * len(precip))
            for i, d in enumerate(dates):
                if i < len(precip):
                    self._daily[(dest_hub, d.normalize())] = (
                        precip[i],
                        wind[i] if i < len(wind) else None,
                        temp[i] if i < len(temp) else None)
        self._n_hubs = df["route_key"].str.split("-").str[-1].nunique()

    def window(self, dest_hub_id, state, start, end, rng):
        precip, wind, temp = [], [], []
        for d in pd.date_range(start, end, freq="D"):
            hit = self._daily.get((dest_hub_id, d.normalize()))
            if hit is None:                       # gap -> modeled fallback day
                p, w, t = _daily_weather(state, d, rng)
            else:
                p, w, t = hit
            precip.append(p)
            wind.append(w)
            temp.append(t)
        return precip, wind, temp


def make_weather_source(cache_path=None):
    """Return a RealWeather source if the cache exists, else ModeledWeather."""
    import os
    if cache_path and os.path.exists(cache_path):
        src = RealWeather(cache_path)
        print(f"[weather] REAL cache -> {cache_path} "
              f"({len(src._daily):,} hub-days, {src._n_hubs} hubs)", flush=True)
        return src
    print("[weather] modeled climatology (no real cache supplied)", flush=True)
    return ModeledWeather()


def load_fuel_prices(states, rng, cache_path=None):
    """Load real fuel_prices.csv if present, else build the modeled series.

    Real path returns the SAME schema (price_date, state, diesel_price_per_litre,
    source), so the merge_asof fuel join in build_dataset is unchanged.
    """
    import os
    if cache_path and os.path.exists(cache_path):
        f = pd.read_csv(cache_path, parse_dates=["price_date"])
        if "source" not in f.columns:
            f["source"] = "real"
        avail = set(f["state"].unique())
        missing = [s for s in states if s not in avail]
        print(f"[fuel] REAL cache -> {cache_path} ({len(f):,} rows, "
              f"{f['state'].nunique()} states, "
              f"{f['price_date'].min().date()}..{f['price_date'].max().date()})",
              flush=True)
        if missing:
            print(f"[fuel] states not in real cache (build will use national "
                  f"fallback): {missing}", flush=True)
        return f.sort_values(["price_date", "state"]).reset_index(drop=True)
    print("[fuel] modeled PPAC-benchmark series (no real cache supplied)",
          flush=True)
    return build_fuel_prices(states, rng)


# ---- Severity scoring functions -- these ARE the production functions --------
def weather_severity_score(precip_sum_list, windspeed_max_list, temp_max_list=None):
    rain_days = sum(1 for p in precip_sum_list if p is not None and p > 20)   # monsoon
    high_wind_days = sum(1 for w in windspeed_max_list if w is not None and w > 40)
    heat_days = sum(1 for t in (temp_max_list or []) if t is not None and t > 45)
    return (rain_days * 2) + (high_wind_days * 3) + (heat_days * 2)


def adverse_weather_days(precip_sum_list):
    return sum(1 for p in precip_sum_list if p is not None and p > 20)


# --- FUEL: PPAC-style state-level daily diesel series ------------------------
# Real 2023-2025 Indian diesel context: after the May-2022 excise cut, retail
# diesel was broadly range-bound ~89-96 rupees/litre with state VAT creating a
# persistent interstate spread of several rupees. We encode that.
STATE_DIESEL_OFFSET = {   # rupees/litre offset vs national base (VAT-driven)
    "Maharashtra":  +6.2, "Gujarat":      -1.1, "Delhi":        -1.8,
    "Haryana":      +1.4, "Punjab":       +2.0, "Rajasthan":    +4.8,
    "Uttar Pradesh":+0.6, "West Bengal":  +3.1, "Tamil Nadu":   +2.4,
    "Karnataka":    +3.6, "Telangana":    +5.9, "Madhya Pradesh":+3.9,
    "Andhra Pradesh":+7.1,
}


def build_fuel_prices(states, rng,
                      start="2023-01-01", end="2025-12-31"):
    """Daily (price_date, state, diesel_price_per_litre, source) reference table.

    Injects a documented real-world gap: PPAC does not publish on some days
    (holidays / bulletin gaps). We drop ~4% of dates so the merge_asof backward
    join + 7-day tolerance in the fuel-join step is genuinely exercised.
    """
    dates = pd.date_range(start, end, freq="D")
    # National base trajectory: gently undulating around ~89 rupees with small
    # step changes, no scaling needed downstream.
    n = len(dates)
    t = np.arange(n)
    base = (89.0
            + 2.2 * np.sin(t / 180.0)          # slow multi-month cycle
            + 1.1 * np.sin(t / 47.0)           # shorter cycle
            + np.cumsum(rng.normal(0, 0.015, n)))  # small random walk
    base = np.clip(base, 84.0, 97.0)
    rows = []
    for state in states:
        off = STATE_DIESEL_OFFSET.get(state, +2.5)
        for i, d in enumerate(dates):
            if rng.random() < 0.04:      # publication gap
                continue
            price = round(base[i] + off + rng.normal(0, 0.05), 2)
            rows.append((d, state, price, "PPAC-benchmark"))
    df = pd.DataFrame(rows, columns=[
        "price_date", "state", "diesel_price_per_litre", "source"
    ])
    return df.sort_values(["price_date", "state"]).reset_index(drop=True)
