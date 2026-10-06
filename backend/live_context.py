"""
Live external context for Model 1 / Model 2 -- weather and diesel price for a
specific route and date, fetched at request time instead of read from the
2023-2025 training caches.

Why: the models were trained on diesel at Rs 86.6-97.4/L. Today several states
are above Rs 100/L. Feeding a stale cached price hides that drift; feeding the
live price and flagging that it is outside the training range makes it visible.

Sources (all free, no key):
  weather  Open-Meteo archive API   (dates older than ~6 days)
           Open-Meteo forecast API  (last ~3 months through +15 days)
           same dates last year     (further in the future -- climatology proxy)
  diesel   PPAC daily cache         (dates covered by the training cache)
           goodreturns.in state page (today / recent / future -- scraped, cached 6h)
           latest cached PPAC price  (if the live scrape fails)

Every value returned carries a `source` string so the UI can say exactly where
it came from. Nothing here raises -- a failed source degrades to the next one.
"""
import html
import os
import re
import threading
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

_BUNDLE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_DIRS = [os.path.join(_BUNDLE, "code", "output"), os.path.join(_BUNDLE, "data")]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DIESEL_URL = "https://www.goodreturns.in/diesel-price.html"
UA = {"user-agent": "Mozilla/5.0 (FreightCost audit; +https://freight-cost-nine.vercel.app)"}

_lock = threading.Lock()
_cache = {}            # key -> (expires_at, value)


def _memo(key, ttl, fn):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    val = fn()
    if val is not None:
        with _lock:
            _cache[key] = (now + ttl, val)
    return val


def _find(name):
    for d in _DATA_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------- reference data
_hubs = None
_fuel = None
_train = None
_routes = None


def hubs():
    global _hubs
    if _hubs is None:
        _hubs = pd.read_csv(_find("hubs.csv")).set_index("hub_id")
    return _hubs


def routes():
    global _routes
    if _routes is None:
        p = _find("route_distance_cache.csv")
        _routes = pd.read_csv(p).set_index(["origin_hub_id", "dest_hub_id"]) if p else pd.DataFrame()
    return _routes


def fuel_cache():
    global _fuel
    if _fuel is None:
        f = pd.read_csv(_find("fuel_prices.csv"), parse_dates=["price_date"])
        _fuel = f.sort_values("price_date")
    return _fuel


def training_ranges():
    """Feature ranges the models actually saw -- the yardstick for drift."""
    global _train
    if _train is None:
        o = pd.read_csv(_find("orders.csv"),
                        usecols=["expected_fuel_price", "expected_weather_score",
                                 "distance_km", "billable_weight_kg"])
        _train = {c: {"min": round(float(o[c].min()), 2), "max": round(float(o[c].max()), 2),
                      "p01": round(float(o[c].quantile(0.01)), 2),
                      "p99": round(float(o[c].quantile(0.99)), 2)}
                  for c in o.columns}
    return _train


def drift_check(feature, value):
    """None if inside the training range, else a human-readable warning."""
    if value is None:
        return None
    r = training_ranges().get(feature)
    if not r:
        return None
    if value > r["max"] or value < r["min"]:
        return (f"{feature} = {value} is outside the training range [{r['min']}, {r['max']}]. "
                f"Model 1 still scales its fuel/toll/driver baseline correctly, but its learned "
                f"correction is held at the nearest training value. Retrain once enough recent "
                f"data has been collected.")
    return None


def route_info(origin_hub_id, dest_hub_id):
    try:
        r = routes().loc[(origin_hub_id, dest_hub_id)]
        return {"distance_km": float(r["distance_km"]),
                "ideal_days": float(r["buffered_ideal_days"])}
    except Exception:
        return None


def _to_date(d):
    if d is None or d == "":
        return date.today()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    s = str(d).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return pd.to_datetime(s, dayfirst=True).date()


# ------------------------------------------------------------------- weather
def weather_severity_score(precip, wind, temp):
    # Identical to code/scripts/weather_fuel.py -- the function the model was trained on.
    rain = sum(1 for p in precip if p is not None and p > 20)
    windy = sum(1 for w in wind if w is not None and w > 40)
    heat = sum(1 for t in (temp or []) if t is not None and t > 45)
    return (rain * 2) + (windy * 3) + (heat * 2)


def _open_meteo(url, lat, lon, start, end):
    params = {"latitude": lat, "longitude": lon,
              "start_date": start.isoformat(), "end_date": end.isoformat(),
              "daily": "precipitation_sum,windspeed_10m_max,temperature_2m_max",
              "timezone": "Asia/Kolkata"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    d = r.json()["daily"]
    return d["time"], d["precipitation_sum"], d["windspeed_10m_max"], d.get("temperature_2m_max")


def _daily_series(hub_id, start, end):
    """Daily (precip, wind, temp) for a hub over [start, end], stitched from
    whichever Open-Meteo endpoint covers each part of the window."""
    h = hubs().loc[hub_id]
    lat, lon = float(h["latitude"]), float(h["longitude"])
    today = date.today()
    archive_end = today - timedelta(days=6)
    forecast_end = today + timedelta(days=15)
    forecast_start = today - timedelta(days=90)

    out, sources = {}, set()

    def add(url, s, e, label, shift_years=0):
        if s > e:
            return
        times, p, w, t = _open_meteo(url, lat, lon, s, e)
        for i, ts in enumerate(times):
            dd = date.fromisoformat(ts)
            if shift_years:
                try:
                    dd = dd.replace(year=dd.year + shift_years)
                except ValueError:      # 29 Feb
                    continue
            out[dd] = (p[i], w[i], t[i] if t else None)
        sources.add(label)

    # 1) archive for the settled past
    add(ARCHIVE_URL, start, min(end, archive_end), "Open-Meteo archive (observed)")
    # 2) forecast endpoint for recent past + next 15 days
    add(FORECAST_URL, max(start, archive_end + timedelta(days=1), forecast_start),
        min(end, forecast_end), "Open-Meteo forecast")
    # 3) beyond the forecast horizon: same dates last year as a climatology proxy
    if end > forecast_end:
        s = max(start, forecast_end + timedelta(days=1))
        add(ARCHIVE_URL, s.replace(year=s.year - 1), end.replace(year=end.year - 1),
            "Open-Meteo climatology (same dates last year)", shift_years=1)
    return out, sorted(sources)


def get_weather(dest_hub_id, start, end):
    """Weather severity score + adverse days for a hub over a date window."""
    start, end = _to_date(start), _to_date(end)
    if end < start:
        start, end = end, start
    end = min(end, start + timedelta(days=45))       # sanity cap

    def fetch():
        try:
            series, sources = _daily_series(dest_hub_id, start, end)
        except Exception as e:
            return {"ok": False, "error": str(e)[:200]}
        days = [series[d] for d in sorted(series) if start <= d <= end]
        if not days:
            return {"ok": False, "error": "no weather data for window"}
        precip = [d[0] for d in days]
        wind = [d[1] for d in days]
        temp = [d[2] for d in days]
        return {
            "ok": True,
            "score": weather_severity_score(precip, wind, temp),
            "adverse_weather_days": sum(1 for p in precip if p is not None and p > 20),
            "rain_mm_total": round(sum(p for p in precip if p is not None), 1),
            "max_wind_kmh": max((w for w in wind if w is not None), default=None),
            "max_temp_c": max((t for t in temp if t is not None), default=None),
            "days": len(days),
            "source": " + ".join(sources),
        }

    res = _memo(("wx", dest_hub_id, start, end), 3 * 3600, fetch)
    res = dict(res)
    res.update({"hub_id": dest_hub_id, "start": start.isoformat(), "end": end.isoformat()})
    if not res.get("ok"):
        res.update({"score": None, "adverse_weather_days": None, "source": "unavailable"})
    return res


# -------------------------------------------------------------------- diesel
_STATE_ALIASES = {"chhattisgarh": "chhatisgarh", "delhi ncr": "delhi"}


def _scrape_live_diesel():
    """{state_lower: price} from goodreturns' all-states table."""
    try:
        r = requests.get(DIESEL_URL, headers=UA, timeout=20)
        r.raise_for_status()
    except Exception:
        return None
    prices = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", r.text, re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        if len(cells) >= 2:
            m = re.search(r"(\d{2,3}\.\d{1,2})", cells[1])
            if m and 50 < float(m.group(1)) < 200:
                prices.setdefault(cells[0].lower(), float(m.group(1)))
    return prices or None


def live_diesel_table():
    return _memo(("diesel_live",), 6 * 3600, _scrape_live_diesel)


def get_diesel(state, on_date=None):
    d = _to_date(on_date)
    f = fuel_cache()
    cache_end = f["price_date"].max().date()
    st = f[f["state"] == state]

    # Historical date inside the PPAC training cache: use the real daily series.
    if d <= cache_end and not st.empty:
        hist = st[st["price_date"].dt.date <= d]
        if not hist.empty:
            row = hist.iloc[-1]
            return {"ok": True, "price": float(row["diesel_price_per_litre"]),
                    "as_of": row["price_date"].date().isoformat(),
                    "source": f"PPAC daily cache ({row.get('source', 'PPAC')})",
                    "state": state}

    # Recent / today / future: live retail price (prices are sticky, so today's
    # price is the best available estimate for a near-future shipment too).
    table = live_diesel_table()
    key = _STATE_ALIASES.get(state.lower(), state.lower())
    if table and key in table:
        note = "" if d <= date.today() else " (today's price used for a future date)"
        if d < date.today() - timedelta(days=7):
            note = " (no daily history for this date; today's price used)"
        return {"ok": True, "price": table[key], "as_of": date.today().isoformat(),
                "source": "goodreturns.in live retail price" + note, "state": state}

    # State missing from the live table (e.g. Andhra Pradesh): carry its last
    # cached price forward by the nationwide change seen in the states we do have.
    if table and not st.empty:
        last = f[f["price_date"] == f["price_date"].max()]
        common = [s for s in last["state"] if s.lower() in table]
        if common:
            then = last[last["state"].isin(common)]["diesel_price_per_litre"].mean()
            now = sum(table[s.lower()] for s in common) / len(common)
            price = round(float(st.iloc[-1]["diesel_price_per_litre"] + now - then), 2)
            return {"ok": True, "price": price, "as_of": date.today().isoformat(),
                    "source": (f"derived: last PPAC price for {state} + live nationwide "
                               f"change ({now - then:+.2f} Rs/L across {len(common)} states)"),
                    "state": state}

    # Fallback: last known cached price.
    if not st.empty:
        row = st.iloc[-1]
        return {"ok": True, "price": float(row["diesel_price_per_litre"]),
                "as_of": row["price_date"].date().isoformat(),
                "source": "last cached PPAC price (live source unavailable -- may be stale)",
                "state": state}
    return {"ok": False, "price": None, "source": "unavailable", "state": state}


# ------------------------------------------------------------------- combined
def shipment_context(origin_hub_id, dest_hub_id, ship_date=None, transit_days=None):
    """Everything the Quote page needs to auto-fill, with sources and drift flags."""
    hub = hubs()
    if origin_hub_id not in hub.index or dest_hub_id not in hub.index:
        raise ValueError("unknown hub id")
    start = _to_date(ship_date)
    route = route_info(origin_hub_id, dest_hub_id)
    days = transit_days or (route["ideal_days"] * 1.3 if route else 2)
    end = start + timedelta(days=max(1, int(round(days + 0.49))))

    # Fuel is bought at the origin; weather that matters is along the route --
    # training used the destination hub's weather, so we keep that convention.
    fuel = get_diesel(hub.loc[origin_hub_id, "state"], start)
    wx = get_weather(dest_hub_id, start, end)

    warnings = [w for w in (drift_check("expected_fuel_price", fuel.get("price")),
                            drift_check("expected_weather_score", wx.get("score"))) if w]
    return {"route": route, "fuel": fuel, "weather": wx,
            "drift_warnings": warnings, "training_ranges": training_ranges()}
