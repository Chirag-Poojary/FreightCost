"""
REAL OPEN-METEO EXTRACTION  --  run this yourself to pull genuine historical
weather for every destination hub. Populates weather_cache.csv.

NO API KEY REQUIRED. Open-Meteo Archive API is free for non-commercial/research
use. Client-side rate-limit only (the doc recommends ~1 call/sec + backoff).

  Endpoint (historical/archive):  archive-api.open-meteo.com/v1/archive

WHY FULL-SERIES INSTEAD OF PER-WINDOW
-------------------------------------
The original design fetched one window per (dest_hub, ISO-week) -- ~2,800 calls.
That returns the SAME underlying daily observations as fetching each hub's full
2023-2025 daily series in ONE call, but 150x more HTTP requests AND only partial
date coverage (an order's transit window can spill onto days the dedup never
fetched). We instead fetch each dest hub's complete daily series (18 calls total,
~20s), which gives COMPLETE real coverage for every order's planned OR actual
window, is trivially resumable, and also captures temperature_2m_max (the old
script requested temp but never wrote it, silently zeroing heat_days scoring).

The output is still written in the documented weather_cache.csv schema -- one row
per (hub, ISO-week) with that week's daily arrays -- so build_dataset.py consumes
it unchanged. A temperature_max_json column is added for complete severity scoring.

RESUMABILITY
------------
Hubs already present in --out are skipped, and each hub is appended as soon as it
returns, so an interrupted run resumes where it stopped -- just re-run the command.

Usage:
  python run_openmeteo.py --hubs ../data/hubs.csv --out ../real_cache/weather_cache.csv \
                          --start 2023-01-01 --end 2026-01-31
"""
import argparse
import os
import time
import json
import requests
import pandas as pd

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_hub_series(lat, lon, start_date, end_date, retries=5):
    """Full daily precip/wind/temp series for one hub over [start, end]."""
    url = (f"{ARCHIVE}?latitude={lat}&longitude={lon}"
           f"&start_date={start_date}&end_date={end_date}"
           f"&daily=precipitation_sum,windspeed_10m_max,temperature_2m_max"
           f"&timezone=Asia%2FKolkata")
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=60).json()
            d = res["daily"]
            return (d["time"], d["precipitation_sum"],
                    d["windspeed_10m_max"], d.get("temperature_2m_max"))
        except Exception as e:
            wait = 2 ** attempt
            print(f"    retry {attempt+1}/{retries} after {wait}s ({e})", flush=True)
            time.sleep(wait)
    return None, None, None, None


def hub_series_to_week_rows(hub_id, times, precip, wind, temp):
    """Chunk a hub's daily series into documented (hub, ISO-week) cache rows."""
    df = pd.DataFrame({"date": pd.to_datetime(times),
                       "precip": precip, "wind": wind, "temp": temp})
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)
    rows = []
    for (iy, iw), g in df.groupby(["iso_year", "iso_week"]):
        g = g.sort_values("date")
        rows.append((
            f"{hub_id}-{hub_id}",                    # route_key: dest parses as hub
            str(g["date"].iloc[0].date()),
            str(g["date"].iloc[-1].date()),
            json.dumps([None if pd.isna(x) else x for x in g["precip"]]),
            json.dumps([None if pd.isna(x) else x for x in g["wind"]]),
            json.dumps([None if pd.isna(x) else x for x in g["temp"]]),
            pd.Timestamp.now().isoformat()))
    return rows


COLUMNS = ["route_key", "window_start", "window_end", "precipitation_sum_json",
           "windspeed_max_json", "temperature_max_json", "api_call_timestamp"]


def main(hubs_csv, out_csv, start, end, sleep_s):
    hubs = pd.read_csv(hubs_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    done_hubs = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done_hubs = {rk.split("-")[0] for rk in prev["route_key"]}
        print(f"Resuming: {len(done_hubs)} hubs already cached -> {sorted(done_hubs)}")
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(out_csv, index=False)

    for r in hubs.itertuples():
        if r.hub_id in done_hubs:
            continue
        print(f"[{r.hub_id}] {r.city} ({r.latitude},{r.longitude}) {start}..{end}",
              flush=True)
        times, precip, wind, temp = fetch_hub_series(
            r.latitude, r.longitude, start, end)
        if times is None:
            print(f"  !! FAILED {r.hub_id}; will retry on next run", flush=True)
            continue
        rows = hub_series_to_week_rows(r.hub_id, times, precip, wind, temp)
        pd.DataFrame(rows, columns=COLUMNS).to_csv(
            out_csv, mode="a", header=False, index=False)
        nsummary = sum(1 for p in precip if p is not None)
        print(f"  wrote {len(rows)} week-rows ({nsummary} real days)", flush=True)
        time.sleep(sleep_s)                       # be polite to the free API

    total = pd.read_csv(out_csv)
    print(f"DONE. {len(total)} week-rows across "
          f"{total['route_key'].str.split('-').str[0].nunique()} hubs -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs", default="../data/hubs.csv")
    ap.add_argument("--out", default="../real_cache/weather_cache.csv")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-01-31")
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    main(a.hubs, a.out, a.start, a.end, a.sleep)
