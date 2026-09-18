"""
REAL ROAD-DISTANCE EXTRACTION via OpenRouteService (ORS) -- no self-hosting.
Produces route_distance_cache.csv with the SAME columns run_osrm.py produces,
so it plugs into build_dataset.py's --route-cache argument unchanged.

Only 153 unique hub pairs are needed (one call per pair, cached and reused
across all order rows) -- this comfortably fits inside ORS's free daily quota.

--- One-time setup ---
  1. Sign up free, no card required: https://openrouteservice.org/dev/#/signup
  2. Copy your API key from the dashboard.
  3. export ORS_API_KEY="your-key-here"      (or pass --api-key)

Then:  python run_ors.py --hubs ../data/hubs.csv --out ../data/route_distance_cache.csv

Uses the driving-hgv (heavy goods vehicle) profile by default -- more
realistic for freight trucks than driving-car, since it respects truck
restrictions where OSM tagging supports them. If your account doesn't have
hgv enabled, pass --profile driving-car.

Free-tier note: the directions endpoint's default daily quota is on the
order of ~2,000 requests/day (check your ORS dashboard for your current
number -- it varies by plan/version). 153 requests is nowhere near that
limit, but this script still rate-limits itself conservatively (~35
requests/minute) so a single run never risks a 429.
"""
import argparse
import os
import time
from itertools import combinations
from math import radians, sin, cos, asin, sqrt

import requests
import pandas as pd
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


TRAFFIC_BUFFER = 1.3
MIN_SECONDS_BETWEEN_CALLS = 1.7   # keeps us under ~35 req/min, safely under ORS limits


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(a))


def get_route(lon1, lat1, lon2, lat2, api_key, profile):
    url = f"https://api.openrouteservice.org/v2/directions/{profile}"
    params = {"api_key": api_key, "start": f"{lon1},{lat1}", "end": f"{lon2},{lat2}"}
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        print(f"  [warn] {r.status_code} for ({lat1},{lon1})->({lat2},{lon2}): "
              f"{r.text[:200]}")
        return None, None
    body = r.json()
    try:
        summary = body["features"][0]["properties"]["summary"]
        dist_km = summary["distance"] / 1000.0
        dur_days = summary["duration"] / 86400.0
        return dist_km, dur_days
    except (KeyError, IndexError):
        print(f"  [warn] unexpected response shape: {str(body)[:200]}")
        return None, None


def main(hubs_csv, out_csv, api_key, profile):
    if not api_key:
        raise SystemExit(
            "No ORS API key. Set ORS_API_KEY env var or pass --api-key. "
            "Get a free key at https://openrouteservice.org/dev/#/signup")

    hubs = pd.read_csv(hubs_csv).set_index("hub_id")
    pairs = list(combinations(hubs.index, 2))
    print(f"Fetching {len(pairs)} unique hub-pair routes from ORS "
          f"(profile={profile})...")

    rows, dropped = [], []
    for i, (a, b) in enumerate(pairs, 1):
        la, lo = hubs.loc[a, "latitude"], hubs.loc[a, "longitude"]
        lb, lob = hubs.loc[b, "latitude"], hubs.loc[b, "longitude"]

        t0 = time.time()
        dist, dur = get_route(lo, la, lob, lb, api_key, profile)
        if dist is None:
            dropped.append((a, b))
        else:
            floor = haversine_km(la, lo, lb, lob)
            if dist < floor:                     # bad routing / coords -> flag
                dropped.append((a, b))
            else:
                rows.append((a, b, round(dist, 1), round(dur, 4),
                             round(dur * TRAFFIC_BUFFER, 3),
                             time.strftime("%Y-%m-%d")))

        if i % 20 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} routes fetched "
                  f"({len(rows)} ok, {len(dropped)} dropped)")

        elapsed = time.time() - t0
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    cache = pd.DataFrame(rows, columns=[
        "origin_hub_id", "dest_hub_id", "distance_km", "osrm_duration_days",
        "buffered_ideal_days", "last_refreshed"])
    cache.to_csv(out_csv, index=False)
    print(f"\nWrote {len(cache)} routes to {out_csv}; dropped {len(dropped)} pairs.")
    if dropped:
        print("Dropped (no route / below haversine floor):", dropped)


if __name__ == "__main__":
    default_hubs = "../data/hubs.csv" if os.path.exists("../data/hubs.csv") else "../output/hubs.csv"
    default_out = "../data/route_distance_cache.csv" if os.path.exists("../data") else "../output/route_distance_cache.csv"
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs", default=default_hubs)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--api-key", default=os.environ.get("ORS_API_KEY"))
    ap.add_argument("--profile", default="driving-hgv",
                    help="ORS routing profile, e.g. driving-hgv or driving-car")
    a = ap.parse_args()
    main(a.hubs, a.out, a.api_key, a.profile)

