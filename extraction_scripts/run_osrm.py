"""
REAL OSRM EXTRACTION  --  run this yourself against a self-hosted OSRM instance.
Produces route_distance_cache.csv with the SAME columns the pipeline expects,
so you can drop it straight into output/ and rebuild with real road distances.

NO API KEY REQUIRED. OSRM is self-hosted (free). Do NOT use the public demo
server (router.project-osrm.org) -- it is not licensed for bulk use.

--- One-time OSRM setup (Docker + free Geofabrik India extract) ---
  wget https://download.geofabrik.de/asia/india-latest.osm.pbf
  docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/india-latest.osm.pbf
  docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/india-latest.osrm
  docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/india-latest.osrm
  docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/india-latest.osrm

Then:  python run_osrm.py --hubs ../output/hubs.csv --out ../output/route_distance_cache.csv
"""
import argparse
import time
from itertools import combinations
import requests
import pandas as pd

TRAFFIC_BUFFER = 1.3


def get_route(lon1, lat1, lon2, lat2, base="http://localhost:5000"):
    url = f"{base}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    r = requests.get(url, timeout=10).json()
    if r.get("code") == "Ok":
        dist_km = r["routes"][0]["distance"] / 1000
        dur_days = r["routes"][0]["duration"] / 86400
        return dist_km, dur_days
    return None, None


def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * asin(sqrt(a))


def main(hubs_csv, out_csv, base_url):
    hubs = pd.read_csv(hubs_csv).set_index("hub_id")
    rows, dropped = [], []
    for a, b in combinations(hubs.index, 2):
        la, lo = hubs.loc[a, "latitude"], hubs.loc[a, "longitude"]
        lb, lob = hubs.loc[b, "latitude"], hubs.loc[b, "longitude"]
        dist, dur = get_route(lo, la, lob, lb, base_url)
        if dist is None:
            dropped.append((a, b)); continue
        floor = haversine_km(la, lo, lb, lob)
        if dist < floor:                     # bad routing / coords -> flag
            dropped.append((a, b)); continue
        rows.append((a, b, round(dist, 1), round(dur, 4),
                     round(dur * TRAFFIC_BUFFER, 3),
                     time.strftime("%Y-%m-%d")))
        time.sleep(0.05)                     # be gentle even when self-hosted
    cache = pd.DataFrame(rows, columns=[
        "origin_hub_id", "dest_hub_id", "distance_km", "osrm_duration_days",
        "buffered_ideal_days", "last_refreshed"])
    cache.to_csv(out_csv, index=False)
    print(f"Wrote {len(cache)} routes to {out_csv}; dropped {len(dropped)} pairs.")
    if dropped:
        print("Dropped (no route / below haversine floor):", dropped)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hubs", default="../output/hubs.csv")
    ap.add_argument("--out", default="../output/route_distance_cache.csv")
    ap.add_argument("--base_url", default="http://localhost:5000")
    a = ap.parse_args()
    main(a.hubs, a.out, a.base_url)
