"""
Phase 3 -- Routing.  Builds route_distance_cache keyed by (origin, dest) hub pair.

Per the reference doc, distance_km and ideal_days both come from ONE OSRM call
per unique hub pair, cached and reused across all order rows.

Local stand-in (used because the sandbox cannot reach an OSRM host):
  road_distance = haversine * road_circuity_factor
The circuity factor (~1.25-1.45) is the well-documented ratio between straight-
line and real Indian road distance; it is sampled per corridor with a fixed seed
so the cache is deterministic. When you run the real OSRM extraction
(extraction_scripts/run_osrm.py) you replace this cache file 1:1 -- same columns.

ideal_days = (osrm_duration_seconds / 86400) * 1.3   [TRAFFIC_BUFFER]
osrm_duration here is modeled from distance and a corridor speed band; the real
extraction takes it straight from OSRM's `duration` field.
"""
import numpy as np
import pandas as pd
from itertools import combinations

TRAFFIC_BUFFER = 1.3  # Indian highway congestion/infrastructure adjustment (doc)
EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def build_route_cache(hubs, rng, refreshed_at="2025-12-31"):
    """One row per unique unordered hub pair (matches OSRM dedup strategy)."""
    h = hubs.set_index("hub_id")
    rows = []
    for a, b in combinations(hubs["hub_id"], 2):
        la, lo = h.loc[a, "latitude"], h.loc[a, "longitude"]
        lb, lob = h.loc[b, "latitude"], h.loc[b, "longitude"]
        straight = haversine_km(la, lo, lb, lob)
        if straight < 1:
            continue
        # circuity: shorter corridors are relatively more indirect
        circuity = np.clip(rng.normal(1.34, 0.06) + (200 / max(straight, 50)) * 0.03,
                           1.20, 1.55)
        dist = round(straight * circuity, 1)
        # effective highway speed band (km/h), heavy-truck realistic for India
        speed = rng.uniform(38, 52)
        osrm_duration_s = (dist / speed) * 3600.0
        osrm_days = osrm_duration_s / 86400.0
        ideal_days = round(osrm_days * TRAFFIC_BUFFER, 3)
        rows.append((a, b, dist, round(osrm_days, 4), ideal_days,
                     round(straight, 1), refreshed_at))
    cache = pd.DataFrame(rows, columns=[
        "origin_hub_id", "dest_hub_id", "distance_km", "osrm_duration_days",
        "buffered_ideal_days", "_haversine_km", "last_refreshed"
    ])
    # Data-quality guard from the doc: distance must exceed haversine lower bound
    bad = cache["distance_km"] < cache["_haversine_km"]
    if bad.any():
        raise ValueError(f"{bad.sum()} routes below haversine floor -- bad coords")
    return cache


def load_route_cache(hubs, rng, cache_path=None):
    """Real cache if supplied (extraction_scripts/run_ors.py or run_osrm.py),
    else the modeled haversine*circuity stand-in. Same swap-in pattern as
    weather_fuel.make_weather_source() / load_fuel_prices().

    The real extraction scripts don't emit `_haversine_km` (write_outputs()
    drops it before saving anyway) -- we recompute it here from hub
    coordinates so the interface matches build_route_cache()'s output exactly
    and the downstream haversine-floor guard still works.
    """
    import os
    if cache_path and os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        h = hubs.set_index("hub_id")
        cache["_haversine_km"] = [
            haversine_km(h.loc[r.origin_hub_id, "latitude"],
                        h.loc[r.origin_hub_id, "longitude"],
                        h.loc[r.dest_hub_id, "latitude"],
                        h.loc[r.dest_hub_id, "longitude"])
            for r in cache.itertuples()]
        bad = cache["distance_km"] < cache["_haversine_km"]
        if bad.any():
            raise ValueError(f"{bad.sum()} real routes below haversine floor -- "
                             f"check the extraction output")
        print(f"[routing] REAL cache -> {cache_path} "
              f"({len(cache)} routes)", flush=True)
        return cache
    print("[routing] modeled haversine x circuity (no real cache supplied)",
          flush=True)
    return build_route_cache(hubs, rng)


def lookup(cache, o, d):
    """Directionless lookup for an ordered (origin,dest) request."""
    m = cache[((cache.origin_hub_id == o) & (cache.dest_hub_id == d)) |
              ((cache.origin_hub_id == d) & (cache.dest_hub_id == o))]
    if m.empty:
        return None
    r = m.iloc[0]
    return float(r.distance_km), float(r.buffered_ideal_days)
