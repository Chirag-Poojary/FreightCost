"""
REAL DIESEL PRICE LOADER  --  normalise a PPAC / IOC retail-price export into
fuel_prices.csv with the schema the pipeline expects:
    price_date, state, diesel_price_per_litre, source

NO API KEY REQUIRED.

Two input modes (auto-detected):

  (A) PPAC/IOC metro-city daily series  [DEFAULT for the DashRSP `rspData.csv`
      mirror of PPAC's daily RSP: columns include 'Products', 'Metro Cities',
      'Calendar Day', and a 'Retail Selling Price ...' column].
      -> The 4 metros (Delhi, Mumbai, Chennai, Kolkata) become DIRECT REAL prices
         for their states (Delhi, Maharashtra, Tamil Nadu, West Bengal).
      -> The remaining hub states are DERIVED from the real national daily
         backbone (mean of the 4 metros) plus the documented state VAT offset,
         so every state gets a daily series grounded in real national price
         movements. `source` distinguishes 'PPAC-IOC-metro' (real) from
         'PPAC-derived' (real national trend + VAT offset).
      -> Forward-filled to daily over [--start, --end]. Real Indian diesel was
         price-static across most of 2023-2025 (OMCs held prices), so carrying
         the last real observation forward to Dec-2025 is faithful, not invented.

  (B) Generic export  [pass --date_col/--state_col/--price_col explicitly].
      Original behaviour: rename, map city->state, dedup, write.

Usage (A, real PPAC/IOC metro mirror):
  python load_ppac_diesel.py --infile ../real_cache/ppac_rsp_raw.csv \
     --out ../real_cache/fuel_prices.csv --start 2023-01-01 --end 2025-12-31

Usage (B, generic):
  python load_ppac_diesel.py --infile ppac_raw.csv --out ../real_cache/fuel_prices.csv \
     --date_col Date --state_col State --price_col "Diesel (Rs/L)"
"""
import argparse
import pandas as pd

# Map metro/city names PPAC sometimes uses -> the state names in hubs.csv.
CITY_TO_STATE = {
    "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra",
    "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Delhi": "Delhi",
    "Gurugram": "Haryana", "Chandigarh": "Punjab", "Ludhiana": "Punjab",
    "Jaipur": "Rajasthan", "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh",
    "Kolkata": "West Bengal", "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
    "Bengaluru": "Karnataka", "Bangalore": "Karnataka", "Hyderabad": "Telangana",
    "Indore": "Madhya Pradesh", "Bhopal": "Madhya Pradesh",
    "Visakhapatnam": "Andhra Pradesh",
}

# Documented state VAT-driven offsets vs national base (kept in sync with
# scripts/weather_fuel.py STATE_DIESEL_OFFSET). Used ONLY to derive daily series
# for non-metro states from the REAL national daily backbone.
STATE_DIESEL_OFFSET = {
    "Maharashtra": +6.2, "Gujarat": -1.1, "Delhi": -1.8, "Haryana": +1.4,
    "Punjab": +2.0, "Rajasthan": +4.8, "Uttar Pradesh": +0.6, "West Bengal": +3.1,
    "Tamil Nadu": +2.4, "Karnataka": +3.6, "Telangana": +5.9,
    "Madhya Pradesh": +3.9, "Andhra Pradesh": +7.1,
}
# The 4 metros in the real series and the state each represents directly.
METRO_TO_STATE = {"Delhi": "Delhi", "Mumbai": "Maharashtra",
                  "Chennai": "Tamil Nadu", "Kolkata": "West Bengal"}


def _daily_full(df_state, start, end):
    """Reindex one state's (date, price) to daily [start,end] with ffill/bfill."""
    s = (df_state.set_index("price_date")["diesel_price_per_litre"]
         .sort_index())
    idx = pd.date_range(start, end, freq="D")
    s = s[~s.index.duplicated(keep="last")].reindex(idx).ffill().bfill()
    return s


def load_metro_series(infile):
    """Parse the PPAC/IOC metro daily RSP export -> long diesel frame."""
    df = pd.read_csv(infile)
    df.columns = [c.strip() for c in df.columns]
    price_col = [c for c in df.columns if c.startswith("Retail Selling Price")][0]
    df["Products"] = df["Products"].astype(str).str.strip()
    df["Metro Cities"] = df["Metro Cities"].astype(str).str.strip()
    d = df[df["Products"] == "Diesel"].copy()
    d["price_date"] = pd.to_datetime(d["Calendar Day"], errors="coerce")
    d["diesel_price_per_litre"] = pd.to_numeric(d[price_col], errors="coerce")
    d = d.dropna(subset=["price_date", "diesel_price_per_litre"])
    return d[["price_date", "Metro Cities", "diesel_price_per_litre"]]


def main_metro(infile, out, start, end):
    d = load_metro_series(infile)
    d = d[d["price_date"].between(start, end)]

    # (1) real per-metro daily series -> 4 real states
    metro_daily = {}
    real_rows = []
    for metro, state in METRO_TO_STATE.items():
        md = d[d["Metro Cities"] == metro][["price_date", "diesel_price_per_litre"]]
        s = _daily_full(md, start, end)
        metro_daily[metro] = s
        real_rows.append(pd.DataFrame({
            "price_date": s.index, "state": state,
            "diesel_price_per_litre": s.values.round(2),
            "source": "PPAC-IOC-metro"}))

    # (2) real national daily backbone = mean of the 4 metros
    national = pd.concat(metro_daily.values(), axis=1).mean(axis=1)
    metro_off_mean = sum(STATE_DIESEL_OFFSET[s] for s in METRO_TO_STATE.values()) / 4.0

    # (3) derive the remaining hub states from real national trend + VAT offset
    derived_rows = []
    for state, off in STATE_DIESEL_OFFSET.items():
        if state in METRO_TO_STATE.values():
            continue
        price = (national + (off - metro_off_mean)).round(2)
        derived_rows.append(pd.DataFrame({
            "price_date": price.index, "state": state,
            "diesel_price_per_litre": price.values,
            "source": "PPAC-derived"}))

    out_df = (pd.concat(real_rows + derived_rows, ignore_index=True)
              .sort_values(["price_date", "state"]).reset_index(drop=True))
    out_df.to_csv(out, index=False)
    real_states = sorted(set(METRO_TO_STATE.values()))
    print(f"Wrote {len(out_df)} (date,state) diesel rows to {out}")
    print(f"  window {start.date()}..{end.date()}, "
          f"{out_df.state.nunique()} states")
    print(f"  DIRECT-REAL states ({len(real_states)}): {real_states}")
    print(f"  DERIVED states: {sorted(set(STATE_DIESEL_OFFSET)-set(real_states))}")
    print(f"  national real backbone range: "
          f"{national.min():.2f}..{national.max():.2f} INR/L")


def main_generic(infile, out, date_col, state_col, price_col):
    df = pd.read_csv(infile)
    df = df.rename(columns={date_col: "price_date", state_col: "region",
                            price_col: "diesel_price_per_litre"})
    df["price_date"] = pd.to_datetime(df["price_date"], dayfirst=True,
                                      errors="coerce")
    df["state"] = df["region"].map(CITY_TO_STATE).fillna(df["region"])
    df["diesel_price_per_litre"] = pd.to_numeric(
        df["diesel_price_per_litre"], errors="coerce")
    df = df.dropna(subset=["price_date", "diesel_price_per_litre"])
    df["source"] = "PPAC"
    out_df = (df[["price_date", "state", "diesel_price_per_litre", "source"]]
              .drop_duplicates(["price_date", "state"])
              .sort_values(["price_date", "state"]))
    out_df.to_csv(out, index=False)
    print(f"Wrote {len(out_df)} (date,state) diesel rows to {out}")
    print("States covered:", sorted(out_df.state.unique()))


def _is_metro_export(infile):
    head = pd.read_csv(infile, nrows=1)
    cols = {c.strip() for c in head.columns}
    return {"Products", "Metro Cities", "Calendar Day"} <= cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--out", default="../real_cache/fuel_prices.csv")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2025-12-31")
    # generic-mode column mapping (mode B)
    ap.add_argument("--date_col", default="Date")
    ap.add_argument("--state_col", default="City")
    ap.add_argument("--price_col", default="Diesel")
    a = ap.parse_args()
    if _is_metro_export(a.infile):
        main_metro(a.infile, a.out, pd.Timestamp(a.start), pd.Timestamp(a.end))
    else:
        main_generic(a.infile, a.out, a.date_col, a.state_col, a.price_col)
