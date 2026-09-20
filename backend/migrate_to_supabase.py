"""
One-time migration: hubs.csv / vendors.csv / orders.csv / invoices.csv /
_ground_truth_audit.csv  -->  Supabase.

Run once, after applying sql/schema.sql in the Supabase SQL Editor:
  python migrate_to_supabase.py --data-dir ../../code/output

Batches inserts (Supabase/PostgREST chokes on very large single payloads)
and upserts, so it's safe to re-run if it's interrupted partway through.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

from db import Db

BATCH = 500


def clean(df):
    """NaN isn't valid JSON -- Postgres/PostgREST wants null. Swap them."""
    return df.replace({np.nan: None})


def push(client, table, df, batch=BATCH, force=False):
    if not force:
        try:
            r = client.table(table).select("*", count="exact").limit(1).execute()
            if r.count is not None and r.count >= len(df):
                print(f"  {table}: already fully migrated ({r.count}/{len(df)} rows). Skipping.")
                return
        except Exception:
            pass

    records = clean(df).to_dict(orient="records")
    n_batches = math.ceil(len(records) / batch)
    for i in range(n_batches):
        chunk = records[i * batch:(i + 1) * batch]
        client.table(table).upsert(chunk).execute()
        print(f"  {table}: {min((i+1)*batch, len(records))}/{len(records)}")


def main(data_dir, force=False):
    db = Db()
    c = db.client

    print("hubs...")
    hubs = pd.read_csv(f"{data_dir}/hubs.csv")
    push(c, "hubs", hubs, force=force)

    print("vendors...")
    vendors = pd.read_csv(f"{data_dir}/vendors.csv")
    push(c, "vendors", vendors, force=force)

    print("orders...")
    orders = pd.read_csv(f"{data_dir}/orders.csv")
    if "order_date" in orders.columns:
        orders["order_date"] = pd.to_datetime(orders["order_date"], format="mixed").dt.strftime("%Y-%m-%d")
    # keep only columns the orders table actually has (drop OOF-internal columns)
    keep = ["order_id", "vendor_id", "origin_hub_id", "dest_hub_id", "product_category",
           "order_date", "weight_kg", "length_cm", "width_cm", "height_cm",
           "quoted_days", "ideal_days", "distance_km", "dimensional_weight_kg",
           "billable_weight_kg", "truck_type", "state", "expected_fuel_price",
           "expected_weather_score", "model_a_predicted_cost"]
    push(c, "orders", orders[keep], force=force)

    print("reference_invoices...")
    invoices = pd.read_csv(f"{data_dir}/invoices.csv").drop_duplicates(
        subset="order_id", keep="first")
    if "vendor_id" in invoices.columns:
        invoices["vendor_id"] = invoices["vendor_id"].astype(str).str.upper()
    push(c, "reference_invoices", invoices, force=force)

    print("ground_truth_audit...")
    audit = pd.read_csv(f"{data_dir}/_ground_truth_audit.csv")
    push(c, "ground_truth_audit", audit, force=force)

    print("\nDone. Row counts:")
    for t in ["hubs", "vendors", "orders", "reference_invoices", "ground_truth_audit"]:
        r = c.table(t).select("*", count="exact").limit(1).execute()
        print(f"  {t}: {r.count}")


def find_default_data_dir():
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "output")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "code", "output")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "code", "output")),
        os.path.abspath("code/output"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "orders.csv")):
            return c
    return candidates[0]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=find_default_data_dir())
    ap.add_argument("--force", action="store_true", help="Force re-upload even if tables are already populated")
    a = ap.parse_args()
    print(f"Using data directory: {a.data_dir}")
    main(a.data_dir, force=a.force)
