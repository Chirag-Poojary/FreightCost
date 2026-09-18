"""
End-to-end build of the Indian B2B road-freight dataset.
Follows Appendix B's 11-phase execution order exactly.

Run:  python build_dataset.py --orders 60000 --seed 42
Outputs the eight reference tables + two model-ready training frames to output/.
"""
import argparse
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import reference_data as ref
import routing
import weather_fuel as wf
import generators as gen


def log(msg):
    print(f"[build] {msg}", flush=True)


def main(n_orders, seed, outdir, weather_cache=None, fuel_cache=None, route_cache_path=None):
    rng = np.random.default_rng(seed)

    # Real-data sources: consume extraction caches when supplied, else modeled.
    weather_src = wf.make_weather_source(weather_cache)

    # === PHASE 1: backbone ==================================================
    log("Phase 1: hubs / vendors / cost assumptions")
    hubs = ref.build_hubs()
    vendors = ref.build_vendors(rng)
    cost_assumptions = ref.build_cost_assumptions()
    hub_state = hubs.set_index("hub_id")["state"].to_dict()

    # === PHASE 3 (routes first; cargo needs nothing from it but we cache now)
    log("Phase 3: route_distance_cache (unique hub pairs)")
    route_cache = routing.load_route_cache(hubs, rng, cache_path=route_cache_path)
    log(f"        {len(route_cache)} unique routes cached")

    # --- order skeleton (Phase 1 tail): vendor/origin/dest/category/date ----
    log("Phase 1: order skeleton")
    hub_ids = hubs["hub_id"].to_numpy()
    # vendor selection weighted by fleet size (bigger fleets => more orders)
    v_weights = vendors["fleet_size"].to_numpy().astype(float)
    v_weights /= v_weights.sum()
    order_vendor = rng.choice(vendors["vendor_id"].to_numpy(), size=n_orders, p=v_weights)

    origins = rng.choice(hub_ids, size=n_orders)
    dests = rng.choice(hub_ids, size=n_orders)
    same = origins == dests
    while same.any():                       # no intra-hub shipments
        dests[same] = rng.choice(hub_ids, size=same.sum())
        same = origins == dests

    categories = rng.choice(ref.PRODUCT_CATEGORIES, size=n_orders,
                            p=[.18, .24, .20, .20, .18])
    order_dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 3, size=n_orders), unit="D")

    orders = pd.DataFrame({
        "order_id": [f"ORD{i+1:06d}" for i in range(n_orders)],
        "vendor_id": order_vendor,
        "origin_hub_id": origins,
        "dest_hub_id": dests,
        "product_category": categories,
        "order_date": order_dates,
    })
    orders["state"] = orders["origin_hub_id"].map(hub_state)  # origin state for fuel

    # === PHASE 2: cargo physicals ==========================================
    log("Phase 2: cargo weight/volume + dimensional weight")
    cargo = np.array([gen.generate_cargo(c, rng) for c in orders["product_category"]])
    orders["weight_kg"] = cargo[:, 0]
    orders["length_cm"] = cargo[:, 1]
    orders["width_cm"] = cargo[:, 2]
    orders["height_cm"] = cargo[:, 3]
    orders["dimensional_weight_kg"] = (
        orders.length_cm * orders.width_cm * orders.height_cm) / gen.DIM_DIVISOR
    orders["billable_weight_kg"] = orders[
        ["weight_kg", "dimensional_weight_kg"]].max(axis=1)
    orders["truck_type"] = orders["weight_kg"].map(ref.assign_truck_type)

    # === PHASE 3 join: distance_km + ideal_days from cache ==================
    log("Phase 3 join: distance / ideal_days onto orders")
    dl = orders.apply(
        lambda r: routing.lookup(route_cache, r.origin_hub_id, r.dest_hub_id),
        axis=1)
    orders["distance_km"] = [x[0] for x in dl]
    orders["ideal_days"] = [x[1] for x in dl]

    # === PHASE 6: quoted days (vendor-clustered padding) ====================
    log("Phase 6: quoted_days")
    vpad = vendors.set_index("vendor_id")["_latent_padding_mu"].to_dict()
    def _quoted(r):
        mu = vpad[r.vendor_id]
        pf = float(np.clip(rng.normal(mu, 0.04), 1.01, 1.60))  # cluster around vendor mu
        return gen.generate_quoted_days(r.ideal_days, pf)
    orders["quoted_days"] = orders.apply(_quoted, axis=1)

    # === PHASE 4: weather over PLANNED window (Model 1 expected weather) ====
    log("Phase 4: expected weather severity (planned window)")
    weather_cache_rows = []
    exp_scores = []
    dest_state = hubs.set_index("hub_id")["state"].to_dict()
    for r in orders.itertuples():
        start = r.order_date
        end = r.order_date + pd.Timedelta(days=int(np.ceil(r.quoted_days)))
        st = dest_state[r.dest_hub_id]
        precip, wind, temp = weather_src.window(r.dest_hub_id, st, start, end, rng)
        exp_scores.append(wf.weather_severity_score(precip, wind, temp))
        if len(weather_cache_rows) < 5000:   # cache a representative sample
            weather_cache_rows.append((
                f"{r.origin_hub_id}-{r.dest_hub_id}", str(start.date()),
                str(end.date()), str(precip), str(wind), str(pd.Timestamp.now())))
    orders["expected_weather_score"] = exp_scores
    if weather_src.is_real:
        # Pass the real extraction cache through to the output unchanged.
        weather_cache_out = weather_src.cache_df
    else:
        weather_cache_out = pd.DataFrame(weather_cache_rows, columns=[
            "route_key", "window_start", "window_end",
            "precipitation_sum_json", "windspeed_max_json", "api_call_timestamp"])

    # === PHASE 5: fuel prices (state-level, as-of backward join) ============
    log("Phase 5: fuel_prices + merge_asof(by=state, backward, 7D)")
    fuel = wf.load_fuel_prices(sorted(orders["state"].unique()), rng,
                               cache_path=fuel_cache)
    orders = orders.sort_values("order_date").reset_index(drop=True)
    fuel_sorted = fuel.sort_values("price_date").reset_index(drop=True)
    orders = pd.merge_asof(
        orders, fuel_sorted[["price_date", "state", "diesel_price_per_litre"]],
        left_on="order_date", right_on="price_date", by="state",
        direction="backward", tolerance=pd.Timedelta("7D"))
    orders = orders.rename(columns={"diesel_price_per_litre": "expected_fuel_price"})
    orders = orders.drop(columns=["price_date"])
    # State-coverage fallback (doc): fill any missing price with national daily median
    missing = orders["expected_fuel_price"].isna()
    if missing.any():
        natl = fuel.groupby("price_date")["diesel_price_per_litre"].median()
        orders.loc[missing, "expected_fuel_price"] = (
            orders.loc[missing, "order_date"].map(natl).fillna(natl.median()))
    log(f"        fuel-price fallback applied to {int(missing.sum())} rows")

    # === PHASE 7: Base_Freight_Cost label ==================================
    log("Phase 7: base_freight_cost label (+/-2% noise)")
    orders["base_freight_cost"] = orders.apply(
        lambda r: gen.add_label_noise(
            gen.base_freight_cost(r.distance_km, r.expected_fuel_price,
                                  r.truck_type, r.ideal_days), rng),
        axis=1).round(2)

    # === PHASE 9 (part): decide anomalies now (needed for actual_days) ======
    log("Phase 9a: anomaly injection (5-8%, three mechanisms)")
    anomaly_rate = float(rng.uniform(0.05, 0.08))
    n = len(orders)
    # vendor fraud propensity weights WHICH rows get picked
    vprop = vendors.set_index("vendor_id")["_latent_fraud_propensity"].to_dict()
    w = orders["vendor_id"].map(vprop).to_numpy().astype(float)
    w /= w.sum()
    k = int(n * anomaly_rate)
    anomaly_idx = rng.choice(n, size=k, replace=False, p=w)
    anomaly_types = rng.choice(
        ["detention_padding", "toll_inflation", "weight_discrepancy"], size=k)
    orders["anomaly_type"] = None            # hidden audit column
    orders["requires_manual_review"] = 0     # the label
    orders.loc[anomaly_idx, "anomaly_type"] = anomaly_types
    orders.loc[anomaly_idx, "requires_manual_review"] = 1
    log(f"        anomaly_rate={anomaly_rate:.3f}  positives={k}")

    # === PHASE 8: Model 1 OOF predictions ==================================
    log("Phase 8: Model 1 (XGB/GBM) 5-fold OOF predictions")
    orders = train_model1_oof(orders, seed)

    # === PHASE 9 (rest): build invoices ====================================
    log("Phase 9b: invoices table (actual_days, weather events, features)")
    invoices = build_invoices(orders, dest_state, rng, weather_src)

    # === PHASE 10: causal vendor historical risk score =====================
    log("Phase 10: vendor_historical_risk_score (causal, Laplace-smoothed)")
    invoices = add_vendor_risk(invoices)

    # === Residual real-world messiness (survives into delivered files) ======
    # The doc describes several real data-quality issues (API nulls, source
    # unit/format drift, capture errors). Most are handled internally; here we
    # deliberately leave a SMALL residue so the dataset reads like real captured
    # data that still needs cleaning -- not a pristine synthetic frame. Every
    # issue below is documented in DATA_QUALITY_NOTES.md.
    log("Injecting residual real-world data-quality issues")
    orders, invoices = inject_residual_issues(orders, invoices, rng)

    # === PHASE 11 handled by train script; here we just assemble outputs ====
    log("Assembling outputs")
    write_outputs(outdir, hubs, vendors, cost_assumptions, route_cache,
                  weather_cache_out, fuel, orders, invoices)
    return orders, invoices


# ---------------------------------------------------------------------------
def train_model1_oof(orders, seed):
    from sklearn.model_selection import KFold
    try:
        from xgboost import XGBRegressor
        def make(): return XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
        algo = "xgboost"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor
        def make(): return HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.05, max_iter=400, random_state=seed)
        algo = "hist-gbm (xgboost unavailable)"
    log(f"        Model 1 algo: {algo}")

    feats_num = ["distance_km", "ideal_days", "quoted_days", "billable_weight_kg",
                 "expected_fuel_price", "expected_weather_score"]
    X = orders[feats_num].copy()
    X = pd.concat([X, pd.get_dummies(orders["product_category"], prefix="cat"),
                   pd.get_dummies(orders["truck_type"], prefix="truck")], axis=1)
    y = orders["base_freight_cost"].to_numpy()

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    orders["oof_fold_id"] = -1
    oof = np.full(len(orders), np.nan)
    for fold, (tr, ho) in enumerate(kf.split(X)):
        m = make()
        m.fit(X.iloc[tr], y[tr])
        oof[ho] = m.predict(X.iloc[ho])
        orders.iloc[ho, orders.columns.get_loc("oof_fold_id")] = fold
    orders["model_a_predicted_cost"] = np.round(oof, 2)
    mae = np.mean(np.abs(oof - y))
    log(f"        OOF MAE = {mae:,.0f} rupees "
        f"({mae / y.mean() * 100:.1f}% of mean cost)")
    return orders


def build_invoices(orders, dest_state, rng, weather_src):
    rows = []
    for r in orders.itertuples():
        atype = r.anomaly_type
        has_delay = atype in gen.DELAY_LINKED_ANOMALIES
        actual_days = gen.generate_actual_days(r.ideal_days, r.quoted_days,
                                               has_delay, rng)
        billed = gen.generate_billed_amount(r.base_freight_cost, atype, rng)
        # Phase 9: actual-window weather events (Model 2)
        start = r.order_date
        end = r.order_date + pd.Timedelta(days=int(np.ceil(actual_days)))
        precip, _, _ = weather_src.window(r.dest_hub_id,
                                          dest_state[r.dest_hub_id],
                                          start, end, rng)
        adverse = wf.adverse_weather_days(precip)
        rows.append((
            f"INV{r.Index+1:06d}", r.order_id, r.vendor_id,
            round(billed, 2), round(actual_days, 2), adverse,
            round(r.ideal_days, 3), round(r.quoted_days, 1),
            r.model_a_predicted_cost, r.order_date,
            r.requires_manual_review, atype))
    inv = pd.DataFrame(rows, columns=[
        "invoice_id", "order_id", "vendor_id", "actual_billed_amount",
        "actual_days", "adverse_weather_days", "ideal_days", "quoted_days",
        "model_a_predicted_cost", "order_date",
        "requires_manual_review", "anomaly_type"])
    # engineered features
    inv["cost_mismatch"] = inv["actual_billed_amount"] - inv["model_a_predicted_cost"]
    inv["true_delay_days"] = inv["actual_days"] - inv["ideal_days"]
    inv["commercial_delay_days"] = inv["actual_days"] - inv["quoted_days"]
    inv["vendor_padding_ratio"] = inv["quoted_days"] / inv["ideal_days"].clip(lower=0.1)
    return inv


def add_vendor_risk(invoices, alpha=2, prior=0.06):
    df = invoices.sort_values("order_date").reset_index(drop=True)
    scores = {}
    for vendor_id, group in df.groupby("vendor_id"):
        flagged, total = 0, 0
        for _, row in group.iterrows():
            score = (flagged + alpha * prior) / (total + alpha)
            scores[row["invoice_id"]] = score
            flagged += row["requires_manual_review"]
            total += 1
    df["vendor_historical_risk_score"] = df["invoice_id"].map(scores)
    return df


def inject_residual_issues(orders, invoices, rng):
    """Leave a small, documented residue of real-world data problems.
    Kept low-rate so the data is usable but clearly needs a cleaning pass."""
    o = orders.copy()
    inv = invoices.copy()
    n = len(o)

    # 1. ~0.4% missing expected_weather_score (Open-Meteo returned all-null window)
    idx = rng.choice(n, size=int(n * 0.004), replace=False)
    o.loc[idx, "expected_weather_score"] = np.nan

    # 2. ~0.3% weight captured in grams instead of kg (unit drift at data entry)
    idx = rng.choice(n, size=int(n * 0.003), replace=False)
    o.loc[idx, "weight_kg"] = (o.loc[idx, "weight_kg"] * 1000).round(0)

    # 3. ~0.2% duplicated invoice rows (double-submission in AP export)
    dup_idx = rng.choice(len(inv), size=int(len(inv) * 0.002), replace=False)
    dups = inv.iloc[dup_idx].copy()
    inv = pd.concat([inv, dups], ignore_index=True)

    # 4. a few order_date values as strings in a different format (source drift)
    idx = rng.choice(n, size=min(40, n), replace=False)
    o["order_date"] = o["order_date"].astype(str)
    o.loc[idx, "order_date"] = pd.to_datetime(
        o.loc[idx, "order_date"]).dt.strftime("%d-%m-%Y")

    # 5. ~0.15% negative/zero actual_billed_amount slipped through (credit notes)
    bidx = rng.choice(len(inv), size=int(len(inv) * 0.0015), replace=False)
    inv.loc[bidx, "actual_billed_amount"] = -inv.loc[bidx, "actual_billed_amount"].abs()

    # 6. inconsistent vendor_id casing on a handful of rows (join hazard)
    vidx = rng.choice(len(inv), size=min(25, len(inv)), replace=False)
    inv.loc[vidx, "vendor_id"] = inv.loc[vidx, "vendor_id"].str.lower()

    return o, inv


def write_outputs(outdir, hubs, vendors, cost_assumptions, route_cache,
                  weather_cache, fuel, orders, invoices):
    import os
    os.makedirs(outdir, exist_ok=True)
    # public vendor table drops latent columns
    vendors_pub = vendors.drop(columns=[c for c in vendors.columns
                                        if c.startswith("_latent")])
    hubs.to_csv(f"{outdir}/hubs.csv", index=False)
    vendors_pub.to_csv(f"{outdir}/vendors.csv", index=False)
    cost_assumptions.to_csv(f"{outdir}/toll_and_cost_assumptions.csv", index=False)
    route_cache.drop(columns=["_haversine_km"]).to_csv(
        f"{outdir}/route_distance_cache.csv", index=False)
    weather_cache.to_csv(f"{outdir}/weather_cache.csv", index=False)
    fuel.to_csv(f"{outdir}/fuel_prices.csv", index=False)

    # orders table -- schema columns from the doc
    orders_cols = ["order_id", "vendor_id", "origin_hub_id", "dest_hub_id",
                   "product_category", "order_date", "weight_kg", "length_cm",
                   "width_cm", "height_cm", "quoted_days", "ideal_days",
                   "distance_km", "dimensional_weight_kg", "billable_weight_kg",
                   "truck_type", "state", "expected_fuel_price",
                   "expected_weather_score", "base_freight_cost",
                   "model_a_predicted_cost", "oof_fold_id"]
    orders[orders_cols].to_csv(f"{outdir}/orders.csv", index=False)
    # hidden audit table (anomaly ground truth) kept SEPARATE from model features
    orders[["order_id", "anomaly_type", "requires_manual_review"]].to_csv(
        f"{outdir}/_ground_truth_audit.csv", index=False)

    inv_cols = ["invoice_id", "order_id", "vendor_id", "actual_billed_amount",
                "actual_days", "adverse_weather_days", "cost_mismatch",
                "true_delay_days", "commercial_delay_days", "vendor_padding_ratio",
                "vendor_historical_risk_score", "requires_manual_review"]
    invoices[inv_cols].to_csv(f"{outdir}/invoices.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--weather-cache", default=None,
                    help="real weather_cache.csv from run_openmeteo.py")
    ap.add_argument("--fuel-cache", default=None,
                    help="real fuel_prices.csv from load_ppac_diesel.py")
    ap.add_argument("--route-cache", default=None,
                    help="real route_distance_cache.csv from run_ors.py or run_osrm.py")
    a = ap.parse_args()
    main(a.orders, a.seed, a.outdir,
         weather_cache=a.weather_cache, fuel_cache=a.fuel_cache,
         route_cache_path=a.route_cache)
