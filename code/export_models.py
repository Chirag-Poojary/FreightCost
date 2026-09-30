"""
Train and EXPORT deployable model artifacts for both models.

The build/train scripts compute an out-of-fold Model 1 and evaluate Model 2, but
never persist a servable model. This script does the documented cleaning pass,
trains a final Model 1 (freight should-cost regressor) and Model 2 (invoice-risk
classifier) on the rebuilt real-data dataset, and writes everything a downstream
app (e.g. a website) needs to load and predict:

  output/models/
    model_1_freight_cost.joblib     sklearn-API XGBRegressor (pipeline-ready)
    model_1_freight_cost.json       xgboost native booster (portable)
    model_1_features.json           exact input schema + one-hot column order
    model_2_invoice_risk.joblib     XGBClassifier
    model_2_invoice_risk.json       xgboost native booster
    model_2_features.json           feature list + decision threshold
    metadata.json                   metrics, versions, build info
    predict_example.py              minimal load-and-predict demo

Usage:
  python export_models.py --outdir output          # after build_dataset.py
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import reference_data as ref
import generators as gen

from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, average_precision_score,
                             precision_score, recall_score, f1_score)
import joblib

M1_NUM = ["distance_km", "ideal_days", "quoted_days", "billable_weight_kg",
          "expected_fuel_price", "expected_weather_score"]
M2_FEATURES = [
    "actual_billed_amount", "model_a_predicted_cost", "cost_mismatch",
    "actual_days", "true_delay_days", "commercial_delay_days",
    "adverse_weather_days", "vendor_padding_ratio",
    "vendor_historical_risk_score",
    # Phase 3: Weight discrepancy anomaly detection features
    "cost_mismatch_ratio",
    "cost_mismatch_upper",
    "cost_mismatch_upper_ratio",
    "billed_to_p90_ratio",
    "excess_rate_per_kg",
    "excess_rate_per_ton_km",
    "cost_mismatch_delay_skew",
    "zero_delay_p90_breach"
]
M2_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# Cleaning (per DATA_QUALITY_NOTES.md) — cleaning lives in training, not raw files
# --------------------------------------------------------------------------- #
def clean_orders(orders):
    o = orders.copy()
    # 6. mixed order_date formats -> robust parse (day-first inference)
    o["order_date"] = pd.to_datetime(o["order_date"], dayfirst=True,
                                     errors="coerce", format="mixed")
    # 2. grams-as-kg: weight above legal Indian GVW -> /1000, recompute derived
    grams = o["weight_kg"] > 30000
    o.loc[grams, "weight_kg"] = (o.loc[grams, "weight_kg"] / 1000).round(1)
    o["billable_weight_kg"] = o[["weight_kg", "dimensional_weight_kg"]].max(axis=1)
    o["truck_type"] = o["weight_kg"].map(ref.assign_truck_type)
    # 1. missing expected_weather_score -> impute (month median, then global)
    if o["expected_weather_score"].isna().any():
        o["_mon"] = o["order_date"].dt.month
        med = o.groupby("_mon")["expected_weather_score"].transform("median")
        o["expected_weather_score"] = o["expected_weather_score"].fillna(med)
        o["expected_weather_score"] = o["expected_weather_score"].fillna(
            o["expected_weather_score"].median())
        o = o.drop(columns="_mon")
    return o


def clean_invoices(inv):
    before = len(inv)
    inv = inv.drop_duplicates(subset="invoice_id", keep="first").copy()
    inv["vendor_id"] = inv["vendor_id"].str.upper()          # 5. casing drift
    inv = inv[inv["actual_billed_amount"] > 0]               # 4. drop credit notes
    print(f"[clean] invoices {before} -> {len(inv)} rows")
    return inv


def build_m1_matrix(o):
    """Numeric + one-hot(category) + one-hot(truck_type), fixed column order."""
    cat_cols = [f"cat_{c}" for c in ref.PRODUCT_CATEGORIES]
    truck_cols = ["truck_6-wheeler", "truck_10-wheeler", "truck_12-wheeler"]
    X = o[M1_NUM].copy()
    for c in ref.PRODUCT_CATEGORIES:
        X[f"cat_{c}"] = (o["product_category"] == c).astype(int)
    for t in ["6-wheeler", "10-wheeler", "12-wheeler"]:
        X[f"truck_{t}"] = (o["truck_type"] == t).astype(int)
    order = M1_NUM + cat_cols + truck_cols
    return X[order], order


def make_reg(seed):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=-1)


def make_quantile_reg(seed, alpha):
    from xgboost import XGBRegressor
    return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                        n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, n_jobs=-1)


def make_clf(seed, spw):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=spw, eval_metric="aucpr",
                         random_state=seed, n_jobs=-1)


def main(out, seed):
    mdir = f"{out}/models"
    os.makedirs(mdir, exist_ok=True)
    meta = {"seed": seed}

    # ---------- MODEL 1 ----------
    print("=== Model 1: freight cost regressor ===")
    orders = pd.read_csv(f"{out}/orders.csv")
    o = clean_orders(orders)
    X, col_order = build_m1_matrix(o)
    y = o["base_freight_cost"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    reg = make_reg(seed); reg.fit(Xtr, ytr)
    reg_p10 = make_quantile_reg(seed, 0.10); reg_p10.fit(Xtr, ytr)
    reg_p90 = make_quantile_reg(seed, 0.90); reg_p90.fit(Xtr, ytr)

    pred = reg.predict(Xte)
    pred_p10 = reg_p10.predict(Xte)
    pred_p90 = reg_p90.predict(Xte)
    m1_mae = float(mean_absolute_error(yte, pred))
    m1_mape = float(np.mean(np.abs(pred - yte) / np.clip(np.abs(yte), 1e-9, None)) * 100)
    coverage = float(np.mean((yte >= pred_p10) & (yte <= pred_p90)))
    mean_width = float(np.mean(pred_p90 - pred_p10))

    print(f"  holdout MAE=Rs {m1_mae:,.1f}  MAPE={m1_mape:.2f}%")
    print(f"  holdout 80% interval coverage={coverage*100:.1f}%  mean width=Rs {mean_width:,.1f}")

    # refit on ALL rows for the deployed models
    reg_full = make_reg(seed); reg_full.fit(X, y)
    reg_p10_full = make_quantile_reg(seed, 0.10); reg_p10_full.fit(X, y)
    reg_p90_full = make_quantile_reg(seed, 0.90); reg_p90_full.fit(X, y)

    joblib.dump(reg_full, f"{mdir}/model_1_freight_cost.joblib")
    reg_full.get_booster().save_model(f"{mdir}/model_1_freight_cost.json")

    joblib.dump(reg_p10_full, f"{mdir}/model_1_freight_cost_p10.joblib")
    reg_p10_full.get_booster().save_model(f"{mdir}/model_1_freight_cost_p10.json")

    joblib.dump(reg_p90_full, f"{mdir}/model_1_freight_cost_p90.joblib")
    reg_p90_full.get_booster().save_model(f"{mdir}/model_1_freight_cost_p90.json")

    json.dump({"numeric": M1_NUM,
               "product_categories": ref.PRODUCT_CATEGORIES,
               "truck_types": ["6-wheeler", "10-wheeler", "12-wheeler"],
               "column_order": col_order,
               "target": "base_freight_cost",
               "prediction_interval": {
                   "alphas": [0.10, 0.90],
                   "p10_artifact": "model_1_freight_cost_p10.joblib",
                   "p90_artifact": "model_1_freight_cost_p90.joblib"
               },
               "notes": "one-hot product_category (cat_*) and truck_type "
                        "(truck_*); truck_type = assign_truck_type(weight_kg); "
                        "impute missing expected_weather_score with month median."},
              open(f"{mdir}/model_1_features.json", "w"), indent=2)
    meta["model_1"] = {"holdout_mae": round(m1_mae, 2),
                       "holdout_mape_pct": round(m1_mape, 3),
                       "n_features": len(col_order),
                       "prediction_interval": {
                           "target_coverage": 0.80,
                           "empirical_coverage": round(coverage, 4),
                           "average_interval_width_inr": round(mean_width, 2)
                       }}

    # ---------- MODEL 2 ----------
    print("=== Model 2: invoice risk classifier ===")
    inv = pd.read_csv(f"{out}/invoices.csv")
    gt = pd.read_csv(f"{out}/_ground_truth_audit.csv")
    inv = clean_invoices(inv)
    inv = inv.merge(gt[["order_id", "anomaly_type"]], on="order_id", how="left")

    # Compute P90 and merge order features for weight discrepancy detection
    o_m1 = o.copy()
    o_m1["p90"] = reg_p90_full.predict(X)
    inv = inv.merge(o_m1[["order_id", "model_a_predicted_cost", "billable_weight_kg", "distance_km", "p90"]], on="order_id")

    # Feature engineering for weight discrepancy & rate inflation
    pred_cost = inv["model_a_predicted_cost"].clip(lower=1.0)
    billed = inv["actual_billed_amount"]
    p90 = inv["p90"].clip(lower=1.0)
    mismatch = inv["cost_mismatch"]
    comm_delay = inv["commercial_delay_days"].clip(lower=0.0)
    weight_tons = (inv["billable_weight_kg"] / 1000.0).clip(lower=0.1)
    dist = inv["distance_km"].clip(lower=10.0)

    inv["cost_mismatch_ratio"] = (mismatch / pred_cost).round(4)
    inv["cost_mismatch_upper"] = np.maximum(0.0, billed - p90).round(2)
    inv["cost_mismatch_upper_ratio"] = (inv["cost_mismatch_upper"] / pred_cost).round(4)
    inv["billed_to_p90_ratio"] = (billed / p90).round(4)
    inv["excess_rate_per_kg"] = (mismatch / inv["billable_weight_kg"].clip(lower=1.0)).round(4)
    inv["excess_rate_per_ton_km"] = (mismatch / (weight_tons * dist)).round(4)
    inv["cost_mismatch_delay_skew"] = (inv["cost_mismatch_ratio"] / (comm_delay + 0.1)).round(4)
    inv["zero_delay_p90_breach"] = ((billed > p90) & (comm_delay <= 0.2)).astype(float)

    feats = [f for f in M2_FEATURES if f in inv.columns]
    inv = inv.dropna(subset=feats)
    Xc = inv[feats]; yc = inv["requires_manual_review"].astype(int)
    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        Xc, yc, inv["anomaly_type"], test_size=0.25, random_state=seed, stratify=yc)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    clf = make_clf(seed, spw); clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    p = (proba >= M2_THRESHOLD).astype(int)
    m2 = {"pr_auc": round(float(average_precision_score(yte, proba)), 3),
          "precision": round(float(precision_score(yte, p)), 3),
          "recall": round(float(recall_score(yte, p)), 3),
          "f1": round(float(f1_score(yte, p)), 3),
          "threshold": M2_THRESHOLD}
    per_type = {}
    te = Xte.copy(); te["y"] = yte.values; te["pred"] = p; te["atype"] = ate.values
    for at, g in te[te.y == 1].groupby("atype"):
        per_type[at] = round(float(g["pred"].mean()), 3)
    m2["per_anomaly_type_recall"] = per_type
    print(f"  PR-AUC={m2['pr_auc']} recall={m2['recall']} precision={m2['precision']}")
    print(f"  per-type recall: {per_type}")
    # refit on ALL cleaned invoices for deployment
    spw_full = (yc == 0).sum() / max((yc == 1).sum(), 1)
    clf_full = make_clf(seed, spw_full); clf_full.fit(Xc, yc)
    joblib.dump(clf_full, f"{mdir}/model_2_invoice_risk.joblib")
    clf_full.get_booster().save_model(f"{mdir}/model_2_invoice_risk.json")
    json.dump({"features": feats, "threshold": M2_THRESHOLD, "target":
               "requires_manual_review",
               "notes": "clean invoices first: dedup invoice_id, upper() "
                        "vendor_id, drop actual_billed_amount<=0. "
                        "anomaly_type is NEVER a feature."},
              open(f"{mdir}/model_2_features.json", "w"), indent=2)
    meta["model_2"] = m2

    import xgboost, sklearn
    meta["versions"] = {"xgboost": xgboost.__version__,
                        "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__}
    json.dump(meta, open(f"{mdir}/metadata.json", "w"), indent=2)

    _write_predict_example(mdir)
    print(f"\nExported artifacts to {mdir}/")
    for f in sorted(os.listdir(mdir)):
        print("  ", f)


def _write_predict_example(mdir):
    import os
    src = os.path.join(mdir, "predict_example.py")
    # If predict_example.py already exists, keep it intact
    if not os.path.exists(src):
        with open(src, "w", encoding="utf-8") as f:
            f.write("# See predict_example.py in output/models\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.outdir, a.seed)
