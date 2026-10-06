"""
Train, TEST and EXPORT deployable artifacts for both models.

What changed vs the first version (see freight_models.py for the reasoning):

  Model 1  cost-baseline hybrid -- XGBoost learns a multiplicative correction
           on top of the fuel + toll + driver cost build-up, so predictions
           keep scaling when diesel or distance leave the training range.
  Model 2  trained on OUT-OF-FOLD Model 1 outputs (the old script used
           in-sample P90, which is optimistic), and on scale-free features
           (ratios to fair cost) when that loses no accuracy, so a nationwide
           price rise does not by itself change the fraud score.
  Tests    besides the random holdout, three stress sets outside the training
           range (cheap diesel, expensive diesel, very long routes) and a
           price-inflation test for Model 2. Results for the previous model
           design are reported alongside for comparison.
  Bounds   plausible-price limits are learned from training data and stored in
           the bundle, used by the serving-side logic checks.

Artifacts (output/models/):
  model_1_freight_cost.pkl     dict bundle: point/p10/p90 XGBRegressors + meta
  model_2_invoice_risk.pkl     dict bundle: XGBClassifier + feature list + threshold
  model_1_*.json, model_2_*.json  native xgboost boosters + schemas (portable)
  metadata.json                metrics (old vs new), stress tests, bounds, versions

Usage:
  python export_models.py --outdir output          # after build_dataset.py
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error,
                             precision_score, recall_score)
from sklearn.model_selection import KFold, train_test_split

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "scripts"))
sys.path.insert(0, _HERE)
import generators as gen                      # noqa: E402
import reference_data as ref                  # noqa: E402
from freight_models import (BASELINE_CONSTANTS, M1_NUM, FreightCostModel,  # noqa: E402
                            baseline_components, build_feature_matrix)

TRUCKS = ["6-wheeler", "10-wheeler", "12-wheeler"]

M2_FEATURES_FULL = [
    "actual_billed_amount", "model_a_predicted_cost", "cost_mismatch",
    "actual_days", "true_delay_days", "commercial_delay_days",
    "adverse_weather_days", "vendor_padding_ratio", "vendor_historical_risk_score",
    "cost_mismatch_ratio", "cost_mismatch_upper", "cost_mismatch_upper_ratio",
    "billed_to_p90_ratio", "excess_rate_per_kg", "excess_rate_per_ton_km",
    "cost_mismatch_delay_skew", "zero_delay_p90_breach",
]
# Same signal without absolute rupee amounts (which drift with the price level).
M2_FEATURES_SCALE_FREE = [
    "actual_days", "true_delay_days", "commercial_delay_days",
    "adverse_weather_days", "vendor_padding_ratio", "vendor_historical_risk_score",
    "cost_mismatch_ratio", "cost_mismatch_upper_ratio", "billed_to_p90_ratio",
    "cost_mismatch_delay_skew", "zero_delay_p90_breach",
]
M2_THRESHOLD = 0.5
# Rupee-valued Model 2 features. In the "deflated" variant these are divided by
# (diesel price / training-average diesel price) -- i.e. expressed in
# training-period rupees -- so a nationwide price rise does not look like fraud,
# while a fixed-rupee padding (e.g. +Rs 5,000 toll) is still visible.
M2_RUPEE_FEATURES = ["actual_billed_amount", "model_a_predicted_cost", "cost_mismatch",
                     "cost_mismatch_upper", "excess_rate_per_kg", "excess_rate_per_ton_km"]


# --------------------------------------------------------------------------- #
# Cleaning (per DATA_QUALITY_NOTES.md)
# --------------------------------------------------------------------------- #
def clean_orders(orders):
    o = orders.copy()
    o["order_date"] = pd.to_datetime(o["order_date"], dayfirst=True, errors="coerce", format="mixed")
    grams = o["weight_kg"] > 30000                       # grams recorded as kg
    o.loc[grams, "weight_kg"] = (o.loc[grams, "weight_kg"] / 1000).round(1)
    o["billable_weight_kg"] = o[["weight_kg", "dimensional_weight_kg"]].max(axis=1)
    o["truck_type"] = o["weight_kg"].map(ref.assign_truck_type)
    if o["expected_weather_score"].isna().any():
        o["_mon"] = o["order_date"].dt.month
        med = o.groupby("_mon")["expected_weather_score"].transform("median")
        o["expected_weather_score"] = o["expected_weather_score"].fillna(med).fillna(
            o["expected_weather_score"].median())
        o = o.drop(columns="_mon")
    return o.reset_index(drop=True)


def clean_invoices(inv):
    before = len(inv)
    inv = inv.drop_duplicates(subset="invoice_id", keep="first").copy()
    inv["vendor_id"] = inv["vendor_id"].str.upper()
    inv = inv[inv["actual_billed_amount"] > 0]           # credit notes are not invoices
    print(f"[clean] invoices {before} -> {len(inv)} rows")
    return inv


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def make_reg(seed, objective="reg:squarederror", alpha=None):
    from xgboost import XGBRegressor
    kw = dict(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, random_state=seed, n_jobs=-1, objective=objective)
    if alpha is not None:
        kw["quantile_alpha"] = alpha
    return XGBRegressor(**kw)


def make_clf(seed, spw, features, monotone=None):
    from xgboost import XGBClassifier
    kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
              random_state=seed, n_jobs=-1)
    if monotone:
        kw["monotone_constraints"] = "(" + ",".join(str(monotone.get(f, 0)) for f in features) + ")"
    return XGBClassifier(**kw)


class HybridM1:
    """Point + P10 + P90 regressors on the ratio target y / baseline."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, df, y):
        base = baseline_components(df)["baseline"].to_numpy()
        r = y / base
        self.point = make_reg(self.seed).fit(X, r)
        self.p10 = make_reg(self.seed, "reg:quantileerror", 0.10).fit(X, r)
        self.p90 = make_reg(self.seed, "reg:quantileerror", 0.90).fit(X, r)
        return self

    def predict(self, X, df):
        base = baseline_components(df)["baseline"].to_numpy()
        p, lo, hi = (m.predict(X) * base for m in (self.point, self.p10, self.p90))
        return p, np.minimum.reduce([lo, hi, p]), np.maximum.reduce([lo, hi, p])


class DirectM1:
    """The previous design: XGBoost straight on rupees (kept for comparison only)."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, df, y):
        self.point = make_reg(self.seed).fit(X, y)
        self.p10 = make_reg(self.seed, "reg:quantileerror", 0.10).fit(X, y)
        self.p90 = make_reg(self.seed, "reg:quantileerror", 0.90).fit(X, y)
        return self

    def predict(self, X, df):
        p, lo, hi = (m.predict(X) for m in (self.point, self.p10, self.p90))
        return p, np.minimum.reduce([lo, hi, p]), np.maximum.reduce([lo, hi, p])


def m1_metrics(y, p, lo, hi):
    return {"mae_inr": round(float(mean_absolute_error(y, p)), 2),
            "mape_pct": round(float(np.mean(np.abs(p - y) / np.abs(y)) * 100), 3),
            "max_abs_pct_error": round(float(np.max(np.abs(p - y) / np.abs(y)) * 100), 2),
            "interval_coverage": round(float(np.mean((y >= lo) & (y <= hi))), 4),
            "negative_predictions": int((p <= 0).sum())}


def stress_sets(df_test, rng):
    """Out-of-range scenarios. Labels use the same cost build-up as the
    training data (+/-2% noise), so they test extrapolation, not new physics."""
    def relabel(d):
        d = d.copy()
        d["base_freight_cost"] = [
            gen.base_freight_cost(r.distance_km, r.expected_fuel_price, r.truck_type, r.ideal_days)
            * rng.normal(1.0, 0.02) for r in d.itertuples()]
        return d

    n = min(3000, len(df_test))
    s = df_test.sample(n, random_state=7).reset_index(drop=True)
    cheap = s.copy(); cheap["expected_fuel_price"] = rng.uniform(70, 86, n)
    dear = s.copy(); dear["expected_fuel_price"] = rng.uniform(98, 125, n)
    long_ = s.copy()
    k = rng.uniform(2900, 4500, n) / long_["distance_km"]
    long_["distance_km"] *= k
    long_["ideal_days"] *= k
    long_["quoted_days"] *= k
    return {"diesel_70_to_86": relabel(cheap), "diesel_98_to_125": relabel(dear),
            "routes_2900_to_4500_km": relabel(long_)}


# --------------------------------------------------------------------------- #
def main(out, seed):
    rng = np.random.default_rng(seed)
    mdir = f"{out}/models"
    os.makedirs(mdir, exist_ok=True)
    meta = {"seed": seed}

    # ============================ MODEL 1 ============================
    print("=== Model 1: freight cost (cost-baseline hybrid) ===")
    o = clean_orders(pd.read_csv(f"{out}/orders.csv"))
    X, col_order = build_feature_matrix(o, ref.PRODUCT_CATEGORIES, TRUCKS)
    y = o["base_freight_cost"].to_numpy()
    idx_tr, idx_te = train_test_split(np.arange(len(o)), test_size=0.2, random_state=seed)

    results = {}
    for name, cls in (("previous_direct_xgb", DirectM1), ("hybrid_baseline_xgb", HybridM1)):
        m = cls(seed).fit(X.iloc[idx_tr], o.iloc[idx_tr], y[idx_tr])
        res = {"holdout": m1_metrics(y[idx_te], *m.predict(X.iloc[idx_te], o.iloc[idx_te]))}
        for sname, sdf in stress_sets(o.iloc[idx_te], np.random.default_rng(seed + 1)).items():
            sX, _ = build_feature_matrix(sdf, ref.PRODUCT_CATEGORIES, TRUCKS)
            res[sname] = m1_metrics(sdf["base_freight_cost"].to_numpy(), *m.predict(sX, sdf))
        results[name] = res
    print(f"  {'test set':<26}{'previous MAPE':>15}{'hybrid MAPE':>13}{'hybrid cover':>14}")
    for k in results["hybrid_baseline_xgb"]:
        a, b = results["previous_direct_xgb"][k], results["hybrid_baseline_xgb"][k]
        print(f"  {k:<26}{a['mape_pct']:>14.2f}%{b['mape_pct']:>12.2f}%{b['interval_coverage']*100:>13.1f}%")
    meta["model_1"] = {"design": "baseline (fuel+toll+driver) x XGBoost ratio",
                       "comparison": results,
                       "n_features": len(col_order)}

    # Out-of-fold predictions for every order -> honest inputs for Model 2.
    print("  computing 5-fold out-of-fold predictions for Model 2 ...")
    oof = np.zeros((len(o), 3))
    for tr, ho in KFold(5, shuffle=True, random_state=seed).split(X):
        m = HybridM1(seed).fit(X.iloc[tr], o.iloc[tr], y[tr])
        oof[ho] = np.column_stack(m.predict(X.iloc[ho], o.iloc[ho]))
    o["m1_pred"], o["m1_p10"], o["m1_p90"] = oof[:, 0], oof[:, 1], oof[:, 2]

    # Final Model 1 on all rows.
    final = HybridM1(seed).fit(X, o, y)

    # Plausible-price bounds learned from the training data.
    cpk = o["base_freight_cost"] / o["distance_km"]
    price_bounds = {
        # half the cheapest / triple the dearest per-km cost seen; wide on purpose,
        # these catch impossible outputs, not ordinary variation
        "cost_per_km_min": round(float(cpk.quantile(0.001)) * 0.5, 2),
        "cost_per_km_max": round(float(cpk.quantile(0.999)) * 3.0, 2),
    }

    # ============================ MODEL 2 ============================
    print("=== Model 2: invoice risk classifier ===")
    inv = clean_invoices(pd.read_csv(f"{out}/invoices.csv"))
    gt = pd.read_csv(f"{out}/_ground_truth_audit.csv")
    inv = inv.merge(gt[["order_id", "anomaly_type"]], on="order_id", how="left")
    inv = inv.merge(o[["order_id", "m1_pred", "m1_p10", "m1_p90", "billable_weight_kg",
                       "distance_km", "quoted_days", "ideal_days", "expected_fuel_price"]],
                    on="order_id", suffixes=("", "_o"))

    ref_fuel = float(o["expected_fuel_price"].mean())

    def add_features(d, scale=1.0, deflate=False):
        """Model 2 features from OOF Model 1 outputs; `scale` multiplies every
        rupee amount (billed and fair) to simulate a nationwide price rise."""
        d = d.copy()
        pred = (d["m1_pred"] * scale).clip(lower=1.0)
        p90 = (d["m1_p90"] * scale).clip(lower=1.0)
        billed = d["actual_billed_amount"] * scale
        mismatch = billed - pred
        comm = d["commercial_delay_days"].clip(lower=0.0)
        tons = (d["billable_weight_kg"] / 1000.0).clip(lower=0.1)
        d["actual_billed_amount"] = billed
        d["model_a_predicted_cost"] = pred
        d["cost_mismatch"] = mismatch
        d["cost_mismatch_ratio"] = mismatch / pred
        d["cost_mismatch_upper"] = np.maximum(0.0, billed - p90)
        d["cost_mismatch_upper_ratio"] = d["cost_mismatch_upper"] / pred
        d["billed_to_p90_ratio"] = billed / p90
        d["excess_rate_per_kg"] = mismatch / d["billable_weight_kg"].clip(lower=1.0)
        d["excess_rate_per_ton_km"] = mismatch / (tons * d["distance_km"].clip(lower=10.0))
        d["cost_mismatch_delay_skew"] = d["cost_mismatch_ratio"] / (comm + 0.1)
        d["zero_delay_p90_breach"] = ((billed > p90) & (comm <= 0.2)).astype(float)
        if deflate:
            defl = (d["expected_fuel_price"] * scale) / ref_fuel
            for c in M2_RUPEE_FEATURES:
                d[c] = d[c] / defl
        return d

    inv_base = inv.dropna(subset=["actual_billed_amount", "m1_pred", "expected_fuel_price"]).reset_index(drop=True)
    frames = {False: add_features(inv_base).dropna(subset=M2_FEATURES_FULL),
              True: add_features(inv_base, deflate=True).dropna(subset=M2_FEATURES_FULL)}
    inv = frames[False]
    yc = inv["requires_manual_review"].astype(int).to_numpy()
    i_tr, i_te = train_test_split(np.arange(len(inv)), test_size=0.25, random_state=seed, stratify=yc)
    # every price (bills, fair cost, diesel) +25%
    inflated = {k: add_features(inv_base.loc[inv.index[i_te]], scale=1.25, deflate=k) for k in (False, True)}

    mono = {"cost_mismatch_ratio": 1, "cost_mismatch_upper_ratio": 1,
            "billed_to_p90_ratio": 1, "zero_delay_p90_breach": 1}
    candidates = {                       # name: (features, monotone, deflate)
        "previous_full_features": (M2_FEATURES_FULL, None, False),
        "scale_free_features": (M2_FEATURES_SCALE_FREE, None, False),
        "scale_free_monotone": (M2_FEATURES_SCALE_FREE, mono, False),
        "deflated_full_features": (M2_FEATURES_FULL, None, True),
        "deflated_full_monotone": (M2_FEATURES_FULL, mono, True),
    }
    m2_results = {}
    ytr, yte = yc[i_tr], yc[i_te]
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    for name, (feats, mon, defl) in candidates.items():
        F = frames[defl]
        clf = make_clf(seed, spw, feats, mon).fit(F.iloc[i_tr][feats], ytr)
        proba = clf.predict_proba(F.iloc[i_te][feats])[:, 1]
        p = (proba >= M2_THRESHOLD).astype(int)
        proba_inf = clf.predict_proba(inflated[defl][feats])[:, 1]
        p_inf = (proba_inf >= M2_THRESHOLD).astype(int)
        at = inv.iloc[i_te]["anomaly_type"].to_numpy()
        per_type = {t: round(float(p[(yte == 1) & (at == t)].mean()), 3)
                    for t in sorted(set(at[yte == 1]))}
        m2_results[name] = {
            "pr_auc": round(float(average_precision_score(yte, proba)), 3),
            "precision": round(float(precision_score(yte, p)), 3),
            "recall": round(float(recall_score(yte, p)), 3),
            "f1": round(float(f1_score(yte, p)), 3),
            "per_anomaly_type_recall": per_type,
            "price_rise_25pct": {
                "verdicts_changed_pct": round(float((p != p_inf).mean() * 100), 2),
                "false_positive_rate_before": round(float(p[yte == 0].mean()), 4),
                "false_positive_rate_after": round(float(p_inf[yte == 0].mean()), 4),
            },
        }
    print(f"  {'variant':<26}{'PR-AUC':>8}{'recall':>8}{'prec':>7}{'F1':>7}{'verdicts changed @+25% prices':>32}")
    for k, v in m2_results.items():
        print(f"  {k:<26}{v['pr_auc']:>8}{v['recall']:>8}{v['precision']:>7}{v['f1']:>7}"
              f"{v['price_rise_25pct']['verdicts_changed_pct']:>30}%")

    # Pick: among variants whose verdicts stay stable under a uniform +25% price
    # rise (<0.5% changed), the best PR-AUC; prefer monotone on ties (<0.005).
    stable = {n: r for n, r in m2_results.items()
              if r["price_rise_25pct"]["verdicts_changed_pct"] < 0.5}
    pool = stable or m2_results
    top = max(r["pr_auc"] for r in pool.values())
    chosen = sorted((n for n in pool if pool[n]["pr_auc"] >= top - 0.005),
                    key=lambda n: (candidates[n][1] is None, -pool[n]["pr_auc"]))[0]
    feats, mon, defl = candidates[chosen]
    print(f"  -> deploying '{chosen}'")
    spw_full = (yc == 0).sum() / max((yc == 1).sum(), 1)
    clf_full = make_clf(seed, spw_full, feats, mon).fit(frames[defl][feats], yc)
    meta["model_2"] = {"chosen_variant": chosen, "features": feats, "threshold": M2_THRESHOLD,
                       "rupee_deflator": defl,
                       "comparison": m2_results,
                       "trained_on": "out-of-fold Model 1 predictions (no in-sample leakage)"}

    # Billed-amount bounds from GENUINE invoices only (OOF fair cost).
    legit = inv[inv["requires_manual_review"] == 0]
    low_ratio = legit["actual_billed_amount"] / legit["m1_p10"].clip(lower=1.0)
    high_ratio = legit["actual_billed_amount"] / legit["m1_p90"].clip(lower=1.0)
    high_abs = legit["actual_billed_amount"] - legit["m1_p90"]
    price_bounds.update({
        # 20% below the lowest genuine bill-to-lower-bound ratio ever seen
        "billed_low_ratio": round(float(low_ratio.min()) * 0.8, 3),
        # both: 50% above the highest genuine ratio AND 50% above the biggest genuine rupee excess
        "billed_high_ratio": round(float(high_ratio.max()) * 1.5, 3),
        "billed_high_abs": round(float(high_abs.max()) * 1.5, 0),
        "derived_from": f"{len(legit):,} genuine invoices, out-of-fold fair cost",
    })
    print(f"  price bounds: {price_bounds}")

    # ============================ SAVE ============================
    import sklearn
    import xgboost
    tr_ranges = {c: {"min": round(float(o[c].min()), 2), "max": round(float(o[c].max()), 2)}
                 for c in M1_NUM}
    m1_meta = {
        "design": "predicted = baseline(fuel+toll+driver) * xgb_ratio",
        "numeric": M1_NUM, "product_categories": ref.PRODUCT_CATEGORIES, "truck_types": TRUCKS,
        "column_order": col_order, "target": "base_freight_cost / baseline",
        "baseline_constants": BASELINE_CONSTANTS, "price_bounds": price_bounds,
        "training_ranges": tr_ranges, "prediction_interval": {"alphas": [0.10, 0.90]},
    }
    m1_bundle = {"point": final.point, "p10": final.p10, "p90": final.p90, "meta": m1_meta}
    deflator = ({"reference_fuel_price": round(ref_fuel, 4), "features": M2_RUPEE_FEATURES}
                if defl else None)
    m2_bundle = {"model": clf_full, "features": feats, "threshold": M2_THRESHOLD,
                 "monotone_constraints": mon or {}, "rupee_deflator": deflator}
    joblib.dump(m1_bundle, f"{mdir}/model_1_freight_cost.pkl")
    joblib.dump(m2_bundle, f"{mdir}/model_2_invoice_risk.pkl")
    for name, mdl in (("model_1_freight_cost", final.point), ("model_1_freight_cost_p10", final.p10),
                      ("model_1_freight_cost_p90", final.p90), ("model_2_invoice_risk", clf_full)):
        mdl.get_booster().save_model(f"{mdir}/{name}.json")
    json.dump(m1_meta, open(f"{mdir}/model_1_features.json", "w"), indent=2)
    json.dump({"features": feats, "threshold": M2_THRESHOLD, "target": "requires_manual_review",
               "monotone_constraints": mon or {}, "rupee_deflator": deflator,
               "notes": "Clean invoices first (dedup invoice_id, upper() vendor_id, drop "
                        "billed<=0). Features are computed from Model 1 outputs; see "
                        "backend/audit_core.build_model2_features. anomaly_type is never a feature."},
              open(f"{mdir}/model_2_features.json", "w"), indent=2)

    # Sanity: the saved bundle reproduces the in-memory model.
    probe = o.iloc[0].to_dict()
    reloaded = FreightCostModel(joblib.load(f"{mdir}/model_1_freight_cost.pkl"))
    assert abs(reloaded.predict(probe) - final.predict(X.iloc[[0]], o.iloc[[0]])[0][0]) < 1e-3

    meta["versions"] = {"xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__}
    json.dump(meta, open(f"{mdir}/metadata.json", "w"), indent=2)
    print(f"\nExported artifacts to {mdir}/")
    for f in sorted(os.listdir(mdir)):
        print("  ", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.outdir, a.seed)
