"""Minimal inference demo -- load the exported models and predict.
Run:  python predict_example.py
For a website, wrap this in a request handler (Flask/FastAPI) instead.
"""
import json, joblib, numpy as np, pandas as pd
import xgboost as xgb
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

M1 = joblib.load("model_1_freight_cost.joblib")
M1_P10 = joblib.load("model_1_freight_cost_p10.joblib")
M1_P90 = joblib.load("model_1_freight_cost_p90.joblib")
M1F = json.load(open("model_1_features.json"))
M2 = joblib.load("model_2_invoice_risk.joblib")
M2F = json.load(open("model_2_features.json"))

def _prepare_m1_df(distance_km, ideal_days, quoted_days, weight_kg,
                    dimensional_weight_kg, expected_fuel_price,
                    expected_weather_score, product_category):
    truck = ("6-wheeler" if weight_kg <= 9000
             else "10-wheeler" if weight_kg <= 16000 else "12-wheeler")
    billable = max(weight_kg, dimensional_weight_kg)
    row = {"distance_km": distance_km, "ideal_days": ideal_days,
           "quoted_days": quoted_days, "billable_weight_kg": billable,
           "expected_fuel_price": expected_fuel_price,
           "expected_weather_score": expected_weather_score}
    for c in M1F["product_categories"]:
        row[f"cat_{c}"] = int(product_category == c)
    for t in M1F["truck_types"]:
        row[f"truck_{t}"] = int(truck == t)
    return pd.DataFrame([row])[M1F["column_order"]]

def predict_freight_cost(distance_km, ideal_days, quoted_days, weight_kg,
                         dimensional_weight_kg, expected_fuel_price,
                         expected_weather_score, product_category):
    X = _prepare_m1_df(distance_km, ideal_days, quoted_days, weight_kg,
                       dimensional_weight_kg, expected_fuel_price,
                       expected_weather_score, product_category)
    point = float(M1.predict(X)[0])
    p10 = float(M1_P10.predict(X)[0])
    p90 = float(M1_P90.predict(X)[0])
    lower = min(p10, p90, point)
    upper = max(p10, p90, point)
    return point, lower, upper

def explain_freight_cost(distance_km, ideal_days, quoted_days, weight_kg,
                         dimensional_weight_kg, expected_fuel_price,
                         expected_weather_score, product_category, top_k=3):
    X = _prepare_m1_df(distance_km, ideal_days, quoted_days, weight_kg,
                       dimensional_weight_kg, expected_fuel_price,
                       expected_weather_score, product_category)
    dmat = xgb.DMatrix(X)
    booster = getattr(M1, "get_booster", lambda: M1)()
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    cols = M1F["column_order"]
    raw = dict(zip(cols, contribs[:-1]))
    base_val = float(contribs[-1])
    unified = {}
    cat_sum = sum(v for k, v in raw.items() if k.startswith("cat_"))
    truck_sum = sum(v for k, v in raw.items() if k.startswith("truck_"))
    for k, v in raw.items():
        if not k.startswith("cat_") and not k.startswith("truck_"):
            unified[k] = float(v)
    if cat_sum: unified["product_category"] = float(cat_sum)
    if truck_sum: unified["truck_type"] = float(truck_sum)
    sorted_drivers = sorted(unified.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return base_val, [{"feature": k, "impact": round(v, 2)} for k, v in sorted_drivers[:top_k]]

def score_invoice_risk(actual_billed_amount, model_a_predicted_cost, cost_mismatch=None,
                       actual_days=None, true_delay_days=None, commercial_delay_days=None,
                       adverse_weather_days=0.0, vendor_padding_ratio=1.10,
                       vendor_historical_risk_score=0.05, p90=None,
                       billable_weight_kg=10000.0, distance_km=1000.0, **extra):
    pred_cost = max(1.0, float(model_a_predicted_cost))
    billed = float(actual_billed_amount)
    mismatch = float(cost_mismatch) if cost_mismatch is not None else (billed - pred_cost)
    days = float(actual_days) if actual_days is not None else 3.0
    true_delay = float(true_delay_days) if true_delay_days is not None else 0.0
    comm_delay = max(0.0, float(commercial_delay_days)) if commercial_delay_days is not None else 0.0

    p90_val = float(p90) if p90 is not None else pred_cost * 1.05
    p90_safe = max(1.0, p90_val)
    cost_upper = max(0.0, billed - p90_val)

    weight_safe = max(1.0, float(billable_weight_kg))
    dist_safe = max(10.0, float(distance_km))
    weight_tons = weight_safe / 1000.0

    cost_mismatch_ratio = mismatch / pred_cost
    cost_mismatch_upper_ratio = cost_upper / pred_cost
    billed_to_p90_ratio = billed / p90_safe
    excess_rate_per_kg = mismatch / weight_safe
    excess_rate_per_ton_km = mismatch / (weight_tons * dist_safe)
    cost_mismatch_delay_skew = cost_mismatch_ratio / (comm_delay + 0.1)
    zero_delay_p90_breach = 1.0 if (billed > p90_val and comm_delay <= 0.2) else 0.0

    row = {
        "actual_billed_amount": billed,
        "model_a_predicted_cost": pred_cost,
        "cost_mismatch": mismatch,
        "actual_days": days,
        "true_delay_days": true_delay,
        "commercial_delay_days": comm_delay,
        "adverse_weather_days": float(adverse_weather_days),
        "vendor_padding_ratio": float(vendor_padding_ratio),
        "vendor_historical_risk_score": float(vendor_historical_risk_score),
        "cost_mismatch_ratio": round(cost_mismatch_ratio, 4),
        "cost_mismatch_upper": round(cost_upper, 2),
        "cost_mismatch_upper_ratio": round(cost_mismatch_upper_ratio, 4),
        "billed_to_p90_ratio": round(billed_to_p90_ratio, 4),
        "excess_rate_per_kg": round(excess_rate_per_kg, 4),
        "excess_rate_per_ton_km": round(excess_rate_per_ton_km, 4),
        "cost_mismatch_delay_skew": round(cost_mismatch_delay_skew, 4),
        "zero_delay_p90_breach": zero_delay_p90_breach,
    }
    for k, v in extra.items():
        if k in M2F["features"]:
            row[k] = v
    X = pd.DataFrame([row])[M2F["features"]]
    proba = float(M2.predict_proba(X)[0, 1])
    return proba, int(proba >= M2F["threshold"])

if __name__ == "__main__":
    cost, p10, p90 = predict_freight_cost(1200, 3.1, 3.5, 8000, 6000, 91.5, 2, "Steel Coils")
    base_val, drivers = explain_freight_cost(1200, 3.1, 3.5, 8000, 6000, 91.5, 2, "Steel Coils")
    print(f"Predicted Should-Cost: ?{cost:,.2f}")
    print(f"80% Uncertainty Band:  ?{p10:,.2f} ? ?{p90:,.2f}")
    print(f"Base Expected Value:   ?{base_val:,.2f}")
    print("Top TreeSHAP Drivers:")
    for d in drivers:
        sign = "+" if d["impact"] > 0 else ""
        print(f"  - {d['feature']:<22} {sign}?{d['impact']:,.2f}")

    proba, flag = score_invoice_risk(
        actual_billed_amount=cost*1.25, model_a_predicted_cost=cost,
        actual_days=3.5, ideal_days=3.1, quoted_days=3.5,
        adverse_weather_days=0, vendor_padding_ratio=1.13,
        vendor_historical_risk_score=0.08, p90=p90,
        billable_weight_kg=8000, distance_km=1200)
    print(f"Invoice Risk Probability: {proba:.1%} | Flag for Review: {bool(flag)}")
