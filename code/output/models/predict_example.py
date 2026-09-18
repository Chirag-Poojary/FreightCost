"""Minimal inference demo -- load the exported models and predict.
Run:  python predict_example.py
For a website, wrap this in a request handler (Flask/FastAPI) instead.
"""
import json, joblib, numpy as np, pandas as pd

M1 = joblib.load("model_1_freight_cost.joblib")
M1F = json.load(open("model_1_features.json"))
M2 = joblib.load("model_2_invoice_risk.joblib")
M2F = json.load(open("model_2_features.json"))

def predict_freight_cost(distance_km, ideal_days, quoted_days, weight_kg,
                         dimensional_weight_kg, expected_fuel_price,
                         expected_weather_score, product_category):
    # truck_type from weight (same rule as the pipeline)
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
    X = pd.DataFrame([row])[M1F["column_order"]]
    return float(M1.predict(X)[0])

def score_invoice_risk(**features):
    X = pd.DataFrame([{f: features[f] for f in M2F["features"]}])
    proba = float(M2.predict_proba(X)[0, 1])
    return proba, int(proba >= M2F["threshold"])

if __name__ == "__main__":
    cost = predict_freight_cost(1200, 3.1, 3.5, 8000, 6000, 91.5, 2,
                                "Steel Coils")
    print(f"predicted should-cost: Rs {cost:,.0f}")
    predicted_cost = cost
    proba, flag = score_invoice_risk(
        actual_billed_amount=cost*1.15, model_a_predicted_cost=predicted_cost,
        cost_mismatch=cost*0.15, actual_days=4.2, true_delay_days=1.1,
        commercial_delay_days=0.7, adverse_weather_days=1,
        vendor_padding_ratio=1.13, vendor_historical_risk_score=0.08)
    print(f"invoice risk proba={proba:.3f}  flag_for_review={flag}")
