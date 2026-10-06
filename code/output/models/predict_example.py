"""
Minimal load-and-predict example for the exported bundles.

    cd code/output/models && python predict_example.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from freight_models import FreightCostModel, InvoiceRiskModel, check_inputs, load_bundle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M1 = FreightCostModel(load_bundle(os.path.join(HERE, "model_1_freight_cost.pkl")))
M2 = InvoiceRiskModel(load_bundle(os.path.join(HERE, "model_2_invoice_risk.pkl")))

order = {
    "distance_km": 1410.0, "ideal_days": 0.98, "quoted_days": 1.2,
    "billable_weight_kg": 12000.0, "expected_fuel_price": 97.8,
    "expected_weather_score": 0.0, "product_category": "Steel Coils",
    "truck_type": "10-wheeler",
}
assert not check_inputs(order), check_inputs(order)

fair, lo, hi = M1.predict_interval(order)
print(f"Model 1 fair cost: Rs {fair:,.0f}  (80% range Rs {lo:,.0f} - Rs {hi:,.0f})")
print("  build-up:", {d["feature"]: round(d["impact"]) for d in M1.explain_prediction(order)["breakdown"]})
print("  prediction check:", M1.check_prediction(order, fair)["status"])

for billed in (fair * 1.02, fair * 1.45, fair * 0.4):
    print(f"Billed Rs {billed:,.0f}: price check = {M1.check_billed(billed, fair, lo, hi)['status']}")

# Model 2 features are built by backend/audit_core.build_model2_features; scored with:
#   M2.predict_proba(features_dict, fuel_price=order["expected_fuel_price"])
