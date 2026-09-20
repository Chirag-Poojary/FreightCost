"""
Storage-agnostic audit core -- confidence gate + Model 1/2 inference.

Pulled out of confidence_gate_benchmark.py so the same logic isn't
duplicated between the CLI benchmark and the FastAPI backend. Nothing here
imports pandas-CSV-specific code or Supabase-specific code: it takes plain
dicts in and returns plain dicts out, so either caller can supply data from
wherever it actually lives (local CSV for the CLI, Supabase for the API).
"""
import json

CRITICAL = ["order_id", "vendor_id", "total", "actual_days"]

MODEL2_FEATURES = ["actual_billed_amount", "model_a_predicted_cost", "cost_mismatch",
                   "actual_days", "true_delay_days", "commercial_delay_days",
                   "adverse_weather_days", "vendor_padding_ratio",
                   "vendor_historical_risk_score"]


def _num(s):
    import re
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)))
    except (ValueError, TypeError):
        return None


def confidence_gate(pred, order_exists, vendor_exists):
    """4-tier validation. `order_exists`/`vendor_exists` are callables
    (id -> bool) so the caller decides where that lookup happens --
    an in-memory pandas index for the CLI, a Supabase point query for the
    API. Returns (passed: bool, reason: str | None); first failing tier wins."""
    missing = [f for f in CRITICAL if not pred.get(f)]
    if missing:
        return False, f"missing_critical_field:{','.join(missing)}"

    if not order_exists(pred["order_id"]):
        return False, "order_id_not_found"
    if not vendor_exists(pred["vendor_id"]):
        return False, "vendor_id_not_found"

    total = _num(pred["total"])
    days = _num(pred["actual_days"])
    if total is None or days is None:
        return False, "unparseable_amount_or_days"

    parts = [_num(pred.get(f)) for f in ("freight_base", "detention", "toll")]
    if all(p is not None for p in parts):
        drift = abs(sum(parts) - total)
        if drift > max(2.0, total * 0.02):
            return False, "arithmetic_mismatch"

    if not (500 < total <= 250000):
        return False, "amount_out_of_plausible_range"
    if not (0.2 <= days <= 30.0):
        return False, "days_out_of_plausible_range"

    return True, None


def _onehot_row(order, feature_meta):
    row = {k: order[k] for k in
          ("distance_km", "ideal_days", "quoted_days", "billable_weight_kg",
           "expected_fuel_price", "expected_weather_score")}
    for c in feature_meta["product_categories"]:
        row[f"cat_{c}"] = 1 if order.get("product_category") == c else 0
    for c in feature_meta["truck_types"]:
        row[f"truck_{c}"] = 1 if order.get("truck_type") == c else 0
    return row


class Model1:
    """Wraps the raw XGBRegressor so callers just pass an order dict.
    Reads product_categories/truck_types straight from model_1_features.json
    -- never hardcoded, so a category list mismatch is structurally
    impossible instead of a silent prediction bug."""
    def __init__(self, booster, feature_meta):
        self.booster = booster
        self.meta = feature_meta
        self.columns = feature_meta["column_order"]

    def predict(self, order):
        import pandas as pd
        row = _onehot_row(order, self.meta)
        for c in self.columns:
            row.setdefault(c, 0)
        x = pd.DataFrame([row])[self.columns]
        return float(self.booster.predict(x)[0])


def build_model2_features(pred, order, context, m1):
    """`order`: dict with distance_km/ideal_days/quoted_days/... (from
    orders table). `context`: dict with adverse_weather_days,
    vendor_padding_ratio, vendor_historical_risk_score -- looked up
    separately since OCR can't read these off a document. cost_mismatch and
    both delay fields are computed from the OCR-extracted total/actual_days,
    not copied from any ground truth, so an OCR slip genuinely changes the
    score, same as it would in production."""
    predicted_cost = m1.predict(order)
    ocr_total = _num(pred["total"])
    ocr_days = _num(pred["actual_days"])
    return {
        "actual_billed_amount": ocr_total,
        "model_a_predicted_cost": predicted_cost,
        "cost_mismatch": ocr_total - predicted_cost,
        "actual_days": ocr_days,
        "true_delay_days": ocr_days - order["ideal_days"],
        "commercial_delay_days": ocr_days - order["quoted_days"],
        "adverse_weather_days": context.get("adverse_weather_days", 0.0),
        "vendor_padding_ratio": context.get("vendor_padding_ratio", 0.0),
        "vendor_historical_risk_score": context["vendor_historical_risk_score"],
    }


def score_model2(features, m2):
    import pandas as pd
    x = pd.DataFrame([features])[MODEL2_FEATURES]
    return float(m2.predict_proba(x)[0, 1])


def causal_vendor_risk(total_invoices, flagged_invoices, alpha=2, prior=0.06):
    """Same Laplace-smoothed formula as build_dataset.py's add_vendor_risk().
    Called with the vendor's stats BEFORE this invoice, to keep it causal."""
    return (flagged_invoices + alpha * prior) / (total_invoices + alpha)
