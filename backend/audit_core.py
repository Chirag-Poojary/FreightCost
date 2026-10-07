"""
Storage-agnostic audit core -- confidence gate + Model 1/2 inference.

Pulled out of confidence_gate_benchmark.py so the same logic isn't
duplicated between the CLI benchmark and the FastAPI backend. Nothing here
imports pandas-CSV-specific code or Supabase-specific code: it takes plain
dicts in and returns plain dicts out, so either caller can supply data from
wherever it actually lives (local CSV for the CLI, Supabase for the API).
"""
import json
import os
import sys

# Model classes + logic checks live in code/freight_models.py, shared with training.
_CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
from freight_models import (FreightCostModel, InvoiceRiskModel,  # noqa: E402,F401
                            check_inputs, load_bundle)

Model1 = FreightCostModel   # backwards-compatible name

CRITICAL = ["order_id", "vendor_id", "total", "actual_days"]

MODEL2_FEATURES = [
    "actual_billed_amount", "model_a_predicted_cost", "cost_mismatch",
    "actual_days", "true_delay_days", "commercial_delay_days",
    "adverse_weather_days", "vendor_padding_ratio",
    "vendor_historical_risk_score",
    # Phase 3: weight discrepancy anomaly detection features
    "cost_mismatch_ratio",
    "cost_mismatch_upper",
    "cost_mismatch_upper_ratio",
    "billed_to_p90_ratio",
    "excess_rate_per_kg",
    "excess_rate_per_ton_km",
    "cost_mismatch_delay_skew",
    "zero_delay_p90_breach"
]


def _num(s):
    import re
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)))
    except (ValueError, TypeError):
        return None


def confidence_gate(pred, order_exists=None, vendor_exists=None, allow_new_vendors=True, allow_new_orders=True):
    """4-tier validation. `order_exists`/`vendor_exists` are callables
    (id -> bool) so the caller decides where that lookup happens --
    an in-memory pandas index for the CLI, a Supabase point query for the
    API. Returns (passed: bool, reason: str | None); first failing tier wins."""
    missing = [f for f in CRITICAL if not pred.get(f)]
    if missing:
        return False, f"missing_critical_field:{','.join(missing)}"

    if not allow_new_orders and order_exists and not order_exists(pred["order_id"]):
        return False, "order_id_not_found"
    if not allow_new_vendors and vendor_exists and not vendor_exists(pred["vendor_id"]):
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


def build_model2_features(pred, order, context, m1):
    """`order`: dict with distance_km/ideal_days/quoted_days/... (from
    orders table). `context`: dict with adverse_weather_days,
    vendor_padding_ratio, vendor_historical_risk_score -- looked up
    separately since OCR can't read these off a document. cost_mismatch and
    both delay fields are computed from the OCR-extracted total/actual_days,
    not copied from any ground truth, so an OCR slip genuinely changes the
    score, same as it would in production."""
    predicted_cost, p10, p90 = m1.predict_interval(order)
    ocr_total = _num(pred["total"])
    ocr_days = _num(pred["actual_days"])
    shap_info = m1.explain_prediction(order, top_k=3)

    pred_safe = max(1.0, float(predicted_cost))
    billed = float(ocr_total)
    days = float(ocr_days)
    mismatch = billed - pred_safe
    p90_safe = max(1.0, float(p90))
    cost_upper = max(0.0, billed - p90)

    billable_wt = max(1.0, float(order.get("billable_weight_kg", 1.0)))
    dist = max(10.0, float(order.get("distance_km", 10.0)))
    weight_tons = billable_wt / 1000.0

    comm_delay = max(0.0, days - float(order.get("quoted_days", 0.0)))
    true_delay = days - float(order.get("ideal_days", 0.0))

    cost_mismatch_ratio = round(mismatch / pred_safe, 4)
    cost_mismatch_upper_ratio = round(cost_upper / pred_safe, 4)
    billed_to_p90_ratio = round(billed / p90_safe, 4)
    excess_rate_per_kg = round(mismatch / billable_wt, 4)
    excess_rate_per_ton_km = round(mismatch / (weight_tons * dist), 4)
    cost_mismatch_delay_skew = round(cost_mismatch_ratio / (comm_delay + 0.1), 4)
    zero_delay_p90_breach = 1.0 if (billed > p90 and comm_delay <= 0.2) else 0.0

    return {
        "actual_billed_amount": billed,
        "model_a_predicted_cost": predicted_cost,
        "cost_interval_lower": p10,
        "cost_interval_upper": p90,
        "cost_mismatch": mismatch,
        "cost_mismatch_upper": cost_upper,
        "top_drivers": shap_info["breakdown"],
        "actual_days": days,
        "true_delay_days": true_delay,
        "commercial_delay_days": comm_delay,
        "adverse_weather_days": float(context.get("adverse_weather_days", 0.0)),
        "vendor_padding_ratio": float(context.get("vendor_padding_ratio", 0.0)),
        "vendor_historical_risk_score": float(context["vendor_historical_risk_score"]),
        # Phase 3: weight discrepancy anomaly detection features
        "cost_mismatch_ratio": cost_mismatch_ratio,
        "cost_mismatch_upper_ratio": cost_mismatch_upper_ratio,
        "billed_to_p90_ratio": billed_to_p90_ratio,
        "excess_rate_per_kg": excess_rate_per_kg,
        "excess_rate_per_ton_km": excess_rate_per_ton_km,
        "cost_mismatch_delay_skew": cost_mismatch_delay_skew,
        "zero_delay_p90_breach": zero_delay_p90_breach,
    }


def score_model2(features, m2, fuel_price=None):
    """`m2` is an InvoiceRiskModel; `fuel_price` is the order's diesel price,
    used to express rupee features in training-period rupees."""
    return m2.predict_proba(features, fuel_price)


def causal_vendor_risk(total_invoices, flagged_invoices, alpha=2, prior=0.06):
    """Same Laplace-smoothed formula as build_dataset.py's add_vendor_risk().
    Called with the vendor's stats BEFORE this invoice, to keep it causal.
    If the vendor is new (total_invoices == 0), defaults to 0.0."""
    if not total_invoices or total_invoices <= 0:
        return 0.0
    return (flagged_invoices + alpha * prior) / (total_invoices + alpha)
