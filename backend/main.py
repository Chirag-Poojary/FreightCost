"""
FastAPI backend for the invoice audit web app.

Audit flow (two steps, human-in-the-loop):

  1. POST /api/extract   image -> cloud vision model -> fields + list of
                         required fields it could not read + validation checks.
                         No ML runs here.
  2. POST /api/score     the user-confirmed fields (AI-extracted + manually
                         typed) -> confidence gate -> Model 1 + Model 2.
                         Refuses to score while any required field is empty.

POST /api/audit is kept for scripts: it extracts and scores in one call, but
only scores when nothing required is missing -- otherwise it returns
`status: "needs_input"` with the missing fields.

Weather and diesel inputs come from live_context.py (Open-Meteo + live retail
diesel) rather than the 2023-2025 training caches, with drift warnings when a
live value falls outside what the models were trained on.
"""
import json
import os
import sys
import tempfile
from datetime import timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.abspath(os.path.join(_CURR_DIR, ".."))

load_dotenv(os.path.join(_CURR_DIR, ".env"))
load_dotenv(os.path.join(_BUNDLE_DIR, ".env"))
load_dotenv()

if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)
_EXTRACT_DIR = os.path.join(_BUNDLE_DIR, "extraction_scripts")
if os.path.isdir(_EXTRACT_DIR) and _EXTRACT_DIR not in sys.path:
    sys.path.insert(0, _EXTRACT_DIR)

import extractors                                                 # noqa: E402
import live_context                                               # noqa: E402
from audit_core import (FreightCostModel, InvoiceRiskModel,  # noqa: E402
                        build_model2_features, causal_vendor_risk, check_inputs,
                        confidence_gate, load_bundle, score_model2)
from db import Db                                                 # noqa: E402
from schemas import (AuditResult, ContextResponse, DashboardStats,  # noqa: E402
                     ExtractResponse, QuoteRequest, QuoteResponse, ScoreRequest,
                     VendorRisk)

MODEL_DIR = os.path.join(_BUNDLE_DIR, "code", "output", "models")

from fastapi.responses import JSONResponse
from fastapi import Request

app = FastAPI(title="FreightCost Audit API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"}
    )

_db = None
_m1 = None
_m2 = None

# Which fields to highlight when a gate tier fails.
GATE_FIELD_MAP = {
    "order_id_not_found": ["order_id"],
    "vendor_id_not_found": ["vendor_id"],
    "unparseable_amount_or_days": ["total", "actual_days"],
    "arithmetic_mismatch": ["freight_base", "detention", "toll", "total"],
    "amount_out_of_plausible_range": ["total"],
    "days_out_of_plausible_range": ["actual_days"],
}


def get_db():
    global _db
    if _db is None:
        try:
            _db = Db()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Database error: {e}")
    return _db


@app.on_event("startup")
def load_models():
    global _m1, _m2
    _m1 = FreightCostModel(load_bundle(os.path.join(MODEL_DIR, "model_1_freight_cost.pkl")))
    _m2 = InvoiceRiskModel(load_bundle(os.path.join(MODEL_DIR, "model_2_invoice_risk.pkl")))


@app.get("/api/health")
def health():
    try:
        get_db()
        db_status = "connected"
    except Exception:
        db_status = "not_connected"
    return {"status": "ok", "models_loaded": _m1 is not None and _m2 is not None,
            "database": db_status, "extractors": extractors.available_extractors()}


@app.get("/api/extractors")
def list_extractors():
    return {"extractors": extractors.available_extractors(),
            "required_fields": extractors.REQUIRED_FIELDS,
            "all_fields": extractors.FIELDS}


# ------------------------------------------------------------------ checks
def _field_checks(fields, db):
    """Per-field problems the user should see before scoring. Does not block
    extraction; /api/score enforces the gate."""
    issues = {}
    order = None
    if fields.get("order_id"):
        if db.order_exists(fields["order_id"]):
            order = db.get_order(fields["order_id"])
        else:
            issues["order_id"] = "Not found in the order master (ERP)."
    if fields.get("vendor_id") and not db.vendor_exists(fields["vendor_id"]):
        issues["vendor_id"] = "Not a registered carrier."
    if order and fields.get("vendor_id") and order.get("vendor_id") \
            and order["vendor_id"] != fields["vendor_id"]:
        issues["vendor_id"] = (f"Order {fields['order_id']} was booked with "
                               f"{order['vendor_id']}, not {fields['vendor_id']}.")
    parts = [fields.get(k) for k in ("freight_base", "detention", "toll")]
    total = fields.get("total")
    if total is not None and all(p is not None for p in parts):
        drift = abs(sum(parts) - total)
        if drift > max(2.0, total * 0.02):
            issues["total"] = (f"Line items add up to {sum(parts):,.2f}, but the total "
                               f"reads {total:,.2f}. Check the amounts.")
    if total is not None and not (500 < total <= 250000):
        issues.setdefault("total", "Outside the plausible range (Rs 500 to Rs 2,50,000).")
    if fields.get("actual_days") is not None and not (0.2 <= fields["actual_days"] <= 30):
        issues["actual_days"] = "Outside the plausible range (0.2 to 30 days)."

    summary = None
    if order:
        hub = live_context.hubs()
        summary = {
            "order_id": fields["order_id"],
            "vendor_id": order.get("vendor_id"),
            "origin": hub.loc[order["origin_hub_id"], "city"] if order.get("origin_hub_id") in hub.index else order.get("origin_hub_id"),
            "destination": hub.loc[order["dest_hub_id"], "city"] if order.get("dest_hub_id") in hub.index else order.get("dest_hub_id"),
            "order_date": str(order.get("order_date")),
            "truck_type": order.get("truck_type"),
            "billable_weight_kg": round(float(order.get("billable_weight_kg") or 0), 1),
            "quoted_days": order.get("quoted_days"),
        }
    return issues, summary


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_invoice(file: UploadFile = File(...), extractor: str = "auto"):
    db = get_db()
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        fields, meta = extractors.extract(tmp_path, extractor)
    except extractors.ExtractionError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        os.unlink(tmp_path)

    issues, order_summary = _field_checks(fields, db)
    missing = extractors.missing_required(fields)
    return {
        "filename": file.filename,
        "extractor_used": meta["provider"],
        "extractor_model": meta["model"],
        "fallback_errors": meta["errors"],
        "fields": fields,
        "required_fields": extractors.REQUIRED_FIELDS,
        "missing_fields": missing,
        "field_issues": issues,
        "order_summary": order_summary,
        "ready_to_score": not missing,
    }


@app.post("/api/check")
def check_fields(req: ScoreRequest):
    """Re-run the per-field checks on user-edited values (live form validation)."""
    fields = extractors.normalise(req.fields)
    issues, summary = _field_checks(fields, get_db())
    return {"fields": fields, "missing_fields": extractors.missing_required(fields),
            "field_issues": issues, "order_summary": summary}


# ------------------------------------------------------------------- score
def _transit_weather(order, fields):
    """Observed weather over the actual transit window from Open-Meteo:
    invoice_date - actual_days -> invoice_date, at the destination hub."""
    try:
        end = live_context._to_date(fields["invoice_date"])
        start = end - timedelta(days=max(1, int(round(float(fields["actual_days"]) + 0.49))))
        return live_context.get_weather(order["dest_hub_id"], start, end)
    except Exception as e:
        return {"ok": False, "source": "unavailable", "error": str(e)[:200]}


def _score(fields, filename, extractor_used, extractor_model, manual_fields, ai_fields):
    db = get_db()
    fields = extractors.normalise(fields)
    missing = extractors.missing_required(fields)
    if missing:
        raise HTTPException(status_code=422, detail={
            "message": "Fill in every required field before the models can run.",
            "missing_fields": missing})

    passed, reason = confidence_gate(fields, db.order_exists, db.vendor_exists)
    result = {"status": "scored" if passed else "gate_failed",
              "filename": filename, "extractor_used": extractor_used,
              "extractor_model": extractor_model, "extracted_fields": fields,
              "manually_entered_fields": manual_fields,
              "gate_passed": passed, "gate_reason": reason, "context": {}}

    if not passed:
        code = (reason or "").split(":")[0]
        result["field_errors"] = GATE_FIELD_MAP.get(code) or (reason.split(":")[1].split(",") if ":" in reason else [])
        try:
            from audit_explanation import explain_rules
            result["explanation"] = explain_rules({"passed": False, "reason": reason})
        except Exception:
            result["explanation"] = f"Validation gate check failed: {reason}"
        _log(db, result)
        return result

    order = dict(db.get_order(fields["order_id"]))
    ref_ctx = db.get_reference_invoice_context(fields["order_id"])
    vstats = db.get_vendor_stats(fields["vendor_id"])
    vendor_risk = causal_vendor_risk(vstats["total_invoices"], vstats["flagged_invoices"])

    # Fill any order-level inputs the ERP row lacks from live sources.
    ctx_notes = []
    if order.get("expected_fuel_price") in (None, "") or order.get("expected_fuel_price") != order.get("expected_fuel_price"):
        fuel = live_context.get_diesel(order.get("state") or "", order.get("order_date"))
        order["expected_fuel_price"] = fuel.get("price") or 91.5
        ctx_notes.append(f"Diesel price from {fuel['source']}")
    if order.get("expected_weather_score") is None or order.get("expected_weather_score") != order.get("expected_weather_score"):
        order["expected_weather_score"] = 0.0

    # Observed transit weather: a reference row exists only for benchmark
    # orders; for every real invoice we measure it live from Open-Meteo.
    wx = _transit_weather(order, fields)
    if ref_ctx.get("_from_reference"):
        adverse = ref_ctx["adverse_weather_days"]
        wx_source = "reference invoice record"
    elif wx.get("ok"):
        adverse = wx["adverse_weather_days"]
        wx_source = wx["source"]
    else:
        adverse = ref_ctx.get("adverse_weather_days", 0.0)
        wx_source = "default (weather service unavailable)"

    context = {"adverse_weather_days": adverse,
               "vendor_padding_ratio": ref_ctx.get("vendor_padding_ratio", 0.0),
               "vendor_historical_risk_score": vendor_risk}

    # Logic check on the ERP order itself -- the models must not run on
    # impossible inputs (negative distance, zero weight, absurd diesel price...).
    bad_inputs = check_inputs(order)
    if bad_inputs:
        result.update({"status": "gate_failed", "gate_passed": False,
                       "gate_reason": "invalid_order_record", "field_errors": ["order_id"],
                       "explanation": "The order record for this invoice has impossible values, so "
                                      "no cost or fraud score was computed: " + " ".join(bad_inputs)})
        _log(db, result)
        return result

    features = build_model2_features(fields, order, context, _m1)
    proba = score_model2(features, _m2, float(order["expected_fuel_price"]))
    model_flag = proba >= _m2.threshold

    # Logic checks on prices: the fair-cost prediction itself, and the billed
    # amount against it. A bill far below the fair cost scores ~0% fraud risk
    # (the model only learned overcharging), so it is caught here instead.
    pred_check = _m1.check_prediction(order, features["model_a_predicted_cost"])
    bill_check = _m1.check_billed(features["actual_billed_amount"], features["model_a_predicted_cost"],
                                  features["cost_interval_lower"], features["cost_interval_upper"])
    rule_flag = bill_check["status"] != "ok" or pred_check["status"] != "ok"
    flagged = bool(model_flag or rule_flag)
    flag_source = ("model+logic_check" if model_flag and rule_flag else
                   "logic_check" if rule_flag else "model" if model_flag else None)

    rec = {"passed": True, "predicted_flag": flagged, "proba": proba, "features": features}
    try:
        from audit_explanation import explain_llm, explain_rules
        try:
            explanation = explain_llm(rec, provider="groq")
        except BaseException:
            explanation = explain_rules(rec)
    except Exception:
        explanation = f"Audit evaluation completed. Estimated fraud probability: {round(proba * 100, 1)}%."
    rule_msgs = [c["message"] for c in (pred_check, bill_check) if c["status"] != "ok"]
    if rule_msgs:
        explanation = "Price logic check: " + " ".join(rule_msgs) + "\n\n" + explanation

    drift = [w for w in (live_context.drift_check("expected_fuel_price", float(order["expected_fuel_price"])),) if w]
    result.update({
        "order_id": fields["order_id"], "vendor_id": fields["vendor_id"],
        "model1_predicted_cost": features["model_a_predicted_cost"],
        "model1_interval_lower": features.get("cost_interval_lower"),
        "model1_interval_upper": features.get("cost_interval_upper"),
        "cost_drivers": features.get("top_drivers"),
        "billed_amount": features["actual_billed_amount"],
        "cost_mismatch": features["cost_mismatch"],
        "model2_proba": proba, "flagged": flagged, "flag_source": flag_source,
        "price_check": {"prediction": pred_check, "billed": bill_check},
        "explanation": explanation,
        "context": {
            "expected_fuel_price": float(order["expected_fuel_price"]),
            "expected_weather_score": float(order["expected_weather_score"]),
            "adverse_weather_days": float(adverse),
            "adverse_weather_source": wx_source,
            "transit_weather": wx if wx.get("ok") else None,
            "vendor_historical_risk_score": round(vendor_risk, 4),
            "notes": ctx_notes,
        },
        "drift_warnings": drift,
    })
    db.bump_vendor_stats(fields["vendor_id"], flagged)
    _log(db, result, ai_fields)
    return result


def _log(db, result, ai_fields=None):
    stored = dict(result["extracted_fields"])
    stored["_meta"] = {"manually_entered": result.get("manually_entered_fields") or [],
                       "extractor_model": result.get("extractor_model"),
                       "ai_extracted": ai_fields}
    try:
        db.insert_audit_log({
            "filename": result["filename"], "extractor_used": result["extractor_used"],
            "extracted_fields": stored, "gate_passed": result["gate_passed"],
            "gate_reason": result["gate_reason"], "order_id": result.get("order_id"),
            "vendor_id": result.get("vendor_id"),
            "model1_predicted_cost": result.get("model1_predicted_cost"),
            "billed_amount": result.get("billed_amount"),
            "cost_mismatch": result.get("cost_mismatch"),
            "model2_proba": result.get("model2_proba"), "flagged": result.get("flagged"),
            "explanation": result.get("explanation"),
        })
    except Exception as e:
        print(f"[audit_log] insert failed: {e}")


@app.post("/api/score", response_model=AuditResult)
def score_invoice(req: ScoreRequest):
    ai = req.ai_fields or {}
    manual = sorted(k for k, v in req.fields.items()
                    if v not in (None, "") and str(v) != str(ai.get(k)) and k in extractors.FIELDS)
    return _score(req.fields, req.filename or "manual-entry", req.extractor_used or "manual",
                  req.extractor_model, manual, ai)


@app.post("/api/audit", response_model=AuditResult)
async def audit_invoice(file: UploadFile = File(...), extractor: str = "auto"):
    """One-shot extract + score, for scripts. Never scores incomplete data."""
    ext = await extract_invoice(file, extractor)
    if ext["missing_fields"]:
        return {"status": "needs_input", "filename": ext["filename"],
                "extractor_used": ext["extractor_used"], "extractor_model": ext["extractor_model"],
                "extracted_fields": ext["fields"], "gate_passed": False,
                "gate_reason": "missing_critical_field:" + ",".join(ext["missing_fields"]),
                "field_errors": ext["missing_fields"],
                "explanation": "Some required fields could not be read. Enter them and call /api/score."}
    return _score(ext["fields"], ext["filename"], ext["extractor_used"],
                  ext["extractor_model"], [], ext["fields"])


# ---------------------------------------------------------- live context
@app.get("/api/context", response_model=ContextResponse)
def shipment_context(origin_hub_id: str, dest_hub_id: str,
                     ship_date: str | None = None, transit_days: float | None = Query(None, gt=0, le=45)):
    try:
        return live_context.shipment_context(origin_hub_id, dest_hub_id, ship_date, transit_days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/hubs")
def list_hubs():
    h = live_context.hubs().reset_index()
    return h[["hub_id", "city", "state"]].to_dict(orient="records")


# ------------------------------------------------------------- dashboard
@app.get("/api/vendors", response_model=list[VendorRisk])
def vendor_leaderboard():
    db = get_db()
    try:
        rows = db.get_vendor_leaderboard()
    except Exception:
        rows = []
    if not rows and getattr(db, "_local", None):
        rows = db._local.get_vendor_leaderboard()
    return [{"vendor_id": r["vendor_id"],
             "vendor_name": r.get("vendor_name") or (r.get("vendors") or {}).get("vendor_name") or r["vendor_id"],
             "total_invoices": r.get("total_invoices", 0),
             "flagged_invoices": r.get("flagged_invoices", 0),
             "risk_score": r.get("risk_score", 0.5)} for r in rows]


@app.get("/api/dashboard", response_model=DashboardStats)
def dashboard(n: int = 300):
    from collections import Counter
    try:
        rows = get_db().get_audit_stats(since_n=n)
    except Exception:
        rows = []
    passed_rows = [r for r in rows if r.get("gate_passed")]
    failed_rows = [r for r in rows if not r.get("gate_passed")]
    manual = sum(1 for r in rows
                 if ((r.get("extracted_fields") or {}).get("_meta") or {}).get("manually_entered"))
    return {
        "total_processed": len(rows),
        "gate_passed": len(passed_rows),
        "gate_rejected": len(failed_rows),
        "auto_processing_rate": len(passed_rows) / len(rows) if rows else 0.0,
        "flagged_count": sum(1 for r in passed_rows if r.get("flagged")),
        "rejection_breakdown": dict(Counter((r.get("gate_reason") or "").split(":")[0] for r in failed_rows)),
        "manual_entry_count": manual,
    }


@app.post("/api/quote", response_model=QuoteResponse)
def quote(req: QuoteRequest):
    if _m1 is None:
        raise HTTPException(status_code=503, detail="Model 1 not loaded yet.")
    payload = req.model_dump()
    problems = check_inputs(payload)
    if problems:
        raise HTTPException(status_code=422, detail={"message": "Some inputs are impossible: " + " ".join(problems),
                                                     "problems": problems})
    predicted, p10, p90 = _m1.predict_interval(payload)
    check = _m1.check_prediction(payload, predicted)
    if check["status"] == "invalid":
        raise HTTPException(status_code=422, detail={"message": check["message"]})
    shap_info = _m1.explain_prediction(payload, top_k=4)
    drift = [w for w in (live_context.drift_check(k, payload[k]) for k in
                         ("expected_fuel_price", "expected_weather_score",
                          "distance_km", "billable_weight_kg")) if w]
    return {"predicted_cost": predicted, "interval_lower": p10, "interval_upper": p90,
            "top_drivers": shap_info["top_drivers"], "cost_breakdown": shap_info["breakdown"],
            "adjustment_drivers": shap_info["adjustment_drivers"],
            "price_check": check, "drift_warnings": drift}


# ------------------------------------------------------------ static files
from fastapi.staticfiles import StaticFiles  # noqa: E402

BENCH_DIR = os.path.join(_BUNDLE_DIR, "phase_a_output")
if os.path.isdir(BENCH_DIR):
    app.mount("/phase_a_output", StaticFiles(directory=BENCH_DIR), name="phase_a_output")

FRONTEND_DIR = os.path.join(_BUNDLE_DIR, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
