"""
FastAPI backend for the invoice audit web app.

Every real Python capability here already existed before this file: OCR
(ocr_extract.py), the confidence gate and model inference (audit_core.py,
itself lifted from confidence_gate_benchmark.py), and explanations
(audit_explanation.py). This file's only job is to wire those into HTTP
endpoints and persist the result to Supabase. No pipeline logic lives here.
"""
import json
import os
import sys
import tempfile

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.abspath(os.path.join(_CURR_DIR, ".."))

# Load environment variables
load_dotenv(os.path.join(_CURR_DIR, ".env"))
load_dotenv(os.path.join(_BUNDLE_DIR, ".env"))
load_dotenv()

# Add backend directory and extraction_scripts to sys.path
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

for candidate in [
    os.path.join(_BUNDLE_DIR, "extraction_scripts"),
    os.path.join(_CURR_DIR, "extraction_scripts"),
    os.path.join(_CURR_DIR, "../../phase_a"),
    os.path.join(_BUNDLE_DIR, "phase_a"),
]:
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.insert(0, os.path.abspath(candidate))
        break

from ocr_extract import ocr, extract_rules, extract_llm          # existing Phase A
from audit_explanation import explain_rules, explain_llm         # existing Phase C
from audit_core import confidence_gate, build_model2_features, score_model2, Model1
from db import Db
from schemas import AuditResult, DashboardStats, VendorRisk, QuoteRequest, QuoteResponse

# Dynamically locate model artifacts directory
MODEL_DIR = None
for candidate in [
    os.path.join(_BUNDLE_DIR, "code", "output", "models"),
    os.path.join(_CURR_DIR, "code", "output", "models"),
    os.path.join(_CURR_DIR, "../../code/output/models"),
    os.path.join(_CURR_DIR, "models"),
]:
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "model_1_freight_cost.joblib")):
        MODEL_DIR = os.path.abspath(candidate)
        break

if not MODEL_DIR:
    MODEL_DIR = os.path.join(_BUNDLE_DIR, "code", "output", "models")

app = FastAPI(title="FreightCost Audit API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_db = None
_m1 = None
_m2 = None


def get_db():
    global _db
    if _db is None:
        try:
            _db = Db()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Database configuration/connection error: {str(e)}"
            )
    return _db


@app.on_event("startup")
def load_models():
    """Model artifacts are files, not database rows -- loaded once from
    disk at startup, same as every other version of this backend."""
    global _m1, _m2
    m1_path = os.path.join(MODEL_DIR, "model_1_freight_cost.joblib")
    meta_path = os.path.join(MODEL_DIR, "model_1_features.json")
    m2_path = os.path.join(MODEL_DIR, "model_2_invoice_risk.joblib")

    booster = joblib.load(m1_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _m1 = Model1(booster, meta)
    _m2 = joblib.load(m2_path)


@app.get("/api/health")
def health():
    db_status = "unconfigured"
    try:
        get_db()
        db_status = "connected"
    except Exception:
        db_status = "not_connected"

    return {
        "status": "ok",
        "models_loaded": _m1 is not None and _m2 is not None,
        "database": db_status
    }


@app.post("/api/audit", response_model=AuditResult)
async def audit_invoice(file: UploadFile = File(...), extractor: str = "rules",
                        provider: str = "groq"):
    db = get_db()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = ocr(tmp_path)
    finally:
        os.unlink(tmp_path)

    pred = extract_rules(text) if extractor == "rules" else extract_llm(text, provider=provider)
    passed, reason = confidence_gate(pred, db.order_exists, db.vendor_exists)

    result = {"filename": file.filename, "extractor_used": extractor,
             "extracted_fields": pred, "gate_passed": passed, "gate_reason": reason}

    if passed:
        order = db.get_order(pred["order_id"])
        ref_ctx = db.get_reference_invoice_context(pred["order_id"])
        vstats = db.get_vendor_stats(pred["vendor_id"])

        # Causal risk score: computed from stats BEFORE this invoice.
        from audit_core import causal_vendor_risk
        vendor_risk = causal_vendor_risk(vstats["total_invoices"], vstats["flagged_invoices"])
        context = {**ref_ctx, "vendor_historical_risk_score": vendor_risk}

        features = build_model2_features(pred, order, context, _m1)
        proba = score_model2(features, _m2)
        flagged = proba >= 0.5

        rec_for_explain = {"passed": True, "predicted_flag": flagged, "proba": proba,
                           "features": features}
        explanation = (explain_rules(rec_for_explain) if extractor == "rules"
                       else explain_llm(rec_for_explain, provider))

        result.update({
            "order_id": pred["order_id"], "vendor_id": pred["vendor_id"],
            "model1_predicted_cost": features["model_a_predicted_cost"],
            "billed_amount": features["actual_billed_amount"],
            "cost_mismatch": features["cost_mismatch"],
            "model2_proba": proba, "flagged": flagged, "explanation": explanation,
        })

        # Update vendor stats AFTER scoring -- keeps the next invoice's score causal.
        db.bump_vendor_stats(pred["vendor_id"], flagged)
    else:
        rec_for_explain = {"passed": False, "reason": reason}
        result["explanation"] = explain_rules(rec_for_explain)

    db.insert_audit_log({
        "filename": result["filename"], "extractor_used": extractor,
        "extracted_fields": result["extracted_fields"], "gate_passed": passed,
        "gate_reason": reason, "order_id": result.get("order_id"),
        "vendor_id": result.get("vendor_id"),
        "model1_predicted_cost": result.get("model1_predicted_cost"),
        "billed_amount": result.get("billed_amount"),
        "cost_mismatch": result.get("cost_mismatch"),
        "model2_proba": result.get("model2_proba"), "flagged": result.get("flagged"),
        "explanation": result.get("explanation"),
    })

    return result


@app.get("/api/vendors", response_model=list[VendorRisk])
def vendor_leaderboard():
    db = get_db()
    rows = db.get_vendor_leaderboard()
    return [{"vendor_id": r["vendor_id"],
            "vendor_name": r.get("vendor_name") or (r.get("vendors") or {}).get("vendor_name"),
            "total_invoices": r["total_invoices"],
            "flagged_invoices": r["flagged_invoices"],
            "risk_score": r["risk_score"]} for r in rows]


@app.get("/api/dashboard", response_model=DashboardStats)
def dashboard(n: int = 300):
    db = get_db()
    rows = db.get_audit_stats(since_n=n)
    total = len(rows)
    passed_rows = [r for r in rows if r["gate_passed"]]
    failed_rows = [r for r in rows if not r["gate_passed"]]

    from collections import Counter
    breakdown = Counter((r["gate_reason"] or "").split(":")[0] for r in failed_rows)

    return {
        "total_processed": total,
        "gate_passed": len(passed_rows),
        "gate_rejected": len(failed_rows),
        "auto_processing_rate": len(passed_rows) / total if total else 0.0,
        "flagged_count": sum(1 for r in passed_rows if r["flagged"]),
        "rejection_breakdown": dict(breakdown),
    }


@app.post("/api/quote", response_model=QuoteResponse)
def quote(req: QuoteRequest):
    if _m1 is None:
        raise HTTPException(status_code=503, detail="Model 1 not loaded yet.")
    predicted = _m1.predict(req.model_dump())
    return {"predicted_cost": predicted}


# Mount static files for the frontend if directory exists (mounted last so /api routes take precedence)
from fastapi.staticfiles import StaticFiles

# Mount sample benchmark images for demo invoice loading
BENCH_DIR = os.path.abspath(os.path.join(_BUNDLE_DIR, "phase_a_output"))
if os.path.isdir(BENCH_DIR):
    app.mount("/phase_a_output", StaticFiles(directory=BENCH_DIR), name="phase_a_output")

FRONTEND_DIR = os.path.abspath(os.path.join(_CURR_DIR, "..", "frontend"))
if not os.path.isdir(FRONTEND_DIR):
    FRONTEND_DIR = os.path.abspath(os.path.join(_CURR_DIR, "frontend"))
if not os.path.isdir(FRONTEND_DIR):
    FRONTEND_DIR = os.path.abspath(os.path.join(_BUNDLE_DIR, "webapp", "frontend"))

if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


