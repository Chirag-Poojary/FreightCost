"""
=================================================================
TWO-NUMBER REPORT SUMMARY (n=300) -- rule-based extractor
=================================================================
Passed confidence gate : 186/300 (62.0%)
Rejected (manual queue): 114/300 (38.0%)
  - missing_critical_field           86
  - days_out_of_plausible_range      11
  - arithmetic_mismatch              9
  - vendor_id_not_found              7
  - amount_out_of_plausible_range    1

Cohort: 186, true fraud cases: 6
Accuracy: 89.2%  Precision: 23.1%  Recall: 100.0%
Confusion: TP=6 FP=20 TN=160 FN=0
=================================================================

4-TIER CONFIDENCE GATE + TWO-NUMBER BENCHMARK (n=300).

Evaluates the end-to-end extraction and fraud audit pipeline across 300
realistic degraded scanned invoices against trained models (Model 1 fair
should-cost regressor, Model 2 invoice risk classifier) and hidden ground-truth
audit labels.

Pipeline per invoice:
  scanned image -> OCR -> rule-based extraction -> 4-TIER CONFIDENCE GATE
        PASS -> join physicals from orders.csv (Model 1 was trained on
                these; OCR never reads them off the document) -> recompute
                cost_mismatch / delay features from the OCR-extracted
                values (not the pipeline's internal ground truth) so an
                OCR error genuinely propagates into a wrong score, exactly
                as it would in production -> Model 1 -> Model 2 -> compare
                to _ground_truth_audit.csv
        FAIL -> Manual Data Entry Queue, tagged with the failing gate

Extractor is rule-based by default (no network dependency, reproducible
without an API key). The LLM extractor can be evaluated using:
  python confidence_gate_benchmark.py --extractor llm --provider groq
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

# Support imports when run from repo root or extraction_scripts directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.dirname(_SCRIPT_DIR)
for p in [_SCRIPT_DIR, os.path.join(_BUNDLE_DIR, "extraction_scripts"), os.path.join(_BUNDLE_DIR, "phase_a"), "../phase_a"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from ocr_extract import ocr, extract_rules, extract_llm, _num  # reuse the tested extractor

# ------------------------------------------------------------- the 4-tier gate
CRITICAL = ["order_id", "vendor_id", "total", "actual_days"]


def confidence_gate(pred, orders_idx, vendors_idx):
    """Returns (passed: bool, reason: str|None). First failing tier wins,
    matching the documented tier order exactly."""
    # Tier 1 -- critical field presence
    missing = [f for f in CRITICAL if not pred.get(f)]
    if missing:
        return False, f"missing_critical_field:{','.join(missing)}"

    # Tier 2 -- referential integrity against the ERP master tables
    if pred["order_id"] not in orders_idx.index:
        return False, "order_id_not_found"
    if pred["vendor_id"] not in vendors_idx.index:
        return False, "vendor_id_not_found"

    total = _num(pred["total"])
    days = _num(pred["actual_days"])
    if total is None or days is None:
        return False, "unparseable_amount_or_days"

    # Tier 3 -- line-item arithmetic reconciliation (+/- 2%, floor 2.0)
    parts = [_num(pred.get(f)) for f in ("freight_base", "detention", "toll")]
    if all(p is not None for p in parts):
        drift = abs(sum(parts) - total)
        if drift > max(2.0, total * 0.02):
            return False, "arithmetic_mismatch"
    # if line items themselves failed to extract, arithmetic can't be
    # checked -- that's a Tier-1-adjacent gap, but we don't double-fail it
    # here since it's already been through Tier 1 on the critical fields.

    # Tier 4 -- physical / economic plausibility
    if not (500 < total <= 250000):
        return False, "amount_out_of_plausible_range"
    if not (0.2 <= days <= 30.0):
        return False, "days_out_of_plausible_range"

    return True, None


# --------------------------------------------------------- feature assembly
MODEL2_FEATURES = [
    "actual_billed_amount",
    "model_a_predicted_cost",
    "cost_mismatch",
    "actual_days",
    "true_delay_days",
    "commercial_delay_days",
    "adverse_weather_days",
    "vendor_padding_ratio",
    "vendor_historical_risk_score",
]


def build_model2_row(pred, order_row, inv_lookup_row, m1):
    """Physicals + Model 1 prediction come from the internal ERP join (OCR
    never reads these off the document). cost_mismatch and the two delay
    fields are RECOMPUTED from the OCR-extracted total/actual_days rather
    than copied from the dataset's internal ground truth -- so an OCR slip
    genuinely changes the score, the same way it would in production."""
    predicted_cost = float(m1.predict(order_row))

    ocr_total = _num(pred["total"])
    ocr_days = _num(pred["actual_days"])

    return {
        "actual_billed_amount": ocr_total,
        "model_a_predicted_cost": predicted_cost,
        "cost_mismatch": ocr_total - predicted_cost,
        "actual_days": ocr_days,
        "true_delay_days": ocr_days - order_row["ideal_days"],
        "commercial_delay_days": ocr_days - order_row["quoted_days"],
        "adverse_weather_days": inv_lookup_row["adverse_weather_days"],
        "vendor_padding_ratio": inv_lookup_row["vendor_padding_ratio"],
        "vendor_historical_risk_score": inv_lookup_row["vendor_historical_risk_score"],
    }


def onehot_row(order_row, m1_meta):
    row = {c: order_row[c] for c in m1_meta["numeric"]}
    for c in m1_meta["product_categories"]:
        row[f"cat_{c}"] = 1 if order_row["product_category"] == c else 0
    for c in m1_meta["truck_types"]:
        row[f"truck_{c}"] = 1 if order_row["truck_type"] == c else 0
    return row


class _M1Wrapper:
    """Adapts the raw XGBRegressor to accept an orders row and return a
    single prediction, handling the one-hot column assembly."""
    def __init__(self, model, meta):
        self.model, self.meta = model, meta

    def predict(self, order_row):
        """order_row: a pandas Series/dict with the original order columns
        (product_category, truck_type, distance_km, ...). Builds the exact
        one-hot column_order Model 1 was trained on and returns one prediction."""
        row = onehot_row(order_row, self.meta)
        full = pd.DataFrame([row])
        for c in self.meta["column_order"]:
            if c not in full.columns:
                full[c] = 0
        return float(self.model.predict(full[self.meta["column_order"]])[0])


def main(indir, n, data_dir, start, checkpoint, extractor, provider):
    global RUN_EXTRACTOR, RUN_PROVIDER
    RUN_EXTRACTOR, RUN_PROVIDER = extractor, provider
    manifest_path = os.path.join(indir, "ground_truth.json") if not indir.endswith(".json") else indir
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Please render invoices first using:\n"
            f"python extraction_scripts/render_invoices.py --n {n} --outdir {indir} --scan"
        )

    base_dir = os.path.dirname(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)[start:start + n]
    print(f"Processing invoices [{start}:{start+len(manifest)}] of this run "
          f"(checkpoint: {checkpoint})\n", flush=True)

    orders = (pd.read_csv(os.path.join(data_dir, "orders.csv"))
              .drop_duplicates(subset="order_id", keep="first")
              .set_index("order_id", drop=False))
    vendors = (pd.read_csv(os.path.join(data_dir, "vendors.csv"))
               .drop_duplicates(subset="vendor_id", keep="first")
               .set_index("vendor_id"))
    invoices = (pd.read_csv(os.path.join(data_dir, "invoices.csv"))
                .drop_duplicates(subset="order_id", keep="first")
                .set_index("order_id"))
    audit = pd.read_csv(os.path.join(data_dir, "_ground_truth_audit.csv")).set_index("order_id")

    m1_path = os.path.join(data_dir, "models", "model_1_freight_cost.joblib")
    if not os.path.exists(m1_path):
        m1_path = os.path.join(data_dir, "model_1_freight_cost.joblib")
    m2_path = os.path.join(data_dir, "models", "model_2_invoice_risk.joblib")
    if not os.path.exists(m2_path):
        m2_path = os.path.join(data_dir, "model_2_invoice_risk.joblib")
    m1_meta_path = os.path.join(data_dir, "models", "model_1_features.json")
    if not os.path.exists(m1_meta_path):
        m1_meta_path = os.path.join(data_dir, "model_1_features.json")

    m1 = joblib.load(m1_path)
    m2 = joblib.load(m2_path)
    with open(m1_meta_path) as f:
        m1_meta = json.load(f)
    m1w = _M1Wrapper(m1, m1_meta)

    done_images = set()
    try:
        with open(checkpoint) as f:
            for line in f:
                if line.strip():
                    done_images.add(json.loads(line)["image"])
    except FileNotFoundError:
        pass
    if done_images:
        print(f"Resuming -- {len(done_images)} already checkpointed", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
    with open(checkpoint, "a") as ckpt:
        for i, entry in enumerate(manifest, 1):
            if entry["image"] in done_images:
                continue
            img_path = os.path.join(base_dir, entry["image"])
            text = ocr(img_path)
            pred = (extract_rules(text) if extractor == "rules"
                    else extract_llm(text, provider=provider))
            passed, reason = confidence_gate(pred, orders, vendors)

            if not passed:
                rec = {"image": entry["image"], "passed": False, "reason": reason}
            else:
                order_row = orders.loc[pred["order_id"]]
                inv_row = invoices.loc[pred["order_id"]]
                feat = build_model2_row(pred, order_row, inv_row, m1w)
                x2 = pd.DataFrame([feat])[MODEL2_FEATURES]
                proba = float(m2.predict_proba(x2)[0, 1])
                truth_row = audit.loc[pred["order_id"]]
                rec = {
                    "image": entry["image"],
                    "passed": True,
                    "order_id": pred["order_id"],
                    "predicted_flag": bool(proba >= 0.5),
                    "proba": proba,
                    "features": feat,
                    "true_flag": bool(truth_row["requires_manual_review"]),
                    "anomaly_type": (truth_row["anomaly_type"]
                                     if pd.notna(truth_row["anomaly_type"]) else None),
                }
            ckpt.write(json.dumps(rec, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
            ckpt.flush()

            if i % 20 == 0 or i == len(manifest):
                print(f"  {i}/{len(manifest)}", flush=True)

    results = [json.loads(l) for l in open(checkpoint) if l.strip()]
    report(results, len(results))


def report(results, total):
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    n_pass, n_fail = len(passed), len(failed)

    print("\n" + "=" * 65)
    print(" " * 18 + "TWO-NUMBER REPORT SUMMARY (n={})".format(total))
    print("=" * 65)
    print(f"[NUMBER 1: AUTO-PROCESSING FRACTION]")
    print(f"  Passed confidence gate (auto-processed) : {n_pass}/{total} ({n_pass/total:.1%})")
    print(f"  Routed to manual data entry (rejected)   : {n_fail}/{total} ({n_fail/total:.1%})")

    if failed:
        from collections import Counter
        reasons = Counter(r["reason"].split(":")[0] for r in failed)
        print(f"\n  Rejection breakdown by failure mode:")
        for reason, cnt in reasons.most_common():
            print(f"    - {reason:<32} {cnt} instance(s)")

    if not passed:
        print("\nNo invoices passed the gate -- cannot compute Model 2 performance.")
        return

    tp = sum(1 for r in passed if r["predicted_flag"] and r["true_flag"])
    fp = sum(1 for r in passed if r["predicted_flag"] and not r["true_flag"])
    tn = sum(1 for r in passed if not r["predicted_flag"] and not r["true_flag"])
    fn = sum(1 for r in passed if not r["predicted_flag"] and r["true_flag"])
    acc = (tp + tn) / n_pass
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    n_fraud = tp + fn

    print(f"\n[NUMBER 2: MODEL 2 PERFORMANCE ON AUTO-PROCESSED FRACTION]")
    print(f"  Cohort size                     : {n_pass} validated invoices")
    print(f"  True fraud cases in cohort       : {n_fraud}  <-- sample size that backs recall")
    print(f"  Classification accuracy          : {acc:.1%}")
    print(f"  Precision                        : {prec:.1%}" if prec == prec else "  Precision : n/a (no positive predictions)")
    print(f"  Recall                           : {rec:.1%}" if rec == rec else "  Recall : n/a (no true positives in cohort)")
    print(f"  F1                                : {f1:.3f}" if f1 == f1 else "  F1 : n/a")
    print(f"  Confusion matrix                 : TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    by_type = {}
    for r in passed:
        if r["true_flag"]:
            t = r["anomaly_type"] or "unknown"
            by_type.setdefault(t, [0, 0])
            by_type[t][0] += 1
            by_type[t][1] += int(r["predicted_flag"])
    if by_type:
        print(f"\n  Per-anomaly-type recall (auto-processed cohort only):")
        for t, (total_t, caught) in sorted(by_type.items()):
            print(f"    - {t:<22} {caught}/{total_t} ({caught/total_t:.1%})")
    print("=" * 65)
    print("\nNote: this run used the {} extractor.".format(
        "RULE-BASED (no API key dependency)" if RUN_EXTRACTOR == "rules"
        else f"LLM ({RUN_PROVIDER})"))
    if RUN_EXTRACTOR == "rules":
        print("To reproduce with the LLM extractor, rotate GROQ_API_KEY first (see")
        print("the security note), then rerun with --extractor llm --provider groq.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Confidence-gate + two-number benchmark (n=300)")
    _def_indir = os.path.join(_BUNDLE_DIR, "phase_a_output", "invoices_bench300")
    if not os.path.exists(_def_indir):
        _def_indir = os.path.join(_BUNDLE_DIR, "phase_a_output", "invoices_scanned")
    _def_data = os.path.join(_BUNDLE_DIR, "code", "output")
    _def_ckpt = os.path.join(_BUNDLE_DIR, "phase_a_output", "bench300_checkpoint.jsonl")

    ap.add_argument("--indir", default=_def_indir, help="Path to input invoice directory containing ground_truth.json")
    ap.add_argument("--n", type=int, default=300, help="Number of invoices to benchmark")
    ap.add_argument("--start", type=int, default=0, help="Start offset")
    ap.add_argument("--checkpoint", default=_def_ckpt, help="Checkpoint jsonl path")
    ap.add_argument("--data-dir", default=_def_data, help="Path to code/output or data dir with orders.csv and models/")
    ap.add_argument("--extractor", choices=["rules", "llm"], default="rules", help="Extractor engine")
    ap.add_argument("--provider", default="groq", help="LLM provider (default: groq)")
    a = ap.parse_args()

    ckpt = a.checkpoint if "--checkpoint" in sys.argv else \
        os.path.join(os.path.dirname(a.checkpoint), f"bench300_checkpoint_{a.extractor}.jsonl")
    main(a.indir, a.n, a.data_dir, a.start, ckpt, a.extractor, a.provider)
