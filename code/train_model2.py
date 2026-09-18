"""
Phase 11 -- Train & evaluate Model 2 (Invoice Risk Flagger).
Class-weighted gradient-boosted classifier. Evaluated with precision/recall/
PR-AUC and PER-ANOMALY-TYPE recall (never plain accuracy), per the doc.

This script also demonstrates a light real-world cleaning pass, since the
delivered invoices.csv intentionally carries residual data-quality issues.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, classification_report,
                             confusion_matrix)

OUT = sys.argv[1] if len(sys.argv) > 1 else "output"


def clean(inv):
    before = len(inv)
    inv = inv.drop_duplicates(subset="invoice_id", keep="first")
    inv["vendor_id"] = inv["vendor_id"].str.upper()          # fix casing drift
    inv = inv[inv["actual_billed_amount"] > 0]               # drop credit notes
    print(f"[clean] {before} -> {len(inv)} rows after dedup / bad-amount drop")
    return inv


def main():
    inv = pd.read_csv(f"{OUT}/invoices.csv")
    gt = pd.read_csv(f"{OUT}/_ground_truth_audit.csv")       # audit only
    inv = clean(inv)
    inv = inv.merge(gt[["order_id", "anomaly_type"]], on="order_id", how="left")

    features = ["actual_billed_amount", "model_a_predicted_cost", "cost_mismatch",
                "actual_days", "true_delay_days", "commercial_delay_days",
                "adverse_weather_days", "vendor_padding_ratio",
                "vendor_historical_risk_score"]
    # model_a_predicted_cost needs to be on invoices for cost_mismatch context
    if "model_a_predicted_cost" not in inv.columns:
        orders = pd.read_csv(f"{OUT}/orders.csv")
        inv = inv.merge(orders[["order_id", "model_a_predicted_cost"]], on="order_id")

    inv = inv.dropna(subset=[f for f in features if f in inv.columns])
    X = inv[[f for f in features if f in inv.columns]]
    y = inv["requires_manual_review"].astype(int)

    Xtr, Xte, ytr, yte, atr, ate = train_test_split(
        X, y, inv["anomaly_type"], test_size=0.25, random_state=42, stratify=y)

    try:
        from xgboost import XGBClassifier
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=spw, eval_metric="aucpr",
                            random_state=42, n_jobs=-1)
        algo = "XGBoost (scale_pos_weight)"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_depth=5, learning_rate=0.05, max_iter=500,
            class_weight="balanced", random_state=42)
        algo = "HistGBM (class_weight=balanced)"

    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"\n=== Model 2: {algo} ===")
    print(f"PR-AUC (avg precision): {average_precision_score(yte, proba):.3f}")
    print(f"Precision: {precision_score(yte, pred):.3f}  "
          f"Recall: {recall_score(yte, pred):.3f}  F1: {f1_score(yte, pred):.3f}")
    print("\nConfusion matrix [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(yte, pred))
    print("\nClassification report:")
    print(classification_report(yte, pred, digits=3))

    print("Per-anomaly-type recall (caught / total in test set):")
    te = Xte.copy(); te["y"] = yte.values; te["pred"] = pred; te["atype"] = ate.values
    for at, g in te[te.y == 1].groupby("atype"):
        print(f"  {at:20s}: {g.pred.sum():4d}/{len(g):4d} = {g.pred.mean():.3f}")

    imp = getattr(clf, "feature_importances_", None)
    if imp is not None:
        print("\nTop features:")
        for f, w in sorted(zip(X.columns, imp), key=lambda z: -z[1])[:6]:
            print(f"  {f:32s} {w:.3f}")


if __name__ == "__main__":
    main()
