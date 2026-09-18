import json
import os
import joblib
import pandas as pd

bench = json.load(open('phase_a_output/benchmark_llm_30.json'))
orders_df = pd.read_csv('data/orders.csv').set_index('order_id')
invoices_df = pd.read_csv('data/invoices.csv').set_index('invoice_id')
M2 = joblib.load('code/output/models/model_2_invoice_risk.joblib')
with open('code/output/models/model_2_features.json') as f:
    M2_CONFIG = json.load(f)

vendor_stats = {}
for vid, g in invoices_df.groupby("vendor_id"):
    vendor_stats[vid] = {
        "padding_ratio": float(g["vendor_padding_ratio"].median()),
        "risk_score": float(g["requires_manual_review"].mean()),
    }

auto_processed_results = []

for rec in bench['records']:
    pred = rec['predicted']
    img = rec['image']
    reasons = []
    
    # 1. Critical fields presence
    for f in ['order_id', 'vendor_id', 'total', 'actual_days']:
        if pred.get(f) is None:
            reasons.append(f"Missing critical field: {f}")
            
    # 2. Referential integrity
    ord_id = pred.get('order_id')
    if ord_id and ord_id not in orders_df.index:
        reasons.append(f"Order ID {ord_id} not found in Orders table")
        
    # 3. Arithmetic reconciliation
    tot = pred.get('total')
    fb = pred.get('freight_base')
    det = pred.get('detention')
    toll = pred.get('toll')
    if None not in (tot, fb, det, toll):
        try:
            line_sum = float(fb) + float(det) + float(toll)
            if abs(line_sum - float(tot)) > max(2.0, float(tot) * 0.02):
                reasons.append(f"Arithmetic mismatch: items sum to Rs {line_sum:,.2f} != total Rs {float(tot):,.2f}")
        except Exception as e:
            reasons.append(f"Arithmetic parse error: {e}")
            
    # 4. Plausibility bounds
    if tot is not None:
        try:
            val = float(tot)
            if val <= 500 or val > 250000:
                reasons.append(f"Billed amount out of bounds: Rs {val:,.2f}")
        except Exception:
            reasons.append("Non-numeric total")
            
    if pred.get('actual_days') is not None:
        try:
            days = float(pred['actual_days'])
            if days <= 0.1 or days > 30:
                reasons.append(f"Transit days out of bounds: {days}")
        except Exception:
            reasons.append("Non-numeric actual_days")

    if reasons:
        continue  # Gated out!

    # AUTO-PROCESSED: Run Model 2
    ord_row = orders_df.loc[ord_id]
    if isinstance(ord_row, pd.DataFrame):
        ord_row = ord_row.iloc[0]
        
    inv_id = pred.get('invoice_id')
    gt_flag = int(invoices_df.loc[inv_id, 'requires_manual_review']) if inv_id in invoices_df.index else None
    
    pred_cost = float(ord_row['model_a_predicted_cost'])
    actual_billed = float(tot)
    cost_mismatch = actual_billed - pred_cost
    actual_days = float(pred['actual_days'])
    ideal_days = float(ord_row['ideal_days'])
    quoted_days = float(ord_row['quoted_days'])
    
    adv_wx = float(invoices_df.loc[inv_id, 'adverse_weather_days']) if inv_id in invoices_df.index else 0.0
    v_stat = vendor_stats.get(pred.get('vendor_id'), {"padding_ratio": 1.10, "risk_score": 0.05})
    
    features = {
        "actual_billed_amount": actual_billed,
        "model_a_predicted_cost": pred_cost,
        "cost_mismatch": cost_mismatch,
        "actual_days": actual_days,
        "true_delay_days": max(0.0, actual_days - ideal_days),
        "commercial_delay_days": max(0.0, actual_days - quoted_days),
        "adverse_weather_days": adv_wx,
        "vendor_padding_ratio": v_stat["padding_ratio"],
        "vendor_historical_risk_score": v_stat["risk_score"],
    }
    
    X = pd.DataFrame([features])[M2_CONFIG["features"]]
    proba = float(M2.predict_proba(X)[0, 1])
    m2_flag = int(proba >= M2_CONFIG["threshold"])
    
    auto_processed_results.append({
        "image": img,
        "invoice_id": inv_id,
        "gt_flag": gt_flag,
        "m2_flag": m2_flag,
        "proba": proba,
        "correct": (gt_flag == m2_flag) if gt_flag is not None else True
    })

res_df = pd.DataFrame(auto_processed_results)
print(f"Auto-Processed Invoices Scored: {len(res_df)}")
print(f"Accuracy on Auto-Processed Fraction: {res_df['correct'].mean():.1%}")

tp = ((res_df['gt_flag'] == 1) & (res_df['m2_flag'] == 1)).sum()
fp = ((res_df['gt_flag'] == 0) & (res_df['m2_flag'] == 1)).sum()
tn = ((res_df['gt_flag'] == 0) & (res_df['m2_flag'] == 0)).sum()
fn = ((res_df['gt_flag'] == 1) & (res_df['m2_flag'] == 0)).sum()

print(f"Confusion Matrix on Auto-Processed:")
print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
print(f"  Precision: {precision:.1%}")
print(f"  Recall:    {recall:.1%}")
