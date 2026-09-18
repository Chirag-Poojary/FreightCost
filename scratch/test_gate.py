import json
import os
import pandas as pd

bench = json.load(open('phase_a_output/benchmark_llm_30.json'))
orders_df = pd.read_csv('data/orders.csv').set_index('order_id')
invoices_df = pd.read_csv('data/invoices.csv').set_index('invoice_id')

passed = []
rejected = []

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
        rejected.append((img, reasons))
    else:
        passed.append((img, pred))

total = len(bench['records'])
print(f"Total Invoices: {total}")
print(f"Passed Confidence Gate (Auto-processed): {len(passed)} ({len(passed)/total:.1%})")
print(f"Rejected to Manual Entry: {len(rejected)} ({len(rejected)/total:.1%})")
print("\nRejections breakdown:")
for img, r in rejected:
    print(f"  {img}: {', '.join(r)}")
