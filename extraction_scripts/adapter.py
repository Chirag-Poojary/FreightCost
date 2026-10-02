"""
Adapter to convert FreightCost benchmark ground_truth.json into the Phase 0 format
expected by evaluate.py, gate.py, and make_splits.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.dirname(_CURR_DIR)
for p in [_CURR_DIR, os.path.join(_CURR_DIR, "phase0"), _BUNDLE_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from schema import CORE_FIELDS, EXTENDED_FIELDS, ALL_FIELDS, norm_date, norm_amount, norm_number, norm_id, norm_text

def adapt_ground_truth(input_path: str, out_gt: str, out_manifest: str | None = None):
    with open(input_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    adapted = []
    manifest = []

    for item in raw:
        doc_id = item.get("ground_truth", {}).get("invoice_id") or os.path.splitext(item.get("image", "unknown"))[0]
        raw_layout = item.get("layout", "unknown")
        # Clean layout name, e.g. layout_classic -> classic
        layout = raw_layout.replace("layout_", "")
        
        gt = item.get("ground_truth", {})
        fields = {}
        for k, v in gt.items():
            if k == "gstin":
                fields["vendor_gstin"] = v
            else:
                fields[k] = v

        adapted.append({
            "doc_id": doc_id,
            "layout": layout,
            "image": item.get("image"),
            "fields": fields
        })

        manifest.append({
            "doc_id": doc_id,
            "layout": layout,
            "image": item.get("image")
        })

    with open(out_gt, "w", encoding="utf-8") as fh:
        json.dump(adapted, fh, indent=2)
    print(f"Wrote {len(adapted)} adapted ground-truth records to {out_gt}")

def adapt_predictions(input_path: str, out_pred: str):
    with open(input_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    # Could be a benchmark result with "records", or list of dicts
    records = raw.get("records") if isinstance(raw, dict) and "records" in raw else raw

    adapted = []
    for item in records:
        pred_fields = item.get("predicted") or item.get("fields") or {}
        doc_id = pred_fields.get("invoice_id") or item.get("doc_id") or os.path.splitext(item.get("image", "unknown"))[0]
        raw_layout = item.get("layout", "unknown")
        layout = raw_layout.replace("layout_", "")

        adapted.append({
            "doc_id": doc_id,
            "layout": layout,
            "fields": pred_fields
        })

    with open(out_pred, "w", encoding="utf-8") as fh:
        json.dump(adapted, fh, indent=2)
    print(f"Wrote {len(adapted)} adapted prediction records to {out_pred}")
    return adapted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ground truth or predictions to Phase 0 format")
    parser.add_argument("--gt-input", default="phase_a_output/invoices_scanned/ground_truth.json")
    parser.add_argument("--out-gt", default="phase_a_output/gt_phase0.json")
    parser.add_argument("--out-manifest", default="phase_a_output/manifest.json")
    parser.add_argument("--pred-input", default=None, help="Optional prediction JSON to convert")
    parser.add_argument("--out-pred", default="phase_a_output/pred_phase0.json")
    args = parser.parse_args()

    if args.gt_input:
        adapt_ground_truth(args.gt_input, args.out_gt, args.out_manifest)
    if args.pred_input:
        adapt_predictions(args.pred_input, args.out_pred)

