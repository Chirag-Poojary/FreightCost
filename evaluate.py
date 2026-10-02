"""FreightCost Phase 0: evaluation harness for invoice extractors.

Usage:
    python evaluate.py --gt ground_truth.json --pred predictions.json [--out report.json]
                       [--orders-csv orders.csv] [--vendors-csv vendors.csv]

File format (JSON list, or dict keyed by doc_id):
    [{"doc_id": "INV000001", "layout": "modern", "fields": {"order_id": "...", ...}}, ...]
If "fields" is absent, the item's remaining keys are treated as the fields.

Headline numbers:
  * mean_field_acc        average per-document field accuracy
  * critical_all_correct  share of documents with every critical field correct
  * silent_error_rate     share of documents that PASS the gate with a wrong critical field
Every headline number carries a 95% bootstrap confidence interval over documents.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict

from gate import GateConfig, run_gate
from schema import (ALL_FIELDS, BAD, CRITICAL_FIELDS, LINE_ITEM_FIELDS, normalise,
                    values_equal)

META_KEYS = {"doc_id", "layout", "source", "image", "path", "split"}
OK = ("correct", "correct_null")


# ------------------------------------------------------------------- loading
def load_docs(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        raw = [{"doc_id": k, **(v if isinstance(v, dict) else {"fields": v})} for k, v in raw.items()]
    docs = {}
    for item in raw:
        fields = item.get("fields")
        if fields is None:
            fields = {k: v for k, v in item.items() if k not in META_KEYS}
        docs[str(item["doc_id"])] = {"layout": item.get("layout", "unknown"), "fields": fields}
    return docs


def _read_id_set(path: str | None, column: str):
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        return {str(r[column]).strip().upper() for r in csv.DictReader(fh)}


# ------------------------------------------------------------------- scoring
def score_field(kind: str, gt_value, pred_value) -> str:
    g = normalise(kind, gt_value)
    p = normalise(kind, pred_value)
    if g is None:
        return "correct_null" if p is None else "hallucinated"
    if p is None:
        return "missing"
    return "correct" if values_equal(kind, g, p) else "wrong"


def score_line_items(gt_items, pred_items) -> str:
    if not gt_items:
        return "correct_null" if not pred_items else "hallucinated"
    if not pred_items:
        return "missing"
    if len(gt_items) != len(pred_items):
        return "wrong"
    for grow, prow in zip(gt_items, pred_items):
        for key, kind in LINE_ITEM_FIELDS.items():
            if key in grow and score_field(kind, grow[key], (prow or {}).get(key)) not in OK:
                return "wrong"
    return "correct"


def score_doc(gt: dict, pred: dict) -> dict:
    out = {}
    for f, g in gt.items():
        if f == "line_items":
            out[f] = score_line_items(g, pred.get(f))
        elif f in ALL_FIELDS:
            out[f] = score_field(ALL_FIELDS[f], g, pred.get(f))
    return out


# ---------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def bootstrap_ci(vals, n_boot: int = 2000, seed: int = 0):
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def _stat(vals):
    if not vals:
        return {"value": None, "ci": None}
    return {"value": sum(vals) / len(vals), "ci": list(bootstrap_ci(vals))}


# ---------------------------------------------------------------- evaluation
def evaluate(gt_docs: dict, pred_docs: dict, gate_cfg: GateConfig | None = None,
             order_ids=None, vendor_ids=None) -> dict:
    per_field = defaultdict(lambda: defaultdict(int))
    doc_acc, crit_ok, silent, per_layout = [], [], [], defaultdict(lambda: defaultdict(list))
    gate_cells = {"auto_correct": 0, "silent_error": 0, "caught": 0, "false_reject": 0}
    reason_counts = defaultdict(int)
    missing_preds = 0

    for doc_id, g in gt_docs.items():
        pred_fields = (pred_docs.get(doc_id) or {}).get("fields", {})
        if doc_id not in pred_docs:
            missing_preds += 1
        scores = score_doc(g["fields"], pred_fields)
        if not scores:
            continue

        for f, status in scores.items():
            per_field[f]["n"] += 1
            per_field[f][status] += 1
            if status in OK:
                per_field[f]["correct_total"] += 1

        acc = sum(s in OK for s in scores.values()) / len(scores)
        crit_scores = [s for f, s in scores.items() if f in CRITICAL_FIELDS]
        crit_good = all(s in OK for s in crit_scores) if crit_scores else True

        gate = run_gate(pred_fields, gate_cfg, order_ids, vendor_ids)
        for r in gate.reasons:
            reason_counts[r] += 1
        is_silent = gate.passed and not crit_good

        if gate.passed and crit_good:
            gate_cells["auto_correct"] += 1
        elif is_silent:
            gate_cells["silent_error"] += 1
        elif not gate.passed and not crit_good:
            gate_cells["caught"] += 1
        else:
            gate_cells["false_reject"] += 1

        doc_acc.append(acc)
        crit_ok.append(1.0 if crit_good else 0.0)
        silent.append(1.0 if is_silent else 0.0)
        lay = g["layout"]
        per_layout[lay]["acc"].append(acc)
        per_layout[lay]["crit"].append(1.0 if crit_good else 0.0)
        per_layout[lay]["silent"].append(1.0 if is_silent else 0.0)

    n = len(doc_acc)
    passed = gate_cells["auto_correct"] + gate_cells["silent_error"]
    fields_report = {}
    for f, c in per_field.items():
        k, tot = c["correct_total"], c["n"]
        fields_report[f] = {
            "n": tot,
            "accuracy": k / tot if tot else None,
            "ci95": list(wilson(k, tot)),
            "wrong": c["wrong"],
            "missing": c["missing"],
            "hallucinated": c["hallucinated"],
        }

    return {
        "n_docs": n,
        "n_docs_without_prediction": missing_preds,
        "per_field": fields_report,
        "mean_field_acc": _stat(doc_acc),
        "critical_all_correct": _stat(crit_ok),
        "silent_error_rate": _stat(silent),
        "gate": {
            "pass_rate": passed / n if n else None,
            "cells": gate_cells,
            "silent_error_rate_of_passed": gate_cells["silent_error"] / passed if passed else None,
            "false_reject_rate": gate_cells["false_reject"] / n if n else None,
            "rejection_reasons": dict(reason_counts),
        },
        "by_layout": {
            lay: {
                "n": len(v["acc"]),
                "mean_field_acc": sum(v["acc"]) / len(v["acc"]),
                "critical_all_correct": sum(v["crit"]) / len(v["crit"]),
                "silent_error_rate": sum(v["silent"]) / len(v["silent"]),
            }
            for lay, v in per_layout.items()
        },
    }


# ----------------------------------------------------------------- reporting
def _pct(x):
    return "n/a" if x is None else f"{100 * x:5.1f}%"


def _stat_line(label: str, s: dict) -> str:
    if s["value"] is None:
        return f"{label:<26}: n/a"
    lo, hi = s["ci"]
    return f"{label:<26}: {_pct(s['value'])}   95% CI [{_pct(lo)}, {_pct(hi)}]"


def format_report(r: dict) -> str:
    lines = [f"Documents scored: {r['n_docs']}  (without any prediction: {r['n_docs_without_prediction']})", ""]
    lines.append(f"{'field':<16}{'n':>5}{'acc':>9}   95% CI (Wilson)    wrong miss halluc")
    for f, v in r["per_field"].items():
        lo, hi = v["ci95"]
        lines.append(f"{f:<16}{v['n']:>5}{_pct(v['accuracy']):>9}   [{_pct(lo)}, {_pct(hi)}]"
                     f"  {v['wrong']:>4} {v['missing']:>4} {v['hallucinated']:>5}")
    lines += ["", _stat_line("Mean field accuracy", r["mean_field_acc"]),
              _stat_line("All critical fields correct", r["critical_all_correct"]),
              _stat_line("SILENT ERROR rate", r["silent_error_rate"]), ""]
    g, c = r["gate"], r["gate"]["cells"]
    lines += [
        f"Gate pass rate            : {_pct(g['pass_rate'])}",
        f"  passed + correct        : {c['auto_correct']}",
        f"  passed + WRONG (silent) : {c['silent_error']}   <- the number to drive to ~0",
        f"  rejected + wrong        : {c['caught']}   (gate did its job)",
        f"  rejected + correct      : {c['false_reject']}   (needless manual work)",
        f"Silent errors among passed: {_pct(g['silent_error_rate_of_passed'])}",
        f"Rejection reasons         : {g['rejection_reasons'] or '-'}",
        "", "By layout:",
    ]
    for lay, v in sorted(r["by_layout"].items()):
        lines.append(f"  {lay:<12} n={v['n']:<4} field acc {_pct(v['mean_field_acc'])}  "
                     f"critical ok {_pct(v['critical_all_correct'])}  silent err {_pct(v['silent_error_rate'])}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out")
    ap.add_argument("--orders-csv", help="CSV with an order_id column (enables referential check)")
    ap.add_argument("--vendors-csv", help="CSV with a vendor_id column (enables referential check)")
    args = ap.parse_args()

    report = evaluate(
        load_docs(args.gt), load_docs(args.pred),
        order_ids=_read_id_set(args.orders_csv, "order_id"),
        vendor_ids=_read_id_set(args.vendors_csv, "vendor_id"),
    )
    print(format_report(report))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
