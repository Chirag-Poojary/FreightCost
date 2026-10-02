"""FreightCost Phase 0: split documents BY LAYOUT, not by document.

Whole layouts are held out for testing, so the test set measures generalisation
to templates the model has never seen. Real scanned invoices should carry
layout="real" and be passed with --real-layouts so they go to test only.

Usage:
    python make_splits.py --manifest manifest.json [--test-layouts 2] [--val-frac 0.1]
                          [--seen-test-frac 0.1] [--real-layouts real] [--seed 42] [--out splits.json]

manifest.json: [{"doc_id": "INV000001", "layout": "classic"}, ...]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict


def make_splits(manifest, test_layouts=None, val_frac=0.1, seen_test_frac=0.1,
                real_layouts=(), seed=42):
    by_layout = defaultdict(list)
    for item in manifest:
        by_layout[item["layout"]].append(item["doc_id"])

    real = [lay for lay in by_layout if lay in set(real_layouts)]
    synth = sorted(lay for lay in by_layout if lay not in real)
    if len(synth) < 2:
        raise ValueError(f"Need at least 2 synthetic layouts to hold one out; found {len(synth)}: {synth}")

    rng = random.Random(seed)
    k = test_layouts if test_layouts is not None else max(1, math.ceil(0.25 * len(synth)))
    k = min(k, len(synth) - 1)
    shuffled = synth[:]
    rng.shuffle(shuffled)
    unseen, train_layouts = shuffled[:k], shuffled[k:]

    train_pool = [d for lay in train_layouts for d in by_layout[lay]]
    rng.shuffle(train_pool)
    n_seen = int(round(seen_test_frac * len(train_pool)))
    n_val = int(round(val_frac * len(train_pool)))
    test_seen = train_pool[:n_seen]
    val = train_pool[n_seen:n_seen + n_val]
    train = train_pool[n_seen + n_val:]

    splits = {
        "train": sorted(train),
        "val": sorted(val),
        "test_seen_layouts": sorted(test_seen),
        "test_unseen_layouts": sorted(d for lay in unseen for d in by_layout[lay]),
        "test_real": sorted(d for lay in real for d in by_layout[lay]),
        "meta": {"train_layouts": sorted(train_layouts), "unseen_test_layouts": sorted(unseen),
                 "real_layouts": sorted(real), "seed": seed},
    }
    _check(splits, by_layout)
    return splits


def _check(splits, by_layout):
    names = ["train", "val", "test_seen_layouts", "test_unseen_layouts", "test_real"]
    seen_ids = set()
    for n in names:
        ids = set(splits[n])
        assert not (ids & seen_ids), f"document appears in more than one split ({n})"
        seen_ids |= ids
    train_layouts = set(splits["meta"]["train_layouts"])
    held_out = set(splits["meta"]["unseen_test_layouts"]) | set(splits["meta"]["real_layouts"])
    assert not (train_layouts & held_out), "layout leakage between train and held-out layouts"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--test-layouts", type=int)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seen-test-frac", type=float, default=0.1)
    ap.add_argument("--real-layouts", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="splits.json")
    a = ap.parse_args()

    with open(a.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)
    s = make_splits(manifest, a.test_layouts, a.val_frac, a.seen_test_frac, a.real_layouts, a.seed)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(s, fh, indent=2)
    print({k: len(v) for k, v in s.items() if k != "meta"})
    print("train layouts :", s["meta"]["train_layouts"])
    print("unseen layouts:", s["meta"]["unseen_test_layouts"])


if __name__ == "__main__":
    main()
