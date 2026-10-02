# FreightCost Phase 0: schema, gate and evaluation harness

Drop this folder into the repo (for example `extraction_scripts/phase0/`). Python 3.10+, standard library only.

| File | Purpose |
|---|---|
| `schema.py` | The 12 core fields, extended fields (GSTIN, LR, e-way bill, taxes, line items), normalisers, and a JSON Schema for constrained decoding |
| `gate.py` | The 4-tier Confidence Gate as a pure function with configurable bounds |
| `evaluate.py` | Scorer: per-field accuracy, critical-fields-correct, **silent-error rate**, per-layout breakdown, 95% confidence intervals |
| `make_splits.py` | Splits by layout (whole templates held out), with real scans kept test-only |
| `test_phase0.py` | Sanity tests: `python test_phase0.py` |

## Data formats

Ground truth and predictions are both JSON lists:

```json
[{"doc_id": "INV000001", "layout": "modern",
  "fields": {"order_id": "ORD123", "vendor_id": "V07", "total": 15700.0, "actual_days": 3, "...": "..."}}]
```

Use `null` for a field that is genuinely absent from the document. A prediction that fills a null field is counted as a hallucination, which the model must learn not to do.

If your existing `benchmark_*.json` files use a different layout, write a small adapter that outputs the format above.

## Commands

```bash
python evaluate.py --gt gt.json --pred pred.json --out report.json
python make_splits.py --manifest manifest.json --test-layouts 2 --real-layouts real --out splits.json
```

## Metrics

- **Mean field accuracy**: average per-document share of correct fields.
- **All critical fields correct**: `order_id`, `vendor_id`, `total`, `actual_days` all right.
- **Silent error rate**: documents that pass the gate with a wrong critical field. This is the number to drive towards zero.
- The gate table splits documents into: passed and correct, **passed and wrong**, rejected and wrong (gate worked), rejected and correct (needless manual work).

## Evaluation protocol (keep these rules)

1. Split by layout, never by document. Report `test_unseen_layouts` and `test_seen_layouts` separately; the gap between them measures overfitting to templates.
2. Real scanned invoices go in the test set only (`layout="real"`), never into training.
3. Use at least a few hundred test documents. With 30 documents the confidence intervals are about 20 points wide.
4. Compare every extractor on the same test set: current rules, current Groq pipeline, zero-shot VLMs, then the fine-tuned model.

## Phase 2 model bake-off (zero-shot, local)

Run these on the 12GB RTX 3060 with the JSON schema from `schema.json_schema()` and score them with `evaluate.py`:

| Candidate | Why | Fits 12GB |
|---|---|---|
| Qwen3-VL-2B-Instruct | Fast baseline, easy to fine-tune | Yes, easily |
| **Qwen3-VL-4B-Instruct** | Default fine-tuning candidate | Yes, with QLoRA |
| Qwen3-VL-8B-Instruct (4-bit) | Upper bound for local inference | Inference yes; QLoRA tight, lower image resolution |
| Qwen3.5-2B / 4B | Newer, native vision | 2B comfortable; 4B tight in bf16 LoRA |
| dots.ocr (1.7B), optional | OCR-to-text stage only, not key extraction | Yes |

Selection rule: pick the smallest model within about 2 points of the best on critical-all-correct, then fine-tune it.
