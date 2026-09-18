# Phase A — Invoice OCR & Structured Extraction

Closes the biggest realism gap in the existing pipeline: it assumed invoice data
arrives as clean CSV rows, when real invoices arrive as scans and phone photos.

```
invoices.csv row  ──▶  rendered invoice image  ──▶  Tesseract OCR  ──▶  field
                       (3 layouts, ±scan noise)                        extraction
                              │                                            │
                              └────────── ground_truth.json ──────────▶ scoring
```

## Files

| File | Role |
|---|---|
| `render_invoices.py` | Renders each invoice row into a realistic Indian freight invoice image (3 rotating layouts) + exact ground-truth JSON. `--scan` adds rotation, blur, sensor noise, and uneven lighting. |
| `ocr_extract.py` | OCR → field extraction → per-field accuracy scoring. Two extractors: `rules` (offline, free, the baseline) and `llm` (needs `ANTHROPIC_API_KEY`). |

## Run it

```bash
# 1. render documents (clean + degraded)
python render_invoices.py --n 100 --outdir ../phase_a_output/invoices_clean
python render_invoices.py --n 100 --outdir ../phase_a_output/invoices_scanned --scan

# 2. baseline extraction + scoring
python ocr_extract.py --indir ../phase_a_output/invoices_scanned

# 3. LLM extraction -- free tier, no credit card
export GROQ_API_KEY="..."          # console.groq.com/keys
python ocr_extract.py --indir ../phase_a_output/invoices_scanned --extractor llm
```

## Free LLM providers

`--extractor llm` speaks the OpenAI-compatible chat API, so any of these work
via `--provider`. All issue keys without a credit card.

| `--provider` | Env var | Free tier | Notes |
|---|---|---|---|
| `groq` *(default)* | `GROQ_API_KEY` | 30 RPM · 14,400/day | Fastest (LPU, 300-500 tok/s). Best fit here. |
| `gemini` | `GEMINI_API_KEY` | generous, no card | Strongest model quality of the free options. |
| `openrouter` | `OPENROUTER_API_KEY` | ~50/day | Widest model choice through one key. |
| `cerebras` | `CEREBRAS_API_KEY` | ~30 RPM | Fast; may require a verified payment method. |
| `anthropic` | `ANTHROPIC_API_KEY` | paid | Kept for parity. |

100 invoices sits far inside Groq's daily ceiling. 429s are retried with
exponential backoff, so a rate limit slows the run rather than killing it.

**Privacy note for the report:** no-credit-card free tiers are generally
funded by your prompts (providers may train on them). Irrelevant for this
synthetic data, but a real deployment with live vendor invoices would need a
paid tier or a locally hosted model.

## Why not just use a public invoice dataset?

SROIE and CORD are the standard public options, and both were considered and
rejected as a *replacement*:

- Both are **receipt** datasets, not B2B freight invoices. SROIE annotates only
  four fields (company, date, address, total) across 626 train / 347 test images.
- Neither carries `order_id`, `vendor_id`, transit days, or fraud labels — so
  nothing extracted from them can flow into Model 2. The link between the
  document and the fraud model is the whole point of this phase, and no public
  dataset provides it.

They remain useful as an **external validation benchmark**: running this
extractor against SROIE's overlapping fields (date, total) and reporting it
alongside the synthetic numbers answers "does this only work on invoices you
generated yourself?" — worth doing as a supplementary result.

## Design decisions worth defending

**Three rotating layouts, not one.** With a single template an extractor just
memorises fixed positions and the accuracy number means nothing. The three
layouts (`classic` bordered table, `modern` dark-banner columns, `compact`
legacy-ERP monospace) force the extractor to generalise.

**Exact line-item decomposition.** `freight_base + detention + toll == total`,
and `total == actual_billed_amount` from the CSV — no rounding drift, so a
wrong extraction is always a real extraction error and never an arithmetic
artifact of the renderer.

**Rule-based baseline kept alongside the LLM.** This turns "we added an LLM"
into a measurable claim. It also means the pipeline runs with no API key and
no network.

**Numeric fields scored with tolerance, text fields exact.** A rupee amount is
correct if it round-trips to the same number, not the same string.

## Baseline results (rule-based, n=100 each)

| | Clean | Scanned |
|---|---|---|
| Mean over all 12 fields | **87.6%** | **87.2%** |
| Mean over Model-2 critical fields | 84.2% | 84.2% |
| Invoices with all critical fields correct | 42.0% | 37.0% |

By layout (scanned): `classic` 94.6% · `compact` 94.7% · `modern` **72.2%**

### What the numbers actually say

**Scan degradation barely matters** (87.6% → 87.2%). Tesseract 5 handles mild
rotation, blur and lighting noise well. The scan pipeline is not the
bottleneck.

**Layout variation is the real bottleneck.** `layout_modern` scores ~22 points
below the other two. It prints a header row above a value row in columns —
a positional relationship regex cannot express, but which a language model
reads trivially. This is the concrete, measured justification for the LLM
extractor, not a hand-wave.

**Two OCR failure modes drive most of the remaining error:**
1. *Glyph confusion in coded IDs* — OCR reads `ORDO24516` for `ORD024516` and
   `VENO17` for `VEN017`. Handled by matching a loose character class and
   normalising `O→0, l→1, S→5, B→8` in the digit half. This lifted critical-field
   accuracy from 48.3% to 81.7% on the first test batch.
2. *Run-on values when two fields share an OCR line* — `Origin: Jaipur
   Destination: Ludhiana` captured both. Handled by cutting each value at the
   next known field label.

**`actual_days`, `origin`, `destination` are the weakest fields** (59–67%),
almost entirely because of `layout_modern`'s column structure. These are the
fields to watch when the LLM extractor is benchmarked against this baseline.

## Next

- Benchmark `--extractor llm` against these baseline numbers; the expected win
  is concentrated in `layout_modern` and the three weak fields above.
- Wire successful extractions into Model 2 (`export_models.py` feature order)
  so a scanned invoice produces a risk score end to end.
