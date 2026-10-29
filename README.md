# RateRight

AI-based freight invoice auditing and fair cost prediction for Indian road freight.

RateRight reads a carrier's freight invoice from an image, asks the user to fill in any field the AI could not read, predicts what the trip should have cost, and estimates the risk that the invoice is overbilled. The result is a clear verdict, approved or flagged for audit, with the fair cost and the reasons behind it.

- Live application: https://freight-cost-nine.vercel.app
- Backend API: https://freightcost.onrender.com (health check at `/api/health`)

This project was built as a TY B.Tech (Information Technology) Specialization Based Project at Vidyalankar Institute of Technology, Mumbai, 2026-27.

---

## Contents

1. [Features](#features)
2. [How it works](#how-it-works)
3. [Models](#models)
4. [Dataset](#dataset)
5. [Results](#results)
6. [Repository structure](#repository-structure)
7. [Running locally](#running-locally)
8. [Configuration](#configuration)
9. [API reference](#api-reference)
10. [Retraining the models](#retraining-the-models)
11. [Deployment](#deployment)
12. [Limitations](#limitations)
13. [Team](#team)

---

## Features

- **Cloud invoice extraction.** A vision language model reads PNG, JPG or WEBP invoice images, including scans and phone photos. Google Gemini is tried first; if it fails or is over quota, the backend falls back to OpenRouter (Qwen2.5-VL-72B), then Groq, then Tesseract OCR with a text LLM.
- **Human-in-the-loop review.** Any required field the model could not read is highlighted and must be typed in. The models never run on incomplete data.
- **Confidence gate.** Before scoring, the invoice must pass four checks: required fields present, numeric values parse, base freight + detention + toll equals the total within 2%, and the total and transit days fall in plausible ranges.
- **Fair cost prediction (Model 1).** Predicts the expected trip cost with an 80% range (P10 to P90).
- **Fraud risk scoring (Model 2).** Estimates the probability that the invoice is overbilled.
- **Price logic check.** Flags bills that are impossibly low, impossibly high or invalid, independent of Model 2.
- **Live context data.** Fetches the diesel price for the shipment date and state, and Open-Meteo weather for the route.
- **Data drift warnings.** Shown when a live input falls outside the range the models were trained on.
- **Pre-shipment quote page.** Estimates a fair cost before booking, with distance, diesel price and weather filled in automatically.
- **Dashboard and carrier risk list.** Audit statistics, a table of recent audits and a carrier risk leaderboard.
- **Sample invoices.** Five sample invoices on the first page (clean, overbilled, missing field, phone photo) so new users can try the full audit before using their own invoices.
- **Light, ERP-style interface.** Sidebar navigation, dropdowns for truck type and goods, city and carrier suggestions, and calendar pickers for dates.

---

## How it works

```
 Frontend (Vercel)                       Backend (FastAPI on Render)                Data and models
 -----------------                       ---------------------------                ---------------
 1. Upload invoice image  ---------->    POST /api/extract  ---------------------->  Gemini / OpenRouter / Groq
                                         (normalise fields, list missing ones)
 2. Review and fill fields  -------->    POST /api/check   ----------------------->  Orders master (Supabase or CSV)
                                         (order exists, carrier matches,
                                          line items add up)
 3. Run models  ------------------->    POST /api/score
                                         a. required-field check
                                         b. confidence gate
                                         c. order lookup (or estimate a new order)
                                         d. live transit weather  ---------------->  Open-Meteo
                                         e. Model 1: fair cost and range  ------->  model_1_freight_cost.pkl
                                         f. Model 2: fraud probability  ---------->  model_2_invoice_risk.pkl
                                         g. price logic check
                                         h. verdict + explanation
 4. Verdict screen  <---------------     (approved / flagged, fair cost build-up,
                                          data sources, fields entered by hand)

 Quote page  ---------------------->    GET /api/context  ----------------------->  live diesel price + Open-Meteo
                                         POST /api/quote   ----------------------->  Model 1
```

An invoice is flagged if the Model 2 fraud probability is 0.5 or higher, or if the price logic check fails.

---

## Models

Both models use XGBoost. Shared model logic, used by both training and the API, lives in `code/freight_models.py`.

### Model 1: fair cost

Model 1 predicts the fair cost as a cost baseline multiplied by a correction learned by XGBoost:

```
Fair cost = (distance / mileage x diesel price  +  distance x 1.75  +  ideal_days x 1200)  x  correction
                fuel                               tolls                driver allowance
```

Truck mileage is 6.5, 4.6 and 4.0 km per litre for 6, 10 and 12 wheelers. The baseline scales with diesel price and distance, so predictions stay accurate outside the training range. XGBoost (400 trees, depth 6, learning rate 0.05) predicts only the correction ratio. Two quantile models predict the P10 and P90 ratios for the 80% range.

**Inputs (14):** `distance_km`, `ideal_days`, `quoted_days`, `billable_weight_kg`, `expected_fuel_price`, `expected_weather_score`, five one-hot cargo categories (`cat_*`) and three one-hot truck types (`truck_*`).

### Model 2: fraud risk

An XGBoost classifier trained on out-of-fold Model 1 predictions, so it never learns from a cost prediction made on its own training row.

**Inputs (17):** `actual_billed_amount`, `model_a_predicted_cost`, `cost_mismatch`, `actual_days`, `true_delay_days`, `commercial_delay_days`, `adverse_weather_days`, `vendor_padding_ratio`, `vendor_historical_risk_score`, `cost_mismatch_ratio`, `cost_mismatch_upper`, `cost_mismatch_upper_ratio`, `billed_to_p90_ratio`, `excess_rate_per_kg`, `excess_rate_per_ton_km`, `cost_mismatch_delay_skew`, `zero_delay_p90_breach`.

- Rupee-valued inputs are divided by the diesel price relative to its training average, so a nationwide price rise does not look like fraud.
- Monotone constraints ensure a larger overcharge can never lower the fraud risk.

### Price logic check

Limits are learned from 56,543 genuine training invoices and stored in the Model 1 bundle:

- Predicted cost must lie between Rs 7.71 and Rs 83.29 per km.
- A bill below 72.4% of the fair lower bound is flagged as too low.
- A bill more than Rs 18,554 above the fair upper bound, or more than 22.8 times it, is flagged as too high.
- Zero or negative amounts, and impossible distances, weights or diesel prices, are rejected.

---

## Dataset

No public dataset links freight invoices to order details, fair costs and fraud labels, so the team built one with `code/build_dataset.py`.

| Parameter | Details |
|---|---|
| Orders | 60,000 (January 2023 to December 2025) |
| Invoices | 60,120 (59,910 after cleaning) |
| Coverage | 18 hubs, 13 states, 306 routes, 25 carriers |
| Weather | Real daily Open-Meteo archive data for every hub (`extraction_scripts/run_openmeteo.py`) |
| Diesel | Real PPAC daily retail prices, 13 states (`extraction_scripts/load_ppac_diesel.py`) |
| Fraud labels | 5.62% of invoices: detention padding (1,133), toll inflation (1,085), weight discrepancy (1,152) |
| Splits | Model 1: 80/20 holdout plus 5-fold out-of-fold predictions. Model 2: 75/25 stratified holdout |

Cleaning removes duplicate invoices and credit notes, fixes weights recorded in grams, normalises carrier ID case, parses mixed date formats and fills missing weather scores with the monthly median.

---

## Results

### Invoice extraction

| Extractor | Test set | Field accuracy | All critical fields correct |
|---|---|---|---|
| Rule-based regex + Tesseract OCR | 100 scanned invoices | 87.2% | 37.0% |
| Tesseract OCR + Groq LLM | 30 scanned invoices | 93.9% | 73.3% |
| Local Qwen3-VL-4B (earlier version) | 13 hard images | 75.0% | 69.2% |
| Cloud vision model with fallback | Same 13 hard images | 99.4% | 92.3% |

Critical fields are order ID, carrier ID, total and transit days. The hard images include rotation, blur, phone photos, Hindi text, stamps and missing fields.

### Model 1 (mean absolute percentage error)

| Test set | Previous direct XGBoost | Current model |
|---|---|---|
| Normal holdout data | 1.68% | 1.61% |
| Diesel Rs 70-86 per litre | 11.02% | 1.70% |
| Diesel Rs 98-125 per litre | 11.97% | 1.63% |
| Routes 2,900-4,500 km | 21.47% | 1.81% |

### Model 2 (14,978 test invoices, 842 fraudulent)

| Method | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Rule: bill above fair upper bound (P90) | 57.44% | 11.64% | 99.64% | 20.84% |
| Rule: bill above 1.10 x fair cost | 82.52% | 23.54% | 93.82% | 37.64% |
| Previous Model 2 (raw rupee features) | 94.27% | 49.45% | 85.51% | 62.66% |
| Current Model 2 | 94.10% | 48.59% | 85.99% | 62.09% |

- PR-AUC: 0.821. Recall by fraud type: detention padding 100%, toll inflation 93.5%, weight discrepancy 65.4%.
- Verdicts changed when every price rises by 25%: 0% (previous model: 2.38%).
- The price logic check flagged none of the 56,543 genuine invoices.

Full metrics are written to `code/output/models/metadata.json` each time the models are trained.

---

## Repository structure

```
backend/
  main.py               FastAPI app and all API endpoints
  extractors.py         Cloud vision extraction, provider fallback, field normalisation
  live_context.py       Live diesel prices, Open-Meteo weather, route lookup, drift checks
  audit_core.py         Confidence gate and Model 2 feature engineering
  db.py                 Supabase access with local CSV fallback
  schemas.py            Request and response models
  schema.sql            Supabase table definitions
  migrate_to_supabase.py  One-time upload of the reference CSVs to Supabase
code/
  freight_models.py     Shared model classes, cost baseline, input and price checks
  build_dataset.py      Builds the orders and invoices dataset
  export_models.py      Trains, tests and exports both models
  forecast_series.py    SARIMA forecasts for diesel and weather series
  scripts/              Dataset generators, reference data, weather and fuel helpers
  output/               Built dataset and trained models (output/models/*.pkl)
extraction_scripts/
  ocr_extract.py        Tesseract OCR and LLM extraction
  render_invoices.py    Renders invoice images with ground truth
  run_openmeteo.py      Downloads historical weather for all hubs
  load_ppac_diesel.py   Builds daily diesel prices from PPAC records
  audit_explanation.py  Rule-based and LLM verdict explanations
  benchmark_hard_images.py, confidence_gate_benchmark.py, invoice_audit_e2e.py
frontend/
  index.html            Single-page app: Audit, Dashboard, Vendors, Quote
  js/                   Page logic and the API client
  demo/                 Demo invoices consistent with the orders master
data/, real_cache/      Reference CSVs, weather and diesel caches, forecasts
images/                 13 hard invoice images used for extraction testing
phase_a_output/         Extraction benchmark images and results
Dockerfile, render.yaml Backend container and Render configuration
```

---

## Running locally

Requirements: Python 3.12 and Tesseract OCR (only needed for the OCR fallback).

```bash
git clone https://github.com/Chirag-Poojary/FreightCost.git
cd FreightCost
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example backend/.env    # then add at least one extraction API key
uvicorn backend.main:app --port 8000
```

Open http://localhost:8000. The backend serves the frontend as well, and the frontend uses `http://localhost:8000` automatically when opened from localhost.

To run the API tests (no API keys or database needed):

```bash
pip install pytest
SUPABASE_URL= SUPABASE_SERVICE_KEY= python -m pytest tests -q
```

Without Supabase credentials the backend runs on the local CSV files in `code/output/`. Without any extraction key, invoices can still be audited by typing the fields in by hand.

---

## Configuration

Set these in `backend/.env` locally, or as environment variables on Render. Configure at least one extraction provider.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Google Gemini vision extraction (recommended primary) |
| `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` | OpenRouter vision models, default `qwen/qwen2.5-vl-72b-instruct` |
| `GROQ_API_KEY`, `GROQ_VISION_MODEL`, `GROQ_TEXT_MODEL` | Groq vision extraction, OCR + LLM fallback and LLM explanations |
| `CLOUD_VLM_URL`, `CLOUD_VLM_MODEL`, `CLOUD_VLM_API_KEY` | A self-hosted OpenAI-compatible vision model, for example Qwen3-VL-4B on a cloud GPU |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Database for orders, vendors, audit log and carrier statistics |

Model names are configurable because providers retire models regularly. Never commit `.env` files; they are excluded by `.gitignore`.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Service status, model status, database status and configured extractors |
| GET | `/api/extractors` | Available extractors, required fields and all fields |
| POST | `/api/extract` | Upload an invoice image; returns fields, missing fields, field issues and order summary. No ML runs |
| POST | `/api/check` | Re-validate user-edited fields |
| POST | `/api/score` | Run the gate, Model 1, Model 2 and price checks on confirmed fields. Returns 422 if a required field is empty |
| POST | `/api/audit` | One-shot extract and score for scripts; returns `needs_input` when fields are missing |
| GET | `/api/context` | Route distance, live diesel price, weather and drift warnings for a route and date |
| GET | `/api/hubs` | List of logistics hubs |
| GET | `/api/audits/recent` | Latest audits for the dashboard table |
| POST | `/api/quote` | Fair cost estimate for a planned shipment |
| GET | `/api/dashboard` | Aggregate audit statistics |
| GET | `/api/vendors` | Carrier risk leaderboard |

Interactive documentation is available at `/docs` on a running backend.

---

## Retraining the models

```bash
cd code
python build_dataset.py --orders 60000 --seed 42 \
    --weather-cache ../real_cache/weather_cache.csv \
    --fuel-cache ../real_cache/fuel_prices.csv     # only to rebuild the dataset
python export_models.py --outdir output
```

`export_models.py` cleans the data, trains Model 1, computes out-of-fold predictions, compares five Model 2 variants, runs the out-of-range and price-rise stress tests, learns the price limits and saves:

- `output/models/model_1_freight_cost.pkl` and `output/models/model_2_invoice_risk.pkl`
- Native XGBoost boosters (`*.json`), feature schemas and `metadata.json`

Training takes about 25 seconds on a laptop CPU. Commit the new model files and redeploy the backend to use them. `code/output/models/predict_example.py` shows how to load and use the bundles.

---

## Deployment

- **Backend:** Docker container on Render, defined by `Dockerfile` and `render.yaml`. The image includes Tesseract OCR, the backend, the trained models and the reference data.
- **Frontend:** Static site on Vercel from the `frontend/` folder. `frontend/vercel.json` forwards `/api/*` requests to the Render backend.
- **Database:** Supabase. Create the tables with `backend/schema.sql`, then load the reference data with `backend/migrate_to_supabase.py`.

---

## Engineering review

A full review of the codebase, listing every problem found, its severity and whether it is fixed, is in [ENGINEERING_REVIEW.md](ENGINEERING_REVIEW.md).

## Limitations

- The orders and invoices are generated; only the weather and diesel inputs are real data. The models should be retrained on real company invoices before production use.
- The current diesel price is read from a public retail price web page, not an official data feed.
- Live diesel and weather values are cached in server memory only (6 hours and 3 hours) and are not saved for retraining.
- The models are retrained manually; there is no scheduled retraining job.
- Extraction requires an internet connection and an API key for a cloud model. Sending real invoices to a free API tier may not be acceptable for confidential data.
- Model 1's 80% range covers the true cost about 79% of the time on normal data and 71-77% outside the training range.

---

## Team

| Member | Contribution |
|---|---|
| Jain Ritik | LLM invoice extraction pipeline |
| Poojary Chirag | Data collection, preparation and maintenance; version 2 ML models |
| Shinde Siddharth | Backend and connectivity |
| Palkar Darshan | Frontend |

Department of Information Technology, Vidyalankar Institute of Technology, Wadala (E), Mumbai.
