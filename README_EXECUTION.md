# Execution Guide — Real-Data Rebuild, Models & Colab

How to reproduce this project from scratch and export the model files, either on
**Google Colab** (recommended, no local setup) or **locally**. For *what the
results were*, see `REPORT.md`. For the intentional data issues, see
`DATA_QUALITY_NOTES.md`.

---

## 0. What's in this package

```
code/
  build_dataset.py        # builds the 60k dataset (real or modeled)
  train_model2.py         # trains + evaluates Model 2 (invoice risk)
  export_models.py        # trains + SAVES deployable Model 1 & Model 2 artifacts
  forecast_series.py      # SARIMA forecaster for the fuel & weather series
  scripts/                # reference_data, routing, weather_fuel, generators
  output/                 # a pre-built real-data run (dataset + models/)
extraction_scripts/
  run_openmeteo.py        # REAL weather fetch  (Open-Meteo Archive, no key)
  load_ppac_diesel.py     # REAL diesel normaliser (PPAC/IOC, no key)
  run_osrm.py             # optional real road distances (self-hosted OSRM)
data/                     # original reference tables + delivered dataset
real_cache/               # REAL caches already fetched: weather + diesel + forecasts
Colab_RealBuild.ipynb     # ready-to-run Colab notebook (this guide, executable)
requirements.txt
README.md                 # dataset design (two-model, 11-phase, anomalies)
DATA_QUALITY_NOTES.md     # intentional data-quality issues + fixes
REPORT.md                 # results of the real-data rebuild
```

---

## A. Run on Google Colab (recommended)

1. **Upload** `chi_freight_realbuild.zip` to your Google Drive (e.g. `MyDrive/`).
2. In Colab: **File ▸ Upload notebook ▸** `Colab_RealBuild.ipynb`.
3. **Run cells top to bottom.** They do:

   | Cell | Does |
   |---|---|
   | 1–2 | Mount Drive, unzip the project into the Colab VM |
   | 3 | `pip install -r requirements.txt` |
   | 4 (optional) | Re-fetch **real** weather + diesel live (else use shipped caches) |
   | 5 | Build the 60k dataset on real data → `code/output/` |
   | 6 | Train + evaluate Model 2 (PR-AUC, recall, per-type recall) |
   | 7 | **Export model files** → `code/output/models/` |
   | 8 (optional) | SARIMA forecaster for fuel & weather |
   | 9 | Copy models to Drive **and** download `chi_freight_models.zip` |
   | 10 | Inference sanity check |

   If your zip isn't at the Drive root, edit `ZIP_ON_DRIVE` in cell 2.

---

## B. Run locally (macOS/Linux)

```bash
unzip chi_freight_realbuild.zip -d chi && cd chi
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) re-fetch real data yourself
cd extraction_scripts
python run_openmeteo.py --hubs ../data/hubs.csv --out ../real_cache/weather_cache.csv \
    --start 2023-01-01 --end 2026-01-31 --sleep 0.8
python load_ppac_diesel.py --infile ../real_cache/ppac_rsp_raw.csv \
    --out ../real_cache/fuel_prices.csv --start 2023-01-01 --end 2025-12-31
cd ..

# build on real caches, train, export models, forecast
cd code
python build_dataset.py --orders 60000 --seed 42 --outdir output \
    --weather-cache ../real_cache/weather_cache.csv \
    --fuel-cache ../real_cache/fuel_prices.csv
python train_model2.py output
python export_models.py --outdir output --seed 42
python forecast_series.py --fuel ../real_cache/fuel_prices.csv \
    --weather ../real_cache/weather_cache.csv --outdir ../real_cache/forecasts
```

**Modeled fallback:** omit `--weather-cache` / `--fuel-cache` and the build uses the
calibrated modeled sources instead of real data — everything else is identical.

---

## C. The exported model files (`code/output/models/`)

| File | Purpose |
|---|---|
| `model_1_freight_cost.pkl` | Model 1 bundle: point/P10/P90 regressors + schema + cost constants + price bounds |
| `model_1_freight_cost*.json` | Same boosters as portable xgboost files (they predict the ratio to the cost baseline) |
| `model_1_features.json` | Model 1 input schema, column order, baseline constants, price bounds |
| `model_2_invoice_risk.pkl` | Model 2 bundle: classifier + feature list + threshold + diesel deflator |
| `model_2_invoice_risk.json` | Portable xgboost booster |
| `model_2_features.json` | Model 2 feature list + decision threshold (0.5) |
| `metadata.json` | Metrics (old vs new design), stress tests, library versions |
| `predict_example.py` | Runnable load-and-predict demo |

See `docs/operations/milestone_4_extrapolation_and_price_checks.md` for the
current metrics and the out-of-range stress tests.

---

## D. Use the models in a website

```python
from freight_models import FreightCostModel, InvoiceRiskModel, check_inputs, load_bundle
M1 = FreightCostModel(load_bundle("model_1_freight_cost.pkl"))
M2 = InvoiceRiskModel(load_bundle("model_2_invoice_risk.pkl"))
fair, lo, hi = M1.predict_interval(order)          # order: dict of the 6 numerics + category + truck
check = M1.check_billed(billed, fair, lo, hi)      # ok / too_low / too_high / invalid
```

- **Model 1** -- always call `check_inputs(order)` first; `predict_interval`
  multiplies the fuel + toll + driver baseline by the learned correction.
- **Model 2** -- build features with `backend/audit_core.build_model2_features`
  and score with `M2.predict_proba(features, fuel_price=order["expected_fuel_price"])`.
- Load the bundles **once** at server start, not per request.
- For a non-Python backend, load the native `.json` boosters via any xgboost binding.

---

## E. Notes & caveats

- **No API keys** are needed (weather = Open-Meteo, diesel = PPAC/IOC mirror).
- **Real coverage:** weather is real for all 18 hubs; diesel is *direct-real* for 4
  metro states and *derived from the real national daily trend + VAT offsets* for
  the other 9 (see `source` column, and REPORT.md §3).
- **Data-quality residue is intentional** and is cleaned inside the training/export
  scripts, not in the raw files (see `DATA_QUALITY_NOTES.md`).
- **No leakage:** `anomaly_type` lives only in `_ground_truth_audit.csv` and is never
  a feature; `model_a_predicted_cost` is out-of-fold.
- **OSRM real distances** were skipped; the route cache is haversine×circuity.
- **Fuel forecast** backtests near-perfectly because real diesel was price-static in
  the holdout window — that's real, not overfitting (REPORT.md §7).
