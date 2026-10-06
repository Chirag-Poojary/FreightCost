# Milestone 4: Price-Range Extrapolation, Price Logic Checks & Model Retraining

## 1. Problem
Probing the deployed models across input ranges showed both were bound by the
price levels seen in training (2023-2025, diesel Rs 86.6-97.35/L):

- **Model 1** (XGBoost on raw rupees) returned a constant outside the training
  range. Diesel at Rs 110/L was under-predicted by 11%, Rs 130/L by 24%; a
  5,000 km route by 43%; a 5 km trip floored at Rs 1,621. Negative distance,
  diesel or weight still produced a normal-looking positive cost.
- **Model 2** gave 0% risk -- and an APPROVED verdict -- to bills far *below*
  the fair cost, including Rs 322 for a 1,410 km trip and negative amounts. Its
  absolute-rupee features also changed 2.4% of verdicts under a uniform +25%
  price rise.

## 2. What changed
- **`code/freight_models.py`** (new, shared by training and the API):
  - `FreightCostModel` -- Model 1 = `baseline x xgb_ratio`, where
    `baseline = distance / mileage(truck) x diesel + distance x Rs 1.75 + ideal_days x Rs 1,200`
    (the standard fuel + toll + driver build-up). XGBoost learns only the
    multiplicative correction, so cost scales correctly with any diesel price or
    distance. Explanations are a rupee build-up: fuel, tolls, driver, model adjustment.
  - `InvoiceRiskModel` -- rupee-valued features are deflated by
    `diesel / training-average diesel` before scoring.
  - `check_inputs()` -- rejects impossible inputs (non-positive distance, weight,
    days; diesel outside Rs 40-250; unknown truck type) before any model runs.
  - `check_prediction()` / `check_billed()` -- flags a predicted or billed price
    that is zero/negative, too low or too high. Bounds are learned from training
    data and stored in the bundle (`price_bounds`).
- **`code/export_models.py`** -- trains Model 1 on the ratio target; trains
  Model 2 on **out-of-fold** Model 1 outputs (previously in-sample P90); adds
  stress tests; compares variants; saves `.pkl` bundles.
- **`backend/`** -- loads the `.pkl` bundles; `/api/quote` rejects impossible
  inputs with 422 and returns `price_check` + `cost_breakdown`; `/api/score`
  flags an invoice when **either** Model 2 (proba >= 0.5) **or** the price logic
  check fires, and reports which (`flag_source`).

## 3. Results (seed 42)

### Model 1 -- MAPE on holdout and out-of-range stress sets
Stress-set labels use the same cost build-up as the training data (+/-2% noise),
so they test extrapolation, not new market behaviour.

| Test set | Previous (direct XGB) | New (baseline x XGB) | New 80% interval coverage |
|---|---|---|---|
| holdout | 1.68% | 1.61% | 78.8% |
| diesel_70_to_86 | 11.02% | 1.70% | 76.8% |
| diesel_98_to_125 | 11.97% | 1.63% | 75.6% |
| routes_2900_to_4500_km | 21.47% | 1.81% | 71.3% |

### Model 2 -- variants (all trained on out-of-fold Model 1 outputs)
| Variant | PR-AUC | Recall | Precision | Weight-discrepancy recall | Verdicts changed at +25% prices |
|---|---|---|---|---|---|
| previous_full_features | 0.819 | 0.855 | 0.495 | 0.637 | 2.38% |
| scale_free_features | 0.805 | 0.874 | 0.466 | 0.699 | 0.0% |
| scale_free_monotone | 0.805 | 0.882 | 0.459 | 0.716 | 0.0% |
| deflated_full_features | 0.821 | 0.863 | 0.496 | 0.668 | 0.0% |
| deflated_full_monotone **(deployed)** | 0.821 | 0.86 | 0.486 | 0.654 | 0.0% |

Selection rule: among variants whose verdicts stay stable under a uniform +25%
price rise (<0.5% changed), the best PR-AUC, preferring monotone constraints on
ties. Monotone constraints guarantee a larger overcharge never lowers the risk.
For reference, the previously shipped Model 2 reported PR-AUC 0.819, recall 0.862.

### Price logic check on all 59,910 cleaned invoices
| | ok | too_high | too_low |
|---|---|---|---|
| genuine | 56,543 | **0** | 0 |
| detention_padding | 824 | 307 | 0 |
| toll_inflation | 1,084 | 0 | 0 |
| weight_discrepancy | 1,102 | 50 | 0 |

No genuine invoice is flagged by the logic check. It acts as a safety net for
gross overcharges and the only line of defence for underbilling and
zero/negative amounts, which Model 2 never learned.

Learned bounds: {"cost_per_km_min": 7.71, "cost_per_km_max": 83.29, "billed_low_ratio": 0.724, "billed_high_ratio": 22.789, "billed_high_abs": 18554.0, "derived_from": "56,543 genuine invoices, out-of-fold fair cost"}

## 4. Artifacts
`code/output/models/model_1_freight_cost.pkl` and `model_2_invoice_risk.pkl`
are plain dict bundles (`joblib.load`), plus native `.json` boosters. The old
`.joblib` files were removed. Retrain with:

```
cd code && python export_models.py --outdir output
```
