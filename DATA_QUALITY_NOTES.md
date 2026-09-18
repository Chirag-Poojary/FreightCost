# Data Quality Notes — Intentional Real-World Issues

This dataset was built to *look real*, not clean-synthetic. Two classes of
imperfection are present: (A) issues the pipeline already **handles internally**
(reproduced faithfully from the reference doc), and (B) a **residue left in the
delivered files** so you must clean it before training. Both are listed here.

---

## A. Issues handled inside the pipeline (documented, not visible in output)

| Issue (from the reference doc) | Where | How handled |
|---|---|---|
| Open-Meteo returns `None` for some days | weather windows | filtered before summing in `weather_severity_score` / `adverse_weather_days` (nulls skipped) |
| PPAC publication gaps (~4% of dates missing) | `fuel_prices.csv` | `merge_asof(direction=backward, tolerance=7D)` + national-median fallback |
| Stale price carry-forward | fuel join | 7-day tolerance cap prevents indefinite forward-fill |
| Bad coordinates / no route | route cache | haversine lower-bound check; failing pairs dropped, not nulled |
| Label too deterministic | `base_freight_cost` | ±2% Gaussian noise added |
| Fraud perfectly separable on cost | billed amount | anomaly magnitudes randomised to overlap legitimate surcharges |

These reflect the doc's *Data Quality Considerations* sections. They are why, e.g.,
`fuel_prices.csv` has gaps but `orders.expected_fuel_price` is still populated.

---

## B. Residue left in the delivered files — YOU must clean these

Each is low-rate and independently reproducible from seed 42. Recommended fixes
are shown; `train_model2.py` already applies the invoice-side ones as an example.

### 1. Missing `expected_weather_score` — ~240 rows in `orders.csv`
Simulates an Open-Meteo window that returned all-null (no usable days).
**Fix:** impute (route×month median) or drop; do not leave NaN for tree models
that don't accept it.

### 2. Weight recorded in grams — ~180 rows in `orders.csv`
Data-entry unit drift: `weight_kg` is ~1000× too large (values > 26,000, some in
the millions). **Fix:** flag `weight_kg > 30000` (above legal Indian GVW) and
divide by 1000, or drop. Note this also corrupts `dimensional_weight_kg`/
`billable_weight_kg`/`truck_type` on those rows — recompute after fixing.

### 3. Duplicate invoice rows — 120 rows in `invoices.csv`
Double-submission in an AP export: same `invoice_id` appears twice.
**Fix:** `drop_duplicates(subset="invoice_id", keep="first")`.

### 4. Negative billed amounts — 90 rows in `invoices.csv`
Credit notes / reversals captured as negative `actual_billed_amount`.
**Fix:** exclude `actual_billed_amount <= 0` from training (they are not invoices
to audit), or handle separately in a credit-note workflow.

### 5. Inconsistent `vendor_id` casing — 25 rows in `invoices.csv`
E.g. `ven003` instead of `VEN003` — silently breaks joins to `vendors.csv` and
splits vendor history. **Fix:** `.str.upper()` before any vendor join or before
recomputing `vendor_historical_risk_score`.

### 6. Mixed `order_date` formats — 80 rows in `orders.csv`
Most dates are ISO `YYYY-MM-DD`; ~40 are `DD-MM-YYYY` (source-system drift).
**Fix:** `pd.to_datetime(..., dayfirst=True)` with format inference, or parse the
two formats separately. Left as strings so a naive `parse_dates=` will misread them.

---

## Suggested cleaning order
1. Standardise `order_date` (issue 6) → parse to datetime.
2. Fix `vendor_id` casing (issue 5) → before any vendor aggregation.
3. Correct grams-as-kg (issue 2) → then recompute weight-derived columns.
4. Dedup invoices (issue 3) and drop credit notes (issue 4).
5. Impute/drop missing weather scores (issue 1).
6. Only then split train/test and train.

## Audit column warning
`_ground_truth_audit.csv` (`anomaly_type`, `requires_manual_review`) is the hidden
generation truth. `anomaly_type` **must never** be used as a Model 2 feature — it
trivially leaks the label. It exists only for per-anomaly-type evaluation after
training.
