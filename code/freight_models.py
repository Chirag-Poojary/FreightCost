"""
Shared model logic for training (export_models.py) and serving (backend/).

MODEL 1 -- cost-baseline hybrid
  A plain XGBRegressor on raw rupees cannot extrapolate: trees return a
  constant outside the training range, so diesel above Rs 97.35/L or a route
  longer than ~2,850 km got the same prediction as the edge of the data.
  Model 1 therefore predicts in two parts:

      baseline  = (distance / truck mileage) * diesel   fuel
                + distance * toll rate                   tolls
                + ideal_days * driver allowance          driver
      predicted = baseline * xgb_ratio(features)

  The baseline is the standard Indian road-freight cost build-up and scales
  linearly with diesel price and distance, so it extrapolates correctly. The
  XGBoost part learns everything the formula misses (cargo type, weight,
  weather, quoted vs ideal days) as a multiplicative correction around 1.0;
  that correction is bounded, which is exactly what we want from trees.

SANITY CHECKS
  check_inputs()  rejects physically impossible inputs before any model runs.
  check_price()   compares a rupee amount (a prediction or a billed total)
                  against the fair cost and against learned bounds, and returns
                  ok / too_low / too_high / invalid with a plain message.

Bundles are saved as plain dicts of xgboost/sklearn objects (+ JSON-able
metadata) so a .pkl loads anywhere without importing this file.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Cost build-up constants (AIMTC / NHAI blended references; same values the
# dataset generator uses). Stored inside the model bundle at training time.
BASELINE_CONSTANTS = {
    "mileage_kmpl": {"12-wheeler": 4.0, "10-wheeler": 4.6, "6-wheeler": 6.5},
    "toll_rate_per_km": 1.75,
    "driver_allowance_per_day": 1200.0,
}

M1_NUM = ["distance_km", "ideal_days", "quoted_days", "billable_weight_kg",
          "expected_fuel_price", "expected_weather_score"]

# Hard physical limits for inputs. Outside these the request is rejected.
INPUT_LIMITS = {
    "distance_km":            (1.0, 6000.0),     # longest Indian road corridor ~4,000 km
    "billable_weight_kg":     (1.0, 60000.0),    # above the heaviest legal GVW
    "expected_fuel_price":    (40.0, 250.0),     # Rs/L
    "ideal_days":             (0.01, 30.0),
    "quoted_days":            (0.01, 45.0),
    "expected_weather_score": (0.0, 200.0),
}


# --------------------------------------------------------------------- inputs
def check_inputs(order: dict) -> list[str]:
    """Return a list of problems; empty list means the inputs are usable."""
    problems = []
    for k, (lo, hi) in INPUT_LIMITS.items():
        v = order.get(k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            problems.append(f"{k} is missing or not a number.")
            continue
        if math.isnan(v) or math.isinf(v):
            problems.append(f"{k} is not a finite number.")
        elif v < lo or v > hi:
            problems.append(f"{k} = {v:g} is outside the possible range {lo:g} to {hi:g}.")
    if order.get("truck_type") not in BASELINE_CONSTANTS["mileage_kmpl"]:
        problems.append(f"truck_type '{order.get('truck_type')}' is not one of "
                        f"{list(BASELINE_CONSTANTS['mileage_kmpl'])}.")
    return problems


# -------------------------------------------------------------------- baseline
def baseline_components(df: pd.DataFrame, const=BASELINE_CONSTANTS) -> pd.DataFrame:
    mileage = df["truck_type"].map(const["mileage_kmpl"]).astype(float)
    fuel = df["distance_km"] / mileage * df["expected_fuel_price"]
    tolls = df["distance_km"] * const["toll_rate_per_km"]
    driver = df["ideal_days"] * const["driver_allowance_per_day"]
    return pd.DataFrame({"fuel": fuel, "tolls": tolls, "driver": driver,
                         "baseline": fuel + tolls + driver}, index=df.index)


def build_feature_matrix(df: pd.DataFrame, categories, truck_types):
    """Numeric + one-hot(category) + one-hot(truck_type), fixed column order."""
    X = df[M1_NUM].astype(float).copy()
    for c in categories:
        X[f"cat_{c}"] = (df["product_category"] == c).astype(int)
    for t in truck_types:
        X[f"truck_{t}"] = (df["truck_type"] == t).astype(int)
    order = M1_NUM + [f"cat_{c}" for c in categories] + [f"truck_{t}" for t in truck_types]
    return X[order], order


# --------------------------------------------------------------------- Model 1
class FreightCostModel:
    """Serving wrapper around a Model 1 bundle.

    bundle = {"point": XGBRegressor, "p10": XGBRegressor, "p90": XGBRegressor,
              "meta": {...column_order, categories, truck_types, baseline
                       constants, price_bounds, training_ranges...}}
    """

    def __init__(self, bundle: dict):
        self.point = bundle["point"]
        self.p10 = bundle.get("p10")
        self.p90 = bundle.get("p90")
        self.meta = bundle["meta"]
        self.columns = self.meta["column_order"]
        self.const = self.meta["baseline_constants"]

    # .. internals
    def _frame(self, order: dict) -> pd.DataFrame:
        row = {k: order.get(k) for k in M1_NUM + ["product_category", "truck_type"]}
        return pd.DataFrame([row])

    def _X(self, df):
        X, _ = build_feature_matrix(df, self.meta["product_categories"], self.meta["truck_types"])
        return X[self.columns]

    # .. public API (same surface as the old audit_core.Model1)
    def predict(self, order: dict) -> float:
        return self.predict_interval(order)[0]

    def predict_interval(self, order: dict):
        df = self._frame(order)
        base = float(baseline_components(df, self.const)["baseline"].iloc[0])
        X = self._X(df)
        r = float(self.point.predict(X)[0])
        lo = float(self.p10.predict(X)[0]) if self.p10 is not None else r * 0.97
        hi = float(self.p90.predict(X)[0]) if self.p90 is not None else r * 1.03
        point = base * r
        lower, upper = base * min(lo, hi, r), base * max(lo, hi, r)
        return point, lower, upper

    def explain_prediction(self, order: dict, top_k: int = 3):
        """Rupee breakdown: the three cost-build-up parts plus the model's
        adjustment, and the features that drove that adjustment."""
        import xgboost as xgb
        df = self._frame(order)
        comp = baseline_components(df, self.const).iloc[0]
        X = self._X(df)
        booster = self.point.get_booster()
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0]
        base_ratio = float(contribs[-1])
        ratio = float(contribs.sum())
        baseline = float(comp["baseline"])
        adjustment = baseline * (ratio - 1.0)

        # SHAP values are in ratio units; convert to rupees on this order's baseline.
        per_feat = dict(zip(self.columns, contribs[:-1]))
        grouped = {}
        for k, v in per_feat.items():
            key = "product_category" if k.startswith("cat_") else "truck_type" if k.startswith("truck_") else k
            grouped[key] = grouped.get(key, 0.0) + float(v) * baseline
        drivers = sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True)

        breakdown = [
            {"feature": "fuel_cost", "impact": round(float(comp["fuel"]), 2)},
            {"feature": "tolls", "impact": round(float(comp["tolls"]), 2)},
            {"feature": "driver_allowance", "impact": round(float(comp["driver"]), 2)},
            {"feature": "model_adjustment", "impact": round(adjustment, 2)},
        ]
        top = sorted(breakdown, key=lambda d: abs(d["impact"]), reverse=True)[:top_k]
        return {
            "base_value": round(baseline * base_ratio, 2),
            "baseline": round(baseline, 2),
            "breakdown": breakdown,
            "attributions": {k: round(v, 2) for k, v in grouped.items()},
            "adjustment_drivers": [{"feature": k, "impact": round(v, 2)} for k, v in drivers[:top_k]],
            "top_drivers": top,
        }

    # .. sanity
    def check_prediction(self, order: dict, predicted: float) -> dict:
        """Is the model's own output plausible for this route and load?"""
        pb = self.meta["price_bounds"]
        dist = max(float(order.get("distance_km") or 0), 1.0)
        if not np.isfinite(predicted) or predicted <= 0:
            return {"status": "invalid", "message": f"Predicted cost {predicted:,.2f} is zero or negative."}
        per_km = predicted / dist
        if per_km < pb["cost_per_km_min"]:
            return {"status": "too_low",
                    "message": f"Predicted Rs {per_km:,.2f}/km is below the plausible minimum "
                               f"Rs {pb['cost_per_km_min']:,.2f}/km. Check the inputs."}
        if per_km > pb["cost_per_km_max"]:
            return {"status": "too_high",
                    "message": f"Predicted Rs {per_km:,.2f}/km is above the plausible maximum "
                               f"Rs {pb['cost_per_km_max']:,.2f}/km. Check the inputs."}
        return {"status": "ok", "message": None}

    def check_billed(self, billed, fair: float, fair_lower: float, fair_upper: float) -> dict:
        """Is a billed invoice amount plausible against the fair cost?"""
        pb = self.meta["price_bounds"]
        try:
            billed = float(billed)
        except (TypeError, ValueError):
            return {"status": "invalid", "message": "Billed amount is not a number."}
        if not np.isfinite(billed) or billed <= 0:
            return {"status": "invalid",
                    "message": f"Billed amount Rs {billed:,.2f} is zero or negative (credit note or data-entry error)."}
        ratio = billed / max(fair, 1.0)
        if billed < fair_lower * pb["billed_low_ratio"]:
            return {"status": "too_low", "ratio": round(ratio, 3),
                    "message": f"Billed Rs {billed:,.0f} is only {ratio:.0%} of the fair cost "
                               f"Rs {fair:,.0f}. No genuine invoice in the training data was billed "
                               f"below {pb['billed_low_ratio']:.0%} of the fair lower bound. Likely a "
                               f"partial invoice, wrong order ID or a typo."}
        excess = billed - fair_upper
        if excess > pb["billed_high_abs"] or billed > fair_upper * pb["billed_high_ratio"]:
            return {"status": "too_high", "ratio": round(ratio, 3),
                    "message": f"Billed Rs {billed:,.0f} is Rs {excess:,.0f} above the fair upper bound "
                               f"Rs {fair_upper:,.0f} ({ratio:.1f}x the fair cost). No genuine invoice in "
                               f"the training data exceeded the upper bound by more than "
                               f"Rs {pb['billed_high_abs'] / 1.5:,.0f}."}
        return {"status": "ok", "ratio": round(ratio, 3), "message": None}


# --------------------------------------------------------------------- Model 2
class InvoiceRiskModel:
    """bundle = {"model": XGBClassifier, "features": [...], "threshold": 0.5,
                 "rupee_deflator": {"reference_fuel_price": 91.5, "features": [...]} | None}

    With a deflator, rupee-valued features are expressed in training-period
    rupees (divided by diesel / reference diesel) before scoring, so a
    nationwide price rise alone does not raise the fraud score."""

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = float(bundle.get("threshold", 0.5))
        self.deflator = bundle.get("rupee_deflator")

    def model_inputs(self, features: dict, fuel_price=None) -> dict:
        x = {k: features[k] for k in self.features}
        if self.deflator and fuel_price:
            d = float(fuel_price) / self.deflator["reference_fuel_price"]
            for k in self.deflator["features"]:
                if k in x:
                    x[k] = x[k] / d
        return x

    def predict_proba(self, features: dict, fuel_price=None) -> float:
        x = pd.DataFrame([self.model_inputs(features, fuel_price)])[self.features]
        return float(self.model.predict_proba(x)[0, 1])


def load_bundle(path: str):
    import joblib
    return joblib.load(path)
