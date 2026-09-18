"""
Synthetic generation functions -- transcribed from the reference doc and kept
faithful to it. Phases 2, 6, 7, 9.

Where the doc specifies an error / anomaly / noise behaviour, it is reproduced
exactly so the dataset "looks real" rather than clean-synthetic.
"""
import numpy as np

# --- Phase 2: cargo (density-band + aspect-ratio) ---------------------------
DENSITY_BANDS_KG_PER_M3 = {
    "Steel Coils":            (2000, 4000),   # dense, low volume
    "Automotive Spare Parts": (400, 900),
    "Industrial Chemicals":   (900, 1400),    # drummed liquids
    "Bulk Textiles":          (150, 300),     # light, bulky
    "Electronics Components": (250, 600),
}


def generate_cargo(category, rng):
    lo, hi = DENSITY_BANDS_KG_PER_M3[category]
    weight_kg = rng.uniform(5000, 25000)
    density = rng.uniform(lo, hi)
    volume_m3 = weight_kg / density
    side = volume_m3 ** (1 / 3)
    length_cm = side * 100 * rng.uniform(0.9, 1.3)
    width_cm = side * 100 * rng.uniform(0.8, 1.1)
    height_cm = (volume_m3 * 1e6) / (length_cm * width_cm)
    return weight_kg, length_cm, width_cm, height_cm


DIM_DIVISOR = 5000  # cm^3 per kg -- standard Indian road-freight convention


# --- Phase 6: quoted days (padding factor, vendor-clustered) -----------------
def generate_quoted_days(ideal_days, padding_factor):
    """Doc uses rng.uniform(1.05,1.35); we pass a vendor-clustered factor in so
    Vendor_Historical_Risk_Score / Vendor_Padding_Ratio have real signal."""
    return round(ideal_days * padding_factor, 1)


# --- Phase 7: Base_Freight_Cost (the Model 1 label) -------------------------
MILEAGE_KMPL = {"12-wheeler": 4.0, "10-wheeler": 4.6, "6-wheeler": 6.5}
TOLL_RATE_PER_KM = 1.75          # rupees/km, NH blended average
DRIVER_ALLOWANCE_PER_DAY = 1200  # rupees/day, AIMTC reference


def base_freight_cost(distance_km, diesel_price, truck_type, ideal_days):
    mileage = MILEAGE_KMPL[truck_type]
    fuel_cost = (distance_km / mileage) * diesel_price
    tolls = distance_km * TOLL_RATE_PER_KM
    driver_cost = ideal_days * DRIVER_ALLOWANCE_PER_DAY
    return fuel_cost + tolls + driver_cost


def add_label_noise(cost, rng):
    """Doc: add +/-2% Gaussian noise so the label isn't perfectly deterministic."""
    return cost * rng.normal(1.0, 0.02)


# --- Phase 9: actual days --------------------------------------------------
def generate_actual_days(ideal_days, quoted_days, has_delay_anomaly, rng):
    # Wider normal variance so genuine late deliveries exist without fraud, and
    # the padded-delay anomaly overlaps them rather than sitting cleanly above.
    normal = max(ideal_days, quoted_days * rng.normal(1.02, 0.14))
    if not has_delay_anomaly:
        # ~8% of clean rows are genuinely (innocently) late
        if rng.random() < 0.08:
            normal += rng.uniform(0.5, 3.0)
        return normal
    # anomalous claimed delay: variable, sometimes small (a fraudster is subtle)
    return normal + rng.uniform(0.5, 3.5)


# --- Phase 9: billed amount by anomaly mechanism ---------------------------
def generate_billed_amount(base_cost, anomaly_type, rng):
    """Doc keeps the three mechanisms but warns the fraud must NOT be perfectly
    separable on cost alone -- the model should need weather/vendor/delay context.
    So normal invoices carry realistic legitimate surcharges (fuel escalation,
    loading/unloading, minor detention) and the anomaly magnitudes are randomised
    and overlap the top of that legitimate band."""
    # legitimate real-world billing variance: base +/-2% PLUS occasional genuine
    # surcharges that look large but are honest (fuel adj, handling, waiting).
    legit = base_cost * rng.normal(1.0, 0.025)
    if rng.random() < 0.35:                       # ~35% carry a real surcharge
        legit += rng.choice([1500, 2500, 4000, 6000, 9000],
                            p=[.30, .28, .22, .12, .08]) * rng.uniform(0.6, 1.1)
    if anomaly_type is None:
        return max(1.0, legit)
    if anomaly_type == "detention_padding":
        # padded wait-time fee, variable and sometimes modest -> overlaps legit
        return legit + rng.uniform(6000, 22000)
    if anomaly_type == "toll_inflation":
        # small inflation, deep inside the legitimate-surcharge band (hard case)
        return legit + rng.uniform(2000, 7000)
    if anomaly_type == "weight_discrepancy":
        return legit * rng.uniform(1.12, 1.40)    # FTL billed vs LTL actual
    raise ValueError(anomaly_type)


# Which anomaly types involve a claimed delay (affects actual_days generation).
DELAY_LINKED_ANOMALIES = {"detention_padding", "toll_inflation"}
