"""
Reference / master data for the Indian B2B road-freight pipeline.
Phase 1 backbone: hubs, vendors, and the toll_and_cost_assumptions constants.

All coordinates are the real lat/long of the actual Indian industrial/logistics
hub cities, so OSRM (when you run the real extraction) resolves genuine road
routes. Everything else in this module is a documented modeling assumption.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# hubs  --  15-20 real Indian industrial / logistics hub cities.
# hub_category: manufacturing | port | consumption | distribution
# ---------------------------------------------------------------------------
HUBS = [
    # hub_id, city, state, lat, lon, category
    ("HUB01", "Mumbai",      "Maharashtra",     19.0760, 72.8777, "port"),
    ("HUB02", "Pune",        "Maharashtra",     18.5204, 73.8567, "manufacturing"),
    ("HUB03", "Ahmedabad",   "Gujarat",         23.0225, 72.5714, "manufacturing"),
    ("HUB04", "Mundra",      "Gujarat",         22.8395, 69.7219, "port"),
    ("HUB05", "Delhi",       "Delhi",           28.7041, 77.1025, "consumption"),
    ("HUB06", "Gurugram",    "Haryana",         28.4595, 77.0266, "manufacturing"),
    ("HUB07", "Ludhiana",    "Punjab",          30.9010, 75.8573, "manufacturing"),
    ("HUB08", "Jaipur",      "Rajasthan",       26.9124, 75.7873, "distribution"),
    ("HUB09", "Kanpur",      "Uttar Pradesh",   26.4499, 80.3319, "manufacturing"),
    ("HUB10", "Kolkata",     "West Bengal",     22.5726, 88.3639, "port"),
    ("HUB11", "Chennai",     "Tamil Nadu",      13.0827, 80.2707, "port"),
    ("HUB12", "Coimbatore",  "Tamil Nadu",      11.0168, 76.9558, "manufacturing"),
    ("HUB13", "Bengaluru",   "Karnataka",       12.9716, 77.5946, "consumption"),
    ("HUB14", "Hyderabad",   "Telangana",       17.3850, 78.4867, "consumption"),
    ("HUB15", "Nagpur",      "Maharashtra",     21.1458, 79.0882, "distribution"),
    ("HUB16", "Indore",      "Madhya Pradesh",  22.7196, 75.8577, "distribution"),
    ("HUB17", "Visakhapatnam","Andhra Pradesh", 17.6868, 83.2185, "port"),
    ("HUB18", "Surat",       "Gujarat",         21.1702, 72.8311, "manufacturing"),
]

def build_hubs():
    return pd.DataFrame(HUBS, columns=[
        "hub_id", "city", "state", "latitude", "longitude", "hub_category"
    ])

# ---------------------------------------------------------------------------
# vendors  --  transport carrier master. fleet_size drives how many orders a
# vendor plausibly handles. Each vendor gets a latent "padding personality" and
# a latent "fraud propensity" so vendor-level signal is learnable downstream
# (these latent columns are NOT exposed to the models -- audit only).
# ---------------------------------------------------------------------------
VENDOR_NAMES = [
    "Bharat Roadways", "Gati Prime Logistics", "Shreeji Transport Co",
    "National Carriers Pvt Ltd", "Konkan Freight Lines", "Deccan Cargo Movers",
    "TCI Express Roadlink", "Safe & Fast Logistics", "Maruti Transport Corp",
    "Ganpati Roadways", "VRL Freight Services", "Blue Dart Surface",
    "Om Sai Transport", "Patel Roadways Ltd", "Rivigo Fleet Partners",
    "Ashok Leyland Logistics", "Jai Bhole Transport", "SpiceRoute Carriers",
    "Coastal Freight India", "Highway King Logistics", "Annapurna Roadlines",
    "Sri Venkatesh Cargo", "Northern Express Freight", "Metro Haul Logistics",
    "Trident Road Carriers",
]

def build_vendors(rng, n_vendors=25):
    names = VENDOR_NAMES[:n_vendors]
    states = build_hubs()["state"].unique()
    rows = []
    for i, name in enumerate(names):
        vid = f"VEN{i+1:03d}"
        fleet = int(rng.choice([8, 12, 20, 35, 60, 120, 200],
                               p=[.18, .22, .20, .15, .12, .08, .05]))
        # onboarding spread across ~6 years; some vendors are "new" (weak history)
        onboard = pd.Timestamp("2020-01-01") + pd.Timedelta(
            days=int(rng.uniform(0, 365 * 5.5)))
        # latent padding personality: mean padding factor this vendor tends to quote
        padding_mu = float(np.clip(rng.normal(1.15, 0.07), 1.02, 1.40))
        # latent fraud propensity: relative weight for being picked for anomalies
        fraud_propensity = float(np.clip(rng.gamma(2.0, 0.5), 0.1, 4.0))
        rows.append((vid, name, rng.choice(states), fleet, onboard,
                     padding_mu, fraud_propensity))
    df = pd.DataFrame(rows, columns=[
        "vendor_id", "vendor_name", "home_state", "fleet_size",
        "onboarding_date", "_latent_padding_mu", "_latent_fraud_propensity"
    ])
    return df

# ---------------------------------------------------------------------------
# toll_and_cost_assumptions  --  Base_Freight_Cost constants (doc Section:
# Base_Freight_Cost). Sourced to SIAM (mileage), NHAI/FASTag (toll/km),
# AIMTC (driver allowance).  effective_from lets historical rows stay
# reproducible after constants are revised.
# ---------------------------------------------------------------------------
def build_cost_assumptions():
    rows = [
        # truck_type, mileage_kmpl, toll_rate_per_km, driver_allowance_per_day, effective_from
        ("12-wheeler", 4.0, 1.75, 1200, "2023-01-01"),
        ("10-wheeler", 4.6, 1.55, 1100, "2023-01-01"),
        ("6-wheeler",  6.5, 1.20,  900, "2023-01-01"),
    ]
    return pd.DataFrame(rows, columns=[
        "truck_type", "mileage_kmpl", "toll_rate_per_km",
        "driver_allowance_per_day", "effective_from"
    ])

# Truck assignment rule based on gross weight (Indian axle-load norms:
# standard 12-wheeler caps ~16-21 t gross). Lighter loads -> smaller trucks.
def assign_truck_type(weight_kg):
    if weight_kg <= 9000:
        return "6-wheeler"
    if weight_kg <= 16000:
        return "10-wheeler"
    return "12-wheeler"

PRODUCT_CATEGORIES = [
    "Steel Coils", "Automotive Spare Parts", "Industrial Chemicals",
    "Bulk Textiles", "Electronics Components",
]
