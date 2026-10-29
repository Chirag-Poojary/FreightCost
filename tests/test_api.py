"""
API smoke tests. They run against the local CSV data (no Supabase, no AI keys)
and never call the extraction models.

    pip install -r backend/requirements.txt pytest
    SUPABASE_URL= SUPABASE_SERVICE_KEY= python -m pytest tests -q
"""
import os
import sys

import pytest

os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SERVICE_KEY"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

CLEAN = {"invoice_id": "INV037976", "order_id": "ORD042771", "vendor_id": "VEN022",
         "invoice_date": "27-11-2024", "actual_days": 3.02, "freight_base": 41360.17,
         "detention": 2280.0, "toll": 3079.65, "total": 46719.82}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["models_loaded"] is True
    assert body["database_mode"] == "local_csv"


def test_clean_invoice_is_approved(client):
    r = client.post("/api/score", json={"fields": CLEAN})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "scored"
    assert body["flagged"] is False
    assert 40000 < body["model1_predicted_cost"] < 55000


def test_missing_required_field_is_refused(client):
    fields = dict(CLEAN, toll=None)
    r = client.post("/api/score", json={"fields": fields})
    assert r.status_code == 422
    assert "toll" in r.json()["detail"]["missing_fields"]


def test_line_items_must_add_up(client):
    r = client.post("/api/score", json={"fields": dict(CLEAN, toll=100.0)})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "gate_failed"
    assert body["gate_reason"] == "arithmetic_mismatch"


def test_bill_far_below_fair_cost_is_flagged(client):
    low = dict(CLEAN, freight_base=14000.0, detention=0.0, toll=0.0, total=14000.0)
    body = client.post("/api/score", json={"fields": low}).json()
    assert body["flagged"] is True
    assert body["price_check"]["billed"]["status"] == "too_low"


def test_new_order_is_estimated(client):
    fields = {"invoice_id": "INV961850", "order_id": "ORD567288", "vendor_id": "VND391",
              "invoice_date": "13-12-2025", "actual_days": 5, "freight_base": 30000.0,
              "detention": 0.0, "toll": 2000.0, "total": 32000.0,
              "origin": "Jaipur", "destination": "Lucknow", "product_category": "Bulk Textiles"}
    body = client.post("/api/score", json={"fields": fields}).json()
    assert body["status"] == "scored"
    assert any("New order" in n for n in body["context"]["notes"])
    assert any("New carrier" in n for n in body["context"]["notes"])


def test_quote_rejects_impossible_input(client):
    payload = {"origin_hub_id": "HUB01", "dest_hub_id": "HUB05", "truck_type": "10-wheeler",
               "product_category": "Steel Coils", "billable_weight_kg": 12000, "distance_km": -5,
               "ideal_days": 1, "quoted_days": 1.2, "expected_fuel_price": 95, "expected_weather_score": 0}
    assert client.post("/api/quote", json=payload).status_code == 422


def test_quote_scales_with_diesel(client):
    base = {"origin_hub_id": "HUB01", "dest_hub_id": "HUB05", "truck_type": "10-wheeler",
            "product_category": "Steel Coils", "billable_weight_kg": 12000, "distance_km": 1410,
            "ideal_days": 0.98, "quoted_days": 1.2, "expected_fuel_price": 95, "expected_weather_score": 0}
    low = client.post("/api/quote", json=base).json()["predicted_cost"]
    high = client.post("/api/quote", json=dict(base, expected_fuel_price=120)).json()["predicted_cost"]
    assert high > low * 1.15   # fuel is most of the cost, so +26% diesel must show


def test_upload_rejects_non_image(client):
    r = client.post("/api/extract", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_carrier_ids_are_not_duplicated_by_case(client):
    ids = [v["vendor_id"] for v in client.get("/api/vendors").json()]
    assert all(i == i.upper() for i in ids)


def test_recent_audits_endpoint(client):
    rows = client.get("/api/audits/recent?limit=5").json()
    assert isinstance(rows, list) and len(rows) <= 5
