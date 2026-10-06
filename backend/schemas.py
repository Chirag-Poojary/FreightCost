from typing import Any, Optional
from pydantic import BaseModel


class ExtractResponse(BaseModel):
    filename: Optional[str] = None
    extractor_used: str
    extractor_model: Optional[str] = None
    fallback_errors: list[dict] = []
    fields: dict
    required_fields: list[str]
    missing_fields: list[str]
    field_issues: dict = {}
    order_summary: Optional[dict] = None
    ready_to_score: bool


class ScoreRequest(BaseModel):
    fields: dict
    ai_fields: Optional[dict] = None          # what the model originally read
    filename: Optional[str] = None
    extractor_used: Optional[str] = None
    extractor_model: Optional[str] = None


class AuditResult(BaseModel):
    status: str = "scored"                    # scored | gate_failed | needs_input
    filename: Optional[str] = None
    extractor_used: str
    extractor_model: Optional[str] = None
    extracted_fields: dict
    manually_entered_fields: list[str] = []
    gate_passed: bool
    gate_reason: Optional[str] = None
    field_errors: list[str] = []
    order_id: Optional[str] = None
    vendor_id: Optional[str] = None
    model1_predicted_cost: Optional[float] = None
    billed_amount: Optional[float] = None
    cost_mismatch: Optional[float] = None
    model1_interval_lower: Optional[float] = None
    model1_interval_upper: Optional[float] = None
    cost_drivers: Optional[list[dict]] = None
    model2_proba: Optional[float] = None
    flagged: Optional[bool] = None
    flag_source: Optional[str] = None          # model | logic_check | model+logic_check
    price_check: Optional[dict] = None
    explanation: Optional[str] = None
    context: dict[str, Any] = {}
    drift_warnings: list[str] = []


class VendorRisk(BaseModel):
    vendor_id: str
    vendor_name: Optional[str] = None
    total_invoices: int
    flagged_invoices: int
    risk_score: float


class DashboardStats(BaseModel):
    total_processed: int
    gate_passed: int
    gate_rejected: int
    auto_processing_rate: float
    flagged_count: int
    rejection_breakdown: dict
    manual_entry_count: int = 0


class QuoteRequest(BaseModel):
    origin_hub_id: str
    dest_hub_id: str
    truck_type: str
    product_category: str
    billable_weight_kg: float
    distance_km: float
    ideal_days: float
    quoted_days: float
    expected_fuel_price: float
    expected_weather_score: float


class QuoteResponse(BaseModel):
    predicted_cost: float
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    top_drivers: Optional[list[dict]] = None
    cost_breakdown: Optional[list[dict]] = None
    adjustment_drivers: Optional[list[dict]] = None
    price_check: Optional[dict] = None
    drift_warnings: list[str] = []


class ContextResponse(BaseModel):
    route: Optional[dict] = None
    fuel: dict
    weather: dict
    drift_warnings: list[str] = []
    training_ranges: dict = {}
