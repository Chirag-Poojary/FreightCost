from typing import Optional
from pydantic import BaseModel


class AuditResult(BaseModel):
    filename: str
    extractor_used: str
    extracted_fields: dict
    gate_passed: bool
    gate_reason: Optional[str] = None
    order_id: Optional[str] = None
    vendor_id: Optional[str] = None
    model1_predicted_cost: Optional[float] = None
    billed_amount: Optional[float] = None
    cost_mismatch: Optional[float] = None
    model2_proba: Optional[float] = None
    flagged: Optional[bool] = None
    explanation: Optional[str] = None


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
