"""FreightCost Phase 0: the 4-tier Confidence Gate as a pure function.

Mirrors the README gate (presence, referential integrity, arithmetic,
plausibility) but with configurable bounds so it can be tuned per document type.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from schema import BAD, norm_amount, norm_id, norm_number


@dataclass
class GateConfig:
    require: tuple = ("order_id", "vendor_id", "total", "actual_days")
    min_total: float = 500.0
    max_total: float = 250_000.0
    min_days: float = 0.2
    max_days: float = 30.0
    arith_abs_tol: float = 2.0
    arith_rel_tol: float = 0.02
    # If components needed for the arithmetic check are missing, fail (safe default).
    fail_on_uncheckable: bool = True


@dataclass
class GateResult:
    passed: bool
    reasons: list = field(default_factory=list)


def _num(fields: dict, key: str, fn):
    v = fn(fields.get(key))
    return None if v == BAD else v


def run_gate(fields: dict, cfg: GateConfig | None = None,
             order_ids: set | None = None, vendor_ids: set | None = None) -> GateResult:
    cfg = cfg or GateConfig()
    reasons: list[str] = []

    total = _num(fields, "total", norm_amount)
    days = _num(fields, "actual_days", norm_number)
    order_id = norm_id(fields.get("order_id"))
    vendor_id = norm_id(fields.get("vendor_id"))
    present = {"order_id": order_id, "vendor_id": vendor_id, "total": total, "actual_days": days}

    # 1. critical field presence
    for key in cfg.require:
        if present.get(key) is None:
            reasons.append(f"missing:{key}")

    # 2. referential integrity (only when reference sets are supplied)
    if order_ids is not None and order_id is not None and order_id not in order_ids:
        reasons.append("unknown_order_id")
    if vendor_ids is not None and vendor_id is not None and vendor_id not in vendor_ids:
        reasons.append("unknown_vendor_id")

    # 3. line-item arithmetic reconciliation
    base = _num(fields, "freight_base", norm_amount)
    det = _num(fields, "detention", norm_amount)
    toll = _num(fields, "toll", norm_amount)
    if total is not None:
        if base is None:
            if cfg.fail_on_uncheckable:
                reasons.append("arithmetic_uncheckable")
        else:
            parts = base + (det or 0.0) + (toll or 0.0)
            if abs(parts - total) > max(cfg.arith_abs_tol, total * cfg.arith_rel_tol):
                reasons.append("arithmetic_mismatch")

    # 4. plausibility bounds
    if total is not None and not (cfg.min_total < total <= cfg.max_total):
        reasons.append("total_out_of_bounds")
    if days is not None and not (cfg.min_days <= days <= cfg.max_days):
        reasons.append("days_out_of_bounds")

    return GateResult(passed=not reasons, reasons=reasons)
