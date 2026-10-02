"""FreightCost Phase 0: extraction schema and value normalisation.

Stdlib only. The 12 CORE fields match the current README benchmark, so existing
ground truth keeps working. EXTENDED fields and line_items are scored only when
the ground truth contains them.
"""
from __future__ import annotations

import re
from datetime import datetime

# field -> kind ("id", "gstin", "date", "text", "amount", "number")
CORE_FIELDS = {
    "invoice_id": "id",
    "order_id": "id",
    "vendor_id": "id",
    "invoice_date": "date",
    "origin": "text",
    "destination": "text",
    "freight_base": "amount",
    "detention": "amount",
    "toll": "amount",
    "total": "amount",
    "weight_kg": "number",
    "actual_days": "number",
}

EXTENDED_FIELDS = {
    "vendor_name": "text",
    "buyer_name": "text",
    "vendor_gstin": "gstin",
    "buyer_gstin": "gstin",
    "lr_number": "id",
    "eway_bill_no": "id",
    "vehicle_no": "id",
    "hsn_sac": "id",
    "cgst": "amount",
    "sgst": "amount",
    "igst": "amount",
    "currency": "text",
}

ALL_FIELDS = {**CORE_FIELDS, **EXTENDED_FIELDS}

# Fields the downstream fraud model depends on (the starred fields in the README).
CRITICAL_FIELDS = ("order_id", "vendor_id", "total", "actual_days")

LINE_ITEM_FIELDS = {
    "description": "text",
    "quantity": "number",
    "unit": "text",
    "rate": "amount",
    "amount": "amount",
}

BAD = "__unparseable__"  # sentinel: value present but could not be normalised

DATE_FORMATS = (
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d %b %Y", "%d-%b-%Y",
    "%d %B %Y", "%d-%b-%y", "%d/%m/%y", "%Y/%m/%d", "%b %d, %Y",
)


# ---------------------------------------------------------------- normalisers
def _is_blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def norm_amount(v):
    if _is_blank(v):
        return None
    if isinstance(v, bool):
        return BAD
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    s = re.sub(r"(?i)(rs\.?|inr|\u20b9)", "", str(v))
    s = s.replace(",", "").replace(" ", "")
    try:
        return round(float(s), 2)
    except ValueError:
        return BAD


def norm_number(v):
    if _is_blank(v):
        return None
    if isinstance(v, bool):
        return BAD
    if isinstance(v, (int, float)):
        return round(float(v), 3)
    s = re.sub(r"(?i)(kgs?|days?)", "", str(v)).replace(",", "").strip()
    try:
        return round(float(s), 3)
    except ValueError:
        return BAD


def norm_date(v):
    if _is_blank(v):
        return None
    s = str(v).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return BAD


def norm_text(v):
    if _is_blank(v):
        return None
    s = re.sub(r"[^0-9a-zA-Z\u0900-\u097F]+", " ", str(v).casefold())
    return " ".join(s.split()) or None


def norm_id(v):
    if _is_blank(v):
        return None
    return "".join(str(v).split()).upper()


def norm_gstin(v):
    return norm_id(v)


NORMALISERS = {
    "amount": norm_amount,
    "number": norm_number,
    "date": norm_date,
    "text": norm_text,
    "id": norm_id,
    "gstin": norm_gstin,
}


def normalise(kind: str, value):
    return NORMALISERS[kind](value)


def values_equal(kind: str, a, b, tol: float = 0.01) -> bool:
    """Compare two already-normalised values."""
    if a == BAD or b == BAD:
        return False
    if kind in ("amount", "number"):
        return abs(a - b) <= tol
    return a == b


# -------------------------------------------------------- JSON schema (decoding)
_JSON_TYPES = {
    "id": ["string", "null"],
    "gstin": ["string", "null"],
    "text": ["string", "null"],
    "date": ["string", "null"],  # YYYY-MM-DD
    "amount": ["number", "null"],
    "number": ["number", "null"],
}


def json_schema(include_extended: bool = True, include_line_items: bool = True) -> dict:
    """JSON Schema for constrained decoding. Every key is required but nullable,
    so the model must emit null instead of guessing when a field is absent."""
    fields = dict(CORE_FIELDS)
    if include_extended:
        fields.update(EXTENDED_FIELDS)
    props = {f: {"type": _JSON_TYPES[k]} for f, k in fields.items()}
    props["invoice_date"]["description"] = "ISO date YYYY-MM-DD"
    required = list(props)
    if include_line_items:
        item_props = {f: {"type": _JSON_TYPES[k]} for f, k in LINE_ITEM_FIELDS.items()}
        props["line_items"] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": item_props,
                "required": list(item_props),
                "additionalProperties": False,
            },
        }
        required.append("line_items")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }
