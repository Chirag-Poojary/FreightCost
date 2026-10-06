"""
Cloud invoice extraction -- replaces the localhost Ollama Qwen3-VL-4B path.

Every provider here speaks the OpenAI chat-completions protocol, so one
function handles all of them; only the base URL, key and model differ:

  gemini      Google Gemini via its OpenAI-compatible endpoint (recommended:
              strongest document OCR per rupee, reads tables/stamps/Hindi well)
  groq        Groq-hosted vision model (fast, free tier, weaker on tiny print)
  openrouter  any vision model on OpenRouter (GPT, Claude, Qwen-VL-72B, ...)
  custom      a self-hosted OpenAI-compatible server (vLLM / TGI / Ollama
              behind a public URL) -- this is how the original Qwen3-VL-4B can
              still be used once it runs on a cloud GPU instead of localhost.
  groq-ocr    Tesseract OCR text -> Groq text LLM (the original cloud path)

Model names are env-configurable because providers retire models often.
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO

FIELDS = [
    "invoice_id", "order_id", "vendor_id", "invoice_date", "origin",
    "destination", "truck_type", "actual_days", "weight_kg",
    "freight_base", "detention", "toll", "total", "vendor_gstin",
]

# Fields the two ML models (and the confidence gate) cannot run without.
# Anything here that the extractor misses must be typed in by the user.
REQUIRED_FIELDS = [
    "invoice_id", "order_id", "vendor_id", "invoice_date",
    "actual_days", "freight_base", "detention", "toll", "total",
]
NUMERIC_FIELDS = {"actual_days", "weight_kg", "freight_base", "detention", "toll", "total"}

PROVIDERS = {
    "gemini": {
        "label": "Google Gemini (cloud vision)",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.5-flash",
    },
    "groq": {
        "label": "Groq vision",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_VISION_MODEL",
        "default_model": "qwen/qwen3.6-27b",
    },
    "openrouter": {
        "label": "OpenRouter (any vision model)",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "qwen/qwen2.5-vl-72b-instruct",
    },
    "custom": {
        "label": "Self-hosted Qwen3-VL (cloud GPU)",
        "url_env": "CLOUD_VLM_URL",
        "key_env": "CLOUD_VLM_API_KEY",
        "model_env": "CLOUD_VLM_MODEL",
        "default_model": "Qwen/Qwen3-VL-4B-Instruct",
    },
}

# Order tried when the requested provider is not configured or errors out.
FALLBACK_ORDER = ["gemini", "openrouter", "groq", "custom", "groq-ocr"]

PROMPT = """You are reading an Indian road-freight (truck transport) tax invoice.
Return ONLY one JSON object with exactly these keys:

invoice_id    invoice number, usually like INV000123
order_id      shipment / order / LR reference, usually like ORD000123
vendor_id     carrier / transporter code, usually like VEN001
invoice_date  invoice date as DD-MM-YYYY
origin        origin city
destination   destination city
truck_type    e.g. 6-wheeler, 10-wheeler, 12-wheeler
actual_days   transit days as a number
weight_kg     billed / chargeable weight in kg as a number
freight_base  basic freight charge, number
detention     detention / halting charge, number (0 if the line exists but is blank or nil)
toll          toll charge, number
total         the final total payable, number
vendor_gstin  15-character GSTIN of the carrier

Rules:
- Amounts: plain numbers, no commas, no currency symbol. Indian grouping
  "1,23,456.00" means 123456.00.
- If a value is not printed on the document or you cannot read it, use null.
  Never guess, never compute a value that is not printed.
- Ignore stamps, signatures and handwritten notes unless they are the only source.
- No markdown, no commentary -- JSON only."""

_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                            "S": "5", "B": "8", "Z": "2"})


class ExtractionError(RuntimeError):
    pass


def _cfg_ready(name):
    if name == "groq-ocr":
        return bool(os.environ.get("GROQ_API_KEY"))
    cfg = PROVIDERS[name]
    if name == "custom":
        return bool(os.environ.get(cfg["url_env"]))
    return bool(os.environ.get(cfg["key_env"]))


def available_extractors():
    """What the frontend should offer, and which one to pre-select."""
    out = []
    for name in FALLBACK_ORDER:
        if name == "groq-ocr":
            label, model = "Tesseract OCR + Groq LLM", os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
        else:
            cfg = PROVIDERS[name]
            label, model = cfg["label"], os.environ.get(cfg["model_env"], cfg["default_model"])
        out.append({"id": name, "label": label, "model": model, "configured": _cfg_ready(name)})
    return out


def _encode_image(path, max_side=2000):
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def _post(url, key, body, timeout=90, retries=3):
    headers = {"content-type": "application/json", "user-agent": "FreightCost/2.0"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = e.headers.get("retry-after") if e.headers else None
                try:
                    time.sleep(min(20.0, float(wait) + 0.5) if wait else 2.0 * (attempt + 1))
                except ValueError:
                    time.sleep(2.0 * (attempt + 1))
                continue
            break
        except Exception as e:  # timeouts, DNS, connection reset
            last = str(e)
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
    raise ExtractionError(last or "request failed")


def _parse_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ExtractionError("model returned no JSON")
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ExtractionError(f"model returned invalid JSON: {e}")
    if not isinstance(parsed, dict):
        raise ExtractionError("model returned non-object JSON")
    return parsed


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().translate(_DIGIT_FIX) if re.search(r"\d", str(v)) else str(v)
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s) if s not in ("", ".", "-") else None
    except ValueError:
        return None


def normalise(parsed):
    """Coerce raw model output into canonical types; unreadable -> None."""
    out = {}
    for k in FIELDS:
        v = parsed.get(k)
        if k == "vendor_gstin" and v is None:
            v = parsed.get("gstin")
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a", "na", "-"):
            v = None
        if k in NUMERIC_FIELDS:
            v = _num(v)
        elif v is not None:
            v = str(v).strip()
        out[k] = v

    for field, prefix, ndig in (("invoice_id", "INV", 6), ("order_id", "ORD", 6), ("vendor_id", "VEN", 3)):
        val = out.get(field)
        if val:
            s = re.sub(r"\s+", "", val.upper())
            if s.startswith(prefix):
                digits = re.sub(r"\D", "", s[len(prefix):].translate(_DIGIT_FIX))
                if digits:
                    out[field] = f"{prefix}{int(digits):0{ndig}d}"
            else:
                out[field] = s

    if out.get("invoice_date"):
        out["invoice_date"] = normalise_date(out["invoice_date"])
    return out


def normalise_date(s):
    """Accept DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD, '05 Mar 2025'; return DD-MM-YYYY."""
    from datetime import datetime
    s = str(s).strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y-%m-%d", "%d-%m-%y", "%d/%m/%y",
                "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%d-%m-%Y")
        except ValueError:
            continue
    return s


def _extract_vision(provider, image_path):
    cfg = PROVIDERS[provider]
    url = os.environ.get(cfg["url_env"]) if provider == "custom" else cfg["url"]
    if provider == "custom":
        url = url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
    key = os.environ.get(cfg["key_env"], "")
    model = os.environ.get(cfg["model_env"], cfg["default_model"])
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
        ]}],
    }
    try:
        resp = _post(url, key, body)
    except ExtractionError as e:
        # Some OpenAI-compatible servers reject response_format; retry once without it.
        if "response_format" in str(e) or "json_object" in str(e):
            body.pop("response_format")
            resp = _post(url, key, body)
        else:
            raise
    content = resp["choices"][0]["message"].get("content") or ""
    return normalise(_parse_json(content)), model


def _extract_ocr_llm(image_path):
    import pytesseract
    from PIL import Image
    text = pytesseract.image_to_string(Image.open(image_path), config="--psm 6")
    model = os.environ.get("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
    body = {
        "model": model, "temperature": 0, "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content":
                      PROMPT + "\n\nThe document has been OCR'd (may contain character errors):\n---\n"
                      + text[:6000] + "\n---"}],
    }
    resp = _post("https://api.groq.com/openai/v1/chat/completions", os.environ.get("GROQ_API_KEY"), body)
    return normalise(_parse_json(resp["choices"][0]["message"]["content"])), model


def extract(image_path, provider="auto"):
    """Run extraction with automatic fallback.

    Returns (fields, meta) where meta records which provider/model actually
    answered and any providers that failed on the way -- surfaced in the UI so
    a silent fallback never hides a misconfigured key.
    """
    order = [p for p in FALLBACK_ORDER if _cfg_ready(p)]
    if provider in order:
        order.remove(provider)
        order.insert(0, provider)
    if not order:
        raise ExtractionError(
            "No extractor configured. Set GEMINI_API_KEY (recommended), GROQ_API_KEY, "
            "OPENROUTER_API_KEY or CLOUD_VLM_URL on the backend.")

    errors = []
    for p in order:
        try:
            if p == "groq-ocr":
                fields, model = _extract_ocr_llm(image_path)
            else:
                fields, model = _extract_vision(p, image_path)
            return fields, {"provider": p, "model": model, "errors": errors}
        except Exception as e:
            errors.append({"provider": p, "error": str(e)[:300]})
    raise ExtractionError("All extractors failed: " + "; ".join(f"{e['provider']}: {e['error']}" for e in errors))


def missing_required(fields):
    return [f for f in REQUIRED_FIELDS if fields.get(f) in (None, "")]
