"""
PHASE A / EXTRACTION -- Multimodal VLM and Cloud LLM Invoice Field Extraction.

Two Extraction Engines Supported:
1. Local Multimodal VLM: Qwen3-VL-4B-Instruct via local Ollama daemon (direct image -> JSON)
2. Cloud LLM: Groq (Qwen3.8-27B) via Tesseract OCR + Fast LPU JSON extraction

Scoring is per-field exact match against ground truth, with numeric fields
compared within tolerance.

Usage:
  python ocr_extract.py --indir ../phase_a_output/invoices_scanned --extractor qwen3-vl
  python ocr_extract.py --indir ../phase_a_output/invoices_scanned --extractor groq
"""
import argparse
import json
import os
import re

import pandas as pd
import pytesseract
from PIL import Image

try:
    from dotenv import load_dotenv
    # Load .env from current directory or workspace root
    load_dotenv()
    _env_parent = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_parent):
        load_dotenv(_env_parent)
except ImportError:
    pass

# Auto-detect Tesseract binary & tessdata if not in system PATH
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.dirname(_SCRIPT_DIR)
_CANDIDATE_TESS = [
    os.path.join(_BUNDLE_DIR, "tesseract_bin", "tesseract.exe"),
    os.path.join(_SCRIPT_DIR, "tesseract_bin", "tesseract.exe"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Chirag\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]
for _p in _CANDIDATE_TESS:
    if os.path.exists(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        _tessdata = os.path.join(os.path.dirname(_p), "tessdata")
        if os.path.exists(_tessdata) and "TESSDATA_PREFIX" not in os.environ:
            os.environ["TESSDATA_PREFIX"] = _tessdata
        break

# Fields the extractor must recover. The starred ones are what Model 2 needs.
TARGET_FIELDS = [
    "invoice_id",      # *
    "order_id",        # *
    "vendor_id",       # *
    "total",           # * -> actual_billed_amount
    "actual_days",     # *
    "invoice_date",
    "origin",
    "destination",
    "truck_type",
    "freight_base",
    "detention",
    "toll",
]
MODEL2_CRITICAL = ["invoice_id", "order_id", "vendor_id", "total", "actual_days"]

AMOUNT_RE = r"[-+]?[\d,]+(?:\.\d{1,2})?"


def ocr(path):
    """OCR with layout preserved. psm 6 = assume a uniform block of text,
    which keeps table rows on their own lines instead of scrambling columns."""
    img = Image.open(path)
    return pytesseract.image_to_string(img, config="--psm 6")


def _num(s):
    """Parse an Indian-format amount string to float. '1,23,456.78' -> 123456.78"""
    try:
        return float(re.sub(r"[^\d.\-]", "", str(s)))
    except (ValueError, TypeError):
        return None


# OCR reliably confuses these glyphs inside the digit half of coded IDs.
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                            "S": "5", "B": "8", "Z": "2"})


def extract_rules(text):
    """Rule-based extraction has been removed from this project.
    Redirecting to Groq LLM extraction.
    """
    return extract_llm(text, provider="groq")


def extract_vlm(image_path, model=None, endpoint=None, backend="api", model_path=None):
    """Direct multimodal Vision-Language extraction using local Qwen3-VL-4B.
    Bypasses Tesseract OCR errors by reading image, layout, and tables directly.
    """
    try:
        from extract_qwen3_vl import Qwen3VLExtractor
        ep = endpoint or os.environ.get("LOCAL_VLM_URL", "http://localhost:11434/v1")
        mdl = model or os.environ.get("LOCAL_VLM_MODEL", "qwen3-vl:4b")
        extractor = Qwen3VLExtractor(endpoint=ep, model_name=mdl, backend=backend, model_path=model_path)
        return extractor.extract_from_image(image_path)
    except Exception as e:
        print(f"[VLM Fallback] Failed to run direct VLM extraction ({e}), falling back to OCR + Groq LLM")
        text = ocr(image_path)
        return extract_llm(text, provider="groq")





LLM_PROMPT = """You are extracting structured data from OCR text of an Indian \
road-freight invoice. The OCR output may contain character errors.

Return ONLY a JSON object, no prose and no markdown fences, with exactly these keys:
invoice_id, order_id, vendor_id, invoice_date, origin, destination, truck_type,
actual_days, freight_base, detention, toll, total

Rules:
- invoice_id looks like INV000123; order_id like ORD000123; vendor_id like VEN001.
- invoice_date must be DD-MM-YYYY.
- freight_base, detention, toll, total are numbers only (no commas, no currency symbol).
- total is the final payable amount.
- actual_days is the transit-days number.
- Use null for any field you cannot find. Never guess a value that is not present.

OCR TEXT:
---
{text}
---"""


# Any OpenAI-compatible provider works. All of these have a permanent free
# tier and issue keys without a credit card (verified 2026). Pick with
# --provider; the key comes from the matching env var.
PROVIDERS = {
    "groq": {          # 30 RPM / 14,400 RPD, very fast -- best default here
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model": "qwen/qwen3.8-27b",
    },
    "gemini": {        # free tier, strongest model quality of the free options
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
    },
    "openrouter": {    # widest model choice through one key
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "cerebras": {
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "key_env": "CEREBRAS_API_KEY",
        "model": "llama-3.3-70b",
    },
    "anthropic": {     # paid; kept for parity if a key is available
        "url": "https://api.anthropic.com/v1/messages",
        "key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
}


def extract_llm(text, provider="groq", model=None, retries=6):
    """LLM-based extraction via any OpenAI-compatible free-tier provider.

    Handles OCR character noise far better than regex because it reads each
    field semantically instead of matching a fixed shape -- which is exactly
    where the rule-based baseline loses ground on column layouts.
    """
    import urllib.error
    import urllib.request
    import time

    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise SystemExit(
            f"Set {cfg['key_env']} to use --provider {provider}. "
            f"Free key, no credit card: see README.")
    model = model or cfg["model"]
    prompt = LLM_PROMPT.format(text=text[:6000])

    if provider == "anthropic":
        body = {"model": model, "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"content-type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01",
                   "user-agent": "FreightOcr/1.0"}
    else:
        body = {"model": model, "max_tokens": 1000, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        headers = {"content-type": "application/json",
                   "authorization": f"Bearer {key}",
                   "user-agent": "FreightOcr/1.0"}

    for attempt in range(retries):
        try:
            req = urllib.request.Request(cfg["url"],
                                         data=json.dumps(body).encode(),
                                         headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            # 429 = free-tier rate limit; back off rather than lose the run.
            if e.code == 429 and attempt < retries - 1:
                retry_after = e.headers.get("retry-after") if hasattr(e, "headers") else None
                try:
                    wait = float(retry_after) + 1.0 if retry_after else (3 * (attempt + 1) + 2.0)
                except (ValueError, TypeError):
                    wait = 5.0 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
    else:
        return {k: None for k in TARGET_FIELDS}

    if provider == "anthropic":
        raw = "".join(b.get("text", "") for b in resp.get("content", []))
    else:
        raw = resp["choices"][0]["message"]["content"]

    # Clean potential reasoning/thinking tags and markdown fences
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m_json = re.search(r"\{.*\}", raw, re.DOTALL)
    if m_json:
        raw = m_json.group(0)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {k: None for k in TARGET_FIELDS}

    res = {k: parsed.get(k) for k in TARGET_FIELDS}
    for field, prefix, ndig in [("invoice_id", "INV", 6), ("order_id", "ORD", 6), ("vendor_id", "VEN", 3)]:
        val = res.get(field)
        if val:
            v_str = str(val).strip().upper()
            if v_str.startswith(prefix):
                tail = v_str[len(prefix):].translate(_DIGIT_FIX)
                digits = re.sub(r"\D", "", tail)
                if digits:
                    res[field] = f"{prefix}{int(digits):0{ndig}d}"
    return res


def match(field, pred, truth):
    """Exact match for text fields; tolerance-based for numerics, since an
    amount is correct if it means the same number, not if the string matches."""
    if pred is None:
        return False
    if field in ("freight_base", "detention", "toll", "total", "actual_days"):
        p, t = _num(pred), _num(truth)
        if p is None or t is None:
            return False
        return abs(p - t) <= max(0.02, abs(t) * 0.001)
    return str(pred).strip().lower() == str(truth).strip().lower()


def main(indir, extractor, limit, outfile, provider, model):
    manifest = json.load(open(os.path.join(indir, "ground_truth.json")))
    if limit:
        manifest = manifest[:limit]
    tag = extractor if extractor == "qwen3-vl" else f"llm:{provider}"
    print(f"Extraction ({tag}) on {len(manifest)} invoices from {indir}")

    rows, records = [], []
    for i, entry in enumerate(manifest, 1):
        path = os.path.join(indir, entry["image"])
        if extractor in ("qwen3-vl", "vlm", "local"):
            pred = extract_vlm(path, model=model or "qwen3-vl:4b")
        else:
            text = ocr(path)
            pred = extract_llm(text, provider="groq", model=model)
            import time
            time.sleep(1.0)
        truth = entry["ground_truth"]

        res = {f: match(f, pred.get(f), truth.get(f)) for f in TARGET_FIELDS}
        rows.append({"image": entry["image"], "layout": entry["layout"], **res})
        records.append({"image": entry["image"], "layout": entry["layout"],
                        "predicted": pred, "ground_truth": truth, "correct": res})
        if i % 25 == 0 or i == len(manifest):
            print(f"  {i}/{len(manifest)}")

    df = pd.DataFrame(rows)
    per_field = df[TARGET_FIELDS].mean().sort_values(ascending=False)
    crit = df[MODEL2_CRITICAL].mean()
    all_crit_ok = df[MODEL2_CRITICAL].all(axis=1).mean()

    print("\n=== Per-field extraction accuracy ===")
    for f, v in per_field.items():
        star = " *" if f in MODEL2_CRITICAL else ""
        print(f"  {f:<16} {v:6.1%}{star}")
    print(f"\n  Mean over all fields          : {per_field.mean():.1%}")
    print(f"  Mean over Model-2 critical (*) : {crit.mean():.1%}")
    print(f"  Invoices with ALL critical OK  : {all_crit_ok:.1%}")

    if df["layout"].nunique() > 1:
        print("\n=== Mean accuracy by layout ===")
        for lay, g in df.groupby("layout"):
            print(f"  {lay:<18} {g[TARGET_FIELDS].mean().mean():.1%}")

    with open(outfile, "w") as f:
        json.dump({"extractor": tag, "n": len(manifest),
                   "per_field": per_field.to_dict(),
                   "critical_mean": float(crit.mean()),
                   "all_critical_ok": float(all_crit_ok),
                   "records": records}, f, indent=1, default=str)
    print(f"\nWrote detailed results -> {outfile}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Invoice Field Extraction (Qwen3-VL & Groq LLM)")
    ap.add_argument("--indir", default="../phase_a_output/invoices_scanned")
    ap.add_argument("--extractor", choices=["qwen3-vl", "groq", "llm"], default="qwen3-vl",
                    help="extraction engine: 'qwen3-vl' (local vision model) or 'groq' (cloud LLM)")
    ap.add_argument("--provider", default="groq", help="cloud provider (default: groq)")
    ap.add_argument("--model", default=None, help="override default model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="extraction_results.json")
    a = ap.parse_args()
    main(a.indir, a.extractor, a.limit, a.out, a.provider, a.model)

