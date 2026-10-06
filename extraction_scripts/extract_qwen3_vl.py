"""
Zero-shot and Local Extraction for Invoices using Qwen3-VL-4B-Instruct.
Conforms strictly to the FreightCost Phase 0 Schema and Gate architecture.

Features:
- Direct Multimodal Vision-Language extraction (no OCR error propagation)
- Constrained output conforming to schema.json_schema()
- Supports:
    1. Local OpenAI-compatible API / vLLM / Ollama server (e.g., http://localhost:8000/v1 or http://localhost:11434/v1)
    2. Local HuggingFace / Transformers weights (AutoModelForVision2Seq / pipeline)
    3. Fallback to Cloud LLMs with structured prompts
- Outputs predictions in Phase 0 evaluate.py format and auto-scores against ground truth.

Usage:
    # Query local VLM server (vLLM / Ollama / llama-server):
    python extraction_scripts/extract_qwen3_vl.py --indir phase_a_output/invoices_scanned --endpoint http://localhost:8000/v1

    # Using local checkpoint directory or HF ID with Transformers:
    python extraction_scripts/extract_qwen3_vl.py --indir phase_a_output/invoices_scanned --backend transformers --model-path /path/to/Qwen3-VL-4B-Instruct

    # Evaluate existing prediction output:
    python evaluate.py --gt phase_a_output/gt_phase0.json --pred phase_a_output/pred_qwen3_vl.json
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.dirname(_CURR_DIR)
for p in [_CURR_DIR, os.path.join(_CURR_DIR, "phase0"), _BUNDLE_DIR]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv
    load_dotenv()
    _env_path = os.path.join(_BUNDLE_DIR, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

from schema import CORE_FIELDS, EXTENDED_FIELDS, ALL_FIELDS, json_schema, normalise, BAD
from gate import run_gate
from evaluate import evaluate, format_report

DEFAULT_ENDPOINT = os.environ.get("LOCAL_VLM_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("LOCAL_VLM_MODEL", "qwen3-vl:4b")

SYSTEM_PROMPT = """Extract the structured invoice fields from this road-freight invoice image.
Return ONLY a valid JSON object with these keys (use null for any absent field):
{
  "invoice_id": "e.g. INV000123",
  "order_id": "e.g. ORD000123",
  "vendor_id": "e.g. VEN001",
  "invoice_date": "YYYY-MM-DD or DD-MM-YYYY",
  "origin": "city name",
  "destination": "city name",
  "actual_days": 1.5,
  "weight_kg": 5000.0,
  "freight_base": 10000.0,
  "detention": 0.0,
  "toll": 500.0,
  "total": 10500.0,
  "vendor_gstin": "string or null"
}
Rules:
- For monetary fields (freight_base, detention, toll, total), output pure numbers only without commas or currency symbols.
- For absent fields, output null. Never guess or hallucinate unstated values.
- Do not output any markdown or commentary outside the JSON object.
"""


def encode_image_to_base64(image_path: str, max_size: int = 1536) -> str:
    """Load image, optionally resize for VRAM efficiency, and convert to base64."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Downscale if excessively large to comfortably fit 6GB-12GB VRAM
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Qwen3VLExtractor:
    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, model_name: str = DEFAULT_MODEL,
                 api_key: str | None = None, backend: str = "api", model_path: str | None = None):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("LOCAL_VLM_API_KEY", "EMPTY")
        self.backend = backend
        self.model_path = model_path
        self._tf_model = None
        self._tf_processor = None

        if self.backend == "transformers":
            self._init_transformers()

    def _init_transformers(self):
        """Lazy load HuggingFace model and processor if requested."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            path = self.model_path or self.model_name
            print(f"[Qwen3-VL] Loading local model from: {path}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self._tf_processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            self._tf_model = AutoModelForVision2Seq.from_pretrained(
                path, torch_dtype=dtype, device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            ).eval()
            print("[Qwen3-VL] Local model loaded successfully.")
        except Exception as e:
            print(f"[Qwen3-VL] Warning: Failed to load local transformers model: {e}")
            print("[Qwen3-VL] Falling back to API endpoint mode.")
            self.backend = "api"

    def extract_from_image(self, image_path: str, retries: int = 3) -> dict:
        """Extract structured fields directly from an invoice image using Qwen3-VL."""
        prompt = SYSTEM_PROMPT
        if self.backend == "transformers" and self._tf_model is not None:
            return self._extract_transformers(image_path, prompt)

        # Encode image to base64
        b64_image = encode_image_to_base64(image_path)
        is_ollama = "11434" in self.endpoint

        if is_ollama:
            base = self.endpoint.replace("/v1", "").rstrip("/")
            url = f"{base}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [b64_image]
                    }
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "num_predict": 2048,
                    "temperature": 0.0
                }
            }
        else:
            url = f"{self.endpoint}/chat/completions" if not self.endpoint.endswith("/chat/completions") else self.endpoint
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                            }
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"}
            }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        for attempt in range(retries):
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=180) as response:
                    res = json.load(response)
                    msg = res.get("message", {})
                    content = msg.get("content", "")
                    # If thinking consumed tokens and content is empty, check thinking block for JSON
                    if not content or not content.strip():
                        if "thinking" in msg and "{" in msg["thinking"]:
                            content = msg["thinking"]
                    if not content and "choices" in res and len(res["choices"]) > 0:
                        content = res["choices"][0]["message"]["content"]
                    if not content:
                        content = str(res)
                    return self._clean_and_parse_json(content)
            except urllib.error.HTTPError as e:
                err_msg = e.read().decode("utf-8", errors="ignore")
                if attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                print(f"[Qwen3-VL Error] HTTP {e.code}: {err_msg}")
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                print(f"[Qwen3-VL Error] {e}")
                break

        return {k: None for k in ALL_FIELDS}

    def _extract_transformers(self, image_path: str, prompt: str) -> dict:
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        text_prompt = self._tf_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._tf_processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt")
        inputs = inputs.to(self._tf_model.device)

        with torch.no_grad():
            output_ids = self._tf_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            output_text = self._tf_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self._clean_and_parse_json(output_text)

    def _clean_and_parse_json(self, raw: str) -> dict:
        # Strip reasoning tokens / markdown fences
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)

        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {k: None for k in ALL_FIELDS}
            
            # Map canonical fields
            out = {}
            for k in ALL_FIELDS:
                out[k] = parsed.get(k)
            return out
        except Exception:
            return {k: None for k in ALL_FIELDS}


def run_extraction_benchmark(indir: str, out_pred: str, extractor: Qwen3VLExtractor, limit: int | None = None):
    # Find images and manifest
    manifest_path = os.path.join(indir, "ground_truth.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(_BUNDLE_DIR, "phase_a_output", "gt_phase0.json")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if limit:
        manifest = manifest[:limit]

    print(f"\n[Qwen3-VL Benchmark] Running extraction on {len(manifest)} invoices from {indir}")
    predictions = []

    for i, item in enumerate(manifest, 1):
        img_name = item.get("image") or f"{item.get('doc_id')}.png"
        img_path = os.path.join(indir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(_BUNDLE_DIR, "phase_a_output", "invoices_scanned", img_name)

        doc_id = item.get("doc_id") or os.path.splitext(img_name)[0]
        layout = item.get("layout", "unknown").replace("layout_", "")

        print(f"[{i}/{len(manifest)}] Extracting {doc_id} ({layout}) ...", end=" ", flush=True)
        t0 = time.time()
        fields = extractor.extract_from_image(img_path)
        dt = time.time() - t0
        print(f"done ({dt:.2f}s)")

        predictions.append({
            "doc_id": doc_id,
            "layout": layout,
            "fields": fields
        })

    with open(out_pred, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nSaved predictions to {out_pred}")
    return predictions


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL Invoice Extraction and Evaluation")
    ap.add_argument("--indir", default="phase_a_output/invoices_scanned", help="Path to invoice images")
    ap.add_argument("--out-pred", default="phase_a_output/pred_qwen3_vl.json", help="Where to save predictions")
    ap.add_argument("--gt", default="phase_a_output/gt_phase0.json", help="Path to Phase 0 ground truth")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Local VLM OpenAI-compatible endpoint URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model name / tag")
    ap.add_argument("--backend", choices=["api", "transformers"], default="api", help="Inference backend")
    ap.add_argument("--model-path", default=None, help="Local directory path to model weights")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of invoices for testing")
    ap.add_argument("--evaluate", action="store_true", default=True, help="Run evaluate.py after extraction")
    args = ap.parse_args()

    vlm = Qwen3VLExtractor(
        endpoint=args.endpoint,
        model_name=args.model,
        backend=args.backend,
        model_path=args.model_path
    )

    preds = run_extraction_benchmark(args.indir, args.out_pred, vlm, limit=args.limit)

    if args.evaluate and os.path.exists(args.gt):
        from evaluate import load_docs
        print("\n" + "=" * 60)
        print("PHASE 0 EVALUATION REPORT (Qwen3-VL-4B-Instruct)")
        print("=" * 60)
        gt_docs = load_docs(args.gt)
        pred_docs = load_docs(args.out_pred)
        report = evaluate(gt_docs, pred_docs)
        print(format_report(report))
        report_path = os.path.splitext(args.out_pred)[0] + "_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
