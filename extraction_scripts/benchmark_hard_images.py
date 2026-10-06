"""
Benchmark & Comparison Harness: Hard Invoices (images/)
Evaluates Local Qwen3-VL (Multimodal Vision) vs Groq Cloud LLM (OCR + LLM)
against ground truth across 13 challenging real-world invoice stress cases.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# Paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for p in [_ROOT_DIR, _SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from schema import ALL_FIELDS, CRITICAL_FIELDS, normalise, values_equal
from extract_qwen3_vl import Qwen3VLExtractor


def load_ground_truth(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    gt_map = {}
    for item in data:
        doc_id = item["doc_id"]
        gt_map[doc_id] = item
    return gt_map


def run_local_vlm_benchmark(images_dir, gt_map, endpoint="http://127.0.0.1:11434", model="qwen3-vl:4b"):
    print(f"\n{'='*70}\n[1/2] RUNNING LOCAL VISION LLM (Qwen3-VL-4B via Ollama)\n{'='*70}")
    extractor = Qwen3VLExtractor(endpoint=endpoint, model_name=model)
    results = {}

    for doc_id, meta in gt_map.items():
        img_name = meta["image"]
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[-] Image not found: {img_path}")
            continue

        print(f"--> Extracting {doc_id} ({meta.get('layout')}) [{img_name}] ...", end=" ", flush=True)
        t0 = time.time()
        try:
            fields = extractor.extract_from_image(img_path)
            elapsed = time.time() - t0
            print(f"DONE in {elapsed:.2f}s")
            results[doc_id] = {
                "doc_id": doc_id,
                "layout": meta.get("layout"),
                "notes": meta.get("notes"),
                "fields": fields,
                "elapsed_sec": round(elapsed, 2)
            }
        except Exception as e:
            print(f"FAILED: {e}")
            results[doc_id] = {
                "doc_id": doc_id,
                "layout": meta.get("layout"),
                "notes": meta.get("notes"),
                "fields": {},
                "error": str(e)
            }

    return results


def run_groq_benchmark(images_dir, gt_map, api_key=None):
    print(f"\n{'='*70}\n[2/2] RUNNING GROQ CLOUD LLM (Tesseract OCR + Qwen3.8-27B on Groq)\n{'='*70}")
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        print("[!] Warning: GROQ_API_KEY is not set. Cannot run live Groq cloud inference.")
        return None

    try:
        from ocr_extract import ocr, extract_llm
    except ImportError as e:
        print(f"[!] Warning: Cannot import ocr_extract: {e}")
        return None

    results = {}
    for doc_id, meta in gt_map.items():
        img_name = meta["image"]
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            continue

        print(f"--> [Groq] OCR + Extracting {doc_id} [{img_name}] ...", end=" ", flush=True)
        t0 = time.time()
        try:
            ocr_text = ocr(img_path)
            fields = extract_llm(ocr_text, provider="groq")
            elapsed = time.time() - t0
            print(f"DONE in {elapsed:.2f}s")
            results[doc_id] = {
                "doc_id": doc_id,
                "layout": meta.get("layout"),
                "fields": fields,
                "elapsed_sec": round(elapsed, 2)
            }
        except Exception as e:
            print(f"FAILED: {e}")
            results[doc_id] = {
                "doc_id": doc_id,
                "layout": meta.get("layout"),
                "fields": {},
                "error": str(e)
            }

    return results


def score_predictions(predictions, gt_map):
    doc_results = []
    field_stats = {}

    for doc_id, pred_info in predictions.items():
        if doc_id not in gt_map:
            continue
        gt_info = gt_map[doc_id]
        gt_fields = gt_info["fields"]
        pred_fields = pred_info.get("fields", {})

        field_scores = {}
        correct_count = 0
        total_eval_fields = 0
        crit_correct = True

        for fname, fval in gt_fields.items():
            total_eval_fields += 1
            pval = pred_fields.get(fname)
            kind = ALL_FIELDS.get(fname, "string")
            norm_g = normalise(kind, fval)
            norm_p = normalise(kind, pval)

            if norm_g is None and norm_p is None:
                is_correct = True
            elif norm_g is not None and norm_p is not None:
                is_correct = values_equal(kind, norm_g, norm_p)
            else:
                is_correct = False
            if is_correct:
                correct_count += 1
                status = "CORRECT"
            else:
                status = "WRONG" if norm_p is not None else "MISSING"

            if fname in CRITICAL_FIELDS and not is_correct:
                crit_correct = False

            field_scores[fname] = {
                "gt": fval,
                "pred": pval,
                "status": status
            }

            if fname not in field_stats:
                field_stats[fname] = {"total": 0, "correct": 0}
            field_stats[fname]["total"] += 1
            if is_correct:
                field_stats[fname]["correct"] += 1

        accuracy = correct_count / total_eval_fields if total_eval_fields > 0 else 0.0
        doc_results.append({
            "doc_id": doc_id,
            "layout": gt_info.get("layout"),
            "notes": gt_info.get("notes"),
            "total_fields": total_eval_fields,
            "correct_fields": correct_count,
            "accuracy": round(accuracy * 100, 1),
            "critical_all_correct": crit_correct,
            "field_scores": field_scores,
            "elapsed_sec": pred_info.get("elapsed_sec")
        })

    return {
        "documents": doc_results,
        "mean_accuracy": round(sum(d["accuracy"] for d in doc_results) / len(doc_results), 1) if doc_results else 0.0,
        "critical_all_correct_rate": round(sum(1 for d in doc_results if d["critical_all_correct"]) / len(doc_results) * 100, 1) if doc_results else 0.0,
        "field_stats": {
            k: {
                "accuracy": round(v["correct"] / v["total"] * 100, 1),
                "correct": v["correct"],
                "total": v["total"]
            }
            for k, v in field_stats.items()
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark hard invoices on Local VLM and Groq")
    parser.add_argument("--images-dir", default=os.path.join(_ROOT_DIR, "images"))
    parser.add_argument("--gt", default=os.path.join(_ROOT_DIR, "ground_truth.json"))
    parser.add_argument("--out-dir", default=os.path.join(_ROOT_DIR, "phase_a_output"))
    parser.add_argument("--skip-local", action="store_true", help="Skip local Qwen3-VL extraction")
    parser.add_argument("--skip-groq", action="store_true", help="Skip Groq cloud extraction")
    parser.add_argument("--groq-key", default=None, help="Explicit Groq API key")
    args = parser.parse_args()

    gt_map = load_ground_truth(args.gt)
    os.makedirs(args.out_dir, exist_ok=True)

    summary_comparison = {
        "ground_truth_total": len(gt_map),
        "local_qwen3_vl": None,
        "groq_cloud": None
    }

    # 1. Run / Load Local Qwen3-VL
    local_pred_file = os.path.join(args.out_dir, "pred_hard_qwen3_vl.json")
    if not args.skip_local:
        local_preds = run_local_vlm_benchmark(args.images_dir, gt_map)
        with open(local_pred_file, "w", encoding="utf-8") as f:
            json.dump(local_preds, f, indent=2)
    elif os.path.exists(local_pred_file):
        with open(local_pred_file, "r", encoding="utf-8") as f:
            local_preds = json.load(f)
    else:
        local_preds = None

    if local_preds:
        local_eval = score_predictions(local_preds, gt_map)
        local_eval_file = os.path.join(args.out_dir, "eval_hard_qwen3_vl.json")
        with open(local_eval_file, "w", encoding="utf-8") as f:
            json.dump(local_eval, f, indent=2)
        summary_comparison["local_qwen3_vl"] = {
            "mean_accuracy": local_eval["mean_accuracy"],
            "critical_all_correct_rate": local_eval["critical_all_correct_rate"],
            "field_stats": local_eval["field_stats"]
        }
        print(f"\n[Local Qwen3-VL Score] Mean Field Acc: {local_eval['mean_accuracy']}% | Critical All Correct: {local_eval['critical_all_correct_rate']}%")

    # 2. Run / Load Groq
    groq_pred_file = os.path.join(args.out_dir, "pred_hard_groq.json")
    if not args.skip_groq:
        groq_preds = run_groq_benchmark(args.images_dir, gt_map, api_key=args.groq_key)
        if groq_preds:
            with open(groq_pred_file, "w", encoding="utf-8") as f:
                json.dump(groq_preds, f, indent=2)
    elif os.path.exists(groq_pred_file):
        with open(groq_pred_file, "r", encoding="utf-8") as f:
            groq_preds = json.load(f)
    else:
        groq_preds = None

    if groq_preds:
        groq_eval = score_predictions(groq_preds, gt_map)
        groq_eval_file = os.path.join(args.out_dir, "eval_hard_groq.json")
        with open(groq_eval_file, "w", encoding="utf-8") as f:
            json.dump(groq_eval, f, indent=2)
        summary_comparison["groq_cloud"] = {
            "mean_accuracy": groq_eval["mean_accuracy"],
            "critical_all_correct_rate": groq_eval["critical_all_correct_rate"],
            "field_stats": groq_eval["field_stats"]
        }
        print(f"\n[Groq Cloud Score] Mean Field Acc: {groq_eval['mean_accuracy']}% | Critical All Correct: {groq_eval['critical_all_correct_rate']}%")

    comp_file = os.path.join(args.out_dir, "hard_invoices_comparison.json")
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(summary_comparison, f, indent=2)
    print(f"\nBenchmark completed. Summary saved to {comp_file}")


if __name__ == "__main__":
    main()
