"""
PHASE C -- Plain-English audit explanations for flagged/approved invoices.

Takes the SAME numbers the confidence gate + Model 2 already computed and
turns them into a short, readable paragraph. The generator (rules or LLM)
is never given the freedom to invent a number: every figure in the prompt
is one that was already computed deterministically upstream, and the
instructions explicitly forbid introducing new ones. This is what makes
"zero-hallucination" an enforced constraint here, not just a claim.

Two backends, same interface as ocr_extract.py's extractor pattern:
  --backend rules  -- offline, free, template-based (default)
  --backend llm    -- reads better, needs an LLM key (see ocr_extract.py's
                       provider table; same PROVIDERS dict is reused)

Usage (standalone):
  python extraction_scripts/audit_explanation.py --checkpoint phase_a_output/bench300_checkpoint_rules.jsonl --n 5
  python extraction_scripts/audit_explanation.py --checkpoint phase_a_output/bench300_checkpoint_rules.jsonl --n 5 --backend llm
"""
import argparse
import json
import os
import sys

# Support imports when run from repo root or extraction_scripts directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BUNDLE_DIR = os.path.dirname(_SCRIPT_DIR)
for p in [_SCRIPT_DIR, os.path.join(_BUNDLE_DIR, "extraction_scripts"), os.path.join(_BUNDLE_DIR, "phase_a"), "../phase_a"]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from ocr_extract import PROVIDERS  # reuse the same free-tier provider table


def explain_rules(rec):
    """Template-based explanation. No network, no key, fully deterministic --
    the honest floor every LLM explanation should be measured against."""
    if not rec.get("passed", False):
        reason_str = rec.get("reason", "unknown failure").replace("_", " ")
        return (f"Invoice routed to manual data entry: {reason_str}. "
                f"No fraud score was computed -- an unvalidated extraction was never "
                f"passed to Model 2.")

    f = rec["features"]
    verdict = "FLAGGED FOR AUDIT" if rec["predicted_flag"] else "APPROVED"
    pred_cost = f["model_a_predicted_cost"]
    p10 = f.get("cost_interval_lower", round(pred_cost * 0.95, 2))
    p90 = f.get("cost_interval_upper", round(pred_cost * 1.05, 2))
    billed = f["actual_billed_amount"]

    lines = [f"Verdict: {verdict} (risk probability {rec.get('proba', 0.0):.1%})."]

    if billed > p90:
        excess = billed - p90
        lines.append(f"Billed amount of ₹{billed:,.0f} breaches the upper uncertainty bound of ₹{p90:,.0f} by ₹{excess:,.0f} (+{excess/p90:.1%}). Model 1 fair should-cost estimate is ₹{pred_cost:,.0f} (80% credible band: ₹{p10:,.0f} – ₹{p90:,.0f}).")
    elif billed < p10:
        discount = p10 - billed
        lines.append(f"Billed amount of ₹{billed:,.0f} is ₹{discount:,.0f} below the expected lower bound of ₹{p10:,.0f} (fair estimate: ₹{pred_cost:,.0f} [₹{p10:,.0f} – ₹{p90:,.0f}]).")
    else:
        lines.append(f"Billed amount of ₹{billed:,.0f} is within the expected 80% credible band of ₹{p10:,.0f} – ₹{p90:,.0f} (fair should-cost estimate: ₹{pred_cost:,.0f}).")

    drivers = rec.get("top_drivers") or f.get("top_drivers")
    if drivers:
        driver_strs = [f"{d['feature'].replace('_', ' ').title()} ({'+' if d['impact'] > 0 else ''}₹{d['impact']:,.0f})" for d in drivers]
        lines.append(f"Top cost drivers: {', '.join(driver_strs)}.")
    if f["commercial_delay_days"] > 0.5:
        lines.append(f"Transit ran {f['commercial_delay_days']:.1f} days beyond the "
                     f"carrier's own quoted time.")
    if f["vendor_historical_risk_score"] > 0.1:
        lines.append(f"This vendor's historical risk score is "
                     f"{f['vendor_historical_risk_score']:.2f} (Laplace-smoothed, "
                     f"causal -- based only on their invoices prior to this one).")
    if f["adverse_weather_days"] > 0:
        lines.append(f"{f['adverse_weather_days']:.0f} day(s) of adverse weather were "
                     f"recorded on this route, which can legitimately explain some delay.")
    return " ".join(lines)


LLM_EXPLAIN_PROMPT = """You are writing a one-paragraph plain-English audit note for a \
logistics manager reviewing a freight invoice. You are given the EXACT numbers a \
fraud-detection model already computed. Do not invent, estimate, or recompute any \
number -- only use the ones given. Do not add a verdict different from the one given.

Verdict: {verdict}
Risk probability: {proba:.1%}
Billed amount: ₹{billed:,.0f}
Model-predicted fair cost: ₹{predicted:,.0f}
Cost mismatch (billed minus predicted): ₹{mismatch:,.0f}
Commercial delay (actual days beyond quoted): {commercial_delay:.2f} days
True delay (actual days beyond physically ideal time): {true_delay:.2f} days
Vendor historical risk score (0-1, causal/Laplace-smoothed): {vendor_risk:.3f}
Adverse weather days on this route: {weather:.0f}

Write 2-4 sentences a non-technical manager can act on. Plain language, no jargon, \
no markdown, no bullet points."""


def explain_llm(rec, provider="groq"):
    import os
    import re
    import urllib.request

    if not rec.get("passed", False):
        return explain_rules(rec)  # nothing for the LLM to add here

    f = rec["features"]
    cfg = PROVIDERS[provider]
    key = os.environ.get(cfg["key_env"])
    if not key:
        raise SystemExit(f"Set {cfg['key_env']} to use --backend llm --provider {provider}")

    prompt = LLM_EXPLAIN_PROMPT.format(
        verdict="FLAGGED FOR AUDIT" if rec["predicted_flag"] else "APPROVED",
        proba=rec.get("proba", 0.0), billed=f["actual_billed_amount"],
        predicted=f["model_a_predicted_cost"], mismatch=f["cost_mismatch"],
        commercial_delay=f["commercial_delay_days"], true_delay=f["true_delay_days"],
        vendor_risk=f["vendor_historical_risk_score"], weather=f["adverse_weather_days"])

    body = json.dumps({"model": cfg["model"], "max_tokens": 300, "temperature": 0.3,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(cfg["url"], data=body,
                                 headers={"content-type": "application/json",
                                          "authorization": f"Bearer {key}",
                                          "user-agent": "FreightAudit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.load(r)
    return resp["choices"][0]["message"]["content"].strip()


if __name__ == "__main__":
    _def_ckpt = os.path.join(_BUNDLE_DIR, "phase_a_output", "bench300_checkpoint_rules.jsonl")
    ap = argparse.ArgumentParser(description="Generate plain-English audit explanations")
    ap.add_argument("--checkpoint", default=_def_ckpt,
                    help="a bench300-style JSONL from confidence_gate_benchmark.py")
    ap.add_argument("--n", type=int, default=5, help="Number of records to explain")
    ap.add_argument("--backend", choices=["rules", "llm"], default="rules", help="Explanation backend")
    ap.add_argument("--provider", default="groq", help="LLM provider (default: groq)")
    a = ap.parse_args()

    if not os.path.exists(a.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint file not found: {a.checkpoint}. Run confidence_gate_benchmark.py first."
        )

    records = [json.loads(l) for l in open(a.checkpoint) if l.strip()][:a.n]
    for rec in records:
        img = rec.get("image", "?")
        text = (explain_rules(rec) if a.backend == "rules" else explain_llm(rec, a.provider))
        print(f"--- {img} ---\n{text}\n")
