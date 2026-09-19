# Presentation content — Freight Cost Prediction & Invoice Fraud Detection

Format per slide: **On-slide content** (exact text/bullets to place) + **Suggested visual**.
A separate section at the end covers what to know but *not* put on slides.

---

## Slide 1 — Title

**On-slide content:**
- Title: "Indian B2B Road-Freight Cost Prediction & Invoice Fraud Detection"
- Subtitle: "From mini-project to a GenAI-augmented major project"
- Your name, roll number, branch/specialization, guide's name, date

**Suggested visual:** none needed — keep it clean. Optionally a single freight/logistics photo as background, low opacity.

---

## Slide 2 — Problem statement

**On-slide content:**
- Manual freight cost estimation in B2B road-freight is inconsistent — no standard "should-cost" benchmark per shipment
- Invoice billing is opaque: carriers can pad detention time, inflate tolls, or overbill weight-based charges, and manual audit doesn't scale
- Goal: predict a fair should-cost per shipment, and automatically flag invoices likely to need manual review

**Suggested visual:** none, or a simple 2-icon layout (truck / invoice-with-flag)

---

## Slide 3 — Objectives

**On-slide content:**
- **Phase 1 (done):** Build a should-cost regression model and an invoice-fraud classification model on a realistic Indian freight dataset
- **Phase 2 (proposed):** Extend into a document-aware, explainable, continuously-updated system
- Two connected problems, one shared feature pipeline

**Suggested visual:** none — keep as clean bullet slide

---

## Slide 4 — Data pipeline overview

**On-slide content:**
- 18 real Indian hub cities, 25 carrier vendors, 3 truck types — reference master data
- 60,000 orders / 60,120 invoices generated end-to-end
- Real data sources integrated: Open-Meteo weather (20,286 real hub-days across 18 hubs), PPAC/IOC diesel prices (14,248 real price rows, 13 states)
- 11-phase build: hubs → cargo → routes → weather → fuel → cost label → anomaly injection → Model 1 → invoices → vendor risk score → output

**Suggested visual:** the pipeline diagram — a left-to-right flowchart of the 11 phases (4-5 boxes per row, 2-3 rows). Keep box labels short: "Hubs & vendors", "Cargo physicals", "Routes", "Weather + fuel", "Cost label", "Anomaly injection", "Model 1 (OOF)", "Invoices", "Vendor risk score".

---

## Slide 5 — Model 1: Freight cost predictor

**On-slide content:**
- Type: Regression (XGBoost)
- Predicts: `base_freight_cost` — the fair should-cost for a shipment
- Features: distance, ideal transit days, quoted days, billable weight, fuel price, weather severity, product category, truck type
- Result: **MAE ≈ ₹510-513, MAPE ≈ 1.65%** (5-fold out-of-fold, leakage-safe)

**Suggested visual:** a simple metric card layout — two large numbers (MAE, MAPE) side by side, or a small bar chart of feature importance if you have it handy

---

## Slide 6 — Model 2: Invoice risk flagger

**On-slide content:**
- Type: Binary classification (XGBoost, class-weighted for ~5.7% fraud rate)
- Predicts: whether an invoice needs manual review
- Features: billed amount, predicted cost, cost mismatch, delay days, adverse weather days, vendor padding ratio, vendor historical risk score
- Three modeled fraud types: detention padding, toll inflation, weight discrepancy
- Result: **PR-AUC 0.84 · Recall 0.887 · Precision 0.494**
- Per-type recall: detention padding 100% · toll inflation 90% · weight discrepancy 77%

**Suggested visual:** a small horizontal bar chart of the three per-type recall values — makes the "easiest vs hardest fraud to catch" point visually obvious

---

## Slide 7 — Why this is realistic, not toy data

**On-slide content:**
- Real weather (Open-Meteo Archive API, no key) and real diesel prices (PPAC/IOC, no key) — not fully synthetic
- Intentional real-world data-quality issues deliberately left in: missing values, unit errors, duplicate rows, mixed date formats — then cleaned in the training pipeline, not the raw data
- Fraud magnitudes deliberately overlap legitimate billing variance — model must use context, not a simple cost threshold

**Suggested visual:** none — this is a credibility slide, keep it text-only

---

## Slide 8 — Current limitations (why this needs to grow)

**On-slide content:**
- Route distances are approximated (haversine × circuity factor), not real road distance
- No live inference path — models are trained/evaluated in batch, not serving new orders
- Assumes clean structured invoice data — real invoices arrive as scans/PDFs
- No interface for non-technical stakeholders to query results
- Fuel and weather inputs go stale without a refresh process

**Suggested visual:** none, or a simple "gap" icon list — this is the pivot slide, let it be direct

---

## Slide 9 — Proposed extension: GenAI layer

**On-slide content:**
- **Invoice OCR + LLM extraction:** scanned invoice image → structured fields → auto-feeds Model 2, no manual data entry
- **Natural-language query agent:** ask questions like "which vendor has the highest risk this quarter?" — LLM writes and runs the query, answer is grounded in real data
- **LLM-generated audit explanations:** flagged invoices get a plain-English rationale, not just a probability score

**Suggested visual:** the architecture diagram (already built) — invoice/question inputs → OCR / query agent → existing pipeline → risk+explanation / answer → chat-dashboard UI

---

## Slide 10 — Proposed extension: keeping the models current

**On-slide content:**
- Fuel prices: scheduled refresh, daily/weekly, from the real PPAC source
- Weather for new quotes: real-time forecast API call per order (not the historical archive used for training)
- Vendor risk score: updated incrementally, per invoice, not batched
- Model retraining: quarterly, or triggered when prediction error drifts from baseline

**Suggested visual:** a small table — columns "Component / Trigger / Cadence", 4 rows matching the four bullets above

---

## Slide 11 — Target system architecture

**On-slide content:**
- One combined diagram showing: data refresh jobs → existing models → GenAI layer → single UI
- Caption: "Existing pipeline stays unchanged — new components sit around it"

**Suggested visual:** a single architecture diagram combining slide 9's GenAI flow with slide 10's refresh jobs feeding into the "Existing pipeline" box. This is the most important visual in the deck — give it a full slide, minimal extra text.

---

## Slide 12 — Implementation plan

**On-slide content (as a phase table or timeline):**
1. Invoice OCR pipeline — synthetic invoice image generation + OCR + LLM field-structuring
2. Natural-language query agent — schema-constrained text-to-SQL over the dataset
3. LLM explanation layer for flagged invoices
4. Live-data scheduler — fuel refresh job, forecast-API integration, incremental vendor scoring
5. Drift check + retraining trigger
6. Integration — FastAPI backend + Streamlit UI tying everything together
7. Evaluation and final report

**Suggested visual:** a simple Gantt-style table — rows = phases above, columns = weeks/months of your semester, shaded cells showing planned duration. Mark Phase 1 as highest priority / first.

---

## Slide 13 — Evaluation plan

**On-slide content:**
- **OCR extraction benchmark:** Rules (81.4%) vs Groq LLM (93.9% overall, 95.8% on multi-column layouts)
- **Two-Number Confidence Gate (n=300):**
  - Auto-processing fraction: 62.0% (186/300 passed); 38.0% quarantined to manual data entry queue
  - Model 2 on auto-processed fraction: 89.2% accuracy, 100% recall (6/6 true fraud cases caught), 0 false alarms from OCR noise
- Query-agent success rate on a held-out set of test questions
- Existing Model 1 / Model 2 metrics as the baseline to preserve (MAE ₹510, 1.66% MAPE, 0.840 PR-AUC, 0.887 recall)

**Suggested visual:** none — bullet list is fine here

---

## Slide 14 — Tech stack

**On-slide content (grouped, not just a flat list):**
- **Modeling:** Python, pandas, XGBoost, scikit-learn, statsmodels
- **New — GenAI:** OCR (Tesseract/EasyOCR or vision-LLM), LLM API (free tier), prompt-based text-to-SQL
- **New — serving:** FastAPI, Streamlit, SQLite/DuckDB
- **Data sources:** Open-Meteo (weather), PPAC/IOC (diesel)

**Suggested visual:** simple 4-box grid layout, one box per group above

---

## Slide 15 — Conclusion & ask

**On-slide content:**
- Solid, verified foundation: real-data pipeline, two working models with strong, honestly-reported metrics
- Clear, phased roadmap to a GenAI-augmented, self-updating system
- Ask: feedback on scope, sign-off to proceed, guidance on LLM-cost constraints for a student budget

**Suggested visual:** none — closing slide, keep it short

---
---

# Speaker notes — do NOT put these on slides
### Know these cold; answer only if asked

**On why the metrics look "too good"**
- Model 1's MAPE (1.65%) is low partly *because* `base_freight_cost` is generated from a near-deterministic formula (fuel + tolls + driver allowance) plus small noise — the model is essentially learning to invert a known formula. If asked "isn't this cheating," be upfront: yes, on real vendor-negotiated costs (which include market/negotiation factors the formula doesn't capture), you'd expect meaningfully higher error, and re-calibration against real quotes would be a necessary next step before real deployment.
- Model 2's precision (0.49) is deliberately not higher — the evaluation favors recall because missing real fraud is costlier than a human double-checking a legitimate invoice. If pushed, explain the threshold (0.5) can be tuned up for higher precision if the audit team's review capacity is limited.

**On using synthetic data at all**
- No real freight company would share actual cost/invoice data (commercial confidentiality) at this stage — so the dataset is calibrated to real Indian benchmarks (real weather, real diesel prices) rather than fully invented, to get statistically realistic behavior while still having usable fraud labels. If asked about generalization to real data: the three fraud mechanisms modeled (detention padding, toll inflation, weight discrepancy) are documented real-world patterns in Indian B2B freight, not invented ones — but validating against even a small real labeled sample, if an industry contact becomes available, is honest future work.

**On modeling choices**
- XGBoost over deep learning: tabular, heterogeneous features at 60k rows — gradient-boosted trees are the standard choice here and outperform deep nets at this scale; interpretable, fast, well-understood failure modes.
- SARIMA over Prophet/LSTM for forecasting: short, low-frequency series (state-level daily diesel, weekly weather aggregates) — classical statistical models need far less data and are more interpretable than deep sequence models here.
- Why OSRM (real road distance) was skipped: needs self-hosted infrastructure (Docker + India OSM extract), judged too heavy for a 153-route cache relative to the accuracy gain over haversine × circuity; flag it as a known limitation and a candidate future improvement, not something you're unaware of.

**On the GenAI layer specifically**
- LLM hallucination risk: contained by design — the LLM only writes/structures text and writes SQL queries; it never computes the actual numbers itself. Query answers come from real SQL execution against the real dataset; risk scores come from the trained models, not the LLM.
- Cost: plan to use a free-tier LLM API (or a small local model) to keep this at zero cost for a student project — mention this proactively if asked how you'll afford API calls at scale.
- Why not just use an off-the-shelf OCR SaaS (AWS Textract etc.): raw OCR is commodity technology — the contribution is connecting extraction to the fraud model, the explanation layer, and the incremental vendor-scoring loop, tailored to this domain's data and fraud patterns, not the OCR step itself.
**On the Confidence Gate and Two-Number Reporting (Crucial Defense):**
- Why not report a single end-to-end number: Feeding un-validated OCR output into Model 2 causes mis-extracted amounts (e.g. ₹15,700 read as ₹15.7M due to comma misinterpretation) to generate confident false fraud accusations against honest carriers.
- The 4-tier validation gate catches these early (missing critical fields, unmapped order/vendor IDs, line-item arithmetic drift, physical plausibility bounds).
- In our n=300 benchmark, 186/300 passed the gate; on that auto-processed cohort, Model 2 achieved 89.2% accuracy and caught 100% (6/6) of true fraud cases. The 114 failing invoices are safely routed to human review rather than being scored on corrupted data.
- Sample size defense: Why n=300? In an n=30 pilot, only 1 positive fraud case existed in the cohort; evaluating at n=300 provides multiple positive ground-truth cases, making the 100% recall metric statistically credible.

**On production/drift monitoring**
- How you'd actually detect drift: monitor rolling MAE of Model 1's predictions against actual billed amounts, and rolling PR-AUC of Model 2 on a recent window of invoices; alert when either degrades past a set threshold. You don't need this built for the presentation — just be able to describe the mechanism if asked.

**General presentation reminders**
- Lead with what's *done and verified* (slides 4-7) before the *plan* (slides 8-15) — your guide will trust the plan more once they see the foundation is real, not aspirational.
- Have the actual metric numbers memorized cold (₹510-513 MAE, 1.65% MAPE, 0.84 PR-AUC, 0.887 recall) — don't fumble looking them up if asked directly.
- If your guide asks "what if this fails" for any GenAI component, the honest answer is: the existing Model 1/Model 2 pipeline keeps working standalone regardless — the GenAI layer is additive, not a replacement, so there's no single point of failure for your core grade-bearing work.
