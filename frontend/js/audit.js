/**
 * audit.js - Three-step audit flow.
 *   1. Upload  -> POST /api/extract   (cloud vision model reads the invoice)
 *   2. Review  -> user confirms AI-read values and types any missing ones;
 *                 POST /api/check re-validates as they edit
 *   3. Verdict -> POST /api/score     (gate + Model 1 + Model 2)
 * The score button stays disabled until every required field has a value,
 * so the models never run on incomplete data.
 */

const FIELD_META = {
  invoice_id:   { label: "Invoice number",        placeholder: "INV000123", type: "text" },
  order_id:     { label: "Order / LR number",     placeholder: "ORD000123", type: "text" },
  vendor_id:    { label: "Carrier (vendor) ID",   placeholder: "VEN001",    type: "text" },
  invoice_date: { label: "Invoice date",          placeholder: "DD-MM-YYYY", type: "date" },
  actual_days:  { label: "Actual transit days",   placeholder: "e.g. 1.5",  type: "number", step: "0.01" },
  freight_base: { label: "Base freight (₹)",      placeholder: "e.g. 12500", type: "number", step: "0.01" },
  detention:    { label: "Detention (₹)",         placeholder: "0 if none", type: "number", step: "0.01" },
  toll:         { label: "Toll (₹)",              placeholder: "e.g. 1035", type: "number", step: "0.01" },
  total:        { label: "Invoice total (₹)",     placeholder: "Final payable", type: "number", step: "0.01" },
  origin:       { label: "Origin",                placeholder: "City", type: "text" },
  destination:  { label: "Destination",           placeholder: "City", type: "text" },
  truck_type:   { label: "Truck type",            placeholder: "e.g. 10-wheeler", type: "text" },
  weight_kg:    { label: "Weight (kg)",           placeholder: "e.g. 9800", type: "number", step: "1" },
  vendor_gstin: { label: "Carrier GSTIN",         placeholder: "15 characters", type: "text" },
};

const REASON_MAPPINGS = {
  missing_critical_field: "One or more required fields are empty.",
  order_id_not_found: "This Order ID does not exist in the order master.",
  vendor_id_not_found: "This Vendor ID is not a registered carrier.",
  unparseable_amount_or_days: "The total or transit days could not be read as numbers.",
  arithmetic_mismatch: "Base freight + detention + toll does not add up to the total.",
  amount_out_of_plausible_range: "The total is outside ₹500 to ₹2,50,000.",
  days_out_of_plausible_range: "Transit days are outside 0.2 to 30 days.",
};

function formatCurrency(val) {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return "₹" + Number(val).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// DD-MM-YYYY <-> YYYY-MM-DD for <input type="date">
const toIsoDate = v => { const m = /^(\d{2})-(\d{2})-(\d{4})$/.exec(v || ""); return m ? `${m[3]}-${m[2]}-${m[1]}` : (v || ""); };
const fromIsoDate = v => { const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(v || ""); return m ? `${m[3]}-${m[2]}-${m[1]}` : v; };

const auditPage = {
  selectedFile: null,
  imageUrl: null,
  extraction: null,     // last /api/extract response
  aiFields: {},         // values exactly as the model read them
  issues: {},
  checkTimer: null,

  init() {
    const $ = id => document.getElementById(id);
    Object.assign(this, {
      dropzone: $("dropzone"), fileInput: $("file-input"), filePreview: $("file-preview"),
      fileThumb: $("file-thumb"), fileName: $("file-name"), fileSize: $("file-size"),
      removeBtn: $("file-remove-btn"), extractBtn: $("extract-btn"), extractSpinner: $("extract-spinner"),
      extractBtnText: $("extract-btn-text"), extractorSelect: $("extractor-select"), extractorHint: $("extractor-hint"),
      demoSelect: $("demo-select"), errorBanner: $("audit-error-banner"),
      uploadCard: $("upload-card"), reviewCard: $("review-card"), resultsCard: $("audit-results-card"),
      fieldsContainer: $("fields-container"), form: $("fields-form"), scoreBtn: $("score-btn"),
      scoreSpinner: $("score-spinner"), scoreBtnText: $("score-btn-text"), reviewBanner: $("review-banner"),
      orderSummary: $("order-summary"),
    });
    this.bindEvents();
    this.loadExtractors();
    this.loadDemos();
  },

  bindEvents() {
    this.dropzone.addEventListener("click", () => this.fileInput.click());
    this.dropzone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.fileInput.click(); } });
    this.dropzone.addEventListener("dragover", e => { e.preventDefault(); this.dropzone.classList.add("dragover"); });
    this.dropzone.addEventListener("dragleave", () => this.dropzone.classList.remove("dragover"));
    this.dropzone.addEventListener("drop", e => {
      e.preventDefault();
      this.dropzone.classList.remove("dragover");
      if (e.dataTransfer.files?.[0]) this.setFile(e.dataTransfer.files[0]);
    });
    this.fileInput.addEventListener("change", () => { if (this.fileInput.files?.[0]) this.setFile(this.fileInput.files[0]); });
    this.removeBtn.addEventListener("click", e => { e.stopPropagation(); this.clearFile(); });
    this.extractBtn.addEventListener("click", () => this.runExtract());
    this.extractorSelect.addEventListener("change", () => this.updateExtractorHint());
    this.demoSelect.addEventListener("change", () => this.loadDemo(this.demoSelect.value));
    this.form.addEventListener("submit", e => { e.preventDefault(); this.runScore(); });
    this.form.addEventListener("input", e => this.onFieldInput(e));
    document.getElementById("back-btn").addEventListener("click", () => this.reset());
    document.getElementById("new-audit-btn").addEventListener("click", () => this.reset());
    document.getElementById("edit-again-btn").addEventListener("click", () => this.showStep(2));
  },

  // ---------------------------------------------------------------- setup
  async loadExtractors() {
    try {
      const { extractors } = await api.extractors();
      this.extractors = extractors;
      extractors.forEach(x => {
        const opt = new Option(`${x.label} · ${x.model}${x.configured ? "" : " (not configured)"}`, x.id);
        opt.disabled = !x.configured;
        this.extractorSelect.add(opt);
      });
      this.updateExtractorHint();
    } catch (_) {
      this.extractorHint.textContent = "Could not load the model list. The backend may be waking up; try again in a minute.";
    }
  },

  updateExtractorHint() {
    const list = this.extractors || [];
    const ready = list.filter(x => x.configured);
    if (!ready.length) {
      this.extractorHint.innerHTML = "<strong>No extraction model is configured on the backend.</strong> Set <code>GEMINI_API_KEY</code> (recommended) or <code>GROQ_API_KEY</code> on Render. You can still enter fields by hand after uploading.";
      return;
    }
    const sel = this.extractorSelect.value;
    const x = sel === "auto" ? ready[0] : list.find(e => e.id === sel);
    this.extractorHint.textContent = sel === "auto"
      ? `Will use ${x.label} (${x.model}) and fall back to the next configured model if it fails.`
      : `${x.label} · ${x.model}`;
  },

  async loadDemos() {
    try {
      const res = await fetch("demo/manifest.json", { cache: "no-store" });
      if (!res.ok) return;
      (await res.json()).forEach(d => this.demoSelect.add(new Option(`${d.image.replace(".png", "")}: ${d.label}`, d.image)));
    } catch (_) { /* demos are optional */ }
  },

  async loadDemo(name) {
    if (!name) return;
    try {
      const resp = await fetch(`demo/${name}`);
      if (!resp.ok) throw new Error();
      const blob = await resp.blob();
      this.setFile(new File([blob], name, { type: blob.type || "image/png" }));
    } catch (_) {
      this.showError("Could not load the demo invoice.");
    } finally {
      this.demoSelect.value = "";
    }
  },

  setFile(file) {
    if (!/^image\/(png|jpe?g|webp)$/.test(file.type)) {
      this.showError("Please upload a PNG, JPG or WEBP image of the invoice.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      this.showError("That image is larger than 10 MB. Please upload a smaller scan or photo.");
      return;
    }
    this.selectedFile = file;
    if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
    this.imageUrl = URL.createObjectURL(file);
    this.fileThumb.src = this.imageUrl;
    this.fileName.textContent = file.name;
    this.fileSize.textContent = `(${(file.size / 1024).toFixed(0)} KB)`;
    this.filePreview.classList.remove("hidden");
    this.extractBtn.disabled = false;
    this.hideError();
  },

  clearFile() {
    this.selectedFile = null;
    this.fileInput.value = "";
    this.filePreview.classList.add("hidden");
    this.extractBtn.disabled = true;
  },

  reset() {
    this.clearFile();
    this.extraction = null;
    this.hideError();
    this.showStep(1);
  },

  showStep(n) {
    this.uploadCard.classList.toggle("hidden", n !== 1);
    this.reviewCard.classList.toggle("hidden", n !== 2);
    this.resultsCard.classList.toggle("hidden", n !== 3);
    document.querySelectorAll("#audit-stepper .step").forEach(s => {
      const k = Number(s.dataset.step);
      s.classList.toggle("active", k === n);
      s.classList.toggle("done", k < n);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  },

  // -------------------------------------------------------------- step 1
  async runExtract() {
    if (!this.selectedFile) return;
    this.hideError();
    this.setBusy(this.extractBtn, this.extractSpinner, this.extractBtnText, true, "Reading invoice…");
    try {
      const res = await api.extract(this.selectedFile, this.extractorSelect.value);
      this.openReview(res);
    } catch (err) {
      // Extraction failed entirely: still let the user type the fields in.
      this.showError(`The model could not read this invoice (${err.message}). Enter the fields by hand below.`);
      this.openReview({
        fields: {}, missing_fields: Object.keys(FIELD_META).slice(0, 9), required_fields: Object.keys(FIELD_META).slice(0, 9),
        field_issues: {}, extractor_used: "manual", extractor_model: null, fallback_errors: [], filename: this.selectedFile.name,
      }, true);
    } finally {
      this.setBusy(this.extractBtn, this.extractSpinner, this.extractBtnText, false, "Read Invoice");
    }
  },

  // -------------------------------------------------------------- step 2
  openReview(res, failed = false) {
    this.extraction = res;
    this.aiFields = { ...res.fields };
    this.issues = res.field_issues || {};
    const required = res.required_fields;
    const optional = Object.keys(FIELD_META).filter(k => !required.includes(k));

    const badge = document.getElementById("extractor-used-badge");
    badge.textContent = failed ? "Manual entry" : `Read by ${res.extractor_used} · ${res.extractor_model}`;
    if (res.fallback_errors?.length) {
      badge.title = "Fell back after: " + res.fallback_errors.map(e => `${e.provider}: ${e.error}`).join("\n");
    }

    document.getElementById("review-image").src = this.imageUrl;
    document.getElementById("review-image-link").href = this.imageUrl;

    const section = (title, keys, isReq) => `
      <fieldset class="field-section">
        <legend>${title}</legend>
        <div class="field-grid">${keys.map(k => this.fieldHtml(k, isReq)).join("")}</div>
      </fieldset>`;
    this.fieldsContainer.innerHTML =
      section("Required to run the models", required, true) +
      `<details class="optional-fields"><summary>Other invoice details (optional)</summary>${section("", optional, false)}</details>`;

    this.renderOrderSummary(res.order_summary);
    this.refreshFieldStates();
    this.showStep(2);
    const firstMissing = this.fieldsContainer.querySelector(".field.missing input");
    if (firstMissing) setTimeout(() => firstMissing.focus({ preventScroll: true }), 400);
  },

  fieldHtml(key, required) {
    const m = FIELD_META[key];
    let v = this.aiFields[key];
    if (v === null || v === undefined) v = "";
    if (m.type === "date") v = toIsoDate(v);
    return `
      <div class="field" data-field="${key}">
        <label for="f-${key}">${m.label}${required ? ' <span class="req" aria-label="required">*</span>' : ""}</label>
        <input id="f-${key}" name="${key}" class="form-input" type="${m.type}" ${m.step ? `step="${m.step}"` : ""}
               placeholder="${m.placeholder}" value="${escapeHtml(v)}" ${required ? "required" : ""}
               autocomplete="off" inputmode="${m.type === "number" ? "decimal" : "text"}">
        <div class="field-status"></div>
      </div>`;
  },

  currentFields() {
    const out = {};
    this.form.querySelectorAll("input[name]").forEach(inp => {
      let v = inp.value.trim();
      if (inp.type === "date") v = fromIsoDate(v);
      if (inp.type === "number") v = v === "" ? null : Number(v);
      out[inp.name] = v === "" ? null : v;
    });
    return out;
  },

  refreshFieldStates() {
    const required = this.extraction.required_fields;
    const fields = this.currentFields();
    let missing = 0;
    this.form.querySelectorAll(".field").forEach(el => {
      const k = el.dataset.field;
      const v = fields[k];
      const ai = this.aiFields[k];
      const status = el.querySelector(".field-status");
      el.classList.remove("missing", "edited", "issue", "ai");
      const empty = v === null || v === undefined || v === "";
      const isReq = required.includes(k);
      if (empty && isReq) {
        missing++;
        el.classList.add("missing");
        status.textContent = ai == null ? "Not found on the invoice. Please enter it." : "Required";
      } else if (this.issues[k]) {
        el.classList.add("issue");
        status.textContent = this.issues[k];
      } else if (!empty && ai != null && String(v) !== String(ai) && !(typeof v === "number" && Number(ai) === v)) {
        el.classList.add("edited");
        status.textContent = `Edited (model read: ${ai})`;
      } else if (!empty && ai != null) {
        el.classList.add("ai");
        status.textContent = "Read by model";
      } else if (!empty) {
        el.classList.add("edited");
        status.textContent = "Entered manually";
      } else {
        status.textContent = "";
      }
    });

    const issueCount = Object.keys(this.issues).length;
    this.scoreBtn.disabled = missing > 0;
    this.scoreBtnText.textContent = missing > 0
      ? `Fill ${missing} required field${missing > 1 ? "s" : ""} to continue`
      : "Run cost & fraud models";

    const b = this.reviewBanner;
    if (missing > 0) {
      b.className = "alert-banner alert-warn";
      b.innerHTML = `<strong>${missing} required field${missing > 1 ? "s" : ""} could not be read.</strong> Look at the invoice and type ${missing > 1 ? "them" : "it"} in. The models only run once every required field is filled.`;
    } else if (issueCount) {
      b.className = "alert-banner alert-warn";
      b.innerHTML = `<strong>All required fields are filled</strong>, but ${issueCount} value${issueCount > 1 ? "s look" : " looks"} wrong. Check the highlighted fields before running the models.`;
    } else {
      b.className = "alert-banner alert-ok";
      b.innerHTML = "<strong>All required fields are filled.</strong> Check the values against the invoice image, then run the models.";
    }
  },

  onFieldInput() {
    this.refreshFieldStates();
    clearTimeout(this.checkTimer);
    this.checkTimer = setTimeout(() => this.recheck(), 600);
  },

  async recheck() {
    try {
      const res = await api.check({ fields: this.currentFields() });
      this.issues = res.field_issues || {};
      this.renderOrderSummary(res.order_summary);
      this.refreshFieldStates();
    } catch (_) { /* validation is advisory; scoring re-checks */ }
  },

  renderOrderSummary(s) {
    if (!s) { this.orderSummary.classList.add("hidden"); return; }
    this.orderSummary.innerHTML = `
      <div class="order-summary-title">Order ${escapeHtml(s.order_id)} in the order master</div>
      <dl>
        <dt>Route</dt><dd>${escapeHtml(s.origin)} → ${escapeHtml(s.destination)}</dd>
        <dt>Booked carrier</dt><dd>${escapeHtml(s.vendor_id)}</dd>
        <dt>Order date</dt><dd>${escapeHtml(s.order_date)}</dd>
        <dt>Truck</dt><dd>${escapeHtml(s.truck_type)}</dd>
        <dt>Billable weight</dt><dd>${Number(s.billable_weight_kg).toLocaleString("en-IN")} kg</dd>
        <dt>Quoted transit</dt><dd>${escapeHtml(s.quoted_days)} days</dd>
      </dl>`;
    this.orderSummary.classList.remove("hidden");
  },

  // -------------------------------------------------------------- step 3
  async runScore() {
    if (this.scoreBtn.disabled) return;
    this.hideError();
    this.setBusy(this.scoreBtn, this.scoreSpinner, this.scoreBtnText, true, "Running models…");
    let res = null;
    try {
      res = await api.score({
        fields: this.currentFields(), ai_fields: this.aiFields,
        filename: this.extraction.filename, extractor_used: this.extraction.extractor_used,
        extractor_model: this.extraction.extractor_model,
      });
    } catch (err) {
      this.showError(`Scoring failed: ${err.message}`);
    }
    this.setBusy(this.scoreBtn, this.scoreSpinner, this.scoreBtnText, false, "Run cost & fraud models");
    this.refreshFieldStates();
    if (res?.status === "gate_failed") {
      this.handleGateFailure(res);
    } else if (res) {
      this.renderResults(res);
      this.showStep(3);
    }
  },

  handleGateFailure(res) {
    const code = (res.gate_reason || "").split(":")[0];
    const msg = REASON_MAPPINGS[code] || res.gate_reason;
    (res.field_errors || []).forEach(k => { this.issues[k] = this.issues[k] || msg; });
    this.refreshFieldStates();
    this.reviewBanner.className = "alert-banner alert-danger";
    this.reviewBanner.innerHTML = `<strong>The validation gate stopped this invoice:</strong> ${escapeHtml(msg)} Fix the highlighted fields and run again. No fraud score was computed.`;
    this.reviewBanner.scrollIntoView({ behavior: "smooth", block: "center" });
  },

  renderResults(res) {
    const verdict = document.getElementById("verdict-badge");
    const explanation = document.getElementById("audit-explanation-box");
    document.getElementById("gate-status-box").classList.add("hidden");

    const billCheck = res.price_check?.billed;
    if (res.flagged) {
      verdict.className = "badge badge-flagged";
      verdict.textContent = res.flag_source === "logic_check"
        ? (billCheck?.status === "too_low" ? "FLAGGED: PRICE TOO LOW" : "FLAGGED: PRICE CHECK")
        : "FLAGGED FOR AUDIT";
      explanation.className = "explanation-box flagged";
    } else {
      verdict.className = "badge badge-approved";
      verdict.textContent = "APPROVED";
      explanation.className = "explanation-box approved";
    }
    explanation.textContent = res.explanation || "";

    document.getElementById("stat-billed-amount").textContent = formatCurrency(res.billed_amount);
    document.getElementById("stat-predicted-cost").textContent = formatCurrency(res.model1_predicted_cost);
    document.getElementById("stat-predicted-range").textContent =
      res.model1_interval_lower != null ? `80% range ${formatCurrency(res.model1_interval_lower)} – ${formatCurrency(res.model1_interval_upper)}` : "";
    const mm = Number(res.cost_mismatch || 0);
    const mmEl = document.getElementById("stat-cost-mismatch");
    mmEl.textContent = (mm > 0 ? "+" : "") + formatCurrency(mm);
    mmEl.style.color = mm > 0 ? "var(--color-flagged-text)" : "var(--color-approved-text)";
    const pct = Number(res.model2_proba || 0) * 100;
    document.getElementById("stat-risk-proba").textContent = `${pct.toFixed(1)}%`;
    const fill = document.getElementById("risk-meter-fill");
    fill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    fill.style.background = pct >= 50 ? "var(--color-flagged)" : "var(--color-approved)";

    const drift = document.getElementById("drift-box");
    if (res.drift_warnings?.length) {
      drift.innerHTML = "<strong>Data drift:</strong> " + res.drift_warnings.map(escapeHtml).join("<br>");
      drift.classList.remove("hidden");
    } else drift.classList.add("hidden");

    const c = res.context || {};
    const tw = c.transit_weather;
    const rows = [
      ["Diesel price used by Model 1", c.expected_fuel_price != null ? `₹${Number(c.expected_fuel_price).toFixed(2)}/L` : "—"],
      ["Adverse weather days in transit", `${c.adverse_weather_days ?? "—"} <span class="src">${escapeHtml(c.adverse_weather_source || "")}</span>`],
      tw ? ["Observed transit weather", `${tw.rain_mm_total} mm rain · max wind ${tw.max_wind_kmh} km/h · max ${tw.max_temp_c}°C <span class="src">${escapeHtml(tw.source)}, ${tw.start} to ${tw.end}</span>`] : null,
      ["Price logic check", billCheck ? (billCheck.status === "ok"
        ? `Passed. Billed amount is ${(billCheck.ratio * 100).toFixed(0)}% of the fair cost.`
        : `<strong>${escapeHtml(billCheck.status.replace("_", " "))}</strong>: ${escapeHtml(billCheck.message)}`) : "—"],
      ["Flagged by", res.flag_source ? escapeHtml(res.flag_source.replace("model", "fraud model").replace("logic_check", "price logic check").replace("+", " + ")) : "Not flagged"],
      ["Carrier historical risk", c.vendor_historical_risk_score != null ? `${(c.vendor_historical_risk_score * 100).toFixed(1)}%` : "—"],
      ["Fields entered or corrected by hand", res.manually_entered_fields?.length ? res.manually_entered_fields.map(k => FIELD_META[k]?.label || k).join(", ") : "None. All read by the model."],
      ["Extraction model", res.extractor_model ? `${escapeHtml(res.extractor_used)} · ${escapeHtml(res.extractor_model)}` : escapeHtml(res.extractor_used)],
    ].filter(Boolean);
    const drivers = (res.cost_drivers || []).map(d =>
      `<li><span>${escapeHtml(d.feature.replace(/_/g, " "))}</span><span class="${d.feature !== "model_adjustment" ? "" : d.impact > 0 ? "neg" : "pos"}">${d.impact > 0 ? "+" : "−"}₹${Math.abs(d.impact).toLocaleString("en-IN", { maximumFractionDigits: 0 })}</span></li>`).join("");
    const panel = document.getElementById("context-panel");
    panel.innerHTML = `
      <div class="context-col">
        <div class="context-title">Inputs and data sources</div>
        <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
      </div>
      ${drivers ? `<div class="context-col"><div class="context-title">Fair-cost build-up</div><ul class="driver-list">${drivers}</ul></div>` : ""}`;
    panel.classList.remove("hidden");
  },

  // ------------------------------------------------------------- helpers
  setBusy(btn, spinner, label, busy, text) {
    btn.disabled = busy;
    spinner.classList.toggle("hidden", !busy);
    label.textContent = text;
  },
  showError(msg) { this.errorBanner.textContent = msg; this.errorBanner.classList.remove("hidden"); },
  hideError() { this.errorBanner.classList.add("hidden"); this.errorBanner.textContent = ""; },
};

document.addEventListener("DOMContentLoaded", () => auditPage.init());
