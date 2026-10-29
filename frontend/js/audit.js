/**
 * audit.js - Three-step audit flow.
 *   1. Upload  -> POST /api/extract   (cloud vision model reads the invoice)
 *   2. Review  -> user confirms AI-read values and types any missing ones;
 *                 POST /api/check re-validates as they edit
 *   3. Verdict -> POST /api/score     (gate + Model 1 + Model 2)
 * The run button stays disabled until every required field has a value,
 * so the models never run on incomplete data.
 */

const TRUCK_TYPES = ["6-wheeler", "10-wheeler", "12-wheeler"];
const PRODUCT_CATEGORIES = ["Steel Coils", "Automotive Spare Parts", "Industrial Chemicals",
  "Bulk Textiles", "Electronics Components"];

// type: text | number | date | select (fixed options) ; list: id of a <datalist> with suggestions
const FIELD_META = {
  invoice_id:       { label: "Invoice number",       placeholder: "INV000123", type: "text" },
  order_id:         { label: "Order / LR number",    placeholder: "ORD000123", type: "text" },
  vendor_id:        { label: "Carrier ID",           placeholder: "VEN001", type: "text", list: "dl-vendors" },
  invoice_date:     { label: "Invoice date",         type: "date" },
  actual_days:      { label: "Actual transit (days)", placeholder: "e.g. 1.5", type: "number", step: "any", min: "0" },
  freight_base:     { label: "Base freight (₹)",     placeholder: "e.g. 12500", type: "number", step: "any", min: "0" },
  detention:        { label: "Detention (₹)",        placeholder: "0 if none", type: "number", step: "any", min: "0" },
  toll:             { label: "Toll (₹)",             placeholder: "0 if none", type: "number", step: "any", min: "0" },
  total:            { label: "Invoice total (₹)",    placeholder: "Final payable", type: "number", step: "any", min: "0" },
  origin:           { label: "From (city)",          placeholder: "Select or type a city", type: "text", list: "dl-cities" },
  destination:      { label: "To (city)",            placeholder: "Select or type a city", type: "text", list: "dl-cities" },
  truck_type:       { label: "Truck",                type: "select", options: TRUCK_TYPES },
  product_category: { label: "Goods",                type: "select", options: PRODUCT_CATEGORIES },
  weight_kg:        { label: "Weight (kg)",          placeholder: "e.g. 9800", type: "number", step: "any", min: "0" },
  vendor_gstin:     { label: "Carrier GSTIN",        placeholder: "15 characters", type: "text" },
};
const REQUIRED_DEFAULT = ["invoice_id", "order_id", "vendor_id", "invoice_date", "actual_days",
  "freight_base", "detention", "toll", "total"];

const REASON_MAPPINGS = {
  missing_critical_field: "One or more required fields are empty.",
  order_id_not_found: "This order number does not exist in the order records.",
  vendor_id_not_found: "This carrier ID is not registered.",
  unparseable_amount_or_days: "The total or transit days could not be read as numbers.",
  arithmetic_mismatch: "Base freight + detention + toll does not add up to the total.",
  amount_out_of_plausible_range: "The total is outside ₹500 to ₹2,50,000.",
  days_out_of_plausible_range: "Transit days are outside 0.2 to 30 days.",
  invalid_order_record: "The order record has impossible values.",
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
      sampleGrid: $("sample-grid"), errorBanner: $("audit-error-banner"),
      uploadCard: $("upload-card"), reviewCard: $("review-card"), resultsCard: $("audit-results-card"),
      fieldsContainer: $("fields-container"), form: $("fields-form"), scoreBtn: $("score-btn"),
      scoreSpinner: $("score-spinner"), scoreBtnText: $("score-btn-text"), reviewBanner: $("review-banner"),
      orderSummary: $("order-summary"),
    });
    this.bindEvents();
    this.loadExtractors();
    this.loadSamples();
    this.loadSuggestionLists();
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
    this.form.addEventListener("submit", e => { e.preventDefault(); this.runScore(); });
    this.form.addEventListener("input", () => this.onFieldInput());
    this.form.addEventListener("change", () => this.onFieldInput());
    document.getElementById("back-btn").addEventListener("click", () => this.reset());
    document.getElementById("new-audit-btn").addEventListener("click", () => this.reset());
    document.getElementById("edit-again-btn").addEventListener("click", () => this.showStep(2));
    document.getElementById("download-report-btn").addEventListener("click", () => this.downloadReport());
  },

  async downloadReport() {
    if (!this.lastResult) return;
    const btn = document.getElementById("download-report-btn");
    const label = document.getElementById("download-report-text");
    btn.disabled = true;
    label.textContent = "Preparing PDF…";
    try {
      const { blob, filename } = await api.reportPdf(this.lastResult);
      const url = URL.createObjectURL(blob);
      const a = Object.assign(document.createElement("a"), { href: url, download: filename });
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      this.showError(`Could not create the PDF report: ${err.message}`);
    } finally {
      btn.disabled = false;
      label.textContent = "Download PDF report";
    }
  },

  // ---------------------------------------------------------------- setup
  async loadExtractors() {
    try {
      const { extractors } = await api.extractors();
      this.extractors = extractors;
      extractors.forEach(x => {
        const opt = new Option(`${x.label} (${x.model})${x.configured ? "" : ", not configured"}`, x.id);
        opt.disabled = !x.configured;
        this.extractorSelect.add(opt);
      });
      this.updateExtractorHint();
    } catch (_) {
      this.extractorHint.textContent = "Could not load the model list. The server may be waking up; try again in a minute.";
    }
  },

  updateExtractorHint() {
    const list = this.extractors || [];
    const ready = list.filter(x => x.configured);
    if (!ready.length) {
      this.extractorHint.textContent = "No reading model is set up on the server. You can still type the invoice details in by hand.";
      return;
    }
    const sel = this.extractorSelect.value;
    const x = sel === "auto" ? ready[0] : list.find(e => e.id === sel);
    this.extractorHint.textContent = sel === "auto"
      ? `Uses ${x.label} first and switches to the next model if it is busy.`
      : `${x.label} (${x.model})`;
  },

  async loadSamples() {
    try {
      const res = await fetch("demo/manifest.json", { cache: "no-store" });
      if (!res.ok) throw new Error();
      const samples = await res.json();
      this.sampleGrid.innerHTML = samples.map((s, i) => `
        <button type="button" class="sample-card" data-index="${i}">
          <div class="sample-thumb"><img src="demo/${escapeHtml(s.thumb || s.image)}" alt="Sample invoice: ${escapeHtml(s.title)}" loading="lazy"></div>
          <div class="sample-body">
            <div class="sample-title">${escapeHtml(s.title)}</div>
            <div class="sample-desc">${escapeHtml(s.description || "")}</div>
            <div class="sample-actions"><span class="use">Use this sample</span><a href="demo/${escapeHtml(s.image)}" target="_blank" rel="noopener" data-view>View</a></div>
          </div>
        </button>`).join("");
      this.sampleGrid.querySelectorAll(".sample-card").forEach(card => {
        card.addEventListener("click", e => {
          if (e.target.closest("[data-view]")) return;   // the View link opens the image
          this.loadSample(samples[Number(card.dataset.index)], card);
        });
      });
    } catch (_) {
      this.sampleGrid.innerHTML = '<div class="muted-note">Sample invoices are not available.</div>';
    }
  },

  async loadSample(sample, card) {
    try {
      const resp = await fetch(`demo/${sample.image}`);
      if (!resp.ok) throw new Error();
      const blob = await resp.blob();
      const type = blob.type || (sample.image.endsWith(".jpg") ? "image/jpeg" : "image/png");
      if (this.setFile(new File([blob], sample.image, { type }))) {
        this.sampleGrid.querySelectorAll(".sample-card").forEach(c => c.classList.toggle("selected", c === card));
        this.extractBtn.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    } catch (_) {
      this.showError("Could not load the sample invoice. Please try again.");
    }
  },

  async loadSuggestionLists() {
    const add = (id, values) => {
      const dl = document.createElement("datalist");
      dl.id = id;
      dl.innerHTML = values.map(v => `<option value="${escapeHtml(v.value)}">${escapeHtml(v.label || "")}</option>`).join("");
      document.body.appendChild(dl);
    };
    try {
      const hubs = await api.hubs();
      add("dl-cities", hubs.map(h => ({ value: h.city, label: h.state })));
    } catch (_) { add("dl-cities", []); }
    try {
      const vendors = await api.vendors();
      add("dl-vendors", vendors.map(v => ({ value: v.vendor_id, label: v.vendor_name })));
    } catch (_) { add("dl-vendors", []); }
  },

  setFile(file) {
    if (!/^image\/(png|jpe?g|webp)$/.test(file.type)) {
      this.showError("Please upload a PNG, JPG or WEBP image of the invoice.");
      return false;
    }
    if (file.size > 10 * 1024 * 1024) {
      this.showError("That image is larger than 10 MB. Please upload a smaller scan or photo.");
      return false;
    }
    this.selectedFile = file;
    if (this.imageUrl) URL.revokeObjectURL(this.imageUrl);
    this.imageUrl = URL.createObjectURL(file);
    this.fileThumb.src = this.imageUrl;
    this.fileName.textContent = file.name;
    this.fileSize.textContent = `${(file.size / 1024).toFixed(0)} KB`;
    this.filePreview.classList.remove("hidden");
    this.extractBtn.disabled = false;
    this.hideError();
    return true;
  },

  clearFile() {
    this.selectedFile = null;
    this.fileInput.value = "";
    this.filePreview.classList.add("hidden");
    this.extractBtn.disabled = true;
    this.sampleGrid.querySelectorAll(".sample-card").forEach(c => c.classList.remove("selected"));
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
      // Reading failed entirely: still let the user type the details in.
      this.showError(`The invoice could not be read automatically (${err.message}). Please enter the details by hand.`);
      this.openReview({
        fields: {}, missing_fields: REQUIRED_DEFAULT, required_fields: REQUIRED_DEFAULT,
        field_issues: {}, extractor_used: "manual", extractor_model: null, fallback_errors: [], filename: this.selectedFile.name,
      }, true);
    } finally {
      this.setBusy(this.extractBtn, this.extractSpinner, this.extractBtnText, false, "Read invoice");
    }
  },

  // -------------------------------------------------------------- step 2
  openReview(res, failed = false) {
    this.extraction = res;
    this.aiFields = { ...res.fields };
    this.issues = res.field_issues || {};
    const required = res.required_fields || REQUIRED_DEFAULT;
    this.extraction.required_fields = required;
    const optional = Object.keys(FIELD_META).filter(k => !required.includes(k));

    const badge = document.getElementById("extractor-used-badge");
    badge.textContent = failed ? "Manual entry" : `Read by ${res.extractor_model || res.extractor_used}`;
    badge.title = res.fallback_errors?.length
      ? "Switched model after: " + res.fallback_errors.map(e => `${e.provider}: ${e.error}`).join("\n") : "";

    document.getElementById("review-image").src = this.imageUrl;
    document.getElementById("review-image-link").href = this.imageUrl;

    const section = (title, keys, isReq) => `
      <fieldset class="field-section">
        ${title ? `<legend>${title}</legend>` : ""}
        <div class="field-grid">${keys.map(k => this.fieldHtml(k, isReq)).join("")}</div>
      </fieldset>`;
    const anyOptionalFilled = optional.some(k => this.aiFields[k] != null && this.aiFields[k] !== "");
    this.fieldsContainer.innerHTML =
      section("Invoice details", required, true) +
      `<details class="optional-fields"${anyOptionalFilled ? " open" : ""}><summary>Trip details (optional, improves the estimate for new orders)</summary>${section("", optional, false)}</details>`;

    this.renderOrderSummary(res.order_summary);
    this.refreshFieldStates();
    this.showStep(2);
    const firstMissing = this.fieldsContainer.querySelector(".field.missing input, .field.missing select");
    if (firstMissing) setTimeout(() => firstMissing.focus({ preventScroll: true }), 400);
  },

  fieldHtml(key, required) {
    const m = FIELD_META[key];
    let v = this.aiFields[key];
    if (v === null || v === undefined) v = "";
    const req = required ? ' <span class="req" aria-hidden="true">*</span>' : "";
    let control;
    if (m.type === "select") {
      const opts = m.options.map(o => `<option value="${escapeHtml(o)}"${o === v ? " selected" : ""}>${escapeHtml(o)}</option>`).join("");
      const extra = v && !m.options.includes(v) ? `<option value="${escapeHtml(v)}" selected>${escapeHtml(v)}</option>` : "";
      control = `<select id="f-${key}" name="${key}" class="form-select"${required ? " required" : ""}>
                   <option value="">Not specified</option>${opts}${extra}</select>`;
    } else {
      if (m.type === "date") v = toIsoDate(v);
      control = `<input id="f-${key}" name="${key}" class="form-input" type="${m.type}"
                   ${m.step ? `step="${m.step}"` : ""} ${m.min ? `min="${m.min}"` : ""} ${m.list ? `list="${m.list}"` : ""}
                   ${m.placeholder ? `placeholder="${escapeHtml(m.placeholder)}"` : ""} value="${escapeHtml(v)}"
                   ${required ? "required" : ""} autocomplete="off" ${m.type === "number" ? 'inputmode="decimal"' : ""}>`;
    }
    return `
      <div class="field" data-field="${key}">
        <label for="f-${key}">${m.label}${req}</label>
        ${control}
        <div class="field-status"></div>
      </div>`;
  },

  currentFields() {
    const out = {};
    this.form.querySelectorAll("input[name], select[name]").forEach(el => {
      let v = el.value.trim();
      if (el.type === "date") v = fromIsoDate(v);
      if (el.type === "number") v = v === "" ? null : Number(v);
      out[el.name] = v === "" ? null : v;
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
      const same = !empty && ai != null && (String(v) === String(ai) || (typeof v === "number" && Number(ai) === v));
      if (empty && isReq) {
        missing++;
        el.classList.add("missing");
        status.textContent = ai == null ? "Not found on the invoice. Please enter it." : "Required";
      } else if (this.issues[k]) {
        el.classList.add("issue");
        status.textContent = this.issues[k];
      } else if (!empty && ai != null && !same) {
        el.classList.add("edited");
        status.textContent = `Changed (read as ${ai})`;
      } else if (same) {
        el.classList.add("ai");
        status.textContent = "Read from invoice";
      } else if (!empty) {
        el.classList.add("edited");
        status.textContent = "Entered by you";
      } else {
        status.textContent = "";
      }
    });

    const issueCount = Object.keys(this.issues).length;
    this.scoreBtn.disabled = missing > 0;
    this.scoreBtnText.textContent = missing > 0
      ? `Fill ${missing} required field${missing > 1 ? "s" : ""} to continue`
      : "Run audit";

    const b = this.reviewBanner;
    if (missing > 0) {
      b.className = "alert-banner alert-warn";
      b.innerHTML = `<strong>${missing} required field${missing > 1 ? "s" : ""} could not be read.</strong> Look at the invoice and enter ${missing > 1 ? "them" : "it"}. The audit runs only when every required field is filled.`;
    } else if (issueCount) {
      b.className = "alert-banner alert-warn";
      b.innerHTML = `<strong>All required fields are filled</strong>, but ${issueCount} value${issueCount > 1 ? "s look" : " looks"} wrong. Check the highlighted fields.`;
    } else {
      b.className = "alert-banner alert-ok";
      b.innerHTML = "<strong>All required fields are filled.</strong> Check the values against the invoice, then run the audit.";
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
    } catch (_) { /* advisory only; the server re-checks when scoring */ }
  },

  renderOrderSummary(s) {
    if (!s) { this.orderSummary.classList.add("hidden"); return; }
    const title = s.is_new
      ? `Order ${escapeHtml(s.order_id)} <span class="new-tag">New order</span>`
      : `Order ${escapeHtml(s.order_id)} in your order records`;
    const weight = Number(s.billable_weight_kg) > 0 ? `${Number(s.billable_weight_kg).toLocaleString("en-IN")} kg` : "Not given";
    this.orderSummary.innerHTML = `
      <div class="order-summary-title">${title}</div>
      <dl>
        <dt>Route</dt><dd>${escapeHtml(s.origin)} to ${escapeHtml(s.destination)}</dd>
        <dt>Carrier</dt><dd>${escapeHtml(s.vendor_id)}</dd>
        <dt>Order date</dt><dd>${escapeHtml(s.order_date || "Not given")}</dd>
        <dt>Truck</dt><dd>${escapeHtml(s.truck_type)}</dd>
        <dt>Weight</dt><dd>${weight}</dd>
        <dt>Quoted transit</dt><dd>${escapeHtml(s.quoted_days)} days</dd>
      </dl>
      ${s.is_new ? '<div class="field-hint" style="margin-top:8px">Not in your order records, so the route and costs are estimated from the invoice. Filling the trip details improves the estimate.</div>' : ""}`;
    this.orderSummary.classList.remove("hidden");
    // For a new order the trip details drive the estimate, so show them.
    if (s.is_new) this.fieldsContainer.querySelector(".optional-fields")?.setAttribute("open", "");
  },

  // -------------------------------------------------------------- step 3
  async runScore() {
    if (this.scoreBtn.disabled) return;
    this.hideError();
    this.setBusy(this.scoreBtn, this.scoreSpinner, this.scoreBtnText, true, "Running audit…");
    let res = null;
    try {
      res = await api.score({
        fields: this.currentFields(), ai_fields: this.aiFields,
        filename: this.extraction.filename, extractor_used: this.extraction.extractor_used,
        extractor_model: this.extraction.extractor_model,
      });
    } catch (err) {
      this.showError(`The audit could not be completed: ${err.message}`);
    }
    this.setBusy(this.scoreBtn, this.scoreSpinner, this.scoreBtnText, false, "Run audit");
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
    this.reviewBanner.innerHTML = `<strong>The invoice did not pass validation:</strong> ${escapeHtml(msg)} Correct the highlighted fields and run again.`;
    this.reviewBanner.scrollIntoView({ behavior: "smooth", block: "center" });
  },

  renderResults(res) {
    this.lastResult = res;
    const verdict = document.getElementById("verdict-badge");
    const explanation = document.getElementById("audit-explanation-box");
    document.getElementById("gate-status-box").classList.add("hidden");
    const f = res.extracted_fields || {};
    document.getElementById("result-subtitle").textContent =
      [f.invoice_id && `Invoice ${f.invoice_id}`, f.order_id && `Order ${f.order_id}`, f.vendor_id && `Carrier ${f.vendor_id}`]
        .filter(Boolean).join("  ·  ");

    const billCheck = res.price_check?.billed;
    if (res.flagged) {
      verdict.className = "badge badge-flagged";
      verdict.textContent = res.flag_source === "logic_check"
        ? (billCheck?.status === "too_low" ? "Flagged: price too low" : "Flagged: price check")
        : "Flagged for audit";
      explanation.className = "explanation-box flagged";
    } else {
      verdict.className = "badge badge-approved";
      verdict.textContent = "Approved";
      explanation.className = "explanation-box approved";
    }
    explanation.textContent = res.explanation || "";

    const reasonsBox = document.getElementById("reasons-box");
    const tags = { pass: "PASS", check: "CHECK", info: "NOTE" };
    document.getElementById("reasons-list").innerHTML = (res.reasons || []).map(r => `
      <li><span class="r-tag ${tags[r.status] ? r.status : "info"}">${tags[r.status] || "NOTE"}</span>
        <div><div class="r-title">${escapeHtml(r.title)}</div><div class="r-detail">${escapeHtml(r.detail)}</div></div></li>`).join("");
    reasonsBox.classList.toggle("hidden", !res.reasons?.length);

    document.getElementById("stat-billed-amount").textContent = formatCurrency(res.billed_amount);
    document.getElementById("stat-predicted-cost").textContent = formatCurrency(res.model1_predicted_cost);
    document.getElementById("stat-predicted-range").textContent =
      res.model1_interval_lower != null ? `Likely range ${formatCurrency(res.model1_interval_lower)} to ${formatCurrency(res.model1_interval_upper)}` : "";
    const mm = Number(res.cost_mismatch || 0);
    const mmEl = document.getElementById("stat-cost-mismatch");
    mmEl.textContent = (mm > 0 ? "+" : mm < 0 ? "−" : "") + formatCurrency(Math.abs(mm));
    mmEl.style.color = mm > 0 ? "var(--danger)" : "var(--success)";
    const pct = Number(res.model2_proba || 0) * 100;
    document.getElementById("stat-risk-proba").textContent = `${pct.toFixed(1)}%`;
    const fill = document.getElementById("risk-meter-fill");
    fill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    fill.style.background = pct >= 50 ? "var(--danger)" : "var(--success)";

    const drift = document.getElementById("drift-box");
    if (res.drift_warnings?.length) {
      drift.innerHTML = "<strong>Note on prices:</strong> " + res.drift_warnings.map(escapeHtml).join("<br>");
      drift.classList.remove("hidden");
    } else drift.classList.add("hidden");

    const c = res.context || {};
    const tw = c.transit_weather;
    const rows = [
      ["Diesel price used", c.expected_fuel_price != null ? `₹${Number(c.expected_fuel_price).toFixed(2)} per litre` : "—"],
      ["Heavy-rain days in transit", `${c.adverse_weather_days ?? "—"} <span class="src">${escapeHtml(c.adverse_weather_source || "")}</span>`],
      tw ? ["Weather during the trip", `${tw.rain_mm_total} mm rain, wind up to ${tw.max_wind_kmh} km/h, up to ${tw.max_temp_c}°C <span class="src">${escapeHtml(tw.source)}, ${escapeHtml(tw.start)} to ${escapeHtml(tw.end)}</span>`] : null,
      ["Price check", billCheck ? (billCheck.status === "ok"
        ? `Passed. Bill is ${(billCheck.ratio * 100).toFixed(0)}% of the fair cost.`
        : `<strong>${escapeHtml(billCheck.status.replace("_", " "))}</strong>: ${escapeHtml(billCheck.message)}`) : "—"],
      ["Flagged by", res.flag_source ? escapeHtml(res.flag_source.replace("model", "risk model").replace("logic_check", "price check").replace("+", " and ")) : "Not flagged"],
      ["Carrier's past risk", c.vendor_historical_risk_score != null ? `${(c.vendor_historical_risk_score * 100).toFixed(1)}%` : "—"],
      ["Entered or changed by you", res.manually_entered_fields?.length ? res.manually_entered_fields.map(k => FIELD_META[k]?.label || k).join(", ") : "None"],
      ...(c.notes || []).map(n => ["Note", escapeHtml(n)]),
      ["Read by", escapeHtml(res.extractor_model || (res.extractor_used === "manual" ? "Entered by hand" : res.extractor_used))],
    ].filter(Boolean);
    const drivers = (res.cost_drivers || []).map(d =>
      `<li><span>${escapeHtml(d.feature.replace(/_/g, " "))}</span><span class="${d.feature !== "model_adjustment" ? "" : d.impact > 0 ? "neg" : "pos"}">${d.impact > 0 ? "+" : "−"}₹${Math.abs(d.impact).toLocaleString("en-IN", { maximumFractionDigits: 0 })}</span></li>`).join("");
    const panel = document.getElementById("context-panel");
    panel.innerHTML = `
      <div class="context-col">
        <div class="context-title">Inputs and sources</div>
        <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
      </div>
      ${drivers ? `<div class="context-col"><div class="context-title">Fair cost build-up</div><ul class="driver-list">${drivers}</ul></div>` : ""}`;
    panel.classList.remove("hidden");
  },

  // ------------------------------------------------------------- helpers
  setBusy(btn, spinner, label, busy, text) {
    btn.disabled = busy;
    spinner.classList.toggle("hidden", !busy);
    label.textContent = text;
  },
  showError(msg) { this.errorBanner.textContent = msg; this.errorBanner.classList.remove("hidden"); this.errorBanner.scrollIntoView({ behavior: "smooth", block: "center" }); },
  hideError() { this.errorBanner.classList.add("hidden"); this.errorBanner.textContent = ""; },
};

document.addEventListener("DOMContentLoaded", () => auditPage.init());
