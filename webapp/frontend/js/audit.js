/**
 * audit.js - Audit page logic.
 * Handles drag-and-drop file upload, extractor selection, API submission,
 * loading states, 3-state verdict badge rendering, and structured results.
 */

const REASON_MAPPINGS = {
  "missing_critical_field": "One or more required fields could not be read from this invoice.",
  "order_id_not_found": "The extracted Order ID does not match any internal order record.",
  "vendor_id_not_found": "The extracted Vendor ID is not registered in our vendor directory.",
  "unparseable_amount_or_days": "Invoice monetary total or transit duration could not be parsed.",
  "arithmetic_mismatch": "Line items (freight base, detention, tolls) do not sum to the billed total.",
  "amount_out_of_plausible_range": "Billed amount is outside plausible freight boundaries (₹500 to ₹250,000).",
  "days_out_of_plausible_range": "Transit duration is outside physical operational limits (0.2 to 30 days)."
};

function formatCurrency(val) {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return "₹" + Number(val).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

const auditPage = {
  selectedFile: null,

  init() {
    this.dropzone = document.getElementById("dropzone");
    this.fileInput = document.getElementById("file-input");
    this.filePreview = document.getElementById("file-preview");
    this.fileName = document.getElementById("file-name");
    this.fileSize = document.getElementById("file-size");
    this.removeBtn = document.getElementById("file-remove-btn");
    this.auditBtn = document.getElementById("audit-btn");
    this.auditSpinner = document.getElementById("audit-spinner");
    this.auditBtnText = document.getElementById("audit-btn-text");
    this.resultsCard = document.getElementById("audit-results-card");
    this.errorBanner = document.getElementById("audit-error-banner");
    this.providerContainer = document.getElementById("llm-provider-container");
    this.demoBtn = document.getElementById("load-demo-btn");

    this.bindEvents();
  },

  bindEvents() {
    // Dropzone drag-drop
    this.dropzone.addEventListener("click", () => this.fileInput.click());
    this.dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      this.dropzone.classList.add("dragover");
    });
    this.dropzone.addEventListener("dragleave", () => this.dropzone.classList.remove("dragover"));
    this.dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      this.dropzone.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        this.setFile(e.dataTransfer.files[0]);
      }
    });

    // File input change
    this.fileInput.addEventListener("change", () => {
      if (this.fileInput.files && this.fileInput.files[0]) {
        this.setFile(this.fileInput.files[0]);
      }
    });

    // Remove file
    this.removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.clearFile();
    });

    // Extractor radio toggle
    document.querySelectorAll("input[name='extractor']").forEach(radio => {
      radio.addEventListener("change", (e) => {
        document.querySelectorAll(".radio-card").forEach(card => card.classList.remove("active"));
        e.target.closest(".radio-card").classList.add("active");
        if (e.target.value === "llm") {
          this.providerContainer.classList.remove("hidden");
        } else {
          this.providerContainer.classList.add("hidden");
        }
      });
    });

    // Primary action button
    this.auditBtn.addEventListener("click", () => this.runAudit());

    // Demo invoice loader
    if (this.demoBtn) {
      this.demoBtn.addEventListener("click", () => this.loadDemoInvoice());
    }
  },

  setFile(file) {
    if (!file.type.startsWith("image/")) {
      this.showError("Please upload an invoice image file (.png, .jpg, .jpeg).");
      return;
    }
    this.selectedFile = file;
    this.fileName.textContent = file.name;
    this.fileSize.textContent = `(${(file.size / 1024).toFixed(1)} KB)`;
    this.filePreview.classList.remove("hidden");
    this.auditBtn.disabled = false;
    this.hideError();
  },

  clearFile() {
    this.selectedFile = null;
    this.fileInput.value = "";
    this.filePreview.classList.add("hidden");
    this.auditBtn.disabled = true;
  },

  async loadDemoInvoice() {
    try {
      this.showError("Loading benchmark demo invoice...");
      // Fetch a sample invoice from phase_a_output
      const resp = await fetch("/phase_a_output/invoices_bench300/INV000026.png");
      if (!resp.ok) {
        throw new Error("Could not fetch demo invoice file from server.");
      }
      const blob = await resp.blob();
      const file = new File([blob], "INV000026.png", { type: "image/png" });
      this.setFile(file);
      this.hideError();
    } catch (err) {
      this.showError("To test, please drop or select an invoice image from your computer.");
    }
  },

  showError(msg) {
    this.errorBanner.textContent = msg;
    this.errorBanner.classList.remove("hidden");
  },

  hideError() {
    this.errorBanner.classList.add("hidden");
    this.errorBanner.textContent = "";
  },

  setLoading(isLoading) {
    this.auditBtn.disabled = isLoading;
    if (isLoading) {
      this.auditSpinner.classList.remove("hidden");
      this.auditBtnText.textContent = "Processing OCR & Models...";
    } else {
      this.auditSpinner.classList.add("hidden");
      this.auditBtnText.textContent = "Audit Invoice";
    }
  },

  async runAudit() {
    if (!this.selectedFile) return;

    this.hideError();
    this.setLoading(true);
    this.resultsCard.classList.add("hidden");

    const extractor = document.querySelector("input[name='extractor']:checked").value;
    const provider = document.getElementById("llm-provider").value;

    try {
      const result = await api.audit(this.selectedFile, extractor, provider);
      this.renderResults(result);
    } catch (err) {
      this.showError(`Audit Failed: ${err.message}`);
    } finally {
      this.setLoading(false);
    }
  },

  renderResults(res) {
    const verdictBadge = document.getElementById("verdict-badge");
    const gateStatusBox = document.getElementById("gate-status-box");
    const statsContainer = document.getElementById("audit-stats-container");
    const explanationBox = document.getElementById("audit-explanation-box");
    const fieldsTableBody = document.getElementById("extracted-fields-tbody");

    // Clear previous
    fieldsTableBody.innerHTML = "";

    // 1. Determine Verdict State (3 states per plan)
    if (!res.gate_passed) {
      verdictBadge.className = "badge badge-pending";
      verdictBadge.textContent = "MANUAL REVIEW REQUIRED";

      // Plain language failure reason
      const rawCode = (res.gate_reason || "").split(":")[0];
      const friendlyMsg = REASON_MAPPINGS[rawCode] || res.gate_reason || "Confidence validation gate check failed.";
      gateStatusBox.innerHTML = `<strong>Gate Rejection:</strong> ${friendlyMsg} <span style="font-family: var(--font-mono); font-size: 0.8rem; opacity: 0.8;">(${res.gate_reason})</span>`;
      gateStatusBox.className = "alert-banner alert-danger";
      gateStatusBox.classList.remove("hidden");

      statsContainer.classList.add("hidden");
      explanationBox.className = "explanation-box pending";
      explanationBox.textContent = res.explanation || friendlyMsg;
    } else {
      // Gate passed: Check flagged state
      gateStatusBox.classList.add("hidden");
      statsContainer.classList.remove("hidden");

      if (res.flagged) {
        verdictBadge.className = "badge badge-flagged";
        verdictBadge.textContent = "FLAGGED FOR AUDIT";
        explanationBox.className = "explanation-box flagged";
      } else {
        verdictBadge.className = "badge badge-approved";
        verdictBadge.textContent = "APPROVED";
        explanationBox.className = "explanation-box approved";
      }

      // Render 4 Stats
      document.getElementById("stat-billed-amount").textContent = formatCurrency(res.billed_amount);
      document.getElementById("stat-predicted-cost").textContent = formatCurrency(res.model1_predicted_cost);

      const mismatchElem = document.getElementById("stat-cost-mismatch");
      const mismatch = Number(res.cost_mismatch || 0);
      mismatchElem.textContent = (mismatch > 0 ? "+" : "") + formatCurrency(mismatch);
      mismatchElem.style.color = mismatch > 0 ? "var(--color-flagged)" : "var(--color-approved)";

      const probaPct = (Number(res.model2_proba || 0) * 100).toFixed(1);
      document.getElementById("stat-risk-proba").textContent = `${probaPct}%`;
      const meterFill = document.getElementById("risk-meter-fill");
      meterFill.style.width = `${Math.min(100, Math.max(0, probaPct))}%`;
      meterFill.style.background = Number(probaPct) > 50 ? "var(--color-flagged)" : "var(--color-approved)";

      explanationBox.textContent = res.explanation || "Invoice successfully scored by Model 1 and Model 2.";
    }

    // 2. Render Extracted Fields Table
    const fields = res.extracted_fields || {};
    const keys = Object.keys(fields);
    if (keys.length === 0) {
      fieldsTableBody.innerHTML = `<tr><td colspan="2" style="text-align: center; color: var(--color-text-dim);">No structured fields extracted</td></tr>`;
    } else {
      keys.forEach(k => {
        const row = document.createElement("tr");
        const val = fields[k] !== null && fields[k] !== undefined ? fields[k] : "—";
        row.innerHTML = `
          <td style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--color-primary);">${k}</td>
          <td style="font-weight: 500;">${val}</td>
        `;
        fieldsTableBody.appendChild(row);
      });
    }

    this.resultsCard.classList.remove("hidden");
    this.resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
};

document.addEventListener("DOMContentLoaded", () => auditPage.init());
