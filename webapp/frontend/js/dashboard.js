/**
 * dashboard.js - Operational Audit Dashboard logic.
 * Renders sample-size controls, 4 high-level KPI cards, and
 * visual CSS percentage breakdown bars for gate rejection reasons.
 */

const REASON_LABELS = {
  "missing_critical_field": "Missing Critical Fields (order_id, vendor, total, days)",
  "order_id_not_found": "Referential Integrity: Order ID Not Found in ERP",
  "vendor_id_not_found": "Carrier Registry: Vendor ID Not Recognized",
  "arithmetic_mismatch": "Arithmetic Checksum: Line Items Disagree With Total",
  "unparseable_amount_or_days": "Data Quality: Unparseable Amount or Transit Days",
  "amount_out_of_plausible_range": "Plausibility Check: Total Outside ₹500 - ₹250K",
  "days_out_of_plausible_range": "Plausibility Check: Transit Outside 0.2 - 30 Days"
};

const dashboardPage = {
  init() {
    this.nInput = document.getElementById("dash-n-input");
    this.refreshBtn = document.getElementById("dash-refresh-btn");
    this.errorBanner = document.getElementById("dash-error-banner");
    this.breakdownContainer = document.getElementById("rejection-breakdown-container");

    this.refreshBtn.addEventListener("click", () => this.load());
  },

  async load() {
    const n = parseInt(this.nInput.value, 10) || 300;
    this.refreshBtn.disabled = true;
    this.refreshBtn.textContent = "Loading...";
    this.errorBanner.classList.add("hidden");

    try {
      const data = await api.dashboard(n);
      this.render(data);
    } catch (err) {
      this.errorBanner.textContent = `Dashboard load failed: ${err.message}`;
      this.errorBanner.classList.remove("hidden");
    } finally {
      this.refreshBtn.disabled = false;
      this.refreshBtn.textContent = "Refresh";
    }
  },

  render(data) {
    // 1. Render 4 Stat Cards
    document.getElementById("dash-stat-total").textContent = data.total_processed.toLocaleString();
    
    const ratePct = (Number(data.auto_processing_rate || 0) * 100).toFixed(1);
    document.getElementById("dash-stat-rate").textContent = `${ratePct}%`;

    document.getElementById("dash-stat-rejected").textContent = data.gate_rejected.toLocaleString();
    document.getElementById("dash-stat-flagged").textContent = data.flagged_count.toLocaleString();

    // 2. Render Rejection Reason Breakdown
    const breakdown = data.rejection_breakdown || {};
    const totalRejected = data.gate_rejected || 0;
    const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);

    this.breakdownContainer.innerHTML = "";

    if (entries.length === 0) {
      this.breakdownContainer.innerHTML = `
        <div style="text-align: center; color: var(--color-text-dim); padding: var(--space-unit) * 3;">
          No gate rejections recorded in the current sample window.
        </div>`;
      return;
    }

    entries.forEach(([reasonCode, count]) => {
      const pct = totalRejected > 0 ? ((count / totalRejected) * 100).toFixed(1) : 0;
      const label = REASON_LABELS[reasonCode] || reasonCode;

      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-header">
          <span style="font-weight: 500;">${label}</span>
          <span style="font-family: var(--font-mono); color: var(--color-text-dim); font-size: 0.85rem;">
            ${count} (${pct}%)
          </span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${pct}%;"></div>
        </div>
      `;
      this.breakdownContainer.appendChild(row);
    });
  }
};

window.dashboardPage = dashboardPage;
document.addEventListener("DOMContentLoaded", () => dashboardPage.init());
