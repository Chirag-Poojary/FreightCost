/**
 * dashboard.js - Audit summary: KPI cards, recent audits table and the
 * reasons invoices were stopped by validation.
 */

const REASON_LABELS = {
  missing_critical_field: "Required details missing",
  order_id_not_found: "Order number not in order records",
  vendor_id_not_found: "Carrier ID not registered",
  arithmetic_mismatch: "Line items do not add up to the total",
  unparseable_amount_or_days: "Amount or transit days not readable",
  amount_out_of_plausible_range: "Total outside ₹500 to ₹2,50,000",
  days_out_of_plausible_range: "Transit outside 0.2 to 30 days",
  invalid_order_record: "Order record has impossible values",
};

const fmtINR = v => (v === null || v === undefined || isNaN(v)) ? "—"
  : Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 });

const fmtDate = v => {
  if (!v) return "—";
  const d = new Date(v);
  return isNaN(d) ? escapeHtml(v) : d.toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
};

const dashboardPage = {
  init() {
    this.nInput = document.getElementById("dash-n-input");
    this.refreshBtn = document.getElementById("dash-refresh-btn");
    this.errorBanner = document.getElementById("dash-error-banner");
    this.breakdownContainer = document.getElementById("rejection-breakdown-container");
    this.recentBody = document.getElementById("recent-audits-tbody");
    this.storageNote = document.getElementById("dash-storage-note");
    this.refreshBtn.addEventListener("click", () => this.load());
    this.nInput.addEventListener("change", () => this.load());
  },

  async load() {
    const n = parseInt(this.nInput.value, 10) || 300;
    this.refreshBtn.disabled = true;
    this.refreshBtn.textContent = "Loading…";
    this.errorBanner.classList.add("hidden");
    try {
      const [data, recent, health] = await Promise.all([
        api.dashboard(n), api.recentAudits(20), api.health().catch(() => ({})),
      ]);
      this.render(data);
      this.renderRecent(recent);
      this.storageNote.textContent = health.database_mode === "local_csv"
        ? "The server is running without a database, so this list resets when the server restarts."
        : "";
    } catch (err) {
      this.errorBanner.textContent = `Could not load the dashboard: ${err.message}`;
      this.errorBanner.classList.remove("hidden");
    } finally {
      this.refreshBtn.disabled = false;
      this.refreshBtn.textContent = "Refresh";
    }
  },

  render(data) {
    document.getElementById("dash-stat-total").textContent = data.total_processed.toLocaleString("en-IN");
    document.getElementById("dash-stat-rate").textContent = `${(Number(data.auto_processing_rate || 0) * 100).toFixed(1)}%`;
    document.getElementById("dash-stat-rejected").textContent = data.gate_rejected.toLocaleString("en-IN");
    document.getElementById("dash-stat-flagged").textContent = data.flagged_count.toLocaleString("en-IN");

    const entries = Object.entries(data.rejection_breakdown || {}).sort((a, b) => b[1] - a[1]);
    const total = data.gate_rejected || 0;
    if (!entries.length) {
      this.breakdownContainer.innerHTML = '<div class="muted-note">No invoices were stopped by validation in this period.</div>';
      return;
    }
    this.breakdownContainer.innerHTML = entries.map(([code, count]) => {
      const pct = total > 0 ? ((count / total) * 100).toFixed(1) : 0;
      return `
        <div class="bar-row">
          <div class="bar-header"><span>${escapeHtml(REASON_LABELS[code] || code || "Other")}</span><span class="count">${count} (${pct}%)</span></div>
          <div class="bar-track"><div class="bar-fill" style="width: ${pct}%"></div></div>
        </div>`;
    }).join("");
  },

  renderRecent(rows) {
    if (!rows || !rows.length) {
      this.recentBody.innerHTML = '<tr><td colspan="8" class="empty-cell">No audits yet. Run one from Invoice Audit.</td></tr>';
      return;
    }
    this.recentBody.innerHTML = rows.map(r => {
      let status;
      if (!r.gate_passed) status = '<span class="badge badge-pending">Stopped</span>';
      else if (r.flagged) status = '<span class="badge badge-flagged">Flagged</span>';
      else status = '<span class="badge badge-approved">Approved</span>';
      const risk = r.model2_proba == null ? "—" : `${(Number(r.model2_proba) * 100).toFixed(1)}%`;
      return `<tr>
        <td>${fmtDate(r.created_at)}</td>
        <td>${escapeHtml(r.filename || "—")}</td>
        <td class="mono">${escapeHtml(r.order_id || "—")}</td>
        <td class="mono">${escapeHtml(r.vendor_id || "—")}</td>
        <td class="num">${fmtINR(r.billed_amount)}</td>
        <td class="num">${fmtINR(r.model1_predicted_cost)}</td>
        <td class="num">${risk}</td>
        <td>${status}</td>
      </tr>`;
    }).join("");
  },
};

window.dashboardPage = dashboardPage;
document.addEventListener("DOMContentLoaded", () => dashboardPage.init());
