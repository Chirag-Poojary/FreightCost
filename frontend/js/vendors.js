/**
 * vendors.js - Vendor Risk Explorer page logic.
 * Renders real-time Laplace-smoothed vendor risk scores, sortable columns,
 * search filtering, and colored risk badges.
 */

const vendorsPage = {
  data: [],
  sortField: "risk_score",
  sortAsc: false,

  init() {
    this.tableBody = document.getElementById("vendors-tbody");
    this.searchInput = document.getElementById("vendor-search-input");
    this.errorBanner = document.getElementById("vendors-error-banner");
    this.countLabel = document.getElementById("vendor-count-label");

    this.bindEvents();
  },

  bindEvents() {
    // Search filter input
    this.searchInput.addEventListener("input", () => this.render());

    // Sortable table headers
    document.querySelectorAll(".data-table th[data-sort]").forEach(th => {
      th.addEventListener("click", () => {
        const field = th.getAttribute("data-sort");
        if (this.sortField === field) {
          this.sortAsc = !this.sortAsc;
        } else {
          this.sortField = field;
          this.sortAsc = (field === "vendor_name" || field === "vendor_id");
        }
        this.updateSortHeaders();
        this.render();
      });
    });
  },

  async load() {
    this.errorBanner.classList.add("hidden");
    try {
      this.data = await api.vendors();
      this.render();
    } catch (err) {
      this.errorBanner.textContent = `Could not load vendor data: ${err.message}`;
      this.errorBanner.classList.remove("hidden");
    }
  },

  updateSortHeaders() {
    document.querySelectorAll(".data-table th[data-sort]").forEach(th => {
      const field = th.getAttribute("data-sort");
      const title = th.getAttribute("data-title") || th.textContent.replace(/[ ▲▼]/g, "");
      if (field === this.sortField) {
        th.textContent = `${title} ${this.sortAsc ? "▲" : "▼"}`;
      } else {
        th.textContent = title;
      }
    });
  },

  render() {
    const query = (this.searchInput.value || "").trim().toLowerCase();
    
    // Filter
    let filtered = this.data.filter(v => {
      const name = (v.vendor_name || "").toLowerCase();
      const id = (v.vendor_id || "").toLowerCase();
      return name.includes(query) || id.includes(query);
    });

    // Sort
    filtered.sort((a, b) => {
      let valA = a[this.sortField];
      let valB = b[this.sortField];

      if (typeof valA === "string") {
        valA = valA.toLowerCase();
        valB = (valB || "").toLowerCase();
        return this.sortAsc ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }

      valA = Number(valA || 0);
      valB = Number(valB || 0);
      return this.sortAsc ? valA - valB : valB - valA;
    });

    // Count label
    this.countLabel.textContent = `Showing ${filtered.length} of ${this.data.length} vendors`;

    // Render Table Rows
    this.tableBody.innerHTML = "";

    if (filtered.length === 0) {
      this.tableBody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align: center; color: var(--color-text-dim); padding: 30px;">
            No vendors match your search criteria.
          </td>
        </tr>`;
      return;
    }

    filtered.forEach(v => {
      const score = Number(v.risk_score || 0);
      let badgeClass = "badge-approved";
      if (score >= 0.35) {
        badgeClass = "badge-flagged";
      } else if (score >= 0.15) {
        badgeClass = "badge-pending";
      }

      const row = document.createElement("tr");
      row.innerHTML = `
        <td style="font-weight: 600;">${v.vendor_name || "Unknown Carrier"}</td>
        <td style="font-family: var(--font-mono); color: var(--color-primary); font-size: 0.88rem;">${v.vendor_id}</td>
        <td style="text-align: right; font-family: var(--font-mono);">${v.total_invoices.toLocaleString()}</td>
        <td style="text-align: right; font-family: var(--font-mono);">${v.flagged_invoices.toLocaleString()}</td>
        <td style="text-align: right;">
          <span class="badge ${badgeClass}" style="font-size: 0.78rem; padding: 3px 10px;">
            ${(score * 100).toFixed(1)}%
          </span>
        </td>
      `;
      this.tableBody.appendChild(row);
    });
  }
};

window.vendorsPage = vendorsPage;
document.addEventListener("DOMContentLoaded", () => vendorsPage.init());
