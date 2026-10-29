/**
 * router.js - Client-side single-page section router.
 * Switches between page sections without full browser reload.
 */

const router = {
  routes: {
    "#audit": { sectionId: "audit-page", title: "Invoice Audit", onShow: () => {} },
    "#dashboard": { sectionId: "dashboard-page", title: "Dashboard", onShow: () => window.dashboardPage?.load() },
    "#vendors": { sectionId: "vendors-page", title: "Carriers", onShow: () => window.vendorsPage?.load() },
    "#quote": { sectionId: "quote-page", title: "Freight Quote", onShow: () => { if (typeof quotePage !== "undefined") quotePage.onShow(); } },
  },

  defaultHash: "#audit",

  init() {
    window.addEventListener("hashchange", () => this.handleRoute());

    // Nav link click events
    document.querySelectorAll(".nav-link").forEach(link => {
      link.addEventListener("click", (e) => {
        const hash = link.getAttribute("href");
        if (hash && hash.startsWith("#")) {
          e.preventDefault();
          window.location.hash = hash;
        }
      });
    });

    // Mobile sidebar
    this.initMobileMenu();

    // Backend connection settings listener
    this.initBackendSettings();

    // Check system health on startup
    this.checkHealth();
    setInterval(() => this.checkHealth(), 60000);

    // Initial navigation
    this.handleRoute();
  },

  initBackendSettings() {
    const saveBtn = document.getElementById("save-backend-url-btn");
    const input = document.getElementById("backend-url-input");
    const banner = document.getElementById("backend-offline-banner");
    const healthContainer = document.getElementById("health-status-container");

    if (input) {
      input.value = localStorage.getItem("FREIGHT_API_BASE") || "";
    }

    const resetBtn = document.getElementById("reset-backend-url-btn");
    if (resetBtn && input) {
      resetBtn.addEventListener("click", async () => {
        localStorage.removeItem("FREIGHT_API_BASE");
        input.value = "https://freightcost.onrender.com";
        api.setBaseUrl("");
        resetBtn.disabled = true;
        resetBtn.textContent = "Resetting...";
        await this.checkHealth();
        resetBtn.disabled = false;
        resetBtn.textContent = "Reset to default";
      });
    }

    if (saveBtn && input) {
      saveBtn.addEventListener("click", async () => {
        const val = input.value.trim();
        api.setBaseUrl(val);
        saveBtn.disabled = true;
        saveBtn.textContent = "Connecting...";
        await this.checkHealth();
        saveBtn.disabled = false;
        saveBtn.textContent = "Connect";
      });
    }

    if (healthContainer && banner) {
      healthContainer.addEventListener("click", () => {
        banner.classList.toggle("hidden");
        const currentCode = document.getElementById("current-api-base");
        if (currentCode) currentCode.textContent = api.getBaseUrl();
      });
    }
  },

  initMobileMenu() {
    const sidebar = document.getElementById("sidebar");
    const btn = document.getElementById("menu-btn");
    if (!sidebar || !btn) return;
    const backdrop = document.createElement("div");
    backdrop.className = "sidebar-backdrop";
    document.body.appendChild(backdrop);
    const close = () => { sidebar.classList.remove("open"); backdrop.classList.remove("open"); };
    btn.addEventListener("click", () => { sidebar.classList.add("open"); backdrop.classList.add("open"); });
    backdrop.addEventListener("click", close);
    sidebar.querySelectorAll(".nav-link").forEach(a => a.addEventListener("click", close));
  },

  handleRoute() {
    let hash = window.location.hash || this.defaultHash;
    if (!this.routes[hash]) hash = this.defaultHash;

    // Toggle active section
    Object.keys(this.routes).forEach(r => {
      const section = document.getElementById(this.routes[r].sectionId);
      if (section) {
        if (r === hash) {
          section.classList.remove("hidden");
        } else {
          section.classList.add("hidden");
        }
      }
    });

    const titleEl = document.getElementById("topbar-title");
    if (titleEl) titleEl.textContent = this.routes[hash].title;
    document.title = `${this.routes[hash].title} | RateRight`;

    // Update active nav link
    document.querySelectorAll(".nav-link").forEach(link => {
      if (link.getAttribute("href") === hash) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });

    // Fire page lifecycle hook
    if (this.routes[hash].onShow) {
      this.routes[hash].onShow();
    }
  },

  async checkHealth() {
    const dot = document.getElementById("health-dot");
    const label = document.getElementById("health-label");
    const banner = document.getElementById("backend-offline-banner");
    const currentCode = document.getElementById("current-api-base");
    if (!dot || !label) return;

    if (currentCode) currentCode.textContent = api.getBaseUrl();

    try {
      const res = await api.health();
      if (res.status === "ok") {
        dot.className = "status-dot online";
        label.textContent = res.database_mode === "local_csv" ? "Online (sample data)" : "System online";
        if (banner) banner.classList.add("hidden");
      } else {
        dot.className = "status-dot offline";
        label.textContent = "Service degraded";
        if (banner) banner.classList.remove("hidden");
      }
    } catch (_) {
      dot.className = "status-dot offline";
      label.textContent = "Server offline";
      if (banner) banner.classList.remove("hidden");
    }
  }
};

document.addEventListener("DOMContentLoaded", () => router.init());
