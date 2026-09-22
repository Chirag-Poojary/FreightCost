/**
 * router.js - Client-side single-page section router.
 * Switches between page sections without full browser reload.
 */

const router = {
  routes: {
    "#audit": { sectionId: "audit-page", onShow: () => {} },
    "#dashboard": { sectionId: "dashboard-page", onShow: () => window.dashboardPage?.load() },
    "#vendors": { sectionId: "vendors-page", onShow: () => window.vendorsPage?.load() },
    "#quote": { sectionId: "quote-page", onShow: () => {} },
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

    // Backend connection settings listener
    this.initBackendSettings();

    // Check system health on startup
    this.checkHealth();
    setInterval(() => this.checkHealth(), 15000);

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
        label.textContent = res.database === "connected" ? "Engine Online" : "DB Disconnected";
        if (banner) banner.classList.add("hidden");
      } else {
        dot.className = "status-dot offline";
        label.textContent = "Engine Degraded";
        if (banner) banner.classList.remove("hidden");
      }
    } catch (_) {
      dot.className = "status-dot offline";
      label.textContent = "Backend Offline";
      if (banner) banner.classList.remove("hidden");
    }
  }
};

document.addEventListener("DOMContentLoaded", () => router.init());
