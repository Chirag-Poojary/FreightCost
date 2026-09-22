/**
 * api.js - Centralized API access layer.
 * All HTTP communication with the FastAPI backend flows through this file.
 */

// Dynamic API base resolution
function getApiBase() {
  const custom = localStorage.getItem("FREIGHT_API_BASE");
  if (custom && custom.trim().startsWith("http")) {
    return custom.trim().replace(/\/+$/, "");
  }
  return window.location.origin && window.location.origin.startsWith("http")
    ? window.location.origin
    : "http://localhost:8000";
}

function setApiBase(url) {
  if (url && url.trim().startsWith("http")) {
    const clean = url.trim().replace(/\/+$/, "");
    localStorage.setItem("FREIGHT_API_BASE", clean);
  } else {
    localStorage.removeItem("FREIGHT_API_BASE");
  }
  return getApiBase();
}

async function apiCall(path, options = {}) {
  const base = getApiBase();
  const url = `${base}${path}`;
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let errDetail = `${res.status} ${res.statusText}`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || JSON.stringify(errJson);
      } catch (_) {
        const rawText = await res.text();
        if (rawText) errDetail = rawText;
      }
      throw new Error(errDetail);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Call failed: ${path}`, err);
    throw err;
  }
}

const api = {
  /**
   * Post an invoice file to the audit pipeline
   */
  audit: (file, extractor = "rules", provider = "groq") => {
    const form = new FormData();
    form.append("file", file);
    return apiCall(
      `/api/audit?extractor=${encodeURIComponent(extractor)}&provider=${encodeURIComponent(provider)}`,
      { method: "POST", body: form }
    );
  },

  /**
   * Fetch aggregate audit analytics
   */
  dashboard: (n = 300) => apiCall(`/api/dashboard?n=${encodeURIComponent(n)}`),

  /**
   * Fetch real-time vendor risk leaderboard
   */
  vendors: () => apiCall("/api/vendors"),

  /**
   * Calculate pre-shipment cost estimate via Model 1
   */
  quote: (payload) => apiCall("/api/quote", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }),

  /**
   * Liveness and database connectivity health check
   */
  health: () => apiCall("/api/health"),

  getBaseUrl: () => getApiBase(),
  setBaseUrl: (url) => setApiBase(url),
};
