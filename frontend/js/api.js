/**
 * api.js - Centralized API access layer.
 * All HTTP communication with the FastAPI backend flows through this file.
 */

const PROD_API_BASE = "https://freightcost.onrender.com";

// Dynamic API base resolution
function getApiBase() {
  const custom = localStorage.getItem("FREIGHT_API_BASE");
  if (custom && custom.trim().startsWith("http")) {
    return custom.trim().replace(/\/+$/, "");
  }
  const isLocal = typeof window !== "undefined" && window.location && (
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.port === "8000" ||
    window.location.port === "3000" ||
    window.location.port === "5500"
  );
  if (isLocal) {
    return "http://localhost:8000";
  }
  return PROD_API_BASE;
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
      const rawText = await res.text();
      try {
        const d = JSON.parse(rawText).detail;
        if (typeof d === "string") errDetail = d;
        else if (d && d.message) errDetail = d.message + (d.missing_fields ? ` (${d.missing_fields.join(", ")})` : "");
        else if (d) errDetail = JSON.stringify(d);
      } catch (_) {
        if (rawText) errDetail = rawText.slice(0, 300);
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
  extract: (file, extractor = "auto") => {
    const form = new FormData();
    form.append("file", file);
    return apiCall(`/api/extract?extractor=${encodeURIComponent(extractor)}`, { method: "POST", body: form });
  },

  /** Re-validate user-edited fields (order exists, vendor matches, line items add up). */
  check: (payload) => apiCall("/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }),

  /** Run gate + Model 1 + Model 2 on confirmed fields. */
  score: (payload) => apiCall("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }),

  extractors: () => apiCall("/api/extractors"),

  /** Live diesel price + Open-Meteo weather for a route and date. */
  context: (origin, dest, shipDate, transitDays) => {
    const q = new URLSearchParams({ origin_hub_id: origin, dest_hub_id: dest });
    if (shipDate) q.set("ship_date", shipDate);
    if (transitDays) q.set("transit_days", transitDays);
    return apiCall(`/api/context?${q}`);
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
