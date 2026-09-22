/**
 * api.js - Centralized API access layer.
 * All HTTP communication with the FastAPI backend flows through this file.
 */

// If served via FastAPI StaticFiles, relative origin works directly.
// Otherwise fallback to local dev port 8000.
const API_BASE = window.location.origin && window.location.origin.startsWith("http")
  ? window.location.origin
  : "http://localhost:8000";

async function apiCall(path, options = {}) {
  const url = `${API_BASE}${path}`;
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
};
