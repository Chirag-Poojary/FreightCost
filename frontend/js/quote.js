/**
 * quote.js - Pre-shipment freight cost estimate (Model 1).
 * Picking a route + date auto-fills distance and ideal days from the route
 * cache, and diesel price + weather severity from live sources (/api/context).
 * Every auto-filled value can still be overridden by hand.
 */

const HUBS = [
  { id: "HUB01", name: "Mumbai (Maharashtra)" },
  { id: "HUB02", name: "Pune (Maharashtra)" },
  { id: "HUB03", name: "Ahmedabad (Gujarat)" },
  { id: "HUB04", name: "Mundra (Gujarat)" },
  { id: "HUB05", name: "Delhi (Delhi NCR)" },
  { id: "HUB06", name: "Gurugram (Haryana)" },
  { id: "HUB07", name: "Ludhiana (Punjab)" },
  { id: "HUB08", name: "Jaipur (Rajasthan)" },
  { id: "HUB09", name: "Kanpur (Uttar Pradesh)" },
  { id: "HUB10", name: "Kolkata (West Bengal)" },
  { id: "HUB11", name: "Chennai (Tamil Nadu)" },
  { id: "HUB12", name: "Coimbatore (Tamil Nadu)" },
  { id: "HUB13", name: "Bengaluru (Karnataka)" },
  { id: "HUB14", name: "Hyderabad (Telangana)" },
  { id: "HUB15", name: "Nagpur (Maharashtra)" },
  { id: "HUB16", name: "Indore (Madhya Pradesh)" },
  { id: "HUB17", name: "Visakhapatnam (Andhra Pradesh)" },
  { id: "HUB18", name: "Surat (Gujarat)" },
];

const quotePage = {
  ctxTimer: null,
  ctxSeq: 0,

  init() {
    const $ = id => document.getElementById(id);
    Object.assign(this, {
      form: $("quote-form"), originSelect: $("quote-origin"), destSelect: $("quote-dest"),
      dateInput: $("quote-date"), submitBtn: $("quote-submit-btn"), btnSpinner: $("quote-spinner"),
      btnText: $("quote-btn-text"), resultCard: $("quote-result-card"), costDisplay: $("quote-cost-display"),
      errorBanner: $("quote-error-banner"), liveBox: $("quote-live-box"), driftBox: $("quote-drift-box"),
    });

    HUBS.forEach(h => {
      this.originSelect.add(new Option(`${h.id} - ${h.name}`, h.id));
      this.destSelect.add(new Option(`${h.id} - ${h.name}`, h.id));
    });
    this.originSelect.value = "HUB01";
    this.destSelect.value = "HUB05";
    this.dateInput.value = new Date().toISOString().slice(0, 10);

    this.form.addEventListener("submit", e => { e.preventDefault(); this.calculateQuote(); });
    [this.originSelect, this.destSelect, this.dateInput].forEach(el =>
      el.addEventListener("change", () => this.scheduleContext()));
    document.getElementById("quote-quoted-days").addEventListener("change", () => this.scheduleContext());
    // A hand-edited value is no longer "live" -- say so next to the label.
    [["quote-distance", "src-distance"], ["quote-ideal-days", "src-ideal"],
     ["quote-fuel-price", "src-fuel"], ["quote-weather", "src-weather"]].forEach(([inp, tag]) =>
      document.getElementById(inp).addEventListener("input", () => { document.getElementById(tag).textContent = "edited"; }));
    if (location.hash === "#quote") this.scheduleContext(0);
  },

  // Called by the router the first time the page is shown.
  onShow() {
    if (this.form && !this.contextLoaded) this.scheduleContext(0);
  },

  scheduleContext(delay = 350) {
    clearTimeout(this.ctxTimer);
    this.ctxTimer = setTimeout(() => this.loadContext(), delay);
  },

  async loadContext() {
    const origin = this.originSelect.value, dest = this.destSelect.value;
    if (origin === dest) {
      this.liveBox.textContent = "Origin and destination are the same hub. Pick two different hubs.";
      return;
    }
    const seq = ++this.ctxSeq;
    this.liveBox.textContent = "Loading live diesel price and weather…";
    const quoted = parseFloat(document.getElementById("quote-quoted-days").value) || undefined;
    try {
      const ctx = await api.context(origin, dest, this.dateInput.value, quoted);
      if (seq !== this.ctxSeq) return;        // a newer request superseded this one
      this.contextLoaded = true;
      const set = (id, tag, v, label) => {
        if (v === null || v === undefined) return;
        document.getElementById(id).value = v;
        document.getElementById(tag).textContent = label;
      };
      if (ctx.route) {
        set("quote-distance", "src-distance", ctx.route.distance_km, "route cache");
        set("quote-ideal-days", "src-ideal", ctx.route.ideal_days, "route cache");
      }
      if (ctx.fuel?.ok) set("quote-fuel-price", "src-fuel", ctx.fuel.price, ctx.fuel.source.startsWith("PPAC") ? "PPAC" : "live");
      if (ctx.weather?.ok) set("quote-weather", "src-weather", ctx.weather.score, "Open-Meteo");

      const w = ctx.weather || {};
      const f = ctx.fuel || {};
      this.liveBox.innerHTML = `
        <div><strong>Diesel (${escapeHtml(f.state || "")}):</strong> ${f.ok ? `₹${Number(f.price).toFixed(2)}/L as of ${escapeHtml(f.as_of)}` : "unavailable"} · <span>${escapeHtml(f.source || "")}</span></div>
        <div style="margin-top:4px;"><strong>Weather at destination, ${escapeHtml(w.start || "")} to ${escapeHtml(w.end || "")}:</strong>
          ${w.ok ? `${w.rain_mm_total} mm rain, max wind ${w.max_wind_kmh} km/h, max ${w.max_temp_c}°C → severity ${w.score}` : "unavailable"} · <span>${escapeHtml(w.source || "")}</span></div>`;
      this.showDrift(ctx.drift_warnings);
    } catch (err) {
      if (seq !== this.ctxSeq) return;
      this.liveBox.textContent = `Live data unavailable (${err.message}). Enter diesel price and weather by hand.`;
    }
  },

  showDrift(warnings) {
    if (warnings?.length) {
      this.driftBox.innerHTML = "<strong>Data drift:</strong> " + warnings.map(escapeHtml).join("<br>");
      this.driftBox.classList.remove("hidden");
    } else {
      this.driftBox.classList.add("hidden");
    }
  },

  async calculateQuote() {
    this.errorBanner.classList.add("hidden");
    if (this.originSelect.value === this.destSelect.value) {
      this.errorBanner.textContent = "Origin and destination must be different hubs.";
      this.errorBanner.classList.remove("hidden");
      return;
    }
    this.submitBtn.disabled = true;
    this.btnSpinner.classList.remove("hidden");
    this.btnText.textContent = "Estimating…";

    const num = id => parseFloat(document.getElementById(id).value);
    const payload = {
      origin_hub_id: this.originSelect.value,
      dest_hub_id: this.destSelect.value,
      truck_type: document.getElementById("quote-truck-type").value,
      product_category: document.getElementById("quote-category").value,
      billable_weight_kg: num("quote-weight"),
      distance_km: num("quote-distance"),
      ideal_days: num("quote-ideal-days"),
      quoted_days: num("quote-quoted-days"),
      expected_fuel_price: num("quote-fuel-price"),
      expected_weather_score: num("quote-weather"),
    };

    try {
      const res = await api.quote(payload);
      const fmt = v => Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 });
      this.costDisplay.textContent = "₹" + Number(res.predicted_cost || 0).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      const intervalEl = document.getElementById("quote-interval-display");
      intervalEl.textContent = res.interval_lower != null ? `80% range: ₹${fmt(res.interval_lower)} – ₹${fmt(res.interval_upper)}` : "";

      const driversCard = document.getElementById("quote-drivers-card");
      const driversList = document.getElementById("quote-drivers-list");
      driversList.innerHTML = "";
      const rows = res.cost_breakdown || res.top_drivers || [];
      rows.forEach(d => {
        const row = document.createElement("div");
        row.className = "label-row";
        const name = d.feature.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
        row.innerHTML = `<span>${escapeHtml(name)}</span><span style="color:${d.feature !== "model_adjustment" ? "var(--color-text)" : d.impact > 0 ? "var(--color-flagged-text)" : "var(--color-approved-text)"};font-weight:600;">${d.impact > 0 ? "+" : "−"}₹${fmt(Math.abs(d.impact))}</span>`;
        driversList.appendChild(row);
      });
      driversCard.classList.toggle("hidden", !rows.length);
      const warn = [...(res.drift_warnings || [])];
      if (res.price_check && res.price_check.status !== "ok") warn.unshift(`Price logic check (${res.price_check.status.replace("_", " ")}): ${res.price_check.message}`);
      this.showDrift(warn);

      this.resultCard.classList.remove("hidden");
      this.resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      this.errorBanner.textContent = `Cost estimation failed: ${err.message}`;
      this.errorBanner.classList.remove("hidden");
    } finally {
      this.submitBtn.disabled = false;
      this.btnSpinner.classList.add("hidden");
      this.btnText.textContent = "Calculate Freight Estimate";
    }
  },
};

document.addEventListener("DOMContentLoaded", () => quotePage.init());
