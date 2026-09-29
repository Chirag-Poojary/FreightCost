/**

 * quote.js - Pre-shipment freight cost estimation page (Model 1).

 * Populates 18 logistics hubs, handles form validation, and calls /api/quote.

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

  init() {

    this.form = document.getElementById("quote-form");

    this.originSelect = document.getElementById("quote-origin");

    this.destSelect = document.getElementById("quote-dest");

    this.submitBtn = document.getElementById("quote-submit-btn");

    this.btnSpinner = document.getElementById("quote-spinner");

    this.btnText = document.getElementById("quote-btn-text");

    this.resultCard = document.getElementById("quote-result-card");

    this.costDisplay = document.getElementById("quote-cost-display");

    this.errorBanner = document.getElementById("quote-error-banner");



    this.populateHubs();

    this.bindEvents();

  },



  populateHubs() {

    HUBS.forEach(h => {

      const opt1 = new Option(`${h.id} - ${h.name}`, h.id);

      const opt2 = new Option(`${h.id} - ${h.name}`, h.id);

      this.originSelect.add(opt1);

      this.destSelect.add(opt2);

    });



    // Default origin/dest selections

    this.originSelect.value = "HUB01"; // Mumbai

    this.destSelect.value = "HUB05";   // Delhi

  },



  bindEvents() {

    this.form.addEventListener("submit", (e) => {

      e.preventDefault();

      this.calculateQuote();

    });

  },



  async calculateQuote() {

    this.errorBanner.classList.add("hidden");

    this.submitBtn.disabled = true;

    this.btnSpinner.classList.remove("hidden");

    this.btnText.textContent = "Estimating...";



    const payload = {

      origin_hub_id: this.originSelect.value,

      dest_hub_id: this.destSelect.value,

      truck_type: document.getElementById("quote-truck-type").value,

      product_category: document.getElementById("quote-category").value,

      billable_weight_kg: parseFloat(document.getElementById("quote-weight").value),

      distance_km: parseFloat(document.getElementById("quote-distance").value),

      ideal_days: parseFloat(document.getElementById("quote-ideal-days").value),

      quoted_days: parseFloat(document.getElementById("quote-quoted-days").value),

      expected_fuel_price: parseFloat(document.getElementById("quote-fuel-price").value),

      expected_weather_score: parseFloat(document.getElementById("quote-weather").value),

    };



    try {

      const res = await api.quote(payload);

      const cost = Number(res.predicted_cost || 0);

      this.costDisplay.textContent = "₹" + cost.toLocaleString("en-IN", {

        minimumFractionDigits: 2,

        maximumFractionDigits: 2,

      });



      const intervalEl = document.getElementById("quote-interval-display");
      if (intervalEl && res.interval_lower !== undefined && res.interval_upper !== undefined) {
        const lo = Number(res.interval_lower).toLocaleString("en-IN", { maximumFractionDigits: 0 });
        const hi = Number(res.interval_upper).toLocaleString("en-IN", { maximumFractionDigits: 0 });
        intervalEl.textContent = `80% Credible Range: ₹${lo} – ₹${hi}`;
      }

      const driversCard = document.getElementById("quote-drivers-card");
      const driversList = document.getElementById("quote-drivers-list");
      if (driversCard && driversList && res.top_drivers && res.top_drivers.length > 0) {
        driversList.innerHTML = "";
        res.top_drivers.forEach(d => {
          const row = document.createElement("div");
          row.style.display = "flex";
          row.style.justifyContent = "space-between";
          row.style.alignItems = "center";
          row.style.fontSize = "0.86rem";
          const sign = d.impact > 0 ? "+" : "";
          const color = d.impact > 0 ? "#f87171" : "#4ade80";
          const name = d.feature.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase());
          row.innerHTML = `<span style="color: var(--color-text);">${name}</span><span style="color: ${color}; font-weight: 600;">${sign}₹${Math.abs(d.impact).toLocaleString("en-IN", {maximumFractionDigits: 0})}</span>`;
          driversList.appendChild(row);
        });
        driversCard.classList.remove("hidden");
      }

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

  }

};



document.addEventListener("DOMContentLoaded", () => quotePage.init());

