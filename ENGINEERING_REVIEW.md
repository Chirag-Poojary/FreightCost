# Engineering Review

A full review of the RateRight codebase (backend, models, frontend and deployment), done before making the app ready for business and MSME users. Each problem is listed with its severity and whether it has been fixed.

Severity: **High** = wrong results, security or data loss. **Medium** = misleading output or a broken feature. **Low** = polish, maintainability or performance.

---

## Summary

| | Count |
|---|---|
| Problems found | 34 |
| Fixed in this change | 25 |
| Open (need a decision, credentials or real data) | 9 |

---

## Fixed

### Prediction correctness

| # | Severity | Problem | Fix | Files |
|---|---|---|---|---|
| 1 | High | `commercial_delay_days` was clipped at 0 when scoring, but Model 2 was trained on the raw value (35% of training invoices are negative, meaning early delivery). Every prediction used a different input from training. | Send the raw value as the feature; use the clipped value only inside the two derived features, exactly as in training. | `backend/audit_core.py` |
| 2 | High | New orders (not in the order records) got `vendor_padding_ratio = 0`. Training values are around 1.16, so 0 was out of range for Model 2. | Use the carrier's median padding ratio from past invoices, or the overall median (1.164) for an unknown carrier. | `backend/main.py`, `backend/db.py` |
| 3 | Medium | New orders always used a weather score of 0 for Model 1, even though live weather was available. | Measure the trip-window weather from Open-Meteo and use it. | `backend/main.py` |
| 4 | Medium | New orders were always treated as "Automotive Spare Parts", whatever the goods were. | Added an optional "Goods" field (dropdown of the five trained categories); the reading model also tries to fill it. The default is used only when it is left empty. | `backend/extractors.py`, `backend/main.py`, `frontend/js/audit.js` |
| 5 | Medium | In CSV mode, carrier IDs in mixed case (`ven001` and `VEN001`) were counted as separate carriers, splitting their history and risk scores. Training already upper-cases them. | Upper-case carrier IDs when loading the CSV data. | `backend/db.py` |
| 6 | Medium | The carrier list in CSV mode used a different risk formula, `(flagged + 1) / (total + 2)`, from the one the model uses, `(flagged + 0.12) / (total + 2)`. | Use the same formula everywhere. | `backend/db.py` |
| 7 | Low | Free-text truck types such as "10 Wheeler" were not recognised by the model. | Normalise truck type and goods category to the values the model knows. | `backend/extractors.py` |

### Backend reliability and security

| # | Severity | Problem | Fix | Files |
|---|---|---|---|---|
| 8 | High | Uploads were not checked on the server: any file type and any size was accepted and written to disk. | Server checks the size (10 MB limit, HTTP 413) and that the file really is a PNG, JPG or WEBP image (HTTP 415). | `backend/main.py` |
| 9 | Medium | Unexpected errors sent the raw Python exception text to the browser, exposing internals. | Errors are logged on the server with a short reference code; the user sees a plain message with that code. | `backend/main.py` |
| 10 | Medium | Supabase order lookup used `.single()`, which throws when an order does not exist, so every new order logged an error and fell back to CSV. | Use `limit(1)` and return nothing when there is no row. | `backend/db.py` |
| 11 | Low | The "New carrier" note was added and then wiped by a second `ctx_notes = []`, so users never saw it. | Removed the second reset; new-order and new-carrier notes now appear in the result. | `backend/main.py` |
| 12 | Low | `/api/health` said "connected" even when the server had silently fallen back to CSV files. | Health now reports `database_mode` (`supabase` or `local_csv`); the UI shows "Online (sample data)" and warns that audit history resets on restart. | `backend/main.py`, `backend/db.py`, `frontend/js/router.js`, `frontend/js/dashboard.js` |
| 13 | Low | Model loading used the deprecated `@app.on_event("startup")`. | Moved to a FastAPI lifespan handler. | `backend/main.py` |
| 14 | Low | Price warnings were written for developers (`expected_fuel_price = 97.83 is outside the training range...`). | Plain-language warning for business users. | `backend/live_context.py` |

### Frontend

| # | Severity | Problem | Fix | Files |
|---|---|---|---|---|
| 15 | Medium | Carrier names were inserted into the page as raw HTML (a script-injection risk if a carrier name contains HTML). | All server data is escaped before display, including the new dashboard table. | `frontend/js/vendors.js`, `frontend/js/dashboard.js` |
| 16 | Medium | Truck type, origin and destination were free-text fields, so typos produced values the model does not know. | Truck and goods are dropdowns; origin, destination and carrier ID offer a list of known values while still allowing new ones. Dates use the calendar picker. | `frontend/js/audit.js` |
| 17 | Medium | The Quote page let users pick a truck that does not match the weight (for example a 6-wheeler for 20 t), unlike the training data. | Truck is chosen automatically from the weight, with a hint if the user overrides it. | `frontend/js/quote.js` |
| 18 | Medium | The demo button fetched a file that Vercel does not serve, and there was no easy way for a new user to try the app. | A gallery of five sample invoices on the first page (clean, overbilled, missing field, phone photo), each with a thumbnail and a View link. | `frontend/index.html`, `frontend/js/audit.js`, `frontend/demo/` |
| 19 | Medium | For a new order, the optional trip details (which drive the estimate) were hidden and the summary said "in the order master". | The section opens automatically and the summary is labelled "New order", with an explanation. | `frontend/js/audit.js` |
| 20 | Low | The Quote date defaulted to yesterday between midnight and 05:30 IST (`toISOString()` is UTC). | Use the local date. | `frontend/js/quote.js` |
| 21 | Low | Invalid CSS (`padding: var(--space-unit) * 3`) in the dashboard empty state. | Rewritten. | `frontend/js/dashboard.js` |
| 22 | Low | Canonical and social-preview links pointed to `freightcost-ai.vercel.app`, which is not the app's domain. | Point to `freight-cost-nine.vercel.app`. | `frontend/index.html` |
| 23 | Low | Dark theme with emoji icons and technical wording did not suit business users. | Light, ERP-style layout: sidebar navigation, one primary colour, SVG icons, plain business wording, RATERIGHT branding, site icon, working mobile menu. | `frontend/index.html`, `frontend/css/styles.css`, `frontend/favicon.svg` |
| 24 | Low | The dashboard showed only totals, with no list of individual audits. | Added a "Recent audits" table backed by a new `/api/audits/recent` endpoint. | `frontend/js/dashboard.js`, `backend/main.py` |

### Testing

| # | Severity | Problem | Fix | Files |
|---|---|---|---|---|
| 25 | Medium | There were no automated tests, so changes could break scoring silently. | Added 11 API tests: health, approved invoice, missing field, line-item mismatch, too-low bill, new order, impossible quote input, diesel scaling, non-image upload, carrier-ID case, recent audits. | `tests/test_api.py` |

---

## Open (not fixed in this change)

| # | Severity | Problem | Why it is open | Suggested next step |
|---|---|---|---|---|
| 26 | High | The API has no login or API key. Anyone who knows the URL can run audits, which also changes carrier risk statistics and the audit log. CORS allows every origin. | Needs a decision on how companies will sign in. | Add user accounts (Supabase Auth) or an API key per company; restrict CORS to the frontend domain. |
| 27 | High | All companies share one set of orders, carriers and audit history. | Needs a multi-company data design. | Add a `company_id` to every table and filter by it. |
| 28 | Medium | Carrier statistics are updated with read-then-write, so two audits for the same carrier at the same moment can lose a count. | Needs a database function. | Use an atomic Supabase RPC (`total = total + 1`). |
| 29 | Medium | Live diesel and weather values are cached only in memory (6 h and 3 h) and are lost on restart; they are never saved for retraining. Training data ends in 2025. | Needs new tables and a scheduled job. | Save daily diesel and weather to Supabase and retrain monthly. |
| 30 | Medium | The training orders and invoices are generated (only weather and diesel are real). | Needs real company data. | Retrain on a pilot partner's invoices before production use. |
| 31 | Medium | Default model names `qwen/qwen3.8-27b` (Groq vision) and `gemini-3.8-flash` could not be verified against the providers' current model lists. The automatic fallback hides failures, and the Gemini key was over its free quota during testing. | Needs the API keys to check. | Confirm the model IDs in each provider's console and update `render.yaml`. |
| 32 | Low | No rate limiting on the AI extraction endpoint, so a burst of uploads can exhaust the AI quota. | Depends on #26. | Rate-limit per user once sign-in exists. |
| 34 | High | The live backend on Render reports `database_mode: local_csv`: Supabase is not configured there or cannot connect, so audit history and carrier statistics are kept in memory and lost whenever the free server restarts. The old health check hid this by always saying "connected". | Needs the Supabase credentials on Render. | Set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` on Render, run `backend/schema.sql` and `backend/migrate_to_supabase.py`, then confirm `/api/health` shows `supabase`. |
| 33 | Low | Live diesel prices are scraped from a public web page, which can change its layout without notice. | No free official API was found. | Use a paid or official price feed for production. |

---

## How this was checked

- 11 API tests pass: `SUPABASE_URL= SUPABASE_SERVICE_KEY= python -m pytest tests -q`
- Browser walk-through with Playwright and Chrome: sample gallery, manual review form, an approved and a flagged invoice, a new order with a missing field, dashboard, carriers, quote, and the mobile layout. No console errors apart from the expected "no reading model configured" when running locally without API keys.
