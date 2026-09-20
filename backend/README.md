# FreightCost audit backend (FastAPI + Supabase)

## What this is

An HTTP layer around the existing, already-working pipeline. No pipeline
logic lives here — `audit_core.py` is a storage-agnostic refactor of
`confidence_gate_benchmark.py`'s gate + model-inference logic, and OCR
extraction / explanation generation are imported straight from `phase_a/`
unchanged. This file's only job is HTTP endpoints + persistence.

## Why the files are split this way

- **`audit_core.py`** — confidence gate, feature assembly, Model 1/2
  inference. Takes plain dicts and callables in, returns plain dicts out.
  No pandas-CSV-specific code, no Supabase-specific code — either caller
  decides where data actually lives.
- **`db.py`** — every network call to Supabase lives in this one file.
  `main.py` never imports the `supabase` client directly. This is what let
  the whole app get tested end-to-end (real OCR, real trained models, real
  rendered invoice image, over a real HTTP request) with a fake `Db` object
  and zero network access — this sandbox can't reach supabase.co, so a live
  connection couldn't be verified from here, but everything except the
  three-line Supabase client call inside `Db` has been.
- **`main.py`** — wires the two together into FastAPI endpoints.

## Setup

**1. Create a Supabase project** (free tier is plenty for this) at
[supabase.com](https://supabase.com), then in the SQL Editor run
`sql/schema.sql` once.

**2. Configure environment**
```bash
cp .env.example .env
# fill in SUPABASE_URL, SUPABASE_SERVICE_KEY (Project Settings > API),
# and a rotated GROQ_API_KEY if you want extractor=llm
```

**3. Migrate the existing dataset into Supabase** (one-time)
```bash
pip install -r requirements.txt
python migrate_to_supabase.py
```
*(The migration script automatically detects the data directory at `code/output`).*
This pushes `hubs`, `vendors`, `orders`, and the reference `invoices`/ground-
truth tables — everything the confidence gate and Model 1 need for
referential-integrity checks and the ERP join. `audit_log` and
`vendor_running_stats` start empty; the app writes those as it runs.

**4. Run the server**
```bash
uvicorn main:app --reload --port 8000
```

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/audit` | POST (multipart file) | Upload an invoice image → OCR → gate → Model 1/2 → explanation. Logs to `audit_log`, updates `vendor_running_stats`. |
| `/api/dashboard` | GET (`?n=300`) | Two-number report over the last N audits, aggregated from `audit_log`. |
| `/api/vendors` | GET | Vendor risk leaderboard from the `vendor_risk_scores` view. |
| `/api/health` | GET | Liveness check. |

## What "causal" means here, concretely

`vendor_running_stats` is only updated *after* an invoice is scored, never
before, and the risk score fed into that invoice's own scoring comes from
the vendor's stats as they stood *before* this invoice. That's what keeps
`vendor_historical_risk_score` genuinely causal — the same guarantee
`build_dataset.py`'s `add_vendor_risk()` had for the offline dataset, now
holding for a live, persistent, multi-invoice stream instead of a one-time
batch computation.

## Not yet wired (next steps)

- **`/api/quote`** (Phase D live shipment quoting) isn't built yet — this
  pass focused on the audit path. The hook point is `audit_core.Model1`,
  which already takes a plain order dict; a quote endpoint just needs to
  build that dict from a form instead of a Supabase row, plus real
  `run_ors.py` / weather-forecast calls for distance and conditions instead
  of a lookup.
- **LLM extractor path** works but wasn't exercised in testing here — it
  needs your rotated `GROQ_API_KEY`, which this sandbox doesn't have.
- **Frontend** — this backend has no UI yet. Streamlit or a custom
  HTML/JS frontend both just call these four endpoints.
