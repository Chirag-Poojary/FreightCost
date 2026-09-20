# FreightCost Audit Backend Guide (FastAPI + Supabase)

Comprehensive documentation for deploying, executing, and independently maintaining the **FreightCost Audit Backend** powered by **FastAPI** and **Supabase (PostgreSQL)**.

---

## 1. Architectural Overview

The backend operates as an API service that connects OCR document extraction, predictive cost modeling (Model 1), confidence gating, fraud/anomaly classification (Model 2), and causal explanation generation with a persistent PostgreSQL database hosted on Supabase.

```
+---------------------------------------------------------------------------------------------------+
|                                          FASTAPI ROUTING                                          |
|                                         (backend/main.py)                                         |
+---------------------------------------------------------------------------------------------------+
        |                                       |                                    |
        v                                       v                                    v
+-------------------+                 +-------------------+                +-------------------+
|    OCR LAYER      |                 |    AUDIT CORE     |                |  DATABASE LAYER   |
| (ocr_extract.py)  |                 |  (audit_core.py)  |                |     (db.py)       |
+-------------------+                 +-------------------+                +-------------------+
  - Tesseract OCR                       - 4-Tier Gate Check                  - Supabase PostgREST
  - Regex Extractor                     - Model 1 Inference                  - Referential Checks
  - LLM Fallback                        - Model 2 Inference                  - Causal Risk Calc
                                        - Explanations                       - Audit Event Log
                                                                                       |
                                                                                       v
                                                                           +-------------------+
                                                                           |     SUPABASE      |
                                                                           |   (PostgreSQL)    |
                                                                           +-------------------+
```

### Key Components

1. **[`backend/audit_core.py`](backend/audit_core.py)**: Completely storage-agnostic. Implements the 4-tier confidence validation gate, Model 1 wrapper, Model 2 feature builder, causal Laplace vendor risk calculation, and model scoring. Operates exclusively on pure Python dictionaries without database or filesystem coupling.
2. **[`backend/db.py`](backend/db.py)**: Encapsulates all network communication with Supabase. Contains single-point queries for referential validation, historical context retrieval, audit log insertion, and atomic vendor risk counters.
3. **[`backend/main.py`](backend/main.py)**: Exposes REST API endpoints, loads ML models (`model_1_freight_cost.joblib` and `model_2_invoice_risk.joblib`) into memory on startup, manages CORS, and coordinates the end-to-end audit lifecycle.
4. **[`backend/schema.sql`](backend/schema.sql)**: DDL defining PostgreSQL tables, indexes, and analytical views for both static reference data and dynamic operational audit trails.
5. **[`backend/migrate_to_supabase.py`](backend/migrate_to_supabase.py)**: Resilient batch migration script that loads pre-generated baseline CSVs into Supabase.

---

## 2. Supabase Requirements & Credentials

To run this backend with live persistence, you need a free Supabase project.

### Credentials Required in `backend/.env`

| Variable Name | Required | Description | Where to Find in Supabase Dashboard |
|---|---|---|---|
| `SUPABASE_URL` | **Yes** | Project API URL (`https://<project-id>.supabase.co`) | **Project Settings** $\rightarrow$ **API** $\rightarrow$ **Project URL** |
| `SUPABASE_SERVICE_KEY` | **Yes** | Secret `service_role` JWT API key | **Project Settings** $\rightarrow$ **API** $\rightarrow$ **Project API keys** $\rightarrow$ `service_role` (secret) |
| `GROQ_API_KEY` | Optional | Groq Cloud API key (for `extractor=llm`) | [Groq Cloud Console](https://console.groq.com/keys) |

> [!IMPORTANT]
> **Why `SUPABASE_SERVICE_KEY` instead of `anon` / public key?**
> The backend runs strictly server-side. It writes system-level audit logs to `audit_log` and modifies running counters in `vendor_running_stats`. The `service_role` key bypasses PostgreSQL Row-Level Security (RLS) policies, ensuring backend microservices can write telemetry and audit events reliably. **Never expose `SUPABASE_SERVICE_KEY` to client-side code or public repositories.**

---

## 3. Step-by-Step Setup Guide

### Step 1: Create a Supabase Project

1. Navigate to [https://supabase.com](https://supabase.com) and log in.
2. Click **New Project**.
3. Choose your organization, assign a project name (e.g. `freightcost-audit`), set a strong database password, and select the region closest to you.
4. Wait approximately 1-2 minutes for project provisioning to complete.

---

### Step 2: Deploy Database Schema via Supabase SQL Editor

1. In your Supabase project dashboard, click on the **SQL Editor** icon in the left navigation sidebar.
2. Click **+ New Query**.
3. Open [`backend/schema.sql`](backend/schema.sql) in your code editor, copy its entire contents, and paste it into the Supabase SQL Editor.
4. Click **Run** (or press `Ctrl` + `Enter`).
5. Verify in the **Table Editor** that the following tables and views have been created:
   - **Reference Tables**: `hubs`, `vendors`, `orders`, `reference_invoices`, `ground_truth_audit`
   - **Operational Tables**: `audit_log`, `vendor_running_stats`
   - **Analytical Views**: `vendor_risk_scores`

---

### Step 3: Configure Environment Variables

Create a file named `.env` inside the `backend/` directory (you can copy `backend/.env.example`):

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your actual credentials:

```env
# Supabase Project Configuration
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_KEY=your_service_role_secret_key_here

# LLM Extractor & Explanation Key (Optional - needed if calling extractor="llm")
GROQ_API_KEY=your_groq_api_key_here
```

*(Note: `backend/.env` is ignored by git in `.gitignore` to prevent secret leaks).*

---

### Step 4: Migrate Baseline Data to Supabase

Before auditing new incoming invoices, populate the reference tables (`hubs`, `vendors`, `orders`, `reference_invoices`, `ground_truth_audit`) from the project's generated CSV datasets.

Activate your Python virtual environment and execute the migration script:

```powershell
# From the repository root (d:\chirag\bundle):
.\.venv\Scripts\Activate.ps1
python backend/migrate_to_supabase.py
```

Or from within `backend/`:

```powershell
cd backend
python migrate_to_supabase.py
```

The script automatically locates `code/output/*.csv`, streams rows in chunks of 500 to respect PostgREST payload limits, and performs idempotent upserts.

#### Expected Verification Output:
```text
Using data directory: D:\chirag\bundle\code\output
hubs...
  hubs: 18/18
vendors...
  vendors: 25/25
orders...
  orders: 500/60000
  ...
  orders: 60000/60000
reference_invoices...
  reference_invoices: 500/60000
  ...
  reference_invoices: 60000/60000
ground_truth_audit...
  ground_truth_audit: 500/60000
  ...
  ground_truth_audit: 60000/60000

Done. Row counts:
  hubs: 18
  vendors: 25
  orders: 60000
  reference_invoices: 60000
  ground_truth_audit: 60000
```

---

### Step 5: Launch the FastAPI Backend Server

Start the Uvicorn development server:

```powershell
# From the repository root:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Or from the `backend/` folder:

```powershell
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal Startup Log:
```text
INFO:     Will watch for changes in ['...']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Visit the interactive API documentation at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 4. API Endpoints Reference

### 1. Health & Readiness Check
- **Endpoint**: `GET /api/health`
- **Description**: Verifies service status, confirms ML models are loaded in memory, and validates database connectivity.
- **Example Response**:
  ```json
  {
    "status": "ok",
    "models_loaded": true,
    "database": "connected"
  }
  ```

---

### 2. Audit Invoice
- **Endpoint**: `POST /api/audit`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (*UploadFile*, required): Invoice image (`.png`, `.jpg`, `.jpeg`).
  - `extractor` (*str*, optional, default: `"rules"`): Extraction engine (`"rules"` or `"llm"`).
  - `provider` (*str*, optional, default: `"groq"`): Provider if using LLM extractor (`"groq"`, `"together"`, etc.).
- **Workflow Executed**:
  1. Image OCR processing via Tesseract.
  2. Field parsing (`order_id`, `vendor_id`, `total`, `actual_days`, line items).
  3. **Confidence Gate Evaluation** (4-tier validation):
     - *Tier 1*: Critical fields present.
     - *Tier 2*: Referential integrity check (`orders` & `vendors` exist in Supabase).
     - *Tier 3*: Arithmetic checksum consistency ($\sum \text{items} \approx \text{total}$).
     - *Tier 4*: Physical range plausibility (₹500 – ₹250,000, 0.2 – 30 days).
  4. If gate **passes**:
     - Pulls shipment context from `orders` and `reference_invoices`.
     - Fetches prior vendor stats and computes **causal risk score**.
     - Executes Model 1 (baseline cost prediction) and constructs Model 2 features.
     - Scores anomaly probability via Model 2 XGBoost model (`flagged = proba >= 0.5`).
     - Generates structured audit explanation.
     - Atomically updates `vendor_running_stats` in Supabase.
  5. Inserts complete execution audit record into `audit_log`.
- **Sample Success Response**:
  ```json
  {
    "filename": "INV-ORD-00042.png",
    "extractor_used": "rules",
    "extracted_fields": {
      "order_id": "ORD-00042",
      "vendor_id": "VND-003",
      "total": "48250.00",
      "actual_days": "4.2",
      "freight_base": "42000.00",
      "detention": "3250.00",
      "toll": "3000.00"
    },
    "gate_passed": true,
    "gate_reason": null,
    "order_id": "ORD-00042",
    "vendor_id": "VND-003",
    "model1_predicted_cost": 41200.50,
    "billed_amount": 48250.00,
    "cost_mismatch": 7049.50,
    "model2_proba": 0.824,
    "flagged": true,
    "explanation": "Flagged for manual review (p=0.82): Billed amount (Rs. 48,250) exceeds expected model cost (Rs. 41,201) by Rs. 7,049 (+17.1%). Vendor historical risk score is 0.28."
  }
  ```

---

### 3. Vendor Risk Leaderboard
- **Endpoint**: `GET /api/vendors`
- **Description**: Returns all vendors ranked by their real-time Laplace-smoothed risk score computed from past audits.
- **Example Response**:
  ```json
  [
    {
      "vendor_id": "VND-007",
      "vendor_name": "Apex Logistics Express",
      "total_invoices": 142,
      "flagged_invoices": 38,
      "risk_score": 0.2647
    },
    {
      "vendor_id": "VND-002",
      "vendor_name": "BlueLine Freightways",
      "total_invoices": 98,
      "flagged_invoices": 12,
      "risk_score": 0.1212
    }
  ]
  ```

---

### 4. Operational Dashboard Analytics
- **Endpoint**: `GET /api/dashboard?n=300`
- **Description**: Returns aggregate metrics over the last `n` audit events, including the two primary production KPIs:
  1. **Auto-Processing Rate**: Fraction of invoices that pass the confidence gate.
  2. **Flag Rate / Anomaly Count**: Fraction of gate-cleared invoices flagged for fraud/error.
- **Example Response**:
  ```json
  {
    "total_processed": 300,
    "gate_passed": 221,
    "gate_rejected": 79,
    "auto_processing_rate": 0.7367,
    "flagged_count": 42,
    "rejection_breakdown": {
      "missing_critical_field": 31,
      "arithmetic_mismatch": 24,
      "order_id_not_found": 14,
      "amount_out_of_plausible_range": 10
    }
  }
  ```

---

## 5. Schema Maintenance & Self-Service Supabase Guide

You have full control over the Supabase database. Below are common maintenance and schema-building procedures you can perform directly.

### A. Inspecting & Querying Data in Supabase

1. **Table Editor**: Click the **Table Editor** tab in the Supabase console. You can view, search, filter, and manually edit records in `audit_log`, `orders`, `vendors`, etc.
2. **SQL Editor**: Execute ad-hoc analytical queries. For example, to check the most frequent failure causes:
   ```sql
   select
     gate_reason,
     count(*) as count
   from audit_log
   where gate_passed = false
   group by gate_reason
   order by count desc;
   ```

---

### B. Adding New Columns or Modifying Schema

If your application evolves (for example, adding user metadata or approval workflows to `audit_log`), run DDL statements in the SQL Editor:

```sql
-- Example 1: Add a manual review status and reviewer notes column
alter table audit_log
  add column if not exists reviewer_status text default 'PENDING',
  add column if not exists reviewer_notes text,
  add column if not exists reviewed_by text,
  add column if not exists reviewed_at timestamptz;

-- Example 2: Create an index to quickly query reviews by status
create index if not exists idx_audit_status on audit_log(reviewer_status);
```

Then update [`backend/schemas.py`](backend/schemas.py) and [`backend/db.py`](backend/db.py) to include the new fields.

---

### C. Resetting or Wiping Audit Logs (Testing / Fresh Starts)

If you want to clear test audits while leaving the reference catalog (`orders`, `vendors`, `hubs`) intact:

```sql
-- Clear operational audit logs and reset vendor stats
truncate table audit_log restart identity;
truncate table vendor_running_stats;
```

---

### D. Understanding the Causal Vendor Risk Mechanism

In [`backend/schema.sql`](backend/schema.sql), the view `vendor_risk_scores` is defined as:

$$\text{Risk Score} = \frac{\text{flagged\_invoices} + \alpha \cdot \text{prior}}{\text{total\_invoices} + \alpha}$$

Where:
- $\alpha = 2$ (smoothing weight)
- $\text{prior} = 0.06$ (empirical baseline anomaly rate)

This matches `build_dataset.py`'s Laplace smoothing formula identically. In `backend/main.py`, the score passed to Model 2 is computed **strictly before** updating the database for the current invoice:
```python
# 1. Read vendor stats BEFORE this invoice
vstats = db.get_vendor_stats(pred["vendor_id"])
vendor_risk = causal_vendor_risk(vstats["total_invoices"], vstats["flagged_invoices"])

# 2. Score with Model 2
proba = score_model2(features, _m2)

# 3. Only now increment the counters for subsequent invoices
db.bump_vendor_stats(pred["vendor_id"], flagged)
```
This guarantees strict temporal causality with zero data leakage.

---

## 6. Testing & Validation Checklist

You can test the entire pipeline locally without deploying a frontend:

### 1. Test Health Endpoint
```powershell
curl http://localhost:8000/api/health
```

### 2. Test Audit Endpoint with an Invoice Image
```powershell
curl -X POST "http://localhost:8000/api/audit?extractor=rules" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@phase_a_output/invoices_bench300/INV-ORD-00000.png"
```

### 3. Verify Leaderboard
```powershell
curl http://localhost:8000/api/vendors
```

### 4. Verify Dashboard Summary
```powershell
curl http://localhost:8000/api/dashboard?n=100
```
