-- ============================================================
-- Supabase schema for the FreightCost audit backend.
-- Run this in the Supabase SQL Editor once, before migrate_to_supabase.py.
--
-- Two kinds of tables, and it matters which is which:
--   REFERENCE  (hubs, vendors, orders, reference_invoices) -- migrated once
--              from the existing CSVs. Read-heavy, effectively static.
--   OPERATIONAL (audit_log, vendor_running_stats)          -- written by
--              the FastAPI app on every real audit. This is the actual
--              "database" the web app owns going forward.
-- ============================================================

-- ---------- REFERENCE: hubs ----------
create table if not exists hubs (
  hub_id      text primary key,
  city        text not null,
  state       text,
  latitude    double precision not null,
  longitude   double precision not null,
  hub_category text
);

-- ---------- REFERENCE: vendors ----------
create table if not exists vendors (
  vendor_id       text primary key,
  vendor_name     text not null,
  home_state      text,
  fleet_size      integer,
  onboarding_date date
);

-- ---------- REFERENCE: orders ----------
-- Everything Model 1 needs, keyed by order_id, so the confidence gate's
-- referential-integrity check and the ERP join are both single-row lookups.
create table if not exists orders (
  order_id                text primary key,
  vendor_id               text references vendors(vendor_id),
  origin_hub_id           text references hubs(hub_id),
  dest_hub_id             text references hubs(hub_id),
  product_category        text,
  order_date              date,
  weight_kg               double precision,
  length_cm               double precision,
  width_cm                double precision,
  height_cm               double precision,
  quoted_days             double precision,
  ideal_days              double precision,
  distance_km             double precision,
  dimensional_weight_kg   double precision,
  billable_weight_kg      double precision,
  truck_type              text,
  state                   text,
  expected_fuel_price     double precision,
  expected_weather_score  double precision,
  model_a_predicted_cost  double precision
);
create index if not exists idx_orders_vendor on orders(vendor_id);

-- ---------- REFERENCE: reference_invoices ----------
-- The synthetic ground-truth invoices from the original dataset build.
-- Used only for orders that exist in the reference set (demo/benchmark
-- invoices) to supply adverse_weather_days / vendor_padding_ratio, which
-- aren't things OCR can read off a document. A genuinely new shipment
-- (Phase D's live-quote path) doesn't have a row here -- that's expected.
create table if not exists reference_invoices (
  invoice_id                   text primary key,
  order_id                     text references orders(order_id),
  vendor_id                    text references vendors(vendor_id),
  actual_billed_amount         double precision,
  actual_days                  double precision,
  adverse_weather_days         double precision,
  cost_mismatch                double precision,
  true_delay_days              double precision,
  commercial_delay_days        double precision,
  vendor_padding_ratio         double precision,
  vendor_historical_risk_score double precision,
  requires_manual_review       boolean
);
create index if not exists idx_refinv_order on reference_invoices(order_id);

-- ---------- REFERENCE: ground_truth_audit ----------
-- Anomaly labels, kept separate from reference_invoices exactly as the
-- original pipeline separated them -- never exposed to the scoring path,
-- only used by the dashboard to grade Model 2 against known outcomes.
create table if not exists ground_truth_audit (
  order_id                text primary key references orders(order_id),
  anomaly_type            text,
  requires_manual_review  boolean not null
);

-- ---------- OPERATIONAL: audit_log ----------
-- One row per invoice actually processed through the web app.
create table if not exists audit_log (
  id                     bigint generated always as identity primary key,
  created_at             timestamptz not null default now(),
  filename               text,
  extractor_used         text not null,
  extracted_fields       jsonb,
  gate_passed            boolean not null,
  gate_reason            text,
  order_id               text,
  vendor_id              text,
  model1_predicted_cost  double precision,
  billed_amount          double precision,
  cost_mismatch          double precision,
  model2_proba           double precision,
  flagged                boolean,
  explanation            text
);
create index if not exists idx_audit_created on audit_log(created_at desc);
create index if not exists idx_audit_vendor on audit_log(vendor_id);

-- ---------- OPERATIONAL: vendor_running_stats ----------
-- Causal, incremental vendor risk -- updated after every real audit, so
-- the score a new invoice sees only reflects invoices strictly before it.
-- This is the durable version of build_dataset.py's Laplace-smoothed
-- vendor_historical_risk_score, now persisted across server restarts.
create table if not exists vendor_running_stats (
  vendor_id         text primary key references vendors(vendor_id),
  total_invoices    integer not null default 0,
  flagged_invoices  integer not null default 0,
  updated_at        timestamptz not null default now()
);

-- Convenience view: current causal risk score per vendor (alpha=2, prior=0.06,
-- matching build_dataset.py's add_vendor_risk() exactly).
create or replace view vendor_risk_scores as
select
  v.vendor_id,
  v.vendor_name,
  coalesce(s.total_invoices, 0) as total_invoices,
  coalesce(s.flagged_invoices, 0) as flagged_invoices,
  (coalesce(s.flagged_invoices, 0)::double precision + 2.0 * 0.06) / (coalesce(s.total_invoices, 0)::double precision + 2.0) as risk_score
from vendors v
left join vendor_running_stats s on v.vendor_id = s.vendor_id;
