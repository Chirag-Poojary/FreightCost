"""
Supabase access layer. Every network call to Supabase lives in this one
file -- main.py and audit_core.py never import the supabase client
directly. That means the rest of the app can be tested with a fake/mock
`Db` object with zero network access, which matters here since this
sandbox can't reach supabase.co to verify a live connection anyway.

Env vars required (see .env.example):
  SUPABASE_URL          -- https://<project-ref>.supabase.co
  SUPABASE_SERVICE_KEY  -- service_role key (server-side only -- NEVER ship
                            this to a browser; it bypasses row-level security)
"""
import os
from dotenv import load_dotenv

# Load credentials from backend/.env or root .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()


class Db:
    def __init__(self, url=None, key=None):
        from supabase import create_client
        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            missing = []
            if not url: missing.append("SUPABASE_URL")
            if not key: missing.append("SUPABASE_SERVICE_KEY")
            raise ValueError(
                f"Missing Supabase configuration: {', '.join(missing)} is not set!\n"
                f"Please add {', '.join(missing)} to your .env file (e.g. SUPABASE_URL=https://your-project-id.supabase.co).\n"
                f"See README_BACKEND.md for setup instructions."
            )
        import re
        url = re.sub(r"/rest/v1/?$", "", str(url).strip()).rstrip("/")
        key = str(key).strip()
        self.client = create_client(url, key)

    # ---------------------------------------------------------- reference
    def order_exists(self, order_id):
        r = self.client.table("orders").select("order_id").eq("order_id", order_id).execute()
        return len(r.data) > 0

    def vendor_exists(self, vendor_id):
        r = self.client.table("vendors").select("vendor_id").eq("vendor_id", vendor_id).execute()
        return len(r.data) > 0

    def get_order(self, order_id):
        r = self.client.table("orders").select("*").eq("order_id", order_id).single().execute()
        return r.data

    def get_reference_invoice_context(self, order_id):
        """adverse_weather_days / vendor_padding_ratio for a demo invoice
        that matches a reference order. Returns zeros if this is a genuinely
        new shipment with no reference row -- that's expected, not an error."""
        r = (self.client.table("reference_invoices")
            .select("adverse_weather_days, vendor_padding_ratio")
            .eq("order_id", order_id).limit(1).execute())
        if r.data:
            return {"adverse_weather_days": r.data[0]["adverse_weather_days"] or 0.0,
                   "vendor_padding_ratio": r.data[0]["vendor_padding_ratio"] or 0.0}
        return {"adverse_weather_days": 0.0, "vendor_padding_ratio": 0.0}

    def get_ground_truth(self, order_id):
        """Only used by the dashboard to grade against known outcomes --
        never read on the scoring path itself."""
        r = (self.client.table("ground_truth_audit").select("*")
            .eq("order_id", order_id).limit(1).execute())
        return r.data[0] if r.data else None

    # --------------------------------------------------- vendor risk (causal)
    def get_vendor_stats(self, vendor_id):
        r = (self.client.table("vendor_running_stats").select("*")
            .eq("vendor_id", vendor_id).limit(1).execute())
        if r.data:
            return r.data[0]
        return {"vendor_id": vendor_id, "total_invoices": 0, "flagged_invoices": 0}

    def bump_vendor_stats(self, vendor_id, was_flagged):
        """Called AFTER scoring, never before -- this is what keeps the risk
        score causal. Upserts so a vendor's first invoice creates the row."""
        stats = self.get_vendor_stats(vendor_id)
        self.client.table("vendor_running_stats").upsert({
            "vendor_id": vendor_id,
            "total_invoices": stats["total_invoices"] + 1,
            "flagged_invoices": stats["flagged_invoices"] + int(was_flagged),
        }).execute()

    # --------------------------------------------------------------- audit_log
    def insert_audit_log(self, record):
        self.client.table("audit_log").insert(record).execute()

    def get_recent_audits(self, limit=50):
        r = (self.client.table("audit_log").select("*")
            .order("created_at", desc=True).limit(limit).execute())
        return r.data

    def get_audit_stats(self, since_n=None):
        """Aggregates for the dashboard. Pulls raw rows and aggregates in
        Python rather than a Postgres RPC -- simpler to read for a student
        project; swap for an RPC function later if row counts get large."""
        q = self.client.table("audit_log").select("*").order("created_at", desc=True)
        if since_n:
            q = q.limit(since_n)
        return q.execute().data

    def get_vendor_leaderboard(self):
        r = self.client.table("vendor_risk_scores").select("*").order("risk_score", desc=True).execute()
        return r.data
