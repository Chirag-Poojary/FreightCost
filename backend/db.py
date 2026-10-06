"""
Supabase access layer with automatic local CSV dataset fallback.

If SUPABASE_URL and SUPABASE_SERVICE_KEY are provided in .env, connects
to the Supabase remote PostgreSQL database. Otherwise, seamlessly falls back
to the local CSV dataset in code/output/, allowing the app to run completely
offline out of the box with zero external dependencies.
"""
import os
from dotenv import load_dotenv

# Load credentials from backend/.env or root .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()


class LocalDb:
    """Offline CSV-backed database implementation when Supabase is not configured."""
    def __init__(self, data_dir=None):
        import pandas as pd
        if not data_dir:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for cand in [os.path.join(base, "code", "output"), os.path.join(base, "data")]:
                if os.path.exists(os.path.join(cand, "orders.csv")):
                    data_dir = cand
                    break
        self.data_dir = data_dir
        self.orders = pd.read_csv(os.path.join(data_dir, "orders.csv")).set_index("order_id")
        self.vendors = pd.read_csv(os.path.join(data_dir, "vendors.csv")).set_index("vendor_id")
        inv_path = os.path.join(data_dir, "invoices.csv")
        self.invoices = pd.read_csv(inv_path).set_index("order_id") if os.path.exists(inv_path) else None
        gt_path = os.path.join(data_dir, "_ground_truth_audit.csv")
        self.gt = pd.read_csv(gt_path).set_index("order_id") if os.path.exists(gt_path) else None

        self.vendor_stats = {}
        if self.invoices is not None and "vendor_id" in self.invoices.columns:
            for vid, g in self.invoices.groupby("vendor_id"):
                flagged = int(g["requires_manual_review"].sum()) if "requires_manual_review" in g.columns else 0
                self.vendor_stats[vid] = {"vendor_id": vid, "total_invoices": len(g), "flagged_invoices": flagged}
        self.audit_logs = []

    def order_exists(self, order_id):
        return order_id in self.orders.index

    def vendor_exists(self, vendor_id):
        return vendor_id in self.vendors.index

    def get_order(self, order_id):
        import pandas as pd
        if order_id not in self.orders.index:
            return None
        row = self.orders.loc[order_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        d = row.to_dict()
        d["order_id"] = order_id
        return d

    def get_reference_invoice_context(self, order_id):
        import pandas as pd
        if self.invoices is not None and order_id in self.invoices.index:
            row = self.invoices.loc[order_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {
                "adverse_weather_days": float(row.get("adverse_weather_days", 0.0) or 0.0),
                "vendor_padding_ratio": float(row.get("vendor_padding_ratio", 0.0) or 0.0),
                "_from_reference": True,
            }
        return {"adverse_weather_days": 0.0, "vendor_padding_ratio": 0.0}

    def get_ground_truth(self, order_id):
        import pandas as pd
        if self.gt is not None and order_id in self.gt.index:
            row = self.gt.loc[order_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.to_dict()
        return None

    def get_vendor_stats(self, vendor_id):
        if vendor_id in self.vendor_stats:
            return self.vendor_stats[vendor_id].copy()
        return {"vendor_id": vendor_id, "total_invoices": 0, "flagged_invoices": 0}

    def bump_vendor_stats(self, vendor_id, was_flagged):
        st = self.get_vendor_stats(vendor_id)
        st["total_invoices"] += 1
        st["flagged_invoices"] += int(was_flagged)
        self.vendor_stats[vendor_id] = st

    def insert_audit_log(self, record):
        from datetime import datetime, timezone
        rec = dict(record)
        rec.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        self.audit_logs.insert(0, rec)

    def get_recent_audits(self, limit=50):
        return self.audit_logs[:limit]

    def get_audit_stats(self, since_n=None):
        return self.audit_logs[:since_n] if since_n else self.audit_logs

    def get_vendor_leaderboard(self):
        board = []
        for vid, st in self.vendor_stats.items():
            tot = st["total_invoices"]
            flg = st["flagged_invoices"]
            score = (flg + 1) / (tot + 2) if tot > 0 else 0.5
            vname = self.vendors.loc[vid, "vendor_name"] if vid in self.vendors.index else vid
            if hasattr(vname, "iloc"):
                vname = vname.iloc[0]
            board.append({
                "vendor_id": vid,
                "vendor_name": str(vname),
                "total_invoices": tot,
                "flagged_invoices": flg,
                "risk_score": float(score)
            })
        board.sort(key=lambda x: x["risk_score"], reverse=True)
        return board


class Db:
    def __init__(self, url=None, key=None):
        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            print("[Db] SUPABASE_URL / SUPABASE_SERVICE_KEY not provided. Running in Local CSV Database mode.")
            self._local = LocalDb()
            self.client = None
        else:
            self._local = None
            from supabase import create_client
            import re
            url = re.sub(r"/rest/v1/?$", "", str(url).strip()).rstrip("/")
            key = str(key).strip()
            self.client = create_client(url, key)

    # ---------------------------------------------------------- reference
    def order_exists(self, order_id):
        if self._local:
            return self._local.order_exists(order_id)
        r = self.client.table("orders").select("order_id").eq("order_id", order_id).execute()
        return len(r.data) > 0

    def vendor_exists(self, vendor_id):
        if self._local:
            return self._local.vendor_exists(vendor_id)
        r = self.client.table("vendors").select("vendor_id").eq("vendor_id", vendor_id).execute()
        return len(r.data) > 0

    def get_order(self, order_id):
        if self._local:
            return self._local.get_order(order_id)
        r = self.client.table("orders").select("*").eq("order_id", order_id).single().execute()
        return r.data

    def get_reference_invoice_context(self, order_id):
        if self._local:
            return self._local.get_reference_invoice_context(order_id)
        r = (self.client.table("reference_invoices")
            .select("adverse_weather_days, vendor_padding_ratio")
            .eq("order_id", order_id).limit(1).execute())
        if r.data:
            return {"adverse_weather_days": r.data[0]["adverse_weather_days"] or 0.0,
                   "vendor_padding_ratio": r.data[0]["vendor_padding_ratio"] or 0.0,
                   "_from_reference": True}
        return {"adverse_weather_days": 0.0, "vendor_padding_ratio": 0.0}

    def get_ground_truth(self, order_id):
        if self._local:
            return self._local.get_ground_truth(order_id)
        r = (self.client.table("ground_truth_audit").select("*")
            .eq("order_id", order_id).limit(1).execute())
        return r.data[0] if r.data else None

    # --------------------------------------------------- vendor risk (causal)
    def get_vendor_stats(self, vendor_id):
        if self._local:
            return self._local.get_vendor_stats(vendor_id)
        r = (self.client.table("vendor_running_stats").select("*")
            .eq("vendor_id", vendor_id).limit(1).execute())
        if r.data:
            return r.data[0]
        return {"vendor_id": vendor_id, "total_invoices": 0, "flagged_invoices": 0}

    def bump_vendor_stats(self, vendor_id, was_flagged):
        if self._local:
            return self._local.bump_vendor_stats(vendor_id, was_flagged)
        stats = self.get_vendor_stats(vendor_id)
        self.client.table("vendor_running_stats").upsert({
            "vendor_id": vendor_id,
            "total_invoices": stats["total_invoices"] + 1,
            "flagged_invoices": stats["flagged_invoices"] + int(was_flagged),
        }).execute()

    # --------------------------------------------------------------- audit_log
    def insert_audit_log(self, record):
        if self._local:
            return self._local.insert_audit_log(record)
        self.client.table("audit_log").insert(record).execute()

    def get_recent_audits(self, limit=50):
        if self._local:
            return self._local.get_recent_audits(limit)
        r = (self.client.table("audit_log").select("*")
            .order("created_at", desc=True).limit(limit).execute())
        return r.data

    def get_audit_stats(self, since_n=None):
        if self._local:
            return self._local.get_audit_stats(since_n)
        q = self.client.table("audit_log").select("*").order("created_at", desc=True)
        if since_n:
            q = q.limit(since_n)
        return q.execute().data

    def get_vendor_leaderboard(self):
        if self._local:
            return self._local.get_vendor_leaderboard()
        r = self.client.table("vendor_risk_scores").select("*").order("risk_score", desc=True).execute()
        return r.data
