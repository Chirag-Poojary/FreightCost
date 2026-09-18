"""
PHASE A / STEP 1 -- Synthetic invoice document rendering.

The existing pipeline assumes invoice data arrives as clean CSV rows. Real
invoices arrive as scanned PDFs and phone photos. This script renders each
invoices.csv row into a realistic Indian freight invoice image, and writes a
ground-truth JSON alongside it so OCR extraction accuracy can be measured
exactly (predicted field vs. the value that was actually printed).

Three layout variants are rotated so the downstream extractor cannot simply
memorise one fixed coordinate template -- which is what would happen with a
single layout, and would make the accuracy numbers meaningless.

Line-item decomposition is exact: freight_base + detention + toll == total,
and total == actual_billed_amount from the CSV. No rounding drift.

Usage:
  python render_invoices.py --n 200 --outdir ../phase_a_output/invoices_clean
  python render_invoices.py --n 200 --scan --outdir ../phase_a_output/invoices_scanned
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1240, 1754                      # A4 at ~150 DPI
TOLL_RATE_PER_KM = 1.75                # matches generators.py
DETENTION_RATE_PER_DAY = 1500.0

if os.name == "nt" or os.path.exists("C:/Windows/Fonts"):
    _WIN_FONTS = "C:/Windows/Fonts"
    FONTS = {
        "sans":       f"{_WIN_FONTS}/arial.ttf",
        "sans_bold":  f"{_WIN_FONTS}/arialbd.ttf",
        "serif":      f"{_WIN_FONTS}/times.ttf",
        "serif_bold": f"{_WIN_FONTS}/timesbd.ttf",
        "mono":       f"{_WIN_FONTS}/cour.ttf",
        "lib":        f"{_WIN_FONTS}/arial.ttf",
        "lib_bold":   f"{_WIN_FONTS}/arialbd.ttf",
    }
else:
    FONT_DIR = "/usr/share/fonts/truetype"
    FONTS = {
        "sans":       f"{FONT_DIR}/dejavu/DejaVuSans.ttf",
        "sans_bold":  f"{FONT_DIR}/dejavu/DejaVuSans-Bold.ttf",
        "serif":      f"{FONT_DIR}/dejavu/DejaVuSerif.ttf",
        "serif_bold": f"{FONT_DIR}/dejavu/DejaVuSerif-Bold.ttf",
        "mono":       f"{FONT_DIR}/dejavu/DejaVuSansMono.ttf",
        "lib":        f"{FONT_DIR}/liberation/LiberationSans-Regular.ttf",
        "lib_bold":   f"{FONT_DIR}/liberation/LiberationSans-Bold.ttf",
    }
_font_cache = {}


def F(name, size):
    key = (name, size)
    if key not in _font_cache:
        path = FONTS.get(name)
        if path and os.path.exists(path):
            try:
                _font_cache[key] = ImageFont.truetype(path, size)
            except Exception:
                _font_cache[key] = ImageFont.load_default()
        else:
            try:
                _font_cache[key] = ImageFont.truetype(name, size)
            except Exception:
                _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def rupees(x):
    """Indian-format currency string, e.g. 1,23,456.78 (lakh grouping)."""
    s = f"{x:,.2f}"                      # western grouping first
    whole, dec = s.replace(",", "").split(".")
    neg = whole.startswith("-")
    whole = whole.lstrip("-")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + "." + dec


def decompose(row):
    """Split the billed total into printable line items that sum EXACTLY."""
    total = float(row["actual_billed_amount"])
    toll = round(float(row["distance_km"]) * TOLL_RATE_PER_KM, 2)
    det_days = max(0.0, float(row["commercial_delay_days"]))
    detention = round(min(det_days * DETENTION_RATE_PER_DAY, total * 0.35), 2)
    freight = round(total - toll - detention, 2)
    if freight < 0:                      # pathological row: collapse to freight only
        toll, detention, freight = 0.0, 0.0, round(total, 2)
    return freight, detention, toll, round(total, 2)


# ------------------------------------------------------------------ layouts
def layout_classic(d, r, fields):
    """Bordered header block, boxed line-item table, totals bottom-right."""
    d.rectangle([40, 40, W - 40, H - 40], outline=0, width=3)
    d.text((70, 70), r["vendor_name"], font=F("serif_bold", 40), fill=0)
    d.text((70, 122), f"{r['home_state']}, India", font=F("sans", 22), fill=60)
    d.text((70, 152), f"GSTIN: {fields['gstin']}", font=F("sans", 21), fill=60)
    d.text((W - 400, 74), "TAX INVOICE", font=F("serif_bold", 38), fill=0)
    d.line([40, 200, W - 40, 200], fill=0, width=2)

    y = 226
    left = [("Invoice No", fields["invoice_id"]), ("Invoice Date", fields["invoice_date"]),
            ("Order Ref", fields["order_id"]), ("Vendor Code", fields["vendor_id"])]
    right = [("Origin", fields["origin"]), ("Destination", fields["destination"]),
             ("Truck Type", fields["truck_type"]), ("Transit Days", fields["actual_days"])]
    for (la, va), (lb, vb) in zip(left, right):
        d.text((70, y), f"{la}:", font=F("sans", 22), fill=90)
        d.text((260, y), str(va), font=F("sans_bold", 23), fill=0)
        d.text((680, y), f"{lb}:", font=F("sans", 22), fill=90)
        d.text((880, y), str(vb), font=F("sans_bold", 23), fill=0)
        y += 40

    y += 26
    d.rectangle([70, y, W - 70, y + 46], fill=235, outline=0, width=2)
    d.text((90, y + 12), "Description", font=F("sans_bold", 23), fill=0)
    d.text((760, y + 12), "Qty / Basis", font=F("sans_bold", 23), fill=0)
    d.text((965, y + 12), "Amount (INR)", font=F("sans_bold", 23), fill=0)
    y += 46
    rows = [("Base Freight Charges", f"{fields['weight_kg']} kg", fields["freight_base"]),
            ("Detention / Waiting Charges", f"{fields['detention_days']} day(s)", fields["detention"]),
            ("Toll & FASTag Charges", f"{fields['distance_km']} km", fields["toll"])]
    for desc, qty, amt in rows:
        d.rectangle([70, y, W - 70, y + 44], outline=0, width=1)
        d.text((90, y + 11), desc, font=F("sans", 22), fill=0)
        d.text((760, y + 11), qty, font=F("sans", 22), fill=0)
        d.text((1000, y + 11), rupees(amt), font=F("mono", 21), fill=0)
        y += 44

    y += 34
    d.rectangle([700, y, W - 70, y + 54], fill=235, outline=0, width=2)
    d.text((726, y + 14), "TOTAL PAYABLE", font=F("sans_bold", 25), fill=0)
    d.text((1000, y + 14), rupees(fields["total"]), font=F("mono", 24), fill=0)

    d.text((70, H - 200), "Payment due within 30 days of invoice date.",
           font=F("sans", 20), fill=90)
    d.text((70, H - 168), "Subject to jurisdiction. E&OE.", font=F("sans", 20), fill=90)
    d.text((W - 420, H - 140), "Authorised Signatory", font=F("sans", 21), fill=60)
    d.line([W - 430, H - 150, W - 120, H - 150], fill=0, width=1)


def layout_modern(d, r, fields):
    """Dark banner header, label-over-value blocks, minimal rules."""
    d.rectangle([0, 0, W, 150], fill=35)
    d.text((60, 38), r["vendor_name"], font=F("lib_bold", 40), fill=255)
    d.text((60, 96), f"GSTIN {fields['gstin']}  |  {r['home_state']}",
           font=F("lib", 21), fill=210)
    d.text((W - 330, 52), "INVOICE", font=F("lib_bold", 44), fill=255)

    y = 210
    cols = [("INVOICE NUMBER", fields["invoice_id"]), ("DATE", fields["invoice_date"]),
            ("ORDER REFERENCE", fields["order_id"]), ("VENDOR CODE", fields["vendor_id"])]
    x = 60
    for lab, val in cols:
        d.text((x, y), lab, font=F("lib", 17), fill=120)
        d.text((x, y + 26), str(val), font=F("lib_bold", 25), fill=0)
        x += 295
    y += 104
    d.line([60, y, W - 60, y], fill=200, width=2)

    y += 30
    x = 60
    for lab, val in [("ORIGIN", fields["origin"]), ("DESTINATION", fields["destination"]),
                     ("VEHICLE", fields["truck_type"]), ("TRANSIT DAYS", fields["actual_days"])]:
        d.text((x, y), lab, font=F("lib", 17), fill=120)
        d.text((x, y + 26), str(val), font=F("lib_bold", 24), fill=0)
        x += 295
    y += 116

    d.text((60, y), "CHARGES", font=F("lib_bold", 22), fill=0)
    y += 44
    for desc, amt in [("Base freight", fields["freight_base"]),
                      ("Detention / waiting", fields["detention"]),
                      ("Toll & FASTag", fields["toll"])]:
        d.text((60, y), desc, font=F("lib", 23), fill=40)
        d.text((W - 320, y), rupees(amt), font=F("mono", 22), fill=0)
        d.line([60, y + 38, W - 60, y + 38], fill=225, width=1)
        y += 56

    y += 22
    d.rectangle([W - 560, y, W - 60, y + 66], fill=35)
    d.text((W - 535, y + 20), "TOTAL", font=F("lib_bold", 26), fill=255)
    d.text((W - 330, y + 20), rupees(fields["total"]), font=F("mono", 25), fill=255)

    d.text((60, H - 150), "Computer generated invoice. Valid without signature.",
           font=F("lib", 19), fill=120)


def layout_compact(d, r, fields):
    """Dense two-column form, typewriter feel -- mimics legacy ERP printouts."""
    d.text((60, 54), r["vendor_name"].upper(), font=F("mono", 32), fill=0)
    d.text((60, 98), f"GSTIN {fields['gstin']}", font=F("mono", 19), fill=70)
    d.text((W - 330, 58), "FREIGHT BILL", font=F("mono", 27), fill=0)
    d.line([60, 136, W - 60, 136], fill=0, width=2)
    d.line([60, 142, W - 60, 142], fill=0, width=1)

    y = 176
    pairs = [("INVOICE NO", fields["invoice_id"]), ("DATE", fields["invoice_date"]),
             ("ORDER NO", fields["order_id"]), ("VENDOR ID", fields["vendor_id"]),
             ("FROM", fields["origin"]), ("TO", fields["destination"]),
             ("TRUCK", fields["truck_type"]), ("DAYS IN TRANSIT", fields["actual_days"]),
             ("CHARGEABLE WT", f"{fields['weight_kg']} KG"),
             ("DISTANCE", f"{fields['distance_km']} KM")]
    for i, (lab, val) in enumerate(pairs):
        col_x = 60 if i % 2 == 0 else 660
        if i % 2 == 0 and i:
            y += 38
        d.text((col_x, y), f"{lab:<16}", font=F("mono", 20), fill=80)
        d.text((col_x + 250, y), str(val), font=F("mono", 21), fill=0)
    y += 76

    d.line([60, y, W - 60, y], fill=0, width=1)
    y += 18
    d.text((60, y), "PARTICULARS", font=F("mono", 21), fill=0)
    d.text((W - 340, y), "AMOUNT INR", font=F("mono", 21), fill=0)
    y += 34
    d.line([60, y, W - 60, y], fill=0, width=1)
    y += 18
    for desc, amt in [("FREIGHT CHARGES", fields["freight_base"]),
                      ("DETENTION CHARGES", fields["detention"]),
                      ("TOLL CHARGES", fields["toll"])]:
        d.text((60, y), desc, font=F("mono", 20), fill=0)
        d.text((W - 340, y), rupees(amt), font=F("mono", 20), fill=0)
        y += 36
    d.line([60, y + 6, W - 60, y + 6], fill=0, width=1)
    y += 26
    d.text((60, y), "GRAND TOTAL", font=F("mono", 23), fill=0)
    d.text((W - 340, y), rupees(fields["total"]), font=F("mono", 23), fill=0)
    d.line([60, y + 38, W - 60, y + 38], fill=0, width=2)

    d.text((60, H - 170), "E.& O.E.   THIS IS A COMPUTER GENERATED DOCUMENT.",
           font=F("mono", 18), fill=90)


LAYOUTS = [layout_classic, layout_modern, layout_compact]


# ------------------------------------------------------------ scan artifacts
def apply_scan(img, rng):
    """Degrade a clean render into something resembling a scan or phone photo."""
    img = img.rotate(rng.uniform(-1.6, 1.6), resample=Image.BICUBIC,
                     fillcolor=255, expand=False)
    if rng.random() < 0.75:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    a = np.asarray(img).astype(np.float32)
    a += rng.normal(0, rng.uniform(4, 11), a.shape)              # sensor noise
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    shade = (1.0 - rng.uniform(0.04, 0.13) * (xx / a.shape[1])   # uneven lighting
                 - rng.uniform(0.02, 0.09) * (yy / a.shape[0]))
    a *= shade
    a = a * rng.uniform(0.92, 1.0) + rng.uniform(0, 14)          # contrast/exposure
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# -------------------------------------------------------------------- driver
def main(n, outdir, scan, seed, data_dir):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    candidates = [data_dir, "data", "../data", "code/output", "../code/output"]
    for c in candidates:
        if os.path.exists(os.path.join(c, "invoices.csv")):
            data_dir = c
            break

    inv = pd.read_csv(f"{data_dir}/invoices.csv")
    orders = pd.read_csv(f"{data_dir}/orders.csv")
    vendors = pd.read_csv(f"{data_dir}/vendors.csv")
    hubs = pd.read_csv(f"{data_dir}/hubs.csv").set_index("hub_id")

    df = (inv.merge(orders, on=["order_id", "vendor_id"], how="left")
             .merge(vendors, on="vendor_id", how="left"))
    df = df.dropna(subset=["distance_km", "vendor_name"])
    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    print(f"Rendering {len(df)} invoices -> {outdir}  (scan artifacts: {scan})")

    manifest = []
    for i, r in df.iterrows():
        freight, detention, toll, total = decompose(r)
        o, dst = hubs.loc[r["origin_hub_id"]], hubs.loc[r["dest_hub_id"]]
        inv_date = pd.to_datetime(r["order_date"]) + pd.Timedelta(days=float(r["actual_days"]))

        fields = {
            "invoice_id": r["invoice_id"],
            "order_id": r["order_id"],
            "vendor_id": r["vendor_id"],
            "invoice_date": inv_date.strftime("%d-%m-%Y"),
            "origin": o["city"],
            "destination": dst["city"],
            "truck_type": r["truck_type"],
            "actual_days": f"{float(r['actual_days']):.2f}",
            "weight_kg": f"{float(r['billable_weight_kg']):.0f}",
            "distance_km": f"{float(r['distance_km']):.1f}",
            "detention_days": f"{max(0.0, float(r['commercial_delay_days'])):.2f}",
            "freight_base": freight,
            "detention": detention,
            "toll": toll,
            "total": total,
            "gstin": f"{rng.integers(10,36):02d}ABCDE{rng.integers(1000,9999)}F1Z{rng.integers(0,9)}",
        }

        img = Image.new("L", (W, H), 255)
        d = ImageDraw.Draw(img)
        layout_fn = LAYOUTS[i % len(LAYOUTS)]
        layout_fn(d, r, fields)
        if scan:
            img = apply_scan(img, rng)

        name = f"{r['invoice_id']}.png"
        img.save(os.path.join(outdir, name), optimize=True)
        manifest.append({"image": name, "layout": layout_fn.__name__,
                         "scanned": bool(scan), "ground_truth": fields})

        if (i + 1) % 50 == 0 or i + 1 == len(df):
            print(f"  {i+1}/{len(df)}")

    with open(os.path.join(outdir, "ground_truth.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f"Wrote {len(manifest)} images + ground_truth.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--outdir", default="../phase_a_output/invoices_clean")
    ap.add_argument("--data-dir", default="../code/output")
    ap.add_argument("--scan", action="store_true", help="apply scan/photo degradation")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.n, a.outdir, a.scan, a.seed, a.data_dir)
