"""
Explainable audit report.

build_reasons()  turns the model outputs into a short list of plain-language
                 reasons ("why this invoice was approved or flagged"), shown on
                 the verdict screen and in the PDF.
render_pdf()     produces the downloadable A4 audit report from a scored result.
"""
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO
from xml.sax.saxutils import escape

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
IST = timezone(timedelta(hours=5, minutes=30))
FLAG_THRESHOLD = 0.5


def signed_pct(v):
    """+1.4% / −1.4% with a proper minus sign."""
    return f"{v:+.1f}%".replace("-", "−")


def inr(v, decimals=0):
    """Rs amount with a proper minus sign: -629.1 -> '−₹629'."""
    if v is None:
        return "—"
    v = float(v)
    s = f"₹{abs(v):,.{decimals}f}"
    return ("−" + s) if v < 0 else s


# ------------------------------------------------------------------ reasons
def build_reasons(features, flagged, flag_source, proba, bill_check, context, manual_fields, drift):
    """Ordered list of {"status": pass|check|info, "title", "detail"}."""
    reasons = []
    billed = features["actual_billed_amount"]
    fair = features["model_a_predicted_cost"]
    lo, hi = features.get("cost_interval_lower"), features.get("cost_interval_upper")
    diff = billed - fair
    pct = diff / fair * 100 if fair else 0

    # 1. bill against the fair cost
    if hi is not None and billed > hi:
        reasons.append({"status": "check", "title": "The bill is above the fair price range",
                        "detail": f"Billed {inr(billed)} against a fair cost of {inr(fair)} (likely range {inr(lo)} to {inr(hi)}). "
                                  f"That is {inr(billed - hi)} above the top of the range, or {signed_pct(pct)} against the fair cost."})
    elif lo is not None and billed < lo:
        too_low = bill_check.get("status") == "too_low"
        reasons.append({"status": "check" if too_low else "pass",
                        "title": "The bill is far below the fair price range" if too_low else "The bill is below the fair price range",
                        "detail": f"Billed {inr(billed)} against a fair cost of {inr(fair)} (likely range {inr(lo)} to {inr(hi)}), "
                                  f"{inr(lo - billed)} below the bottom of the range. "
                                  + ("This usually means a partial invoice or a wrong order number." if too_low
                                     else "A slightly lower bill is not a concern.")})
    else:
        reasons.append({"status": "pass", "title": "The bill is within the fair price range",
                        "detail": f"Billed {inr(billed)} against a fair cost of {inr(fair)} (likely range {inr(lo)} to {inr(hi)}), "
                                  f"a difference of {signed_pct(pct)}."})

    # 2. risk model
    if proba >= FLAG_THRESHOLD:
        reasons.append({"status": "check", "title": f"Overbilling risk is high ({proba:.1%})",
                        "detail": "The risk model compares this bill with the fair cost, the delivery delay and the "
                                  f"carrier's history. Invoices at or above {FLAG_THRESHOLD:.0%} are flagged for audit."})
    else:
        reasons.append({"status": "pass", "title": f"Overbilling risk is low ({proba:.1%})",
                        "detail": f"Below the {FLAG_THRESHOLD:.0%} level at which invoices are flagged for audit."})

    # 3. price logic check
    if bill_check.get("status") == "ok":
        reasons.append({"status": "pass", "title": "Price check passed",
                        "detail": "The amount is within the limits seen across 56,543 genuine invoices."})
    else:
        reasons.append({"status": "check", "title": "Price check failed",
                        "detail": bill_check.get("message") or "The billed amount is outside plausible limits."})

    # 4. delivery delay against the carrier's own quote
    delay = features.get("commercial_delay_days", 0.0)
    if delay > 0.5 and proba >= FLAG_THRESHOLD:
        reasons.append({"status": "check", "title": f"Delivered {delay:.1f} days after the quoted time",
                        "detail": "Late delivery is how extra waiting (detention) charges are usually justified, so the "
                                  "detention line deserves a closer look."})
    elif delay > 0.5:
        reasons.append({"status": "info", "title": f"Delivered {delay:.1f} days after the quoted time",
                        "detail": "Delays can justify extra waiting charges. The risk model did not find them excessive here."})
    else:
        reasons.append({"status": "pass", "title": "Delivered within the quoted time",
                        "detail": f"Actual transit {features.get('actual_days', 0):.2f} days."})

    # 5. weather
    adverse = context.get("adverse_weather_days") or 0
    if adverse > 0:
        reasons.append({"status": "info", "title": f"{adverse:.0f} heavy-rain day(s) during the trip",
                        "detail": "Bad weather can legitimately explain some delay and waiting charges."})
    else:
        reasons.append({"status": "info", "title": "No heavy rain during the trip",
                        "detail": f"Source: {context.get('adverse_weather_source') or 'Open-Meteo'}."})

    # 6. carrier history
    vr = context.get("vendor_historical_risk_score") or 0
    if any("New carrier" in n for n in context.get("notes", [])):
        reasons.append({"status": "info", "title": "New carrier with no audit history",
                        "detail": "There is no past record for this carrier, so its history does not affect the score."})
    elif vr > 0.1:
        reasons.append({"status": "check", "title": f"Carrier's past invoices were often flagged ({vr:.1%})",
                        "detail": "Based only on this carrier's invoices audited before this one."})
    else:
        reasons.append({"status": "pass", "title": f"Carrier has a good record ({vr:.1%} past flag rate)",
                        "detail": "Based only on this carrier's invoices audited before this one."})

    # 7. data entry
    if manual_fields:
        reasons.append({"status": "info", "title": (f"{len(manual_fields)} details were" if len(manual_fields) > 1 else "1 detail was") + " entered or corrected by the reviewer",
                        "detail": ", ".join(f.replace("_", " ") for f in manual_fields).capitalize() + "."})

    # 8. price drift
    for w in drift or []:
        reasons.append({"status": "info", "title": "Note on prices", "detail": w})
    return reasons


def build_summary(flagged, flag_source, bill_check, features):
    diff = features["actual_billed_amount"] - features["model_a_predicted_cost"]
    if not flagged:
        return (f"No signs of overbilling were found. The bill is {inr(abs(diff))} "
                f"{'below' if diff < 0 else 'above'} the fair cost, which is within the normal range.")
    if flag_source == "logic_check" and bill_check.get("status") == "too_low":
        return "The bill is far below the fair cost for this trip. Check the order number and amounts before paying."
    return (f"This invoice should be checked before payment. The bill is {inr(abs(diff))} "
            f"{'above' if diff > 0 else 'below'} the fair cost.")


# ---------------------------------------------------------------------- PDF
def render_pdf(result):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(FONT_DIR, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf")))

    NAVY = colors.HexColor("#1d4f91")
    INK = colors.HexColor("#1f2937")
    MUTED = colors.HexColor("#64748b")
    LINE = colors.HexColor("#e2e8f0")
    ALT = colors.HexColor("#f8fafc")
    STATUS = {"pass": ("PASS", colors.HexColor("#15803d"), colors.HexColor("#ecfdf3")),
              "check": ("CHECK", colors.HexColor("#b91c1c"), colors.HexColor("#fef2f2")),
              "info": ("NOTE", colors.HexColor("#475569"), colors.HexColor("#f1f5f9"))}

    def st(name, **kw):
        base = dict(fontName="DejaVu", fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_LEFT)
        base.update(kw)
        return ParagraphStyle(name, **base)

    body = st("body")
    small = st("small", fontSize=8.5, leading=12, textColor=MUTED)
    h2 = st("h2", fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=NAVY, spaceBefore=10, spaceAfter=6)
    label = st("label", fontSize=8, leading=10, textColor=MUTED)
    big = st("big", fontName="DejaVu-Bold", fontSize=14, leading=17)
    bold = st("bold", fontName="DejaVu-Bold")
    P = lambda t, s=body: Paragraph(escape(str(t)), s)

    f = result.get("extracted_fields") or {}
    trip = result.get("trip") or {}
    ctx = result.get("context") or {}
    flagged = bool(result.get("flagged"))
    now = datetime.now(IST)
    ref = f"RR-{now:%Y%m%d-%H%M%S}"

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm,
                            title=f"RateRight audit report {f.get('invoice_id') or ''}".strip(),
                            author="RateRight")
    W = doc.width - 12
    story = []

    # --- header
    head = Table([[Paragraph("RATERIGHT", st("brand", fontName="DejaVu-Bold", fontSize=15, leading=18, textColor=NAVY)),
                   Paragraph(f"Generated {now:%d %b %Y, %I:%M %p} IST<br/>Report {ref}",
                             st("hr", fontSize=8.5, leading=12, textColor=MUTED, alignment=2))],
                  [Paragraph("Freight Invoice Audit Report", st("t", fontName="DejaVu-Bold", fontSize=18, leading=23)), ""]],
                 colWidths=[W * 0.62, W * 0.38])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("SPAN", (0, 1), (1, 1)),
                              ("LINEBELOW", (0, 1), (-1, 1), 1, LINE), ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [head, Spacer(1, 10)]

    # --- verdict panel
    vcol, vbg = (STATUS["check"][1], STATUS["check"][2]) if flagged else (STATUS["pass"][1], STATUS["pass"][2])
    verdict_txt = "FLAGGED FOR AUDIT" if flagged else "APPROVED"
    ids = "   ·   ".join(x for x in [f.get("invoice_id") and f"Invoice {f['invoice_id']}",
                                       f.get("order_id") and f"Order {f['order_id']}",
                                       f.get("vendor_id") and f"Carrier {f['vendor_id']}"] if x)
    vt = Table([[Paragraph(verdict_txt, st("v", fontName="DejaVu-Bold", fontSize=16, leading=20, textColor=vcol))],
                [P(result.get("summary") or "", st("vs", fontSize=10.5, leading=15))],
                [P(ids, small)]], colWidths=[W])
    vt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), vbg), ("BOX", (0, 0), (-1, -1), 1, vcol),
                            ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                            ("TOPPADDING", (0, 0), (0, 0), 10), ("BOTTOMPADDING", (0, -1), (0, -1), 10)]))
    story += [vt, Spacer(1, 10)]

    # --- key figures
    billed = result.get("billed_amount")
    fair = result.get("model1_predicted_cost")
    diff = result.get("cost_mismatch")
    pct = (diff / fair * 100) if (diff is not None and fair) else None
    cells = [("Billed amount", inr(billed, 2), ""),
             ("Fair cost", inr(fair, 2), f"Likely {inr(result.get('model1_interval_lower'))} to {inr(result.get('model1_interval_upper'))}"),
             ("Difference", ("+" if (diff or 0) > 0 else "") + inr(diff, 2), f"{signed_pct(pct)} against fair cost" if pct is not None else ""),
             ("Overbilling risk", f"{(result.get('model2_proba') or 0):.1%}", f"Flag level {FLAG_THRESHOLD:.0%}")]
    kf = Table([[[P(a, label), Spacer(1, 3), P(b, big), Spacer(1, 2), P(c, small)] for a, b, c in cells]],
               colWidths=[W / 4] * 4)
    kf.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.75, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.75, LINE),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 8),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("LEFTPADDING", (0, 0), (-1, -1), 9)]))
    story += [kf]

    # --- why
    story.append(Paragraph("Why this decision", h2))
    rows = []
    for r in result.get("reasons") or []:
        lab, fg, bg = STATUS.get(r.get("status"), STATUS["info"])
        tag = Table([[Paragraph(lab, st("tag", fontName="DejaVu-Bold", fontSize=7.5, leading=9, textColor=fg, alignment=1))]],
                    colWidths=[15 * mm])
        tag.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.6, fg),
                                 ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        rows.append([tag, [P(r.get("title", ""), bold), P(r.get("detail", ""), small)]])
    if rows:
        wt = Table(rows, colWidths=[20 * mm, W - 20 * mm])
        wt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE),
                                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                ("LEFTPADDING", (0, 0), (0, -1), 0)]))
        story.append(wt)

    # --- invoice details
    def kv_table(pairs, widths=(0.32, 0.68)):
        t = Table([[P(k, label), P(v)] for k, v in pairs], colWidths=[W * widths[0], W * widths[1]])
        t.setStyle(TableStyle([("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, ALT]),
                               ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
        return t

    route = " to ".join(x for x in [trip.get("origin") or f.get("origin"), trip.get("destination") or f.get("destination")] if x)
    details = [("Invoice number", f.get("invoice_id") or "—"), ("Invoice date", f.get("invoice_date") or "—"),
               ("Order / LR no.", f.get("order_id") or "—"), ("Carrier ID", f.get("vendor_id") or "—"),
               ("Route", route or "—"),
               ("Truck, goods", ", ".join(x for x in [trip.get("truck_type"), trip.get("product_category")] if x) or "—"),
               ("Base freight", inr(f.get("freight_base"), 2)), ("Detention", inr(f.get("detention"), 2)),
               ("Toll", inr(f.get("toll"), 2)), ("Invoice total", inr(f.get("total"), 2)),
               ("Distance", f"{trip['distance_km']:,.0f} km" if trip.get("distance_km") else "—"),
               ("Transit days", f"{f.get('actual_days', '—')} actual, {trip.get('quoted_days') or '—'} quoted")]
    half = (len(details) + 1) // 2
    grid = [[P(k1, label), P(v1), P(k2, label), P(v2)]
            for (k1, v1), (k2, v2) in zip(details[:half], details[half:] + [("", "")])]
    dt = Table(grid, colWidths=[W * 0.17, W * 0.33, W * 0.17, W * 0.33])
    dt.setStyle(TableStyle([("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, ALT]),
                            ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("LINEBEFORE", (2, 0), (2, -1), 0.6, LINE),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story += [KeepTogether([Paragraph("Invoice details", h2), dt])]

    # --- fair cost build-up
    drivers = result.get("cost_drivers") or []
    if drivers:
        brows = [[P("Component", label), P("Amount", st("lr", fontSize=8, leading=10, textColor=MUTED, alignment=2))]]
        names = {"fuel_cost": "Fuel (distance ÷ mileage × diesel price)", "tolls": "Tolls (₹1.75 per km)",
                 "driver_allowance": "Driver allowance (₹1,200 per day)", "model_adjustment": "Adjustment learned from past shipments"}
        for d in drivers:
            brows.append([P(names.get(d["feature"], d["feature"].replace("_", " ").title())),
                          P(("+" if d["impact"] > 0 else "") + inr(d["impact"]), st("r", alignment=2))])
        brows.append([P("Fair cost", bold), P(inr(fair, 2), st("rb", fontName="DejaVu-Bold", alignment=2))])
        bt = Table(brows, colWidths=[W * 0.7, W * 0.3])
        bt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE), ("LINEABOVE", (0, -1), (-1, -1), 0.9, INK),
                                ("BOX", (0, 0), (-1, -1), 0.6, LINE), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
        story += [KeepTogether([Paragraph("How the fair cost was calculated", h2),
                                P("The fair cost is the standard trip cost (fuel, tolls and driver pay) adjusted by a model "
                                  "trained on 60,000 past shipments.", small), Spacer(1, 5), bt])]

    # --- detailed explanation + sources
    if result.get("explanation"):
        story += [Paragraph("Model explanation", h2)]
        for para in str(result["explanation"]).split("\n"):
            if para.strip():
                story += [P(para.strip()), Spacer(1, 4)]
    tw = ctx.get("transit_weather") or {}
    sources = [("Diesel price used", f"₹{ctx['expected_fuel_price']:.2f} per litre" if ctx.get("expected_fuel_price") else "—"),
               ("Weather during the trip", (f"{tw.get('rain_mm_total')} mm rain, wind up to {tw.get('max_wind_kmh')} km/h, "
                                            f"up to {tw.get('max_temp_c')}°C ({tw.get('source')}, {tw.get('start')} to {tw.get('end')})")
                if tw else (ctx.get("adverse_weather_source") or "—")),
               ("Invoice read by", result.get("extractor_model") or
                ("Entered by hand" if result.get("extractor_used") in (None, "manual") else result["extractor_used"])),
               ("Entered or changed by reviewer", ", ".join(x.replace("_", " ") for x in result.get("manually_entered_fields") or []) or "None")]
    sources += [("Note", n) for n in ctx.get("notes") or []]
    story += [KeepTogether([Paragraph("Data used", h2), kv_table(sources)])]

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(MUTED)
        canvas.setStrokeColor(LINE)
        canvas.line(16 * mm, 13.5 * mm, A4[0] - 16 * mm, 13.5 * mm)
        canvas.drawString(16 * mm, 9.5 * mm, "RateRight supports, but does not replace, a reviewer's judgement. "
                                             "Check flagged invoices against the original documents.")
        canvas.drawRightString(A4[0] - 16 * mm, 9.5 * mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue(), ref
