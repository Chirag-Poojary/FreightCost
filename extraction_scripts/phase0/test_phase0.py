"""Sanity tests for the Phase 0 kit. Run: python test_phase0.py"""
from evaluate import evaluate, format_report
from gate import run_gate
from make_splits import make_splits
from schema import BAD, json_schema, norm_amount, norm_date, norm_id, norm_text


def test_normalisers():
    assert norm_amount("\u20b91,570.50") == 1570.5
    assert norm_amount("Rs. 15,700") == 15700.0
    assert norm_amount("abc") == BAD
    assert norm_amount("") is None
    assert norm_date("15-Mar-2025") == "2025-03-15"
    assert norm_date("15/03/2025") == "2025-03-15"
    assert norm_id(" inv 000001 ") == "INV000001"
    assert norm_text("Navi-Mumbai ") == "navi mumbai"


def test_gate():
    ok = {"order_id": "O1", "vendor_id": "V1", "total": 1000, "actual_days": 3,
          "freight_base": 900, "detention": 50, "toll": 50}
    assert run_gate(ok).passed
    comma_misread = dict(ok, total=1_000_000)
    r = run_gate(comma_misread)
    assert not r.passed and "arithmetic_mismatch" in r.reasons and "total_out_of_bounds" in r.reasons
    assert "missing:order_id" in run_gate(dict(ok, order_id=None)).reasons
    assert "unknown_order_id" in run_gate(ok, order_ids={"X"}).reasons


def _doc(layout, **kw):
    base = {"order_id": "O1", "vendor_id": "V1", "total": 1000.0, "actual_days": 3,
            "freight_base": 900.0, "detention": 50.0, "toll": 50.0}
    base.update(kw)
    return {"layout": layout, "fields": base}


def test_evaluate_cells():
    gt = {"a": _doc("x"), "b": _doc("x"), "c": _doc("y"), "d": _doc("y")}
    pred = {
        "a": _doc("x"),                                                  # auto + correct
        # consistent but wrong total + base: passes gate, wrong -> SILENT ERROR
        "b": _doc("x", total=1100.0, freight_base=1000.0),
        "c": _doc("y", total=1_000_000.0),                               # wrong, caught by gate
        "d": _doc("y", freight_base=None),                               # uncheckable -> rejected, critical ok
    }
    r = evaluate(gt, pred)
    assert r["gate"]["cells"] == {"auto_correct": 1, "silent_error": 1, "caught": 1, "false_reject": 1}
    assert abs(r["silent_error_rate"]["value"] - 0.25) < 1e-9
    assert abs(r["critical_all_correct"]["value"] - 0.5) < 1e-9
    assert r["by_layout"]["x"]["silent_error_rate"] == 0.5
    format_report(r)  # must not raise


def test_hallucination_and_missing():
    gt = {"a": {"layout": "x", "fields": {"order_id": "O1", "toll": None}}}
    pred = {"a": {"fields": {"order_id": None, "toll": 120}}}
    r = evaluate(gt, pred)
    assert r["per_field"]["order_id"]["missing"] == 1
    assert r["per_field"]["toll"]["hallucinated"] == 1


def test_splits():
    manifest = [{"doc_id": f"{lay}{i}", "layout": lay} for lay in "ABCD" for i in range(20)]
    manifest += [{"doc_id": f"R{i}", "layout": "real"} for i in range(5)]
    s = make_splits(manifest, test_layouts=1, real_layouts=["real"], seed=1)
    assert len(s["test_real"]) == 5
    unseen = set(s["meta"]["unseen_test_layouts"])
    assert len(unseen) == 1
    assert all(d[0] in unseen for d in s["test_unseen_layouts"])
    assert not any(d[0] in unseen for d in s["train"] + s["val"])


def test_json_schema():
    sch = json_schema()
    assert "order_id" in sch["required"] and "line_items" in sch["required"]
    assert sch["properties"]["total"]["type"] == ["number", "null"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all passed")
