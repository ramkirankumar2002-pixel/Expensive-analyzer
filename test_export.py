# test_export.py - small tests for export.py (plain asserts). Run with:  python test_export.py

import os
import re
import tempfile

import pandas as pd

import export as ex
import insights as ins
import merger as mg
from parser import parse_statement

SECTIONS = ["header", "summary", "categories", "merchants", "unusual", "subscriptions",
            "budgets", "goals", "forecast"]
TODAY = pd.Timestamp("2026-11-27").date()
AI = {"summary": "A steady month.", "tips": ["Cook at home twice more."]}

# Full IDs that must never appear in a report
SECRETS = ["50100123456789", "rahul.sharma@okaxis", "123456789012345678901234", "5010 0123 4567 89",
           "9876543210@ybl", "00112233445566"]


def table():
    """Jan to May 2026: monthly salary, rent and Netflix, plus food."""
    rows = []
    for m in range(1, 6):
        rows += [(f"2026-{m:02d}-01", "Salary", 50000, "credit", "Income"),
                 (f"2026-{m:02d}-03", "Landlord", 12000, "debit", "Rent"),
                 (f"2026-{m:02d}-05", "Netflix", 649, "debit", "Entertainment"),
                 (f"2026-{m:02d}-12", "Swiggy", 1000 * m, "debit", "Food"),
                 (f"2026-{m:02d}-30" if m != 2 else "2026-02-27", "Metro", 500, "debit", "Travel")]
    df = pd.DataFrame(rows, columns=["date", "merchant", "amount", "type", "category"])
    df["date"] = pd.to_datetime(df["date"])
    df["description"] = df["merchant"]
    return df


def without_scripts(page):
    """The page minus the Plotly library and chart code, i.e. what the reader sees (plus the CSS)."""
    return re.sub(r"<script\b.*?</script>", "", page, flags=re.DOTALL)


def full_report(**changes):
    options = {"budgets": {"Food": 4000, "Travel": 2000}, "ai_report": AI, "today": TODAY,
               "goals": [{"id": "1", "name": "Laptop", "target": 60000.0,
                          "target_date": "2027-11-01", "saved": 6000.0}]}
    options.update(changes)
    return ex.build_report_html(table(), "2026-05-01", **options)


def test_file_is_created():
    with tempfile.TemporaryDirectory() as folder:
        path = ex.save_report(folder, table(), pd.Timestamp("2026-05-15"), today=TODAY)
        assert os.path.basename(path) == "expense-report-2026-05.html"
        with open(path, encoding="utf-8") as f:
            page = f.read()
        assert page.startswith("<!DOCTYPE html>") and page.endswith("</html>")


def test_has_all_sections():
    page = full_report()
    for name in SECTIONS + ["ai"]:
        assert f'id="{name}"' in page, name
    visible = without_scripts(page)
    for text in ["May 2026", "01 May 2026 to 30 May 2026", "Savings %", "Netflix", "Laptop",
                 "Over budget", "Predicted for June 2026", "A steady month.", "Cook at home twice more."]:
        assert text in visible, text
    assert 'id="category-chart"' in page and "Plotly.newPlot" in page


def test_ai_section_only_when_generated():
    assert 'id="ai"' not in full_report(ai_report=None)


def test_hindi_report_is_written_as_utf8():
    ai = {"summary": "इस महीने आपने Food पर ₹1,25,000 (42.5%) खर्च किए।",
          "tips": ["घर पर खाना बनाइए।"], "language": "hi"}
    with tempfile.TemporaryDirectory() as folder:
        path = ex.save_report(folder, table(), "2026-05-01", ai_report=ai, today=TODAY)
        with open(path, "rb") as f:
            raw = f.read()
    for text in [ai["summary"], ai["tips"][0]]:
        assert text.encode("utf-8") in raw, text            # real Devanagari bytes, not masked or escaped
    page = raw.decode("utf-8")
    assert '<meta charset="utf-8">' in page and '<div lang="hi">' in page
    assert '"Nirmala UI", "Noto Sans Devanagari", sans-serif' in ex.CSS
    assert "<link" not in without_scripts(page) and "@font-face" not in ex.CSS


def test_hinglish_report_language_tag():
    ai = {"summary": "Is mahine aapne Food par ₹4,800 kharch kiye.", "tips": [], "language": "hinglish"}
    page = full_report(ai_report=ai)
    assert '<div lang="hi-Latn"><p>' in page and ai["summary"] in page


def test_empty_extras_still_give_all_sections():
    page = full_report(budgets={}, goals=[], ai_report=None)
    for name in SECTIONS:
        assert f'id="{name}"' in page, name
    assert "No budgets set." in page and "No savings goals set." in page


def test_self_contained():
    page = full_report()
    # nothing is loaded from outside: no <script src>, <link>, <img>, <iframe> or CSS imports.
    # (The Plotly library's own code contains web addresses as text, but a pie chart never loads them.)
    visible = without_scripts(page)
    assert not re.search(r'<script[^>]*\ssrc\s*=', page)
    for bad in ["<link", "<img", "<iframe", "@import", "url(", "http://", "https://"]:
        assert bad not in visible, bad
    assert len(page) > 1_000_000                          # the Plotly library really is inside


def test_clean_masks_ids_and_escapes():
    for secret in SECRETS:
        assert secret not in ex.clean(f"pay {secret} now"), secret
    assert ex.clean("acct 50100123456789") == "acct **********6789"
    assert ex.clean("<b>x</b>") == "&lt;b&gt;x&lt;/b&gt;"
    assert ex.clean("Big Bazaar 24") == "Big Bazaar 24"


def test_no_unmasked_ids_anywhere():
    """Put raw IDs into every text that can reach the report, even ones the parser would have masked."""
    df = table()
    df.loc[df["merchant"] == "Swiggy", "merchant"] = "rahul.sharma@okaxis 50100123456789"
    df.loc[df["merchant"] == "Netflix", "merchant"] = "Netflix 9876543210@ybl"
    df["description"] = "UPI/00112233445566/" + df["merchant"]
    df["source"] = "stmt_123456789012345678901234"          # a file name used as the source label
    goals = [{"id": "1", "name": "Pay 5010 0123 4567 89", "target": 100.0,
              "target_date": "2027-01-01", "saved": 0.0}]
    ai = {"summary": "Account 50100123456789.", "tips": ["Ask rahul.sharma@okaxis"]}
    page = ex.build_report_html(df, "2026-05-01", {"Food": 10}, goals, ai, today=TODAY)
    for secret in SECRETS:
        assert secret not in page, secret
    visible = without_scripts(page)
    assert not re.search(r"\d{9,}", visible)
    assert not re.search(r"[A-Za-z0-9._-]{3,}@[A-Za-z]{2,}", visible.replace("@media", "").replace("@page", ""))
    assert "ra****@okaxis" in visible and "**********6789" in visible


def test_multiple_statements():
    """The two sample bank files, merged: both sources are listed and no raw ID survives."""
    here = os.path.dirname(os.path.abspath(__file__))
    paths = {name: os.path.join(here, "data", f"sample_{name}.csv") for name in ("bank_a", "bank_b")}
    merged, _ = mg.merge_statements([(name, parse_statement(p)) for name, p in paths.items()])
    merged["category"] = "Others"
    merged.loc[merged["type"] == "credit", "category"] = "Income"
    merged.loc[merged["self_transfer"], "category"] = "Self transfer"
    month = merged["date"].max()
    page = ex.build_report_html(merged, month, today=TODAY)
    assert "Sources included: bank_a (" in page and "bank_b (" in page
    raw = "".join(open(p, encoding="utf-8").read() for p in paths.values())
    for secret in re.findall(r"\d{9,}|[A-Za-z0-9._-]+@[A-Za-z]+", raw):
        assert secret not in without_scripts(page), secret


def test_older_month_uses_only_data_up_to_then():
    df = table()
    page = ex.build_report_html(df, "2026-04-01", today=TODAY)
    assert "Predicted for May 2026" in page and "is incomplete" not in page
    changed = df.copy()
    changed.loc[(changed["date"].dt.month == 5) & (changed["category"] == "Food"), "amount"] = 999999
    assert ex.forecast_section(changed, pd.Period("2026-04")) == ex.forecast_section(df, pd.Period("2026-04"))
    # an older month whose last payment is early in the month is still complete
    quiet = df[~((df["date"].dt.month == 4) & (df["date"].dt.day > 20))]
    assert "Predicted for May 2026" in ex.forecast_section(quiet, pd.Period("2026-04"))
    # but if the statement itself stops mid-month, that month is left out
    cut = df[df["date"] <= "2026-05-12"]
    assert "May 2026 is incomplete" in ex.forecast_section(cut, pd.Period("2026-05"))


def test_forecast_flag_default_is_unchanged():
    cut = table()[lambda d: d["date"] <= "2026-05-12"]
    assert ins.forecast_next_month(cut)[2] is True
    assert ins.forecast_next_month(cut, month_is_complete=True)[2] is False


def test_month_without_data_is_an_error():
    try:
        ex.build_report_html(table(), "2020-01-01")
        assert False, "should have raised"
    except ValueError as error:
        assert "January 2020" in str(error)


def test_month_with_only_income():
    df = table()[lambda d: d["type"] == "credit"]
    page = ex.build_report_html(df, "2026-05-01", today=TODAY)
    assert "No spending in this month." in page
    for name in SECTIONS:
        assert f'id="{name}"' in page, name


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All export tests passed.")
