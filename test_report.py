# test_report.py - small tests for report.py (plain asserts). No network: Gemini is replaced by a fake.
# Run with:  python test_report.py

import json
import os

import pandas as pd

import report as rp


def table():
    """May and June: Swiggy food, a salary credit, and a transfer to a person."""
    rows = [("2026-05-02", "Swiggy order 8123", "Swiggy", 1000, "Food", "debit"),
            ("2026-05-03", "Salary ACME", "ACME", 20000, "Income", "credit"),
            ("2026-06-02", "Swiggy order 9911", "Swiggy", 1500, "Food", "debit"),
            ("2026-06-04", "Metro card", "Metro", 500, "Travel", "debit"),
            ("2026-06-05", "Salary ACME", "ACME", 20000, "Income", "credit"),
            ("2026-06-06", "Sent to Ramesh", "Ramesh", 7000, "Transfers", "debit")]
    df = pd.DataFrame(rows, columns=["date", "description", "merchant", "amount", "category", "type"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_stats_numbers():
    s = rp.build_report_stats(table(), pd.Timestamp("2026-06-01"), {"Food": 1600})
    assert s["month"] == "June 2026" and s["total_spent"] == 2000          # transfer not counted
    assert s["spent_by_category"] == {"Food": 1500, "Travel": 500}
    assert s["change_vs_last_month_pct"]["total"] == 100.0                  # 1000 -> 2000
    assert s["change_vs_last_month_pct"]["by_category"]["Food"] == 50.0
    assert s["savings_pct"] == 90.0                                         # (20000 - 2000) / 20000
    assert s["budget_alerts"][0]["category"] == "Food" and s["budget_alerts"][0]["status"] == "warning"


def test_stats_first_month_has_no_comparison():
    s = rp.build_report_stats(table(), pd.Timestamp("2026-05-01"), {})
    assert s["change_vs_last_month_pct"]["total"] is None and s["budget_alerts"] == []


def test_stats_contain_no_raw_text():
    text = json.dumps(rp.build_report_stats(table(), pd.Timestamp("2026-06-01"), {"Food": 1600}))
    for secret in ["Swiggy", "order", "Salary", "ACME", "Metro", "Ramesh", "Sent to"]:
        assert secret not in text


def test_missing_key_is_a_friendly_error():
    old = os.environ.pop("GEMINI_API_KEY", None)
    rp.load_dotenv = lambda *a, **k: None          # don't let a real .env file fill the key back in
    try:
        rp.generate_report({}, ask=lambda p, k: "{}")
        assert False, "should have raised"
    except rp.ReportError as e:
        assert "GEMINI_API_KEY" in str(e)
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old


def test_generate_report_parses_and_falls_back():
    os.environ["GEMINI_API_KEY"] = "fake"
    rp.load_dotenv = lambda *a, **k: None
    seen = {}

    def good(prompt, key):
        seen["prompt"] = prompt
        return json.dumps({"summary": "Fine month.", "tips": ["a", "b", "c", "d", "e"]})
    out = rp.generate_report({"total_spent": 5}, ask=good)
    assert out == {"summary": "Fine month.", "tips": ["a", "b", "c", "d"]}      # max 4 tips
    assert '"total_spent": 5' in seen["prompt"]

    assert rp.generate_report({}, ask=lambda p, k: "just text") == {"summary": "just text", "tips": []}

    def broken(prompt, key):
        raise RuntimeError("401 secret details")
    try:
        rp.generate_report({}, ask=broken)
        assert False, "should have raised"
    except rp.ReportError as e:
        assert "secret" not in str(e)

    # a temporary error (503) is retried once and then succeeds
    rp.time.sleep = lambda s: None
    calls = []

    def flaky(prompt, key):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("503 UNAVAILABLE model overloaded")
        return json.dumps({"summary": "Second try worked.", "tips": []})
    assert rp.generate_report({}, ask=flaky)["summary"] == "Second try worked." and len(calls) == 2

    # a bad key is reported as a bad key, not as "no internet"
    def bad_key(prompt, key):
        raise RuntimeError("403 API key not valid " + key)
    try:
        rp.generate_report({}, ask=bad_key)
    except rp.ReportError as e:
        assert "API key" in str(e) and "fake" not in str(e)


def test_api_key_prefers_secrets_then_env():
    import types
    rp.load_dotenv = lambda *a, **k: None
    os.environ["GEMINI_API_KEY"] = "from-env"
    import streamlit as st
    real = st.secrets
    try:
        st.secrets = {"GEMINI_API_KEY": " from-secrets "}
        assert rp.get_api_key() == "from-secrets"          # deployed: secrets win (and are stripped)
        st.secrets = {}                                     # no secret set -> fall back to the environment / .env
        assert rp.get_api_key() == "from-env"
    finally:
        st.secrets = real


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all report tests passed")
