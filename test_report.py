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


def test_stats_include_goals_without_names():
    goal = {"id": "1", "name": "Ramesh wedding gift", "target": 5000.0, "target_date": "2099-01-01", "saved": 100.0}
    s = rp.build_report_stats(table(), pd.Timestamp("2026-06-01"), {}, goals=[goal])
    assert s["savings_goals"][0]["target"] == 5000 and s["savings_goals"][0]["status"] == "On track"
    assert "Ramesh" not in json.dumps(s) and "wedding" not in json.dumps(s)
    assert "savings_goals" not in rp.build_report_stats(table(), pd.Timestamp("2026-06-01"), {})


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
    assert out == {"summary": "Fine month.", "tips": ["a", "b", "c", "d"], "language": "en"}      # max 4 tips
    assert '"total_spent": "₹5"' in seen["prompt"]

    assert rp.generate_report({}, ask=lambda p, k: "just text") == {"summary": "just text", "tips": [],
                                                                    "language": "en"}

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

def fake_gemini(seen, reply=None):
    """A stand-in for Gemini that remembers every prompt it gets."""
    os.environ["GEMINI_API_KEY"] = "fake"
    rp.load_dotenv = lambda *a, **k: None

    def ask(prompt, key):
        seen.append(prompt)
        return reply or json.dumps({"summary": "ok", "tips": []})
    return ask


def test_language_goes_into_the_prompt():
    expected = {"en": "simple English", "hi": "Hindi in Devanagari script", "hinglish": "Roman script"}
    assert list(rp.LANGUAGES) == list(expected) and rp.DEFAULT_LANGUAGE == "en"
    for language, words in expected.items():
        seen = []
        out = rp.generate_report({}, language, ask=fake_gemini(seen))
        assert words in seen[0] and out["language"] == language
        assert "Do not translate category names or numbers" in seen[0]
        for other, other_words in expected.items():
            assert other == language or other_words not in seen[0]
    seen = []
    rp.generate_report({}, ask=fake_gemini(seen))                    # no language given = English
    assert "simple English" in seen[0]


def test_hindi_reply_is_kept_as_is():
    reply = json.dumps({"summary": "इस महीने आपने Food पर ₹4,800 खर्च किए।", "tips": ["घर पर खाना बनाइए।"]})
    out = rp.generate_report({}, "hi", ask=fake_gemini([], reply))
    assert out["summary"] == "इस महीने आपने Food पर ₹4,800 खर्च किए।" and out["tips"] == ["घर पर खाना बनाइए।"]


def test_friendly_error_in_every_language():
    fake_gemini([])                                                  # sets the fake key

    def broken(prompt, key):
        raise RuntimeError("something secret broke")
    for language in rp.LANGUAGES:
        try:
            rp.generate_report({}, language, ask=broken)
            assert False, "should have raised"
        except rp.ReportError as e:
            assert str(e).startswith("Couldn't get a report from Gemini") and "secret" not in str(e)


def test_cache_key_has_month_and_language():
    may, june = pd.Period("2026-05"), pd.Period("2026-06")
    keys = {rp.report_cache_key(m, language) for m in (may, june) for language in rp.LANGUAGES}
    assert len(keys) == 6                                            # all different
    assert rp.report_cache_key(may, "hi") == rp.report_cache_key(may, "hi") == "report_2026-05_hi"


def test_prompt_numbers_are_formatted():
    stats = {"total_spent": 125000, "spent_by_category": {"Food": 4800},
             "change_vs_last_month_pct": {"total": -12.5, "by_category": {"Food": None}},
             "savings_pct": 90.0, "subscriptions": {"count": 3, "total_per_month": 1500},
             "budget_alerts": [{"category": "Food", "budget": 5000, "spent": 4800, "used_pct": 96,
                                "status": "warning"}],
             "savings_goals": [{"target": 100000, "months_left": 5, "status": "On track"}]}
    out = rp.format_for_prompt(stats)
    assert out["total_spent"] == "₹1,25,000" and out["spent_by_category"] == {"Food": "₹4,800"}
    assert out["change_vs_last_month_pct"] == {"total": "-12.5%", "by_category": {"Food": None}}
    assert out["savings_pct"] == "90.0%" and out["subscriptions"] == {"count": 3, "total_per_month": "₹1,500"}
    assert out["budget_alerts"] == [{"category": "Food", "budget": "₹5,000", "spent": "₹4,800",
                                     "used_pct": "96%", "status": "warning"}]
    assert out["savings_goals"] == [{"target": "₹1,00,000", "months_left": 5, "status": "On track"}]
    assert stats["total_spent"] == 125000                            # the original is not changed


def test_prompt_has_no_merchant_names_in_any_language():
    goal = {"id": "1", "name": "Wedding gift", "target": 5000.0, "target_date": "2099-01-01", "saved": 100.0}
    stats = rp.build_report_stats(table(), pd.Timestamp("2026-06-01"), {"Food": 1600}, goals=[goal])
    for language in rp.LANGUAGES:
        seen = []
        rp.generate_report(stats, language, ask=fake_gemini(seen))
        data = seen[0].split("\n\n", 1)[1]                          # the part after the instructions
        assert json.loads(data) == rp.format_for_prompt(stats)
        for secret in ["Swiggy", "order", "Salary", "ACME", "Metro card", "Ramesh", "Sent to", "Wedding"]:
            assert secret not in seen[0], (language, secret)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all report tests passed")
