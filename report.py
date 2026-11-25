# report.py - the AI monthly report: build a small summary of AGGREGATED numbers, send it to Gemini.
# (No Streamlit in here, so it is easy to test.)
#
# Privacy rule: only totals, percentages and counts leave the computer. No transaction
# descriptions, no merchant names, no account numbers or UPI IDs.

import json
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv

import analytics as an
import insights as ins

MODEL_NAME = "gemini-3.5-flash-lite"   # model names get retired often: change it here if Gemini returns 404


class ReportError(Exception):
    """A problem we can show to the user as-is (message is already friendly)."""


def _pct_change(now, before):
    """% change from before to now; None when there is nothing to compare with."""
    return None if not before else round((now - before) / before * 100, 1)


def build_report_stats(df, month, budgets):
    """Aggregated stats for one month (month = any date inside it). Returns a plain dict."""
    period = pd.Timestamp(month).to_period("M")
    in_month = df[df["date"].dt.to_period("M") == period]
    in_prev = df[df["date"].dt.to_period("M") == period - 1]

    totals = an.category_totals(in_month).set_index("category")["amount"]
    prev_totals = an.category_totals(in_prev).set_index("category")["amount"]
    total, prev_total = float(totals.sum()), float(prev_totals.sum())
    has_prev = not in_prev.empty

    summary = an.summary(in_month)
    odd = ins.unusual_spends(df)                          # judged against the whole statement...
    odd = odd[pd.to_datetime(odd["date"]).dt.to_period("M") == period]     # ...but only this month's are reported
    subs = ins.find_subscriptions(df)
    status = ins.budget_status(df, period.to_timestamp(), budgets) if budgets else pd.DataFrame()
    alerts = status[status["status"] != "ok"] if not status.empty else status

    return {
        "month": period.strftime("%B %Y"),
        "total_spent": round(total),
        "spent_by_category": {c: round(float(a)) for c, a in totals.items()},
        "change_vs_last_month_pct": {
            "total": _pct_change(total, prev_total) if has_prev else None,
            "by_category": {c: _pct_change(float(a), float(prev_totals.get(c, 0))) for c, a in totals.items()}
                           if has_prev else {},
        },
        "savings_pct": None if summary["savings_pct"] is None else round(summary["savings_pct"], 1),
        # category and amount only: the merchant name is left out on purpose
        "unusual_spends": [{"category": r.category, "amount": round(float(r.amount))} for r in odd.itertuples()],
        "subscriptions": {"count": len(subs), "total_per_month": round(ins.subscription_total(subs))},
        "budget_alerts": [{"category": r.category, "budget": round(r.budget), "spent": round(r.spent),
                           "used_pct": round(r.pct), "status": r.status} for r in alerts.itertuples()],
    }


def _ask_gemini(prompt, api_key):
    """Send the prompt to Gemini and return its text. Imported here so tests never need the package."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))   # 30 s (in ms): never hang the page
    response = client.models.generate_content(
        model=MODEL_NAME, contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"))      # ask for JSON back
    return response.text


def _is_temporary(error):
    """True for errors that usually go away on a retry: timeouts, overload (503), rate limit (429), network drops."""
    text = f"{type(error).__name__} {error}".lower()
    return any(w in text for w in ("timeout", "timed out", "503", "429", "500", "unavailable",
                                   "overloaded", "connect", "resource_exhausted"))


def _friendly_message(error):
    """Turn a Gemini/network error into a short message that says what to fix. Never includes raw details."""
    text = f"{type(error).__name__} {error}".lower()
    if any(w in text for w in ("api key", "api_key", "401", "403", "permission", "unauthenticated")):
        return "Gemini rejected the API key. Check GEMINI_API_KEY in the .env file and restart the app."
    if any(w in text for w in ("404", "not found", "not_found")):
        return f"The Gemini model '{MODEL_NAME}' was not found. Change MODEL_NAME at the top of report.py."
    if any(w in text for w in ("429", "quota", "resource_exhausted")):
        return "Gemini's free quota or rate limit was reached. Wait a minute and try again."
    if _is_temporary(error):
        return "Gemini is busy or the connection timed out. Check your internet and try again in a moment."
    return "Couldn't get a report from Gemini. Check the server console for the reason, then try again."


def get_api_key():
    """The Gemini key: Streamlit secrets when deployed (Streamlit Cloud), otherwise the local .env file."""
    try:
        import streamlit as st
        key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()     # raises if there is no secrets file
    except Exception:                                               # running locally without secrets.toml
        key = ""
    if key:
        return key
    load_dotenv()
    return os.getenv("GEMINI_API_KEY", "").strip()


def generate_report(stats, ask=None):
    """Turn the stats into {'summary': str, 'tips': [str, ...]} using Gemini.
    ask(prompt, api_key) -> text can be swapped out in tests. Raises ReportError with a friendly message."""
    api_key = get_api_key()
    if not api_key:
        raise ReportError("No Gemini API key found. Add GEMINI_API_KEY to the .env file (local) "
                          "or to the app's Secrets (Streamlit Cloud), then restart the app.")

    prompt = (
        "You are a friendly personal finance coach. Below are aggregated spending stats for one month "
        "(amounts in Indian rupees). Reply ONLY with JSON of the form "
        '{"summary": "2-3 sentences on how the month went", "tips": ["3 to 4 short, personal saving tips"]}. '
        "Mention specific numbers and categories from the stats.\n\n" + json.dumps(stats, ensure_ascii=False)
    )
    ask = ask or _ask_gemini
    try:
        try:
            text = ask(prompt, api_key)
        except Exception as error:
            if not _is_temporary(error):
                raise
            time.sleep(2)                               # busy/slow server: wait a moment and try once more
            text = ask(prompt, api_key)
    except Exception as error:
        # Print the REAL cause in the server console (key removed) so it can be debugged
        print(f"[report] Gemini call failed: {type(error).__name__}: {str(error).replace(api_key, '<key>')[:300]}",
              file=sys.stderr)
        raise ReportError(_friendly_message(error))

    try:
        data = json.loads(text)
        summary = str(data["summary"]).strip()
        tips = [str(t).strip() for t in data["tips"] if str(t).strip()][:4]
    except (ValueError, KeyError, TypeError):           # not the JSON we asked for: show the raw text instead
        return {"summary": str(text).strip(), "tips": []}
    return {"summary": summary, "tips": tips}
