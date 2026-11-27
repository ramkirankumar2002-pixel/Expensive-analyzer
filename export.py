# export.py - the downloadable monthly report: ONE self-contained HTML file (no Streamlit in here, so it is
# easy to test). All the numbers come from analytics.py, insights.py and goals.py; this file only lays them out.
#
# Self-contained = the CSS and the Plotly library are written inside the file, so it opens offline and
# never loads anything from the internet.
#
# Privacy rule: EVERY piece of text that came from the user or the AI goes through clean() before it is
# written, which masks account numbers and UPI IDs again (parser.py already masked the statement, but file
# names, goal names and the AI's text never went through the parser).

import datetime as dt
import html
import os
import re

import pandas as pd
import plotly.express as px

import analytics as an
import goals as gl
import insights as ins
import theme as th
from parser import mask_sensitive

TOP_MERCHANTS = 10

# 9 or more digits, also when written in groups: '5010 0123 4567 89' or '5010-0123-4567-89' (any length)
LONG_ID_PATTERN = re.compile(r"\d(?:[ -]?\d){8,}")

STATUS_TEXT = {"ok": "OK", "warning": "Warning", "over": "Over budget"}
# status -> CSS class (budgets and goals share the three colours)
STATUS_CLASS = {"ok": "good", "warning": "warn", "over": "bad",
                gl.ON_TRACK: "good", gl.SLIGHTLY_BEHIND: "warn", gl.OFF_TRACK: "bad"}

# A light theme, because the report is meant to be printed
CSS = """
* { box-sizing: border-box; }
body { margin: 0; background: #f3f4f6; color: #111827; font: 14px/1.5 "Segoe UI", Arial, sans-serif; }
.page { max-width: 780px; margin: 24px auto; background: #fff; padding: 36px 40px; border-radius: 12px; }
h1 { margin: 0; font-size: 26px; }
h2 { font-size: 17px; margin: 28px 0 10px; padding-bottom: 6px; border-bottom: 2px solid #0f766e;
     break-after: avoid; }
.muted { color: #6b7280; font-size: 12.5px; }
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }
.card { border: 1px solid #d1d5db; border-radius: 10px; padding: 12px 14px; }
.card .name { color: #6b7280; font-size: 12px; }
.card .value { font-size: 22px; font-weight: 700; }
table { width: 100%; border-collapse: collapse; margin-top: 6px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #e5e7eb; vertical-align: top;
         overflow-wrap: anywhere; }
th { font-size: 12px; color: #374151; background: #f9fafb; }
td.num, th.num { text-align: right; white-space: nowrap; }
tr { break-inside: avoid; }
tr.total td { font-weight: 700; border-top: 2px solid #9ca3af; }
.chart { overflow-x: auto; break-inside: avoid; }
.good { color: #047857; font-weight: 600; }
.warn { color: #b45309; font-weight: 600; }
.bad  { color: #b91c1c; font-weight: 600; }
.bar { height: 7px; background: #e5e7eb; border-radius: 99px; overflow: hidden; margin-top: 4px; }
.bar div { height: 100%; background: #0f766e; }
li { margin-bottom: 4px; }
footer { margin-top: 30px; }

@page { size: A4; margin: 12mm; }
@media print {
  body { background: #fff; font-size: 12px; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .page { max-width: none; margin: 0; padding: 0; border-radius: 0; }
  .chart { overflow: visible; }
}
@media (max-width: 600px) {
  .page { margin: 0; padding: 20px 16px; border-radius: 0; }
  .cards { grid-template-columns: 1fr 1fr; }
}
"""


def clean(text):
    """Mask account numbers / UPI IDs, then escape the text so it is safe inside HTML."""
    def hide_digits(match):
        digits = re.sub(r"\D", "", match.group())
        return "*" * (len(digits) - 4) + digits[-4:]        # keep only the last 4 digits
    # long numbers first: mask_sensitive alone only handles runs of up to 18 digits
    masked = mask_sensitive(LONG_ID_PATTERN.sub(hide_digits, str(text)))
    return html.escape(masked)


def report_filename(month):
    """month = any date inside it -> 'expense-report-2026-05.html'."""
    return f"expense-report-{pd.Timestamp(month):%Y-%m}.html"


# =================================================================================================
# Small HTML helpers
# =================================================================================================
def html_table(headers, rows, numeric=(), total=None):
    """headers = column titles; rows = lists of cells that are ALREADY safe HTML (use clean() for text).
    numeric = the column numbers to right-align; total = an optional bold last row."""
    def line(cells, tag="td", css=""):
        body = ""
        for i, cell in enumerate(cells):
            align = ' class="num"' if i in numeric else ""
            body += f"<{tag}{align}>{cell}</{tag}>"
        return f"<tr{css}>{body}</tr>"
    body = "".join(line(row) for row in rows)
    if total:
        body += line(total, css=' class="total"')
    return f"<table><thead>{line(headers, 'th')}</thead><tbody>{body}</tbody></table>"


def section(section_id, title, body):
    return f'<section id="{section_id}"><h2>{html.escape(title)}</h2>{body}</section>'


def note(text):
    """A small grey line (explanations and 'nothing to show' messages)."""
    return f'<p class="muted">{html.escape(text)}</p>'


# =================================================================================================
# One function per section
# =================================================================================================
def header_section(table, in_month, period):
    """Title, date range and the accounts (sources) in the statement."""
    first, last = in_month["date"].min(), in_month["date"].max()
    out = (f'<header id="header"><h1>Expense report: {period.strftime("%B %Y")}</h1>'
           f'<div class="muted">Transactions from {first:%d %b %Y} to {last:%d %b %Y} '
           f'({len(in_month)} transactions)</div>')
    if "source" in table.columns:
        counts = in_month["source"].value_counts()
        items = [f"{clean(s)} ({counts.get(s, 0)} transactions)" for s in sorted(table["source"].unique())]
        out += f'<div class="muted">Sources included: {", ".join(items)}</div>'
    return out + "</header>"


def summary_section(in_month):
    s = an.summary(in_month)
    savings_pct = "-" if s["savings_pct"] is None else f"{s['savings_pct']:.1f}%"
    cards = [("Spent", an.inr(s["spent"])), ("Income", an.inr(s["income"])), ("Savings %", savings_pct)]
    body = "".join(f'<div class="card"><div class="name">{name}</div><div class="value">{value}</div></div>'
                   for name, value in cards)
    text = (f"Savings (income - spent): {an.inr(s['savings'])}. Transfers out: {an.inr(s['transfers'])}. "
            "Spending leaves out Transfers; self transfers between your own accounts count as neither "
            "spending nor income.")
    return f'<section id="summary"><div class="cards">{body}</div>{note(text)}</section>'


def category_chart(totals):
    """A Plotly donut written as HTML. staticPlot = a plain picture (no zoom or hover buttons).
    The Plotly library itself is put inside the file, which is why the report is a few MB."""
    data = totals.assign(amount=totals["amount"].round(2))
    colours = {c: colour for c, (_, colour) in th.CATEGORY_STYLE.items()}
    fig = px.pie(data, names="category", values="amount", hole=0.55, color="category",
                 color_discrete_map=colours)
    fig.update_traces(textinfo="percent", textposition="inside")
    # a fixed width that fits on an A4 page (a self-resizing chart prints badly)
    fig.update_layout(width=660, height=340, margin=dict(l=10, r=10, t=10, b=10),
                      uniformtext_minsize=10, uniformtext_mode="hide")
    return fig.to_html(full_html=False, include_plotlyjs=True, div_id="category-chart",
                       config={"staticPlot": True})


def category_section(in_month):
    totals = an.category_totals(in_month)
    if totals.empty or totals["amount"].sum() == 0:
        return section("categories", "Spending by category", note("No spending in this month."))
    total = totals["amount"].sum()
    rows = [[clean(r.category), an.inr(r.amount), f"{r.amount / total:.1%}"] for r in totals.itertuples()]
    table = html_table(["Category", "Amount", "Share"], rows, numeric=(1, 2),
                       total=["Total", an.inr(total), "100%"])
    return section("categories", "Spending by category",
                   f'<div class="chart">{category_chart(totals)}</div>{table}')


def merchants_section(in_month):
    top = an.top_merchants(in_month, n=TOP_MERCHANTS)
    if top.empty:
        body = note("No spending in this month.")
    else:
        rows = [[clean(r.merchant), str(r.count), an.inr(r.amount)] for r in top.itertuples()]
        body = html_table(["Merchant", "Payments", "Amount"], rows, numeric=(1, 2))
    return section("merchants", "Top merchants", body)


def unusual_section(table, period):
    """Judged against the whole statement (like the app); only this month's are listed."""
    odd = ins.unusual_spends(table)
    odd = odd[pd.to_datetime(odd["date"]).dt.to_period("M") == period]
    if odd.empty:
        body = note("Nothing unusual found in this month (or too few transactions to tell).")
    else:
        rows = [[f"{r.date:%d %b %Y}", clean(r.merchant), clean(r.category), an.inr(r.amount), clean(r.reason)]
                for r in odd.itertuples()]
        body = html_table(["Date", "Merchant", "Category", "Amount", "Why"], rows, numeric=(3,))
    return section("unusual", "Unusual spends", body)


def subscriptions_section(table):
    subs = ins.find_subscriptions(table)
    if subs.empty:
        body = note("No recurring payments found.")
    else:
        rows = [[clean(r.merchant), clean(r.category), an.inr(r.monthly_amount), str(r.payments),
                 f"{r.last_paid:%d %b %Y}"] for r in subs.itertuples()]
        body = html_table(["Merchant", "Category", "Per month", "Payments", "Last paid"], rows, numeric=(2, 3),
                          total=["Total", "", an.inr(ins.subscription_total(subs)), "", ""])
        body += note("Detected across the whole statement, not only this month.")
    return section("subscriptions", "Subscriptions and recurring payments", body)


def budget_section(table, period, budgets):
    status = ins.budget_status(table, period.to_timestamp(), budgets or {})
    if status.empty:
        body = note("No budgets set.")
    else:
        rows = [[clean(r.category), an.inr(r.budget), an.inr(r.spent), f"{r.pct:.0f}%",
                 f'<span class="{STATUS_CLASS[r.status]}">{STATUS_TEXT[r.status]}</span>']
                for r in status.itertuples()]
        body = html_table(["Category", "Budget", "Spent", "Used", "Status"], rows, numeric=(1, 2, 3))
    return section("budgets", "Budget status", body)


def goals_section(table, goals, today):
    """The same numbers as the Goals page: they describe TODAY, not the month of the report."""
    if not goals:
        return section("goals", "Goal progress", note("No savings goals set."))
    savings = gl.recent_savings(table)["savings"]
    rows = []
    for goal in goals:
        r = gl.evaluate(goal, savings, today)
        progress = (f'{an.inr(goal["saved"])} of {an.inr(goal["target"])} ({r["progress"]:.0%})'
                    f'<div class="bar"><div style="width:{r["progress"] * 100:.1f}%"></div></div>')
        due = pd.Timestamp(goal["target_date"])
        status = "Reached" if r["reached"] else r["status"]
        css = "good" if r["reached"] else STATUS_CLASS[r["status"]]
        rows.append([clean(goal["name"]), progress, f"{due:%d %b %Y}", an.inr(r["needed"]),
                     f'<span class="{css}">{status}</span>'])
    body = html_table(["Goal", "Progress", "Target date", "Needed per month", "Status"], rows, numeric=(3,))
    body += note(f"As of {today:%d %b %Y}, based on average savings of {an.inr(savings)} per month "
                 "(recent months of the whole statement).")
    return section("goals", "Goal progress", body)


def forecast_section(table, period):
    """The forecast uses only the data up to the end of the report's month, so a report for an older
    month shows what could be predicted at that time."""
    up_to_month = table[table["date"].dt.to_period("M") <= period]
    # If later months exist, this month is over: it must not be treated as 'ends mid-month'
    month_is_over = bool((table["date"].dt.to_period("M") > period).any())
    history, forecast, dropped_partial = ins.forecast_next_month(up_to_month, month_is_complete=month_is_over)
    if forecast.empty:
        return section("forecast", "Forecast for next month",
                       note("Need at least 3 complete months of spending to forecast."))
    predicted_month = (history.index[-1] + pd.DateOffset(months=1)).strftime("%B %Y")
    rows = [[clean(r.category), an.inr(r.predicted)] for r in forecast.itertuples()]
    body = html_table(["Category", f"Predicted for {predicted_month}"], rows, numeric=(1,),
                      total=["Total", an.inr(forecast["predicted"].sum())])
    text = "This is only an estimate based on past trends, not a guarantee."
    if dropped_partial:
        text += (f" {period.strftime('%B %Y')} is incomplete, so it was left out and the estimate is "
                 f"for {predicted_month}.")
    return section("forecast", "Forecast for next month", body + note(text))


def ai_section(ai_report):
    """ai_report = {'summary': str, 'tips': [str]} from report.generate_report, or None."""
    if not ai_report:
        return ""                                   # not generated: leave the section out
    body = f"<p>{clean(ai_report['summary'])}</p>"
    if ai_report["tips"]:
        body += "<ul>" + "".join(f"<li>{clean(tip)}</li>" for tip in ai_report["tips"]) + "</ul>"
    return section("ai", "AI summary and tips", body + note("Written by Gemini from aggregated numbers only."))


# =================================================================================================
# Putting it together
# =================================================================================================
def build_report_html(table, month, budgets=None, goals=None, ai_report=None, today=None):
    """The whole report as one HTML string.
    table = the categorized statement (all months; with a 'source' column when it has one),
    month = any date inside the month to report. Raises ValueError if that month has no transactions."""
    period = pd.Timestamp(month).to_period("M")
    in_month = table[table["date"].dt.to_period("M") == period]
    if in_month.empty:
        raise ValueError(f"There are no transactions in {period.strftime('%B %Y')}.")
    today = today or dt.date.today()

    parts = [
        header_section(table, in_month, period),
        summary_section(in_month),
        category_section(in_month),
        merchants_section(in_month),
        unusual_section(table, period),
        subscriptions_section(table),
        budget_section(table, period, budgets),
        goals_section(table, goals, today),
        forecast_section(table, period),
        ai_section(ai_report),
        f'<footer class="muted">Generated on {today:%d %b %Y} by Personal Expense Analyzer. '
        "Account numbers and UPI IDs are masked.</footer>",
    ]
    return ('<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Expense report {period.strftime("%B %Y")}</title><style>{CSS}</style></head>'
            f'<body><div class="page">{"".join(parts)}</div></body></html>')


def save_report(folder, table, month, **options):
    """Write the report into a folder and return the file path. (The app does not need this: it sends the
    HTML straight to the browser, so nothing is written on the server, also on Streamlit Cloud.)"""
    path = os.path.join(folder, report_filename(month))
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report_html(table, month, **options))
    return path
