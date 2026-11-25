# app.py - Streamlit dashboard: upload a statement, see the analysis, fix wrong categories, retrain.
# Run with:  streamlit run app.py
#
# Pages (sidebar):  Dashboard  |  Transactions  |  Insights  |  Model
# The number crunching lives in analytics.py; this file only draws things.

import os

import pandas as pd
import plotly.express as px
import streamlit as st

import analytics as an
import categorizer as cz
import insights as ins
import report as rp
import theme as th
from parser import MissingColumnError, parse_statement

st.set_page_config(page_title="Expense Analyzer", page_icon="💸", layout="wide")
st.markdown(th.CSS, unsafe_allow_html=True)        # the dark fintech look (see theme.py)

# --- Remember things between reruns (Streamlit re-runs this whole file on every click) -------
if "bundle" not in st.session_state:
    st.session_state.bundle = cz.load_model()   # the trained model (loaded once)
if "scores" not in st.session_state:
    st.session_state.scores = cz.messy_scores(st.session_state.bundle)  # accuracy of the current model
st.session_state.setdefault("editor_version", 0)  # bump it to reset the table's pending edits


# The built-in fake statement used by the "Try with sample data" button (path relative to this file)
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "transactions.csv")
PRIVACY_NOTE = "Files are processed in memory for this session; don't upload real statements to the public demo."

# Credits get these two labels instead of a spending category (see categorizer.credit_label)
CREDIT_LABELS = ["Income", "Refund"]


def is_public_demo():
    """True on Streamlit Community Cloud (the app lives under /mount/src there) or when PUBLIC_DEMO=1 is set
    (a top-level Streamlit secret also becomes an environment variable). Locally this is False."""
    on_cloud = os.path.abspath(__file__).replace("\\", "/").startswith("/mount/src")
    return on_cloud or os.environ.get("PUBLIC_DEMO", "").lower() in ("1", "true", "yes")


PUBLIC_DEMO = is_public_demo()


# In the public demo every visitor gets their OWN corrections and budgets, kept in st.session_state
# (gone when the tab closes). Locally they are saved in corrections.csv / budgets.json as before.
def current_corrections():
    """{cleaned description: category} the user has saved."""
    if PUBLIC_DEMO:
        return st.session_state.setdefault("corrections", {})
    return cz.load_corrections()


def save_correction_rows(rows):
    """rows = [(description, category), ...]"""
    if not PUBLIC_DEMO:
        cz.save_corrections(rows)
        return
    for _, category in rows:
        if category not in cz.CATEGORIES:
            raise ValueError(f"'{category}' is not one of the fixed categories")
    current_corrections().update({cz.clean_text(d): c for d, c in rows})


def current_budgets():
    if PUBLIC_DEMO:
        return st.session_state.get("budgets", {})
    return ins.load_budgets()


def save_current_budgets(new):
    if PUBLIC_DEMO:
        st.session_state.budgets = {k: float(v) for k, v in new.items() if float(v) > 0}
    else:
        ins.save_budgets(new)


def categorize(parsed):
    """Rules + ML + the user's saved corrections for every row of the parsed statement."""
    return cz.categorize_dataframe(parsed, bundle=st.session_state.bundle, corrections=current_corrections())


# --- Sidebar: navigation + upload ------------------------------------------------------------
st.sidebar.markdown('<div class="brand"><span class="logo">💸</span><div><div class="name">Expense Analyzer</div>'
                    '<div class="tag">Know where your money goes</div></div></div>', unsafe_allow_html=True)
PAGE_ICONS = {"Dashboard": "📊", "Transactions": "🧾", "Insights": "✨", "Model": "🧠"}
page = st.sidebar.radio("Go to", list(PAGE_ICONS), format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
                        label_visibility="collapsed")
st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)   # pushes the upload box down

uploaded = st.sidebar.file_uploader("📤 Upload your bank / UPI statement", type=["csv", "xlsx", "xls"])
st.sidebar.caption("🔒 " + PRIVACY_NOTE)
if st.sidebar.button("🧪 Try with sample data", help="Loads a fake statement so you can explore the app"):
    st.session_state.source = "sample"                    # the newest choice (button or upload) wins
if uploaded is not None and (uploaded.name, uploaded.size) != st.session_state.get("seen_upload"):
    st.session_state.seen_upload = (uploaded.name, uploaded.size)
    st.session_state.source = "upload"

# `statement` is what we parse: the sample file's path, or the uploaded file
statement = SAMPLE_PATH if st.session_state.get("source") == "sample" else uploaded
if statement is not None:
    file_id = ("sample",) if statement == SAMPLE_PATH else (uploaded.name, uploaded.size)
    if st.session_state.get("file_id") != file_id:       # a new file: parse it once
        try:
            st.session_state.parsed = parse_statement(statement)
        except MissingColumnError as error:
            # No column we recognise: let the user say which one holds the description.
            st.sidebar.warning(f"{error}. Pick the column that describes each transaction.")
            chosen = st.sidebar.selectbox("Description column", error.columns, index=None,
                                          placeholder="Choose a column...")
            if chosen is None:
                st.stop()
            try:
                st.session_state.parsed = parse_statement(statement, description_column=chosen)
            except ValueError as error2:
                st.sidebar.error(str(error2))
                st.stop()
        except ValueError as error:
            st.sidebar.error(str(error))
            st.stop()
        st.session_state.file_id = file_id
        st.session_state.table = categorize(st.session_state.parsed)
        st.session_state.editor_version += 1


# =================================================================================================
# Page 1: Dashboard
# =================================================================================================
def show_dashboard(table):
    st.title("📊 Dashboard")

    # --- Summary cards: spent / income / savings / savings %, each with the change vs last month ---
    s = an.summary(table)
    mom = an.month_over_month(table)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(th.metric_card("💸", "Total spent", an.inr(s["spent"]), mom["spent"], invert=True), unsafe_allow_html=True)
    c2.markdown(th.metric_card("🔁", "Transfers out", an.inr(s["transfers"]), mom["transfers"], invert=True), unsafe_allow_html=True)
    c3.markdown(th.metric_card("💰", "Income", an.inr(s["income"]), mom["income"]), unsafe_allow_html=True)
    c4.markdown(th.metric_card("🏦", "Savings", an.inr(s["savings"]), mom["savings"]), unsafe_allow_html=True)
    c5.markdown(th.metric_card("📈", "Savings %", "-" if s["savings_pct"] is None else f"{s['savings_pct']:.1f}%",
                               mom["savings_pct"], unit=" pts"), unsafe_allow_html=True)
    st.caption("Spending = debits except Transfers (shown separately). Credits (salary, refunds) count as income. "
               "Savings = income - spent.")

    if s["spent"] == 0:
        st.info("No debit transactions in this statement, so there is nothing to chart.")
        return

    # --- Category pie + monthly trend, side by side ---
    st.markdown(th.section("Where your money goes", "Spending by category and month"), unsafe_allow_html=True)
    left, right = st.columns(2)

    cats = an.category_totals(table)
    cats["category"] = th.label_column(cats["category"])      # '🍔 Food' - same emoji and colour everywhere
    cats["label"] = cats["amount"].map(an.inr)               # Indian-format text for the hover box
    pie = px.pie(cats, names="category", values="amount", hole=0.6, custom_data="label",
                 title="Spending by category", color="category", color_discrete_map=th.COLOR_MAP)
    pie.update_traces(hovertemplate="%{label}: %{customdata[0]} (%{percent})<extra></extra>",
                      textinfo="percent", textposition="inside", marker_line_width=0)
    pie.update_layout(uniformtext_minsize=11, uniformtext_mode="hide")   # hide labels on tiny slices
    left.plotly_chart(th.style_fig(pie), width="stretch")

    months = an.monthly_spending(table)
    months["label"] = months["amount"].map(an.inr)
    months["month_name"] = months["month"].dt.strftime("%b %Y")
    trend = px.bar(months, x="month_name", y="amount", text="label", title="Monthly spending",
                   labels={"month_name": "Month", "amount": "Spent (₹)"},
                   color_discrete_sequence=[th.ACCENT])
    trend.update_traces(hovertemplate="%{x}: %{text}<extra></extra>", textposition="outside",
                        marker_cornerradius=8)
    trend.update_yaxes(visible=False)
    right.plotly_chart(th.style_fig(trend), width="stretch")

    # --- Top 5 merchants + biggest 5 transactions, side by side ---
    st.markdown(th.section("Top spends", "Your favourite merchants and biggest payments"), unsafe_allow_html=True)
    left, right = st.columns(2)

    top = an.top_merchants(table, n=5)
    top["label"] = top["amount"].map(an.inr)
    bars = px.bar(top.iloc[::-1], x="amount", y="merchant", orientation="h", text="label",   # reversed: biggest on top
                  title="Top 5 merchants", labels={"amount": "Spent (₹)", "merchant": ""},
                  color_discrete_sequence=[th.ACCENT])
    bars.update_traces(hovertemplate="%{y}: %{text}<extra></extra>", marker_cornerradius=8)
    bars.update_xaxes(visible=False)
    left.plotly_chart(th.style_fig(bars), width="stretch")

    right.markdown("**Biggest 5 transactions**")
    big = an.biggest_transactions(table, n=5)
    right.dataframe(
        big.assign(date=big["date"].dt.date, amount=big["amount"].map(lambda a: an.inr(a, 2)),
                   category=th.label_column(big["category"]))
        [["date", "merchant", "amount", "category"]],
        hide_index=True, width="stretch",
    )


# =================================================================================================
# Page 2: Transactions (search, filters, and the category correction editor)
# =================================================================================================
def show_transactions(table):
    st.title("🧾 Transactions")

    # --- Filters ---
    search = st.text_input("Search merchant or description", placeholder="e.g. swiggy")
    f1, f2, f3 = st.columns([2, 2, 2])
    first, last = table["date"].min().date(), table["date"].max().date()
    date_range = f1.date_input("Date range", value=(first, last), min_value=first, max_value=last)
    categories = f2.multiselect("Category", cz.CATEGORIES + CREDIT_LABELS, placeholder="All categories")
    top_amount = float(table["amount"].max())
    a1, a2 = f3.columns(2)
    min_amount = a1.number_input("Min ₹", min_value=0.0, max_value=top_amount, value=0.0)
    max_amount = a2.number_input("Max ₹", min_value=0.0, max_value=top_amount, value=top_amount)

    # date_input returns 1 date while the user is still picking the 2nd one: wait for both
    start, end = date_range if len(date_range) == 2 else (None, None)
    shown = an.filter_transactions(table, start, end, categories, min_amount, max_amount, search)

    # Totals of what is currently shown (debits and credits kept apart)
    spent, income = an.spending(shown)["amount"].sum(), an.credits(shown)["amount"].sum()
    transfers = an.transfers_out(shown)["amount"].sum()
    st.markdown(th.chips([("Showing", f"{len(shown)} of {len(table)}"), ("Spent", an.inr(spent)),
                          ("Transfers out", an.inr(transfers)), ("Income", an.inr(income))]),
                unsafe_allow_html=True)

    # --- Editable table: change a category with the dropdown ---
    st.caption("Wrong category? Pick the right one in the dropdown, then click 'Save corrections'. "
               "Credits are always Income / Refund and cannot be corrected.")
    view = shown[["date", "merchant", "type", "amount", "category"]].copy()
    view["date"] = view["date"].dt.date                               # date only, no 00:00:00
    view["category"] = th.label_column(view["category"])              # show '🍔 Food' (mapped back below)
    # The editor key includes the filters: if they change, the editor starts fresh instead of
    # mixing up edits made on a different set of rows.
    filter_key = hash((str(date_range), tuple(categories), min_amount, max_amount, search))
    edited = st.data_editor(
        view,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "merchant": "Merchant",
            "type": "Type",
            "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),   # a real number, so it sorts correctly
            "category": st.column_config.SelectboxColumn("Category", options=[th.label(c) for c in cz.CATEGORIES + CREDIT_LABELS], required=True),
        },
        disabled=["date", "merchant", "type", "amount"],              # only the category can be changed
        hide_index=True,
        width="stretch",
        key=f"editor_{st.session_state.editor_version}_{filter_key}",
    )

    # Rows (by their original row number, kept in the index) where the dropdown differs from the prediction
    # Only debits can be corrected, and only to one of the 10 spending categories
    edited["category"] = edited["category"].map(th.unlabel)           # '🍔 Food' -> 'Food' again
    differs = edited["category"] != shown["category"]
    changed = edited.index[differs & (shown["type"] == "debit") & edited["category"].isin(cz.CATEGORIES)]
    if (differs.sum() - len(changed)) > 0:
        st.warning("Some edits are ignored: credits stay Income / Refund, and debits need a spending category.")
    if st.button(f"Save corrections ({len(changed)} changed)", disabled=len(changed) == 0):
        # Save the masked description (not the merchant) because that is what the model trains on
        rows = [(st.session_state.parsed.at[i, "description"], edited.at[i, "category"])
                for i in changed]
        save_correction_rows(rows)
        # Re-categorize: other rows with the same merchant pick up the correction too
        st.session_state.table = categorize(st.session_state.parsed)
        st.session_state.editor_version += 1
        st.toast(f"Saved {len(rows)} correction(s) " + ("for this session." if PUBLIC_DEMO else f"to {cz.CORRECTIONS_PATH}."))  # survives the rerun
        st.rerun()


# =================================================================================================
# Page 3: Model (retrain with the saved corrections)
# =================================================================================================
def show_model():
    st.title("🧠 Model")
    n_corrections = len(current_corrections())
    st.write(f"Saved corrections: **{n_corrections}**  |  Current model: **{st.session_state.bundle['name']}**")

    if PUBLIC_DEMO:
        st.info("Retraining is disabled in the public demo")
    if st.button("Retrain model", disabled=PUBLIC_DEMO or n_corrections == 0,
                 help="Retraining is disabled in the public demo" if PUBLIC_DEMO
                 else "Save at least one correction first." if n_corrections == 0 else None):
        with st.spinner("Retraining... this takes a few seconds"):
            before = st.session_state.scores                      # accuracy of the old model
            new_bundle = cz.train_and_save(corrections_path=cz.CORRECTIONS_PATH)
            after = cz.messy_scores(new_bundle)                   # same test set, new model
        st.session_state.bundle = new_bundle
        st.session_state.scores = after
        st.session_state.retrain_result = {"before": before, "after": after, "bundle": new_bundle}
        if "parsed" in st.session_state:                          # show the new predictions
            st.session_state.table = categorize(st.session_state.parsed)
            st.session_state.editor_version += 1
        st.rerun()

    # Accuracy on the fixed messy test set: hybrid (rules + ML) next to ML-only.
    # Both use the same never-changing test rows, so before/after can be compared fairly.
    scores = st.session_state.scores
    result = st.session_state.get("retrain_result")

    col1, col2, col3 = st.columns(3)

    def card(col, icon, title, value, key=None):
        """Model score card; the change after a retrain goes in the small line."""
        d = None if key is None or not result else (result["after"][key] - result["before"][key]) * 100
        col.markdown(th.metric_card(icon, title, value, d, unit=" pts", note="since retrain"), unsafe_allow_html=True)

    card(col1, "🎯", "Hybrid (rules + ML), messy set", f"{scores['hybrid']:.1%}", "hybrid")
    card(col2, "🤖", "ML only, messy set", f"{scores['ml_only']:.1%}", "ml_only")
    if result:
        card(col3, "🧪", "Held-out split (incl. corrections)", f"{result['bundle']['clean_accuracy']:.1%}")
        st.caption(f"Retrained on the original data + {result['bundle']['n_corrections']} correction(s).")


# =================================================================================================
# Page 4: Insights (unusual spends, subscriptions, budgets). Logic lives in insights.py
# =================================================================================================
STATUS_ICON = {"ok": "🟢", "warning": "🟡", "over": "🔴"}


def show_insights(table):
    st.title("✨ Insights")

    # --- 1. Unusual spends ---
    st.markdown(th.section("Unusual spends"), unsafe_allow_html=True)
    st.caption("An Isolation Forest looks at amount, category and day of week and flags debits that "
               "don't look like the rest.")
    odd = ins.unusual_spends(table)
    if odd.empty:
        st.info("Nothing unusual found (or too few transactions to tell).")
    else:
        st.dataframe(
            odd.assign(date=odd["date"].dt.date, amount=odd["amount"].map(an.inr),
                       category=th.label_column(odd["category"]))
            [["date", "merchant", "category", "amount", "reason"]],
            hide_index=True, width="stretch")

    # --- 2. Subscriptions and recurring payments ---
    st.markdown(th.section("Subscriptions and recurring payments"), unsafe_allow_html=True)
    st.caption("Same merchant, similar amount, about 30 days apart (at least 3 payments).")
    subs = ins.find_subscriptions(table)
    if subs.empty:
        st.info("No recurring payments found.")
    else:
        st.markdown(th.chips([("You spend", f"{an.inr(ins.subscription_total(subs))}/month on subscriptions")]),
                    unsafe_allow_html=True)
        st.dataframe(
            subs.assign(monthly_amount=subs["monthly_amount"].map(an.inr),
                        last_paid=subs["last_paid"].dt.date, next_due=subs["next_due"].dt.date,
                        category=th.label_column(subs["category"]))
            [["merchant", "category", "monthly_amount", "payments", "last_paid", "next_due"]],
            hide_index=True, width="stretch")

    # --- 3. Budgets ---
    st.markdown(th.section("Budgets", "Monthly limits per category"), unsafe_allow_html=True)
    budgets = current_budgets()
    spendable = [c for c in cz.CATEGORIES if c != "Transfers"]      # transfers are not spending
    with st.expander("Set monthly budgets (₹, 0 = no budget)"):
        cols = st.columns(3)
        new = {c: cols[i % 3].number_input(c, min_value=0, step=500, value=int(budgets.get(c, 0)),
                                           key=f"budget_{c}")
               for i, c in enumerate(spendable)}
        if st.button("Save budgets"):
            save_current_budgets(new)
            st.toast("Saved budgets for this session." if PUBLIC_DEMO else f"Saved budgets to {ins.BUDGETS_PATH}.")
            st.rerun()

    months = sorted(table["date"].dt.to_period("M").unique(), reverse=True)     # newest first
    month = st.selectbox("Month (used for budgets and the AI report)", months,
                         format_func=lambda p: p.strftime("%B %Y"))
    if not budgets:
        st.info("Set a budget above to track it month by month.")
    for row in ins.budget_status(table, month.to_timestamp(), budgets).itertuples():
        st.markdown(th.budget_row(row.category, an.inr(row.spent), an.inr(row.budget), row.pct, row.status),
                    unsafe_allow_html=True)
        if row.status == "over":
            st.error(f"{row.category} is over budget by {an.inr(row.spent - row.budget)}.")
        elif row.status == "warning":
            st.warning(f"{row.category} has used {row.pct:.0f}% of its budget.")

    show_forecast(table)
    show_ai_report(table, month, budgets)


def show_forecast(table):
    """Next month's spending per category, predicted with Linear Regression on past monthly totals."""
    st.markdown(th.section("Spending forecast", "Next month, predicted from past trends"), unsafe_allow_html=True)
    history, forecast, dropped_partial = ins.forecast_next_month(table)
    if forecast.empty:
        st.info("Need at least 3 complete months of spending to forecast.")
        return

    # Chart: one stacked bar per past month + one for the predicted month
    actual = history.rename_axis("month").reset_index().melt(id_vars="month", var_name="category", value_name="amount")
    actual["label"] = actual["month"].dt.strftime("%b %Y")
    next_month = history.index[-1] + pd.DateOffset(months=1)
    predicted = forecast.rename(columns={"predicted": "amount"}).assign(label=next_month.strftime("%b %Y") + " (predicted)")
    chart_data = pd.concat([actual[["label", "category", "amount"]], predicted[["label", "category", "amount"]]])
    chart_data["text"] = chart_data["amount"].map(an.inr)
    chart_data["category"] = th.label_column(chart_data["category"])
    fig = px.bar(chart_data, x="label", y="amount", color="category", custom_data="text",
                 title="Actual vs predicted spending", labels={"label": "Month", "amount": "Spent (₹)"},
                 color_discrete_map=th.COLOR_MAP)
    fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{customdata[0]}<extra></extra>")
    st.plotly_chart(th.style_fig(fig, height=420), width="stretch")

    # Table: predicted amount per category + a Total row
    total = forecast["predicted"].sum()
    shown = pd.concat([forecast.assign(category=th.label_column(forecast["category"])),
                       pd.DataFrame({"category": ["Total"], "predicted": [total]})])
    st.dataframe(shown.assign(predicted=shown["predicted"].map(an.inr)).rename(
        columns={"category": "Category", "predicted": f"Predicted for {next_month.strftime('%B %Y')}"}),
        hide_index=True, width="stretch")
    note = "This is only an estimate based on past trends, not a guarantee."
    if dropped_partial:
        note += " The last month is incomplete, so it was left out of the forecast."
    st.caption(note)


def show_ai_report(table, month, budgets):
    """Ask Gemini for a short summary and saving tips, sending aggregated stats only."""
    st.markdown(th.section("AI monthly report", "A short summary and saving tips from Gemini"), unsafe_allow_html=True)
    st.caption("Only totals, percentages and counts are sent to Gemini: never transaction descriptions or merchant names.")
    key = f"report_{month}"                                   # one saved report per month
    if st.button("Generate AI Report"):
        stats = rp.build_report_stats(table, month.to_timestamp(), budgets)
        with st.spinner("Asking Gemini..."):
            try:
                st.session_state[key] = rp.generate_report(stats)
            except rp.ReportError as error:
                st.session_state.pop(key, None)
                st.error(str(error))
    if key in st.session_state:
        result = st.session_state[key]
        st.write(result["summary"])
        for tip in result["tips"]:
            st.markdown(f"- 💡 {tip}")


def show_empty_state():
    """Friendly screen shown until a statement is uploaded."""
    st.markdown(
        '<div class="hero"><div class="big">💸</div><h2>Upload a statement to get started</h2>'
        '<div class="sub">Drop your bank or UPI statement in the sidebar, or click <b>Try with sample data</b>.</div>'
        '<div class="steps">'
        '<div class="step"><span class="n">1</span><br>Download your statement as <b>CSV or Excel</b> '
        'from your bank or UPI app.</div>'
        '<div class="step"><span class="n">2</span><br>Drop the file into the <b>upload box</b> in the sidebar.</div>'
        '<div class="step"><span class="n">3</span><br>Explore your <b>dashboard, insights</b> and AI report.</div>'
        '</div><div class="sub" style="margin-top:22px">🔒 ' + PRIVACY_NOTE + ' Account numbers and UPI IDs are masked, '
        'and only totals are ever sent to the AI.</div></div>', unsafe_allow_html=True)


# --- Route to the chosen page ----------------------------------------------------------------
if page == "Model":
    show_model()
elif "table" not in st.session_state:
    show_empty_state()
elif page == "Dashboard":
    show_dashboard(st.session_state.table)
elif page == "Insights":
    show_insights(st.session_state.table)
else:
    show_transactions(st.session_state.table)
