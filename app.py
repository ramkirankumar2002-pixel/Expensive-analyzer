# app.py - Streamlit dashboard: upload a statement, see the analysis, fix wrong categories, retrain.
# Run with:  streamlit run app.py
#
# Pages (sidebar):  Dashboard  |  Transactions  |  Insights  |  Goals  |  Model
# The number crunching lives in analytics.py; this file only draws things.

import datetime as dt
import os

import pandas as pd
import plotly.express as px
import streamlit as st

import analytics as an
import categorizer as cz
import export as ex
import goals as gl
import insights as ins
import merger as mg
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


# In the public demo every visitor gets their OWN corrections, budgets and goals, kept in st.session_state
# (gone when the tab closes). Locally they are saved in corrections.csv / budgets.json / goals.json.
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


def current_goals():
    if PUBLIC_DEMO:
        return st.session_state.get("goals", [])
    return gl.load_goals()


def save_current_goals(new):
    if PUBLIC_DEMO:
        st.session_state.goals = new
    else:
        gl.save_goals(new)


def categorize(parsed):
    """Rules + ML + the user's saved corrections for every row of the parsed statement.
    Self transfers found by merger.py then get their own label, whatever the rules said."""
    table = cz.categorize_dataframe(parsed, bundle=st.session_state.bundle, corrections=current_corrections())
    if "self_transfer" in table.columns:                  # only there when several files were merged
        table.loc[table["self_transfer"], "category"] = an.SELF_TRANSFER
    return table


# --- Sidebar: navigation + upload ------------------------------------------------------------
st.sidebar.markdown('<div class="brand"><span class="logo">💸</span><div><div class="name">Expense Analyzer</div>'
                    '<div class="tag">Know where your money goes</div></div></div>', unsafe_allow_html=True)
PAGE_ICONS = {"Dashboard": "📊", "Transactions": "🧾", "Insights": "✨", "Goals": "🎯", "Model": "🧠"}
page = st.sidebar.radio("Go to", list(PAGE_ICONS), format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
                        label_visibility="collapsed")
st.sidebar.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)   # pushes the upload box down

uploads = st.sidebar.file_uploader("📤 Upload your bank / UPI statement(s)", type=["csv", "xlsx", "xls"],
                                   accept_multiple_files=True,
                                   help="You can add statements from several banks or UPI apps at once.")
st.sidebar.caption("🔒 " + PRIVACY_NOTE)
if st.sidebar.button("🧪 Try with sample data", help="Loads a fake statement so you can explore the app"):
    st.session_state.source = "sample"                    # the newest choice (button or upload) wins
upload_ids = tuple(f.file_id for f in uploads)            # changes when a file is added or removed
if uploads and upload_ids != st.session_state.get("seen_upload"):
    st.session_state.seen_upload = upload_ids
    st.session_state.source = "upload"


def parse_once(file, key, error_prefix=""):
    """Parse one file and remember the result, so it is not parsed again on every click.
    A problem is shown in the sidebar and stops the page. error_prefix says which file it was."""
    cache = st.session_state.setdefault("parsed_files", {})
    if key not in cache:
        try:
            cache[key] = parse_statement(file)
        except MissingColumnError as error:
            # No column we recognise: let the user say which one holds the description.
            st.sidebar.warning(f"{error_prefix}{error}. Pick the column that describes each transaction.")
            chosen = st.sidebar.selectbox("Description column", error.columns, index=None,
                                          placeholder="Choose a column...", key=f"description_{key}")
            if chosen is None:
                st.stop()
            try:
                cache[key] = parse_statement(file, description_column=chosen)
            except ValueError as error2:
                st.sidebar.error(f"{error_prefix}{error2}")
                st.stop()
        except ValueError as error:
            st.sidebar.error(f"{error_prefix}{error}")
            st.stop()
    return cache[key]


def ask_labels(files):
    """One editable account label per uploaded file, pre-filled from the file name."""
    defaults = mg.default_labels([(f.name, f.getvalue()) for f in files])
    st.sidebar.markdown("**Account labels**")
    st.sidebar.caption("Use the same label for two files from one account, so overlapping rows are removed.")
    return [st.sidebar.text_input(f.name, value=default, key=f"label_{f.file_id}").strip() or default
            for f, default in zip(files, defaults)]


def show_upload_summary(summary):
    """Rows per file, duplicates removed and self transfers found (only after a multi-file upload)."""
    with st.sidebar.expander("📋 Upload summary", expanded=True):
        # just two columns: the sidebar is narrow, and the labels are already shown in the boxes above
        st.dataframe(pd.DataFrame(summary["files"], columns=["File", "Rows"]), hide_index=True, width="stretch")
        st.markdown(f"Duplicates removed: **{summary['duplicates_removed']}**")
        st.markdown(f"Self transfers found: **{summary['self_transfers']}** (each is a debit + credit pair)")


# What are we showing? The sample file, or the uploaded file(s).
# data_id describes that choice; the table is only rebuilt when data_id changes.
using_sample = st.session_state.get("source") == "sample"
files = [] if using_sample else uploads
labels = ask_labels(files) if len(files) > 1 else []
data_id = None
if using_sample:
    data_id = ("sample",)
elif files:
    data_id = (upload_ids, tuple(labels))

if data_id is not None and st.session_state.get("data_id") != data_id:
    summary = None
    if using_sample:
        parsed = mg.tag_source(parse_once(SAMPLE_PATH, "sample"), "sample data")
    elif len(files) == 1:                                 # one file: nothing to merge, no row is removed
        label = mg.default_labels([(files[0].name, files[0].getvalue())])[0]
        parsed = mg.tag_source(parse_once(files[0], files[0].file_id), label)
    else:
        frames = [parse_once(f, f.file_id, error_prefix=f"{f.name}: ") for f in files]
        parsed, summary = mg.merge_statements(list(zip(labels, frames)))
        summary["files"] = [(f.name, len(df)) for f, df in zip(files, frames)]     # rows read from each file
    st.session_state.parsed = parsed
    st.session_state.upload_summary = summary
    st.session_state.data_id = data_id
    # AI reports were written about the previous statement: forget them, so they can't end up in this one
    for key in [k for k in st.session_state if str(k).startswith("report_")]:
        del st.session_state[key]
    st.session_state.table = categorize(parsed)
    st.session_state.editor_version += 1

if st.session_state.get("upload_summary"):
    show_upload_summary(st.session_state.upload_summary)


def source_filter(table, key):
    """'Source' dropdown, "All" by default. Returns the chosen account label, or None for all.
    It is only shown when the table holds more than one account."""
    sources = sorted(table["source"].unique())
    if len(sources) < 2:
        return None
    # "" stands for All (a label can never be empty). The key includes the accounts, so new data resets the choice.
    choice = st.selectbox("Source", [""] + sources, format_func=lambda s: s or "All",
                          key=f"{key}_{'|'.join(sources)}")
    return choice or None


# =================================================================================================
# Page 1: Dashboard
# =================================================================================================
def show_dashboard(table):
    st.title("📊 Dashboard")
    table = an.filter_transactions(table, source=source_filter(table, "dashboard_source"))   # everything below uses this

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
               "Savings = income - spent."
               + (" Self transfers between your own accounts count as neither."
                  if (table["category"] == an.SELF_TRANSFER).any() else ""))

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
    source = source_filter(table, "transactions_source")
    several_sources = table["source"].nunique() > 1
    # "Self transfer" is only offered when the data has some
    extra_labels = [an.SELF_TRANSFER] if (table["category"] == an.SELF_TRANSFER).any() else []
    search = st.text_input("Search merchant or description", placeholder="e.g. swiggy")
    f1, f2, f3 = st.columns([2, 2, 2])
    first, last = table["date"].min().date(), table["date"].max().date()
    date_range = f1.date_input("Date range", value=(first, last), min_value=first, max_value=last)
    categories = f2.multiselect("Category", cz.CATEGORIES + CREDIT_LABELS + extra_labels, placeholder="All categories")
    top_amount = float(table["amount"].max())
    a1, a2 = f3.columns(2)
    min_amount = a1.number_input("Min ₹", min_value=0.0, max_value=top_amount, value=0.0)
    max_amount = a2.number_input("Max ₹", min_value=0.0, max_value=top_amount, value=top_amount)

    # date_input returns 1 date while the user is still picking the 2nd one: wait for both
    start, end = date_range if len(date_range) == 2 else (None, None)
    shown = an.filter_transactions(table, start, end, categories, min_amount, max_amount, search, source)

    # Totals of what is currently shown (debits and credits kept apart)
    spent, income = an.spending(shown)["amount"].sum(), an.credits(shown)["amount"].sum()
    transfers = an.transfers_out(shown)["amount"].sum()
    st.markdown(th.chips([("Showing", f"{len(shown)} of {len(table)}"), ("Spent", an.inr(spent)),
                          ("Transfers out", an.inr(transfers)), ("Income", an.inr(income))]),
                unsafe_allow_html=True)

    # --- Editable table: change a category with the dropdown ---
    st.caption("Wrong category? Pick the right one in the dropdown, then click 'Save corrections'. "
               "Credits are always Income / Refund and cannot be corrected."
               + (" Self transfers cannot be changed either." if extra_labels else ""))
    columns = ["date", "merchant", "type", "amount", "category"] + (["source"] if several_sources else [])
    view = shown[columns].copy()
    view["date"] = view["date"].dt.date                               # date only, no 00:00:00
    view["category"] = th.label_column(view["category"])              # show '🍔 Food' (mapped back below)
    # The editor key includes the filters: if they change, the editor starts fresh instead of
    # mixing up edits made on a different set of rows.
    filter_key = hash((str(date_range), tuple(categories), min_amount, max_amount, search, source))
    edited = st.data_editor(
        view,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "merchant": "Merchant",
            "type": "Type",
            "amount": st.column_config.NumberColumn("Amount", format="₹%.2f"),   # a real number, so it sorts correctly
            "category": st.column_config.SelectboxColumn("Category", options=[th.label(c) for c in cz.CATEGORIES + CREDIT_LABELS + extra_labels], required=True),
            "source": "Source",
        },
        disabled=["date", "merchant", "type", "amount", "source"],    # only the category can be changed
        hide_index=True,
        width="stretch",
        key=f"editor_{st.session_state.editor_version}_{filter_key}",
    )

    # Rows (by their original row number, kept in the index) where the dropdown differs from the prediction
    # Only debits can be corrected, and only to one of the 10 spending categories
    edited["category"] = edited["category"].map(th.unlabel)           # '🍔 Food' -> 'Food' again
    differs = edited["category"] != shown["category"]
    editable = (shown["type"] == "debit") & (shown["category"] != an.SELF_TRANSFER)
    changed = edited.index[differs & editable & edited["category"].isin(cz.CATEGORIES)]
    if (differs.sum() - len(changed)) > 0:
        st.warning("Some edits are ignored: credits stay Income / Refund, self transfers stay as they are, "
                   "and debits need a spending category.")
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
    month = st.selectbox("Month (used for budgets, the AI report and the download)", months,
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
    language = show_ai_report(table, month, budgets)
    show_download(table, month, budgets, language)    # last, so a just-generated AI report is included


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
    goals = current_goals()
    include_goals = bool(goals) and st.checkbox("Include savings goal status (numbers only, no goal names)", value=True)
    button_column, language_column = st.columns([1, 2], vertical_alignment="bottom")
    language = language_column.selectbox("Report language", list(rp.LANGUAGES), format_func=rp.LANGUAGES.get,
                                         key="report_language")    # the first one (English) is the default
    # One saved report per month AND language: switching back to one you already made needs no new Gemini call
    key = rp.report_cache_key(month, language)
    if button_column.button("Generate AI Report"):
        stats = rp.build_report_stats(table, month.to_timestamp(), budgets, goals if include_goals else None)
        with st.spinner("Asking Gemini..."):
            try:
                st.session_state[key] = rp.generate_report(stats, language)
            except rp.ReportError as error:
                st.session_state.pop(key, None)
                st.error(str(error))
    if key in st.session_state:
        result = st.session_state[key]
        st.write(result["summary"])
        for tip in result["tips"]:
            st.markdown(f"- 💡 {tip}")
    return language                                           # the download uses the same language


def show_download(table, month, budgets, language):
    """'Download monthly report': one self-contained HTML file for the chosen month (built by export.py)."""
    period = month.strftime("%B %Y")
    st.markdown(th.section("Download", "The whole month as one HTML file you can keep or print"),
                unsafe_allow_html=True)
    # Read these now: the function below runs later, outside the page script, where session_state is not available
    goals = current_goals()
    ai_report = st.session_state.get(rp.report_cache_key(month, language))     # only the chosen language
    st.caption(f"Report for {period} (change the month above). "
               + (f"Includes the AI summary and tips in {rp.LANGUAGES[language]}." if ai_report
                  else f"Generate the AI report in {rp.LANGUAGES[language]} first if you want it included.")
               + " It contains your merchant and goal names, so share it with care.")
    st.download_button(
        "📥 Download monthly report",
        # a function = the report is only built when the button is clicked, not on every rerun.
        # It is built in memory and sent to the browser: no file is written on the server (works on Streamlit Cloud).
        data=lambda: ex.build_report_html(table, month.to_timestamp(), budgets, goals, ai_report),
        file_name=ex.report_filename(month.to_timestamp()),
        mime="text/html",
        on_click="ignore",                       # don't rerun the page after the download
    )
    st.caption("To print it or save it as PDF: open the file in your browser and press Ctrl+P.")


# =================================================================================================
# Page 5: Goals (savings goals). Logic lives in goals.py
# =================================================================================================
def show_goals(table):
    st.title("🎯 Goals")
    goals = current_goals()
    savings = gl.recent_savings(table)             # whole statement: all accounts together

    months = savings["months"]
    period = months[0].strftime("%b %Y") + (f" to {months[-1].strftime('%b %Y')}" if len(months) > 1 else "")
    st.markdown(th.chips([("Average monthly savings", an.inr(savings["savings"])), ("Based on", period)]),
                unsafe_allow_html=True)
    st.caption("Savings = income - spending, as on the Dashboard (money sent in Transfers is not subtracted). "
               f"Averaged over {len(months)} month(s) of your statement. If the statement ends mid-month, "
               "that month is left out.")

    # --- Add a goal ---
    with st.expander("➕ Add a goal", expanded=not goals):
        with st.form("add_goal"):
            name = st.text_input("Goal name", placeholder="e.g. Laptop")
            c1, c2, c3 = st.columns(3)
            target = c1.number_input("Target amount (₹)", min_value=0, step=1000)
            today = dt.date.today()
            target_date = c2.date_input("Target date", value=today + dt.timedelta(days=365),
                                        min_value=today + dt.timedelta(days=1),
                                        max_value=dt.date(today.year + 30, 12, 31), format="DD/MM/YYYY")
            saved = c3.number_input("Already saved (₹)", min_value=0, step=1000)
            if st.form_submit_button("Add goal"):
                try:
                    save_current_goals(goals + [gl.make_goal(name, target, target_date, saved)])
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.toast("Saved the goal for this session." if PUBLIC_DEMO else f"Saved the goal to {gl.GOALS_PATH}.")
                    st.rerun()
    if not goals:
        st.info("No goals yet. Add one above to see if you are on track.")
        return

    results = [gl.evaluate(goal, savings["savings"]) for goal in goals]
    if len(goals) > 1:
        st.caption("Each goal is checked on its own, as if all your savings went to it.")
        total_needed = sum(r["needed"] for r in results if not r["date_passed"])
        if total_needed > max(savings["savings"], 0):
            st.warning(f"Together your goals need {an.inr(total_needed)}/month, more than your average "
                       f"savings of {an.inr(savings['savings'])}.")

    for goal, result in zip(goals, results):
        show_goal(goal, result, goals, savings["spend_by_category"])


def show_goal(goal, r, goals, spend_by_category):
    """One goal: status card with progress bar, advice, and update / delete controls."""
    due = dt.date.fromisoformat(goal["target_date"])
    if r["reached"]:
        reach_text = "Reached 🎉"
    elif r["reach_date"] is None:
        reach_text = "Never at your current rate"
    else:
        reach_text = r["reach_date"].strftime("%b %Y")
    st.markdown(th.goal_card(goal["name"], r["status"], r["progress"],
                             f"{an.inr(goal['saved'])} of {an.inr(goal['target'])}",
                             [("Target date", due.strftime("%d %b %Y")),
                              ("Months left", str(r["months_left"])),
                              ("Needed per month", an.inr(r["needed"])),
                              ("You save per month", an.inr(r["savings"])),
                              ("You would reach it", reach_text)]),
                unsafe_allow_html=True)

    if r["date_passed"]:
        st.error(f"The target date has passed and {an.inr(r['remaining'])} is still missing. "
                 "Delete this goal and add it again with a new date.")
    elif r["status"] == gl.OFF_TRACK:
        cuts, uncovered = gl.suggest_cuts(r["gap"], spend_by_category)
        st.markdown(f"**To close the gap of {an.inr(r['gap'])}/month:**")
        for cut in cuts:
            st.markdown(f"- ✂️ {gl.cut_text(cut)}")
        if uncovered >= 1:
            st.warning(f"Cuts of up to {gl.MAX_CUT:.0%} in {', '.join(gl.CUTTABLE)} still leave "
                       f"{an.inr(uncovered)}/month short. Try a later date or a smaller target.")

    # the id keeps two goals with the same name apart
    with st.expander(f"Update or delete: {goal['name']}"):
        new_saved = st.number_input("Already saved (₹)", min_value=0, step=1000, value=int(goal["saved"]),
                                    key=f"saved_{goal['id']}")
        c1, c2 = st.columns(2)
        if c1.button("Update saved amount", key=f"update_{goal['id']}"):
            save_current_goals([{**g, "saved": float(new_saved)} if g["id"] == goal["id"] else g for g in goals])
            st.rerun()
        if c2.button("Delete goal", key=f"delete_{goal['id']}"):
            save_current_goals([g for g in goals if g["id"] != goal["id"]])
            st.rerun()


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
elif page == "Goals":
    show_goals(st.session_state.table)
else:
    show_transactions(st.session_state.table)
