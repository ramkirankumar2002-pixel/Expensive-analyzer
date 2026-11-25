# app.py - Streamlit dashboard: upload a statement, see the analysis, fix wrong categories, retrain.
# Run with:  streamlit run app.py
#
# Pages (sidebar):  Dashboard  |  Transactions  |  Model
# The number crunching lives in analytics.py; this file only draws things.

import plotly.express as px
import streamlit as st

import analytics as an
import categorizer as cz
from parser import MissingColumnError, parse_statement

st.set_page_config(page_title="Expense Analyzer", page_icon="₹", layout="wide")

# --- Remember things between reruns (Streamlit re-runs this whole file on every click) -------
if "bundle" not in st.session_state:
    st.session_state.bundle = cz.load_model()   # the trained model (loaded once)
if "scores" not in st.session_state:
    st.session_state.scores = cz.messy_scores(st.session_state.bundle)  # accuracy of the current model
st.session_state.setdefault("editor_version", 0)  # bump it to reset the table's pending edits


def categorize(parsed):
    """Rules + ML + the user's saved corrections for every row of the parsed statement."""
    return cz.categorize_dataframe(parsed, bundle=st.session_state.bundle)


# --- Sidebar: navigation + upload ------------------------------------------------------------
st.sidebar.title("Expense Analyzer")
page = st.sidebar.radio("Go to", ["Dashboard", "Transactions", "Model"], label_visibility="collapsed")
st.sidebar.divider()

uploaded = st.sidebar.file_uploader("Upload your bank / UPI statement", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("file_id") != file_id:       # a new file: parse it once
        try:
            st.session_state.parsed = parse_statement(uploaded)
        except MissingColumnError as error:
            # No column we recognise: let the user say which one holds the description.
            st.sidebar.warning(f"{error}. Pick the column that describes each transaction.")
            chosen = st.sidebar.selectbox("Description column", error.columns, index=None,
                                          placeholder="Choose a column...")
            if chosen is None:
                st.stop()
            try:
                st.session_state.parsed = parse_statement(uploaded, description_column=chosen)
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
    st.title("Dashboard")

    # --- Summary cards: spent / income / savings / savings % ---
    s = an.summary(table)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spent", an.inr(s["spent"]))
    c2.metric("Income", an.inr(s["income"]))
    c3.metric("Savings", an.inr(s["savings"]))
    c4.metric("Savings %", "-" if s["savings_pct"] is None else f"{s['savings_pct']:.1f}%")
    st.caption("Spending = debits only. Credits (salary, refunds) count as income.")

    if s["spent"] == 0:
        st.info("No debit transactions in this statement, so there is nothing to chart.")
        return

    # --- Category pie + monthly trend, side by side ---
    left, right = st.columns(2)

    cats = an.category_totals(table)
    cats["label"] = cats["amount"].map(an.inr)               # Indian-format text for the hover box
    pie = px.pie(cats, names="category", values="amount", hole=0.4, custom_data="label",
                 title="Spending by category")
    pie.update_traces(hovertemplate="%{label}: %{customdata[0]} (%{percent})<extra></extra>")
    left.plotly_chart(pie, width="stretch")

    months = an.monthly_spending(table)
    months["label"] = months["amount"].map(an.inr)
    months["month_name"] = months["month"].dt.strftime("%b %Y")
    trend = px.bar(months, x="month_name", y="amount", text="label", title="Monthly spending",
                   labels={"month_name": "Month", "amount": "Spent (₹)"})
    trend.update_traces(hovertemplate="%{x}: %{text}<extra></extra>", textposition="outside")
    right.plotly_chart(trend, width="stretch")

    # --- Top 5 merchants + biggest 5 transactions, side by side ---
    left, right = st.columns(2)

    top = an.top_merchants(table, n=5)
    top["label"] = top["amount"].map(an.inr)
    bars = px.bar(top.iloc[::-1], x="amount", y="merchant", orientation="h", text="label",   # reversed: biggest on top
                  title="Top 5 merchants", labels={"amount": "Spent (₹)", "merchant": ""})
    bars.update_traces(hovertemplate="%{y}: %{text}<extra></extra>")
    left.plotly_chart(bars, width="stretch")

    right.subheader("Biggest 5 transactions")
    big = an.biggest_transactions(table, n=5)
    right.dataframe(
        big.assign(date=big["date"].dt.date, amount=big["amount"].map(lambda a: an.inr(a, 2)))
        [["date", "merchant", "amount", "category"]],
        hide_index=True, width="stretch",
    )


# =================================================================================================
# Page 2: Transactions (search, filters, and the category correction editor)
# =================================================================================================
def show_transactions(table):
    st.title("Transactions")

    # --- Filters ---
    search = st.text_input("Search merchant or description", placeholder="e.g. swiggy")
    f1, f2, f3 = st.columns([2, 2, 2])
    first, last = table["date"].min().date(), table["date"].max().date()
    date_range = f1.date_input("Date range", value=(first, last), min_value=first, max_value=last)
    categories = f2.multiselect("Category", cz.CATEGORIES, placeholder="All categories")
    top_amount = float(table["amount"].max())
    a1, a2 = f3.columns(2)
    min_amount = a1.number_input("Min ₹", min_value=0.0, max_value=top_amount, value=0.0)
    max_amount = a2.number_input("Max ₹", min_value=0.0, max_value=top_amount, value=top_amount)

    # date_input returns 1 date while the user is still picking the 2nd one: wait for both
    start, end = date_range if len(date_range) == 2 else (None, None)
    shown = an.filter_transactions(table, start, end, categories, min_amount, max_amount, search)

    # Totals of what is currently shown (debits and credits kept apart)
    spent, income = an.debits(shown)["amount"].sum(), an.credits(shown)["amount"].sum()
    st.caption(f"Showing **{len(shown)}** of {len(table)} transactions  |  "
               f"Spent {an.inr(spent)}  |  Income {an.inr(income)}")

    # --- Editable table: change a category with the dropdown ---
    st.caption("Wrong category? Pick the right one in the dropdown, then click 'Save corrections'.")
    view = shown[["date", "merchant", "type", "amount", "category"]].copy()
    view["date"] = view["date"].dt.date                               # date only, no 00:00:00
    view["amount"] = view["amount"].map(lambda a: an.inr(a, 2))       # ₹1,25,000.00 (text, so no sorting by amount)
    # The editor key includes the filters: if they change, the editor starts fresh instead of
    # mixing up edits made on a different set of rows.
    filter_key = hash((str(date_range), tuple(categories), min_amount, max_amount, search))
    edited = st.data_editor(
        view,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "merchant": "Merchant",
            "type": "Type",
            "amount": st.column_config.TextColumn("Amount"),
            "category": st.column_config.SelectboxColumn("Category", options=cz.CATEGORIES, required=True),
        },
        disabled=["date", "merchant", "type", "amount"],              # only the category can be changed
        hide_index=True,
        width="stretch",
        key=f"editor_{st.session_state.editor_version}_{filter_key}",
    )

    # Rows (by their original row number, kept in the index) where the dropdown differs from the prediction
    changed = edited.index[edited["category"] != shown["category"]]
    if st.button(f"Save corrections ({len(changed)} changed)", disabled=len(changed) == 0):
        # Save the masked description (not the merchant) because that is what the model trains on
        rows = [(st.session_state.parsed.at[i, "description"], edited.at[i, "category"])
                for i in changed]
        cz.save_corrections(rows)
        # Re-categorize: other rows with the same merchant pick up the correction too
        st.session_state.table = categorize(st.session_state.parsed)
        st.session_state.editor_version += 1
        st.toast(f"Saved {len(rows)} correction(s) to {cz.CORRECTIONS_PATH}.")  # survives the rerun
        st.rerun()


# =================================================================================================
# Page 3: Model (retrain with the saved corrections)
# =================================================================================================
def show_model():
    st.title("Model")
    n_corrections = len(cz.load_corrections())
    st.write(f"Saved corrections: **{n_corrections}**  |  Current model: **{st.session_state.bundle['name']}**")

    if st.button("Retrain model", disabled=n_corrections == 0,
                 help="Save at least one correction first." if n_corrections == 0 else None):
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

    def delta(key):
        """'+1.2 pts (was 66.0%)' after a retrain, nothing before one."""
        if not result:
            return None
        return f"{(result['after'][key] - result['before'][key]) * 100:+.1f} pts (was {result['before'][key]:.1%})"

    col1, col2, col3 = st.columns(3)
    col1.metric("Hybrid (rules + ML), messy set", f"{scores['hybrid']:.1%}", delta=delta("hybrid"))
    col2.metric("ML only, messy set", f"{scores['ml_only']:.1%}", delta=delta("ml_only"))
    if result:
        col3.metric("Held-out split (incl. corrections)", f"{result['bundle']['clean_accuracy']:.1%}")
        st.caption(f"Retrained on the original data + {result['bundle']['n_corrections']} correction(s).")


# --- Route to the chosen page ----------------------------------------------------------------
if page == "Model":
    show_model()
elif "table" not in st.session_state:
    st.title(page)
    st.info("Upload a bank / UPI statement (CSV or Excel) in the sidebar to get started.")
elif page == "Dashboard":
    show_dashboard(st.session_state.table)
else:
    show_transactions(st.session_state.table)
