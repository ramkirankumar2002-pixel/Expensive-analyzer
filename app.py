# app.py - Streamlit app: upload a statement, fix wrong categories, retrain the model.
# Run with:  streamlit run app.py

import streamlit as st

import categorizer as cz
from parser import MissingColumnError, parse_statement

st.title("Expense Analyzer")

# --- Remember things between reruns (Streamlit re-runs this whole file on every click) -------
if "bundle" not in st.session_state:
    st.session_state.bundle = cz.load_model()   # the trained model (loaded once)
if "scores" not in st.session_state:
    st.session_state.scores = cz.messy_scores(st.session_state.bundle)  # accuracy of the current model
st.session_state.setdefault("editor_version", 0)  # bump it to reset the table's pending edits


def categorize(parsed):
    """Rules + ML + the user's saved corrections for every row of the parsed statement."""
    return cz.categorize_dataframe(parsed, bundle=st.session_state.bundle)


# --- 1. Upload and categorize ----------------------------------------------------------------
uploaded = st.file_uploader("Upload your bank / UPI statement", type=["csv", "xlsx", "xls"])
if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.get("file_id") != file_id:       # a new file: parse it once
        try:
            st.session_state.parsed = parse_statement(uploaded)
        except MissingColumnError as error:
            # No column we recognise: let the user say which one holds the description.
            st.warning(f"{error}. Pick the column that describes each transaction.")
            chosen = st.selectbox("Description column", error.columns, index=None,
                                  placeholder="Choose a column...")
            if chosen is None:
                st.stop()
            try:
                st.session_state.parsed = parse_statement(uploaded, description_column=chosen)
            except ValueError as error2:
                st.error(str(error2))
                st.stop()
        except ValueError as error:
            st.error(str(error))
            st.stop()
        st.session_state.file_id = file_id
        st.session_state.table = categorize(st.session_state.parsed)
        st.session_state.editor_version += 1

# --- 2. Editable table: change a category with the dropdown ----------------------------------
if "table" in st.session_state:
    table = st.session_state.table
    st.subheader("Transactions")
    st.caption("Wrong category? Pick the right one in the dropdown, then click 'Save corrections'.")
    edited = st.data_editor(
        table[["date", "merchant", "amount", "category"]],
        column_config={
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),  # no 00:00:00
            "category": st.column_config.SelectboxColumn("Category", options=cz.CATEGORIES, required=True),
        },
        disabled=["date", "merchant", "amount"],          # only the category can be changed
        hide_index=True,
        key=f"editor_{st.session_state.editor_version}",
    )

    # Rows where the dropdown value differs from what we predicted
    changed = edited.index[edited["category"] != table["category"]]
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

# --- 3. Retrain with the original data + corrections -----------------------------------------
st.divider()
st.subheader("Model")
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
