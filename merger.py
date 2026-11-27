# merger.py - combine several parsed statements into one table (no Streamlit in here, so it is easy to test).
#
# Pipeline (see merge_statements at the bottom):
#   1. tag_source           -> add a "source" column (which account each row came from)
#   2. remove_duplicates    -> drop rows repeated within the same source
#   3. sort by date
#   4. find_self_transfers  -> money moved between the user's own accounts
#
# Input is the table from parser.parse_statement: date, description (masked), amount, type, merchant

import os

import pandas as pd

# A debit and a credit this many days apart (or less) can be the two sides of one self transfer
MAX_GAP_DAYS = 2

# Two rows are "the same transaction" when all of these match. "type" is included so that a payment and
# its reversal (same day, amount and text, but money going the opposite way) are both kept.
DUPLICATE_COLUMNS = ["date", "amount", "description", "type", "source"]


def default_labels(files):
    """Suggest a short source label for each file. files = [(file name, file contents), ...]

    The label is the file name without its extension ('hdfc_march.csv' -> 'hdfc_march').
    Banks often use the same name ('statement.csv'), so a DIFFERENT file with an already used name
    gets ' (2)', ' (3)'...  The very same file added twice keeps one label, so its rows are
    recognised as duplicates."""
    labels, owner = [], {}                       # owner = {label: the contents that use it}
    for name, contents in files:
        stem = os.path.splitext(name)[0]
        label, number = stem, 2
        while label in owner and owner[label] != contents:
            label, number = f"{stem} ({number})", number + 1
        owner[label] = contents
        labels.append(label)
    return labels


def tag_source(df, source):
    """A copy of the table with every row marked with the account it came from."""
    return df.assign(source=source)


def remove_duplicates(df):
    """Drop exact duplicates (same date, amount, description, type and source); the first one is kept.
    Returns (table, number of rows removed)."""
    out = df.drop_duplicates(subset=DUPLICATE_COLUMNS, keep="first")
    return out, len(df) - len(out)


def find_self_transfers(df, max_gap_days=MAX_GAP_DAYS):
    """Find money moved between the user's own accounts: a debit in one source and a credit of the
    same amount in ANOTHER source, at most max_gap_days apart (either one may be dated first, because
    banks do not always post on the same day).

    Each row can belong to one pair only. When there are several choices, the closest dates are paired
    first. Returns a list of (debit row number, credit row number)."""
    rows = df.assign(day=df["date"].dt.normalize(),             # ignore the time of day
                     paise=(df["amount"] * 100).round())        # whole numbers compare exactly (decimals may not)
    rows["row"] = rows.index
    columns = ["row", "day", "paise", "source"]
    debit_rows = rows.loc[rows["type"] == "debit", columns]
    credit_rows = rows.loc[rows["type"] == "credit", columns]

    # Every debit next to every credit of the same amount, then keep only the possible ones
    pairs = debit_rows.merge(credit_rows, on="paise", suffixes=("_debit", "_credit"))
    pairs["gap"] = (pairs["day_credit"] - pairs["day_debit"]).dt.days.abs()
    pairs = pairs[(pairs["source_debit"] != pairs["source_credit"]) & (pairs["gap"] <= max_gap_days)]

    found, used = [], set()
    for pair in pairs.sort_values(["gap", "day_debit", "row_debit", "row_credit"]).itertuples():
        if pair.row_debit not in used and pair.row_credit not in used:
            found.append((pair.row_debit, pair.row_credit))
            used.update([pair.row_debit, pair.row_credit])
    return found


def merge_statements(statements):
    """Combine several statements. statements = [(source label, parsed table), ...]

    Returns (table, summary):
      table   = all rows sorted by date, with "source" and "self_transfer" (True / False) columns
      summary = {"duplicates_removed": rows dropped, "self_transfers": number of debit + credit pairs}"""
    table = pd.concat([tag_source(df, label) for label, df in statements], ignore_index=True)
    table, removed = remove_duplicates(table)
    # mergesort keeps rows of the same date in their original order
    table = table.sort_values("date", kind="mergesort").reset_index(drop=True)

    pairs = find_self_transfers(table)
    table["self_transfer"] = False
    table.loc[[row for pair in pairs for row in pair], "self_transfer"] = True
    return table, {"duplicates_removed": removed, "self_transfers": len(pairs)}
