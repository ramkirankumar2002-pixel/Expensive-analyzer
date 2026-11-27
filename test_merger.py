# test_merger.py - small tests for merger.py (plain asserts, no pytest needed).
# Run with:  python test_merger.py

import pandas as pd

import analytics as an
import categorizer as cz
import merger as mg
import parser as p

BANK_A = "data/sample_bank_a.csv"
BANK_B = "data/sample_bank_b.csv"


def make(*rows):
    """A tiny parsed statement from rows of (date, description, amount, type)."""
    df = pd.DataFrame(rows, columns=["date", "description", "amount", "type"])
    df["date"] = pd.to_datetime(df["date"])
    df["merchant"] = df["description"]
    return df


def test_default_labels():
    assert mg.default_labels([("hdfc.csv", b"1"), ("sbi.xlsx", b"2")]) == ["hdfc", "sbi"]
    # two DIFFERENT files with one name must not be treated as the same account
    assert mg.default_labels([("statement.csv", b"1"), ("statement.csv", b"2")]) == ["statement", "statement (2)"]
    # the same file added twice keeps one label, so its rows count as duplicates
    assert mg.default_labels([("statement.csv", b"1"), ("statement.csv", b"1")]) == ["statement", "statement"]


def test_merge_tags_and_sorts():
    a = make(("2026-05-03", "Late", 10, "debit"), ("2026-05-01", "Early", 20, "debit"))
    b = make(("2026-05-02", "Middle", 30, "debit"))
    table, summary = mg.merge_statements([("A", a), ("B", b)])
    assert list(table["description"]) == ["Early", "Middle", "Late"]
    assert list(table["source"]) == ["A", "B", "A"]
    assert list(table.index) == [0, 1, 2]                 # the app finds rows by these numbers
    assert summary == {"duplicates_removed": 0, "self_transfers": 0}


def test_remove_duplicates():
    row = ("2026-05-01", "Netflix", 649, "debit")
    a = make(row, row,                                    # repeated inside one source -> one is removed
             ("2026-05-02", "Netflix", 649, "debit"),     # other date -> kept
             ("2026-05-01", "Netflix", 650, "debit"),     # other amount -> kept
             ("2026-05-01", "Spotify", 649, "debit"),     # other description -> kept
             ("2026-05-01", "Netflix", 649, "credit"))    # a reversal is not a duplicate -> kept
    table, summary = mg.merge_statements([("A", a), ("B", make(row))])   # same row in B -> kept
    assert summary["duplicates_removed"] == 1 and len(table) == 6
    # two uploads with the same label (overlapping statements of one account)
    table, summary = mg.merge_statements([("A", make(row)), ("A", make(row))])
    assert summary["duplicates_removed"] == 1 and len(table) == 1


def found(a_rows, b_rows):
    """How many self-transfer pairs are found between source A and source B."""
    return mg.merge_statements([("A", make(*a_rows)), ("B", make(*b_rows))])[1]["self_transfers"]


def test_self_transfer_rules():
    out = [("2026-05-10", "sent", 5000, "debit")]
    got = lambda day, amount=5000, kind="credit": [(day, "got", amount, kind)]
    assert found(out, got("2026-05-10")) == 1
    assert found(out, got("2026-05-12")) == 1             # exactly 2 days later
    assert found(out, got("2026-05-08")) == 1             # the credit may be posted first
    assert found(out, got("2026-05-13")) == 0             # 3 days: too far apart
    assert found(out, got("2026-05-10", 5000.5)) == 0     # different amount
    assert found(out, got("2026-05-10", 5000, "debit")) == 0   # two debits
    # times of day are ignored: 23:00 and 01:00 two calendar days later still count
    assert found([("2026-05-10 23:00", "sent", 5000, "debit")], got("2026-05-12 01:00")) == 1
    # a debit and credit inside the SAME file are not a transfer between two accounts
    assert found(out + got("2026-05-10"), [("2026-05-10", "other", 1, "debit")]) == 0


def test_self_transfer_pairs_one_to_one():
    a = make(("2026-05-10", "sent", 5000, "debit"))
    b = make(("2026-05-12", "far", 5000, "credit"), ("2026-05-10", "near", 5000, "credit"))
    table, summary = mg.merge_statements([("A", a), ("B", b)])
    assert summary["self_transfers"] == 1                 # one debit cannot match two credits
    assert set(table[table["self_transfer"]]["description"]) == {"sent", "near"}   # closest date wins


def test_self_transfers_are_not_spending_or_income():
    a = make(("2026-05-01", "salary", 50000, "credit"), ("2026-05-02", "sent", 5000, "debit"),
             ("2026-05-03", "food", 300, "debit"))
    b = make(("2026-05-02", "got", 5000, "credit"))
    table, _ = mg.merge_statements([("A", a), ("B", b)])
    table["category"] = ["Others", "Transfers", "Food", "Others"]
    table.loc[table["self_transfer"], "category"] = an.SELF_TRANSFER
    s = an.summary(table)
    assert (s["spent"], s["transfers"], s["income"]) == (300, 0, 50000)
    assert len(an.debits(table)) == 1 and len(an.credits(table)) == 1
    assert len(an.filter_transactions(table, source="B")) == 1
    assert len(an.filter_transactions(table, source=None)) == 4
    assert an.SELF_TRANSFER not in cz.CATEGORIES          # the fixed category list is unchanged


def test_sample_bank_files():
    a, b = p.parse_statement(BANK_A), p.parse_statement(BANK_B)
    assert (len(a), len(b)) == (14, 10)
    table, summary = mg.merge_statements([("bank_a", a), ("bank_b", b)])
    assert summary == {"duplicates_removed": 1, "self_transfers": 3}
    assert len(table) == 23 and table["date"].is_monotonic_increasing
    own = table[table["self_transfer"]]
    assert sorted(own["amount"]) == [5000, 5000, 15000, 15000, 20000, 20000]
    assert 2000 not in set(own["amount"])                 # the 4-days-apart look-alike is left alone
    assert not table["description"].str.contains(r"\d{9,}", regex=True).any()   # still masked


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All merger tests passed.")
