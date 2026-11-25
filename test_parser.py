# test_parser.py - small tests for parser.py (plain asserts, no pytest needed).
# Run with:  python test_parser.py

import os
import tempfile

import pandas as pd

import parser as p


def test_rename_columns():
    # HDFC-style headers
    hdfc = pd.DataFrame(columns=["Date", "Narration", "Withdrawal Amt.", "Deposit Amt.", "Balance"])
    assert list(p.rename_columns(hdfc).columns) == ["date", "description", "debit", "credit", "Balance"]
    # SBI-style headers
    sbi = pd.DataFrame(columns=["Txn Date", "Description", "Debit", "Credit"])
    assert list(p.rename_columns(sbi).columns) == ["date", "description", "debit", "credit"]
    # two date-like columns must not both become "date"
    two = pd.DataFrame(columns=["Txn Date", "Value Date"])
    assert list(p.rename_columns(two).columns).count("date") == 1


def test_description_aliases():
    # other common names for the description column
    for name in ["Particulars", "Remarks", "Details", "Note", "Expenses", "Transaction Remarks"]:
        df = pd.DataFrame(columns=["Date", name, "Amount"])
        assert list(p.rename_columns(df).columns) == ["date", "description", "amount"], name


def test_pick_description_column():
    df = pd.DataFrame({"Date": ["2026-01-05"], "What": ["UPI/123456789012/Swiggy/swiggy@ybl"], "Amount": [250]})
    path = os.path.join(tempfile.gettempdir(), "no_desc.csv")
    df.to_csv(path, index=False)
    try:
        p.parse_statement(path)          # "What" is not a known name -> asks for help
        assert False, "should have raised"
    except p.MissingColumnError as error:
        assert error.columns == ["Date", "What", "Amount"]   # the app shows these in the dropdown
    out = p.parse_statement(path, description_column="What")  # the user picked it
    assert out["merchant"][0] == "Swiggy" and out["amount"][0] == 250


def test_parse_expenses_workbook():
    path = "D:/Copy of Expenses.xlsx"   # the user's own sheet (Date, Expenses, Category, Amount, ...)
    if not os.path.exists(path):
        return
    out = p.parse_statement(path)
    assert len(out) > 1000 and set(out["type"]) == {"debit"}
    assert out["description"].notna().all()


def test_build_amount_and_type():
    # A) separate debit / credit columns (commas and blanks included)
    a = pd.DataFrame({"debit": ["1,200.50", None], "credit": [None, "500"]})
    a = p.build_amount_and_type(a)
    assert list(a["amount"]) == [1200.5, 500.0]
    assert list(a["type"]) == ["debit", "credit"]
    # B) Dr/Cr type column
    b = p.build_amount_and_type(pd.DataFrame({"amount": [100, 200], "type": ["DR", "CR"]}))
    assert list(b["type"]) == ["debit", "credit"]
    # C) signed amount: negative = money out, amount becomes positive
    c = p.build_amount_and_type(pd.DataFrame({"amount": [-250.0, 900.0]}))
    assert list(c["type"]) == ["debit", "credit"]
    assert list(c["amount"]) == [250.0, 900.0]
    # no hint at all -> every row is a debit
    d = p.build_amount_and_type(pd.DataFrame({"amount": [50, 60]}))
    assert list(d["type"]) == ["debit", "debit"]


def test_convert_dates():
    df = pd.DataFrame({"date": ["25/11/2026", "01/02/2026", "not a date"]})
    out = p.convert_dates(df)
    assert len(out) == 2                                   # bad row dropped
    assert out["date"].iloc[0] == pd.Timestamp(2026, 11, 25)
    assert out["date"].iloc[1] == pd.Timestamp(2026, 2, 1)  # day first, not January 2
    # ISO dates (year first) must keep their meaning: 2026-05-01 is 1 May, 2026-10-27 is valid
    iso = p.convert_dates(pd.DataFrame({"date": ["2026-05-01", "2026-10-27"]}))
    assert list(iso["date"]) == [pd.Timestamp(2026, 5, 1), pd.Timestamp(2026, 10, 27)]


def test_mask_sensitive():
    assert p.mask_sensitive("rahul@ybl") == "ra****@ybl"
    assert p.mask_sensitive("A/c 50100123456789 debited") == "A/c **********6789 debited"
    assert p.mask_sensitive("pay to john.doe@okaxis") == "pay to jo****@okaxis"
    assert p.mask_sensitive("Netflix") == "Netflix"        # nothing to hide


def test_extract_merchant():
    assert p.extract_merchant("UPI/418273650192/Swiggy/swiggy@ybl") == "Swiggy"
    assert p.extract_merchant("UPI/418273650192/P2P/Rahul S/ra****@ybl") == "Rahul S"
    assert p.extract_merchant("NEFT-HDFCN52026112501-RAHUL SHARMA-SALARY") == "Rahul Sharma"
    assert p.extract_merchant("IMPS/412345678901/PRIYA M/HDFC/Rent") == "Priya M"
    assert p.extract_merchant("POS 4567XXXXXX1234 BIG BAZAAR MUMBAI") == "Big Bazaar Mumbai"
    assert p.extract_merchant("Rent - Landlord Mr Sharma") == "Rent - Landlord Mr Sharma"
    assert p.extract_merchant("Netflix") == "Netflix"


def test_parse_statement_csv():
    df = p.parse_statement("data/transactions.csv")
    assert list(df.columns)[:5] == ["date", "description", "amount", "type", "merchant"]
    assert len(df) == len(pd.read_csv("data/transactions.csv"))  # no row lost while parsing
    assert df["date"].notna().all()
    assert (df["amount"] > 0).all()
    # every UPI id must be masked: "xx****@bank", never a full id
    assert not df["description"].str.contains(r"[A-Za-z0-9._-]{3,}@", regex=True).any()
    assert not df["description"].str.contains(r"\d{9,}", regex=True).any()   # no long numbers left
    assert "Swiggy" in set(df["merchant"])


def test_parse_statement_excel():
    # build a tiny HDFC-style Excel file, parse it, then clean up
    raw = pd.DataFrame({
        "Date": ["25/11/2026", "26/11/2026"],
        "Narration": ["UPI/418273650192/Zomato/zomato@icici", "NEFT-HDFCN52026112501-ACME CORP-SALARY"],
        "Withdrawal Amt.": [350.0, None],
        "Deposit Amt.": [None, 50000.0],
    })
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "statement.xlsx")
    raw.to_excel(path, index=False)
    df = p.parse_statement(path)
    assert list(df["merchant"]) == ["Zomato", "Acme Corp"]
    assert list(df["type"]) == ["debit", "credit"]
    assert list(df["amount"]) == [350.0, 50000.0]
    assert "zomato@icici" not in df["description"].iloc[0]


if __name__ == "__main__":
    # run every function whose name starts with "test_"
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All parser tests passed.")
