# test_analytics.py - small tests for analytics.py (plain asserts, no pytest needed).
# Run with:  python test_analytics.py

import pandas as pd

import analytics as an


def sample():
    """8 transactions: 6 debits (one is a Transfer), 2 credits (a salary and a refund)."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-04-30", "2026-05-01", "2026-05-02",
                                "2026-05-10", "2026-05-15", "2026-06-01",
                                "2026-06-02", "2026-06-03"]),
        "description": ["UPI Swiggy", "UPI Swiggy", "Rent", "Salary", "Amazon refund", "UPI DMart", "UPI Rahul", "Sent to Mom"],
        "merchant": ["Swiggy", "Swiggy", "Landlord", "Employer", "Amazon", "DMart", "Rahul", "Mom"],
        "amount": [500.0, 300.0, 10000.0, 50000.0, 700.0, 1000.0, 5000.0, 20000.0],
        "type": ["debit", "debit", "debit", "credit", "credit", "debit", "debit", "debit"],
        "category": ["Food", "Food", "Rent", "Others", "Shopping", "Groceries", "Transfers", "Transfers"],
    })


def test_inr():
    assert an.inr(125000) == "₹1,25,000"
    assert an.inr(999) == "₹999"
    assert an.inr(1000) == "₹1,000"
    assert an.inr(12345678) == "₹1,23,45,678"
    assert an.inr(1234.5, decimals=2) == "₹1,234.50"
    assert an.inr(-2500) == "-₹2,500"
    assert an.inr(0) == "₹0"
    assert an.inr(float("nan")) == "-"


def test_summary_counts_credits_as_income_only():
    s = an.summary(sample())
    assert s["spent"] == 11800.0           # 500 + 300 + 10000 + 1000 (Transfers and credits not included)
    assert s["transfers"] == 25000.0       # 5000 + 20000 shown separately
    assert s["income"] == 50700.0          # salary + refund
    assert s["savings"] == 50700.0 - 11800.0
    assert abs(s["savings_pct"] - (38900 / 50700 * 100)) < 1e-9


def test_summary_without_income():
    s = an.summary(sample()[lambda d: d["type"] == "debit"])
    assert s["income"] == 0 and s["savings_pct"] is None


def test_category_totals_ignore_credits():
    totals = an.category_totals(sample())
    assert dict(zip(totals["category"], totals["amount"])) == {"Rent": 10000.0, "Food": 800.0, "Groceries": 1000.0}
    assert totals.iloc[0]["category"] == "Rent"      # biggest first
    assert "Shopping" not in set(totals["category"])  # only a refund was in Shopping


def test_transfers_are_not_spending():
    df = sample()
    assert "Transfers" not in set(an.category_totals(df)["category"])
    assert "Rahul" not in set(an.top_merchants(df, n=10)["merchant"])
    assert 20000.0 not in list(an.biggest_transactions(df, n=10)["amount"])
    assert an.monthly_spending(df)["amount"].sum() == 11800.0
    assert list(an.transfers_out(df)["amount"]) == [5000.0, 20000.0]


def test_monthly_spending():
    m = an.monthly_spending(sample())
    assert list(m["month"]) == [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-01")]
    assert list(m["amount"]) == [500.0, 10300.0, 1000.0]


def test_top_merchants():
    top = an.top_merchants(sample(), n=2)
    assert list(top["merchant"]) == ["Landlord", "DMart"]
    swiggy = an.top_merchants(sample())
    assert swiggy[swiggy["merchant"] == "Swiggy"].iloc[0]["count"] == 2


def test_biggest_transactions_are_debits():
    big = an.biggest_transactions(sample(), n=2)
    assert list(big["amount"]) == [10000.0, 1000.0]    # the 50000 salary is a credit, so not here


def test_filter_transactions():
    df = sample()
    assert len(an.filter_transactions(df)) == 8                                    # no filters
    assert len(an.filter_transactions(df, start="2026-05-01", end="2026-05-10")) == 3   # end day included
    assert len(an.filter_transactions(df, categories=["Food"])) == 2
    assert len(an.filter_transactions(df, min_amount=1000, max_amount=10000)) == 3
    assert len(an.filter_transactions(df, search="swiggy")) == 2                   # case-insensitive
    assert len(an.filter_transactions(df, search="  ")) == 8                       # blank search = no filter
    assert len(an.filter_transactions(df, categories=["Food"], min_amount=400)) == 1   # filters combine


def test_month_over_month():
    df = sample()
    mom = an.month_over_month(df)                      # June vs May
    # May: spent 10300 (rent + swiggy), income 50700 ; June: spent 1000 (DMart), income 0
    assert abs(mom["spent"] - (1000 - 10300) / 10300 * 100) < 1e-9
    assert abs(mom["income"] - (-100.0)) < 1e-9
    assert mom["savings_pct"] is None                  # June has no income, so no savings %
    one_month = df[df["date"] >= "2026-06-01"]
    assert all(v is None for v in an.month_over_month(one_month).values())


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All analytics tests passed.")
