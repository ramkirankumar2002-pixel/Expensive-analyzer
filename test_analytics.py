# test_analytics.py - small tests for analytics.py (plain asserts, no pytest needed).
# Run with:  python test_analytics.py

import pandas as pd

import analytics as an


def sample():
    """6 transactions: 4 debits, 2 credits (a salary and a refund)."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-04-30", "2026-05-01", "2026-05-02",
                                "2026-05-10", "2026-05-15", "2026-06-01"]),
        "description": ["UPI Swiggy", "UPI Swiggy", "Rent", "Salary", "Amazon refund", "UPI DMart"],
        "merchant": ["Swiggy", "Swiggy", "Landlord", "Employer", "Amazon", "DMart"],
        "amount": [500.0, 300.0, 10000.0, 50000.0, 700.0, 1000.0],
        "type": ["debit", "debit", "debit", "credit", "credit", "debit"],
        "category": ["Food", "Food", "Rent", "Others", "Shopping", "Groceries"],
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
    assert s["spent"] == 11800.0           # 500 + 300 + 10000 + 1000 (refund NOT subtracted/added here)
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
    assert len(an.filter_transactions(df)) == 6                                    # no filters
    assert len(an.filter_transactions(df, start="2026-05-01", end="2026-05-10")) == 3   # end day included
    assert len(an.filter_transactions(df, categories=["Food"])) == 2
    assert len(an.filter_transactions(df, min_amount=1000, max_amount=10000)) == 2
    assert len(an.filter_transactions(df, search="swiggy")) == 2                   # case-insensitive
    assert len(an.filter_transactions(df, search="  ")) == 6                       # blank search = no filter
    assert len(an.filter_transactions(df, categories=["Food"], min_amount=400)) == 1   # filters combine


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All analytics tests passed.")
