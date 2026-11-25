# analytics.py - the number-crunching behind the dashboard (no Streamlit in here, so it is easy to test).
#
# Golden rule: only DEBITS are spending, only CREDITS are income.
# Refunds and salary are credits, so they never show up in the category totals or spending charts.
#
# Input everywhere is the table from parser + categorizer:
#   date, description, amount (always positive), type ("debit"/"credit"), merchant, category

import re

import pandas as pd


def inr(amount, decimals=0):
    """Format a number the Indian way with the rupee sign: 125000 -> '₹1,25,000'.
    Indian grouping = the last 3 digits, then groups of 2 (lakhs, crores)."""
    if pd.isna(amount):
        return "-"
    sign = "-" if amount < 0 else ""
    whole, _, paise = f"{abs(amount):.{decimals}f}".partition(".")   # also rounds for us
    head, tail = whole[:-3], whole[-3:]
    head = re.sub(r"(?<=\d)(?=(\d{2})+$)", ",", head)               # a comma before every 2 digits from the right
    grouped = f"{head},{tail}" if head else tail
    return f"{sign}₹{grouped}" + (f".{paise}" if paise else "")


def debits(df):
    """Only the rows where money went out (spending)."""
    return df[df["type"] == "debit"]


def credits(df):
    """Only the rows where money came in (salary, refunds, ...)."""
    return df[df["type"] == "credit"]


def summary(df):
    """Numbers for the top cards: total spent, income, savings and savings %.
    Savings % is None when there is no income (it would be a division by zero)."""
    spent = debits(df)["amount"].sum()
    income = credits(df)["amount"].sum()
    savings = income - spent
    savings_pct = savings / income * 100 if income > 0 else None
    return {"spent": spent, "income": income, "savings": savings, "savings_pct": savings_pct}


def category_totals(df):
    """Spending per category (debits only), biggest first. Columns: category, amount."""
    out = debits(df).groupby("category", as_index=False)["amount"].sum()
    return out.sort_values("amount", ascending=False).reset_index(drop=True)


def monthly_spending(df):
    """Spending per calendar month (debits only), oldest first. Columns: month, amount.
    'month' is the first day of the month, so the chart sorts correctly across years."""
    spend = debits(df)
    month = spend["date"].dt.to_period("M").dt.to_timestamp()
    out = spend.groupby(month)["amount"].sum()
    out.index.name = "month"
    return out.reset_index()


def top_merchants(df, n=5):
    """The n merchants where the most money was spent. Columns: merchant, amount, count."""
    out = (debits(df).groupby("merchant")["amount"]
           .agg(amount="sum", count="count").reset_index())
    return out.sort_values("amount", ascending=False).head(n).reset_index(drop=True)


def biggest_transactions(df, n=5):
    """The n largest single payments (debits only)."""
    return debits(df).sort_values("amount", ascending=False).head(n)


def filter_transactions(df, start=None, end=None, categories=None,
                        min_amount=None, max_amount=None, search=""):
    """Apply the Transactions-page filters. Any filter left as None / empty is skipped.
    start / end are dates (inclusive); search matches the merchant or description text."""
    keep = pd.Series(True, index=df.index)
    if start is not None:
        keep &= df["date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= df["date"] < pd.Timestamp(end) + pd.Timedelta(days=1)   # include the whole end day
    if categories:
        keep &= df["category"].isin(categories)
    if min_amount is not None:
        keep &= df["amount"] >= min_amount
    if max_amount is not None:
        keep &= df["amount"] <= max_amount
    if search and search.strip():
        text = df["merchant"].astype(str) + " " + df["description"].astype(str)
        keep &= text.str.contains(search.strip(), case=False, regex=False)
    return df[keep]
