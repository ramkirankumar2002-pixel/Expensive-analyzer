# insights.py - the "smart" analysis behind the Insights page (no Streamlit in here, so it is easy to test).
#
#   1. unusual_spends     -> Isolation Forest flags debits that look unlike the rest
#   2. find_subscriptions -> same merchant, similar amount, about 30 days apart
#   3. budgets            -> monthly limit per category, saved in budgets.json
#
# Input is the same table as analytics.py: date, description, amount (positive), type, merchant, category

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

import analytics as an

BUDGETS_PATH = "budgets.json"
WARN_AT = 80      # % of budget used -> yellow warning
OVER_AT = 100     # % of budget used -> red alert


# =================================================================================================
# 1. Unusual spends (Isolation Forest)
# =================================================================================================
def unusual_spends(df, contamination=0.03, min_rows=30):
    """Spending (debits except Transfers) that looks unusual. Columns: date, merchant, category, amount, reason (biggest first).

    Isolation Forest isolates odd points quickly with random splits; points isolated in few splits
    are 'anomalies'. Features: amount (log scale), category (one-hot), day of week.
    contamination = the share of rows we expect to be unusual (3%)."""
    cols = ["date", "merchant", "category", "amount", "reason"]
    d = an.spending(df).reset_index(drop=True)   # debits except Transfers: sending money to people is not "unusual spending"
    if len(d) < min_rows:                      # too little data to know what "usual" is
        return pd.DataFrame(columns=cols)

    features = pd.get_dummies(d["category"]).astype(float)       # one 0/1 column per category
    features["log_amount"] = np.log1p(d["amount"])               # log: 8000 stands out, rent doesn't swamp everything
    features["weekday"] = d["date"].dt.dayofweek                 # 0 = Monday ... 6 = Sunday
    flagged = IsolationForest(contamination=contamination, random_state=42).fit_predict(features) == -1

    rows = []
    for i in d.index[flagged]:
        row = d.loc[i]
        # "Your usual" = median of the OTHER payments to this merchant (or this category if too few)
        same = d[(d["merchant"] == row["merchant"]) & (d["category"] == row["category"]) & (d.index != i)]["amount"]
        if len(same) >= 3:
            usual, place = same.median(), f"at {row['merchant']}"
        else:
            same = d[(d["category"] == row["category"]) & (d.index != i)]["amount"]
            if same.empty:
                continue
            usual, place = same.median(), f"on {row['merchant']}"
        if row["amount"] <= usual:             # unusually LOW is not worth warning about
            continue
        reason = f"{an.inr(row['amount'])} {place} — your usual is about {an.inr(usual)}"
        rows.append((row["date"], row["merchant"], row["category"], row["amount"], reason))

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("amount", ascending=False).reset_index(drop=True)


# =================================================================================================
# 2. Subscriptions and recurring payments
# =================================================================================================
def find_subscriptions(df, min_payments=3, gap_range=(25, 35), amount_tolerance=0.10):
    """Merchants paid about every month for about the same amount (Netflix, rent, gym...).
    Columns: merchant, category, monthly_amount, payments, last_paid, next_due (biggest first).

    A merchant counts when it has >= 3 payments, most gaps between payments are 25-35 days,
    and every amount is within +-10% of its median amount."""
    cols = ["merchant", "category", "monthly_amount", "payments", "last_paid", "next_due"]
    d = an.debits(df)

    def looks_monthly(g):
        """>= 3 payments, most gaps 25-35 days, every amount within +-10% of the median."""
        g = g.sort_values("date")
        if len(g) < min_payments:
            return False
        monthly_gaps = g["date"].diff().dt.days.dropna().between(*gap_range)
        typical = g["amount"].median()
        steady = ((g["amount"] - typical).abs() <= typical * amount_tolerance).all()
        return monthly_gaps.mean() >= 0.7 and steady

    rows, used = [], pd.Series(False, index=d.index)
    # Pass 1: same merchant name (Netflix, Spotify, gym...)
    for (merchant, category), g in d.groupby(["merchant", "category"]):
        if looks_monthly(g):
            rows.append((merchant, category, g["amount"].median(), len(g), g["date"].max()))
            used[g.index] = True
    # Pass 2: rent-like payments whose narration changes every month ("Room Rent", "Monthly Rent"...):
    # group by category + exact amount instead, and show the most common name.
    rest = d[~used]
    for (category, _), g in rest.groupby(["category", rest["amount"].round(0)]):
        if looks_monthly(g):
            rows.append((g["merchant"].mode().iloc[0], category, g["amount"].median(), len(g), g["date"].max()))

    out = pd.DataFrame(rows, columns=cols[:5])
    out["next_due"] = out["last_paid"] + pd.Timedelta(days=30)
    return out.sort_values("monthly_amount", ascending=False).reset_index(drop=True)[cols]


def subscription_total(subs):
    """What the recurring payments cost per month."""
    return float(subs["monthly_amount"].sum())


# =================================================================================================
# 3. Budgets
# =================================================================================================
def load_budgets(path=BUDGETS_PATH):
    """{category: monthly limit}. A missing or broken file just means 'no budgets yet'."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items() if float(v) > 0}
    except (OSError, ValueError, AttributeError):
        return {}


def save_budgets(budgets, path=BUDGETS_PATH):
    """Write the budgets to JSON; categories with a limit of 0 are dropped."""
    clean = {k: float(v) for k, v in budgets.items() if float(v) > 0}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2)


def status_of(pct):
    """'ok' below 80%, 'warning' from 80%, 'over' from 100%."""
    if pct >= OVER_AT:
        return "over"
    if pct >= WARN_AT:
        return "warning"
    return "ok"


def budget_status(df, month, budgets):
    """Spent vs budget for one month. Columns: category, budget, spent, pct, status.
    month = any date inside the month (e.g. pd.Timestamp('2026-05-01')). Transfers are not spending."""
    spend = an.spending(df)
    in_month = spend[spend["date"].dt.to_period("M") == pd.Timestamp(month).to_period("M")]
    spent_by_cat = in_month.groupby("category")["amount"].sum()
    rows = []
    for category, limit in budgets.items():
        spent = float(spent_by_cat.get(category, 0.0))
        pct = spent / limit * 100
        rows.append((category, limit, spent, pct, status_of(pct)))
    return pd.DataFrame(rows, columns=["category", "budget", "spent", "pct", "status"])


# =================================================================================================
# 4. Spending forecast (Linear Regression)
# =================================================================================================
def monthly_by_category(df):
    """Spending per month (rows) and category (columns); months with no spending in a category = 0.
    Debits except Transfers only. The index is the first day of each month, oldest first."""
    spend = an.spending(df)
    month = spend["date"].dt.to_period("M").dt.to_timestamp()
    table = spend.pivot_table(index=month, columns="category", values="amount", aggfunc="sum", fill_value=0.0)
    return table.sort_index()


def forecast_next_month(df, min_months=3):
    """Predict next month's spending per category from past monthly totals.
    Returns (history, forecast, dropped_partial):
      history  = monthly_by_category() of the complete months used for training
      forecast = DataFrame(category, predicted), biggest first (empty if too little data)
      dropped_partial = True when the last month was left out because the statement ends mid-month.

    For each category we fit a straight line  amount = a * month_number + b  (Linear Regression)
    and read it at the next month number. Negative predictions are set to 0."""
    history = monthly_by_category(df)
    empty = pd.DataFrame(columns=["category", "predicted"])
    if history.empty:
        return history, empty, False

    # A statement that stops mid-month would make its last month look cheap and bend the line down.
    last_day = df["date"].max()
    dropped_partial = (last_day.to_period("M").end_time.normalize() - last_day.normalize()).days > 3
    if dropped_partial:
        history = history.iloc[:-1]
    if len(history) < min_months:
        return history, empty, dropped_partial

    x = np.arange(len(history)).reshape(-1, 1)          # month numbers 0, 1, 2, ...
    next_x = [[len(history)]]                           # the month we want to predict
    rows = []
    for category in history.columns:
        model = LinearRegression().fit(x, history[category].values)
        rows.append((category, max(float(model.predict(next_x)[0]), 0.0)))
    forecast = pd.DataFrame(rows, columns=["category", "predicted"])
    return history, forecast.sort_values("predicted", ascending=False).reset_index(drop=True), dropped_partial
