# goals.py - savings goals: how much to save per month, are you on track, and what to cut.
# (No Streamlit in here, so it is easy to test.)
#
# A goal is a plain dict, so it can be saved as JSON:
#   {"id": "a1b2c3", "name": "Laptop", "target": 60000.0, "target_date": "2027-06-01", "saved": 10000.0}

import datetime as dt
import json
import math
import uuid

import pandas as pd

import analytics as an
import insights as ins

GOALS_PATH = "goals.json"

ON_TRACK, SLIGHTLY_BEHIND, OFF_TRACK = "On track", "Slightly behind", "Off track"
BEHIND_AT = 0.8     # saving at least 80% of what is needed = "Slightly behind", less = "Off track"

CUTTABLE = ["Food", "Shopping", "Entertainment", "Travel"]   # Rent, Bills etc. are never suggested
MAX_CUT = 0.5       # never suggest cutting more than half of a category (it would not be realistic)


# =================================================================================================
# 1. Creating, saving and loading goals
# =================================================================================================
def make_goal(name, target, target_date, saved=0.0):
    """Check the values and build the goal dict. Raises ValueError with a message for the user."""
    name = str(name).strip()
    if not name:
        raise ValueError("Give the goal a name.")
    target, saved = float(target), float(saved)
    if target <= 0:
        raise ValueError("The target amount must be more than 0.")
    if saved < 0:
        raise ValueError("The amount already saved cannot be negative.")
    return {"id": uuid.uuid4().hex[:8], "name": name, "target": target,
            "target_date": pd.Timestamp(target_date).date().isoformat(), "saved": saved}


def load_goals(path=GOALS_PATH):
    """List of goals. A missing or broken file means 'no goals yet'; a broken entry is skipped."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    goals = []
    for item in data if isinstance(data, list) else []:
        try:
            goal = make_goal(item["name"], item["target"], item["target_date"], item["saved"])
            goal["id"] = str(item["id"])                    # keep the saved id
            goals.append(goal)
        except (KeyError, TypeError, ValueError):
            continue
    return goals


def save_goals(goals, path=GOALS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(goals, f, indent=2, ensure_ascii=False)


# =================================================================================================
# 2. What you save now (from the statement)
# =================================================================================================
def recent_savings(df, n_months=3):
    """Average monthly savings (income - spending) over the last n months of the statement.

    Returns {"savings": average per month,
             "months": the months used (oldest first),
             "spend_by_category": {category: average spend per month}}

    - Spending and income are counted exactly like on the Dashboard (Transfers are not spending,
      self transfers are neither).
    - If the statement ends mid-month, that month is left out: it may have the salary but only part
      of the spending. (If it is the only month we have, we have to use it.)
    - With less than n months of data, we average over the months we do have."""
    if df.empty:
        return {"savings": 0.0, "months": [], "spend_by_category": {}}
    month_of_row = df["date"].dt.to_period("M")
    first, last = month_of_row.min(), month_of_row.max()
    if ins.ends_mid_month(df) and last > first:
        last -= 1
    start = max(first, last - (n_months - 1))
    months = list(pd.period_range(start, last, freq="M"))    # calendar months, so an empty month counts too

    window = df[month_of_row.isin(months)]
    totals = an.summary(window)
    by_category = an.category_totals(window).set_index("category")["amount"]
    return {"savings": float(totals["savings"]) / len(months),
            "months": months,
            "spend_by_category": {c: float(a) / len(months) for c, a in by_category.items()}}


# =================================================================================================
# 3. Is the goal on track?
# =================================================================================================
def months_between(today, target_date):
    """Whole months from today to the target date. 0 if the date is today or in the past;
    a future date less than a month away counts as 1 (there is still one chance to save)."""
    today, target_date = pd.Timestamp(today).normalize(), pd.Timestamp(target_date).normalize()
    if target_date <= today:
        return 0
    months = (target_date.year - today.year) * 12 + (target_date.month - today.month)
    if target_date.day < today.day:                          # the last month is not complete
        months -= 1
    return max(months, 1)


def status_of(needed, savings):
    """Compare what you save per month with what you need to save per month."""
    if savings >= needed:
        return ON_TRACK
    if savings >= needed * BEHIND_AT:
        return SLIGHTLY_BEHIND
    return OFF_TRACK


def reach_date(remaining, savings, today):
    """The date you reach the goal if you keep saving `savings` per month.
    None = never at this rate (nothing saved per month, or it would take over 100 years)."""
    today = pd.Timestamp(today).normalize()
    if remaining <= 0:
        return today.date()
    if savings <= 0 or remaining / savings > 1200:
        return None
    return (today + pd.DateOffset(months=math.ceil(remaining / savings))).date()


def evaluate(goal, savings, today=None):
    """All the numbers shown for one goal. savings = average monthly savings from recent_savings().

    Returns months_left, remaining, needed (per month), savings, gap (per month),
    progress (0 to 1), reached, date_passed, status, reach_date."""
    today = today or dt.date.today()
    remaining = max(goal["target"] - goal["saved"], 0.0)
    months_left = months_between(today, goal["target_date"])
    reached = remaining == 0
    date_passed = months_left == 0 and not reached
    needed = remaining / months_left if months_left else remaining   # no months left = all of it now

    if reached:
        status = ON_TRACK
    elif date_passed:
        status = OFF_TRACK                                   # the date is gone and money is still missing
    else:
        status = status_of(needed, savings)

    return {"months_left": months_left, "remaining": remaining, "needed": needed, "savings": savings,
            "gap": max(needed - savings, 0.0), "progress": min(goal["saved"] / goal["target"], 1.0),
            "reached": reached, "date_passed": date_passed, "status": status,
            "reach_date": reach_date(remaining, savings, today)}


# =================================================================================================
# 4. What to cut when a goal is off track
# =================================================================================================
def suggest_cuts(gap, spend_by_category):
    """Share the monthly gap between the categories that can be cut.

    Every category is cut by the same percentage (gap / total cuttable spend), so the one you spend
    the most on gives the most rupees. The percentage is never more than MAX_CUT (50%).

    Returns (cuts, uncovered):
      cuts      = [{"category", "cut", "current", "new"}, ...] biggest first, whole rupees per month
      uncovered = the part of the gap that these cuts cannot close"""
    current = {c: spend_by_category.get(c, 0.0) for c in CUTTABLE}
    current = {c: a for c, a in current.items() if round(a) > 0}
    total = sum(current.values())
    if gap <= 0:
        return [], 0.0
    if total == 0:
        return [], float(gap)

    share = min(gap / total, MAX_CUT)
    cuts = []
    for category, amount in sorted(current.items(), key=lambda item: item[1], reverse=True):
        now = round(amount)
        cut = math.ceil(round(amount * share, 6))            # round UP so the cuts really close the gap
        if cut > 0:
            cuts.append({"category": category, "cut": cut, "current": now, "new": now - cut})
    return cuts, max(float(gap) - sum(c["cut"] for c in cuts), 0.0)


def cut_text(c):
    """'Cut Food by ₹1,200/month (from ₹4,800 to ₹3,600)'"""
    return f"Cut {c['category']} by {an.inr(c['cut'])}/month (from {an.inr(c['current'])} to {an.inr(c['new'])})"


# =================================================================================================
# 5. Numbers for the AI report
# =================================================================================================
def report_stats(goals, df, today=None):
    """Goal numbers for the AI report. The goal NAME is left out on purpose (it is free text typed by
    the user); the only text is the fixed status label."""
    savings = recent_savings(df)["savings"]
    out = []
    for goal in goals:
        r = evaluate(goal, savings, today)
        out.append({"target": round(goal["target"]), "saved": round(goal["saved"]),
                    "months_left": r["months_left"], "needed_per_month": round(r["needed"]),
                    "average_monthly_savings": round(savings), "status": r["status"]})
    return out
