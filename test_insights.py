# test_insights.py - small tests for insights.py (plain asserts, no pytest needed).
# Run with:  python test_insights.py

import os
import tempfile

import numpy as np
import pandas as pd

import insights as ins


def frame(rows):
    """rows = (date, merchant, amount, category) -> a table like the app's."""
    df = pd.DataFrame(rows, columns=["date", "merchant", "amount", "category"])
    df["date"] = pd.to_datetime(df["date"])
    df["description"] = df["merchant"]
    df["type"] = "debit"
    return df


def test_unusual_spends_flags_the_big_zomato_order():
    rng = np.random.default_rng(0)
    days = pd.date_range("2026-01-01", periods=60)
    rows = [(d, "Zomato", 350 + rng.integers(-30, 30), "Food") for d in days]
    rows.append((pd.Timestamp("2026-03-05"), "Zomato", 8000, "Food"))
    out = ins.unusual_spends(frame(rows))
    assert len(out) >= 1
    top = out.iloc[0]
    assert top["amount"] == 8000 and "Zomato" in top["reason"] and "₹8,000" in top["reason"]
    assert "₹3" in top["reason"]                       # "your usual is about ₹350"


def test_unusual_spends_needs_enough_rows():
    rows = [("2026-01-01", "Zomato", 300, "Food")] * 5 + [("2026-01-02", "Zomato", 9000, "Food")]
    assert ins.unusual_spends(frame(rows)).empty


def test_unusual_spends_ignores_transfers():
    rng = np.random.default_rng(1)
    rows = [(d, "Zomato", 350 + rng.integers(-30, 30), "Food") for d in pd.date_range("2026-01-01", periods=60)]
    rows.append((pd.Timestamp("2026-03-05"), "Mom", 50000, "Transfers"))      # a huge transfer to a person
    assert "Mom" not in set(ins.unusual_spends(frame(rows))["merchant"])


def test_find_subscriptions():
    rows = []
    for m in range(6):
        rows.append((pd.Timestamp("2026-01-05") + pd.Timedelta(days=30 * m), "Netflix", 649, "Entertainment"))
        rows.append((pd.Timestamp("2026-01-01") + pd.Timedelta(days=30 * m), "Landlord", 12000, "Rent"))
    for d in ["2026-01-02", "2026-01-09", "2026-01-11", "2026-02-20"]:     # random, not monthly
        rows.append((pd.Timestamp(d), "Swiggy", 300, "Food"))
    for m, amt in enumerate([100, 900, 250, 4000]):                          # monthly but amounts vary a lot
        rows.append((pd.Timestamp("2026-01-10") + pd.Timedelta(days=30 * m), "Mixed", amt, "Others"))
    subs = ins.find_subscriptions(frame(rows))
    assert set(subs["merchant"]) == {"Netflix", "Landlord"}
    assert ins.subscription_total(subs) == 12649.0
    assert subs.iloc[0]["merchant"] == "Landlord"      # biggest first


def test_status_thresholds():
    assert ins.status_of(79.9) == "ok"
    assert ins.status_of(80) == "warning"
    assert ins.status_of(99.9) == "warning"
    assert ins.status_of(100) == "over"


def test_budget_status_filters_by_month():
    df = frame([("2026-05-02", "Swiggy", 800, "Food"), ("2026-05-20", "Zomato", 100, "Food"),
                ("2026-06-02", "Swiggy", 5000, "Food"), ("2026-05-03", "Mom", 9000, "Transfers")])
    out = ins.budget_status(df, pd.Timestamp("2026-05-01"), {"Food": 1000, "Travel": 500})
    food = out[out["category"] == "Food"].iloc[0]
    assert food["spent"] == 900 and food["pct"] == 90 and food["status"] == "warning"
    travel = out[out["category"] == "Travel"].iloc[0]
    assert travel["spent"] == 0 and travel["status"] == "ok"


def test_budgets_save_and_load():
    path = os.path.join(tempfile.mkdtemp(), "budgets.json")
    assert ins.load_budgets(path) == {}                # missing file
    ins.save_budgets({"Food": 5000, "Travel": 0}, path)
    assert ins.load_budgets(path) == {"Food": 5000.0}  # zero limit is dropped
    with open(path, "w") as f:
        f.write("not json")
    assert ins.load_budgets(path) == {}                # broken file


def month_rows(amounts, category="Food", last_day="2026-%02d-28"):
    """One payment per month on the 5th, plus a payment on the 28th so each month counts as complete."""
    rows = []
    for m, amt in enumerate(amounts, start=1):
        rows.append((f"2026-{m:02d}-05", "Shop", amt, category))
        rows.append((last_day % m, "Shop", 0, category))
    return rows


def test_forecast_continues_a_straight_line():
    history, forecast, dropped = ins.forecast_next_month(frame(month_rows([100, 200, 300])))
    assert not dropped and len(history) == 3
    assert round(forecast.iloc[0]["predicted"]) == 400


def test_forecast_never_negative():
    _, forecast, _ = ins.forecast_next_month(frame(month_rows([300, 100, 0])))
    assert forecast.iloc[0]["predicted"] == 0


def test_forecast_needs_three_months_and_ignores_transfers():
    _, forecast, _ = ins.forecast_next_month(frame(month_rows([100, 200])))
    assert forecast.empty
    rows = month_rows([100, 200, 300]) + [("2026-02-10", "Mom", 99999, "Transfers")]
    _, forecast, _ = ins.forecast_next_month(frame(rows))
    assert list(forecast["category"]) == ["Food"]


def test_forecast_drops_partial_last_month():
    rows = month_rows([100, 200, 300]) + [("2026-04-05", "Shop", 10, "Food")]    # statement ends on 5 April
    history, forecast, dropped = ins.forecast_next_month(frame(rows))
    assert dropped and len(history) == 3 and round(forecast.iloc[0]["predicted"]) == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all insights tests passed")
