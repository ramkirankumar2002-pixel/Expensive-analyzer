# test_goals.py - small tests for goals.py (plain asserts).
# Run with:  python test_goals.py

import datetime as dt
import json
import os
import tempfile

import pandas as pd

import goals as gl

TODAY = dt.date(2026, 1, 15)     # a fixed "today", so the tests give the same result every day


def table():
    """Feb-May 2026. Each full month: salary 50,000, rent 20,000, food 6,000, shopping 3,000, transfer 5,000.
    So savings = 50,000 - 29,000 = 21,000 a month (the transfer is not spending).
    February is older than the last 3 months and has a huge spend that must be ignored."""
    rows = [("2026-02-10", 40000, "Shopping", "debit")]
    for month in ["02", "03", "04", "05"]:
        rows += [(f"2026-{month}-01", 50000, "Income", "credit"),
                 (f"2026-{month}-02", 20000, "Rent", "debit"),
                 (f"2026-{month}-05", 6000, "Food", "debit"),
                 (f"2026-{month}-09", 3000, "Shopping", "debit"),
                 (f"2026-{month}-30" if month != "02" else "2026-02-28", 5000, "Transfers", "debit")]
    df = pd.DataFrame(rows, columns=["date", "amount", "category", "type"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def goal(target=70000, date="2026-11-15", saved=10000):
    return gl.make_goal("Laptop", target, date, saved)          # 10 months after TODAY


def test_recent_savings_uses_last_3_months():
    r = gl.recent_savings(table())
    assert [str(m) for m in r["months"]] == ["2026-03", "2026-04", "2026-05"]
    assert r["savings"] == 21000
    assert r["spend_by_category"] == {"Rent": 20000, "Food": 6000, "Shopping": 3000}


def test_recent_savings_skips_a_half_month():
    june = pd.DataFrame([(pd.Timestamp("2026-06-01"), 50000, "Income", "credit")],
                        columns=["date", "amount", "category", "type"])
    r = gl.recent_savings(pd.concat([table(), june], ignore_index=True))
    assert str(r["months"][-1]) == "2026-05" and r["savings"] == 21000
    only = gl.recent_savings(june)                              # the only month there is: use it
    assert len(only["months"]) == 1 and only["savings"] == 50000
    assert gl.recent_savings(june.iloc[:0])["savings"] == 0     # no rows at all


def test_months_between():
    assert gl.months_between(TODAY, "2026-11-15") == 10
    assert gl.months_between(TODAY, "2026-11-14") == 9          # last month not complete
    assert gl.months_between(TODAY, "2026-01-20") == 1          # less than a month left
    assert gl.months_between(TODAY, "2026-01-15") == 0          # today
    assert gl.months_between(TODAY, "2025-12-01") == 0          # past


def test_on_track():
    r = gl.evaluate(goal(), savings=8000, today=TODAY)
    assert r["months_left"] == 10 and r["needed"] == 6000 and r["status"] == gl.ON_TRACK
    assert r["gap"] == 0 and round(r["progress"], 3) == 0.143
    assert r["reach_date"] == dt.date(2026, 9, 15)              # 60,000 / 8,000 = 7.5 -> 8 months


def test_slightly_behind_and_boundaries():
    assert gl.evaluate(goal(), 5000, TODAY)["status"] == gl.SLIGHTLY_BEHIND
    assert gl.evaluate(goal(), 6000, TODAY)["status"] == gl.ON_TRACK           # exactly enough
    assert gl.evaluate(goal(), 4800, TODAY)["status"] == gl.SLIGHTLY_BEHIND    # exactly 80%
    assert gl.evaluate(goal(), 4799, TODAY)["status"] == gl.OFF_TRACK


def test_off_track():
    r = gl.evaluate(goal(), savings=3000, today=TODAY)
    assert r["status"] == gl.OFF_TRACK and r["gap"] == 3000
    assert r["reach_date"] == dt.date(2027, 9, 15)              # 20 months


def test_goal_date_in_the_past():
    r = gl.evaluate(goal(date="2025-12-01"), savings=100000, today=TODAY)
    assert r["months_left"] == 0 and r["date_passed"] and r["status"] == gl.OFF_TRACK
    assert r["needed"] == 60000                                 # no division by zero
    assert r["reach_date"] == dt.date(2026, 2, 15)


def test_zero_and_negative_savings():
    for savings in [0.0, -2000.0]:
        r = gl.evaluate(goal(), savings, TODAY)
        assert r["status"] == gl.OFF_TRACK and r["reach_date"] is None
        assert r["gap"] == 6000 - savings                       # overspending makes the gap bigger
    assert gl.evaluate(goal(saved=0), 0.0, TODAY)["progress"] == 0      # nothing saved yet either
    assert gl.reach_date(10**9, 0.01, TODAY) is None            # absurdly far away: no date, no crash


def test_goal_already_reached():
    r = gl.evaluate(goal(saved=80000, date="2025-01-01"), savings=0, today=TODAY)
    assert r["reached"] and r["status"] == gl.ON_TRACK and not r["date_passed"]
    assert r["progress"] == 1 and r["needed"] == 0 and r["reach_date"] == TODAY


def test_suggest_cuts():
    spend = {"Rent": 20000, "Bills": 5000, "Groceries": 4000, "Shopping": 3200, "Food": 4800}
    cuts, uncovered = gl.suggest_cuts(2000, spend)
    assert [c["category"] for c in cuts] == ["Food", "Shopping"]        # ranked; no Rent / Bills / Groceries
    assert cuts[0] == {"category": "Food", "cut": 1200, "current": 4800, "new": 3600}
    assert cuts[1]["cut"] == 800 and uncovered == 0
    assert gl.cut_text(cuts[0]) == "Cut Food by ₹1,200/month (from ₹4,800 to ₹3,600)"


def test_suggest_cuts_is_capped():
    cuts, uncovered = gl.suggest_cuts(10000, {"Food": 4800, "Travel": 1000, "Entertainment": 0})
    assert [(c["category"], c["cut"]) for c in cuts] == [("Food", 2400), ("Travel", 500)]   # half each
    assert uncovered == 7100
    assert gl.suggest_cuts(500, {"Rent": 20000}) == ([], 500.0)         # nothing that can be cut
    assert gl.suggest_cuts(0, {"Food": 4800}) == ([], 0.0)              # no gap


def test_cuts_always_close_the_gap_when_possible():
    cuts, uncovered = gl.suggest_cuts(1000, {"Food": 3333.33, "Travel": 1111.11, "Shopping": 777.77})
    assert sum(c["cut"] for c in cuts) >= 1000 and uncovered == 0
    assert all(c["current"] - c["cut"] == c["new"] >= 0 for c in cuts)


def test_make_goal_rejects_bad_values():
    for bad in [("  ", 100, "2027-01-01", 0), ("x", 0, "2027-01-01", 0), ("x", 100, "2027-01-01", -1)]:
        try:
            gl.make_goal(*bad)
            assert False, f"should have raised for {bad}"
        except ValueError:
            pass


def test_save_and_load():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "goals.json")
        assert gl.load_goals(path) == []                        # no file yet
        two = [goal(), gl.make_goal("Trip", 20000, dt.date(2027, 3, 1))]
        gl.save_goals(two, path)
        assert gl.load_goals(path) == two and two[0]["id"] != two[1]["id"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump([two[0], {"name": "broken"}, 5], f)       # bad entries are skipped
        assert gl.load_goals(path) == [two[0]]
        with open(path, "w", encoding="utf-8") as f:
            f.write("not json")
        assert gl.load_goals(path) == []


def test_report_stats_have_no_names():
    stats = gl.report_stats([goal()], table(), TODAY)
    assert stats == [{"target": 70000, "saved": 10000, "months_left": 10, "needed_per_month": 6000,
                      "average_monthly_savings": 21000, "status": "On track"}]
    assert "Laptop" not in json.dumps(stats)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all goals tests passed")
