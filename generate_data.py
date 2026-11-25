# generate_data.py - creates fake but realistic Indian bank/UPI transactions.
# Run with:  python generate_data.py
# Output:    data/transactions.csv  (columns: date, description, amount, category)

import os
import random
from datetime import date, timedelta

import pandas as pd

random.seed(42)  # fixed seed -> same data every run (easy to demo and test)

START = date(2026, 5, 1)   # 6 months: May 2026 ... Oct 2026
END = date(2026, 10, 31)
TARGET_ROWS = 1500

# ---------------------------------------------------------------------------
# 1. Everyday spending: category -> (merchants, min amount, max amount, weight)
#    "weight" = how often this category is picked relative to the others.
# ---------------------------------------------------------------------------
EVERYDAY = {
    "Food":          (["Swiggy", "Zomato", "Dominos", "McDonalds", "Starbucks", "Chai Point"], 80, 700, 30),
    "Groceries":     (["Blinkit", "BigBasket", "Zepto", "DMart", "Reliance Fresh"], 100, 2500, 22),
    "Travel":        (["Uber", "Ola", "Rapido", "IRCTC", "Metro Card Recharge"], 40, 600, 16),
    "Shopping":      (["Amazon", "Flipkart", "Myntra", "Ajio", "Decathlon"], 300, 4000, 10),
    "Entertainment": (["BookMyShow", "PVR Cinemas", "Spotify", "Steam Games"], 100, 900, 5),
    "Health":        (["Apollo Pharmacy", "PharmEasy", "1mg", "Cult Fit"], 100, 1800, 4),
    "Bills":         (["Airtel Recharge", "Tata Power", "Mahanagar Gas", "Jio Prepaid"], 150, 1500, 3),
}

# P2P transfers to friends/family (UPI IDs are masked, as the project rules say)
PEOPLE = ["Rahul S", "Priya M", "Amit K", "Neha G", "Mom", "Dad", "Sneha R"]
BANK_HANDLES = ["okhdfcbank", "oksbi", "ybl", "paytm", "okaxis"]

# ---------------------------------------------------------------------------
# 2. Fixed monthly payments: (merchant, day of month, amount, category)
# ---------------------------------------------------------------------------
MONTHLY = [
    ("Rent - Landlord Mr Sharma", 1, 18000, "Rent"),
    ("Netflix", 5, 649, "Entertainment"),
    ("Jio Postpaid", 7, 599, "Bills"),
    ("Airtel Broadband", 10, 999, "Bills"),
    ("BESCOM Electricity", 15, 1450, "Bills"),
]

# ---------------------------------------------------------------------------
# 3. A few big unusual spends (the anomalies we want to detect later)
# ---------------------------------------------------------------------------
UNUSUAL = [
    (date(2026, 6, 14), "Croma Electronics - Laptop", 78500, "Shopping"),
    (date(2026, 7, 22), "MakeMyTrip - Flight Tickets", 34200, "Travel"),
    (date(2026, 8, 9), "Fortis Hospital", 46800, "Health"),
    (date(2026, 9, 18), "Tanishq Jewellers", 62000, "Shopping"),
    (date(2026, 10, 5), "Apple Store - iPhone", 89900, "Shopping"),
    (date(2026, 10, 27), "Goa Resort Booking", 41500, "Travel"),
]


def mask_upi(name):
    """Hide most of a UPI ID, e.g. 'rahul s' -> 'ra****@ybl'."""
    handle = random.choice(BANK_HANDLES)
    clean = name.lower().replace(" ", "")
    return f"{clean[:2]}****@{handle}"


def make_description(merchant, category):
    """Build a UPI-style bank narration like 'UPI/418273650192/Swiggy/Order'."""
    ref = random.randint(10**11, 10**12 - 1)  # 12-digit UPI reference number
    if category == "Transfers":
        return f"UPI/{ref}/P2P/{merchant}/{mask_upi(merchant)}"
    handle = random.choice(["ybl", "okaxis", "paytm", "icici"])
    return f"UPI/{ref}/{merchant}/{merchant.lower().replace(' ', '')[:10]}@{handle}"


def random_date():
    """Pick a random day between START and END."""
    return START + timedelta(days=random.randint(0, (END - START).days))


rows = []  # each row = [date, description, amount, category]

# Fixed monthly payments (6 months)
for month in range(5, 11):
    for merchant, day, amount, category in MONTHLY:
        rows.append([date(2026, month, day), merchant, amount, category])

# Unusual big spends
for d, merchant, amount, category in UNUSUAL:
    rows.append([d, make_description(merchant, category), amount, category])

# P2P transfers (about 5% of rows): amounts are round-ish numbers
for _ in range(int(TARGET_ROWS * 0.05)):
    person = random.choice(PEOPLE)
    amount = random.choice([100, 200, 250, 500, 500, 1000, 1500, 2000, 3000])
    rows.append([random_date(), make_description(person, "Transfers"), amount, "Transfers"])

# Fill the rest with everyday spending until we reach the target row count
categories = list(EVERYDAY)
weights = [EVERYDAY[c][3] for c in categories]
while len(rows) < TARGET_ROWS:
    category = random.choices(categories, weights=weights)[0]
    merchants, low, high, _ = EVERYDAY[category]
    merchant = random.choice(merchants)
    amount = round(random.uniform(low, high), 2)
    rows.append([random_date(), make_description(merchant, category), amount, category])

# Put into a DataFrame, sort by date and save
df = pd.DataFrame(rows, columns=["date", "description", "amount", "category"])
df = df.sort_values("date").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print(f"Saved {len(df)} transactions to data/transactions.csv")
print(df.groupby("category")["amount"].agg(["count", "sum"]).round(0))
