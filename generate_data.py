# generate_data.py - creates fake but realistic Indian bank/UPI transactions.
# Run with:  python generate_data.py
# Output:    data/transactions.csv  (columns: date, description, amount, category)
#
# Two parts: (1) the clean UPI-style rows, (2) EXTRA rows that add variety so the ML model
# generalises: name variations, typos, other narration styles, short generic labels, more people.
# The extra rows use their own random generator, so part (1) is identical to before.
# Merchants in make_realistic_test.py (the messy TEST set) are NOT used here, so that test stays unseen.

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

# ---------------------------------------------------------------------------
# 4. EXTRA variety (added after the clean rows; own random generator -> part 1 unchanged)
# ---------------------------------------------------------------------------
rng = random.Random(2026)

# 4a. More merchant names and spelling variations per category (none of them is in the messy test set)
VARIED_MERCHANTS = {
    "Food": ["Swiggy Order", "Zomato Online", "Dominos Pizza", "Domino's", "McDonald's", "Mc Donalds",
             "Starbucks Coffee", "Chai Point Cafe", "Pizza Hut", "Chaayos", "Cafe Coffee Day", "Wow Momo",
             "Biryani By Kilo", "Ovenstory Pizza", "Theobroma", "Baskin Robbins", "KFC", "Eatfit"],
    "Groceries": ["Blinkit Instant", "BigBasket.com", "Big Basket", "Zepto Now", "DMart Ready", "D Mart",
                  "Reliance Fresh Store", "JioMart", "Milkbasket", "Star Bazaar", "Ratnadeep Supermarket",
                  "Easyday", "Vegetable Market", "Local Kirana"],
    "Travel": ["Uber India", "Uber Trip", "Ola Cabs", "Rapido Bike Taxi", "IRCTC Tickets", "RedBus", "MakeMyTrip",
               "Air India", "SpiceJet", "Meru Cabs", "Metro Rail Card", "HP Petrol Pump", "Indian Oil Fuel",
               "Shell Petrol"],
    "Shopping": ["Amazon.in", "Amazon Seller", "Flipkart.com", "Myntra.com", "Ajio Fashion", "Decathlon Sports",
                 "Tata Cliq", "Snapdeal", "Shoppers Stop", "Westside", "Max Fashion", "Croma Store",
                 "Vijay Sales", "Puma Store"],
    "Entertainment": ["BookMyShow Tickets", "PVR INOX", "INOX Movies", "Spotify Premium", "Netflix India",
                      "Hotstar Subscription", "YouTube Premium", "Audible", "Xbox Live", "PlayStation Store"],
    "Health": ["Apollo Pharmacy Store", "Apollo 24x7", "PharmEasy Order", "Cult.fit", "Tata 1mg", "Fortis Clinic",
               "Max Hospital", "Wellness Forever", "Apollo Diagnostics", "City Dental Clinic"],
    "Bills": ["Airtel Prepaid", "Airtel Postpaid", "Tata Power Delhi", "Jio Recharge", "Jio Fiber",
              "Mahanagar Gas Ltd", "Vodafone Recharge", "Hathway Broadband", "Torrent Power", "Bharat Gas",
              "BWSSB Water Bill", "Electricity Board"],
    "Rent": ["Monthly Rent", "Room Rent", "Rent Payment", "Landlord Mr Gupta", "Rent - Landlord Mrs Rao"],
}
AMOUNTS = {"Food": (60, 800), "Groceries": (80, 3000), "Travel": (30, 2500), "Shopping": (200, 5000),
           "Entertainment": (100, 1200), "Health": (80, 2500), "Bills": (150, 2500), "Rent": (5000, 25000),
           "Transfers": (100, 5000), "Others": (20, 3000)}

CITIES = ["Bangalore", "Mumbai", "Delhi", "Pune", "Hyderabad", "Chennai", "Kolkata"]


def typo(name):
    """Misspell one letter (drop / swap / double / wrong vowel). Names under 5 letters are kept."""
    if len(name) < 5:
        return name
    i = rng.randint(1, len(name) - 2)
    kind = rng.choice(["drop", "swap", "double", "vowel"])
    if kind == "drop":
        return name[:i] + name[i + 1:]
    if kind == "swap":
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    if kind == "double":
        return name[:i] + name[i] + name[i:]
    return name[:i] + rng.choice("aeiou") + name[i + 1:]


def varied_description(name):
    """Wrap a name in a narration style (these differ from the styles used in the messy TEST set)."""
    ref = rng.randint(10**9, 10**12)
    handle = rng.choice(["ybl", "okaxis", "paytm", "icici", "oksbi"])
    style = rng.choice(["plain", "upper", "lower", "paid", "payment", "city", "drupi", "purchase", "bill", "billpay"])
    return {
        "plain": name, "upper": name.upper(), "lower": name.lower(),
        "paid": f"Paid to {name}",
        "payment": f"{name} Payment",
        "city": f"{name} - {rng.choice(CITIES)}",
        "drupi": f"UPI/DR/{ref}/{name}/{handle}",
        "purchase": f"Purchase at {name}",
        "bill": f"{name} Bill {rng.randint(1000, 9999)}",
        "billpay": f"IB BILLPAY {name}",
    }[style]


def extra_row(description, category):
    low, high = AMOUNTS[category]
    return [random_date(), description, round(rng.uniform(low, high), 2), category]


# Name variations and typos (about 1 in 4 names is misspelled)
for category, names in VARIED_MERCHANTS.items():
    for name in names:
        for _ in range(rng.randint(4, 7)):
            shown = typo(name) if rng.random() < 0.25 else name
            rows.append(extra_row(varied_description(shown), category))

# 4b. Short generic labels, like the ones people type in their own expense sheets
GENERIC = {
    "Food": ["Eating Out", "Dining Out", "Snacks", "Lunch", "Dinner", "Breakfast", "Tea Coffee", "Takeaway",
             "Street Food", "Canteen", "Meals"],
    "Groceries": ["Vegetables", "Fruits", "Kirana Store", "Milk", "Provisions", "Daily Needs", "Weekly Groceries"],
    "Travel": ["Transportation", "Transport", "Cab Fare", "Auto Rickshaw", "Bus Ticket", "Train Ticket", "Fuel",
               "Parking", "Toll", "Commute", "Taxi"],
    "Shopping": ["Clothes", "Shoes", "Gadgets", "Electronics", "Gifts", "Accessories", "Online Shopping",
                 "Household Items", "Furniture"],
    "Entertainment": ["Movies", "Games", "Streaming", "Concert", "Hobbies", "Party", "Outing"],
    "Health": ["Medicines", "Doctor Visit", "Medical Checkup", "Dental", "Gym", "Lab Tests", "Vitamins"],
    "Bills": ["Electricity Bill", "Water Bill", "Internet Bill", "Mobile Recharge", "Utilities", "DTH Recharge",
              "Credit Card Bill", "Loan EMI", "Maintenance"],
    "Rent": ["Rent", "House Rent", "Flat Rent", "Rent Deposit"],
    "Transfers": ["Sent to Friend", "Money Transfer", "Family Support", "Loan to Friend", "Pocket Money"],
    "Others": ["Others", "Other", "Miscellaneous", "Misc Expense", "Unknown", "Other Expenses", "Donation", "Cash Withdrawal",
               "Service Charge", "Cheque Clearing", "Interest Paid"],
}
for category, labels in GENERIC.items():
    for label in labels:
        for _ in range(rng.randint(3, 5)):
            text = rng.choice([label, label, label.lower(), label.upper()])   # mostly as written
            rows.append(extra_row(text, category))

# 4c. Transfers to more people, in formats that do NOT contain the word P2P
MORE_PEOPLE = ["Rohan", "Pooja Sharma", "Vikram", "Kavya Nair", "Arjun Reddy", "Divya", "Manoj Patel", "Isha"]
for person in MORE_PEOPLE:
    for _ in range(8):
        ref = rng.randint(10**9, 10**12)
        handle = rng.choice(["ybl", "okaxis", "paytm", "oksbi"])
        text = rng.choice([f"Sent to {person}", f"Transfer to {person}",
                           f"UPI/{ref}/{person}/{person.lower().replace(' ', '')[:4]}****@{handle}",
                           f"{person} - UPI"])
        rows.append(extra_row(text, "Transfers"))

# Put into a DataFrame, sort by date and save
df = pd.DataFrame(rows, columns=["date", "description", "amount", "category"])
df = df.sort_values("date").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/transactions.csv", index=False)

print(f"Saved {len(df)} transactions to data/transactions.csv")
print(df.groupby("category")["amount"].agg(["count", "sum"]).round(0))
