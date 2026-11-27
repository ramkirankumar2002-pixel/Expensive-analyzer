# generate_data.py - creates fake but realistic Indian bank/UPI transactions.
# Run with:  python generate_data.py
# Output:    data/transactions.csv (demo statement) and data/training.csv (ML training data)
#            columns: date, description, amount, type, category
#            type = "debit" (money out) or "credit" (monthly salary, a few refunds)
#            credits are labelled "Income" / "Refund" (they never go through the ML model)
#
# Two parts: (1) the clean UPI-style rows, (2) EXTRA rows that add variety so the ML model
# generalises: name variations, typos, other narration styles, short generic labels, more people.
# The extra rows use their own random generator, so part (1) is identical to before.
# Merchants in make_realistic_test.py (the messy TEST set) are NOT used here, so that test stays unseen.

import os
import random
from datetime import date, timedelta

import pandas as pd

from merchants import NEW_MERCHANTS

START = date(2023, 11, 1)  # 36 months: Nov 2023 ... Oct 2026
END = date(2026, 10, 31)

# Every (year, month) in the range, used for the monthly salary and fixed payments
MONTHS = [(START.year + (START.month - 1 + i) // 12, (START.month - 1 + i) % 12 + 1) for i in range(36)]

# ---------------------------------------------------------------------------
# 1. Everyday spending: category -> (merchants, min amount, max amount, weight)
#    "weight" = how often this category is picked relative to the others.
# ---------------------------------------------------------------------------
EVERYDAY = {
    "Food":          (["Swiggy", "Zomato", "Dominos", "McDonalds", "Starbucks", "Chai Point"], 150, 600, 30),
    "Groceries":     (["Blinkit", "BigBasket", "Zepto", "DMart", "Reliance Fresh"], 300, 2500, 12),
    "Travel":        (["Uber", "Ola", "Rapido", "IRCTC", "Metro Card Recharge"], 50, 800, 18),
    "Shopping":      (["Amazon", "Flipkart", "Myntra", "Ajio", "Decathlon"], 500, 4000, 6),
    "Entertainment": (["BookMyShow", "PVR Cinemas", "Steam Games"], 100, 900, 5),
    "Health":        (["Apollo Pharmacy", "PharmEasy", "1mg", "Cult Fit"], 100, 1800, 4),
    "Bills":         (["Airtel Recharge", "Tata Power", "Mahanagar Gas", "Jio Prepaid"], 300, 2000, 3),
}

# P2P transfers to friends/family (UPI IDs are masked, as the project rules say)
PEOPLE = ["Rahul S", "Priya M", "Amit K", "Neha G", "Mom", "Dad", "Sneha R"]
BANK_HANDLES = ["okhdfcbank", "oksbi", "ybl", "paytm", "okaxis"]

# ---------------------------------------------------------------------------
# 2. Fixed monthly payments: (merchant, day of month, amount, category)
#    (rent is added in part 4d, with a different bank narration style each month)
# ---------------------------------------------------------------------------
RENT = 12000
MONTHLY = [
    ("Netflix", 5, 649, "Entertainment"),
    ("Jio Postpaid", 7, 599, "Bills"),
    ("Airtel Broadband", 10, 999, "Bills"),
    ("BESCOM Electricity", 15, 1450, "Bills"),
    ("Spotify Premium", 12, 119, "Entertainment"),   # fixed amount, same day every month
    ("Gym Membership", 20, 1200, "Health"),
]

# ---------------------------------------------------------------------------
# 3. A few big unusual spends (the anomalies we want to detect later)
#    Sized to stand out from everyday spending while fitting a Rs 50,000-60,000 monthly salary.
# ---------------------------------------------------------------------------
UNUSUAL = [
    (date(2026, 6, 14), "Croma Electronics - Laptop", 42000, "Shopping"),
    (date(2026, 7, 22), "MakeMyTrip - Flight Tickets", 15000, "Travel"),
    (date(2026, 8, 9), "Fortis Hospital", 12000, "Health"),
    (date(2026, 9, 18), "Tanishq Jewellers", 18000, "Shopping"),
    (date(2026, 10, 5), "Apple Store - iPhone", 48000, "Shopping"),
    (date(2026, 10, 27), "Goa Resort Booking", 20000, "Travel"),
]
# The dates above are the ones in training.csv (the ML model only reads the text, not the date).
# In the demo statement the same spends happen earlier, at most one a month, so that its last
# 3 months (Aug-Oct 2026) show normal savings and the Goals page has something sensible to work with.
DEMO_UNUSUAL_DATES = {
    "Fortis Hospital": date(2025, 10, 9),
    "Tanishq Jewellers": date(2025, 11, 18),
    "MakeMyTrip - Flight Tickets": date(2025, 12, 22),
    "Goa Resort Booking": date(2026, 1, 27),
    "Croma Electronics - Laptop": date(2026, 2, 14),
    "Apple Store - iPhone": date(2026, 5, 5),
}

# Demo statement only: some ordinary extra purchases in the last 3 months (festival season),
# so savings there are about Rs 12,000 a month. Each amount is normal for its category.
DEMO_RECENT = [
    (date(2026, 8, 6), "Myntra", 2349.0, "Shopping"),
    (date(2026, 8, 12), "BigBasket", 1865.0, "Groceries"),
    (date(2026, 8, 21), "Amazon", 2899.0, "Shopping"),
    (date(2026, 8, 28), "Swiggy", 486.0, "Food"),
    (date(2026, 9, 5), "Flipkart", 3499.0, "Shopping"),
    (date(2026, 9, 13), "DMart", 2140.0, "Groceries"),
    (date(2026, 9, 19), "Apollo Pharmacy", 1285.0, "Health"),
    (date(2026, 9, 26), "Ajio", 1799.0, "Shopping"),
    (date(2026, 10, 8), "Amazon", 3649.0, "Shopping"),
    (date(2026, 10, 17), "Reliance Fresh", 1920.0, "Groceries"),
    (date(2026, 10, 24), "Myntra", 1299.0, "Shopping"),
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


def build_dataset(target_rows, extra_merchants=False):
    """Build one fake statement (DataFrame) with about `target_rows` rows of everyday spending
    plus the fixed extras. Same seeds every time, so the same target gives the same data.
    extra_merchants=True builds the training file; False builds the demo statement."""
    demo = not extra_merchants
    random.seed(42)   # fixed seed -> same data every run (easy to demo and test)

    rows = []  # each row = [date, description, amount, category]

    # Fixed monthly payments (every month)
    for year, month in MONTHS:
        for merchant, day, amount, category in MONTHLY:
            rows.append([date(year, month, day), merchant, amount, category])

    # Unusual big spends
    for d, merchant, amount, category in UNUSUAL:
        if demo:
            d = DEMO_UNUSUAL_DATES[merchant]
        rows.append([d, make_description(merchant, category), amount, category])

    # P2P transfers (about 5% of rows): amounts are round-ish numbers
    for _ in range(int(target_rows * 0.05)):
        person = random.choice(PEOPLE)
        amount = random.choice([100, 200, 250, 500, 500, 1000, 1500, 2000, 3000])
        rows.append([random_date(), make_description(person, "Transfers"), amount, "Transfers"])

    # Fill the rest with everyday spending until we reach the target row count
    categories = list(EVERYDAY)
    weights = [EVERYDAY[c][3] for c in categories]
    while len(rows) < target_rows:
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
    AMOUNTS = {"Food": (150, 600), "Groceries": (300, 2500), "Travel": (50, 800), "Shopping": (500, 4000),
               "Entertainment": (100, 900), "Health": (100, 1800), "Bills": (300, 2000),
               "Transfers": (500, 5000), "Others": (50, 1500), "Rent": (10000, 15000)}

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
        if category == "Rent":
            continue   # rent is added once a month in part 4d
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
        if category == "Rent":
            continue
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

    # 4d. Monthly rent: always Rs 12,000, but the bank narration changes (so the model sees many styles)
    RENT_NAMES = ["Rent - Landlord Mr Sharma"] + VARIED_MERCHANTS["Rent"] + GENERIC["Rent"]
    for year, month in MONTHS:
        name = rng.choice(RENT_NAMES)
        rows.append([date(year, month, 1), rng.choice([name, varied_description(name)]), float(RENT), "Rent"])

    # 4e. More well-known merchants (merchants.py), only for the ML training file: they would
    #     make the demo statement spend more than it earns. Same narration styles and typos as 4a.
    if extra_merchants:
        for category, names in NEW_MERCHANTS.items():
            for name in names:
                for _ in range(rng.randint(3, 5)):
                    shown = typo(name) if rng.random() < 0.25 else name
                    rows.append(extra_row(varied_description(shown), category))

    # 4f. Demo statement only: the extra recent purchases (last, so the rows above stay the same)
    if demo:
        for d, merchant, amount, category in DEMO_RECENT:
            rows.append([d, make_description(merchant, category), amount, category])

    # ---------------------------------------------------------------------------
    # 5. Income: a monthly salary and a few refunds (credits).
    #    Credits are labelled "Income" / "Refund" here; the app does the same without using the ML model.
    # ---------------------------------------------------------------------------
    income_rng = random.Random(7)   # own generator, so adding income does not change the rows above

    salary_rows = []   # salary lands on the 1st, between Rs 50,000 and Rs 60,000 (variable pay)
    for year, month in MONTHS:
        amount = income_rng.randrange(50000, 60001, 500)
        salary_rows.append([date(year, month, 1), f"Salary - Acme Technologies {date(year, month, 1):%b %Y}",
                            float(amount), "Income"])

    REFUND_FROM = ["Amazon", "Myntra", "Flipkart", "Ajio", "Zomato", "Swiggy", "Uber", "BookMyShow"]
    refund_rows = []
    for merchant in REFUND_FROM:
        day = START + timedelta(days=income_rng.randint(0, (END - START).days))
        refund_rows.append([day, f"Refund - {merchant}", round(income_rng.uniform(150, 1800), 2), "Refund"])

    # Put into a DataFrame, sort by date and save
    df = pd.DataFrame(rows, columns=["date", "description", "amount", "category"])
    df["type"] = "debit"
    credits = pd.DataFrame(salary_rows + refund_rows, columns=["date", "description", "amount", "category"])
    credits["type"] = "credit"
    df = pd.concat([df, credits], ignore_index=True)[["date", "description", "amount", "type", "category"]]
    df = df.sort_values("date", kind="stable").reset_index(drop=True)
    return df


# Two files from the same generator:
#   transactions.csv = the demo statement you upload to the app: realistic amounts, ~22% savings
#   training.csv     = what the ML model learns from: many more rows (amounts do not matter to the
#                      model, only the description text and the category do)
if __name__ == "__main__":   # importing this file does not overwrite any data
    for path, target, extra in [("data/transactions.csv", 200, False), ("data/training.csv", 1500, True)]:
        df = build_dataset(target, extra_merchants=extra)
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} rows to {path}")
        if "transactions" in path:   # savings only means something for the demo statement
            spent = df[(df["type"] == "debit") & (df["category"] != "Transfers")]["amount"].sum()
            earned = df[df["type"] == "credit"]["amount"].sum()
            print(f"  income {earned:,.0f}  spent {spent:,.0f}  savings {(earned - spent) / earned:.1%}")
