# make_realistic_test.py - builds a deliberately MESSY test set: data/test_realistic.csv
# Run with:  python make_realistic_test.py
#
# The training data (generate_data.py) is very clean, so 100% accuracy on it means little.
# This file makes harder rows the model is NEVER trained on. Each row has a "kind":
#   messy_known     - merchants from the training data, but written like real bank narrations
#   typo            - merchants from the training data with spelling mistakes
#   unseen_merchant - merchants that do not exist in the training data at all
#   unseen_person   - money sent to people WITHOUT the "P2P" word the training data always has
#   other           - things that fit no category (ATM, bank charges...) -> correct answer is "Others"

import os
import random

import pandas as pd

random.seed(7)  # fixed seed -> the same test set every time

# merchants that DO appear in the training data (name, category)
KNOWN = [
    ("Swiggy", "Food"), ("Zomato", "Food"), ("Dominos", "Food"), ("McDonalds", "Food"),
    ("Starbucks", "Food"), ("Chai Point", "Food"),
    ("Blinkit", "Groceries"), ("BigBasket", "Groceries"), ("Zepto", "Groceries"),
    ("DMart", "Groceries"), ("Reliance Fresh", "Groceries"),
    ("Uber", "Travel"), ("Ola", "Travel"), ("Rapido", "Travel"), ("IRCTC", "Travel"),
    ("Amazon", "Shopping"), ("Flipkart", "Shopping"), ("Myntra", "Shopping"),
    ("Ajio", "Shopping"), ("Decathlon", "Shopping"),
    ("BookMyShow", "Entertainment"), ("PVR Cinemas", "Entertainment"),
    ("Spotify", "Entertainment"), ("Netflix", "Entertainment"),
    ("Apollo Pharmacy", "Health"), ("PharmEasy", "Health"), ("Cult Fit", "Health"),
    ("Airtel Recharge", "Bills"), ("Tata Power", "Bills"), ("Jio Prepaid", "Bills"),
    ("Mahanagar Gas", "Bills"),
]

# merchants that do NOT appear in the training data
UNSEEN = {
    "Food": ["Behrouz Biryani", "Faasos", "Subway", "Burger King", "Haldirams",
             "Barbeque Nation", "Sagar Ratna", "Third Wave Coffee"],
    "Groceries": ["Licious", "Big Bazaar", "More Retail", "Spencers", "Natures Basket",
                  "Country Delight", "Dunzo Daily"],
    "Travel": ["Cleartrip", "Goibibo", "Namma Yatri", "BluSmart", "Indigo Airlines",
               "Delhi Metro", "FASTag Recharge", "Yatra Online"],
    "Shopping": ["Lenskart", "Zudio", "Nykaa", "Meesho", "Pantaloons",
                 "Reliance Digital", "Ikea", "Lifestyle Stores"],
    "Bills": ["BSNL Landline", "Tata Play", "Vi Postpaid", "Adani Electricity",
              "Indane Gas", "Municipal Water Board", "ACT Fibernet", "LIC Premium"],
    "Rent": ["House Rent Aug", "Rent to Owner", "PG Rent Sunrise Hostel", "Flat Rent Transfer"],
    "Entertainment": ["Prime Video", "Disney Plus", "Zee5", "Sony Liv", "Gaana",
                      "Smaaash", "Wonderla"],
    "Health": ["Practo", "Netmeds", "Lal PathLabs", "Dr Agarwal Eye Clinic", "Medplus",
               "Healthians", "Manipal Hospital"],
}
PEOPLE = ["Ravi Kumar", "Anita Singh", "Suresh", "Karthik", "Meena Iyer", "Deepak Jain"]
OTHERS = ["ATM WDL Cash", "Bank Charges GST", "Interest Credit", "Cheque Deposit",
          "Zxq Traders", "Mystery Merchant 77", "Annual Card Fee"]

CITIES = ["BANGALORE", "MUMBAI", "DELHI", "PUNE", "HYDERABAD", "CHENNAI"]
HANDLES = ["ybl", "okaxis", "paytm", "icici", "oksbi", "okhdfcbank"]


def make_typo(name):
    """Spell the name wrongly in one of 4 common ways (never touches the first letter)."""
    if len(name) < 5:
        return name
    i = random.randint(1, len(name) - 2)
    kind = random.choice(["drop", "swap", "double", "vowel"])
    if kind == "drop":      # swiggy -> swigy
        return name[:i] + name[i + 1:]
    if kind == "swap":      # zomato -> zomaot
        return name[:i] + name[i + 1] + name[i] + name[i + 2:]
    if kind == "double":    # amazon -> amazzon
        return name[:i] + name[i] + name[i:]
    return name[:i] + random.choice("aeiou") + name[i + 1:]  # flipkart -> flipkort


def messy_case(name):
    """Random messy capitalisation: UPPER, lower or rAnDoM."""
    style = random.choice(["upper", "lower", "random"])
    if style == "upper":
        return name.upper()
    if style == "lower":
        return name.lower()
    return "".join(ch.upper() if random.random() < 0.5 else ch.lower() for ch in name)


def messy_description(name):
    """Wrap a merchant name in one of several real-looking bank narration styles."""
    ref = random.randint(10**9, 10**12)
    card = f"{random.randint(4000, 5999)}XXXXXX{random.randint(1000, 9999)}"
    handle = random.choice(HANDLES)
    slug = name.lower().replace(" ", "")[:10]
    style = random.choice(range(7))
    n = messy_case(name)
    if style == 0:
        return f"POS {card} {n} {random.choice(CITIES)}"
    if style == 1:
        return f"UPI-{n}-{slug}@{handle}-{ref}-UPI"
    if style == 2:
        return f"{n} ORDER #{random.randint(1000, 99999)}"
    if style == 3:
        return f"PAYMENT TO  {n}   REF{ref}"
    if style == 4:
        return f"ECOM PUR/{n}/{random.choice(CITIES)} IN"
    if style == 5:
        return f"  {n}  "
    return f"UPI/{ref}/{n}/{slug}@{handle}/Payment from PhonePe"


def person_description(person):
    """A transfer to a person, in formats that do NOT contain the word P2P."""
    ref = random.randint(10**9, 10**12)
    handle = random.choice(HANDLES)
    slug = person.lower().replace(" ", "")[:5]
    return random.choice([
        f"UPI-{person.upper()}-{slug}@{handle}-{ref}",
        f"IMPS/{ref}/{person}",
        f"NEFT-{person.upper()}-{ref}",
        f"To {person} - UPI/{ref}",
    ])


rows = []  # [description, category, kind]

# 1. known merchants in messy formats
for _ in range(120):
    name, cat = random.choice(KNOWN)
    rows.append([messy_description(name), cat, "messy_known"])

# 2. known merchants with typos (also messy format and case)
for _ in range(100):
    name, cat = random.choice(KNOWN)
    rows.append([messy_description(make_typo(name)), cat, "typo"])

# 3. merchants never seen in training (3 different narrations each)
for cat, names in UNSEEN.items():
    for name in names:
        for _ in range(3):
            rows.append([messy_description(name), cat, "unseen_merchant"])

# 4. transfers to people without "P2P"
for person in PEOPLE:
    for _ in range(4):
        rows.append([person_description(person), "Transfers", "unseen_person"])

# 5. rows that belong to no category
for name in OTHERS:
    for _ in range(3):
        rows.append([messy_description(name), "Others", "other"])

df = pd.DataFrame(rows, columns=["description", "category", "kind"]).sample(frac=1, random_state=7)
os.makedirs("data", exist_ok=True)
df.to_csv("data/test_realistic.csv", index=False)
print(f"Saved {len(df)} messy test rows to data/test_realistic.csv")
print(df["kind"].value_counts().to_string())
print("\nExamples:")
print(df.groupby("kind").head(2).sort_values("kind").to_string(index=False))
