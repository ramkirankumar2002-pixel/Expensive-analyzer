# test_categorizer.py - small tests for categorizer.py (plain asserts, no pytest needed).
# Run with:  python test_categorizer.py

import os

import pandas as pd

import categorizer as c


def test_clean_text():
    # reference number, UPI handle and the word UPI are removed; only useful words stay
    assert c.clean_text("UPI/428799513174/Swiggy/swiggy@ybl") == "swiggy"
    assert c.clean_text("UPI/123456789012/P2P/Rahul S/ra****@oksbi") == "p2p rahul s"
    # dashes instead of slashes: the merchant name must survive next to the UPI handle
    assert c.clean_text("UPI-SWIGGY-swiggy@icici-688498519526-UPI") == "swiggy"
    assert c.clean_text(None) == "none"  # never crashes on odd input


def test_rule_category():
    assert c.rule_category("UPI/1/Swiggy/swiggy@ybl") == "Food"
    assert c.rule_category("Airtel Broadband") == "Bills"
    assert c.rule_category("Rent - Landlord Mr Sharma") == "Rent"
    assert c.rule_category("UPI/1/Uber/uber@paytm") == "Travel"
    assert c.rule_category("UPI/123456789012/P2P/Mom/mo****@ybl") == "Transfers"  # digits inside "p2p" survive
    assert c.rule_category("UPI/1/1mg/1mg@ybl") == "Health"
    # whole words only: "current" contains "rent" but must not count as Rent
    assert c.rule_category("Current account fee") is None
    # unknown merchant -> no rule fires, so the ML layer gets a chance
    assert c.rule_category("UPI/1/Zxqv Traders/zx@ybl") is None


def test_model_file_and_classes():
    bundle = c.load_model()  # trains + saves automatically if models/categorizer.joblib is missing
    assert os.path.exists(c.MODEL_PATH)
    assert bundle["name"] in ("Logistic Regression", "Naive Bayes")
    # the model may only ever predict our fixed categories
    assert set(bundle["model"].classes_) <= set(c.CATEGORIES)


def test_ml_category():
    model = c.load_model()["model"]
    # a merchant the ML model knows from training (bypassing the rules on purpose)
    assert c.ml_category("UPI/1/Myntra/myntra@ybl", model) == "Shopping"
    # nothing readable left after cleaning -> Others
    assert c.ml_category("UPI/123456/", model) == "Others"
    # a threshold of 1.1 is impossible to reach, so everything becomes Others
    assert c.ml_category("UPI/1/Myntra/myntra@ybl", model, threshold=1.1) == "Others"


def test_categorize_text_layers():
    bundle = c.load_model()
    assert c.categorize_text("UPI/1/Zomato/zomato@ybl", bundle) == "Food"     # layer 1
    assert c.categorize_text("Chai Point order", bundle) == "Food"             # layer 1
    result = c.categorize_text("UPI/1/Qwxz Pvt Ltd/qw@ybl", bundle)           # unknown merchant
    assert result in c.CATEGORIES


def test_categorize_dataframe():
    df = pd.DataFrame({"description": ["UPI/1/Uber/uber@ybl", "NEFT to friend", "Netflix"],
                       "amount": [200, 500, 649]})
    out = c.categorize_dataframe(df)
    assert list(out["category"]) == ["Travel", "Transfers", "Entertainment"]
    assert "category" not in df.columns  # the original DataFrame is left untouched


def test_evaluate_unseen_merchants():
    df = pd.read_csv(c.DATA_PATH)
    X = df["description"].map(c.clean_text)
    results = c.evaluate_unseen_merchants(X, df["category"], df["description"])
    for ml_only, full in results.values():
        assert len(ml_only) == len(df)
        # rules + ML must do at least as well as ML alone on merchants it has never seen
        assert (full == df["category"]).mean() >= (ml_only == df["category"]).mean()


def test_messy_test_set():
    messy = pd.read_csv(c.REALISTIC_PATH)  # made by make_realistic_test.py
    assert set(messy["kind"]) == {"messy_known", "typo", "unseen_merchant", "unseen_person", "other"}
    assert set(messy["category"]) <= set(c.CATEGORIES)
    bundle = c.load_model()
    pred = [c.categorize_text(d, bundle) for d in messy["description"]]
    known = messy["kind"].isin(["messy_known", "typo"])
    # merchants from training in messy formats / with typos must be mostly right
    accuracy = (pd.Series(pred, index=messy.index)[known] == messy["category"][known]).mean()
    assert accuracy > 0.85, accuracy
    # a typo must still find its category through the character n-grams
    assert c.categorize_text("UPI-SWGGY-swggy@oksbi-325939859950-UPI", bundle) == "Food"


def test_confusion_matrix_png():
    assert os.path.getsize(c.MATRIX_PATH) > 1000  # created by training


if __name__ == "__main__":
    # run every function whose name starts with "test_"
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All categorizer tests passed.")
