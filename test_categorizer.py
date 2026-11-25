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
    assert c.rule_category("UPI/123456789012/Zxqv Traders/zx@ybl") is None


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
    df = pd.DataFrame({"description": ["UPI/123456789012/Uber/uber@ybl", "NEFT to friend", "Netflix"],
                       "amount": [200, 500, 649]})
    out = c.categorize_dataframe(df, corrections_path=None)  # ignore any real corrections.csv
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


def test_save_and_load_corrections():
    path = _tmp("corr_roundtrip.csv")
    assert c.load_corrections(path) == {}  # no file yet -> empty
    c.save_corrections([("UPI/123456789012/Zxqv Traders/zx@ybl", "Shopping")], path)
    assert c.load_corrections(path) == {"zxqv traders": "Shopping"}
    # correcting the same description again keeps only the newest answer (one row in the file)
    c.save_corrections([("UPI/210987654321/Zxqv Traders/zx@ybl", "Food")], path)
    assert c.load_corrections(path) == {"zxqv traders": "Food"}
    assert len(pd.read_csv(path)) == 1
    # only the fixed categories are allowed
    try:
        c.save_corrections([("x", "Holidays")], path)
        assert False, "should have raised"
    except ValueError:
        pass


def test_correction_beats_rule():
    bundle = c.load_model()
    assert c.categorize_text("UPI/123456789012/Uber/uber@ybl", bundle) == "Travel"              # the rule says Travel
    fixes = {c.clean_text("UPI/123456789012/Uber/uber@ybl"): "Shopping"}
    assert c.categorize_text("UPI/999999999999/Uber/uber@paytm", bundle, fixes) == "Shopping"  # the correction wins


def test_training_includes_corrections():
    path = _tmp("corr_training.csv")
    c.save_corrections([("UPI/123456789012/Zxqv Traders/zx@ybl", "Shopping"),
                        ("UPI/123456789012/Qwxz Pvt Ltd/qw@ybl", "Health")], path)
    X_plain, _ = c.load_training_data()
    X_with, y_with = c.load_training_data(corrections_path=path)
    assert len(X_with) == len(X_plain) + 2
    assert list(y_with.tail(2)) == ["Shopping", "Health"]


def test_retrain_with_corrections():
    path, model_path = _tmp("corr_retrain.csv"), _tmp("model_retrain.joblib")
    # one lone row in a category the data has not used before ("Others") must not break training
    c.save_corrections([("UPI/123456789012/Zxqv Traders/zx@ybl", "Shopping"),
                        ("UPI/123456789012/Qwxz Pvt Ltd/qw@ybl", "Others")], path)
    bundle = c.train_and_save(corrections_path=path, model_path=model_path)
    assert os.path.exists(model_path)
    assert bundle["n_corrections"] == 2
    assert 0 <= bundle["clean_accuracy"] <= 1
    scores = c.messy_scores(bundle)
    assert 0 <= scores["ml_only"] <= 1 and 0 <= scores["hybrid"] <= 1
    assert c.messy_accuracy(bundle) == scores["hybrid"]
    # rules fix what the ML model alone gets wrong, so the hybrid must be at least as good
    assert scores["hybrid"] >= scores["ml_only"]


def _tmp(name):
    """A fresh temp file path (any old copy is removed) so tests never touch the real corrections/model."""
    import tempfile
    path = os.path.join(tempfile.gettempdir(), name)
    if os.path.exists(path):
        os.remove(path)
    return path


def test_expenses_workbook_labels():
    # every label in the user's "Copy of Expenses.xlsx" sheet -> the category it should get
    expected = {
        "Food/Drinks": "Food", "Eating/Going Out": "Food", "Grocery": "Groceries",
        "Transportation": "Travel", "Phone/Internet": "Bills", "Electricity/Water": "Bills",
        "Garbage": "Bills", "Cable": "Bills", "Bank fees": "Bills", "Pension Fund": "Bills",
        "SSS": "Bills", "Home": "Bills", "Apartment": "Rent",
        "Debt repayment": "Transfers", "Credit Card Payment": "Transfers", "Revi Credit": "Transfers",
        "Clothing/Shoes/Personal Items": "Shopping", "Hair Cut": "Shopping",
        "Personal care items (drugstore)": "Shopping",
        "Entertainment/Subscriptions": "Entertainment", "Magazines/Newspapers/Books": "Entertainment",
        "Health Insurance": "Health", "Healthcare": "Health", "PhilHealth": "Health",
        "Others": "Others",
    }
    bundle = c.load_model()
    for label, category in expected.items():
        assert c.categorize_text(label, bundle) == category, (label, c.categorize_text(label, bundle))


def test_generic_labels_ml_only():
    # Short generic labels the ML model has NEVER seen (none is in generate_data.py's GENERIC list).
    # rule_category is bypassed on purpose: this checks what the model itself learned.
    model = c.load_model()["model"]
    unseen = {"Cab Rides": "Travel", "Eating Out Weekend": "Food", "Vegetable Shopping": "Groceries",
              "Doctor Fees": "Health", "Movie Night": "Entertainment", "Mobile Bill": "Bills"}
    right = sum(c.ml_category(label, model) == cat for label, cat in unseen.items())
    assert right >= 4, right
    # the two labels named in the request are in the training data, so they must be right
    assert c.ml_category("Transportation", model) == "Travel"
    assert c.ml_category("Eating Out", model) == "Food"


def test_training_data_is_varied():
    df = pd.read_csv(c.DATA_PATH)
    assert len(df) > 2000 and set(df["category"]) == set(c.CATEGORIES)
    # the messy TEST merchants must not leak into training (they are what "unseen" means)
    test_only = ["Behrouz", "Faasos", "Lenskart", "Prime Video", "Practo", "Ravi Kumar", "Cleartrip"]
    for name in test_only:
        assert not df["description"].str.contains(name, case=False).any(), name


def test_confusion_matrix_png():
    assert os.path.getsize(c.MATRIX_PATH) > 1000  # created by training


if __name__ == "__main__":
    # run every function whose name starts with "test_"
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"OK  {name}")
    print("All categorizer tests passed.")
