# categorizer.py - puts every transaction into one of our fixed categories.
#
# Two layers (see categorize_text at the bottom):
#   Layer 1: keyword rules   -> fast, 100% explainable, handles well-known merchants
#   Layer 2: ML model        -> TF-IDF + Logistic Regression (or Naive Bayes, whichever scores better)
#                               used only when no rule matches
#   Safety net: if the model is not confident enough, the answer becomes "Others".
#
# Run with:  python categorizer.py     (trains, compares the models, saves the best one)

import os
import re
from difflib import SequenceMatcher

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to a file, no pop-up window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from merchants import NEW_MERCHANTS   # more well-known Indian merchants (also used for training rows)
from people import BUSINESS_WORDS, FIRST_NAMES, RELATIONS   # to spot payments to people

# The fixed category list from the project rules (do not change).
CATEGORIES = ["Food", "Groceries", "Travel", "Shopping", "Bills", "Rent",
              "Entertainment", "Health", "Transfers", "Others"]

DATA_PATH = "data/training.csv"
MODEL_PATH = "models/categorizer.joblib"
MATRIX_PATH = "models/confusion_matrix.png"                 # messy (realistic) test set
CLEAN_MATRIX_PATH = "models/confusion_matrix_clean.png"     # clean random split (too easy)
REALISTIC_PATH = "data/test_realistic.csv"                  # made by make_realistic_test.py
CORRECTIONS_PATH = "corrections.csv"                        # user fixes from the app: description,category

# If the model's best guess has less than 40% probability, we do not trust it.
CONFIDENCE_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Layer 1: keyword rules. category -> words that identify it.
# The first category (top to bottom) with a matching word wins.
# Words are matched as whole words, so "rent" will not match "current".
# ---------------------------------------------------------------------------
KEYWORD_RULES = {
    "Transfers": ["p2p", "neft", "imps", "self transfer",
                  "debt repayment", "credit card payment", "revi credit",
                  "sent to", "transfer to", "money transfer", "family support", "pocket money", "to friend"],
    "Rent": ["rent", "landlord", "apartment"],
    "Food": ["swiggy", "zomato", "dominos", "mcdonalds", "starbucks", "kfc",
             "pizza", "burger", "restaurant", "cafe", "chai",
             "food", "eating"],
    "Groceries": ["blinkit", "bigbasket", "zepto", "dmart", "reliance fresh",
                  "grocery", "supermarket", "instamart"],
    "Travel": ["uber", "ola", "rapido", "irctc", "metro", "makemytrip", "redbus",
               "flight", "airlines", "railway", "petrol", "fuel", "transportation"],
    "Entertainment": ["bookmyshow", "pvr", "inox", "spotify", "netflix", "hotstar",
                      "steam", "cinema", "movie", "youtube", "entertainment", "subscriptions",
                      "magazines", "newspapers", "books"],
    "Health": ["pharmacy", "pharmeasy", "1mg", "apollo", "hospital", "clinic",
               "cult fit", "diagnostics", "medical", "healthcare", "health insurance", "philhealth"],
    "Bills": ["airtel", "jio", "vodafone", "electricity", "bescom", "tata power",
              "gas", "broadband", "recharge", "insurance", "water bill",
              "phone", "internet", "cable", "garbage", "bank fees", "bank fee",
              "pension fund", "sss", "home"],
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "decathlon", "croma",
                 "tanishq", "apple store", "nykaa", "mall", "clothing", "personal care", "hair cut"],
}


def clean_text(description):
    """Turn a raw narration into plain lowercase words.
    'UPI/428799513174/Swiggy/swiggy@ybl' -> 'swiggy swiggy'
    We drop 'upi', numbers/reference ids and UPI handles because they say nothing about the category."""
    text = str(description).lower().replace("'", "")   # Domino's -> dominos
    text = re.sub(r"[\w.*]+@[a-z0-9.]+", " ", text)  # UPI ids like swiggy@ybl or ra****@oksbi (stops at "/", "-" or space)
    text = re.sub(r"\bupi\b", " ", text)       # the word UPI is on every row
    text = re.sub(r"\d{4,}", " ", text)       # long number runs = reference ids (short ones stay: p2p, 1mg)
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep letters/digits only (removes / - * etc.)
    return re.sub(r"\s+", " ", text).strip()


# Every merchant in merchants.py is also a keyword for its category.
for _category, _names in NEW_MERCHANTS.items():
    for _name in _names:
        _keyword = clean_text(_name)
        if _keyword not in KEYWORD_RULES[_category]:
            KEYWORD_RULES[_category].append(_keyword)


# Fuzzy matching: a keyword also matches a slightly misspelled word ("swigy" -> "swiggy").
# difflib's ratio is 1.0 for identical text and 0.0 for nothing in common.
FUZZY_MIN_RATIO = 0.80    # how similar a word must be to the keyword
FUZZY_MIN_LENGTH = 5      # short keywords (ola, jio, kfc, gas) would match too many wrong words
FUZZY_REAL_WORDS = {"interest"}   # real words that only look like a keyword ("interest" is not "internet")


def fuzzy_rule_category(text):
    """Best category whose keyword is a close misspelling of words in the (cleaned) text, or None.
    A typo keeps the first letter and changes the length by at most 1, so other words are skipped
    ('phonepe' is NOT a typo of the keyword 'phone'). Multi-word keywords like 'cult fit'
    are compared with the same number of consecutive words."""
    tokens = text.split()
    best_ratio, best_category = 0.0, None
    for category, words in KEYWORD_RULES.items():
        for word in words:
            if len(word) < FUZZY_MIN_LENGTH:
                continue
            size = len(word.split())
            for i in range(len(tokens) - size + 1):
                candidate = " ".join(tokens[i:i + size])
                if abs(len(candidate) - len(word)) > 1 or candidate[0] != word[0]:
                    continue
                if candidate in FUZZY_REAL_WORDS:
                    continue
                ratio = SequenceMatcher(None, candidate, word).ratio()
                if ratio >= FUZZY_MIN_RATIO and ratio > best_ratio:
                    best_ratio, best_category = ratio, category
    return best_category


# Person-to-person payments (UPI to a friend, family member, ...). No merchant word is involved,
# so we recognise the PERSON instead. A mobile number starts with 6-9 and has 10 digits.
PHONE_NUMBER = re.compile(r"(?<![\d*])[6-9]\d{9}(?!\d)")              # 9876543210@ybl, "paid 9876543210"
MASKED_PHONE_HANDLE = re.compile(r"(?<![\w*])[6-9]\d\*{2,}\d*@")      # 98****@ybl  (how the parser masks it)
UPI_HANDLE = re.compile(r"([a-z0-9._*-]+)@[a-z]+")                    # the part before the @
PERSONAL_HANDLE = re.compile(r"[a-z]+(?:[._-][a-z0-9]+)?\d{0,4}")     # rahul / rahul.k / priya_m / amit123


def person_transfer(description):
    """True if the payment looks like money sent to a person, not to a business:
      - a mobile number as the UPI id or in the text, or
      - a UPI id like rahul.k@okaxis whose first part is a common first name, or
      - a common first name (or Mom, Dad, Bhai...) in the text.
    A business word (Store, Traders, Pharmacy...) in the text cancels all of this."""
    text = str(description).lower()
    words = set(clean_text(text).split())
    if words & BUSINESS_WORDS:
        return False
    if PHONE_NUMBER.search(text) or MASKED_PHONE_HANDLE.search(text):
        return True
    for handle in UPI_HANDLE.findall(text):
        if PERSONAL_HANDLE.fullmatch(handle) and re.match(r"[a-z]+", handle).group() in FIRST_NAMES | RELATIONS:
            return True
    return bool(words & (FIRST_NAMES | RELATIONS))


def rule_category(description, fuzzy=True):
    """Layer 1. Return the category whose keyword appears in the text, or None if nothing matches.
    Order: exact keywords (the longest wins), then misspelled keywords (fuzzy), then 'a payment to a person'."""
    text = clean_text(description)
    best_length, best_category = 0, None
    for category, words in KEYWORD_RULES.items():
        for word in words:
            # the longest matching keyword is the most specific one ("swiggy instamart" beats "swiggy")
            if len(word) > best_length and re.search(rf"\b{re.escape(word)}\b", text):
                best_length, best_category = len(word), category
    if best_category:
        return best_category
    if fuzzy:
        fuzzy_hit = fuzzy_rule_category(text)
        if fuzzy_hit:
            return fuzzy_hit
    return "Transfers" if person_transfer(description) else None


# ---------------------------------------------------------------------------
# Layer 2: machine-learning models
# ---------------------------------------------------------------------------
def make_pipeline(classifier):
    """TF-IDF turns text into numbers; the classifier learns text -> category.
    char_wb n-grams (2-4 letters) cope with truncated or slightly different names,
    e.g. 'swigy' is still close to 'swiggy'."""
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    return Pipeline([("tfidf", tfidf), ("clf", classifier)])


# ---------------------------------------------------------------------------
# Corrections: categories the user fixed by hand in the app (saved in corrections.csv)
# ---------------------------------------------------------------------------
def load_corrections(path=CORRECTIONS_PATH):
    """Return {cleaned description: category}. Empty dict if the file does not exist yet.
    If the same description was corrected twice, the latest correction wins."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {clean_text(d): c for d, c in zip(df["description"], df["category"])}


def save_corrections(rows, path=CORRECTIONS_PATH):
    """Save corrections. rows = list of (description, category) pairs.
    The file keeps one row per cleaned description (the newest one), so it never grows with repeats."""
    for _, category in rows:
        if category not in CATEGORIES:  # only our fixed categories are allowed
            raise ValueError(f"'{category}' is not one of the fixed categories")
    new = pd.DataFrame(rows, columns=["description", "category"])
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new["key"] = new["description"].map(clean_text)
    new = new.drop_duplicates("key", keep="last").drop(columns="key")  # newest correction wins
    new.to_csv(path, index=False)


def load_training_frame(path=DATA_PATH, corrections_path=None):
    """The original training data, plus the user's corrections when a corrections file is given.
    Credits (salary, refunds) are skipped: they are labelled by credit_label(), never by the model."""
    df = pd.read_csv(path)
    if "type" in df.columns:
        df = df[df["type"] == "debit"]
    df = df[["description", "category"]]
    if corrections_path and os.path.exists(corrections_path):
        df = pd.concat([df, pd.read_csv(corrections_path)[["description", "category"]]],
                       ignore_index=True)
    return df


def load_training_data(path=DATA_PATH, corrections_path=None):
    """Return (cleaned texts, categories) for training (original data + corrections if given)."""
    df = load_training_frame(path, corrections_path)
    return df["description"].map(clean_text), df["category"]


def evaluate(name, model, X_test, y_test):
    """Print accuracy and F1 for one model and return the macro F1 (used to pick the winner).
    Macro F1 = average F1 of every category, so small categories (Rent) count as much as big ones."""
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1_macro = f1_score(y_test, pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, pred, average="weighted", zero_division=0)
    print(f"{name:<20} accuracy={acc:.3f}  macro-F1={f1_macro:.3f}  weighted-F1={f1_weighted:.3f}")
    return f1_macro


def evaluate_unseen_merchants(X, y, raw_descriptions):
    """Harder test: the model is scored ONLY on merchants it never saw in training.
    GroupKFold keeps all rows of one merchant (same cleaned text) in the same fold, so a
    merchant is either in training or in testing, never both. This shows how well each model
    copes with new names (the normal random split above lets them memorise names)."""
    groups = X  # cleaned text identifies the merchant
    print("\nUnseen-merchant test (5-fold GroupKFold), ML models only:")
    results = {}
    for name, make in [("Logistic Regression", lambda: LogisticRegression(C=10, max_iter=1000)),
                       ("Naive Bayes", lambda: MultinomialNB(alpha=0.1))]:
        pred_ml, pred_full = pd.Series(index=y.index, dtype=object), pd.Series(index=y.index, dtype=object)
        for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
            model = make_pipeline(make()).fit(X.iloc[train_idx], y.iloc[train_idx])
            for i in test_idx:
                ml = ml_category(raw_descriptions.iloc[i], model)  # ML with the "Others" safety net
                pred_ml.iloc[i] = ml
                pred_full.iloc[i] = rule_category(raw_descriptions.iloc[i]) or ml
        results[name] = (pred_ml, pred_full)
        print(f"{name:<20} ML only: accuracy={accuracy_score(y, pred_ml):.3f} "
              f"macro-F1={f1_score(y, pred_ml, average='macro', zero_division=0):.3f} | "
              f"rules+ML: accuracy={accuracy_score(y, pred_full):.3f} "
              f"macro-F1={f1_score(y, pred_full, average='macro', zero_division=0):.3f}")
    return results


def save_confusion_matrix(y_true, y_pred, path, title):
    """Draw a confusion matrix and save it as a PNG.
    Rows = true category, columns = predicted category; a perfect model has numbers only on the diagonal."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = [c for c in CATEGORIES if c in set(y_true) | set(y_pred)]  # keep our fixed order
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, ax=ax, cmap="Blues",
        xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_realistic(candidates, path=REALISTIC_PATH):
    """Score every model on the MESSY test set (typos, odd formats, unseen merchants, non-spending rows).
    Two numbers per model:
      ML only   = the model alone (with the low-confidence -> Others rule)
      rules+ML  = the real two-layer system
    Returns {model name: predictions of the full system} and the test DataFrame."""
    df = pd.read_csv(path)
    print(f"\nMessy test set: {len(df)} rows that were never used for training")
    full_predictions = {}
    for name, model in candidates.items():
        ml = [ml_category(d, model) for d in df["description"]]
        full = [rule_category(d) or m for d, m in zip(df["description"], ml)]
        full_predictions[name] = full
        print(f"{name:<20} ML only: accuracy={accuracy_score(df['category'], ml):.3f} "
              f"macro-F1={f1_score(df['category'], ml, average='macro', zero_division=0):.3f} | "
              f"rules+ML: accuracy={accuracy_score(df['category'], full):.3f} "
              f"macro-F1={f1_score(df['category'], full, average='macro', zero_division=0):.3f}")
        # accuracy of the full system for each kind of messy row
        by_kind = (pd.Series(full, index=df.index) == df["category"]).groupby(df["kind"]).mean()
        print("   rules+ML accuracy by kind: " + ", ".join(f"{k}={v:.2f}" for k, v in by_kind.items()))
    return full_predictions, df


def train_and_save(data_path=DATA_PATH, corrections_path=None, model_path=MODEL_PATH):
    """Train both models, compare them, save the best with joblib. Returns the saved bundle (a dict).
    Pass corrections_path to train on the original data PLUS the user's corrections."""
    X, y = load_training_data(data_path, corrections_path)
    raw_descriptions = load_training_frame(data_path, corrections_path)["description"]
    # Hold back 20% of the data the models never see, so the scores are honest.
    # stratify=y keeps the same category mix in both parts.
    # (stratifying needs at least 2 rows per category; a lone correction in a new category has 1.)
    can_stratify = y.value_counts().min() >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None)

    candidates = {
        "Logistic Regression": make_pipeline(LogisticRegression(C=10, max_iter=1000)),
        "Naive Bayes": make_pipeline(MultinomialNB(alpha=0.1)),
    }

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows\n")
    scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        scores[name] = evaluate(name, model, X_test, y_test)

    # The clean split is too easy (both models get 100%), so it cannot tell them apart.
    # Pick the winner on the messy test set instead (macro-F1 of the full two-layer system).
    if os.path.exists(REALISTIC_PATH):
        full_predictions, messy = evaluate_realistic(candidates)
        scores = {n: f1_score(messy["category"], p, average="macro", zero_division=0)
                  for n, p in full_predictions.items()}
    else:
        print(f"\n({REALISTIC_PATH} not found - run make_realistic_test.py. Choosing on the clean split.)")
        full_predictions, messy = None, None
    best_name = max(scores, key=scores.get)  # on a tie the first one (Logistic Regression) wins
    best = candidates[best_name]
    print(f"\nBest model: {best_name}\n")

    if messy is not None:
        print(classification_report(messy["category"], full_predictions[best_name], zero_division=0))
        save_confusion_matrix(messy["category"], full_predictions[best_name], MATRIX_PATH,
                              f"Confusion matrix - messy test set ({best_name}, rules + ML)")
        print(f"Confusion matrix (messy set) saved to {MATRIX_PATH}")
    save_confusion_matrix(y_test, best.predict(X_test),
                          MATRIX_PATH if messy is None else CLEAN_MATRIX_PATH,
                          "Confusion matrix - clean test split (ML only)")

    # For comparison: the full two-layer system on the clean split.
    test_descriptions = raw_descriptions.loc[X_test.index]
    hybrid_pred = [categorize_text(d, best) for d in test_descriptions]
    print(f"Clean split, full system (rules + ML + Others): accuracy={accuracy_score(y_test, hybrid_pred):.3f}  "
          f"macro-F1={f1_score(y_test, hybrid_pred, average='macro', zero_division=0):.3f}")

    evaluate_unseen_merchants(X, y, raw_descriptions)

    # clean_accuracy = accuracy on the held-out 20% (this split includes the corrections)
    bundle = {"model": best, "name": best_name, "threshold": CONFIDENCE_THRESHOLD,
              "clean_accuracy": accuracy_score(y_test, best.predict(X_test)),
              "n_corrections": len(X) - len(load_training_frame(data_path))}
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"Model saved to {model_path}")
    return bundle


def messy_scores(bundle, path=REALISTIC_PATH):
    """Accuracy on the fixed messy test set (never used for training, never changes), so the
    numbers are comparable before and after retraining. Two numbers:
      ml_only = the ML model alone (with the low-confidence -> Others rule)
      hybrid  = the real system: keyword rules first, ML only when no rule matches
    The user-corrections layer is NOT used, so this measures what the model itself learned."""
    df = pd.read_csv(path)
    ml = [ml_category(d, bundle["model"], bundle["threshold"]) for d in df["description"]]
    hybrid = [categorize_text(d, bundle) for d in df["description"]]
    return {"ml_only": accuracy_score(df["category"], ml),
            "hybrid": accuracy_score(df["category"], hybrid)}


def messy_accuracy(bundle, path=REALISTIC_PATH):
    """Accuracy of the full hybrid system (rules + ML) on the messy test set."""
    return messy_scores(bundle, path)["hybrid"]


def load_model(path=MODEL_PATH):
    """Load the saved model bundle; train it first if the file does not exist yet."""
    if not os.path.exists(path):
        return train_and_save()
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Putting the layers together
# ---------------------------------------------------------------------------
def ml_category(description, model, threshold=CONFIDENCE_THRESHOLD):
    """Layer 2. Ask the model; if its best probability is below the threshold, answer 'Others'."""
    text = clean_text(description)
    if not text:                       # nothing readable left after cleaning
        return "Others"
    probabilities = model.predict_proba([text])[0]
    best = probabilities.argmax()
    if probabilities[best] < threshold:
        return "Others"
    return model.classes_[best]


def categorize_text(description, bundle_or_model=None, corrections=None):
    """Category for ONE description: user corrections first, then rules, then ML, then 'Others'.
    corrections = {cleaned description: category} from load_corrections() (optional)."""
    if corrections:
        fixed = corrections.get(clean_text(description))
        if fixed:
            return fixed
    rule = rule_category(description)
    if rule:
        return rule
    if bundle_or_model is None:
        bundle_or_model = load_model()
    if isinstance(bundle_or_model, dict):
        return ml_category(description, bundle_or_model["model"], bundle_or_model["threshold"])
    return ml_category(description, bundle_or_model)


def credit_label(description):
    """Label for money coming IN. No ML here: a credit is 'Refund' if the text says so
    (refund / reversal / cashback), otherwise 'Income' (salary, interest, anything else).
    These two labels are only for credits; the 10 spending categories stay unchanged."""
    text = str(description).lower()
    return "Refund" if any(word in text for word in ("refund", "reversal", "cashback")) else "Income"


def categorize_dataframe(df, column="description", bundle=None, corrections_path=CORRECTIONS_PATH):
    """Add a 'category' column to a parsed statement (the DataFrame from parser.parse_statement).
    Debits go through corrections + rules + ML; credits get 'Income' / 'Refund' without the model.
    Saved user corrections are applied on top (pass corrections_path=None to ignore them)."""
    bundle = bundle or load_model()
    corrections = load_corrections(corrections_path) if corrections_path else {}
    df = df.copy()
    is_credit = df["type"] == "credit" if "type" in df.columns else pd.Series(False, index=df.index)
    df["category"] = [credit_label(d) if credit else categorize_text(d, bundle, corrections)
                      for d, credit in zip(df[column], is_credit)]
    return df


if __name__ == "__main__":
    train_and_save()
