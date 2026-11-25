# parser.py - reads a bank/UPI statement (CSV or Excel) and cleans it up.
#
# Pipeline (see parse_statement at the bottom):
#   1. read_statement        -> load the file into a DataFrame
#   2. rename_columns        -> bank-specific headers -> date, description, amount, debit, credit, type
#   3. build_amount_and_type -> one positive "amount" column + a "type" (debit/credit)
#   4. convert_dates         -> text dates -> real datetime values
#   5. extract_merchant      -> "UPI/123/Swiggy/swiggy@ybl" -> "Swiggy"   (regex)
#   6. mask_sensitive        -> hide UPI IDs and account numbers           (regex)
#
# Final columns: date, description (masked), amount, type, merchant  (+ category if the file has one)

import os
import re

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Column names used by different banks -> our standard names
#    Keys are written in lowercase, without dots (we normalise headers first).
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "date": ["date", "txn date", "tran date", "transaction date", "value date",
             "value dt", "posting date"],
    "description": ["description", "narration", "particulars", "remarks",
                    "transaction details", "transaction remarks",
                    "details", "detail", "note", "notes", "memo", "payee",
                    "expenses", "expense", "item"],  # last row: simple expense-tracker sheets
    "amount": ["amount", "txn amount", "transaction amount"],
    "debit": ["debit", "debit amount", "withdrawal", "withdrawal amt",
              "withdrawal amount", "dr"],
    "credit": ["credit", "credit amount", "deposit", "deposit amt",
               "deposit amount", "cr"],
    "type": ["type", "dr/cr", "cr/dr", "txn type", "transaction type"],
    "category": ["category"],
}

# Flip the dictionary so we can look up "narration" -> "description" quickly
ALIAS_TO_STANDARD = {
    alias: standard
    for standard, aliases in COLUMN_ALIASES.items()
    for alias in aliases
}


def read_statement(file):
    """Load a CSV or Excel statement. `file` is a path or a Streamlit upload."""
    # A path is a string; an uploaded file has a .name attribute
    name = file if isinstance(file, (str, os.PathLike)) else file.name
    extension = os.path.splitext(str(name))[1].lower()
    if hasattr(file, "seek"):
        file.seek(0)  # an upload may already have been read once (e.g. before the user picked a column)

    if extension == ".csv":
        return pd.read_csv(file)
    if extension in (".xlsx", ".xls"):
        return pd.read_excel(file)
    raise ValueError(f"Unsupported file type '{extension}'. Please upload a CSV or Excel file.")


class MissingColumnError(ValueError):
    """A required column was not found. `.columns` lists the file's columns so the app
    can let the user pick the right one from a dropdown."""

    def __init__(self, message, columns):
        super().__init__(message)
        self.columns = list(columns)


def rename_columns(df, description_column=None):
    """Rename bank-specific column headers to our standard names.
    description_column = the file's column the user picked by hand to be the description."""
    new_names = {}
    used = set()  # make sure two columns never get the same standard name
    if description_column is not None:
        new_names[description_column] = "description"   # the user's choice beats the alias guess
        used.add("description")
    for col in df.columns:
        if col == description_column:
            continue
        # normalise: lowercase, remove dots, collapse spaces ("Withdrawal Amt." -> "withdrawal amt")
        clean = re.sub(r"\s+", " ", str(col).lower().replace(".", "")).strip()
        standard = ALIAS_TO_STANDARD.get(clean)
        if standard and standard not in used:
            new_names[col] = standard
            used.add(standard)
    return df.rename(columns=new_names)


def _to_number(series):
    """Turn text like '1,250.50' or 'Rs. 99' into numbers (blank -> NaN)."""
    cleaned = series.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def build_amount_and_type(df):
    """Create a positive 'amount' column and a 'type' column ('debit' or 'credit').

    Banks store this in three different ways, so we check each one:
      A) separate Debit / Credit columns
      B) a Dr/Cr (or Debit/Credit) type column next to the amount
      C) a single signed amount (negative = money out)
    If there is no hint at all we treat every row as a debit (a spend).
    """
    df = df.copy()

    if "debit" in df.columns or "credit" in df.columns:        # case A
        zeros = pd.Series(0.0, index=df.index)   # used if one of the two columns is missing
        debit = _to_number(df["debit"]).fillna(0) if "debit" in df.columns else zeros
        credit = _to_number(df["credit"]).fillna(0) if "credit" in df.columns else zeros
        df["amount"] = debit.where(debit > 0, credit)      # whichever side has a value
        is_credit = (debit <= 0) & (credit > 0)
        df["type"] = is_credit.map({True: "credit", False: "debit"})

    elif "amount" in df.columns:
        amount = _to_number(df["amount"])
        if "type" in df.columns:                                # case B
            # "CR", "Credit", "C" -> credit ; everything else -> debit
            is_credit = df["type"].astype(str).str.strip().str.lower().str.startswith("c")
            df["type"] = is_credit.map({True: "credit", False: "debit"})
        elif (amount < 0).any():                                # case C
            df["type"] = (amount < 0).map({True: "debit", False: "credit"})
        else:                                                   # no hint -> spending
            df["type"] = "debit"
        df["amount"] = amount.abs()

    else:
        raise ValueError("Could not find an amount (or debit/credit) column in the statement.")

    return df


def convert_dates(df):
    """Convert the 'date' column to real dates. Unreadable dates are dropped."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):   # Excel may already give real dates
        text = df["date"].astype(str).str.strip()
        # 1) ISO dates like 2026-05-01 (year first) - must NOT be read day-first
        iso = pd.to_datetime(text, format="ISO8601", errors="coerce")
        # 2) everything else, e.g. 25/11/2026 - Indian statements are day/month/year
        other = pd.to_datetime(text, dayfirst=True, errors="coerce")
        df["date"] = iso.fillna(other)
    return df.dropna(subset=["date"])


# ---------------------------------------------------------------------------
# Regex helpers for cleaning descriptions
# ---------------------------------------------------------------------------
UPI_ID_PATTERN = re.compile(r"([A-Za-z0-9._-]+)@([A-Za-z]+)")   # name@bank
LONG_NUMBER_PATTERN = re.compile(r"\d{9,18}")                    # account / reference numbers
PREFIX_PATTERN = re.compile(r"^\s*(UPI|NEFT|IMPS|RTGS|POS)\b[\s/\-:]*", re.IGNORECASE)


def mask_sensitive(text):
    """Hide UPI IDs and long numbers.

    'rahul@ybl'      -> 'ra****@ybl'     (keep first 2 letters + the bank handle)
    '50100123456789' -> '**********6789' (keep only the last 4 digits)
    """
    text = str(text)
    text = UPI_ID_PATTERN.sub(lambda m: f"{m.group(1)[:2]}****@{m.group(2)}", text)
    text = LONG_NUMBER_PATTERN.sub(lambda m: "*" * (len(m.group()) - 4) + m.group()[-4:], text)
    return text


def _is_reference(token):
    """True for tokens that are codes, not names: '4182736', 'HDFCN52026', '4567XXXXXX1234'."""
    token = token.strip()
    has_digit = any(ch.isdigit() for ch in token)
    return (" " not in token and has_digit and len(token) >= 6) or bool(re.fullmatch(r"[\dXx*]{4,}", token))


def _tidy_case(name):
    """'SWIGGY' / 'swiggy' -> 'Swiggy'. Mixed case like 'McDonalds' is left alone."""
    name = re.sub(r"\s+", " ", name).strip(" -/:.")
    return name.title() if name.isupper() or name.islower() else name


def extract_merchant(description):
    """Pull a clean merchant / person name out of a bank narration.

    UPI/418273650192/Swiggy/swiggy@ybl       -> Swiggy
    UPI/418273650192/P2P/Rahul S/ra****@ybl  -> Rahul S
    NEFT-HDFCN52026112-RAHUL SHARMA-SALARY   -> Rahul Sharma
    IMPS/412345678901/PRIYA M/HDFC/Rent      -> Priya M
    POS 4567XXXXXX1234 BIG BAZAAR MUMBAI     -> Big Bazaar Mumbai
    Netflix                                  -> Netflix (already clean)
    """
    text = str(description).strip()
    match = PREFIX_PATTERN.match(text)
    if not match:
        return _tidy_case(text)            # plain text: nothing to strip

    mode = match.group(1).upper()
    rest = text[match.end():]

    if mode == "POS":
        # POS narrations are space separated: drop the card number, keep the shop name
        parts = [p for p in re.split(r"[\s/]+", rest) if p and not _is_reference(p)]
        return _tidy_case(" ".join(parts))

    # UPI / NEFT / IMPS / RTGS: parts are separated by "/" or "-"
    parts = [p.strip() for p in re.split(r"[/\-]", rest) if p.strip()]
    parts = [p for p in parts
             if "@" not in p                         # UPI ids
             and not _is_reference(p)                # reference numbers
             and p.upper() not in ("P2P", "P2M", "PAY", "PAID")]
    return _tidy_case(parts[0]) if parts else _tidy_case(text)


def parse_statement(file, description_column=None):
    """Read a statement and return a clean DataFrame:
    date, description (masked), amount, type, merchant  (+ category if present).
    If no description-like column is found, raises MissingColumnError (with the column list);
    call again with description_column=<the column the user chose>."""
    df = read_statement(file)
    original_columns = list(df.columns)
    df = rename_columns(df, description_column)

    missing = [c for c in ("date", "description") if c not in df.columns]
    if missing:
        raise MissingColumnError(
            f"Statement is missing required column(s): {', '.join(missing)}", original_columns)

    df = build_amount_and_type(df)
    df = convert_dates(df)

    # Extract the merchant FIRST (needs the raw text), then mask the description
    df["merchant"] = df["description"].apply(extract_merchant).apply(mask_sensitive)
    df["description"] = df["description"].apply(mask_sensitive)

    columns = ["date", "description", "amount", "type", "merchant"]
    if "category" in df.columns:
        columns.append("category")
    return df[columns].reset_index(drop=True)
