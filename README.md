# Personal Expense Analyzer

An AI-based analytical system that reads a bank / UPI statement, categorizes every transaction with
machine learning, shows a spending dashboard and writes a short AI summary of your month.
(Semester project.)

## Problem statement

Bank and UPI statements are long lists of cryptic narrations such as
`UPI/418273650192/Swiggy/swiggy@ybl`. They do not tell you where your money actually goes. Doing it by
hand in a spreadsheet is slow, and every bank uses a different file layout.

This project takes the raw statement (CSV or Excel), cleans it, **labels each transaction with one
of ten fixed categories automatically**, and turns the result into charts, budget alerts, a forecast
and a plain-English summary, without sending any personal transaction text to an outside service.

## Features

- **Upload CSV or Excel** statements from different banks (column names are matched automatically; if
  no description column is found, you choose it from a dropdown).
- **Several statements at once** (different banks or UPI apps): they are merged into one table sorted
  by date, each row tagged with an account label you can edit. Exact duplicates within an account (same
  date, amount, description and debit/credit type) are removed. A debit in one account and a credit of
  the same amount in another within 2 days is labelled **Self transfer** and counts as neither spending
  nor income. An upload summary shows rows per file, duplicates removed and self transfers found, and
  the Dashboard and Transactions pages get a **Source** filter.
  Limits of these simple rules: two unrelated payments of the same amount can be mistaken for a self
  transfer, a transfer with a bank fee (different amounts) is missed, and two truly identical payments
  on the same day are counted once. Nothing is removed or paired when you upload a single file.
  Try it with `data/sample_bank_a.csv` and `data/sample_bank_b.csv` (fake data).
- **Friendly error messages** for a wrong file type, an empty or damaged file, a missing date or
  amount column, or rows with unreadable dates and amounts.
- **Privacy by design:** UPI IDs and long account / reference numbers are masked while parsing.
- **Automatic categorization** into Food, Groceries, Travel, Shopping, Bills, Rent, Entertainment,
  Health, Transfers, Others (credits are labelled Income or Refund).
- **Dashboard:** total spent, transfers out, income, savings, category pie chart, monthly trend,
  top 5 merchants, biggest 5 transactions.
- **Transactions page:** search, date / category / amount filters, and an editable category column.
  Corrections are saved and take effect immediately.
- **Insights page:** unusual spends (Isolation Forest), subscriptions and recurring payments,
  monthly budgets with warnings, next-month spending forecast (Linear Regression) and an AI monthly
  report.
- **Goals page:** savings goals (name, target, date, amount already saved) with the months left, the
  saving needed per month, your average savings over the last 3 months of the statement, a status
  (On track / Slightly behind = saving at least 80% of what is needed / Off track), a progress bar and
  the date you would reach it. If off track, it suggests cuts in Food, Shopping, Entertainment and
  Travel only (the same percentage from each, never more than 50%).
- **Model page:** retrain the model with your corrections and see accuracy before and after.
- **AI report (Gemini):** only totals, percentages and counts are sent. Never merchant names or
  transaction descriptions.

## Tech stack

| Part | Tool |
|---|---|
| Language | Python 3.12 |
| UI | Streamlit |
| Data | Pandas, openpyxl (Excel) |
| ML | Scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Isolation Forest, Linear Regression) |
| Charts | Plotly (dashboard), Matplotlib (confusion matrix images) |
| LLM | Google Gemini API (`google-genai`) |
| Config | python-dotenv (`.env` file) |

## Architecture

```
 Statement (CSV / Excel)
          |
          v
   parser.py ---- read file, map column names, build amount + debit/credit type,
          |        convert dates, extract merchant, MASK UPI IDs and account numbers
          v
 categorizer.py -- layer 0: your saved corrections (corrections.csv)
          |        layer 1: keyword rules (fast, explainable)
          |        layer 2: ML model (TF-IDF + Logistic Regression), only if no rule matched
          |        safety net: model not confident (< 40%)  ->  "Others"
          v
   analytics.py    insights.py     report.py
   totals, charts  unusual spends  aggregated stats only ---> Gemini ---> summary + tips
                   subscriptions
                   budgets, forecast
          \             |             /
           v            v            v
                    app.py (Streamlit)
        Dashboard | Transactions | Insights | Model
```

## ML approach and results

**How it works.** A hybrid, two-layer categorizer:

1. **Keyword rules** (with a small typo-tolerant match) catch well-known merchants such as Swiggy,
   Netflix or "rent". A helper also spots payments to people ("Rahul S", "IMPS", "P2P") as Transfers.
2. **ML model** for everything the rules miss: descriptions are cleaned, turned into TF-IDF word and
   character features, and classified by Logistic Regression (Naive Bayes is trained too, and the
   better one on the messy test set is kept). If the best guess has under 40% probability the row is
   labelled "Others" instead of guessing.

Your corrections override both layers, and "Retrain model" adds them to the training data.

**Data.** There is no public labelled statement data, so the training set (about 4,000 rows,
`data/training.csv`) is **synthetic**, generated from templates by `generate_data.py`. To avoid fooling
ourselves, a separate **messy test set** (436 rows, `data/test_realistic.csv`, written by
`make_realistic_test.py`) has typos, odd formats, unseen merchants and unseen people, and none of its
merchants appear in the training data (a test checks this).

**Results**

| Test | Accuracy |
|---|---|
| Held-out 20% of the synthetic training data (too easy, only shows the pipeline works) | about 99% |
| **Messy test set, hybrid (rules + ML)** | **75.0%** |
| **Messy test set, ML only** | **65.6%** |

On the messy set the rules add about 9 points over the ML model alone. Breakdown of the hybrid
system by kind of row: known merchants 100%, typos 100%, new people 92%, **new merchants only 42%**.
A confusion matrix is saved in `models/confusion_matrix.png`.

**Honest limitations** (full list in [docs/limitations.md](docs/limitations.md))

- The training data is synthetic, and the messy test set was written by me and also used to pick the
  winning model, so 75% is probably slightly optimistic. Real statements would score differently, and
  the accuracy on them has not been measured.
- A merchant the system has never seen is usually wrong (the ML model alone scores about 37%).
- "Others" is over-used: only about 16% of rows labelled Others really are Others.
- Rules always win over the model, so a bad or too-broad keyword cannot be fixed by retraining.
- Corrections only match descriptions that are identical after cleaning, and there is no undo screen.
- The keyword rules are tuned to Indian merchants and UPI-style narrations.
- The spending forecast is a straight-line trend and needs at least 3 complete months.
- The AI report is only as good as the numbers sent, and needs an internet connection and an API key.

## Setup on Windows

You need Python 3.12 installed. Open PowerShell in the project folder.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a file named `.env` in the project folder with your Gemini API key (get one free at
Google AI Studio). The app works without it, only the AI report needs it:

```
GEMINI_API_KEY=your_key_here
```

`.env` is git-ignored: never commit it.

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub (public).
2. On share.streamlit.io choose **Create app**, pick this repo, branch `master`, main file `app.py` (Python 3.12).
3. Under **Advanced settings -> Secrets** add `GEMINI_API_KEY = "your_key_here"`. The app reads it from
   `st.secrets` and falls back to the local `.env` file. Without a key everything works except the AI report.
4. Visitors can click **Try with sample data**. On the cloud the app runs in public-demo mode (detected
   automatically, or set `PUBLIC_DEMO = "1"`): each visitor's corrections, budgets and goals live only in their own
   session (`st.session_state`) and disappear when the tab closes, nothing is written to `corrections.csv`,
   `budgets.json` or `goals.json`, and **Retrain model** is disabled. Locally the app behaves as before.

## How to run

```powershell
venv\Scripts\activate
streamlit run app.py
```

Open the address Streamlit prints (usually http://localhost:8501), then upload a statement in the
sidebar. `data/transactions.csv` is a fake statement you can try straight away.

Run the tests (each file prints OK lines and finishes with "passed"):

```powershell
python test_parser.py
python test_merger.py
python test_categorizer.py
python test_analytics.py
python test_insights.py
python test_report.py
python test_goals.py
python test_theme.py
```

To rebuild the data and the model from scratch:

```powershell
python generate_data.py
python make_realistic_test.py
python categorizer.py
```

## Folder structure

```
Expensive analyzer/
|-- app.py                 Streamlit UI (5 pages)
|-- theme.py               look and feel: category emojis/colours, CSS, cards, chart styling
|-- .streamlit/config.toml dark theme (teal accent)
|-- parser.py              read + clean + mask a statement
|-- merger.py              combine several statements: duplicates, self transfers
|-- categorizer.py         rules + ML categorizer, training and evaluation
|-- analytics.py           totals and chart data for the dashboard
|-- insights.py            unusual spends, subscriptions, budgets, forecast
|-- goals.py               savings goals: status, reach date, suggested cuts
|-- report.py              builds aggregated stats and calls Gemini
|-- merchants.py           extra merchant names (training data + rules)
|-- people.py              first names / business words to spot person transfers
|-- generate_data.py       makes the synthetic training and demo data
|-- make_realistic_test.py makes the messy test set
|-- test_*.py              one test file per module
|-- requirements.txt
|-- data/
|   |-- training.csv           synthetic training data
|   |-- test_realistic.csv     messy test set (never used for training)
|   |-- transactions.csv       fake demo statement
|   `-- sample_bank_*.csv      two small fake statements with self transfers between them
|-- models/
|   |-- categorizer.joblib     trained model
|   `-- confusion_matrix*.png  evaluation images
`-- docs/limitations.md    detailed limitations of the categorizer
```

Created while you use the app (git-ignored, personal): `corrections.csv`, `budgets.json`, `goals.json`, `.env`.
