# Personal Expense Analyzer — semester project (AI-based analytical system)

## Goal
User uploads a bank/UPI statement (CSV/Excel). The app parses it, auto-categorizes
transactions with ML, shows a dashboard, and writes an AI summary of the spending.

## Stack
- Python 3.12, Streamlit (UI), Pandas, Scikit-learn, Plotly
- LLM: Gemini API (key in .env as GEMINI_API_KEY, never commit it)
- python-dotenv loads the .env file

## Categories (fixed, do not change)
Food, Groceries, Travel, Shopping, Bills, Rent, Entertainment, Health,
Transfers, Others

## Rules
- Keep the code simple and well-commented — I must explain every part in my viva.
- Mask account numbers and UPI IDs when parsing.
- Send only aggregated stats to the LLM, never raw transactions.
- Write a small test for each parser/ML function and run it.
- Don't add libraries beyond requirements.txt without asking me.

## Run
    venv\Scripts\activate
    streamlit run app.py
