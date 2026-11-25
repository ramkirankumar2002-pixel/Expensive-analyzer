# Limitations of the categorizer

Numbers below come from the final model (Logistic Regression, trained on 2,398 rows) and the fixed
messy test set `data/test_realistic.csv` (436 rows). Re-run `python categorizer.py` to refresh them.

## 1. What the accuracy numbers mean

| Measure | Result | What it tells us |
|---|---|---|
| Held-out 20% split of the training data | about 100% | Only shows the pipeline works. |
| Messy test set, hybrid (rules + ML) | 68.6% | The realistic number. |
| Messy test set, ML only | 62.4% | What the model itself learned. |

- **The held-out score is not evidence of generalization.** The training data is generated from
  templates (`generate_data.py`), so test rows look like training rows.
- **The messy test set is the real measure, but it is not fully independent.** I wrote it myself, and
  it was also used to pick the winning model (Logistic Regression over Naive Bayes), so the 68.6% is
  slightly optimistic. A set of real, unseen statements would be a fairer test.
- **The messy set was kept unseen.** Its merchants and narration styles are not in the training data
  (checked by `test_training_data_is_varied`). That stays true only if nobody copies them in later.
- **The "unseen-merchant" cross-validation printed by `categorizer.py` is inflated.** After adding
  name variants (e.g. "Swiggy Order" next to "Swiggy"), related names land in different folds, so
  the score (about 88%) leaks. Do not quote it.

## 2. Known weaknesses of the ML model

- **New merchants are mostly missed.** On unseen merchants the ML model alone scores about 29%, and
  the hybrid about 37%.
- **New people are not recognised.** Transfers to people whose names are not in the training data
  score 0% from the ML model alone. It cannot tell a new name from a new merchant. The keyword rules
  ("imps", "neft", "p2p") rescue some of them: the hybrid reaches 46% on this group.
- **Too many rows fall back to "Others".** When the best guess has less than 40% probability
  (`CONFIDENCE_THRESHOLD`), the answer becomes "Others". This is a safe default, but "Others" has
  only 13% precision on the messy set: most rows labelled Others are really something else.
  Lowering the threshold would fix some of them and wrongly categorize others.
- **Short generic labels are only partly learned.** On labels from my own Excel sheet that the model
  never saw, the ML model alone is right for 9 of 23. It misses "Pension Fund", "SSS",
  "Healthcare" and "Cable", so the keyword rules carry those.

## 3. Limits of the keyword rules

- **Rules always win over the model**, so a wrong or too-broad rule cannot be fixed by retraining.
  The broad words I added for the Excel file ("home", "phone", "internet", "food") also apply to
  bank statements. For example "Home Centre" would be read as Bills.
- **Rules match whole words in a fixed order**, and the first category that matches wins. A narration
  with two keywords gets whichever category comes first in `KEYWORD_RULES`.
- **They are tuned to my data:** Indian merchants, UPI-style narrations and one expense-tracker
  spreadsheet. Statements from other countries or formats will mostly fall through to the ML model.

## 4. User corrections

- **Corrections override everything** (layer 0), but only for descriptions that match exactly after
  cleaning. A fix for "Swiggy Order 123" does not fix "Swiggy Instamart".
- **Corrections do not change the headline accuracy.** The messy-set score deliberately ignores them,
  so the before/after numbers show only what retraining changed in the model.
- **A single wrong correction is trusted completely.** There is no undo screen and no check for
  conflicting corrections, so a mistaken fix stays until `corrections.csv` is edited by hand.
- **Retraining on a handful of corrections changes the model very little.** The corrections layer is
  what makes a fix take effect immediately.

## 5. Data and scope

- **Training data is synthetic** (fake, template-based) apart from the user's corrections.
- **The category list is fixed** at ten categories (project rule), so anything outside them has to
  be forced into the nearest one or into "Others".
- **Credits are categorized like debits.** The parser records whether a row is money in or out, but
  the categorizer ignores it, so a refund from "Amazon" is labelled Shopping.
- **Privacy:** only masked descriptions are saved in `corrections.csv`. The file is git-ignored,
  but it still contains personal transaction text.

## 6. What would improve it most

1. Test on a few real, hand-labelled statements, and keep them out of training and model selection.
2. Collect real corrections from use and retrain on them. This is the cheapest source of realistic data.
3. Add a separate validation set for choosing the model, so the messy test set stays untouched.
4. Tune the confidence threshold and report precision and recall for "Others" separately.
