# Titanic Survival Classification — ML Foundations Refresh

Refresher exercise for Phase 1 of the AI Engineering roadmap. Goal: apply core evaluation
metrics (precision, recall, F1, ROC-AUC) and cross-validation correctly, not to build a
resume-worthy project — Titanic is a well-worn dataset used here purely for reps on process.

## Dataset
891 passengers, 12 raw columns. Target: `Survived` (0/1). Class balance: 61.6% did not
survive, 38.4% survived — mildly imbalanced, not severe.

## Data cleaning decisions
| Column | Issue | Decision | Reasoning |
|---|---|---|---|
| `Cabin` | 77% missing | Replaced with binary `has_cabin` flag | Too much missing to impute reliably; missingness itself is a weak signal (correlates with class) |
| `Age` | 20% missing | Median imputation | Median is robust to outliers; retains an important predictive feature |
| `Embarked` | 2 missing | Dropped those 2 rows | Negligible impact, not worth imputing |
| `Sex` | categorical | Mapped to 0/1 | Binary category |
| `Embarked` | categorical, no order | One-hot encoded (`drop_first=True`) | Avoids implying a false ranking between ports |
| `PassengerId`, `Name`, `Ticket` | no direct signal / raw text | Dropped | Out of scope for this baseline |

## Models compared

| Metric | Logistic Regression | Random Forest (200 trees) |
|---|---|---|
| Test ROC-AUC | 0.861 | 0.841 |
| 5-fold CV Mean ROC-AUC | 0.849 | 0.860 |
| 5-fold CV Std | 0.016 | 0.040 |

Logistic Regression scored higher on the single test split; Random Forest edged ahead on
CV mean but with more than double the standard deviation across folds. On a small dataset
(~890 rows), the added complexity of Random Forest didn't translate into a clear win — the
higher variance across folds outweighs its lower bias here. This is bias-variance tradeoff
playing out directly: a simpler model can be competitive, or better, when data is limited.

## Random Forest feature importance
```
Sex          0.271
Fare         0.251
Age          0.240
Pclass       0.072
SibSp        0.049
has_cabin    0.044
Parch        0.037
Embarked_S   0.023
Embarked_Q   0.014
```
`Sex`, `Fare`, and `Age` dominate — consistent with the historical "women and children
first" evacuation pattern, and wealthier/higher-class passengers having better survival
odds.

## Full classification report (Logistic Regression, test set)
```
              precision    recall  f1-score   support
           0       0.84      0.88      0.86       110
           1       0.79      0.72      0.75        68
    accuracy                           0.82       178
```

## Key takeaways
- Accuracy alone (0.82) would have hidden the class-level differences — recall for the
  minority class (survived, class 1) is meaningfully lower (0.72) than precision (0.79),
  meaning the model misses more real survivors than it falsely flags.
- ROC-AUC (0.86) confirms strong ranking ability independent of the classification
  threshold used.
- Cross-validation confirmed the single test-split result wasn't a lucky split — Logistic
  Regression's CV std (0.016) shows genuinely stable performance across folds.
- More model complexity isn't automatically better — worth testing, not assuming.

## What I'd try next (not built here — out of scope for a refresher exercise)
- Per-class median age imputation (by `Pclass`) instead of a single global median
- Feature engineering: extract titles (Mr/Mrs/Miss/Master) from the `Name` column before
  dropping it — likely a strong signal
- Hyperparameter tuning on Random Forest (may close the CV std gap)
