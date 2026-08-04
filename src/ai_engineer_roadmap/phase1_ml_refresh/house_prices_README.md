# House Prices Regression — ML Foundations Refresh

Refresher exercise for Phase 1 of the AI Engineering roadmap. Goal: apply regression
evaluation metrics (MAE, RMSE, R²) and cross-validation correctly, with a hand-picked
focused feature set rather than all ~80 raw columns.

## Dataset
1460 houses, ~80 raw columns (Kaggle House Prices). Target: `SalePrice`, right-skewed
(mean $181k > median $163k, range $35k–$755k) — a handful of expensive houses pull the
average up.

## Feature selection
Rather than cleaning all ~80 columns (several >90% missing, e.g. `PoolQC`, `Alley`,
`Fence`), hand-picked 9 features with strong intuitive/known relevance to price:
`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`,
`YearRemodAdd`, `LotArea`, `Neighborhood`. None of these had missing values, so no
imputation was needed. `Neighborhood` (categorical, no natural order) was one-hot encoded.

## Models compared

| Metric | Linear Regression | Random Forest (200 trees) |
|---|---|---|
| MAE | $21,570 | $17,933 |
| RMSE | $35,960 | $28,215 |
| R² (test) | 0.831 | 0.896 |
| 5-fold CV Mean R² | 0.802 | 0.844 |
| 5-fold CV Std | 0.052 | 0.037 |

Random Forest outperformed Linear Regression on every metric, and was also more stable
across folds. This is the opposite result from the Titanic classification project (where
Random Forest didn't clearly beat Logistic Regression and was less stable) — a reminder
that model choice depends on the problem, not a fixed rule. House price likely has
genuine non-linear relationships (e.g. the price jump from quality 8→9 isn't the same
dollar amount as 3→4), which linear regression structurally can't capture.

## MAE vs RMSE gap
RMSE ($35,960) is notably higher than MAE ($21,570) for Linear Regression. Since RMSE
squares errors before averaging, this gap signals a subset of predictions with large
errors — most likely the higher-priced houses at the tail of the right-skewed
distribution, which a linear model struggles to fit as well as the bulk of mid-range
houses.

## Random Forest feature importance
```
OverallQual   0.576
GrLivArea     0.170
TotalBsmtSF   0.086
LotArea       0.045
YearBuilt     0.035
GarageCars    0.028
YearRemodAdd  0.025
FullBath      0.011
```
`OverallQual` alone accounts for more than half the model's decision weight — overall
material/finish quality is by far the strongest driver of price in this feature set,
followed by living area and basement size.

## Cross-validation note
One fold scored notably lower (R² = 0.702 for Linear Regression) than the other four
(0.70–0.85 range) — a flag that a particular data slice (likely containing more
high-price outliers) is harder to predict. Random Forest's CV scores were both higher
and less spread out, reinforcing that it handles this dataset's skew better.

## Key takeaways
- A focused, intuitive feature subset (9 columns) achieved R² = 0.896 without needing to
  clean all ~80 raw columns — feature selection can matter more than feature quantity.
- Cross-validation caught fold-to-fold instability that a single train/test split would
  have missed.
- Random Forest clearly won here, in contrast to the Titanic project — reinforces testing
  multiple models rather than assuming one approach is always better.

## What I'd try next (not built here — out of scope for a refresher exercise)
- Log-transform `SalePrice` before modeling, given its right skew
- Use more of the ~80 available columns with proper missing-value handling per column
- Hyperparameter tuning on Random Forest (max_depth, min_samples_leaf)
