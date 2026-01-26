# Margin Requirements (Futu/MooMoo) — Analysis Plan

## Goal
Infer the underlying patterns/rules Futu/MooMoo use to set **per-stock margin requirements** for US-listed securities, using `Regression_data.csv` (one row per ticker).

**Targets (Y):**
- `im_long`, `im_short` (initial margin, long/short)
- `mm_long`, `mm_short` (maintenance margin, long/short)

**Candidate drivers (X):**
- Price/liquidity: `price`, `ADT`, `turnover_rate`, `market_cap`
- Risk/return: `var_10d_pct`, `volatility_annual_pct`, `mean_return_annual_pct`
- Data quality: `n_observations`

Deliverables:
- EDA plots that reveal structure (nonlinearities, buckets, thresholds).
- Interpretable model(s) explaining margin settings (ideally “rules of thumb”).
- A validated baseline regression + a “rule-extraction” model for piecewise logic.

---

## 0) Decide conventions (before any modeling)
- Confirm units/scales:
  - Margin columns are stored in **percentage points** (e.g., `30` means **30%**).
  - Decide whether to model targets in percent points (`30`) or decimals (`0.30`) and be consistent (often: fit on decimals, report in %).
- Modeling approach:
  - Start with **4 separate runs** (one per target) for clarity and diagnostics.
  - Then move to a **shared model** that learns common structure across targets (better generalization + more data).
- Define derived quantities to analyze:
  - Spread: `im_long - mm_long`, `im_short - mm_short`
  - Short-vs-long: `im_short - im_long`, `mm_short - mm_long`
  - Ratios (optional): `im_short / im_long` etc.

---

## 1) Load + sanity checks
1. Parse `Regression_data.csv` with explicit dtypes.
2. Validate:
   - unique `ticker` per row (or dedupe rules if not).
   - missing values per column; decide drop vs impute.
   - obvious invalid values (negative vol/market cap, zero shares implied, etc.).
3. Quick summaries:
   - distributions for each target and each feature.
   - min/median/max by column; count extreme outliers.

---

## 2) Feature engineering (for stability + interpretability)
Create model-ready features (keep raw columns too):
- Log transforms for scale-heavy variables (common in finance):
  - `log_price = log(price)`, `log_ADT = log(ADT)`, `log_mktcap = log(market_cap)`
  - `log_turnover = log(turnover_rate)` (guard for zeros)
- Winsorize heavy tails (e.g., top/bottom 0.5–1%) or use robust models.
- Standardize numeric features for linear models (z-score).
- Optional: interaction terms to test hypotheses:
  - risk × liquidity (e.g., `var_10d_pct / log_ADT`, `volatility_annual_pct / log_mktcap`)
- Data quality controls:
  - include `n_observations` or filter low-`n_observations` rows and compare results.

---

## 3) EDA: visualize relationships and check “bucketed rules”
### 3.1 Single-variable relationships (fast signal)
For each target `Y ∈ {im_long, im_short, mm_long, mm_short}`:
- Scatter plots with smoothing (LOWESS):
  - `Y` vs `var_10d_pct`, `volatility_annual_pct`, `ADT`, `turnover_rate`, `market_cap`, `price`
  - Use log scales for `ADT/market_cap/price` plots
- Binned plots:
  - bucket each X into quantiles and plot mean/median of `Y` with error bars
- Distribution plots:
  - histogram/KDE of `Y` + check for discrete levels (e.g., 25%, 30%, 50%, 100%)

### 3.2 Multivariate structure
- Correlation heatmap (Spearman preferred for monotone/nonlinear).
- Pairplot for a subset of features (or `Y` vs top 3–4 X’s).
- Check collinearity:
  - `ADT`, `market_cap`, `turnover_rate`, `price` often correlate strongly; plan for this in modeling.

### 3.3 “Rule-like” diagnostics
- Look for step functions:
  - plot sorted `Y` and see if it clusters at a few levels.
- For each potential threshold variable (e.g., `price`, `ADT`, `var_10d_pct`):
  - plot `Y` vs X with vertical lines at candidate cut points (quantiles), visually inspect jumps.

---

## 4) Baseline statistical models (interpretable regression)
Run separate models per target first, then a shared model.

### 4.1 Linear baselines
- OLS on engineered features (with robust standard errors).
- Regularized regression (Lasso/Ridge/ElasticNet) to manage collinearity and select drivers.
- Evaluate with cross-validation (RMSE/MAE + R²).

### 4.2 Robust alternatives (if outliers/heavy tails)
- Huber regression / RANSAC (if many outliers).
- Quantile regression (useful if broker margins behave like upper quantiles of risk).

Key outputs:
- Coefficients with signs vs intuition (risk ↑ → margin ↑, liquidity ↑ → margin ↓).
- Partial residual plots to catch remaining nonlinearity.

### 4.3 Shared model (single model for all 4 targets)
Reshape to “long” format (4 rows per ticker):
- `ticker`, all X features, plus categorical columns:
  - `margin_kind ∈ {im, mm}`
  - `side ∈ {long, short}`
  - `target_value` (the corresponding margin)

Fit a single model that includes target indicators and allows different baselines (and optionally different sensitivities):
- Minimum: `target_value ~ X + margin_kind + side`
- Better: add interactions (regularized): `target_value ~ X + margin_kind + side + X:margin_kind + X:side`

Compare shared vs separate:
- same CV scheme, report per-target metrics and overall metrics
- check whether interactions materially improve fit (if not, you have strong evidence of a mostly-shared rule)

---

## 5) Nonlinear + “rule extraction” models (to uncover piecewise logic)
If EDA suggests buckets/thresholds:
- Decision tree regressor (shallow, e.g., depth 3–5) to get human-readable rules.
- Random forest / gradient boosting (better predictive power).
- Extract explanations:
  - feature importance
  - partial dependence plots (PDP) / ICE curves
  - SHAP values (if using boosting models)

Optional (if you strongly suspect monotonic rules):
- Fit a monotonic gradient boosting model with constraints (risk metrics monotone ↑, liquidity monotone ↓) and compare.

---

## 6) Model diagnostics + validation
- Train/test split + K-fold CV.
- Residual analysis:
  - errors vs fitted values (heteroscedasticity)
  - errors vs key features (misspecification)
- Stability checks:
  - re-fit after removing extreme outliers
  - re-fit on high `n_observations` subset
- Compare long vs short targets:
  - do the same drivers explain both?
  - is short margin driven by additional nonlinearities?

---

## 7) Translate models into “broker-style” rules
Goal: produce a small set of statements like:
- “If `var_10d_pct > A` and `ADT < B`, margin increases to ~C%.”
- “Below a `price` threshold, margins jump.”

Steps:
- Use the shallow tree as the candidate ruleset.
- Validate each rule:
  - coverage (% of tickers affected)
  - average target value within leaf vs outside
  - error metrics for the rule-based approximation
- Summarize rules separately for:
  - `im_long`, `mm_long`, `im_short`, `mm_short`

---

## 8) Reporting + next iteration
- Make a 1–2 page writeup:
  - top drivers (by consistent importance across models)
  - key thresholds/buckets (with plots)
  - where the model fails (tickers with large residuals; investigate common traits)
- Iterate:
  - add missing features if needed (sector, borrow availability, price history length, gap risk, halted status, etc.).

---

## Suggested “first run” checklist (fastest path to insight)
1. Confirm units and whether `Y` is bucketed (discrete).
2. Make 6–8 scatter+LOWESS plots: `Y` vs `var_10d_pct`, `volatility_annual_pct`, `log_ADT`, `log_mktcap`, `price`.
3. Fit ElasticNet and a depth-4 decision tree per target; compare.
4. Extract 3–6 candidate rules from the tree; validate coverage + error.
