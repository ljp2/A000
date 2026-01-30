# Baseline-v2 Model Summary (SPY Intraday ML)

## Overview
- **Instrument:** SPY
- **Data:** 1-minute bars, Regular Trading Hours (09:30–16:00 ET)
- **Prediction Horizon:** 2 minutes
- **Model Version:** baseline-v2
- **Status:** Frozen, production-ready for paper trading

Baseline-v2 differs from baseline-v1 **only on the label side** via a Regime-Conditional Deadband (RCD).  
The feature set, model architecture, and training procedure are unchanged.

---

## Model Architecture
- **Model:** XGBoost Classifier
- **Objective:** binary:logistic
- **Tree Method:** hist
- **Estimators:** 500
- **Max Depth:** 4
- **Learning Rate:** 0.05
- **Subsample:** 0.8
- **Column Subsample:** 0.8
- **Min Child Weight:** 50
- **L2 Regularization (λ):** 1.0
- **Random Seed:** 42

Saved model:
- `spy_xgb_v2.json`

---

## Feature Set
- **Total Features:** 64
- **Change vs v1:** None (feature set frozen)

### Feature Families
- Multi-horizon log returns and momentum
- ATR & realized volatility normalization
- VWAP distance (bar-level and session-level)
- Candle anatomy & range expansion
- Volume & trade-count regimes
- Intraday time-of-day encoding
- Efficiency / path-dependence metrics

Feature order stored in:
- `spy_feature_cols_v2.json`

---

## Label Definition (Key v2 Change)
### Regime-Conditional Deadband (RCD)

- **Base Deadband:** 0.05 × ATR (log-return space)
- **Volatility Regime Metric:** `rv_ratio_20_120`

| Regime | Condition | Deadband Multiplier |
|------|----------|---------------------|
| Low RV | < 0.90 | 0.85× |
| Normal RV | 0.90 – 1.10 | 1.00× |
| High RV | > 1.10 | 1.25× |

- **Target:** Direction of forward 2-minute log return
- **Purpose:** Adaptive noise filtering by volatility regime
- **Effect:** Improved probability calibration and cost robustness

---

## Cross-Validation Setup
- **Method:** Purged day-based K-Fold
- **Folds:** 5
- **Purge Window:** ±2 minutes per validation day
- **Training Rows:** ~37,200
- **OOF Rows:** ~36,800
- **Class Balance:** ~50.7% long / 49.3% short

---

## Cross-Validated Performance (OOF)
- **Mean AUC:** ~0.519
- **Mean Logloss:** ~0.698
- **Fold Stability:** Consistent across all folds

---

## Threshold Performance (No Transaction Cost)
Edge improves monotonically with confidence threshold.

| Threshold | Hit Rate | Avg Log Return |
|---------:|---------:|---------------:|
| 0.56 | ~0.524 | 1.3e-05 |
| 0.58 | ~0.530 | 1.6e-05 |
| 0.59 | ~0.538 | 2.1e-05 |
| 0.60 | ~0.542 | 2.4e-05 |

---

## Cost-Adjusted Results
- **Assumed Cost:** 1 bp round-trip (log-return space)
- **Cost-Positive Region:** ≥ 0.55
- **Best Trade-Off Zone:** 0.58 – 0.60
- **Conclusion:** Edge survives realistic intraday costs

---

## Trading Policy (Current)
Stored in:
- `policy_thresholds_v2.json`

```json
{
  "LONG_ENTER": 0.59,
  "SHORT_ENTER": 0.41,
  "LONG_EXIT": 0.52,
  "SHORT_EXIT": 0.48
}
