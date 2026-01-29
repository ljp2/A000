# SPY Intraday ML System v1 — One‑Page Summary

## Objective
Use an XGBoost classifier on **1‑minute SPY RTH bars** to estimate the probability of an **up move over the next 2 minutes**, and convert that probability into **state‑dependent trading actions** (flat / long / short management).

---

## Data & Target

### Data
- Instrument: **SPY**
- Bars: **1‑minute OHLCV**
- Additional fields: `trade_count`, `vwp` (per‑bar VWAP)
- Session: **RTH only (09:30–16:00 ET)**

### Target
- Horizon: **+2 minutes**
- Forward return:  
  `fwd_lr_2 = ln(close[t+2] / close[t])`
- Label:  
  `y = 1 if fwd_lr_2 > 0 else 0`
- **Deadband filter**: drop samples where  
  `|fwd_lr_2| <= 0.05 × (ATR20 / close)`

Purpose: reduce label noise in ultra‑short horizons.

---

## Feature Schema (Baseline v1)

All features are computed using information available **up to bar t**.

### Session‑Reset Features (reset each RTH day)
Computed via `session_date = index.date` grouping:
- `open_session`
- `hod`, `lod`
- `vwap_sess` (cumulative VWAP)
- Time‑of‑day:
  - `minute_of_session`
  - `sin_tod`, `cos_tod`
  - `is_opening_30m`, `is_closing_30m`

### Volatility & Normalization
- `lr_1`
- `TR`
- `ATR_20`, `ATR_120`
- `rv_20`, `rv_120`
- `rv_ratio_20_120`
- `atr_ratio_20_120`
- `vov_60`

### Returns & Momentum
For k ∈ {1,2,3,5,10,15,30,60}:
- `ret_lr_k`
- `ret_atr_k`

Additional:
- `lr_z_20`, `lr_z_60`
- `mom_mean_20`, `mom_std_20`
- `ema_spread_atr`
- `ema12_slope_10_atr`

### Candle Structure (ATR‑scaled)
- `range_atr`
- `body_atr`, `body_abs_atr`
- `upper_wick_atr`, `lower_wick_atr`
- `close_loc`
- `body_to_range`
- `range_expand_20`

### VWAP & Anchors
- Session VWAP:
  - `dist_vwapsess_atr`
  - `dist_vwapsess_z_60`
  - `vwapsess_slope_10_atr`
- Per‑bar VWAP:
  - `dist_vwapbar_atr`
  - `dist_vwapbar_z_60`
- Open anchor:
  - `dist_open_atr`

### Day‑Range Context
- `dist_hod_atr`, `dist_lod_atr`
- `pos_day_range`
- `day_range_atr`

### Volume & Microstructure Proxies
- `vol_ratio_20/60`, `vol_z_20/60`
- `signed_vol`, `signed_vol_ema_20`
- `volXrange`, `volXdistVWAP`
- `tc_ratio_20/60`, `tc_z_20/60`
- `trades_per_vol`, `trades_per_vol_z_60`
- `eff_ratio_20/60`

**Explicitly excluded from v1**:
- Polynomial slope/curvature features
- Acceleration features
- Heiken‑Ashi features (experimental only)

---

## Model Training & Validation

- Model: **XGBClassifier**
- Tree method: `hist`
- Regularized, shallow trees
- Validation: **purged, day‑based CV**
- Baseline performance:
  - **AUC ≈ 0.518**
  - Weak but real ranking signal (expected for SPY 1‑min, 2‑min horizon)

Primary value comes from **thresholding and position management**, not raw accuracy.

---

## Probability Calibration & Thresholds

Using out‑of‑fold (OOF) probabilities:

- Long signal: `p_up ≥ T`
- Short signal: `p_up ≤ 1 − T`

Typical operating range:
- `T ≈ 0.60 – 0.62`

Default policy:
- Entry:
  - `LONG_ENTER = 0.62`
  - `SHORT_ENTER = 0.38`
- Exit (hysteresis):
  - `LONG_EXIT = 0.52`
  - `SHORT_EXIT = 0.48`

---

## Trading Policy (State Machine)

At each bar close:

- **FLAT**
  - `GO_LONG` / `GO_SHORT` / `WAIT`
- **LONG**
  - `SELL_TO_CLOSE` / `HOLD`
- **SHORT**
  - `BUY_TO_CLOSE` / `HOLD`

Decision at bar close, execution at **next bar open**.

---

## Production Architecture

1. 1‑minute bar feed
2. Rolling buffer (~600 bars, preload last ~200 RTH bars)
3. Feature engine
   - Session features reset by `session_date`
   - Rolling ATR/RV carried across sessions
   - Missing features neutral‑filled during warm‑up
4. XGBoost inference (`spy_xgb.json`)
5. Policy engine (state machine)
6. Execution & risk layer

### Warm‑up rules
- Minimum bars before trading: **~30**
- Neutral fill:
  - ratios → 1
  - z‑scores / distances / returns → 0
  - flags → 0

---

## Saved Artifacts

From training:
- `spy_xgb.json` — trained XGBoost model
- `spy_feature_cols.json` — exact feature order
- `policy_thresholds.json` — entry/exit thresholds

---

**Status**: This defines **SPY Intraday ML System v1 (frozen baseline)**.
