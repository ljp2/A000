import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

EPS = 1e-12

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
INPUT_CSV = "SPY_1min_RTH_full.csv"

MODEL_OUT = "spy_xgb.json"
FEATURE_COLS_OUT = "spy_feature_cols.json"
POLICY_OUT = "policy_thresholds.json"

# Target config
HORIZON = 2
DEADBAND_FRAC_ATR = 0.05

# XGBoost params (your baseline)
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

# Policy thresholds (start with what you calibrated; tune later)
POLICY = dict(
    LONG_ENTER=0.62,
    SHORT_ENTER=0.38,
    LONG_EXIT=0.52,
    SHORT_EXIT=0.48,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - m) / (sd + EPS)

def enforce_rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "16:00", inclusive="left")

# ------------------------------------------------------------
# Feature blocks (baseline 64 features)
# ------------------------------------------------------------
def add_session_keys(df: pd.DataFrame) -> pd.DataFrame:
    df["session_date"] = df.index.date
    return df

def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    mins = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    df["minute_of_session"] = mins.astype(np.int16)
    ang = 2.0 * np.pi * df["minute_of_session"] / 390.0
    df["sin_tod"] = np.sin(ang)
    df["cos_tod"] = np.cos(ang)
    df["is_opening_30m"] = (df["minute_of_session"] < 30).astype(np.int8)
    df["is_closing_30m"] = (df["minute_of_session"] >= 360).astype(np.int8)
    return df

def add_session_anchors(df: pd.DataFrame) -> pd.DataFrame:
    df["open_session"] = df.groupby("session_date")["open"].transform("first")
    df["hod"] = df.groupby("session_date")["high"].cummax()
    df["lod"] = df.groupby("session_date")["low"].cummin()
    return df

def add_session_cum_vwap(df: pd.DataFrame) -> pd.DataFrame:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    df["vwap_sess"] = (
        pv.groupby(df["session_date"]).cumsum()
        / (df["volume"].groupby(df["session_date"]).cumsum() + EPS)
    )
    return df

def add_normalizers(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    h = df["high"]
    l = df["low"]
    prev_c = c.shift(1)

    df["lr_1"] = np.log(c / prev_c)

    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))

    df["atr_20"] = ema(df["tr"], 20)
    df["atr_120"] = ema(df["tr"], 120)

    df["rv_20"] = df["lr_1"].rolling(20).std(ddof=0)
    df["rv_120"] = df["lr_1"].rolling(120).std(ddof=0)

    df["atr_ret_20"] = df["atr_20"] / (df["close"] + EPS)
    return df

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    for k in [1,2,3,5,10,15,30,60]:
        lr_k = np.log(df["close"] / df["close"].shift(k))
        df[f"ret_lr_{k}"] = lr_k
        df[f"ret_atr_{k}"] = lr_k / (df["atr_ret_20"] + EPS)

    df["lr_z_20"] = rolling_zscore(df["lr_1"], 20)
    df["lr_z_60"] = rolling_zscore(df["lr_1"], 60)
    df["mom_mean_20"] = df["lr_1"].rolling(20).mean()
    df["mom_std_20"] = df["lr_1"].rolling(20).std(ddof=0)

    ema_12 = ema(df["close"], 12)
    ema_48 = ema(df["close"], 48)
    df["ema_spread_atr"] = (ema_12 - ema_48) / (df["atr_20"] + EPS)
    df["ema12_slope_10_atr"] = (ema_12 - ema_12.shift(10)) / (df["atr_20"] + EPS)
    return df

def add_candle_anatomy(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    atr = df["atr_20"] + EPS

    df["range_atr"] = (h - l) / atr
    df["body_atr"] = (c - o) / atr
    df["body_abs_atr"] = (c - o).abs() / atr
    df["upper_wick_atr"] = (h - np.maximum(o, c)) / atr
    df["lower_wick_atr"] = (np.minimum(o, c) - l) / atr

    rng = (h - l) + EPS
    df["close_loc"] = (c - l) / rng
    df["body_to_range"] = (c - o).abs() / rng
    df["range_expand_20"] = df["range_atr"] / (df["range_atr"].rolling(20).mean() + EPS)
    return df

def add_vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    df["rv_ratio_20_120"] = df["rv_20"] / (df["rv_120"] + EPS)
    df["atr_ratio_20_120"] = df["atr_20"] / (df["atr_120"] + EPS)
    df["vov_60"] = df["rv_20"].rolling(60).std(ddof=0)
    return df

def add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    atr = df["atr_20"] + EPS

    # per-bar VWAP from file (vwp)
    df["dist_vwapbar_atr"] = (df["close"] - df["vwp"]) / atr
    df["dist_vwapbar_z_60"] = rolling_zscore(df["close"] - df["vwp"], 60)

    # session VWAP
    df["dist_vwapsess_atr"] = (df["close"] - df["vwap_sess"]) / atr
    df["dist_vwapsess_z_60"] = rolling_zscore(df["close"] - df["vwap_sess"], 60)
    df["vwapsess_slope_10_atr"] = (df["vwap_sess"] - df["vwap_sess"].shift(10)) / atr
    df["dist_open_atr"] = (df["close"] - df["open_session"]) / atr
    return df

def add_hod_lod_features(df: pd.DataFrame) -> pd.DataFrame:
    atr = df["atr_20"] + EPS
    df["dist_hod_atr"] = (df["hod"] - df["close"]) / atr
    df["dist_lod_atr"] = (df["close"] - df["lod"]) / atr

    day_rng = (df["hod"] - df["lod"]) + EPS
    df["pos_day_range"] = (df["close"] - df["lod"]) / day_rng
    df["day_range_atr"] = (df["hod"] - df["lod"]) / atr
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    v = df["volume"].astype(float)
    for n in [20,60]:
        mv = v.rolling(n).mean()
        sv = v.rolling(n).std(ddof=0)
        df[f"vol_ratio_{n}"] = v / (mv + EPS)
        df[f"vol_z_{n}"] = (v - mv) / (sv + EPS)

    df["signed_vol"] = v * (2.0 * df["close_loc"] - 1.0)
    df["signed_vol_ema_20"] = ema(df["signed_vol"], 20) / (v.rolling(20).mean() + EPS)
    df["volXrange"] = df["vol_ratio_20"] * df["range_atr"]
    df["volXdistVWAP"] = df["vol_ratio_20"] * df["dist_vwapsess_atr"]
    return df

def add_trade_count_features(df: pd.DataFrame) -> pd.DataFrame:
    tc = df["trade_count"].astype(float)
    for n in [20,60]:
        mt = tc.rolling(n).mean()
        st = tc.rolling(n).std(ddof=0)
        df[f"tc_ratio_{n}"] = tc / (mt + EPS)
        df[f"tc_z_{n}"] = (tc - mt) / (st + EPS)

    df["trades_per_vol"] = tc / (df["volume"].astype(float) + EPS)
    df["trades_per_vol_z_60"] = rolling_zscore(df["trades_per_vol"], 60)
    return df

def add_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    abs_step = (c - c.shift(1)).abs()
    for n in [20,60]:
        net = (c - c.shift(n)).abs()
        path = abs_step.rolling(n).sum()
        df[f"eff_ratio_{n}"] = net / (path + EPS)
    return df

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    fwd_lr = np.log(df["close"].shift(-HORIZON) / df["close"])
    df["fwd_lr_2"] = fwd_lr
    df["y"] = (fwd_lr > 0).astype(np.int8)

    db = DEADBAND_FRAC_ATR * df["atr_ret_20"]
    keep = fwd_lr.abs() > db
    df.loc[~keep, "y"] = np.nan
    return df

# ------------------------------------------------------------
# Build dataset
# ------------------------------------------------------------
def build_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.set_index("timestamp").sort_index()

    needed = ["open","high","low","close","volume","trade_count","vwp"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df[needed].astype(float)

    df = enforce_rth(df)
    df = add_session_keys(df)
    df = add_time_of_day(df)
    df = add_session_anchors(df)
    df = add_session_cum_vwap(df)
    df = add_normalizers(df)

    df = add_return_features(df)
    df = add_candle_anatomy(df)
    df = add_vol_regime(df)
    df = add_vwap_features(df)
    df = add_hod_lod_features(df)
    df = add_volume_features(df)
    df = add_trade_count_features(df)
    df = add_efficiency(df)

    df = add_target(df)

    return df

def finalize_xy(feat_df: pd.DataFrame):
    df2 = feat_df.dropna().copy()

    exclude = {
        "open","high","low","close","volume","trade_count","vwp",
        "open_session","hod","lod","vwap_sess","session_date",
        "tr","atr_20","atr_120","rv_20","rv_120","atr_ret_20","lr_1",
        "fwd_lr_2","y"
    }
    feature_cols = [c for c in df2.columns if c not in exclude]
    X = df2[feature_cols]
    y = df2["y"].astype(int)
    return X, y

# ------------------------------------------------------------
# Train and save
# ------------------------------------------------------------
def main():
    feat_df = build_features(INPUT_CSV)
    X, y = finalize_xy(feat_df)

    print("Training rows:", len(X))
    print("Num features :", X.shape[1])
    print("y balance    :", y.value_counts(normalize=True).to_dict())

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X, y)

    # Save model (production-safe format)
    model.save_model(MODEL_OUT)

    # Save feature order for production inference
    with open(FEATURE_COLS_OUT, "w") as f:
        json.dump(X.columns.tolist(), f)

    # Save policy thresholds (optional, but useful)
    with open(POLICY_OUT, "w") as f:
        json.dump(POLICY, f, indent=2)

    print(f"\nSaved model:        {MODEL_OUT}")
    print(f"Saved feature cols: {FEATURE_COLS_OUT}")
    print(f"Saved policy:       {POLICY_OUT}")

if __name__ == "__main__":
    main()
