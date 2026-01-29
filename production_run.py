import json
import numpy as np
import pandas as pd
from enum import IntEnum
from typing import Optional, Dict, Tuple

from xgboost import XGBClassifier

EPS = 1e-12

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "spy_xgb.json"
FEATURE_COLS_PATH = "spy_feature_cols.json"

# Warm-up: minimum bars required before we even attempt prediction
MIN_BARS = 30

# Buffer length: keep enough history for 120-bar features + safety
MAX_BUFFER_BARS = 600

# Policy thresholds (start with symmetric entry, mild hysteresis exits)
LONG_ENTER = 0.62
SHORT_ENTER = 0.38
LONG_EXIT = 0.52
SHORT_EXIT = 0.48


# ============================================================
# POSITION / POLICY
# ============================================================
class Pos(IntEnum):
    FLAT = 0
    LONG = 1
    SHORT = -1


def action_policy(p_up: float, pos: Pos,
                  long_enter: float = LONG_ENTER,
                  short_enter: float = SHORT_ENTER,
                  long_exit: float = LONG_EXIT,
                  short_exit: float = SHORT_EXIT) -> str:
    """
    State-dependent policy:
      FLAT  -> GO_LONG / GO_SHORT / WAIT
      LONG  -> SELL_TO_CLOSE / HOLD
      SHORT -> BUY_TO_CLOSE / HOLD
    """
    if pos == Pos.FLAT:
        if p_up >= long_enter:
            return "GO_LONG"
        if p_up <= short_enter:
            return "GO_SHORT"
        return "WAIT"

    if pos == Pos.LONG:
        return "SELL_TO_CLOSE" if p_up <= long_exit else "HOLD"

    if pos == Pos.SHORT:
        return "BUY_TO_CLOSE" if p_up >= short_exit else "HOLD"

    raise ValueError("Unknown position state")


# ============================================================
# MODEL LOADING
# ============================================================
def load_model_and_features(model_path: str, feature_cols_path: str) -> Tuple[XGBClassifier, list]:
    model = XGBClassifier()
    model.load_model(model_path)

    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError("Feature columns file is invalid or empty.")

    return model, feature_cols


# ============================================================
# FEATURE ENGINE HELPERS
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rolling_z(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return (s - m) / (sd + EPS)


def enforce_rth(df: pd.DataFrame) -> pd.DataFrame:
    # assumes timestamp is already NY time tz-aware or tz-naive NY local
    return df.between_time("09:30", "16:00", inclusive="left")


def fill_neutral_features(x: pd.Series) -> pd.Series:
    """
    Neutral-fill missing values so the model can run during warm-up.
    Heuristics:
      - ratios -> 1.0
      - zscores/distances/returns/slopes/oscillators -> 0.0
      - flags -> 0.0
      - sin/cos -> 0.0
      - fallback -> 0.0
    """
    x = x.copy()
    for col in x.index:
        if pd.notna(x[col]):
            continue

        if "ratio" in col:
            x[col] = 1.0
        elif ("_z_" in col) or col.startswith("dist_") or col.startswith("ret_") or ("slope" in col) or ("mom_" in col) or ("eff_ratio" in col) or ("vov" in col) or ("rv_" in col) or ("atr_" in col):
            x[col] = 0.0
        elif col.startswith("is_"):
            x[col] = 0.0
        elif col in ("sin_tod", "cos_tod"):
            x[col] = 0.0
        else:
            x[col] = 0.0
    return x


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Session-reset features (computed per session_date).
    Works even if df contains yesterday + today.
    """
    df = df.copy()
    df["session_date"] = df.index.date

    # Session open, HOD/LOD
    df["open_session"] = df.groupby("session_date")["open"].transform("first")
    df["hod"] = df.groupby("session_date")["high"].cummax()
    df["lod"] = df.groupby("session_date")["low"].cummin()

    # Session VWAP (cumulative)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    df["vwap_sess"] = (
        pv.groupby(df["session_date"]).cumsum()
        / (df["volume"].groupby(df["session_date"]).cumsum() + EPS)
    )

    # Time-of-day encodings
    mins = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    df["minute_of_session"] = mins.astype(np.int16)
    ang = 2.0 * np.pi * df["minute_of_session"] / 390.0
    df["sin_tod"] = np.sin(ang)
    df["cos_tod"] = np.cos(ang)
    df["is_opening_30m"] = (df["minute_of_session"] < 30).astype(np.int8)
    df["is_closing_30m"] = (df["minute_of_session"] >= 360).astype(np.int8)

    return df


def compute_features_from_buffer(buf: pd.DataFrame, feature_cols: list) -> Optional[pd.Series]:
    """
    Compute a single feature vector for the most recent bar in buf.

    buf columns required:
      open, high, low, close, volume, trade_count, vwp

    Returns:
      pd.Series indexed by feature_cols (exact order), or None if insufficient data.
    """
    if buf is None or len(buf) < MIN_BARS:
        return None

    df = buf.copy().sort_index()

    # Only RTH in production (recommended). If your feed includes non-RTH, enforce it here:
    df = enforce_rth(df)
    if len(df) < MIN_BARS:
        return None

    # Session-reset features (grouped by day)
    df = add_session_features(df)

    # ------------------------------------------------------------
    # Rolling features (carry across sessions via buffer)
    # Optional: break at session boundary to avoid close->open gap contamination
    # ------------------------------------------------------------
    prev_c = df["close"].shift(1)
    session_change = df["session_date"] != df["session_date"].shift(1)
    # break gap: treat prev close at new session as current close -> lr_1 = 0, TR gap suppressed
    prev_c = prev_c.where(~session_change, df["close"])

    df["lr_1"] = np.log(df["close"] / (prev_c + EPS))

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_c).abs()
    tr3 = (df["low"] - prev_c).abs()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))

    df["atr_20"] = ema(df["tr"], 20)
    df["atr_120"] = ema(df["tr"], 120)
    df["atr_ret_20"] = df["atr_20"] / (df["close"] + EPS)

    df["rv_20"] = df["lr_1"].rolling(20).std(ddof=0)
    df["rv_120"] = df["lr_1"].rolling(120).std(ddof=0)

    # Returns
    for k in [1, 2, 3, 5, 10, 15, 30, 60]:
        lr_k = np.log(df["close"] / (df["close"].shift(k) + EPS))
        df[f"ret_lr_{k}"] = lr_k
        df[f"ret_atr_{k}"] = lr_k / (df["atr_ret_20"] + EPS)

    # Z-scores & moments
    df["lr_z_20"] = rolling_z(df["lr_1"], 20)
    df["lr_z_60"] = rolling_z(df["lr_1"], 60)
    df["mom_mean_20"] = df["lr_1"].rolling(20).mean()
    df["mom_std_20"] = df["lr_1"].rolling(20).std(ddof=0)

    # EMA spread and slope
    ema12 = ema(df["close"], 12)
    ema48 = ema(df["close"], 48)
    df["ema_spread_atr"] = (ema12 - ema48) / (df["atr_20"] + EPS)
    df["ema12_slope_10_atr"] = (ema12 - ema12.shift(10)) / (df["atr_20"] + EPS)

    # Candle anatomy
    atr = df["atr_20"] + EPS
    df["range_atr"] = (df["high"] - df["low"]) / atr
    df["body_atr"] = (df["close"] - df["open"]) / atr
    df["body_abs_atr"] = (df["close"] - df["open"]).abs() / atr

    df["upper_wick_atr"] = (df["high"] - np.maximum(df["open"], df["close"])) / atr
    df["lower_wick_atr"] = (np.minimum(df["open"], df["close"]) - df["low"]) / atr

    rng = (df["high"] - df["low"]) + EPS
    df["close_loc"] = (df["close"] - df["low"]) / rng
    df["body_to_range"] = (df["close"] - df["open"]).abs() / rng

    df["range_expand_20"] = df["range_atr"] / (df["range_atr"].rolling(20).mean() + EPS)

    # VWAP features (session)
    df["dist_vwapsess_atr"] = (df["close"] - df["vwap_sess"]) / atr
    df["dist_vwapsess_z_60"] = rolling_z(df["close"] - df["vwap_sess"], 60)
    df["vwapsess_slope_10_atr"] = (df["vwap_sess"] - df["vwap_sess"].shift(10)) / atr
    df["dist_open_atr"] = (df["close"] - df["open_session"]) / atr

    # VWAP features (per-bar vwp from feed)
    df["dist_vwapbar_atr"] = (df["close"] - df["vwp"]) / atr
    df["dist_vwapbar_z_60"] = rolling_z(df["close"] - df["vwp"], 60)

    # HOD/LOD + day range context
    df["dist_hod_atr"] = (df["hod"] - df["close"]) / atr
    df["dist_lod_atr"] = (df["close"] - df["lod"]) / atr
    day_rng = (df["hod"] - df["lod"]) + EPS
    df["pos_day_range"] = (df["close"] - df["lod"]) / day_rng
    df["day_range_atr"] = (df["hod"] - df["lod"]) / atr

    # Volume features + interactions
    v = df["volume"].astype(float)
    for n in [20, 60]:
        mv = v.rolling(n).mean()
        sv = v.rolling(n).std(ddof=0)
        df[f"vol_ratio_{n}"] = v / (mv + EPS)
        df[f"vol_z_{n}"] = (v - mv) / (sv + EPS)

    df["signed_vol"] = v * (2.0 * df["close_loc"] - 1.0)
    df["signed_vol_ema_20"] = ema(df["signed_vol"], 20) / (v.rolling(20).mean() + EPS)
    df["volXrange"] = df["vol_ratio_20"] * df["range_atr"]
    df["volXdistVWAP"] = df["vol_ratio_20"] * df["dist_vwapsess_atr"]

    # Trade count features
    tc = df["trade_count"].astype(float)
    for n in [20, 60]:
        mt = tc.rolling(n).mean()
        st = tc.rolling(n).std(ddof=0)
        df[f"tc_ratio_{n}"] = tc / (mt + EPS)
        df[f"tc_z_{n}"] = (tc - mt) / (st + EPS)

    df["trades_per_vol"] = tc / (v + EPS)
    df["trades_per_vol_z_60"] = rolling_z(df["trades_per_vol"], 60)

    # Regime
    df["rv_ratio_20_120"] = df["rv_20"] / (df["rv_120"] + EPS)
    df["atr_ratio_20_120"] = df["atr_20"] / (df["atr_120"] + EPS)
    df["vov_60"] = df["rv_20"].rolling(60).std(ddof=0)

    # Efficiency
    abs_step = (df["close"] - df["close"].shift(1)).abs()
    for n in [20, 60]:
        net = (df["close"] - df["close"].shift(n)).abs()
        path = abs_step.rolling(n).sum()
        df[f"eff_ratio_{n}"] = net / (path + EPS)

    # OPTIONAL accel features (only if your trained FEATURE_COLS contain them)
    if "ema_spread_accel_5" in feature_cols:
        df["ema_spread_accel_5"] = df["ema_spread_atr"] - df["ema_spread_atr"].shift(5)
    if "vwapsess_pull_accel_5" in feature_cols:
        df["vwapsess_pull_accel_5"] = df["dist_vwapsess_atr"] - df["dist_vwapsess_atr"].shift(5)

    # Last-row feature vector in the exact training order
    last = df.iloc[-1]
    x = last.reindex(feature_cols)

    # Neutral fill missing values (warm-up safe)
    x = fill_neutral_features(x)

    return x


# ============================================================
# BUFFER / STREAM HANDLER
# ============================================================
def append_bar_to_buffer(buf: pd.DataFrame, bar: Dict) -> pd.DataFrame:
    """
    bar keys required:
      timestamp, open, high, low, close, volume, trade_count, vwp
    timestamp should be tz-aware NY time if possible.
    """
    ts = pd.to_datetime(bar["timestamp"])
    row = pd.DataFrame([{
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "close": float(bar["close"]),
        "volume": float(bar["volume"]),
        "trade_count": float(bar["trade_count"]),
        "vwp": float(bar["vwp"]),
    }], index=[ts])

    if buf is None or len(buf) == 0:
        buf = row
    else:
        buf = pd.concat([buf, row]).sort_index()

    # de-dup timestamps (keep last)
    buf = buf[~buf.index.duplicated(keep="last")]

    # cap buffer size
    if len(buf) > MAX_BUFFER_BARS:
        buf = buf.iloc[-MAX_BUFFER_BARS:]

    return buf


def on_new_bar(model: XGBClassifier,
               feature_cols: list,
               buf: pd.DataFrame,
               position: Pos,
               bar: Dict) -> Tuple[pd.DataFrame, Pos, str, Optional[float]]:
    """
    Called each time a new 1-minute bar is received.
    Returns updated buffer, (unchanged) position, action string, p_up.
    Note: position should be updated after actual fills by your execution layer.
    """
    buf = append_bar_to_buffer(buf, bar)

    x = compute_features_from_buffer(buf, feature_cols)
    if x is None:
        return buf, position, "WAIT", None

    Xrow = pd.DataFrame([x.values], columns=feature_cols)
    p_up = float(model.predict_proba(Xrow)[0, 1])

    action = action_policy(p_up, position)
    return buf, position, action, p_up


# ============================================================
# EXAMPLE MAIN (replace with your real streaming feed)
# ============================================================
def main():
    model, feature_cols = load_model_and_features(MODEL_PATH, FEATURE_COLS_PATH)

    # Buffer can be preloaded from disk to enable trading at open immediately.
    # For example, load last ~200 bars from yesterday:
    # buf = pd.read_csv("last_bars.csv", parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    buf = pd.DataFrame()
    position = Pos.FLAT

    # --- Simulated stream input ---
    # Replace with your real bar feed callback / websocket / broker API
    example_bars = [
        # timestamp must be comparable to your training timezone
        {"timestamp": "2026-01-13 09:30:00-05:00", "open": 480.0, "high": 480.2, "low": 479.9, "close": 480.1, "volume": 120000, "trade_count": 2500, "vwp": 480.05},
        {"timestamp": "2026-01-13 09:31:00-05:00", "open": 480.1, "high": 480.3, "low": 480.0, "close": 480.2, "volume": 90000, "trade_count": 2200, "vwp": 480.15},
        # add more bars in real usage...
    ]

    for bar in example_bars:
        buf, position, action, p_up = on_new_bar(model, feature_cols, buf, position, bar)
        print(bar["timestamp"], "p_up=", None if p_up is None else round(p_up, 4), "pos=", int(position), "action=", action)

        # In production:
        # - send action to execution layer
        # - execution layer updates 'position' upon fill confirmation

if __name__ == "__main__":
    main()
