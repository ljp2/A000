# train_cv_10m.py
# 10-minute horizon evaluation (same framework as your 5m test)

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

EPS = 1e-12

BASELINE_VERSION = "v2_10m"

RCD_PARAMS = dict(
    horizon=10,           # <<< CHANGED
    deadband_base=0.05,
    rv_low=0.90,
    rv_high=1.10,
    mult_low=0.85,
    mult_mid=1.00,
    mult_high=1.25,
)

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def enforce_rth(df):
    return df.between_time("09:30", "16:00", inclusive="left")

def add_session_keys(df):
    df["session_date"] = df.index.date
    return df

def add_time_of_day(df):
    mins = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    ang = 2 * np.pi * mins / 390.0
    df["sin_tod"] = np.sin(ang)
    df["cos_tod"] = np.cos(ang)
    df["is_opening_30m"] = (mins < 30).astype(int)
    df["is_closing_30m"] = (mins >= 360).astype(int)
    return df

def add_session_anchors(df):
    df["open_session"] = df.groupby("session_date")["open"].transform("first")
    df["hod"] = df.groupby("session_date")["high"].cummax()
    df["lod"] = df.groupby("session_date")["low"].cummin()
    return df

def add_session_cum_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    df["vwap_sess"] = pv.groupby(df["session_date"]).cumsum() / (
        df["volume"].groupby(df["session_date"]).cumsum() + EPS
    )
    return df

def add_normalizers(df):
    c = df["close"]
    prev_c = c.shift(1)
    prev_c = prev_c.where(df["session_date"] == df["session_date"].shift(1), c)

    df["lr_1"] = np.log(c / (prev_c + EPS))

    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum((df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()),
    )
    df["atr_20"] = ema(tr, 20)
    df["rv_20"] = df["lr_1"].rolling(20).std(ddof=0)
    df["rv_120"] = df["lr_1"].rolling(120).std(ddof=0)
    df["atr_ret_20"] = df["atr_20"] / (c + EPS)

    df["rv_ratio_20_120"] = df["rv_20"] / (df["rv_120"] + EPS)
    return df

def add_return_features(df):
    for k in [1, 2, 3, 5, 10, 15, 30, 60]:
        lrk = np.log(df["close"] / (df["close"].shift(k) + EPS))
        df[f"ret_lr_{k}"] = lrk
        df[f"ret_atr_{k}"] = lrk / (df["atr_ret_20"] + EPS)
    return df

def add_target(df):
    p = RCD_PARAMS
    fwd = np.log(df["close"].shift(-p["horizon"]) / df["close"])
    df["fwd_lr_10"] = fwd
    df["y"] = (fwd > 0).astype(int)

    mult = np.where(
        df["rv_ratio_20_120"] < p["rv_low"], p["mult_low"],
        np.where(df["rv_ratio_20_120"] > p["rv_high"], p["mult_high"], p["mult_mid"])
    )
    deadband = p["deadband_base"] * mult * df["atr_ret_20"]
    df.loc[fwd.abs() <= deadband, "y"] = np.nan
    return df

def build_features_from_csv(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df.columns = [c.lower() for c in df.columns]
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume", "trade_count", "vwp"]]

    df = enforce_rth(df)
    df = add_session_keys(df)
    df = add_time_of_day(df)
    df = add_session_anchors(df)
    df = add_session_cum_vwap(df)
    df = add_normalizers(df)
    df = add_return_features(df)
    df = add_target(df)
    return df.dropna()

def finalize_xy(df):
    exclude = {
        "open", "high", "low", "close", "volume", "trade_count", "vwp",
        "session_date", "atr_20", "rv_20", "rv_120", "atr_ret_20", "lr_1",
        "fwd_lr_10", "y"
    }
    X = df[[c for c in df.columns if c not in exclude]]
    y = df["y"].astype(int)
    return X, y

def purged_day_splits(df, n_splits=5, purge_minutes=10):  # <<< CHANGED
    days = sorted(df["session_date"].unique())
    folds = np.array_split(days, n_splits)

    for val_days in folds:
        train_days = [d for d in days if d not in val_days]
        val = df[df["session_date"].isin(val_days)]

        def purge(g):
            if len(g) <= 2 * purge_minutes:
                return g.iloc[0:0]
            return g.iloc[purge_minutes:-purge_minutes]

        val = val.groupby("session_date", group_keys=False).apply(purge)

        yield (
            df[df["session_date"].isin(train_days)].index,
            val.index,
        )

def collect_oof(X, y, df, splits, params):
    out = []
    for i, (tr, va) in enumerate(splits):
        m = XGBClassifier(**params)
        m.fit(X.loc[tr], y.loc[tr])
        p = m.predict_proba(X.loc[va])[:, 1]

        tmp = df.loc[va, ["fwd_lr_10"]].copy()
        tmp["p"] = p
        tmp["y"] = y.loc[va].values
        tmp["fold"] = i
        out.append(tmp)

    return pd.concat(out).sort_index()

def eval_thresholds(oof, thresholds, cost=0.0):
    rows = []
    for t in thresholds:
        long = oof["p"] >= t
        short = oof["p"] <= (1 - t)
        rets = pd.concat([
            oof.loc[long, "fwd_lr_10"] - cost,
            -oof.loc[short, "fwd_lr_10"] - cost
        ])
        rows.append(dict(
            threshold=float(t),
            n=int(len(rets)),
            hit=float((rets > 0).mean()) if len(rets) else np.nan,
            avg_lr=float(rets.mean()) if len(rets) else np.nan
        ))
    return pd.DataFrame(rows)

def main():
    params = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=50,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    thresholds = np.arange(0.52, 0.66, 0.01)

    df = build_features_from_csv("SPY_1min_RTH_full.csv")
    X, y = finalize_xy(df)

    splits = list(purged_day_splits(df))
    oof = collect_oof(X, y, df, splits, params)

    print("\n==== 10-MIN THRESHOLDS (NO COST) ====")
    print(eval_thresholds(oof, thresholds, 0.0).to_string(index=False))

    print("\n==== 10-MIN THRESHOLDS (0.5 BP COST) ====")
    print(eval_thresholds(oof, thresholds, 0.00005).to_string(index=False))

    print("\n==== 10-MIN THRESHOLDS (1.0 BP COST) ====")
    print(eval_thresholds(oof, thresholds, 0.0001).to_string(index=False))

if __name__ == "__main__":
    main()
