import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, roc_auc_score

EPS = 1e-12

# ============================================================
# Baseline / evaluation config
# ============================================================
BASELINE_VERSION = "v2"

# Transaction cost (round-trip, log-return space)
COST_LR = 0.00001   # 1 bp realistic baseline

# RCD label params (locked)
RCD_PARAMS = dict(
    horizon=2,
    deadband_base=0.05,
    rv_low=0.90,
    rv_high=1.10,
    mult_low=0.85,
    mult_mid=1.00,
    mult_high=1.25,
)

# ============================================================
# Helpers
# ============================================================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - m) / (sd + EPS)

def enforce_rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "16:00", inclusive="left")

# ============================================================
# Feature blocks (baseline 64)
# ============================================================
def add_session_keys(df):
    df["session_date"] = df.index.date
    return df

def add_time_of_day(df):
    mins = (df.index.hour * 60 + df.index.minute) - (9 * 60 + 30)
    df["minute_of_session"] = mins.astype(np.int16)
    ang = 2.0 * np.pi * df["minute_of_session"] / 390.0
    df["sin_tod"] = np.sin(ang)
    df["cos_tod"] = np.cos(ang)
    df["is_opening_30m"] = (df["minute_of_session"] < 30).astype(np.int8)
    df["is_closing_30m"] = (df["minute_of_session"] >= 360).astype(np.int8)
    return df

def add_session_anchors(df):
    df["open_session"] = df.groupby("session_date")["open"].transform("first")
    df["hod"] = df.groupby("session_date")["high"].cummax()
    df["lod"] = df.groupby("session_date")["low"].cummin()
    return df

def add_session_cum_vwap(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    df["vwap_sess"] = (
        pv.groupby(df["session_date"]).cumsum()
        / (df["volume"].groupby(df["session_date"]).cumsum() + EPS)
    )
    return df

def add_normalizers(df):
    c,h,l = df["close"],df["high"],df["low"]
    prev_c = c.shift(1)
    sc = df["session_date"] != df["session_date"].shift(1)
    prev_c = prev_c.where(~sc, c)

    df["lr_1"] = np.log(c / (prev_c + EPS))

    tr = np.maximum(h-l, np.maximum((h-prev_c).abs(), (l-prev_c).abs()))
    df["tr"] = tr
    df["atr_20"] = ema(tr,20)
    df["atr_120"] = ema(tr,120)

    df["rv_20"] = df["lr_1"].rolling(20).std(ddof=0)
    df["rv_120"] = df["lr_1"].rolling(120).std(ddof=0)

    df["atr_ret_20"] = df["atr_20"] / (c + EPS)
    return df

def add_return_features(df):
    for k in [1,2,3,5,10,15,30,60]:
        lr_k = np.log(df["close"]/(df["close"].shift(k)+EPS))
        df[f"ret_lr_{k}"] = lr_k
        df[f"ret_atr_{k}"] = lr_k/(df["atr_ret_20"]+EPS)

    df["lr_z_20"] = rolling_zscore(df["lr_1"],20)
    df["lr_z_60"] = rolling_zscore(df["lr_1"],60)
    df["mom_mean_20"] = df["lr_1"].rolling(20).mean()
    df["mom_std_20"] = df["lr_1"].rolling(20).std(ddof=0)

    e12 = ema(df["close"],12)
    e48 = ema(df["close"],48)
    df["ema_spread_atr"] = (e12-e48)/(df["atr_20"]+EPS)
    df["ema12_slope_10_atr"] = (e12-e12.shift(10))/(df["atr_20"]+EPS)
    return df

def add_candle_anatomy(df):
    o,h,l,c = df["open"],df["high"],df["low"],df["close"]
    atr = df["atr_20"]+EPS
    df["range_atr"] = (h-l)/atr
    df["body_atr"] = (c-o)/atr
    df["body_abs_atr"] = (c-o).abs()/atr
    df["upper_wick_atr"] = (h-np.maximum(o,c))/atr
    df["lower_wick_atr"] = (np.minimum(o,c)-l)/atr
    rng = (h-l)+EPS
    df["close_loc"] = (c-l)/rng
    df["body_to_range"] = (c-o).abs()/rng
    df["range_expand_20"] = df["range_atr"]/(df["range_atr"].rolling(20).mean()+EPS)
    return df

def add_vol_regime(df):
    df["rv_ratio_20_120"] = df["rv_20"]/(df["rv_120"]+EPS)
    df["atr_ratio_20_120"] = df["atr_20"]/(df["atr_120"]+EPS)
    df["vov_60"] = df["rv_20"].rolling(60).std(ddof=0)
    return df

def add_vwap_features(df):
    atr = df["atr_20"]+EPS
    df["dist_vwapbar_atr"] = (df["close"]-df["vwp"])/atr
    df["dist_vwapbar_z_60"] = rolling_zscore(df["close"]-df["vwp"],60)
    df["dist_vwapsess_atr"] = (df["close"]-df["vwap_sess"])/atr
    df["dist_vwapsess_z_60"] = rolling_zscore(df["close"]-df["vwap_sess"],60)
    df["vwapsess_slope_10_atr"] = (df["vwap_sess"]-df["vwap_sess"].shift(10))/atr
    df["dist_open_atr"] = (df["close"]-df["open_session"])/atr
    return df

def add_hod_lod_features(df):
    atr = df["atr_20"]+EPS
    df["dist_hod_atr"] = (df["hod"]-df["close"])/atr
    df["dist_lod_atr"] = (df["close"]-df["lod"])/atr
    rng = (df["hod"]-df["lod"])+EPS
    df["pos_day_range"] = (df["close"]-df["lod"])/rng
    df["day_range_atr"] = (df["hod"]-df["lod"])/atr
    return df

def add_volume_features(df):
    v = df["volume"].astype(float)
    for n in [20,60]:
        mv,sv = v.rolling(n).mean(),v.rolling(n).std(ddof=0)
        df[f"vol_ratio_{n}"] = v/(mv+EPS)
        df[f"vol_z_{n}"] = (v-mv)/(sv+EPS)
    df["signed_vol"] = v*(2*df["close_loc"]-1)
    df["signed_vol_ema_20"] = ema(df["signed_vol"],20)/(v.rolling(20).mean()+EPS)
    df["volXrange"] = df["vol_ratio_20"]*df["range_atr"]
    df["volXdistVWAP"] = df["vol_ratio_20"]*df["dist_vwapsess_atr"]
    return df

def add_trade_count_features(df):
    tc = df["trade_count"].astype(float)
    for n in [20,60]:
        mt,st = tc.rolling(n).mean(),tc.rolling(n).std(ddof=0)
        df[f"tc_ratio_{n}"] = tc/(mt+EPS)
        df[f"tc_z_{n}"] = (tc-mt)/(st+EPS)
    df["trades_per_vol"] = tc/(df["volume"]+EPS)
    df["trades_per_vol_z_60"] = rolling_zscore(df["trades_per_vol"],60)
    return df

def add_efficiency(df):
    c = df["close"]
    step = (c-c.shift(1)).abs()
    for n in [20,60]:
        net = (c-c.shift(n)).abs()
        path = step.rolling(n).sum()
        df[f"eff_ratio_{n}"] = net/(path+EPS)
    return df

# ============================================================
# Target (baseline-v2 RCD)
# ============================================================
def add_target(df):
    p = RCD_PARAMS
    fwd_lr = np.log(df["close"].shift(-p["horizon"])/df["close"])
    df["fwd_lr_2"] = fwd_lr
    df["y"] = (fwd_lr>0).astype(np.int8)

    rv = df["rv_ratio_20_120"]
    mult = np.where(rv<p["rv_low"],p["mult_low"],
           np.where(rv>p["rv_high"],p["mult_high"],p["mult_mid"]))
    db = (p["deadband_base"]*mult)*df["atr_ret_20"]
    df.loc[fwd_lr.abs()<=db,"y"]=np.nan
    return df

# ============================================================
# Build features
# ============================================================
def build_features_from_csv(path):
    df = pd.read_csv(path,parse_dates=["timestamp"])
    df.columns=[c.strip().lower() for c in df.columns]
    df=df.set_index("timestamp").sort_index()
    df=df[["open","high","low","close","volume","trade_count","vwp"]].astype(float)

    df=enforce_rth(df)
    df=add_session_keys(df)
    df=add_time_of_day(df)
    df=add_session_anchors(df)
    df=add_session_cum_vwap(df)
    df=add_normalizers(df)

    df=add_return_features(df)
    df=add_candle_anatomy(df)
    df=add_vol_regime(df)
    df=add_vwap_features(df)
    df=add_hod_lod_features(df)
    df=add_volume_features(df)
    df=add_trade_count_features(df)
    df=add_efficiency(df)
    df=add_target(df)
    return df.copy()

# ============================================================
# Final X / y
# ============================================================
def finalize_xy(feat_df):
    df2 = feat_df.dropna().copy()
    exclude = {
        "open","high","low","close","volume","trade_count","vwp",
        "open_session","hod","lod","vwap_sess","session_date",
        "tr","atr_20","atr_120","rv_20","rv_120","atr_ret_20","lr_1",
        "fwd_lr_2","y"
    }
    X = df2[[c for c in df2.columns if c not in exclude]]
    y = df2["y"].astype(int)
    return X,y

# ============================================================
# Purged day-based CV
# ============================================================
def purged_day_splits(df,n_splits=5,purge_minutes=2):
    days = pd.Index(sorted(pd.unique(df["session_date"])))
    fold_sizes = np.full(n_splits,len(days)//n_splits,dtype=int)
    fold_sizes[:len(days)%n_splits]+=1
    start=0
    for fs in fold_sizes:
        val_days = days[start:start+fs]
        train_days = days.drop(val_days)
        val_df = df[df["session_date"].isin(val_days)]

        def purge(g):
            return g.iloc[purge_minutes:-purge_minutes] if len(g)>2*purge_minutes else g.iloc[0:0]

        val_kept = val_df.groupby("session_date",group_keys=False).apply(purge)
        yield (
            df[df["session_date"].isin(train_days)].index.values,
            val_kept.index.values
        )
        start+=fs

# ============================================================
# OOF predictions
# ============================================================
def collect_oof_predictions(X,y,feat_df,splits,params):
    out=[]
    for i,(tr,va) in enumerate(splits):
        m = XGBClassifier(**params)
        m.fit(X.loc[tr],y.loc[tr])
        p = m.predict_proba(X.loc[va])[:,1]
        tmp = feat_df.loc[va,["fwd_lr_2"]].copy()
        tmp["p_up"]=p
        tmp["y"]=y.loc[va].values
        tmp["fold"]=i
        out.append(tmp)
    return pd.concat(out).sort_index()

# ============================================================
# Threshold evaluation
# ============================================================
def eval_thresholds(oof,thresholds,cost_lr=0.0):
    rows=[]
    for t in thresholds:
        long = oof["p_up"]>=t
        short = oof["p_up"]<=(1-t)

        lr = oof["fwd_lr_2"]
        rets = pd.concat([
            lr[long]-cost_lr,
            -lr[short]-cost_lr
        ])

        rows.append(dict(
            threshold=t,
            n_total=len(rets),
            hit_rate=(rets>0).mean() if len(rets) else np.nan,
            avg_lr=rets.mean() if len(rets) else np.nan,
            median_lr=rets.median() if len(rets) else np.nan
        ))
    return pd.DataFrame(rows)

# ============================================================
# Main
# ============================================================
def main():
    params=dict(
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

    thresholds=np.arange(0.50,0.66,0.01)

    feat_df = build_features_from_csv("SPY_1min_RTH_full.csv")
    feat_df_nonan = feat_df.dropna().copy()
    X,y = finalize_xy(feat_df_nonan)

    splits=list(purged_day_splits(feat_df_nonan))
    oof = collect_oof_predictions(X,y,feat_df_nonan,splits,params)

    print("\n==== THRESHOLD STATS (no cost) ====")
    print(eval_thresholds(oof,thresholds).to_string(index=False))

    print("\n==== THRESHOLD STATS (cost-adjusted) ====")
    print(eval_thresholds(oof,thresholds,COST_LR).to_string(index=False))

if __name__=="__main__":
    main()
