import pandas as pd
from pathlib import Path

DATA_DIR = Path("HistData")
OUT_FILE = "SPY_1min_RTH_full.csv"

REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]
OPTIONAL = ["trade_count", "vwp"]

def load_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")

    # Keep required + optional (if present)
    keep = REQUIRED + [c for c in OPTIONAL if c in df.columns]
    df = df[keep].copy()

    # Parse timestamp (tz-aware strings like 2026-01-13 09:30:00-05:00)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    return df

def main():
    paths = sorted(DATA_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {DATA_DIR.resolve()}")

    dfs = []
    for p in paths:
        print(f"Loading: {p.name}")
        dfs.append(load_one_csv(p))

    out = pd.concat(dfs, ignore_index=True)

    # Sort + set index for time-based filtering
    out = out.sort_values("timestamp")
    
    # Handle timezone removal - data may have mixed timezones causing object dtype
    # IMPORTANT: Preserve local times (don't convert to UTC) since between_time expects local RTH
    if out["timestamp"].dtype == 'object':
        # Object dtype means Python datetime objects - strip timezone from each
        out["timestamp"] = out["timestamp"].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x)
        # Convert to datetime64
        out["timestamp"] = pd.to_datetime(out["timestamp"])
    elif hasattr(out["timestamp"].dtype, 'tz') and out["timestamp"].dt.tz is not None:
        # Strip timezone keeping local time
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    
    # Now set index - should be timezone-naive
    out = out.set_index("timestamp").sort_index()

    # Enforce RTH (between_time requires timezone-naive DatetimeIndex)
    out = out.between_time("09:30", "16:00", inclusive="left")

    # De-duplicate timestamps (keep last in case of overlap)
    out = out[~out.index.duplicated(keep="last")]

    # Back to a normal column
    out = out.reset_index()

    # Ensure all expected columns exist; if optional missing, add as NaN
    for col in OPTIONAL:
        if col not in out.columns:
            out[col] = pd.NA

    # Reorder columns consistently
    out = out[REQUIRED + OPTIONAL]

    # Quick summary
    print("\nOutput columns:", out.columns.tolist())
    print("Total rows:", len(out))
    bars_per_day = out.groupby(out["timestamp"].dt.date).size()
    print("\nBars per day summary:")
    print(bars_per_day.describe())

    out.to_csv(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")

if __name__ == "__main__":
    main()
