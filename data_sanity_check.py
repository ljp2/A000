import pandas as pd

df = pd.read_csv(
    "SPY_1min_RTH_full.csv",
    parse_dates=["timestamp"],
    index_col="timestamp"
)

# 1) Sorted, unique timestamps
assert df.index.is_monotonic_increasing
assert not df.index.duplicated().any()

# 2) Bars per day
bars_per_day = df.groupby(df.index.date).size()
print(bars_per_day.describe())

# 3) Inspect one day
import numpy as np

random_day = np.random.choice(df.index.date)
print("Random session:", random_day)
print(df.loc[str(random_day)].head())

