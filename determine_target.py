import pandas as pd
import numpy as np



def add_future_gt_column(
    df: pd.DataFrame, column: str, future_rows: int = 2, new_column: str = "target"
) -> pd.DataFrame:
    """Add a boolean column that's True when a future value is greater."""
    df["shifted"] = df[column].shift(-future_rows)
    df[new_column] = df[column] < df[column].shift(-future_rows)
    df[new_column] = df[new_column].fillna(False)
    return df


def add_future_lt_column(
    df: pd.DataFrame, column: str, future_rows: int = 2, new_column: str = "target"
) -> pd.DataFrame:
    """Add a boolean column that's True when a future value is less."""
    df["shifted"] = df[column].shift(-future_rows)
    df[new_column] = df[column] > df[column].shift(-future_rows)
    df[new_column] = df[new_column].fillna(False)
    return df


# Example usage
df = pd.DataFrame(
    {
        "Close": [100.0, 101.5, 99.0, 102.0, 98.5, 103.0],
        "Volume": [1000, 1100, 1050, 1200, 1150, 1300],
    }
)

df = add_future_gt_column(df, column="Close", future_rows=2, new_column="close_gt_plus2")

df = add_future_lt_column(df, column="Close", future_rows=2, new_column="close_lt_plus2")
print(df)

