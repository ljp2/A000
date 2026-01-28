"""Feature preparation utilities."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def fit_quadratic_last_point(series: Iterable[float]) -> Tuple[float, float, float]:
    """
    Fit a 2nd-order polynomial to a series and return value, slope, acceleration
    at the last point in the series.

    Uses x = [0, 1, 2, ..., n-1] as the independent variable.
    Returns (value, slope, acceleration) at x = n-1.
    """
    y = np.asarray(list(series), dtype=float)
    if y.size < 3:
        raise ValueError("series must contain at least 3 points to fit a quadratic")

    x = np.arange(y.size, dtype=float)
    a, b, c = np.polyfit(x, y, 2)
    x_last = x[-1]

    value = a * x_last**2 + b * x_last + c
    slope = 2.0 * a * x_last + b
    acceleration = 2.0 * a

    return float(value), float(slope), float(acceleration)


def fit_cubic_last_point(series: Iterable[float]) -> Tuple[float, float, float]:
    """
    Fit a 3rd-order polynomial to a series and return value, slope, acceleration
    at the last point in the series.

    Uses x = [0, 1, 2, ..., n-1] as the independent variable.
    Returns (value, slope, acceleration) at x = n-1.
    """
    y = np.asarray(list(series), dtype=float)
    if y.size < 4:
        raise ValueError("series must contain at least 4 points to fit a cubic")

    x = np.arange(y.size, dtype=float)
    a, b, c, d = np.polyfit(x, y, 3)
    x_last = x[-1]

    value = a * x_last**3 + b * x_last**2 + c * x_last + d
    slope = 3.0 * a * x_last**2 + 2.0 * b * x_last + c
    acceleration = 6.0 * a * x_last + 2.0 * b

    return float(value), float(slope), float(acceleration)


def heiken_ashi_bars(ohlc: Iterable[dict]) -> np.ndarray:
    """
    Convert OHLC bars to Heikin-Ashi bars.

    Input: iterable of dicts with keys open/high/low/close (or o/h/l/c),
    case-insensitive (e.g., Open/High/Low/Close).
    Output: ndarray shaped (n, 4) as [ha_open, ha_high, ha_low, ha_close].
    """
    rows = list(ohlc)
    if not rows:
        return np.zeros((0, 4), dtype=float)

    data = np.empty((len(rows), 4), dtype=float)
    for i, bar in enumerate(rows):
        if not isinstance(bar, dict):
            raise ValueError("each bar must be a dict with open/high/low/close keys")
        if all(k in bar for k in ("open", "high", "low", "close")):
            data[i, 0] = bar["open"]
            data[i, 1] = bar["high"]
            data[i, 2] = bar["low"]
            data[i, 3] = bar["close"]
        elif all(k in bar for k in ("Open", "High", "Low", "Close")):
            data[i, 0] = bar["Open"]
            data[i, 1] = bar["High"]
            data[i, 2] = bar["Low"]
            data[i, 3] = bar["Close"]
        elif all(k in bar for k in ("o", "h", "l", "c")):
            data[i, 0] = bar["o"]
            data[i, 1] = bar["h"]
            data[i, 2] = bar["l"]
            data[i, 3] = bar["c"]
        else:
            raise ValueError("bar keys must be open/high/low/close or o/h/l/c")

    o = data[:, 0]
    h = data[:, 1]
    l = data[:, 2]
    c = data[:, 3]

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, data.shape[0]):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])



    return np.column_stack((ha_open, ha_high, ha_low, ha_close))


def add_heiken_ashi_columns(
    df: pd.DataFrame,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    prefix: str = "ha_",
) -> pd.DataFrame:
    """
    Add Heikin-Ashi columns to a DataFrame containing OHLC data.

    Columns added: {prefix}open, {prefix}high, {prefix}low, {prefix}close.
    Operates in-place and returns the same DataFrame.
    """
    if df.empty:
        for name in ("open", "high", "low", "close"):
            df[f"{prefix}{name}"] = pd.Series(dtype=float)
        return df

    o = df[open_col].astype(float).to_numpy()
    h = df[high_col].astype(float).to_numpy()
    l = df[low_col].astype(float).to_numpy()
    c = df[close_col].astype(float).to_numpy()

    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])

    df[f"{prefix}open"] = ha_open
    df[f"{prefix}high"] = ha_high
    df[f"{prefix}low"] = ha_low
    df[f"{prefix}close"] = ha_close

    return df


def rolling_linear_features(
    series: pd.Series, window: int, prefix: str = ""
) -> pd.DataFrame:
    """
    Compute value and slope from a linear fit over the last N points.

    Input series must have a DateTimeIndex. Output columns: {prefix}value,
    {prefix}slope.
    The first (window - 1) rows are NaN because there is insufficient history.
    """
    if window < 2:
        raise ValueError("window must be at least 2 to fit a line")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series index must be a DateTimeIndex")

    y = series.astype(float).to_numpy()
    n = y.size
    if n == 0:
        return pd.DataFrame(
            index=series.index,
            columns=[f"{prefix}value", f"{prefix}slope"],
            dtype=float,
        )

    values = np.full(n, np.nan, dtype=float)
    slopes = np.full(n, np.nan, dtype=float)

    x = np.arange(window, dtype=float)
    x_last = x[-1]
    for i in range(window - 1, n):
        y_window = y[i - window + 1 : i + 1]
        m, b = np.polyfit(x, y_window, 1)
        values[i] = m * x_last + b
        slopes[i] = m

    return pd.DataFrame(
        {f"{prefix}value": values, f"{prefix}slope": slopes},
        index=series.index,
    )


def rolling_quadratic_features(
    series: pd.Series, window: int, prefix: str = ""
) -> pd.DataFrame:
    """
    Compute value, slope, and acceleration from a quadratic fit over the last N points.

    Input series must have a DateTimeIndex. Output columns: {prefix}value,
    {prefix}slope, {prefix}acceleration.
    The first (window - 1) rows are NaN because there is insufficient history.

    """
    if window < 3:
        raise ValueError("window must be at least 3 to fit a quadratic")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series index must be a DateTimeIndex")

    y = series.astype(float).to_numpy()
    n = y.size
    if n == 0:
        return pd.DataFrame(
            index=series.index,
            columns=[f"{prefix}value", f"{prefix}slope", f"{prefix}acceleration"],
            dtype=float,
        )

    values = np.full(n, np.nan, dtype=float)
    slopes = np.full(n, np.nan, dtype=float)
    accels = np.full(n, np.nan, dtype=float)

    x = np.arange(window, dtype=float)
    x_last = x[-1]
    for i in range(window - 1, n):
        y_window = y[i - window + 1 : i + 1]
        a, b, c = np.polyfit(x, y_window, 2)
        values[i] = a * x_last**2 + b * x_last + c
        slopes[i] = 2.0 * a * x_last + b
        accels[i] = 2.0 * a

    return pd.DataFrame(
        {
            f"{prefix}value": values,
            f"{prefix}slope": slopes,
            f"{prefix}acceleration": accels,
        },
        index=series.index,
    )


def rolling_cubic_features(
    series: pd.Series, window: int, prefix: str = ""
) -> pd.DataFrame:
    """
    Compute value, slope, and acceleration from a cubic fit over the last N points.

    Input series must have a DateTimeIndex. Output columns: {prefix}value,
    {prefix}slope, {prefix}acceleration.
    The first (window - 1) rows are NaN because there is insufficient history.
    """
    if window < 4:
        raise ValueError("window must be at least 4 to fit a cubic")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series index must be a DateTimeIndex")

    y = series.astype(float).to_numpy()
    n = y.size
    if n == 0:
        return pd.DataFrame(
            index=series.index,
            columns=[f"{prefix}value", f"{prefix}slope", f"{prefix}acceleration"],
            dtype=float,
        )

    values = np.full(n, np.nan, dtype=float)
    slopes = np.full(n, np.nan, dtype=float)
    accels = np.full(n, np.nan, dtype=float)

    x = np.arange(window, dtype=float)
    x_last = x[-1]
    for i in range(window - 1, n):
        y_window = y[i - window + 1 : i + 1]
        a, b, c, d = np.polyfit(x, y_window, 3)
        values[i] = a * x_last**3 + b * x_last**2 + c * x_last + d
        slopes[i] = 3.0 * a * x_last**2 + 2.0 * b * x_last + c
        accels[i] = 6.0 * a * x_last + 2.0 * b

    return pd.DataFrame(
        {
            f"{prefix}value": values,
            f"{prefix}slope": slopes,
            f"{prefix}acceleration": accels,
        },
        index=series.index,
    )


