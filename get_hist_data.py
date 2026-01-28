

import sys
import os
from datetime import date, datetime, time, timedelta
import pandas as pd
import pytz
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Column renaming mapping
COL_RENAME = {"vwap": "vwp"}



def check_missing_data(df):
    """
    Check for missing rows (gaps in timestamps) and missing values in a DataFrame
    with a timestamp index representing 1-minute financial data, and impute missing data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with DatetimeIndex or time index

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing timestamps filled and missing values imputed
    """
    df_imputed = df.copy()

    # Convert index to datetime if needed for processing
    if isinstance(df_imputed.index, pd.DatetimeIndex):
        time_index = df_imputed.index
        needs_conversion = False
    else:
        # If index is time objects, convert to datetime for processing
        try:
            time_index = pd.to_datetime(df_imputed.index.astype(str), format="%H:%M:%S")
            needs_conversion = True
        except (ValueError, TypeError):
            # If conversion fails, just impute existing missing values
            return _impute_missing_values(df_imputed)

    # Check for timestamp gaps and add missing rows
    if len(df_imputed) > 1:
        # Calculate expected frequency (1 minute)
        expected_freq = pd.Timedelta(minutes=1)

        # Find gaps larger than 1 minute
        time_diffs = time_index.to_series().diff()
        gaps = time_diffs[time_diffs > expected_freq]

        missing_rows = []

        # For each gap, create missing rows
        for idx, gap_size in gaps.items():
            gap_minutes = int(gap_size.total_seconds() / 60)
            if gap_minutes > 1:
                # Find the previous timestamp and its data
                prev_idx = time_index.get_loc(idx) - 1
                prev_time = time_index[prev_idx]

                # Get data from previous and current rows for interpolation
                if needs_conversion:
                    prev_row = df_imputed.iloc[prev_idx]
                    curr_row = df_imputed.loc[df_imputed.index[time_index.get_loc(idx)]]
                else:
                    prev_row = df_imputed.iloc[prev_idx]
                    curr_row = df_imputed.iloc[time_index.get_loc(idx)]

                # Generate missing timestamps and interpolated data
                for i in range(1, gap_minutes):
                    missing_time = prev_time + pd.Timedelta(minutes=i)

                    # Create interpolated row
                    interpolation_ratio = i / gap_minutes
                    missing_row = {}

                    for col in df_imputed.columns:
                        if pd.api.types.is_numeric_dtype(df_imputed[col]):
                            # Linear interpolation for numeric columns
                            missing_row[col] = (
                                prev_row[col]
                                + (curr_row[col] - prev_row[col]) * interpolation_ratio
                            )
                        else:
                            # Forward fill for non-numeric columns
                            missing_row[col] = prev_row[col]

                    # Convert back to time index if needed
                    if needs_conversion:
                        missing_time_idx = missing_time.time()
                    else:
                        missing_time_idx = missing_time

                    missing_rows.append((missing_time_idx, missing_row))

        # Add missing rows to dataframe
        if missing_rows:
            for time_idx, row_data in missing_rows:
                df_imputed.loc[time_idx] = row_data

            # Sort by index to maintain chronological order
            df_imputed = df_imputed.sort_index()

    # Impute remaining missing values
    df_imputed = _impute_missing_values(df_imputed)

    return df_imputed


def _impute_missing_values(df):
    """
    Helper function to impute missing values in DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with potential missing values

    Returns
    -------
    pandas.DataFrame
        DataFrame with missing values imputed
    """
    df_imputed = df.copy()

    for col in df_imputed.columns:
        if df_imputed[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df_imputed[col]):
                # For numeric columns, use forward fill then backward fill
                df_imputed[col] = df_imputed[col].ffill().bfill()

                # If still NaN (all values were NaN), fill with 0
                if df_imputed[col].isnull().any():
                    df_imputed[col] = df_imputed[col].fillna(0)
            else:
                # For non-numeric columns, use forward fill then backward fill
                df_imputed[col] = df_imputed[col].ffill().bfill()

                # If still NaN, fill with empty string or appropriate default
                if df_imputed[col].isnull().any():
                    df_imputed[col] = df_imputed[col].fillna("")

    return df_imputed



def get_nyse_open_close_times() -> tuple[time, time]:
    """
    Returns NYSE open and close times (Eastern Time).
    Returns
    -------
    tuple[time, time]
        (open_time, close_time)
    """
    return time(9, 30), time(15, 59)


def get_nyse_minute_bars(ticker: str, trade_date: date) -> pd.DataFrame | None:
    """
    Fetch 1-minute historical bars for a ticker during NYSE trading hours.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY')
    trade_date : date
        The trading date to fetch data for

    Returns
    -------
    pd.DataFrame or None
        DataFrame with OHLCV data indexed by timestamp in US/Eastern timezone,
        or None if no data is available
    """

    # NYSE open/close times (Eastern Time)
    nyse_open, nyse_close = get_nyse_open_close_times()

    # Convert date to datetime for start/end
    start_dt = datetime.combine(trade_date, nyse_open)
    end_dt = datetime.combine(trade_date, nyse_close)

    # Alpaca expects UTC, so convert from US/Eastern to UTC
    eastern = pytz.timezone("US/Eastern")
    utc = pytz.utc
    start_utc = eastern.localize(start_dt).astimezone(utc)
    end_utc = eastern.localize(end_dt).astimezone(utc)

    # Initialize Alpaca client (requires ALPACA_API_KEY and ALPACA_SECRET_KEY env vars)
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")

    client = StockHistoricalDataClient(api_key, api_secret)

    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start_utc,
        end=end_utc,
    )

    bars = client.get_stock_bars(request_params)
    if bars.df.empty:
        print("No bars returned")
        return None
    df = bars.df.loc[ticker]

    # Ensure index is DatetimeIndex for between_time
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            raise ValueError(
                "DataFrame does not have a 'timestamp' column for indexing."
            )

    # Convert index from UTC to US/Eastern for correct filtering
    df.index = df.index.tz_convert("US/Eastern")

    # Filter for NYSE hours just in case
    df = df.between_time(nyse_open, nyse_close)
    return df


def get_nyse_bars_for_range(
    ticker: str, start_date: date, end_date: date, data_dir: str = "HistData"
) -> None:
    """
    Fetch NYSE minute bars for a ticker over a date range and save to CSV files.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY')
    start_date : date
        Start date for data collection (inclusive)
    end_date : date
        End date for data collection (inclusive)
    data_dir : str, optional
        Directory to save CSV files (default: 'HistData')

    Raises
    ------
    ValueError
        If date range is invalid or dates are in the future
    """
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)
    
    # Get current date and time in Eastern timezone
    today = datetime.now(pytz.timezone("US/Eastern")).date()
    now_et = datetime.now(pytz.timezone("US/Eastern")).time()
    nyse_open, nyse_close = get_nyse_open_close_times()

    # Validation checks
    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    if start_date > today:
        raise ValueError(
            f"start_date ({start_date}) cannot be in the future (today is {today})"
        )

    if end_date > today:
        raise ValueError(
            f"end_date ({end_date}) cannot be in the future (today is {today})"
        )

    # Adjust end_date if NYSE is still open today
    actual_end_date = end_date
    if end_date == today and now_et < nyse_close:
        actual_end_date = today - timedelta(days=1)
        print(
            f"NYSE is still open today, adjusting end_date from {end_date} to {actual_end_date}"
        )

    current_date = start_date
    os.makedirs(data_dir, exist_ok=True)

    while current_date <= actual_end_date:
        if current_date.weekday() < 5:  # 0=Monday, 4=Friday
            filename = f"{ticker}_{current_date.strftime('%Y%m%d')}.csv"
            filepath = os.path.join(data_dir, filename)

            # Check if file already exists
            if os.path.exists(filepath):
                print(
                    f"Data for {ticker} on {current_date} already exists, skipping..."
                )
            else:
                print(f"Fetching data for {ticker} on {current_date}")
                # Get the minute bars for the current date
                df = get_nyse_minute_bars(ticker, current_date)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df_complete = check_missing_data(df)
                    df_complete.rename(columns=COL_RENAME, inplace=True)
                    df_complete.to_csv(filepath)

                else:
                    print(
                        f"No data available for {ticker} on {current_date} (market closed or holiday), skipping..."
                    )
        else:
            print(f"Skipping {current_date} (weekend)")
        current_date += timedelta(days=1)


def get_nyse_bars_last_n_days(
    ticker: str, number_days: int, data_dir: str = "HistData"
) -> None:
    """
    Fetch NYSE minute bars for the last N trading days and save to CSV files.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY')
    number_days : int
        Number of trading days to fetch (excludes weekends)
    data_dir : str, optional
        Directory to save CSV files (default: 'HistData')
    """
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, data_dir)
    os.makedirs(data_dir, exist_ok=True)
    today = datetime.now(pytz.timezone("US/Eastern")).date()
    now_et = datetime.now(pytz.timezone("US/Eastern")).time()
    nyse_open, nyse_close = get_nyse_open_close_times()

    # If NYSE is still open today, skip today
    if now_et < nyse_close:
        end_date = today - timedelta(days=1)
    else:
        end_date = today

    bars_collected = 0
    current_date = end_date
    while bars_collected < number_days:
        if current_date.weekday() < 5:  # 0=Monday, 4=Friday
            filename = f"{ticker}_{current_date.strftime('%Y%m%d')}.csv"
            filepath = os.path.join(data_dir, filename)

            # Check if file already exists
            if os.path.exists(filepath):
                print(
                    f"Data for {ticker} on {current_date} already exists, skipping..."
                )
                bars_collected += 1
            else:
                print(f"Fetching data for {ticker} on {current_date}")
                df = get_nyse_minute_bars(ticker, current_date)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Check and impute missing data
                    df_complete = check_missing_data(df)
                    df_complete.rename(columns=COL_RENAME, inplace=True)
                    df_complete.to_csv(filepath)
                    print(
                        f"Data saved for {ticker} on {current_date} ({len(df_complete)} rows)"
                    )
                    bars_collected += 1
                else:
                    print(
                        f"No data available for {ticker} on {current_date} (market closed or holiday), skipping..."
                    )
        else:
            print(f"Skipping {current_date} (weekend)")
        current_date -= timedelta(days=1)


if __name__ == "__main__":
    ticker = "SPY"
    get_nyse_bars_last_n_days(ticker, 100)
    # get_nyse_bars_for_range(ticker, date(2024, 12, 21), date(2024, 12, 29))
    # get_nyse_bars_for_range(ticker, date(2025, 6, 1), date(2025, 6, 13))
