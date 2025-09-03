import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers time-based features from the DataFrame's index.

    This function was renamed from 'create_time_features' to 'create_features'
    to match the import in the main app.py file.

    Args:
        df (pd.DataFrame): DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with added time-based features.
    """
    # Ensure the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['week_of_year'] = df.index.isocalendar().week.astype(int)

    # Lag features (e.g., usage in the previous hour)
    df['cpu_lag_1'] = df['cpu_usage'].shift(1).fillna(method='bfill')
    df['gpu_lag_1'] = df['gpu_usage'].shift(1).fillna(method='bfill')

    # Rolling window features (e.g., average usage over the last 3 hours)
    df['cpu_rolling_mean_3'] = df['cpu_usage'].rolling(window=3, min_periods=1).mean()
    df['gpu_rolling_mean_3'] = df['gpu_usage'].rolling(window=3, min_periods=1).mean()

    return df

