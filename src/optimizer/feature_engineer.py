import pandas as pd

def create_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Creates time-based features from a timestamp column.

    Args:
        df (pd.DataFrame): The input DataFrame with a timestamp column.
        timestamp_col (str): The name of the timestamp column.

    Returns:
        pd.DataFrame: The DataFrame with added time-based features.
    """
    if timestamp_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        raise ValueError(f"Column '{timestamp_col}' must be a datetime type.")

    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek # Monday=0, Sunday=6
    df['day_of_month'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    
    print("Created time-based features: hour, day_of_week, day_of_month, month.")
    return df

def create_rolling_features(df: pd.DataFrame, columns: list, window_sizes: list) -> pd.DataFrame:
    """
    Creates rolling window features (e.g., moving averages) for specified columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to create rolling features for.
        window_sizes (list): A list of integer window sizes (e.g., [3, 7, 14]).

    Returns:
        pd.DataFrame: The DataFrame with added rolling features.
    """
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found for rolling feature creation. Skipping.")
            continue
        for window in window_sizes:
            feature_name = f'{col}_rolling_mean_{window}'
            df[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
            print(f"Created rolling feature: {feature_name}")
            
    return df

if __name__ == '__main__':
    # Example Usage
    print("\n--- Feature Engineer Example ---")
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00', '2023-01-01 03:00']),
        'cpu_usage': [20, 25, 22, 30]
    }
    example_df = pd.DataFrame(data)
    
    print("\nOriginal DataFrame:")
    print(example_df)
    
    # 1. Create time features
    df_with_time_features = create_time_features(example_df.copy(), 'timestamp')
    print("\nDataFrame with Time Features:")
    print(df_with_time_features)

    # 2. Create rolling features
    df_with_rolling_features = create_rolling_features(
        df_with_time_features,
        columns=['cpu_usage'],
        window_sizes=[2, 3]
    )
    print("\nDataFrame with Rolling Features:")
    print(df_with_rolling_features)
