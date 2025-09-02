import pandas as pd
from typing import Optional

def load_metrics_data(filepath: str, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """
    Loads telemetry data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.
        parse_dates (Optional[list]): A list of column names to parse as dates.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    
    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        Exception: For other potential errors during file loading.
    """
    try:
        print(f"Attempting to load data from: {filepath}")
        df = pd.read_csv(filepath, parse_dates=parse_dates)
        print("Data loaded successfully.")
        
        # --- Basic Preprocessing ---
        
        # Handle missing values - for simplicity, we'll forward-fill
        # This assumes that a missing value is likely the same as the previous reading.
        df.ffill(inplace=True)
        print("Missing values handled using forward-fill.")
        
        return df

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the data: {e}")
        raise

if __name__ == '__main__':
    # Example usage when running this script directly
    # Note: Adjust the path based on your project's root directory structure.
    # This might fail if run from src/optimizer; run from the project root.
    try:
        sample_path = 'data/sample_metrics.csv'
        metrics_df = load_metrics_data(sample_path, parse_dates=['timestamp'])
        print("\n--- Data Loader Example ---")
        print("Data loaded and preprocessed. First 5 rows:")
        print(metrics_df.head())
        print("\nData Info:")
        metrics_df.info()
    except FileNotFoundError:
        print("\nCould not run example: Ensure 'data/sample_metrics.csv' exists and you are running from the project root.")
    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
