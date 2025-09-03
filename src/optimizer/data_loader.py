import pandas as pd

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and performs basic preprocessing.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A preprocessed DataFrame.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # --- Preprocessing Steps ---

        # 1. Convert timestamp to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # 2. Set timestamp as the index
        df.set_index('timestamp', inplace=True)

        # 3. Handle any potential missing values (e.g., forward fill)
        df.ffill(inplace=True)

        return df

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        raise
    except Exception as e:
        print(f"An error occurred during data loading or preprocessing: {e}")
        raise

