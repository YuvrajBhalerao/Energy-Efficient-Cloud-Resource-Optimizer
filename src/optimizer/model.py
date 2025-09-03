import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from typing import List

class UsagePredictor:
    """
    A machine learning model to predict resource usage.
    This class wraps a RandomForestRegressor model.
    """
    def __init__(self, model_params: dict = None):
        """
        Initializes the UsagePredictor.

        Args:
            model_params (dict, optional): Parameters for the RandomForestRegressor.
                                           Defaults to a basic configuration.
        """
        if model_params is None:
            model_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        
        self.model = RandomForestRegressor(**model_params)
        self.features: List[str] = []
        self.target: str = ""
        print("UsagePredictor initialized.")

    def train(self, df: pd.DataFrame, features: List[str], target: str):
        """
        Trains the model on the provided data.

        Args:
            df (pd.DataFrame): The training DataFrame.
            features (List[str]): The list of feature column names.
            target (str): The name of the target column to predict.
        """
        self.features = features
        self.target = target
        
        X = df[self.features]
        y = df[self.target]
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training model to predict '{self.target}' with features: {self.features}")
        self.model.fit(X_train, y_train)
        
        # Evaluate on the test set
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model training complete. Test MSE: {mse:.4f}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Makes predictions on new data.

        Args:
            df (pd.DataFrame): DataFrame with the necessary features.

        Returns:
            pd.Series: A pandas Series with the predictions.
        """
        if not self.features:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
            
        print(f"Making predictions for '{self.target}'.")
        # Ensure columns are in the same order as during training
        X_predict = df[self.features]
        return self.model.predict(X_predict)

    def save_model(self, filepath: str):
        """Saves the trained model to a file."""
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """Loads a model from a file."""
        print(f"Loading model from {filepath}")
        return joblib.load(filepath)

if __name__ == '__main__':
    # Example Usage
    print("\n--- Model Example ---")
    data = {
        'hour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'day_of_week': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'cpu_usage_rolling_mean_3': [10, 12, 11, 15, 14, 18, 20, 22, 21, 25],
        'cpu_usage': [12, 11, 15, 14, 18, 20, 22, 21, 25, 28] # Target
    }
    example_df = pd.DataFrame(data)

    features_to_use = ['hour', 'day_of_week', 'cpu_usage_rolling_mean_3']
    target_to_predict = 'cpu_usage'

    # 1. Initialize and train the model
    predictor = UsagePredictor()
    predictor.train(example_df, features_to_use, target_to_predict)

    # 2. Make a prediction on new data (using last row as an example)
    new_data = example_df.tail(1)[features_to_use]
    prediction = predictor.predict(new_data)
    print(f"\nExample prediction for data:\n{new_data}")
    print(f"Predicted CPU Usage: {prediction[0]:.2f}")
    
    # 3. Save and load the model
    model_path = 'temp_model.joblib'
    predictor.save_model(model_path)
    loaded_predictor = UsagePredictor.load_model(model_path)
    loaded_prediction = loaded_predictor.predict(new_data)
    print(f"Prediction from loaded model: {loaded_prediction[0]:.2f}")

    # Clean up the temporary file
    import os
    os.remove(model_path)
