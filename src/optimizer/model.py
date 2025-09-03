import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class UsagePredictor:
    """
    A wrapper for the machine learning model to predict resource usage.
    """
    def __init__(self):
        """
        Initializes the UsagePredictor. It uses a dictionary to store a
        separate model for each resource type (e.g., 'cpu', 'gpu').
        """
        self.models = {}

    def train(self, features: pd.DataFrame, target: pd.Series, resource_name: str):
        """
        Trains or retrains a model for a specific resource.

        FIX: Added 'resource_name' to the function signature to accept the
        argument being passed from app.py.

        Args:
            features (pd.DataFrame): The input features for training.
            target (pd.Series): The target values (e.g., actual cpu_usage).
            resource_name (str): The name of the resource to model (e.g., 'cpu_usage').
        """
        if resource_name not in self.models:
            # If a model for this resource doesn't exist, create one.
            self.models[resource_name] = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                min_samples_split=10,
                n_jobs=-1  # Use all available cores for training
            )

        model = self.models[resource_name]
        
        # Fit the model to the data
        model.fit(features, target)
        print(f"Model for {resource_name} trained successfully.")

    def predict(self, features: pd.DataFrame, resource_name: str) -> pd.Series:
        """
        Makes predictions for a specific resource.

        Args:
            features (pd.DataFrame): The features to use for prediction.
            resource_name (str): The name of the resource model to use.

        Returns:
            pd.Series: The predicted values.
        """
        if resource_name not in self.models:
            raise ValueError(f"No model trained for resource: {resource_name}. Please train the model first.")
        
        model = self.models[resource_name]
        predictions = model.predict(features)
        
        return pd.Series(predictions, index=features.index, name=f'predicted_{resource_name}')

