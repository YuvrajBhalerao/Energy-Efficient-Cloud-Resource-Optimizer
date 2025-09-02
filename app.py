import os
import pandas as pd
from flask import Flask, jsonify, render_template

# Import your custom modules from the optimizer package
from src.optimizer.data_loader import load_metrics_data
from src.optimizer.feature_engineer import create_time_features, create_rolling_features
from src.optimizer.model import UsagePredictor
from src.optimizer.resource_allocator import allocate_resources
from src.optimizer.simulator import run_simulation

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Global Variables & Constants ---
# Define paths to data. Using os.path.join for cross-platform compatibility.
DATA_DIR = "data"
METRICS_FILEPATH = os.path.join(DATA_DIR, 'sample_metrics.csv')

# This would ideally be a pre-trained model saved to disk.
# For this demo, we'll train it on the fly.
CPU_MODEL = UsagePredictor()


def create_dummy_data():
    """Creates dummy CSV files if they don't exist to ensure the app can run."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(METRICS_FILEPATH):
        print(f"Creating dummy data at: {METRICS_FILEPATH}")
        # Create a time series for one week
        timestamps = pd.to_datetime(pd.date_range(start='2023-01-01', periods=168, freq='H'))
        
        # Simulate cyclical daily patterns for CPU and GPU
        hours = timestamps.hour
        cpu_usage = 30 + 20 * (1 + pd.np.sin(2 * pd.np.pi * hours / 24)) + pd.np.random.rand(168) * 10
        gpu_usage = 15 + 10 * (1 + pd.np.sin(2 * pd.np.pi * (hours - 6) / 24)) + pd.np.random.rand(168) * 5 # Peaking later than CPU
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cpu_usage': cpu_usage.clip(0, 100),
            'gpu_usage': gpu_usage.clip(0, 100),
            'memory_usage': cpu_usage * 0.5 + pd.np.random.rand(168) * 20
        })
        df.to_csv(METRICS_FILEPATH, index=False)


# --- Main Application Logic ---

def run_optimization_pipeline():
    """
    Orchestrates the full data processing, modeling, and simulation pipeline.
    """
    # 1. Load Data
    df = load_metrics_data(METRICS_FILEPATH, parse_dates=['timestamp'])

    # 2. Engineer Features
    df = create_time_features(df, 'timestamp')
    df = create_rolling_features(df, columns=['cpu_usage', 'gpu_usage'], window_sizes=[3, 6])
    df.dropna(inplace=True) # Drop rows with NaN from rolling features

    # 3. Train Model and Predict
    # In a real-world app, you'd load a pre-trained model.
    # Here, we train it on the full dataset for demonstration purposes.
    features = ['hour', 'day_of_week', 'cpu_usage_rolling_mean_3', 'cpu_usage_rolling_mean_6']
    target = 'cpu_usage'
    CPU_MODEL.train(df, features, target)
    
    # Predict on the same data to simulate a real-time scenario
    cpu_predictions = CPU_MODEL.predict(df)
    
    # For GPU, we'll just pass a placeholder or a simple shift for this example
    gpu_predictions = df['gpu_usage'].shift(-1).fillna(method='ffill').values

    predictions = {'cpu': cpu_predictions, 'gpu': gpu_predictions}

    # 4. Run Simulation
    simulation_results = run_simulation(
        df_metrics=df,
        predictions=predictions,
        allocation_logic=allocate_resources
    )

    return simulation_results


# --- API Routes ---

@app.route('/')
def home():
    """
    Renders the main dashboard page.
    The actual 'index.html' needs to be created in a 'templates' folder.
    """
    # This will fail if /templates/index.html does not exist.
    # We will assume it exists and handles the API call.
    return render_template('index.html')

@app.route('/api/run-optimization', methods=['POST'])
def api_run_optimization():
    """
    API endpoint that triggers the optimization pipeline and returns the results.
    """
    try:
        results = run_optimization_pipeline()
        return jsonify({
            "status": "success",
            "data": results
        })
    except FileNotFoundError:
        return jsonify({
            "status": "error",
            "message": f"Data file not found at {METRICS_FILEPATH}. Please ensure the data exists."
        }), 404
    except Exception as e:
        # It's good practice to log the error here
        print(f"An error occurred: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# --- Main Entry Point ---

if __name__ == '__main__':
    # Create dummy data for local development if it's missing
    create_dummy_data()
    # Run the Flask app
    # The host='0.0.0.0' makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
