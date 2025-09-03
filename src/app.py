import os
import logging
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np

# Import optimizer modules
from optimizer.data_loader import load_and_preprocess_data
from optimizer.feature_engineer import create_features
from optimizer.model import UsagePredictor
from optimizer.resource_allocator import allocate_resources
from optimizer.simulator import simulate_costs

# --- App Initialization ---
# Use __name__ so Flask knows where to look for templates and static files
# The path is relative to the `src` directory where this file lives.
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# --- Data & Model Initialization ---
DATA_FILE = './data/sample_metrics.csv'
# Initialize one predictor instance to be shared across requests
predictor = UsagePredictor()

def ensure_data_exists():
    """Creates dummy data if the CSV file doesn't exist."""
    # Correct the path since we are running from within the `src` directory
    if not os.path.exists(DATA_FILE):
        print("Creating dummy metrics data...")
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        timestamps = pd.to_datetime(pd.date_range(start="2025-01-01", periods=168, freq="h"))
        data = {
            "timestamp": timestamps,
            "cpu_usage": np.random.uniform(30, 90, size=168),
            "gpu_usage": np.random.uniform(20, 80, size=168),
            "memory_usage": np.random.uniform(40, 95, size=168),
        }
        df = pd.DataFrame(data)
        df.to_csv(DATA_FILE, index=False)
        print("Dummy data created at", DATA_FILE)

# Run this once when the app starts
ensure_data_exists()

# --- API Routes ---
@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/api/run-optimization', methods=['POST'])
def run_optimization():
    """
    API endpoint to trigger the optimization process and return results.
    """
    try:
        # Get simulation parameters from the user's request
        simulation_params = request.get_json()
        
        df = load_and_preprocess_data(DATA_FILE)
        features_df = create_features(df.copy())
        
        features = [col for col in features_df.columns if col not in ['timestamp', 'cpu_usage', 'gpu_usage', 'memory_usage']]
        targets = ['cpu_usage', 'gpu_usage', 'memory_usage']
        X = features_df[features]

        all_predictions = pd.DataFrame(index=X.index)
        for target in targets:
            y = features_df[target]
            predictor.train(X, y, resource_name=target)
            predictions = predictor.predict(X, resource_name=target)
            all_predictions[predictions.name] = predictions
        
        allocations = allocate_resources(all_predictions)
        combined_df = df.join(all_predictions).join(allocations)
        
        # Pass the user-defined parameters to the simulator
        results_df = simulate_costs(combined_df, simulation_params)

        total_original_cost = results_df['original_cost'].sum()
        total_optimized_cost = results_df['optimized_cost'].sum()
        total_cost_savings = results_df['cost_saved'].sum()
        total_energy_saved = results_df['energy_saved_kwh'].sum()
        
        if total_original_cost > 0:
            percent_savings = (total_cost_savings / total_original_cost) * 100
        else:
            percent_savings = 0
            
        summary = {
            "total_intervals": len(results_df),
            "original_cost": round(total_original_cost, 2),
            "optimized_cost": round(total_optimized_cost, 2),
            "cost_savings": round(total_cost_savings, 2),
            "cost_savings_percent": round(percent_savings, 2),
            "energy_saved_kwh": round(total_energy_saved, 2),
        }
        
        return jsonify({"status": "success", "data": summary})

    except Exception as e:
        app.logger.error(f"An error occurred during optimization: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)

