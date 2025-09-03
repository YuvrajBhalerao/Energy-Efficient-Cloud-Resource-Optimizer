import os
import pandas as pd
from flask import Flask, render_template, jsonify, logging

# --- Corrected Imports (No 'src.' prefix) ---
# Because the start command now runs from inside the 'src' directory,
# Python can find the 'optimizer' package directly.
from optimizer.data_loader import load_and_preprocess_data
from optimizer.feature_engineer import create_features
from optimizer.model import UsagePredictor
from optimizer.resource_allocator import allocate_resources
from optimizer.simulator import simulate_costs

# --- App Initialization ---
# The 'templates' folder is now one level up from this file's execution context.
app = Flask(__name__, template_folder='../templates')

# --- Dummy Data Generation ---
def create_dummy_data_if_not_exists():
    """Creates dummy CSV files if they don't already exist."""
    # The 'data' directory is one level up.
    data_dir = '../data'
    metrics_path = os.path.join(data_dir, 'sample_metrics.csv')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(metrics_path):
        print("Creating dummy metrics data...")
        timestamps = pd.to_datetime(pd.date_range(start='2025-01-01', periods=100, freq='h'))
        data = {
            'timestamp': timestamps,
            'cpu_usage': [50 + i * 0.5 + 10 * (i % 6) for i in range(100)],
            'gpu_usage': [30 + i * 0.2 - 5 * (i % 8) for i in range(100)],
            'memory_usage': [60 - i * 0.3 + 15 * (i % 5) for i in range(100)]
        }
        df_metrics = pd.DataFrame(data)
        df_metrics.to_csv(metrics_path, index=False)

# Run data creation at startup
create_dummy_data_if_not_exists()


# --- API Routes ---

@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/api/run-optimization', methods=['POST'])
def run_optimization_endpoint():
    """
    Full pipeline: Load data, train model, predict, allocate, and simulate.
    """
    try:
        metrics_path = '../data/sample_metrics.csv'
        df = load_and_preprocess_data(metrics_path)
        df = create_features(df)

        predictor = UsagePredictor()
        predictor.train(df, ['cpu_usage', 'gpu_usage', 'memory_usage'])
        predictions = predictor.predict(df)
        df['predicted_cpu'] = predictions[:, 0]
        df['predicted_gpu'] = predictions[:, 1]
        df['predicted_mem'] = predictions[:, 2]

        df_allocations = allocate_resources(df, 'predicted_cpu', 'predicted_gpu', 'predicted_mem')
        simulation_results = simulate_costs(df_allocations)

        original_cost_sum = simulation_results['original_cost'].sum()
        cost_savings_percent = (simulation_results['cost_saved'].sum() / original_cost_sum) * 100 if original_cost_sum > 0 else 0

        response_data = {
            "total_intervals": len(simulation_results),
            "original_cost": original_cost_sum,
            "optimized_cost": simulation_results['optimized_cost'].sum(),
            "cost_savings": simulation_results['cost_saved'].sum(),
            "cost_savings_percent": cost_savings_percent,
            "energy_saved_kwh": simulation_results['energy_saved_kwh'].sum()
        }
        
        return jsonify({"status": "success", "data": response_data})

    except FileNotFoundError:
        app.logger.error("Sample data file not found.")
        return jsonify({"status": "error", "message": "Sample data file not found. Ensure '../data/sample_metrics.csv' exists."}), 500
    except Exception as e:
        app.logger.error(f"An error occurred during optimization: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

