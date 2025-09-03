import os
import pandas as pd
import logging # <-- FIX: Import Python's standard logging library
from flask import Flask, render_template, jsonify

# --- Optimizer Module Imports ---
# These imports work because the start command is `cd src && gunicorn app:app`
from optimizer.data_loader import load_and_preprocess_data
from optimizer.feature_engineer import create_features
from optimizer.model import UsagePredictor
from optimizer.resource_allocator import allocate_resources
from optimizer.simulator import simulate_costs

# --- Create Dummy Data (if it doesn't exist) ---
def ensure_data_exists():
    """Creates a sample CSV file if one isn't found."""
    if not os.path.exists('../data/sample_metrics.csv'):
        print("Creating dummy metrics data...")
        os.makedirs('../data', exist_ok=True)
        timestamps = pd.to_datetime(pd.date_range(start='2025-01-01', periods=168, freq='h'))
        data = {
            'timestamp': timestamps,
            'cpu_usage': [max(10, min(90, 50 + 20 * (i % 24 - 12) + (i % 7) * 5)) for i in range(168)],
            'gpu_usage': [max(5, min(80, 40 + 15 * (i % 24 - 10) + (i % 7) * 3)) for i in range(168)],
            'memory_usage': [max(20, min(95, 60 + 10 * (i % 24 - 15) + (i % 7) * 4)) for i in range(168)]
        }
        pd.DataFrame(data).to_csv('../data/sample_metrics.csv', index=False)

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='../templates')

# Configure logging
if not app.debug:
    # In production, log to stderr.
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    app.logger.addHandler(stream_handler)

# --- API Routes ---
@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/api/run-optimization', methods=['POST'])
def run_optimization():
    """
    API endpoint to trigger the full optimization pipeline.
    """
    try:
        # 1. Load and preprocess data
        metrics_df = load_and_preprocess_data('../data/sample_metrics.csv')

        # 2. Engineer features
        features_df = create_features(metrics_df.copy())

        # 3. Define features (X) and targets (y)
        target_cols = ['cpu_usage', 'gpu_usage', 'memory_usage']
        # Drop original usage columns and any non-feature columns to create the feature set
        feature_cols = features_df.drop(columns=target_cols).columns
        
        features = features_df[feature_cols]
        targets = features_df[target_cols]

        # 4. Train model and predict for each target resource
        predictor = UsagePredictor()
        print("UsagePredictor initialized.")

        predictions = {}
        for col in target_cols:
            print(f"Training model for {col}...")
            # Pass both features and the specific target series for training
            predictor.train(features, targets[col], resource_name=col)
            
            print(f"Making predictions for {col}...")
            # Predict using the same features
            predictions[f'predicted_{col}'] = predictor.predict(features, resource_name=col)

        predictions_df = pd.DataFrame(predictions, index=features_df.index)

        # 5. Get resource allocations based on predictions
        allocations_df = allocate_resources(predictions_df)

        # 6. Combine original data with allocations for simulation
        combined_df = metrics_df.join(allocations_df)

        # 7. Simulate cost and energy savings
        results_df = simulate_costs(combined_df)

        # 8. Aggregate results for the final JSON response
        total_original_cost = results_df['original_cost'].sum()
        total_optimized_cost = results_df['optimized_cost'].sum()
        total_cost_savings = results_df['cost_saved'].sum()
        total_energy_saved = results_df['energy_saved_kwh'].sum()

        response_data = {
            "total_intervals": len(results_df),
            "original_cost": round(total_original_cost, 2),
            "optimized_cost": round(total_optimized_cost, 2),
            "cost_savings": round(total_cost_savings, 2),
            "cost_savings_percent": round((total_cost_savings / total_original_cost) * 100, 2) if total_original_cost > 0 else 0,
            "energy_saved_kwh": round(total_energy_saved, 2)
        }
        
        return jsonify({"status": "success", "data": response_data})

    except Exception as e:
        app.logger.error(f"An error occurred during optimization: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Main Execution ---
if __name__ == '__main__':
    ensure_data_exists()
    app.run(debug=True, port=5001)

