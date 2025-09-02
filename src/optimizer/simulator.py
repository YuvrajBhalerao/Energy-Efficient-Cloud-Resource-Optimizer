import pandas as pd

# --- Constants for Simulation (Azure-like pricing) ---
# These are hypothetical costs. Replace with actuals from a cloud provider.
COST_CPU_PER_HOUR = 0.05  # Cost for one vCPU core per hour
COST_GPU_PER_HOUR = 0.50  # Cost for one GPU (e.g., T4) per hour
COST_MEM_GB_PER_HOUR = 0.01 # Cost for 1 GB of RAM per hour

# Power constants (hypothetical)
POWER_CPU_WATT_PER_PERCENT = 1.5 # Watts per 1% of CPU utilization
POWER_GPU_WATT_PER_PERCENT = 3.0 # Watts per 1% of GPU utilization

def run_simulation(
    df_metrics: pd.DataFrame, 
    predictions: dict,
    allocation_logic: callable
) -> dict:
    """
    Simulates the cost and energy savings based on optimized resource allocation.

    Args:
        df_metrics (pd.DataFrame): DataFrame with actual usage columns 
                                   (e.g., 'cpu_usage', 'gpu_usage').
        predictions (dict): A dictionary with predicted usage, e.g., 
                            {'cpu': [pred1, pred2], 'gpu': [pred1, pred2]}.
        allocation_logic (callable): A function (like `allocate_resources`) that 
                                     returns an action string.

    Returns:
        dict: A dictionary summarizing the simulation results.
    """
    total_original_cost = 0
    total_optimized_cost = 0
    total_original_kwh = 0
    total_optimized_kwh = 0
    
    num_intervals = len(df_metrics)

    for i in range(num_intervals):
        # --- Original Cost Calculation ---
        current_cpu = df_metrics['cpu_usage'].iloc[i]
        current_gpu = df_metrics['gpu_usage'].iloc[i]
        
        # Assume original plan is always provisioned at 100% capacity
        original_cost_interval = COST_CPU_PER_HOUR + COST_GPU_PER_HOUR
        original_power_interval = (POWER_CPU_WATT_PER_PERCENT * current_cpu) + (POWER_GPU_WATT_PER_PERCENT * current_gpu)
        
        total_original_cost += original_cost_interval
        total_original_kwh += original_power_interval / 1000 # Convert watts to kWh

        # --- Optimized Cost Calculation ---
        pred_cpu = predictions['cpu'][i]
        
        # Get allocation action from the provided logic
        cpu_action = allocation_logic(current_usage=current_cpu, predicted_usage=pred_cpu)['action']
        
        # Simplified cost model for optimizer
        # Assume we can scale to a smaller instance (e.g., half size)
        if cpu_action == 'SCALE_DOWN':
            optimized_cost_interval = original_cost_interval * 0.5 # Half the cost
            optimized_power_interval = original_power_interval * 0.6 # Reduced power
        else: # MAINTAIN or SCALE_UP (assume we stay on the same instance size for simplicity)
            optimized_cost_interval = original_cost_interval
            optimized_power_interval = original_power_interval

        total_optimized_cost += optimized_cost_interval
        total_optimized_kwh += optimized_power_interval / 1000

    # --- Summarize Results ---
    cost_saved = total_original_cost - total_optimized_cost
    energy_saved_kwh = total_original_kwh - total_optimized_kwh
    
    return {
        "total_intervals": num_intervals,
        "original_cost": round(total_original_cost, 2),
        "optimized_cost": round(total_optimized_cost, 2),
        "cost_savings": round(cost_saved, 2),
        "cost_savings_percent": round((cost_saved / total_original_cost) * 100 if total_original_cost > 0 else 0, 2),
        "energy_saved_kwh": round(energy_saved_kwh, 2),
    }

if __name__ == '__main__':
    from resource_allocator import allocate_resources # For example usage

    print("\n--- Simulator Example ---")
    
    # Create sample data
    actual_data = pd.DataFrame({
        'cpu_usage': [25, 22, 28, 70, 80, 25],
        'gpu_usage': [10, 12, 15, 50, 55, 10]
    })

    # Create sample predictions
    predicted_vals = {
        'cpu': [20, 21, 25, 75, 78, 22],
        'gpu': [11, 13, 14, 52, 53, 12]
    }

    results = run_simulation(actual_data, predicted_vals, allocate_resources)
    
    print("\nSimulation Results:")
    for key, value in results.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
