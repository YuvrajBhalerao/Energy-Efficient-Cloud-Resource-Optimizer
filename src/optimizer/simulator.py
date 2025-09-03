import pandas as pd

# --- Constants for Simulation ---
CPU_COST_PER_HOUR = 0.05  # Arbitrary cost per % CPU utilization per hour
GPU_COST_PER_HOUR = 0.15  # Arbitrary cost per % GPU utilization per hour
MEMORY_COST_PER_HOUR = 0.02 # Arbitrary cost per % Memory utilization per hour

# Energy constants (e.g., kWh per % utilization per hour)
CPU_ENERGY_KWH = 0.01
GPU_ENERGY_KWH = 0.03
MEMORY_ENERGY_KWH = 0.005

def simulate_costs(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates the cost and energy savings based on resource allocations.

    Args:
        combined_df (pd.DataFrame): DataFrame containing original usage
                                    and allocation decisions.

    Returns:
        pd.DataFrame: DataFrame with original cost, optimized cost, and savings.
    """
    results_df = combined_df.copy()

    # --- Calculate Original Cost & Energy ---
    results_df['original_cost'] = (
        (results_df['cpu_usage'] / 100 * CPU_COST_PER_HOUR) +
        (results_df['gpu_usage'] / 100 * GPU_COST_PER_HOUR) +
        (results_df['memory_usage'] / 100 * MEMORY_COST_PER_HOUR)
    )
    results_df['original_energy_kwh'] = (
        (results_df['cpu_usage'] / 100 * CPU_ENERGY_KWH) +
        (results_df['gpu_usage'] / 100 * GPU_ENERGY_KWH) +
        (results_df['memory_usage'] / 100 * MEMORY_ENERGY_KWH)
    )

    # --- Calculate Optimized Usage ---
    def get_allocated_usage(row, resource_type):
        """Calculates the new usage percentage based on the allocation decision."""
        # FIX: Column names now correctly match the output of resource_allocator.py
        # e.g., 'cpu_usage_allocation', 'gpu_usage_allocation'
        allocation_decision = row[f'{resource_type}_allocation']
        original_usage = row[resource_type]

        # Define how much to scale up or down (can be fine-tuned)
        SCALE_DOWN_TARGET = 30.0  # Target usage % after scaling down
        SCALE_UP_TARGET = 60.0    # Target usage % after scaling up (assumes new capacity)

        if allocation_decision == 'SCALE_DOWN':
            return SCALE_DOWN_TARGET
        elif allocation_decision == 'SCALE_UP':
            return SCALE_UP_TARGET
        else: # MAINTAIN
            return original_usage

    # Create new columns for the calculated optimized usage
    results_df['optimized_cpu_usage'] = results_df.apply(lambda row: get_allocated_usage(row, 'cpu_usage'), axis=1)
    results_df['optimized_gpu_usage'] = results_df.apply(lambda row: get_allocated_usage(row, 'gpu_usage'), axis=1)
    results_df['optimized_memory_usage'] = results_df.apply(lambda row: get_allocated_usage(row, 'memory_usage'), axis=1)

    # --- Calculate Optimized Cost & Energy ---
    results_df['optimized_cost'] = (
        (results_df['optimized_cpu_usage'] / 100 * CPU_COST_PER_HOUR) +
        (results_df['optimized_gpu_usage'] / 100 * GPU_COST_PER_HOUR) +
        (results_df['optimized_memory_usage'] / 100 * MEMORY_COST_PER_HOUR)
    )
    results_df['optimized_energy_kwh'] = (
        (results_df['optimized_cpu_usage'] / 100 * CPU_ENERGY_KWH) +
        (results_df['optimized_gpu_usage'] / 100 * GPU_ENERGY_KWH) +
        (results_df['optimized_memory_usage'] / 100 * MEMORY_ENERGY_KWH)
    )

    # --- Calculate Savings ---
    results_df['cost_saved'] = results_df['original_cost'] - results_df['optimized_cost']
    results_df['energy_saved_kwh'] = results_df['original_energy_kwh'] - results_df['optimized_energy_kwh']

    return results_df

