import pandas as pd

def simulate_costs(combined_df: pd.DataFrame, simulation_params: dict) -> pd.DataFrame:
    """
    Simulates the cost and energy savings based on resource allocations.

    Args:
        combined_df (pd.DataFrame): DataFrame containing original usage
                                    and allocation decisions.
        simulation_params (dict): A dictionary containing the cost, energy,
                                  and scaling parameters for the simulation.

    Returns:
        pd.DataFrame: DataFrame with original cost, optimized cost, and savings.
    """
    results_df = combined_df.copy()

    # --- Use parameters from the input dictionary ---
    CPU_COST_PER_HOUR = simulation_params.get('cpu_cost_per_hour', 0.05)
    GPU_COST_PER_HOUR = simulation_params.get('gpu_cost_per_hour', 0.15)
    MEMORY_COST_PER_HOUR = simulation_params.get('memory_cost_per_hour', 0.02)
    CPU_ENERGY_KWH = simulation_params.get('cpu_energy_kwh', 0.01)
    GPU_ENERGY_KWH = simulation_params.get('gpu_energy_kwh', 0.03)
    MEMORY_ENERGY_KWH = simulation_params.get('memory_energy_kwh', 0.005)
    SCALE_DOWN_TARGET = simulation_params.get('scale_down_target', 30.0)
    SCALE_UP_TARGET = simulation_params.get('scale_up_target', 60.0)


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
        allocation_decision = row[f'{resource_type}_allocation']
        original_usage = row[resource_type]

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

