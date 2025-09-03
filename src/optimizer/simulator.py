import pandas as pd

def simulate_costs(df_allocations: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates the cost savings and energy reduction based on allocation decisions.

    This function calculates costs based on predefined rates for CPU, GPU, and memory,
    and then computes the savings achieved by the resource allocator's decisions.

    Args:
        df_allocations (pd.DataFrame): The DataFrame containing original usage
                                       and optimized allocation columns.

    Returns:
        pd.DataFrame: The input DataFrame with added columns for original cost,
                      optimized cost, cost saved, and energy saved.
    """
    # --- Define Cost and Energy Parameters ---
    # These values can be adjusted or moved to a configuration file.
    CPU_COST_PER_HOUR = 0.05        # Cost for 100% CPU utilization for one hour
    GPU_COST_PER_HOUR = 0.25        # Cost for 100% GPU utilization for one hour
    MEMORY_GB_COST_PER_HOUR = 0.01  # Cost per GB of memory per hour
    TOTAL_MEMORY_GB = 16            # Assuming a total of 16 GB memory for percentage calculations

    CPU_ENERGY_FACTOR_KWH = 0.001   # kWh saved per 1% CPU reduction per hour
    GPU_ENERGY_FACTOR_KWH = 0.003   # kWh saved per 1% GPU reduction per hour

    # --- Cost Calculation ---

    # 1. Calculate the original cost based on actual historical usage
    df_allocations['original_cost'] = (
        (df_allocations['cpu_usage'] / 100 * CPU_COST_PER_HOUR) +
        (df_allocations['gpu_usage'] / 100 * GPU_COST_PER_HOUR) +
        (df_allocations['memory_usage'] / 100 * TOTAL_MEMORY_GB * MEMORY_GB_COST_PER_HOUR)
    )

    # 2. Calculate the new, optimized cost based on the allocated resources
    df_allocations['optimized_cost'] = (
        (df_allocations['allocated_cpu'] / 100 * CPU_COST_PER_HOUR) +
        (df_allocations['allocated_gpu'] / 100 * GPU_COST_PER_HOUR) +
        (df_allocations['allocated_memory'] / 100 * TOTAL_MEMORY_GB * MEMORY_GB_COST_PER_HOUR)
    )

    # 3. Calculate the savings
    df_allocations['cost_saved'] = df_allocations['original_cost'] - df_allocations['optimized_cost']

    # --- Energy Savings Calculation ---

    # Calculate the reduction in resource usage. Use .clip(lower=0) to ignore cases where usage increased.
    cpu_reduction = (df_allocations['cpu_usage'] - df_allocations['allocated_cpu']).clip(lower=0)
    gpu_reduction = (df_allocations['gpu_usage'] - df_allocations['allocated_gpu']).clip(lower=0)

    df_allocations['energy_saved_kwh'] = (
        (cpu_reduction * CPU_ENERGY_FACTOR_KWH) +
        (gpu_reduction * GPU_ENERGY_FACTOR_KWH)
    )

    return df_allocations

