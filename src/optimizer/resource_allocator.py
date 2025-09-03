import pandas as pd

def allocate_resources(predicted_usage: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the appropriate resource allocation action based on predicted usage.

    FIX: The function signature now correctly accepts the 'predicted_usage'
    DataFrame passed from app.py.

    Args:
        predicted_usage (pd.DataFrame): A DataFrame with columns for
                                        predicted CPU, GPU, and memory usage.

    Returns:
        pd.DataFrame: A DataFrame with the allocation decision ('SCALE_UP',
                      'SCALE_DOWN', 'MAINTAIN') for each resource.
    """
    allocations = pd.DataFrame(index=predicted_usage.index)

    # Define thresholds (these can be fine-tuned)
    SCALE_UP_THRESHOLD = 80.0
    SCALE_DOWN_THRESHOLD = 40.0

    for column in predicted_usage.columns:
        # Extract the base resource name (e.g., 'cpu_usage' from 'predicted_cpu_usage')
        resource_name = column.replace('predicted_', '')
        
        def get_allocation(prediction):
            if prediction > SCALE_UP_THRESHOLD:
                return 'SCALE_UP'
            elif prediction < SCALE_DOWN_THRESHOLD:
                return 'SCALE_DOWN'
            else:
                return 'MAINTAIN'

        allocations[f'{resource_name}_allocation'] = predicted_usage[column].apply(get_allocation)

    return allocations

