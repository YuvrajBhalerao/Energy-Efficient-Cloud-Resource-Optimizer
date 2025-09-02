def allocate_resources(
    current_usage: float, 
    predicted_usage: float, 
    scale_up_threshold: float = 75.0,
    scale_down_threshold: float = 30.0
) -> dict:
    """
    Determines the resource allocation action based on current and predicted usage.

    This is a simple rule-based allocator. A more complex system might consider
    prediction confidence, rate of change, or business rules.

    Args:
        current_usage (float): The current resource usage percentage.
        predicted_usage (float): The predicted resource usage percentage for the next interval.
        scale_up_threshold (float): The percentage threshold above which to scale up.
        scale_down_threshold (float): The percentage threshold below which to scale down.

    Returns:
        dict: A dictionary containing the suggested action and a reason.
    """
    action = "MAINTAIN"
    reason = "Usage is within optimal thresholds."

    if predicted_usage > scale_up_threshold:
        action = "SCALE_UP"
        reason = f"Predicted usage ({predicted_usage:.1f}%) exceeds scale-up threshold of {scale_up_threshold}%. Proactive scaling recommended."
    elif predicted_usage < scale_down_threshold and current_usage < scale_down_threshold:
        action = "SCALE_DOWN"
        reason = f"Both current ({current_usage:.1f}%) and predicted ({predicted_usage:.1f}%) usage are below scale-down threshold of {scale_down_threshold}%. Resources can be conserved."
    
    return {"action": action, "reason": reason}

if __name__ == '__main__':
    # Example Usage
    print("\n--- Resource Allocator Example ---")

    # Scenario 1: High predicted usage -> Scale up
    result1 = allocate_resources(current_usage=70.0, predicted_usage=85.0)
    print(f"Scenario 1 (High Prediction): {result1}")

    # Scenario 2: Low usage -> Scale down
    result2 = allocate_resources(current_usage=25.0, predicted_usage=20.0)
    print(f"Scenario 2 (Low Prediction): {result2}")

    # Scenario 3: Stable usage -> Maintain
    result3 = allocate_resources(current_usage=50.0, predicted_usage=55.0)
    print(f"Scenario 3 (Stable Prediction): {result3}")

    # Scenario 4: Usage spike predicted, but current is low -> Maintain (prevents thrashing)
    result4 = allocate_resources(current_usage=28.0, predicted_usage=45.0)
    print(f"Scenario 4 (Spike Predicted): {result4}")
