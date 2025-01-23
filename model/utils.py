from typing import List

def unconditional_to_conditional(unconditional_probs: List[float]) -> List[float]:
    """
    Convert unconditional probabilities to conditional probabilities.
    
    Args:
        unconditional_probs: List of unconditional probabilities
        
    Returns:
        List of conditional probabilities
    """
    if sum(unconditional_probs) > 1:
        raise ValueError("Sum of probabilities cannot exceed 1") # It's ok if it's less than 1, the remainder is the probability of never achieving TAI
    
    n = len(unconditional_probs)
    conditional_probs = [0.0] * n  # Initialize list of zeros
    sum_unconditional = 0.0  # Sum of unconditional probabilities up to time t-1

    for t in range(n):
        # Calculate conditional probability
        conditional_probs[t] = unconditional_probs[t] / (1 - sum_unconditional)
        # Update sum for next iteration 
        sum_unconditional += unconditional_probs[t]

    return conditional_probs

def get_TAI_probs_conditional_on_year(unconditional_probs: List[float], t: int) -> List[float]:
    """
    Get probabilities of TAI occurring in each future year, conditional on reaching year t without TAI.
    
    Args:
        unconditional_probs: List of unconditional probabilities for each year
        t: Current year (0-based index)
        
    Returns:
        List of conditional probabilities for each year (= 0 for t and earlier years)
    """
    if t >= len(unconditional_probs):
        return [0.0] * len(unconditional_probs)
        
    # Calculate sum of probabilities up to t-1
    sum_before_t = sum(unconditional_probs[:t])
    
    # Initialize output list
    conditional_probs = [0.0] * len(unconditional_probs)
    
    # Calculate conditional probabilities for remaining years
    for i in range(t, len(unconditional_probs)):
        conditional_probs[i] = unconditional_probs[i] / (1 - sum_before_t)
        
    return conditional_probs