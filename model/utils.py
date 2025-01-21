import numpy as np
from typing import List
from model.transition_paths import TransitionPaths
from model.economy import Economy

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

def calculate_interest_rate(paths: TransitionPaths, economy: Economy, year_0: int, horizon: int, TAI_year: int = 0) -> float:
    """
    Calculate the interest rate starting from year_0 over the specified horizon assuming no TAI.
    
    Args:
        paths: TransitionPaths object containing consumption paths
        economy: Economy object containing model parameters
        year_0: Starting year for calculation
        horizon: Number of years for rate calculation
        TAI_year: Year TAI is invented. If 0, assumes no TAI so far. If TAI_year is provided, it must be < year_0.
        
    Returns:
        float: Annualized interest rate
    """
    if TAI_year >= year_0 and TAI_year > 0:
        raise ValueError("TAI year must be less than the starting year")
    elif TAI_year > 0:
        # Get consumption at start and end years
        c_0 = paths.consumption[TAI_year, year_0]
        c_T = paths.consumption[TAI_year, year_0 + horizon]

        # Calculate marginal utilities
        mu_0 = economy.du(c_0)
        mu_T = economy.du(c_T)
        ratio = mu_0 / mu_T

        # Convert to annualized rate
        r = (ratio)**(1/horizon) / economy.beta - 1
        return r
    else:
        # Get consumption at start and end years
        c_0 = paths.consumption[:, year_0]
        c_T = paths.consumption[:, year_0 + horizon]
        
        # Calculate marginal utilities
        mu_0 = economy.du(c_0)
        mu_T = economy.du(c_T)
        
        # Get probabilities for each path
        if year_0 == 0:
            probs = np.array([1 - sum(economy.unconditional_TAI_probs)] + 
                            list(economy.unconditional_TAI_probs))
        else:
            conditional_probs = get_TAI_probs_conditional_on_year(
                economy.unconditional_TAI_probs, 
                year_0
            )
            probs = np.array([1 - sum(conditional_probs)] + conditional_probs)
        
        # Calculate expected marginal utility ratio
        ratio = mu_0[0] / (np.sum(probs * mu_T))
        
        # Convert to annualized rate
        r = (ratio)**(1/horizon) / economy.beta - 1
        
        return r