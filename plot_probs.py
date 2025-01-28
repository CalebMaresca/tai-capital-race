import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model.utils import unconditional_to_conditional
from scipy import stats
from scipy.optimize import minimize

def discretized_lognormal_pmf(x, mu, sigma, lower_bound=None, upper_bound=None):
    """
    Calculate PMF of a discretized lognormal distribution using [x-1, x] intervals
    """
    x = np.asarray(x)
    
    # Calculate CDF at x and x-1
    cdf_x = stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))
    cdf_x_minus_1 = stats.lognorm.cdf(x - 1, s=sigma, scale=np.exp(mu))
    
    # PMF is the difference between consecutive CDFs
    pmf = cdf_x - cdf_x_minus_1
    
    # Normalize if bounds are provided
    if lower_bound is not None and upper_bound is not None:
        total_prob = stats.lognorm.cdf(upper_bound, s=sigma, scale=np.exp(mu)) - \
                     stats.lognorm.cdf(lower_bound, s=sigma, scale=np.exp(mu))
        pmf = pmf / total_prob
        
    return pmf*0.8

def compute_annual_betanbinom_pmf(n_param, a, b, scaling=0.8, max_years=30):
    """
    Compute annual beta-negative binomial PMF.
    
    Args:
        n_param: Either an integer or a list/array of integers representing possible n values with their probabilities
        a, b: Beta distribution parameters
        scaling: Total probability that TAI will be achieved (default: 0.8)
        max_years: Maximum number of years to compute probabilities for
    
    Returns:
        Array of annual probabilities
    """
    if isinstance(n_param, (int, float)):
        n_values = [int(n_param)]
        n_probs = [1.0]
    else:
        n_values, n_probs = n_param
        
    final_pmf = np.zeros(max_years)
    
    for n, n_prob in zip(n_values, n_probs):
        # Compute monthly probabilities
        monthly_pmf = [0] * (n-1) + list(stats.betanbinom.pmf(range(0, 12*max_years+12), n, a, b))
        
        # Convert to annual by summing every 12 elements
        annual_pmf = [sum(monthly_pmf[12*i-11:12*i+1]) 
                     for i in range(1, len(monthly_pmf)//12 + 1)][:max_years]
        
        final_pmf += np.array(annual_pmf) * n_prob
    
    # Apply scaling
    return list(final_pmf * scaling)

def fit_betanbinom_params(target_probs, max_n=15, a_init=4, b_init=60, scaling_init=0.8, bounds=None):
    """
    Fit beta-negative binomial parameters to match target probabilities.
    
    Args:
        target_probs: Array of target probabilities to match
        max_n: Maximum value of n to consider in the distribution
        a_init: Initial guess for a parameter
        b_init: Initial guess for b parameter
        scaling_init: Initial guess for scaling parameter
        bounds: Optional bounds for parameters (n_probs, a, b, scaling)
    
    Returns:
        Tuple of ((n_values, n_probs), a, b, scaling) where n_values and n_probs define
        the distribution over n
    """
    n_values = list(range(1, max_n + 1))
    n_probs_init = [1.0 / max_n] * max_n  # Start with uniform distribution
    
    if bounds is None:
        # Bounds for n_probs (must sum to 1), then a, b, scaling
        n_prob_bounds = [(0, 1)] * max_n
        other_bounds = [(0.1, 10), (1, 100), (0.1, 1.0)]
        bounds = n_prob_bounds + other_bounds
    
    def objective(params):
        n_probs = params[:max_n]
        a, b, scaling = params[max_n:]
        
        # Normalize n_probs to sum to 1
        n_probs = np.array(n_probs)
        n_probs = n_probs / np.sum(n_probs)
        
        predicted = compute_annual_betanbinom_pmf((n_values, n_probs), a, b, scaling, len(target_probs))
        return np.sum((np.array(predicted) - np.array(target_probs))**2)
    
    initial_params = n_probs_init + [a_init, b_init, scaling_init]
    
    result = minimize(
        objective,
        x0=initial_params,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    # Extract and normalize the n probabilities
    n_probs_result = result.x[:max_n]
    n_probs_result = n_probs_result / np.sum(n_probs_result)
    
    # Return the distribution of n along with other parameters
    return ((n_values, list(n_probs_result)), result.x[max_n], result.x[max_n + 1], result.x[max_n + 2])

# Use the baseline TAI probs from analysis.py
cotra_pmf = [
    0.025, 0.03, 0.035, 0.045, 0.05, 0.055, 0.06, 0.06, 0.055, 0.051,
    0.048, 0.045, 0.04, 0.035, 0.03, 0.027, 0.022, 0.02, 0.018, 0.016,
    0.014, 0.012, 0.01, 0.0085, 0.007, 0.0055, 0.004, 0.003, 0.002, 0.001
]

# Load Metaculus forecast data
forecast_df = pd.read_csv('data/forecast_data.csv')
metaculus_cdf = eval(forecast_df['Continuous CDF'].iloc[-1])  # Get last CDF

# Convert CDF to PDF starting from 2025 (5 years from 2020)
# and truncate at 30 years from 2025
metaculus_pmf = []
for i in range(5, min(35, len(metaculus_cdf)-1)):  # From 2025 to 2025+30 years
    pmf_value = metaculus_cdf[i+1] - metaculus_cdf[i]
    metaculus_pmf.append(pmf_value)

# Get conditional probabilities for both forecasts
cotra_conditional = unconditional_to_conditional(cotra_pmf)
metaculus_conditional = unconditional_to_conditional(metaculus_pmf)

# Create years array for x-axis
years = range(1, len(cotra_pmf)+1)
years_metaculus = range(1, len(metaculus_pmf)+1)

# Fit the parameters to both probability distributions
print("Fitting baseline probabilities:")
n_dist, a, b, scaling = fit_betanbinom_params(cotra_pmf)
n_values, n_probs = n_dist
print(f"Baseline fitted parameters:")
print(f"n distribution:")
for n, p in zip(n_values, n_probs):
    if p > 0.001:  # Only print probabilities > 0.1%
        print(f"  n={n}: {p:.3f}")
print(f"a={a:.2f}, b={b:.2f}, scaling={scaling:.2f}")

print("\nFitting Metaculus probabilities:")
n_dist_meta, a_meta, b_meta, scaling_meta = fit_betanbinom_params(metaculus_pmf)
n_values_meta, n_probs_meta = n_dist_meta
print(f"Metaculus fitted parameters:")
print(f"n distribution:")
for n, p in zip(n_values_meta, n_probs_meta):
    if p > 0.001:  # Only print probabilities > 0.1%
        print(f"  n={n}: {p:.3f}")
print(f"a={a_meta:.2f}, b={b_meta:.2f}, scaling={scaling_meta:.2f}")

# Compute fitted distributions
annual_betanbinom_pmf_cotra = compute_annual_betanbinom_pmf(n_dist, a, b, scaling, max_years=len(cotra_pmf))
conditional_betanbinom_pmf_cotra = unconditional_to_conditional(annual_betanbinom_pmf_cotra)

annual_betanbinom_pmf_meta = compute_annual_betanbinom_pmf(n_dist_meta, a_meta, b_meta, scaling_meta, max_years=len(metaculus_pmf))
conditional_betanbinom_pmf_meta = unconditional_to_conditional(annual_betanbinom_pmf_meta)

# Create the plot with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot Cotra distributions
ax1.plot(years, cotra_pmf, 'b-', label='Cotra Unconditional')
ax1.plot(years, cotra_conditional, 'r-', label='Cotra Conditional')
ax1.plot(years, annual_betanbinom_pmf_cotra, 'm-', label='Cotra Fitted Beta-NB')
ax1.plot(years, conditional_betanbinom_pmf_cotra, 'c-', label='Cotra Fitted Conditional')

# Add text showing sum of Cotra probabilities
sum_cotra = sum(cotra_pmf)
ax1.text(0.02, 0.95, f'Sum of Cotra probabilities: {sum_cotra:.3f}', 
         transform=ax1.transAxes)

# Customize first subplot
ax1.set_title('Cotra TAI Probability Distribution')
ax1.set_xlabel('Years from 2025')
ax1.set_ylabel('Probability')
ax1.grid(True)
ax1.legend()

# Plot Metaculus distributions
ax2.plot(years_metaculus, metaculus_pmf, 'b-', label='Metaculus Unconditional')
ax2.plot(years_metaculus, metaculus_conditional, 'r-', label='Metaculus Conditional')
ax2.plot(years_metaculus, annual_betanbinom_pmf_meta, 'm-', label='Metaculus Fitted Beta-NB')
ax2.plot(years_metaculus, conditional_betanbinom_pmf_meta, 'c-', label='Metaculus Fitted Conditional')

# Add text showing sum of Metaculus probabilities
sum_metaculus = sum(metaculus_pmf)
ax2.text(0.02, 0.95, f'Sum of Metaculus probabilities: {sum_metaculus:.3f}', 
         transform=ax2.transAxes)

# Customize second subplot
ax2.set_title('Metaculus TAI Probability Distribution')
ax2.set_xlabel('Years from 2025')
ax2.set_ylabel('Probability')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()