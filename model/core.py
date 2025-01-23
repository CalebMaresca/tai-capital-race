import numpy as np
from dataclasses import dataclass, field
from model.utils import unconditional_to_conditional, get_TAI_probs_conditional_on_year

class Economy():
    def __init__(self, rho, beta, alpha, delta, zeta, TFP_0, g_N, g_TAI, unconditional_TAI_probs):
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.zeta = zeta
        self.TFP_0 = TFP_0
        self.g_N = g_N
        self.g_TAI = g_TAI
        self.unconditional_TAI_probs = unconditional_TAI_probs
        self.conditional_TAI_probs = unconditional_to_conditional(unconditional_TAI_probs)
        self.max_TAI_year = len(unconditional_TAI_probs)
        self.g = np.array([g_N] + [g_TAI] * self.max_TAI_year)
    
    def du(self, c):
        """Calculate marginal utility of consumption"""
        if self.rho == 1:
            return 1/c
        else:
            return c**(-self.rho)
    
    def dv_da_t(self, c, r):
        """Derivative of value function with respect to this period's assets"""
        return (1+r) * self.du(c)
    
    def dv_TAI_da_TAI(self, c, w, a_TAI, dv_TAI_da_TAI_next):
        """Derivative of value function with respect to assets in year TAI is invented"""
        return w * self.df_z_da(a_TAI) * self.du(c) + self.beta * dv_TAI_da_TAI_next
    
    def dv_TAI_da_TAI_final(self, c, w, a_TAI):
        """Final period derivative of value function with respect to assets in year TAI is invented"""
        return w * self.df_z_da(a_TAI) / (c**self.rho * (1 - self.beta * (1+self.g_TAI)**(1-self.rho)))
    
    def dv_TAI_da_t_full(self, c, r, w, a, dv_TAI_da_TAI_next):
        """
        Full derivative of value function with respect to this period's assets for year when TAI is invented.
        Full meaning that it includes the effect of an increase in assets on the share of AI labor received.
        """
        return (1+r + w*self.df_z_da(a)) * self.du(c) + self.beta * dv_TAI_da_TAI_next
    
    def dv_N_da_next(self, c, dv_N_next_da_next, dv_TAI_next_da_next, TAI_prob):
        """Derivative of value function with respect to next period's assets for the pre-TAI regime"""
        return - self.du(c) + self.beta * (TAI_prob * dv_TAI_next_da_next + (1 - TAI_prob) * dv_N_next_da_next)
    
    def dv_TAI_da_next(self, c, dv_da_next):
        """Derivative of value function with respect to next period's assets for the TAI regime"""
        return - self.du(c) + self.beta * dv_da_next
    
    def df_z_da(self, a):
        """Derivative of share of AI labor with respect to assets"""
        return self.zeta / a
    
    def rd(self, K, TFP):
        """
        Calculate the interest rate using marginal product of capital
        
        Parameters:
        K: aggregate capital
        TFP: total factor productivity
        
        Returns:
        r: capital rental rate
        """
        return self.alpha * (TFP / K)**(1 - self.alpha) - self.delta

    def r_to_w(self, r, TFP):
        """
        Convert capital rental rate to wage rate using Cobb-Douglas production function
        
        Parameters:
        r: capital rental rate
        
        Returns:
        w: wage rate
        """
        return (1 - self.alpha) * TFP * (self.alpha / (r + self.delta))**(self.alpha / (1 - self.alpha))
    
@dataclass
class TransitionPaths:
    """Class to store and manage transition paths"""
    T: int  # Number of time periods
    max_TAI_year: int  # Maximum year TAI can occur
    TFP: np.ndarray  # Total Factor Productivity [TAI_year × time]
    capital: np.ndarray  # Matrix of capital paths [TAI_year × time]
    wages: np.ndarray = field(init=False)
    capital_rental_rates: np.ndarray = field(init=False)
    consumption: np.ndarray = field(init=False)
    dv_da_next: np.ndarray = field(init=False)  # Derivative of value function w.r.t next period assets
    dv_da: np.ndarray = field(init=False)  # Derivative of value function w.r.t current assets
    dv_da_TAI: np.ndarray = field(init=False)  # Derivative of value function w.r.t assets when TAI invented
    interest_rates_1y: np.ndarray = field(init=False, default=None) # 1-year ahead interest rate paths
    interest_rates_30y: np.ndarray = field(init=False, default=None) # 30-year ahead interest rate paths
    

    def __post_init__(self):
        """Initialize arrays after the required fields are set"""
        # Initialize all arrays that should be zeros_like(capital)
        for field_name in ['wages', 'capital_rental_rates', 'consumption', 
                          'dv_da_next', 'dv_da', 'dv_da_TAI']:
            setattr(self, field_name, np.zeros_like(self.capital))

    @property
    def n_paths(self):
        """Number of paths (TAI never + possible TAI years)"""
        return self.max_TAI_year + 1
    
def create_initial_guess(economy: Economy, r_N: float, r_TAI: float, T: int) -> TransitionPaths:
    """
    Create initial guess for transition paths
    """

    alpha = economy.alpha
    delta = economy.delta
    TFP_0 = economy.TFP_0
    g_N = economy.g_N
    g_TAI = economy.g_TAI
    max_TAI_year = economy.max_TAI_year

    # Number of paths
    n_paths = max_TAI_year + 1  # Including "TAI never happens" path

    TFP = np.zeros((n_paths, T))
    TFP[0] = TFP_0 * (1 + g_N)**np.arange(T)
    for i in range(1, n_paths):
        TFP[i, :i] = TFP[0, :i]
        TFP[i, i:] = TFP[i, i-1] * (1 + g_TAI)**np.arange(1, T-i+1)

    # First calculate normalized steady state capitals
    K_N_norm = (alpha/(r_N + delta))**(1/(1-alpha))
    K_TAI_norm = (alpha/(r_TAI + delta))**(1/(1-alpha))
    
    # Initialize normalized capital paths matrix
    capital_norm = np.zeros((n_paths, T))
    
    # Generate normalized guesses for each path
    for i in range(n_paths):
        if i == 0:  # Never TAI path
            capital_norm[i, :] = K_N_norm  # Constant in normalized terms
        else:  # TAI occurs in year i
            # Until TAI: same as never-TAI normalized path
            capital_norm[i, :i+1] = K_N_norm
            # After TAI: interpolate to TAI steady state over 10 periods, then constant
            if T - (i+1) < 10:
                raise ValueError(f"Need at least 10 periods after TAI invention in year {i}, but only have {T-(i+1)}")
            transition_end = i + 11  # 10 periods after TAI year i
            if transition_end < T:  # Fill remaining periods with steady state
                capital_norm[i, transition_end:] = K_TAI_norm
    
    # Now unnormalize using the appropriate growth rates for each path
    capital = capital_norm * TFP
    # We want the initial guess for the capital in the year TAI is invented to stay in the old steady state, 
    # but since TFP grew at a higher rate, a constant normalized capital will lead to a higher than desired unnormalized capital
    # thuswe need to scale down the capital in the year TAI is invented
    for i in range(1, n_paths):
        capital[i,i] = capital[i,i] *(1 + g_N) / (1 + g_TAI)

        # Now interpolate in log space between TAI year and transition end
        transition_end = i + 11
        start_capital = capital[i, i]
        end_capital = capital[i, transition_end]
        capital[i, i+1:transition_end] = np.exp(np.linspace(np.log(start_capital), np.log(end_capital), transition_end - i - 1))
    
    print('Initial guess max capital:', np.max(capital))
    return TransitionPaths(T, max_TAI_year, TFP, capital)

def compute_transition_path(economy: Economy, T: int, lr: float = 0.1, tolerance: float = 1e-6, max_iterations: int = 100000) -> TransitionPaths:
    """
    Compute transition paths using vectorized operations
    
    Parameters:
    economy: Economy instance containing model parameters
    T: number of time periods
    tolerance: convergence tolerance
    max_iterations: maximum number of iterations
    """
    # 1. Calculate steady state values
    r_N = ((1 + economy.g_N)**economy.rho)/economy.beta - 1
    r_TAI = ((1 + economy.g_TAI)**economy.rho)/economy.beta - 1
    
    # 2-4. Initialize paths
    paths = create_initial_guess(economy, r_N, r_TAI, T)

    # Calculating some things we need for below, but don't change over iterations
    capital_T = paths.capital[:, -1] # Capital in final period is determined by equilibrium, we save this to make sure it doesn't change
    capital_after_T = capital_T*(1 + economy.g) # used to create capital_next below
    never_TAI_mask = np.zeros(paths.n_paths, dtype=bool)
    never_TAI_mask[0] = True
    
    for iteration in range(max_iterations):
        # 5. Calculate prices and consumption (vectorized)
        # Capital rental rates: shape (n_paths, T)
        paths.capital_rental_rates = economy.rd(paths.capital, paths.TFP)
        
        # Wages: shape (n_paths, T)
        paths.wages = economy.r_to_w(paths.capital_rental_rates, paths.TFP)
        
        # Consumption: shape (n_paths, T)
        capital_next = np.column_stack([paths.capital[:, 1:], capital_after_T])
        paths.consumption = (paths.wages + 
                           (1 + paths.capital_rental_rates) * paths.capital - 
                           capital_next)
        if np.any(paths.consumption < 0):
            raise ValueError("Consumption is negative! This might happen if the initial guess forces households to save too much, or if the learning rate is too high.")
        
        # 6. Calculate terminal period derivatives
        paths.dv_da[:, -1] = economy.dv_da_t(paths.consumption[:, -1], paths.capital_rental_rates[:, -1]) # Terminal derivatives w.r.t. current period assets

        TAI_year_assets = np.array([paths.capital[i, i] for i in range(1, paths.n_paths)])
        paths.dv_da_TAI[~never_TAI_mask, -1] = economy.dv_TAI_da_TAI_final( # Terminal derivatives w.r.t. assets in year TAI is invented
            paths.consumption[~never_TAI_mask, -1],
            paths.wages[~never_TAI_mask, -1],
            TAI_year_assets
        )

        # 7-8. Backward iteration
        for t in range(T-2, -1, -1):

            invented_TAI_mask = np.array([i > 0 and i <= t for i in range(paths.n_paths)])
            now_invented_TAI_mask = np.array([i > 0 and i == t for i in range(paths.n_paths)])
            past_invented_TAI_mask = invented_TAI_mask & ~now_invented_TAI_mask

            # Derivative of value function w.r.t. next period assets
            paths.dv_da_next[~invented_TAI_mask, t] = economy.dv_N_da_next(
                paths.consumption[~invented_TAI_mask, t],
                paths.dv_da[0, t+1],
                paths.dv_da_TAI[t+1, t+1] if t < economy.max_TAI_year else 0,
                economy.conditional_TAI_probs[t] if t < economy.max_TAI_year else 0
            )        
            paths.dv_da_next[invented_TAI_mask, t] = economy.dv_TAI_da_next(
                paths.consumption[invented_TAI_mask, t],
                paths.dv_da[invented_TAI_mask, t+1]
            )
            
            # Derivative of value function w.r.t. current period assets
            paths.dv_da[:, t] = economy.dv_da_t(paths.consumption[:, t], paths.capital_rental_rates[:, t])
            if t <= economy.max_TAI_year:
                paths.dv_da_TAI[t, t] = economy.dv_TAI_da_t_full(
                    paths.consumption[t, t],
                    paths.capital_rental_rates[t, t],
                    paths.wages[t, t],
                    paths.capital[t, t],
                    paths.dv_da_TAI[t, t+1]
                )

            # Derivative of value function w.r.t. assets in year TAI is invented
            paths.dv_da_TAI[past_invented_TAI_mask, t] = economy.dv_TAI_da_TAI( # TODO: check this
                paths.consumption[past_invented_TAI_mask, t],
                paths.wages[past_invented_TAI_mask, t],
                TAI_year_assets[past_invented_TAI_mask[1:]],
                paths.dv_da_TAI[past_invented_TAI_mask, t+1]
            )
        
        # 9. Update capital using gradient ascent
        new_capital = paths.capital.copy()
        # Convert to normalized values
        capital_norm = paths.capital[:, 1:] / paths.TFP[:, 1:]
        dv_da_next_norm = paths.dv_da_next[:, :-1] * paths.TFP[:, :-1]  # Multiply by TFP because derivative is with respect to unnormalized capital

        # Gradient ascent step on normalized values
        new_capital_norm = np.maximum(1e-10, capital_norm + lr * dv_da_next_norm)

        # Convert back to unnormalized values
        new_capital[:, 1:] = new_capital_norm * paths.TFP[:, 1:]
        
        # Check convergence
        max_diff = np.max(np.abs(new_capital_norm - capital_norm))
        ave_diff = np.mean(np.abs(new_capital_norm - capital_norm))
        if max_diff < tolerance:
            paths.capital = new_capital
            print(f"Converged after {iteration+1} iterations")
            return paths
            
        paths.capital = new_capital
        paths.capital[:, -1] = capital_T # Enforce terminal condition

        if (iteration+1) % 1000 == 0:
            print(f"Iteration {iteration+1}, max diff: {max_diff}, ave diff: {ave_diff}, learning rate: {lr}")

        if (iteration+1) % 10000 == 0:
            lr *= 0.9
    
    print(f"Did not converge after {max_iterations} iterations, final max diff: {max_diff}, final ave diff: {ave_diff}")
    return paths

def calculate_interest_rate_paths(paths: TransitionPaths, economy: Economy, n_years: int) -> TransitionPaths:
    """
    Calculate the 1y and 30y interest rate paths for each possible TAI scenario over n_years
    and store them in the TransitionPaths object
    
    Args:
        paths: TransitionPaths object containing consumption paths
        economy: Economy object containing model parameters
        n_years: Number of years to calculate rates for
    
    Returns:
        TransitionPaths: Updated paths object containing interest rate paths
    """
    n_paths = paths.n_paths
    
    # Initialize arrays for 1-year and 30-year rates
    paths.interest_rates_1y = np.zeros((n_paths, n_years))
    paths.interest_rates_30y = np.zeros((n_paths, n_years))
    
    # Calculate rates for no-TAI path
    paths.interest_rates_1y[0] = [calculate_interest_rate(paths, economy, year_0=y, horizon=1) 
                                 for y in range(n_years)]
    paths.interest_rates_30y[0] = [calculate_interest_rate(paths, economy, year_0=y, horizon=30) 
                                  for y in range(n_years)]
    
    # Calculate rates for each TAI year path
    for tai_year in range(1, n_paths):
        paths.interest_rates_1y[tai_year] = [calculate_interest_rate(paths, economy, year_0=y, horizon=1, TAI_year=tai_year) 
                                           if y > tai_year else paths.interest_rates_1y[0, y] 
                                           for y in range(n_years)]
        paths.interest_rates_30y[tai_year] = [calculate_interest_rate(paths, economy, year_0=y, horizon=30, TAI_year=tai_year) 
                                            if y > tai_year else paths.interest_rates_30y[0, y] 
                                            for y in range(n_years)]
    
    return paths

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