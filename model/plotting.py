import matplotlib.pyplot as plt
from typing import List, Optional, Literal
from model.economy import TransitionPaths

class TransitionPathVisualizer:
    """
    Visualizer for transition paths data.
    
    Attributes:
        paths (TransitionPaths): The transition paths data to visualize
    """
    
    # Define valid plot types
    VALID_VARIABLES = Literal[
        'capital_rental_rates', 'capital', 'wages', 
        'consumption', 'TFP', 'dv_da_next', 
        'dv_da', 'dv_da_TAI', 'interest_rates_1y', 'interest_rates_30y'
    ]
    
    def __init__(self, paths: TransitionPaths):
        self.paths = paths
        
        # Map variable names to their display properties
        self.variable_properties = {
            'capital_rental_rates': {'title': 'Capital Rental Rate Paths', 'ylabel': 'Capital Rental Rate'},
            'capital': {'title': 'Capital Paths', 'ylabel': 'Capital'},
            'wages': {'title': 'Wage Paths', 'ylabel': 'Wages'},
            'consumption': {'title': 'Consumption Paths', 'ylabel': 'Consumption'},
            'TFP': {'title': 'Total Factor Productivity Paths', 'ylabel': 'TFP'},
            'dv_da_next': {'title': 'Value Function Derivative (Next Period)', 'ylabel': 'dV/da\''},
            'dv_da': {'title': 'Value Function Derivative (Current Period)', 'ylabel': 'dV/da'},
            'dv_da_TAI': {'title': 'Value Function Derivative (TAI Period)', 'ylabel': 'dV/da_TAI'},
            'interest_rates_1y': {'title': '1-Year Interest Rate Paths', 'ylabel': '1-Year Interest Rate'},
            'interest_rates_30y': {'title': '30-Year Interest Rate Paths', 'ylabel': '30-Year Interest Rate'}
        }

    def plot_variable(self, 
                     variable: VALID_VARIABLES,
                     selected_paths: Optional[List[int]] = None,
                     time_periods: Optional[int] = None,
                     title: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     figsize: tuple = (12, 6)) -> None:
        """
        Plot specified variable for selected paths.
        
        Args:
            variable: Name of the variable to plot
            selected_paths: List of path indices to plot. If None, plots all paths
            time_periods: Number of time periods to plot. If None, plots all periods
            title: Custom title for the plot. If None, uses default
            ylabel: Custom y-axis label. If None, uses default
            figsize: Figure size as (width, height)
        """
        if not hasattr(self.paths, variable):
            raise ValueError(f"Invalid variable: {variable}")
            
        data = getattr(self.paths, variable)
        
        plt.figure(figsize=figsize)
        
        # Determine which paths to plot
        paths_to_plot = selected_paths or range(data.shape[0])
        
        # Determine time range
        t_max = time_periods or data.shape[1]
        
        # Plot each path
        for i in paths_to_plot:
            label = 'No TAI' if i == 0 else f'TAI in year {i}'
            plt.plot(data[i, :t_max], label=label)
            
        # Set labels and title
        plt.xlabel('Time Period')
        plt.ylabel(ylabel or self.variable_properties[variable]['ylabel'])
        plt.title(title or self.variable_properties[variable]['title'])
        plt.legend()
        plt.grid(True)
    
    def plot_comparison(self,
                       variables: List[VALID_VARIABLES],
                       selected_paths: Optional[List[int]] = None,
                       time_periods: Optional[int] = None,
                       figsize: tuple = (15, 5)) -> None:
        """
        Plot multiple variables side by side for comparison.
        
        Args:
            variables: List of variables to compare
            selected_paths: List of path indices to plot. If None, plots all paths
            time_periods: Number of time periods to plot. If None, plots all periods
            figsize: Figure size as (width, height)
        """
        n_vars = len(variables)
        plt.figure(figsize=figsize)
        
        for idx, var in enumerate(variables, 1):
            plt.subplot(1, n_vars, idx)
            data = getattr(self.paths, var)
            
            paths_to_plot = selected_paths or range(data.shape[0])
            t_max = time_periods or data.shape[1]
            
            for i in paths_to_plot:
                label = 'No TAI' if i == 0 else f'TAI in year {i}'
                plt.plot(data[i, :t_max], label=label)
            
            plt.title(self.variable_properties[var]['title'])
            plt.xlabel('Time Period')
            plt.ylabel(self.variable_properties[var]['ylabel'])
            plt.grid(True)
            if idx == 1:  # Only show legend for first subplot
                plt.legend()
                
        plt.tight_layout()