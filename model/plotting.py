import matplotlib.pyplot as plt
from typing import List, Optional, Literal, Dict
from model.core import TransitionPaths

class TransitionPathVisualizer:
    """
    Visualizer for transition paths data.
    
    Attributes:
        paths_dict (Dict[str, TransitionPaths]): Dictionary mapping path names to their TransitionPaths objects
    """
    
    # Define valid plot types
    VALID_VARIABLES = Literal[
        'capital_rental_rates', 'capital', 'wages', 
        'consumption', 'TFP', 'dv_da_next', 
        'dv_da', 'dv_da_TAI', 'interest_rates_1y', 'interest_rates_30y'
    ]
    
    def __init__(self, paths_dict: Dict[str, TransitionPaths]):
        """
        Initialize with either a single TransitionPaths object or a dictionary of them.
        
        Args:
            paths_dict: Dictionary mapping path names to their TransitionPaths objects,
                       or a single TransitionPaths object which will be stored with name 'default'
        """
        if isinstance(paths_dict, dict):
            self.paths_dict = paths_dict
        else:
            self.paths_dict = {'default': paths_dict}
        
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
                     figsize: tuple = (12, 6),
                     path_names: Optional[List[str]] = None,
                     line_styles: Optional[Dict[str, str]] = None) -> plt.Figure:
        """
        Plot specified variable for selected paths.
        
        Args:
            variable: Name of the variable to plot
            selected_paths: List of path indices to plot. If None, plots all paths
            time_periods: Number of time periods to plot. If None, plots all periods
            title: Custom title for the plot. If None, uses default
            ylabel: Custom y-axis label. If None, uses default
            figsize: Figure size as (width, height)
            path_names: List of path names to plot. If None, plots all paths
            line_styles: Dictionary mapping path names to their line styles
        """
        fig = plt.figure(figsize=figsize)
        
        # Default line styles if none provided
        if line_styles is None:
            line_styles = {name: '-' for name in self.paths_dict.keys()}
            
        # Determine which path sets to plot
        paths_to_use = {name: paths for name, paths in self.paths_dict.items() 
                       if path_names is None or name in path_names}
        
        # Plot each path set
        for name, paths in paths_to_use.items():
            if not hasattr(paths, variable):
                continue
                
            data = getattr(paths, variable)
            indices = selected_paths or range(data.shape[0])
            t_max = time_periods or data.shape[1]
            
            for i in indices:
                label = f'{name} - {"No TAI" if i == 0 else f"TAI in year {i}"}'
                plt.plot(data[i, :t_max], line_styles[name], label=label)
            
        # Set labels and title
        plt.xlabel('Time Period')
        plt.ylabel(ylabel or self.variable_properties[variable]['ylabel'])
        plt.title(title or self.variable_properties[variable]['title'])
        plt.legend()
        plt.grid(True)

        return fig
    
    def plot_comparison(self,
                       variables: List[VALID_VARIABLES],
                       selected_paths: Optional[List[int]] = None,
                       time_periods: Optional[int] = None,
                       titles: Optional[List[str]] = None,
                       figsize: Optional[tuple] = None,
                       shared_y_range: bool = False,
                       path_names: Optional[List[str]] = None,
                       line_styles: Optional[Dict[str, str]] = None) -> plt.Figure:
        """
        Plot multiple variables side by side for comparison.
        
        Args:
            variables: List of variables to compare
            selected_paths: List of path indices to plot. If None, plots all paths
            time_periods: Number of time periods to plot. If None, plots all periods
            titles: Optional list of custom titles for each subplot
            figsize: Figure size as (width, height)
            shared_y_range: If True, all subplots will share the same y-axis range
            path_names: List of path names to plot. If None, plots all paths
            line_styles: Dictionary mapping path names to their line styles
        """
        n_vars = len(variables)
        if figsize is None:
            figsize = (5 * n_vars, 5)
        fig = plt.figure(figsize=figsize)

        # Default line styles if none provided
        if line_styles is None:
            line_styles = {name: '-' for name in self.paths_dict.keys()}

        # Determine which path sets to plot
        paths_to_use = {name: paths for name, paths in self.paths_dict.items() 
                       if path_names is None or name in path_names}

        if shared_y_range:
            global_min = float('inf')
            global_max = float('-inf')
            for var in variables:
                for paths in paths_to_use.values():
                    data = getattr(paths, var)
                    paths_to_plot = selected_paths or range(data.shape[0])
                    t_max = time_periods or data.shape[1]
                    for i in paths_to_plot:
                        values = data[i, :t_max]
                        global_min = min(global_min, values.min())
                        global_max = max(global_max, values.max())
            
            # Add 5% padding on each end
            y_range = global_max - global_min
            global_min -= 0.05 * y_range
            global_max += 0.05 * y_range

        for idx, var in enumerate(variables, 1):
            plt.subplot(1, n_vars, idx)
            
            for name, paths in paths_to_use.items():
                data = getattr(paths, var)
                indices = selected_paths or range(data.shape[0])
                t_max = time_periods or data.shape[1]
                
                for i in indices:
                    label = f'{name} - {"No TAI" if i == 0 else f"TAI in year {i}"}'
                    plt.plot(data[i, :t_max], line_styles[name], label=label)
            
            title = titles[idx-1] if titles and idx <= len(titles) else self.variable_properties[var]['title']
            plt.title(title)
            plt.xlabel('Time Period')
            plt.ylabel(self.variable_properties[var]['ylabel'])
            plt.grid(True)
            if idx == 1:  # Only show legend for first subplot
                plt.legend()

            if shared_y_range:
                plt.ylim(global_min, global_max)
                
        plt.tight_layout()
        
        return fig

    @staticmethod
    def compare_paths(paths_dict: Dict[str, TransitionPaths],
                     variable: VALID_VARIABLES,
                     selected_paths: Optional[List[int]] = None,
                     time_periods: Optional[int] = None,
                     title: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     figsize: tuple = (12, 6),
                     line_styles: Optional[Dict[str, str]] = None) -> plt.Figure:
        """
        Compare the same variable across different path sets.
        
        Args:
            paths_dict: Dictionary mapping path set names to their TransitionPaths objects
            variable: Name of the variable to plot
            selected_paths: List of path indices to plot. If None, plots all paths
            time_periods: Number of time periods to plot. If None, plots all periods
            title: Custom title for the plot. If None, uses default
            ylabel: Custom y-axis label. If None, uses default
            figsize: Figure size as (width, height)
            line_styles: Dictionary mapping path set names to their line styles
        """
        fig = plt.figure(figsize=figsize)
        
        # Default line styles if none provided
        if line_styles is None:
            line_styles = {name: '-' for name in paths_dict.keys()}
        
        # Plot each path set with different line styles
        for name, paths in paths_dict.items():
            data = getattr(paths, variable)
            paths_to_plot = selected_paths or range(data.shape[0])
            t_max = time_periods or data.shape[1]
            
            for i in paths_to_plot:
                label = f'{name} - {"No TAI" if i == 0 else f"TAI in year {i}"}'
                plt.plot(data[i, :t_max], line_styles[name], label=label)
        
        # Set labels and title
        plt.xlabel('Time Period')
        plt.ylabel(ylabel or TransitionPathVisualizer(next(iter(paths_dict.values()))).variable_properties[variable]['ylabel'])
        plt.title(title or f'Comparison of {variable}')
        plt.legend()
        plt.grid(True)
        
        return fig