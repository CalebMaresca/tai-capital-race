# analysis.py

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

from model.core import Economy, compute_transition_path, calculate_interest_rate_paths
from model.plotting import TransitionPathVisualizer

baseline_TAI_probs = [
        0.025, 0.03, 0.035, 0.045, 0.05, 0.055, 0.06, 0.06, 0.055, 0.051,
        0.048, 0.045, 0.04, 0.035, 0.03, 0.027, 0.022, 0.02, 0.018, 0.016,
        0.014, 0.012, 0.01, 0.008, 0.006, 0.004, 0.002
    ]

def main():
    """
    Main analysis script. Edit parameter sets and output directory here.
    """
    # === USER PARAMETERS ===
    
    # Set base output directory
    BASE_OUTPUT_DIR = "output"
    
    # Define parameter sets to analyze
    parameter_sets = [
        ParameterSet(
            name="baseline",
            eta=1,
            beta=0.96,
            alpha=0.36,
            delta=0.025,
            lambda_param=1,
            g_SQ=0.018,
            g_TAI=0.3,
            unconditional_TAI_probs=baseline_TAI_probs,
            description="Baseline model specification"
        ),
        ParameterSet(
            name="no_competition",
            eta=1,
            beta=0.96,
            alpha=0.36,
            delta=0.025,
            lambda_param=0,
            g_SQ=0.018,
            g_TAI=0.3,
            unconditional_TAI_probs=baseline_TAI_probs,
            description="Model without competition over AI labor"
        ),
        ParameterSet(
            name="lambda_2",
            eta=1,
            beta=0.96,
            alpha=0.36,
            delta=0.025,
            lambda_param=2,
            g_SQ=0.018,
            g_TAI=0.3,
            unconditional_TAI_probs=baseline_TAI_probs,
            description="Model with lambda=2"
        ),
        ParameterSet(
            name="lambda_4",
            eta=1,
            beta=0.96,
            alpha=0.36,
            delta=0.025,
            lambda_param=4,
            g_SQ=0.018,
            g_TAI=0.3,
            unconditional_TAI_probs=baseline_TAI_probs,
            lr=0.075,
            description="Model with lambda=4"
        ),
        # Add more parameter sets as needed
    ]
    
    # === ANALYSIS EXECUTION ===
    print(f"Starting analysis for {len(parameter_sets)} parameter sets...")
    
    # Run analysis for each parameter set
    for param_set in parameter_sets:
        print(f"\nRunning analysis for {param_set.name}")
        print(f"Description: {param_set.description}")
        
        # Create output directories
        output_dirs = create_output_dirs(BASE_OUTPUT_DIR, param_set)
        
        # Run analysis
        try:
            paths = run_analysis(param_set, output_dirs)
            print(f"Analysis completed successfully for {param_set.name}")
        except Exception as e:
            print(f"Error in analysis for {param_set.name}: {str(e)}")
            continue

    print("\nAnalysis complete!")


# === SUPPORTING CLASSES AND FUNCTIONS ===

@dataclass
class ParameterSet:
    """Configuration for a single model run"""
    name: str
    eta: float
    beta: float
    alpha: float
    delta: float
    lambda_param: float
    g_SQ: float
    g_TAI: float
    unconditional_TAI_probs: List[float]
    lr: float = 0.1
    description: str = ""  # Optional description of this parameter set

def create_output_dirs(base_dir: str, param_set: ParameterSet) -> Dict[str, str]:
    """Create output directory structure for a parameter set"""
    # Create main output dir if it doesn't exist
    output_dir = Path(base_dir) / param_set.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of output
    dirs = {
        'figures': output_dir / 'figures',
        'data': output_dir / 'data',
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True)
        
    return dirs

def run_analysis(param_set: ParameterSet, output_dirs: Dict[str, str]):
    """Run full analysis for a parameter set and save results"""
    # Initialize model
    unconditional_TAI_probs = param_set.unconditional_TAI_probs
    
    economy = Economy(
        eta=param_set.eta,
        beta=param_set.beta,
        alpha=param_set.alpha,
        delta=param_set.delta,
        lambda_param=param_set.lambda_param,
        TFP_0=1,
        g_SQ=param_set.g_SQ,
        g_TAI=param_set.g_TAI,
        unconditional_TAI_probs=unconditional_TAI_probs
    )
    
    # Compute paths
    paths = compute_transition_path(economy, T=200, lr=param_set.lr, tolerance=1e-5)
    paths = calculate_interest_rate_paths(paths, economy, n_years=31)
    
    # Create visualizations
    plotter = TransitionPathVisualizer(paths)
    
    # Generate and save all plots
    plot_specs = [
        ('capital_rental_rates', {'title': 'Selected Capital Rental Rate Paths', 'selected_paths': [0, 1, 9, 18, economy.max_TAI_year], 'time_periods': 31}),
        ('interest_rates_1y', {'title': 'Selected 1-Year Interest Rate Paths', 'selected_paths': [0, 1, 9, 18, economy.max_TAI_year], 'time_periods': 31}),
        ('interest_rates_30y', {'title': 'Selected 30-Year Interest Rate Paths', 'selected_paths': [0, 1, 9, 18, economy.max_TAI_year], 'time_periods': 31}),
        # Add more plot specifications as needed
    ]
    
    for var_name, plot_kwargs in plot_specs:
        fig = plotter.plot_variable(var_name, **plot_kwargs)
        fig.savefig(output_dirs['figures'] / f"{var_name}.png")

    # Generate comparison plots
    fig = plotter.plot_comparison(
        variables=['capital_rental_rates', 'interest_rates_1y', 'interest_rates_30y'],
        titles=['Capital Rental Rate - No TAI Path', '1-Year Interest Rate - No TAI Path', '30-Year Interest Rate - No TAI Path'],
        selected_paths=[0],
        time_periods=31,
        shared_y_range=True
    )
    fig.savefig(output_dirs['figures'] / "rates_comparison.png")
        
    # Save key data
    # (Add code here to save relevant numerical results if desired)
    
    return paths  # Return paths object in case needed for further analysis


if __name__ == "__main__":
    main()