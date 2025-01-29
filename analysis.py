# analysis.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from model.core import Economy, compute_transition_path, calculate_interest_rate_paths
from model.plotting import TransitionPathVisualizer

# Read TAI probabilities from CSV
probs_df = pd.read_csv('output/probabilities/tai_probabilities.csv')

# Get Cotra and Metaculus probabilities, removing any NaN values
cotra_probs = [p for p in probs_df['cotra-fitted'].dropna()]
metaculus_probs = [p for p in probs_df['metaculus-fitted'].dropna()]

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

@dataclass
class ParameterGroup:
    """A group of related parameter sets to be analyzed together"""
    name: str
    description: str
    parameter_sets: List[ParameterSet]
    line_styles: Optional[Dict[str, str]] = None

def create_output_dirs(base_dir: str, group_name: str) -> Dict[str, Path]:
    """Create output directory structure"""
    output_dir = Path(base_dir) / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dirs = {
        'figures': output_dir / 'figures',
        'data': output_dir / 'data',
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True)
        
    return dirs

def run_analysis(param_group: ParameterGroup, base_output_dir: str):
    """Run full analysis for a parameter group and save results"""
    # Create output directories
    output_dirs = create_output_dirs(base_output_dir, param_group.name)
    
    # Dictionary to store paths for all parameter sets in the group
    paths_dict = {}
    
    # Run analysis for each parameter set in the group
    for param_set in param_group.parameter_sets:
        print(f"\nAnalyzing {param_set.name}")
        print(f"Description: {param_set.description}")
        
        # Initialize model
        economy = Economy(
            eta=param_set.eta,
            beta=param_set.beta,
            alpha=param_set.alpha,
            delta=param_set.delta,
            lambda_param=param_set.lambda_param,
            TFP_0=1,
            g_SQ=param_set.g_SQ,
            g_TAI=param_set.g_TAI,
            unconditional_TAI_probs=param_set.unconditional_TAI_probs
        )
        
        # Compute paths
        paths = compute_transition_path(economy, T=200, lr=param_set.lr, tolerance=1e-5)
        paths = calculate_interest_rate_paths(paths, economy, n_years=31)
        paths_dict[param_set.name] = paths
    
    # Create visualizer with all paths
    plotter = TransitionPathVisualizer(paths_dict)
    
    # Generate plots comparing all parameter sets in the group
    variables = ['capital_rental_rates', 'interest_rates_1y', 'interest_rates_30y']
    selected_paths = [0, 1, 9, 18]
    line_styles = param_group.line_styles or {name: '-' for name in paths_dict.keys()}
    
    for var_name in variables:
        # Single variable plot
        fig = plotter.plot_variable(
            var_name,
            selected_paths=selected_paths,
            time_periods=31,
            title=f'{var_name.replace("_", " ").title()} - {param_group.name}',
            line_styles=line_styles
        )
        fig.savefig(output_dirs['figures'] / f"{var_name}_comparison.png")
        plt.close(fig)
    
    # Multi-variable comparison plot
    fig = plotter.plot_comparison(
        variables=variables,
        selected_paths=[0],  # Only show No TAI path for clarity
        time_periods=31,
        titles=[f'{var.replace("_", " ").title()} - {param_group.name}' for var in variables],
        shared_y_range=True,
        line_styles=line_styles
    )
    fig.savefig(output_dirs['figures'] / "rates_comparison.png")
    plt.close(fig)
    
    return paths_dict

def main():
    """Main analysis script"""
    # === USER PARAMETERS ===
    BASE_OUTPUT_DIR = "output"
    
    # Define parameter groups
    parameter_groups = [
        ParameterGroup(
            name="baseline",
            description="Baseline model comparison",
            parameter_sets=[
                ParameterSet(
                    name="cotra",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=1, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    description="Baseline model with Cotra probabilities"
                ),
                ParameterSet(
                    name="metaculus",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=1, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    description="Baseline model with Metaculus probabilities"
                )
            ],
            line_styles={'cotra': '-', 'metaculus': '--'}
        ),
        ParameterGroup(
            name="no_competition",
            description="Model without competition over AI labor",
            parameter_sets=[
                ParameterSet(
                    name="cotra",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    description="No competition model with Cotra probabilities"
                ),
                ParameterSet(
                    name="metaculus",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    description="No competition model with Metaculus probabilities"
                )
            ],
            line_styles={'cotra': '-', 'metaculus': '--'}
        ),
        ParameterGroup(
            name="lambda_2",
            description="Model with lambda=2",
            parameter_sets=[
                ParameterSet(
                    name="cotra",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=2, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    description="Lambda=2 model with Cotra probabilities"
                ),
                ParameterSet(
                    name="metaculus",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=2, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    description="Lambda=2 model with Metaculus probabilities"
                )
            ],
            line_styles={'cotra': '-', 'metaculus': '--'}
        ),
        ParameterGroup(
            name="lambda_4",
            description="Model with lambda=4",
            parameter_sets=[
                ParameterSet(
                    name="cotra",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=4, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    lr=0.075,
                    description="Lambda=4 model with Cotra probabilities"
                ),
                ParameterSet(
                    name="metaculus",
                    eta=1, beta=0.96, alpha=0.36, delta=0.025,
                    lambda_param=4, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    lr=0.075,
                    description="Lambda=4 model with Metaculus probabilities"
                )
            ],
            line_styles={'cotra': '-', 'metaculus': '--'}
        )
    ]
    
    # === ANALYSIS EXECUTION ===
    print(f"Starting analysis for {len(parameter_groups)} parameter groups...")
    
    # Run analysis for each parameter group
    for group in parameter_groups:
        print(f"\nAnalyzing group: {group.name}")
        print(f"Description: {group.description}")
        try:
            paths_dict = run_analysis(group, BASE_OUTPUT_DIR)
            print(f"Analysis completed successfully for group {group.name}")
        except Exception as e:
            print(f"Error in analysis for group {group.name}: {str(e)}")
            continue

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()