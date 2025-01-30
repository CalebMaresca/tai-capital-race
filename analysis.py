# analysis.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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
        'paths': output_dir / 'paths',  # New directory for saved paths
    }
    
    for d in dirs.values():
        d.mkdir(exist_ok=True)
        
    return dirs

def save_paths(paths_dict: Dict, output_path: Path, group_name: str):
    """Save paths dictionary to a pickle file"""
    file_path = output_path / f"{group_name}_paths.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(paths_dict, f)
    print(f"Saved paths to {file_path}")

def load_paths(output_path: Path, group_name: str) -> Optional[Dict]:
    """Load paths dictionary from a pickle file if it exists"""
    file_path = output_path / f"{group_name}_paths.pkl"
    if file_path.exists():
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def compute_paths(param_group: ParameterGroup, base_output_dir: str, force_recompute: bool = False) -> Dict:
    """Compute or load transition paths for a parameter group"""
    # Create output directories
    output_dirs = create_output_dirs(base_output_dir, param_group.name)
    
    # Try to load existing paths if not forcing recomputation
    if not force_recompute:
        paths_dict = load_paths(output_dirs['paths'], param_group.name)
        if paths_dict is not None:
            print(f"Loaded existing paths for {param_group.name}")
            return paths_dict
    
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
        paths = calculate_interest_rate_paths(paths, economy, n_years=26)
        paths_dict[param_set.name] = paths
    
    # Save the computed paths
    save_paths(paths_dict, output_dirs['paths'], param_group.name)
    return paths_dict

def generate_plots(param_group: ParameterGroup, paths_dict: Dict, base_output_dir: str):
    """Generate plots for a parameter group using the provided paths"""
    output_dirs = create_output_dirs(base_output_dir, param_group.name)
    
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
            time_periods=26,
            title=f'{var_name.replace("_", " ").title()} - {param_group.name}',
            line_styles=line_styles
        )
        fig.savefig(output_dirs['figures'] / f"{var_name}_comparison.png")
        plt.close(fig)
    
    # Multi-variable comparison plot
    fig = plotter.plot_comparison(
        variables=variables,
        selected_paths=[0],  # Only show No TAI path for clarity
        time_periods=26,
        titles=[f'{var.replace("_", " ").title()} - {param_group.name}' for var in variables],
        shared_y_range=True,
        line_styles=line_styles
    )
    fig.savefig(output_dirs['figures'] / "rates_comparison.png")
    plt.close(fig)

def main():
    """Main analysis script"""
    # === USER PARAMETERS ===
    BASE_OUTPUT_DIR = "output"
    FORCE_RECOMPUTE = False  # Set to True to force recomputation of paths
    
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
    
    # Process each parameter group
    for group in parameter_groups:
        print(f"\nProcessing group: {group.name}")
        print(f"Description: {group.description}")
        try:
            # First compute or load the paths
            paths_dict = compute_paths(group, BASE_OUTPUT_DIR, force_recompute=FORCE_RECOMPUTE)
            print(f"Paths obtained successfully for group {group.name}")
            
            # Then generate the plots
            generate_plots(group, paths_dict, BASE_OUTPUT_DIR)
            print(f"Plots generated successfully for group {group.name}")
        except Exception as e:
            print(f"Error processing group {group.name}: {str(e)}")
            continue

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()