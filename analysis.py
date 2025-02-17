# analysis.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import OrderedDict
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
    warmup_iters: int = 10000
    decay_iters: int = 10000
    decay_rate: float = 0.9
    max_iterations: int = 1000000
    max_grad: float = 1.0
    description: str = ""  # Optional description of this parameter set

@dataclass
class ParameterGroup:
    """A group of related parameter sets to be analyzed together"""
    name: str
    description: str
    parameter_sets: List[ParameterSet]
    line_styles: Optional[Dict[str, Dict[str, float]]] = None

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
        paths = compute_transition_path(economy, T=300, max_iterations=param_set.max_iterations, lr=param_set.lr, warmup_iters=param_set.warmup_iters, decay_iters=param_set.decay_iters, decay_rate=param_set.decay_rate, tolerance=1e-5, max_grad=param_set.max_grad)
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
    
    # Calculate stationary equilibrium values using parameters from the first parameter set
    # (they should all have the same g_SQ, eta, beta, alpha, delta)
    first_params = param_group.parameter_sets[0]
    r_ss = ((1 + first_params.g_SQ) ** first_params.eta) / first_params.beta - 1
    
    # Calculate equilibrium capital (which is the initial capital)
    k_ss = (first_params.alpha / (r_ss + first_params.delta)) ** (1 / (1 - first_params.alpha))
    
    # Calculate equilibrium savings rate
    # f(k)/k = (r + delta)/alpha in equilibrium
    fk_over_k_ss = (r_ss + first_params.delta) / first_params.alpha
    s_ss = (first_params.g_SQ + first_params.delta) / fk_over_k_ss
    
    # Generate plots comparing all parameter sets in the group
    variables = OrderedDict([
        ('capital_rental_rates', 'Capital Rental Rates'),
        ('interest_rates_1y', '1-Year Interest Rate'),
        ('interest_rates_30y', '30-Year Interest Rate'),
        ('savings_rates', 'Savings Rate')
    ])
    selected_paths = [0, 1, 9, 18]
    line_styles = param_group.line_styles or {name: {'style': '-', 'alpha': 1.0} for name in paths_dict.keys()}
    
    for var_name, var_title in variables.items():
        # Single variable plot
        fig = plotter.plot_variable(
            var_name,
            selected_paths=selected_paths,
            time_periods=26,
            title=var_title,
            line_styles=line_styles
        )
        
        # Add horizontal line for stationary equilibrium values
        if var_name == 'savings_rates':
            plt.plot([0, 25], [s_ss, s_ss], color='black', linestyle=':', alpha=1.0,
                    label='Stationary Equilibrium Rate', zorder=-1)
        else:
            plt.plot([0, 25], [r_ss, r_ss], color='black', linestyle=':', alpha=1.0,
                    label='Stationary Equilibrium Rate', zorder=-1)
        # Refresh legend to include the new line
        plt.legend(fontsize=12)
        
        fig.savefig(output_dirs['figures'] / f"{var_name}_comparison.png")
        plt.close(fig)
    
    # Multi-variable comparison plot with 2x2 layout
    fig = plt.figure(figsize=(15, 12))  # Adjusted figure size for 2x2 layout
    
    # First find global min/max for interest rate plots
    interest_vars = ['capital_rental_rates', 'interest_rates_1y', 'interest_rates_30y']
    global_min = float('inf')
    global_max = float('-inf')
    
    for var_name in interest_vars:
        for name, paths in paths_dict.items():
            data = getattr(paths, var_name)
            values = data[0, :26]  # Only looking at No TAI path for 26 periods
            global_min = min(global_min, values.min())
            global_max = max(global_max, values.max())
    
    # Add 5% padding on each end
    y_range = global_max - global_min
    global_min -= 0.05 * y_range
    global_max += 0.05 * y_range
    
    for idx, (var_name, var_title) in enumerate(variables.items(), 1):
        plt.subplot(2, 2, idx)
        
        for name, paths in paths_dict.items():
            if var_name == 'savings_rates':
                data = plotter.calculate_savings_rates(paths)
            else:
                data = getattr(paths, var_name)
            
            style = line_styles[name]
            # Only show No TAI path for clarity
            label = f'{name} - No TAI'
            plt.plot(data[0, :26], style['style'], label=label, linewidth=2, alpha=style['alpha'], color=style['color'])
        
        # Add horizontal line for stationary equilibrium values
        if var_name == 'savings_rates':
            plt.plot([0, 25], [s_ss, s_ss], color='black', linestyle=':', alpha=1.0,
                    label='Stationary Equilibrium Rate', zorder=-1)
        else:
            plt.plot([0, 25], [r_ss, r_ss], color='black', linestyle=':', alpha=1.0,
                    label='Stationary Equilibrium Rate', zorder=-1)
            # Set shared y-axis limits for interest rate plots
            plt.ylim(global_min, global_max)
            
        plt.title(var_title, fontsize=14, pad=20)
        plt.xlabel('Years from Present', fontsize=12)
        plt.ylabel(plotter.variable_properties[var_name]['ylabel'], fontsize=12)
        plt.grid(True)
        if idx == 1:  # Only show legend for first subplot
            plt.legend(fontsize=10)
        
        # Increase tick label sizes
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    
    plt.tight_layout()
    fig.savefig(output_dirs['figures'] / "rates_comparison.png")
    plt.close(fig)

def generate_latex_table(param_groups: List[ParameterGroup], group_dict: Dict) -> str:
    """Generate a LaTeX table comparing initial values across parameter sets"""
    latex_table = [
        "\\begin{table}[H]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Parameter Set & 1y Interest Rate & 30y Interest Rate \\\\",
        "\\hline"
    ]
    
    # Collect all rates by source and lambda
    rates = {
        'Cotra': {},
        'Metaculus': {}
    }
    
    # First collect all the rates
    for group in param_groups:
        paths_dict = group_dict[group.name]
        for param_set in group.parameter_sets:
            if param_set.name in paths_dict:
                paths = paths_dict[param_set.name]
                # Get year 1 values (index 1)
                rate_1y = paths.interest_rates_1y[0,1]
                rate_30y = paths.interest_rates_30y[0,1]
                
                # Extract lambda value and source from name
                if 'Cotra' in param_set.name:
                    source = 'Cotra'
                else:
                    source = 'Metaculus'
                
                lambda_val = param_set.lambda_param
                rates[source][lambda_val] = (rate_1y, rate_30y)
    
    # Now generate the table in the desired order
    for source in ['Cotra', 'Metaculus']:
        latex_table.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{source}}}}} \\\\")
        
        # Sort by lambda value
        for lambda_val in sorted(rates[source].keys()):
            rate_1y, rate_30y = rates[source][lambda_val]
            row = (
                f"\\quad $\\lambda={lambda_val}$ & "
                f"{rate_1y*100:.2f}\\% & "
                f"{rate_30y*100:.2f}\\% \\\\"
            )
            latex_table.append(row)
        latex_table.append("\\hline")
    
    latex_table.extend([
        "\\end{tabular}",
        "\\caption{Interest Rates in Year 1}",
        "\\label{tab:initial_values}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_table)

def generate_savings_rate_table(param_groups: List[ParameterGroup], group_dict: Dict, plotter: TransitionPathVisualizer) -> str:
    """Generate a LaTeX table comparing savings rates across parameter sets"""
    # Get all unique lambda values
    lambda_values = sorted({param_set.lambda_param 
                          for group in param_groups 
                          for param_set in group.parameter_sets})
    
    # Create column specification for all lambdas
    column_spec = "l" + "c" * len(lambda_values)
    
    latex_table = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\hline",
        "Source & " + " & ".join([f"$\\lambda={l}$" for l in lambda_values]) + " \\\\",
        "\\hline"
    ]
    
    # Collect all rates by source and lambda
    rates = {
        'Cotra': {},
        'Metaculus': {}
    }
    
    # First collect all the rates
    for group in param_groups:
        paths_dict = group_dict[group.name]
        for param_set in group.parameter_sets:
            if param_set.name in paths_dict:
                paths = paths_dict[param_set.name]
                # Calculate savings rate for year 0
                savings_rates = plotter.calculate_savings_rates(paths)
                rate = savings_rates[0,0]  # Get year 0 rate from No TAI path
                
                # Extract lambda value and source from name
                if 'Cotra' in param_set.name:
                    source = 'Cotra'
                else:
                    source = 'Metaculus'
                
                lambda_val = param_set.lambda_param
                rates[source][lambda_val] = rate
    
    # Generate rows
    for source in ['Cotra', 'Metaculus']:
        row = [source]
        for lambda_val in lambda_values:
            rate = rates[source][lambda_val]
            row.append(f"{rate*100:.2f}\\%")
        latex_table.append(" & ".join(row) + " \\\\")
    
    latex_table.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Savings Rates in Year 0}",
        "\\label{tab:savings_rates}",
        "\\end{table}"
    ])
    
    return "\n".join(latex_table)

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
                    name="Cotra λ=1",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=1, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="Baseline model with Cotra probabilities"
                ),
                ParameterSet(
                    name="Metaculus λ=1",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=1, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="Baseline model with Metaculus probabilities"
                ),
                ParameterSet(
                    name="Cotra λ=0",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="No competition model with Cotra probabilities"
                ),
                ParameterSet(
                    name="Metaculus λ=0",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="No competition model with Metaculus probabilities"
                )
            ],
            line_styles={
                'Cotra λ=1': {'style': '-', 'alpha': 1.0, 'color': 'tab:blue'},
                'Metaculus λ=1': {'style': '-', 'alpha': 1.0, 'color': 'tab:orange'},
                'Cotra λ=0': {'style': '--', 'alpha': 1.0, 'color': 'tab:blue'},
                'Metaculus λ=0': {'style': '--', 'alpha': 1.0, 'color': 'tab:orange'}
            }
        ),
        ParameterGroup(
            name="lambda_2",
            description="Model with lambda=2",
            parameter_sets=[
                ParameterSet(
                    name="Cotra λ=2",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=2, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    lr=0.03, decay_iters=50000, decay_rate=0.95,
                    description="Lambda=2 model with Cotra probabilities"
                ),
                ParameterSet(
                    name="Metaculus λ=2",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=2, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    lr=0.03, decay_iters=50000, decay_rate=0.95,
                    description="Lambda=2 model with Metaculus probabilities"
                ),
                ParameterSet(
                    name="Cotra λ=0",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="No competition model with Cotra probabilities"
                ),
                ParameterSet(
                    name="Metaculus λ=0",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="No competition model with Metaculus probabilities"
                )
            ],
            line_styles={
                'Cotra λ=2': {'style': '-', 'alpha': 1.0, 'color': 'tab:blue'},
                'Metaculus λ=2': {'style': '-', 'alpha': 1.0, 'color': 'tab:orange'},
                'Cotra λ=0': {'style': '--', 'alpha': 1.0, 'color': 'tab:blue'},
                'Metaculus λ=0': {'style': '--', 'alpha': 1.0, 'color': 'tab:orange'}
            }
        ),
        ParameterGroup(
            name="lambda_4",
            description="Model with lambda=4",
            parameter_sets=[
                ParameterSet(
                    name="Cotra λ=4",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=4, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    lr=0.01, warmup_iters=50000, decay_iters=50000, decay_rate=0.95, max_grad=1,
                    description="Lambda=4 model with Cotra probabilities"
                ),
                ParameterSet(
                    name="Metaculus λ=4",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=4, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    lr=0.01, warmup_iters=50000, decay_iters=50000, decay_rate=0.95, max_grad=1,
                    description="Lambda=4 model with Metaculus probabilities"
                ),
                ParameterSet(
                    name="Cotra λ=0",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=cotra_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="No competition model with Cotra probabilities"
                ),
                ParameterSet(
                    name="Metaculus λ=0",
                    eta=1, beta=0.99, alpha=0.36, delta=0.025,
                    lambda_param=0, g_SQ=0.018, g_TAI=0.3,
                    unconditional_TAI_probs=metaculus_probs,
                    warmup_iters=0, decay_rate=0.95,
                    description="No competition model with Metaculus probabilities"
                )
            ],
            line_styles={
                'Cotra λ=4': {'style': '-', 'alpha': 1.0, 'color': 'tab:blue'},
                'Metaculus λ=4': {'style': '-', 'alpha': 1.0, 'color': 'tab:orange'},
                'Cotra λ=0': {'style': '--', 'alpha': 1.0, 'color': 'tab:blue'},
                'Metaculus λ=0': {'style': '--', 'alpha': 1.0, 'color': 'tab:orange'}
            }
        )
    ]
    
    # === ANALYSIS EXECUTION ===
    print(f"Starting analysis for {len(parameter_groups)} parameter groups...")
    
    # Dictionary to store all paths
    all_paths_dict = {}
    
    # Process each parameter group
    for group in parameter_groups:
        print(f"\nProcessing group: {group.name}")
        print(f"Description: {group.description}")
        try:
            # First compute or load the paths
            paths_dict = compute_paths(group, BASE_OUTPUT_DIR, force_recompute=FORCE_RECOMPUTE)
            all_paths_dict[group.name] = paths_dict
            print(f"Paths obtained successfully for group {group.name}")
            
            # Then generate the plots
            generate_plots(group, paths_dict, BASE_OUTPUT_DIR)
            print(f"Plots generated successfully for group {group.name}")
        except Exception as e:
            print(f"Error processing group {group.name}: {str(e)}")
            continue
    
    # Create a plotter instance for savings rate calculations
    plotter = TransitionPathVisualizer(all_paths_dict['baseline'])
    
    # Generate and save the LaTeX tables
    latex_table = generate_latex_table(parameter_groups, all_paths_dict)
    output_path = Path(BASE_OUTPUT_DIR) / "initial_values_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {output_path}")
    
    savings_table = generate_savings_rate_table(parameter_groups, all_paths_dict, plotter)
    output_path = Path(BASE_OUTPUT_DIR) / "savings_rates_table.tex"
    with open(output_path, 'w') as f:
        f.write(savings_table)
    print(f"Savings rates table saved to {output_path}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()