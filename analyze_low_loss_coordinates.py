import numpy as np
import matplotlib.pyplot as plt
from read_tmi_compare_results import TMIAnalyzer
import os
import glob
import re

def get_available_thresholds(p_fixed_name, p_fixed):
    """Get list of thresholds that have data available."""
    # Find all CSV files with threshold values
    pattern = f'tmi_compare_results/data_collapse_results_{p_fixed_name}{p_fixed:.3f}_threshold*.csv'
    files = glob.glob(pattern)
    
    # Extract threshold values from filenames
    thresholds = []
    for f in files:
        match = re.search(r'threshold(\d+\.\d+e-\d+)\.csv', f)
        if match:
            thresholds.append(float(match.group(1)))
    
    return sorted(thresholds)

def analyze_threshold_dependence(
    pc_guess,
    nu_guess,
    p_fixed,
    p_fixed_name,
    p_range,
    nu_range,
    L_min,
    loss_threshold_delta,  # Amount above minimum loss to consider as "low loss"
    n_points,
    thresholds=None
):
    """
    Analyze how the low-loss (pc, nu) coordinates depend on threshold values.
    
    Parameters:
    -----------
    thresholds : list, optional
        List of threshold values to analyze. If None, will use all available thresholds.
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    p_range : tuple
        (min_p, max_p) range to explore
    nu_range : tuple
        (min_nu, max_nu) range to explore
    L_min : int
        Minimum system size to include
    loss_threshold_delta : float
        Amount above minimum loss to consider as "low loss"
    n_points : int
        Number of points to sample in each dimension
    """
    # If no thresholds provided, get all available ones
    if thresholds is None:
        thresholds = get_available_thresholds(p_fixed_name, p_fixed)
        print(f"Found {len(thresholds)} threshold values with data")
    
    # Store results for each implementation
    results = {}
    
    # Create meshgrid of p and nu values
    p_vals = np.linspace(p_range[0], p_range[1], n_points)
    nu_vals = np.linspace(nu_range[0], nu_range[1], n_points)
    PC, NU = np.meshgrid(p_vals, nu_vals)
    
    # Process each threshold
    for threshold in thresholds:
        print(f"\nProcessing threshold: {threshold:.1e}")
        
        # Initialize analyzer for this threshold
        analyzer = TMIAnalyzer(
            pc_guess=pc_guess,  # Initial guess, will be refined
            nu_guess=nu_guess,  # Initial guess, will be refined
            p_fixed=p_fixed,
            p_fixed_name=p_fixed_name,
            threshold=threshold
        )
        
        # Read data
        df = analyzer.read_from_csv()
        if df is None:
            print(f"No data found for threshold {threshold:.1e}, skipping...")
            continue
            
        # Get unique implementations
        implementations = df.index.get_level_values('implementation').unique()
        
        # Process each implementation
        for impl in implementations:
            # Initialize results dictionary for this implementation if not exists
            if impl not in results:
                results[impl] = {
                    'threshold': [],
                    'pc': [],
                    'nu': [],
                    'loss': []
                }
            
            # Filter data for this implementation
            impl_df = analyzer.unscaled_df.xs(impl, level='implementation')
            
            # Create a DataCollapse object
            from FSS.DataCollapse import DataCollapse
            dc = DataCollapse(
                impl_df,
                p_='p',
                L_='L',
                params={},
                p_range=p_range,
                Lmin=L_min
            )
            
            # Calculate loss for each point
            Z = np.zeros_like(PC)
            for j in range(n_points):
                for k in range(n_points):
                    loss_vals = dc.loss(PC[j,k], NU[j,k], beta=0)
                    Z[j,k] = np.sum(loss_vals**2) / (len(loss_vals) - 2)
            
            # Calculate minimum loss and set threshold
            min_loss = np.nanmin(Z)
            max_loss = np.nanmax(Z)
            loss_threshold = min_loss + loss_threshold_delta
            
            # Print min/max loss values for diagnostics
            print(f"Implementation {impl} at threshold {threshold:.1e}:")
            print(f"  Min loss: {min_loss:.3f}")
            print(f"  Max loss: {max_loss:.3f}")
            print(f"  Mean loss: {np.mean(Z):.3f}")
            print(f"  Loss threshold: {loss_threshold:.3f}")
            print(f"  Points with loss < {loss_threshold}: {np.sum(Z < loss_threshold)}")
            
            # Find low loss points
            low_loss_mask = Z < loss_threshold
            low_loss_points = np.where(low_loss_mask)
            
            if len(low_loss_points[0]) > 0:
                # Get the lowest loss point
                min_loss_idx = np.argmin(Z)
                min_loss_j, min_loss_k = np.unravel_index(min_loss_idx, Z.shape)
                print(f"  Best fit: pc={PC[min_loss_j,min_loss_k]:.3f}, nu={NU[min_loss_j,min_loss_k]:.3f}, loss={Z[min_loss_j,min_loss_k]:.3f}")
            
            # Store results
            for point_idx in range(len(low_loss_points[0])):
                j, k = low_loss_points[0][point_idx], low_loss_points[1][point_idx]
                results[impl]['threshold'].append(threshold)
                results[impl]['pc'].append(PC[j,k])
                results[impl]['nu'].append(NU[j,k])
                results[impl]['loss'].append(Z[j,k])
    
    # Create separate plots for each implementation
    for impl, impl_results in results.items():
        if not impl_results['threshold']:
            print(f"\nNo low-loss points found for implementation {impl}")
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot pc vs threshold
        scatter1 = ax1.scatter(impl_results['threshold'], impl_results['pc'], 
                             alpha=0.6, c=impl_results['loss'], 
                             cmap='viridis', s=50)
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('p_c')
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add horizontal line at pc_guess
        ax1.axhline(y=pc_guess, color='r', linestyle='--', alpha=0.5, label='Initial guess')
        ax1.legend(loc='upper right')
        ax1.set_ylim(p_range)
        
        # Customize colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Loss', rotation=270, labelpad=15)
        
        # Plot nu vs threshold
        scatter2 = ax2.scatter(impl_results['threshold'], impl_results['nu'],
                             alpha=0.6, c=impl_results['loss'],
                             cmap='viridis', s=50)
        
        ax2.set_xscale('log')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('nu')
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add horizontal lines for nu
        ax2.axhline(y=nu_guess, color='r', linestyle='--', alpha=0.5, label='Initial guess')
        ax2.legend(loc='upper right')
        ax2.set_ylim(nu_range)
        
        # Customize colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Loss', rotation=270, labelpad=15)
        
        # Add title
        plt.suptitle(f'Critical Parameters vs Threshold - {impl.capitalize()} Implementation', y=1.02, fontsize=14)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('tmi_compare_results', exist_ok=True)
        fig_path = f'tmi_compare_results/threshold_dependence_{p_fixed_name}{p_fixed:.3f}_{impl}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved threshold dependence plot for {impl} to {fig_path}")
        
        # Close the figure to free memory
        plt.close(fig)
    
    return results

if __name__ == '__main__':
    # Run analysis with all available thresholds
    p_fixed = 0.0
    p_fixed_name = 'pctrl'
    pc_guess = 0.5
    nu_guess = 1.33
    delta_pc = 0.2  # Increased range
    delta_nu = 0.5   # Increased range
    p_range = (pc_guess - delta_pc, pc_guess + delta_pc)
    nu_range = (nu_guess - delta_nu, nu_guess + delta_nu)
    results = analyze_threshold_dependence(
        p_fixed=p_fixed,
        p_fixed_name=p_fixed_name,
        pc_guess=pc_guess,
        nu_guess=nu_guess,
        p_range=p_range,
        nu_range=nu_range,
        L_min=12,
        loss_threshold_delta=0.05,  # Small increment above minimum loss
        n_points=150  # Increased resolution
    ) 

