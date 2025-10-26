#!/usr/bin/env python3
"""
Perform data collapse analysis on TMI data from CSV file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fss import DataCollapse
import os

def create_observations_from_stats(mean, sem, n_samples=100):
    """
    Create synthetic observations from mean and standard error.
    This is needed because DataCollapse expects individual observations.
    
    Parameters:
    -----------
    mean : float
        Mean value
    sem : float
        Standard error of the mean
    n_samples : int
        Number of synthetic samples to generate
        
    Returns:
    --------
    numpy.ndarray
        Array of synthetic observations
    """
    # Standard deviation = SEM * sqrt(n_samples)
    std = sem * np.sqrt(n_samples)
    
    # Generate samples from normal distribution
    observations = np.random.normal(mean, std, n_samples)
    
    return observations

def load_csv_for_datacollapse(csv_file, n_samples=100):
    """
    Load CSV file and convert to format needed for DataCollapse.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file with columns: L, pctrl, pproj, tmi_mean, tmi_sem
    n_samples : int
        Number of synthetic samples to generate from mean/sem
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with multi-index (p, L) and 'observations' column
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"Loaded CSV with {len(df)} rows")
    print(f"L values: {sorted(df['L'].unique())}")
    print(f"pproj range: [{df['pproj'].min():.3f}, {df['pproj'].max():.3f}]")
    
    # Create data list for DataCollapse format
    data_list = []
    
    for _, row in df.iterrows():
        # Create synthetic observations from mean and SEM
        observations = create_observations_from_stats(
            row['tmi_mean'], 
            row['tmi_sem'], 
            n_samples
        )
        
        data_list.append({
            'p': row['pproj'],  # Using pproj as the varying parameter
            'L': row['L'],
            'observations': observations
        })
    
    # Convert to DataFrame with multi-index
    result_df = pd.DataFrame(data_list)
    result_df = result_df.set_index(['p', 'L'])
    
    return result_df

def perform_data_collapse_analysis(csv_file, 
                                   pc_guess=0.75, 
                                   nu_guess=1.0, 
                                   beta=0,
                                   L_min=None, 
                                   L_max=None, 
                                   p_range=None,
                                   nu_vary=True, 
                                   p_c_vary=True,
                                   n_samples=100,
                                   output_dir='/scratch/ty296/plots'):
    """
    Perform data collapse analysis on CSV data.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file
    pc_guess : float
        Initial guess for critical point
    nu_guess : float
        Initial guess for correlation length exponent
    beta : float
        Scaling dimension (usually 0 for TMI)
    L_min : int, optional
        Minimum system size to include
    L_max : int, optional
        Maximum system size to include
    p_range : tuple, optional
        (min, max) range of p values to include
    nu_vary : bool
        Whether to fit nu
    p_c_vary : bool
        Whether to fit p_c
    n_samples : int
        Number of synthetic samples to generate
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    tuple
        (fig, axes, result) containing plots and fit result
    """
    # Load and format data
    print("Loading and formatting data...")
    df = load_csv_for_datacollapse(csv_file, n_samples=n_samples)
    
    # Apply filters
    if L_min is not None or L_max is not None:
        L_values = df.index.get_level_values('L')
        mask = np.ones(len(df), dtype=bool)
        if L_min is not None:
            mask &= L_values >= L_min
        if L_max is not None:
            mask &= L_values <= L_max
        df = df.loc[mask]
        print(f"Filtered to L range: [{L_min}, {L_max}]")
    
    if p_range is None:
        p_min = df.index.get_level_values('p').min()
        p_max = df.index.get_level_values('p').max()
        p_range = [p_min, p_max]
    else:
        # Filter by p range
        p_values = df.index.get_level_values('p')
        mask = (p_values >= p_range[0]) & (p_values <= p_range[1])
        df = df.loc[mask]
        print(f"Filtered to p range: [{p_range[0]}, {p_range[1]}]")
    
    # Perform data collapse fit
    print("\nPerforming data collapse fit...")
    dc = DataCollapse(df=df, 
                     p_='p', 
                     L_='L',
                     params={},
                     p_range=p_range,
                     Lmin=L_min,
                     Lmax=L_max)
    
    try:
        result = dc.datacollapse(p_c=pc_guess, 
                               nu=nu_guess, 
                               beta=beta,
                               p_c_vary=p_c_vary,
                               nu_vary=nu_vary,
                               beta_vary=False)
        
        # Extract fitted parameters
        fitted_pc = result.params['p_c'].value
        fitted_pc_err = result.params['p_c'].stderr if result.params['p_c'].stderr is not None else 0
        fitted_nu = result.params['nu'].value
        fitted_nu_err = result.params['nu'].stderr if result.params['nu'].stderr is not None else 0
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Data Collapse Results:")
        print(f"{'='*60}")
        print(f"  p_c = {fitted_pc:.5f} ± {fitted_pc_err:.5f}")
        print(f"  nu  = {fitted_nu:.5f} ± {fitted_nu_err:.5f}")
        print(f"  beta = {beta:.3f} (fixed)")
        print(f"  Reduced chi^2 = {result.redchi:.5f}")
        print(f"  Degrees of freedom: {result.nfree}")
        print(f"{'='*60}\n")
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get unique L values and colors
        unique_L = sorted(df.index.get_level_values('L').unique())
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(unique_L)))
        
        # Plot raw and collapsed data with error bars for each L
        for i, L in enumerate(unique_L):
            L_data = df.loc[(slice(None), L), :]
            p_values = L_data.index.get_level_values('p')
            
            # Calculate mean and standard error for each point
            means = [np.mean(obs) for obs in L_data['observations']]
            stderrs = [np.std(obs) / np.sqrt(len(obs)) for obs in L_data['observations']]
            
            # Raw data plot with error bars
            axes[0].errorbar(p_values, means, yerr=stderrs,
                           marker='o', linestyle='-', color=colors[i],
                           label=f'L = {L}', capsize=3, markersize=6, alpha=0.8)
            
            # Collapsed data plot with error bars
            x_scaled = (p_values - fitted_pc) * L**(1/fitted_nu)
            y_scaled = np.array(means) * L**beta
            yerr_scaled = np.array(stderrs) * L**beta
            
            # Sort points by x_scaled for proper line connection
            sort_idx = np.argsort(x_scaled)
            x_scaled = x_scaled[sort_idx]
            y_scaled = y_scaled[sort_idx]
            yerr_scaled = yerr_scaled[sort_idx]
            
            axes[1].errorbar(x_scaled, y_scaled, yerr=yerr_scaled,
                           marker='o', linestyle='', color=colors[i],
                           label=f'L = {L}', capsize=3, markersize=6, alpha=0.8)
        
        # Add vertical line at p_c in raw data plot
        axes[0].axvline(x=fitted_pc, color='red', linestyle='--', linewidth=2,
                       alpha=0.7, label=f'$p_c$ = {fitted_pc:.3f}')
        
        # Add shaded region for p_c uncertainty
        axes[0].axvspan(fitted_pc - fitted_pc_err, fitted_pc + fitted_pc_err,
                       alpha=0.2, color='red')
        
        # Set labels and titles
        axes[0].set_xlabel('$p_{proj}$', fontsize=14)
        axes[0].set_ylabel('TMI', fontsize=14)
        axes[0].set_title('Raw TMI Data', fontsize=15, fontweight='bold')
        axes[0].legend(loc='best', fontsize=11)
        axes[0].grid(alpha=0.3, linestyle='--')
        
        axes[1].set_xlabel(r'$(p_{proj} - p_c) \, L^{1/\nu}$', fontsize=14)
        axes[1].set_ylabel(r'TMI $\cdot L^\beta$', fontsize=14)
        title_str = f'Data Collapse: $p_c$ = {fitted_pc:.4f}±{fitted_pc_err:.4f}, '
        title_str += f'$\\nu$ = {fitted_nu:.3f}±{fitted_nu_err:.3f}'
        axes[1].set_title(title_str, fontsize=15, fontweight='bold')
        axes[1].legend(loc='best', fontsize=11)
        axes[1].grid(alpha=0.3, linestyle='--')
        axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        axes[1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
        fig_path = os.path.join(output_dir, f'data_collapse_{csv_basename}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved data collapse figure to: {fig_path}")
        
        # Save results to text file
        results_path = os.path.join(output_dir, f'data_collapse_{csv_basename}_results.txt')
        with open(results_path, 'w') as f:
            f.write("Data Collapse Analysis Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Input file: {csv_file}\n")
            f.write(f"L values: {unique_L}\n")
            f.write(f"p range: [{p_range[0]:.3f}, {p_range[1]:.3f}]\n\n")
            f.write("Fitted Parameters:\n")
            f.write(f"  p_c  = {fitted_pc:.6f} ± {fitted_pc_err:.6f}\n")
            f.write(f"  nu   = {fitted_nu:.6f} ± {fitted_nu_err:.6f}\n")
            f.write(f"  beta = {beta:.3f} (fixed)\n\n")
            f.write("Fit Quality:\n")
            f.write(f"  Reduced chi^2 = {result.redchi:.6f}\n")
            f.write(f"  Degrees of freedom = {result.nfree}\n")
        print(f"Saved results to: {results_path}")
        
        return fig, axes, result
        
    except Exception as e:
        print(f"Error performing data collapse: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Path to CSV file
    csv_file = '/scratch/ty296/plots/tmi_data_L8-12-16-20_pctrl0.400_n0_threshold1e-15.csv'
    
    # Perform data collapse analysis
    # Initial guesses: p_c ~ 0.75, nu ~ 1.0
    result = perform_data_collapse_analysis(
        csv_file=csv_file,
        pc_guess=0.75,      # Initial guess for critical point
        nu_guess=1.0,       # Initial guess for correlation length exponent
        beta=0,             # Scaling dimension (0 for TMI)
        L_min=None,         # Include all L values (8, 12, 16, 20)
        L_max=None,
        p_range=None,       # Include all pproj values (0.5 to 1.0)
        nu_vary=True,       # Fit nu
        p_c_vary=True,      # Fit p_c
        n_samples=100,      # Number of synthetic samples from mean/sem
        output_dir='/scratch/ty296/plots'
    )
    
    if result is not None:
        fig, axes, fit_result = result
        plt.show()
    else:
        print("Data collapse analysis failed!")

