#!/usr/bin/env python3
"""
Final attempt at data collapse with multiple strategies.
"""
import sys
sys.path.append('/scratch/ty296/CT_toy')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fss import DataCollapse
import os

def create_observations_from_stats(mean, sem, n_samples=100):
    """Create synthetic observations from mean and standard error."""
    std = sem * np.sqrt(n_samples)
    observations = np.random.normal(mean, std, n_samples)
    return observations

def load_and_prepare_data(csv_file, n_samples=100):
    """Load and prepare data for DataCollapse."""
    df_raw = pd.read_csv(csv_file)
    
    data_list = []
    for _, row in df_raw.iterrows():
        observations = create_observations_from_stats(
            row['tmi_mean'], 
            row['tmi_sem'], 
            n_samples
        )
        data_list.append({
            'p': row['pproj'],
            'L': row['L'],
            'observations': observations
        })
    
    df = pd.DataFrame(data_list)
    df = df.set_index(['p', 'L'])
    return df

def plot_datacollapse_result(df, result, output_name, p_c, nu):
    """Create and save data collapse plots."""
    fitted_pc = result.params['p_c'].value
    fitted_pc_err = result.params['p_c'].stderr if result.params['p_c'].stderr is not None else 0
    fitted_nu = result.params['nu'].value
    fitted_nu_err = result.params['nu'].stderr if result.params['nu'].stderr is not None else 0
    beta = 0
    
    # Create main figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    unique_L = sorted(df.index.get_level_values('L').unique())
    # Use seaborn blue palette - deeper blue for larger L
    colors = sns.color_palette("Blues", n_colors=len(unique_L)+2)[2:]  # Skip the lightest shades
    
    for i, L in enumerate(unique_L):
        L_data = df.loc[(slice(None), L), :]
        p_values = L_data.index.get_level_values('p')
        
        means = [np.mean(obs) for obs in L_data['observations']]
        stderrs = [np.std(obs) / np.sqrt(len(obs)) for obs in L_data['observations']]
        
        # Raw data on main plot
        ax.errorbar(p_values, means, yerr=stderrs,
                   marker='o', linestyle='-', color=colors[i],
                   label=f'L = {int(L)}', capsize=3, markersize=5, alpha=0.7, linewidth=1.5)
        
    ax.set_xlabel(r'$p_{\mathrm{proj}}$', fontsize=16)
    ax.set_ylabel(r'$I^{(0)}_3(p_{\mathrm{proj}})$', fontsize=16)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    # Create inset for collapsed data (moved higher)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_inset = inset_axes(ax, width="45%", height="45%", loc='lower right',
                         bbox_to_anchor=(0, 0.15, 1, 1), bbox_transform=ax.transAxes)
    
    # Plot collapsed data in inset
    for i, L in enumerate(unique_L):
        L_data = df.loc[(slice(None), L), :]
        p_values = L_data.index.get_level_values('p')
        
        means = [np.mean(obs) for obs in L_data['observations']]
        stderrs = [np.std(obs) / np.sqrt(len(obs)) for obs in L_data['observations']]
        
        # Collapsed data
        x_scaled = (p_values - fitted_pc) * L**(1/fitted_nu)
        y_scaled = np.array(means) * L**beta
        yerr_scaled = np.array(stderrs) * L**beta
        
        sort_idx = np.argsort(x_scaled)
        x_scaled = x_scaled[sort_idx]
        y_scaled = y_scaled[sort_idx]
        yerr_scaled = yerr_scaled[sort_idx]
        
        ax_inset.errorbar(x_scaled, y_scaled, yerr=yerr_scaled,
                         marker='o', linestyle='', color=colors[i],
                         capsize=2, markersize=3, alpha=0.7)
    
    # Build the xlabel with formatted values
    xlabel_text = r'$(p_{\mathrm{proj}} - ' + f'{p_c:.3f})' + r' L^{1/' + f'{nu:.3f}' + r'}$'
    ax_inset.set_xlabel(xlabel_text, fontsize=11)
    ax_inset.grid(alpha=0.3, linestyle='--')
    ax_inset.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_inset.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax_inset.tick_params(labelsize=9)
    
    plt.tight_layout()
    
    output_dir = '/scratch/ty296/plots'
    fig_path = os.path.join(output_dir, output_name)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {fig_path}")
    
    return fig

# Main script
csv_file = '/scratch/ty296/plots/tmi_data_merged_L8-12-16-20_pc0.6-0.8_n0_threshold1e-15.csv'

print("="*60)
print("Data Collapse on Combined Dataset - Multiple Strategies")
print("="*60)

df = load_and_prepare_data(csv_file, n_samples=100)
print(f"\nLoaded {len(df)} data points")
print(f"L values: {sorted(df.index.get_level_values('L').unique())}")

# Strategy 1: Try with fixed nu first, then optimize p_c
print("\n" + "="*60)
print("Strategy 1: Fix nu=0.67, optimize p_c")
print("="*60)

try:
    dc1 = DataCollapse(df=df, p_='p', L_='L', params={}, 
                      p_range=[0.5, 0.9], Lmin=None, Lmax=None)
    
    result1 = dc1.datacollapse(p_c=0.75, nu=0.67, beta=0,
                              p_c_vary=True, nu_vary=False, beta_vary=False)
    
    print("\nSTRATEGY 1 SUCCESS!")
    pc_err = result1.params['p_c'].stderr if result1.params['p_c'].stderr is not None else 0.0
    print(f"  p_c = {result1.params['p_c'].value:.5f} ± {pc_err:.5f}")
    print(f"  nu  = {result1.params['nu'].value:.5f} (fixed)")
    print(f"  Reduced chi^2 = {result1.redchi:.5f}")
    
    fig1 = plot_datacollapse_result(df, result1, 'data_collapse_combined_strategy1_fixed_nu.png', 
                                     result1.params['p_c'].value, result1.params['nu'].value)
    
    # Now try optimizing both with the found p_c as starting point
    print("\n  Attempting to optimize both parameters from this starting point...")
    try:
        result1b = dc1.datacollapse(p_c=result1.params['p_c'].value, nu=0.67, beta=0,
                                   p_c_vary=True, nu_vary=True, beta_vary=False)
        
        print("  BOTH PARAMETERS OPTIMIZED!")
        pc_err_b = result1b.params['p_c'].stderr if result1b.params['p_c'].stderr is not None else 0.0
        nu_err_b = result1b.params['nu'].stderr if result1b.params['nu'].stderr is not None else 0.0
        print(f"  p_c = {result1b.params['p_c'].value:.5f} ± {pc_err_b:.5f}")
        print(f"  nu  = {result1b.params['nu'].value:.5f} ± {nu_err_b:.5f}")
        print(f"  Reduced chi^2 = {result1b.redchi:.5f}")
        
        fig1b = plot_datacollapse_result(df, result1b, 'data_collapse_combined_full_optimization.png',
                                          result1b.params['p_c'].value, result1b.params['nu'].value)
        
    except Exception as e:
        print(f"  Full optimization failed: {str(e)}")
        print("  Using fixed-nu result instead")
    
except Exception as e:
    print(f"Strategy 1 failed: {str(e)}")

# Strategy 2: Exclude L=8 and use p_range for fitting
print("\n" + "="*60)
print("Strategy 2: L=12,16,20 only")
print("="*60)

# Filter data to exclude L=8 only
L_vals = df.index.get_level_values('L')
mask = (L_vals >= 12)
df_restricted = df.loc[mask]
print(f"Filtered to {len(df_restricted)} data points with L >= 12")
print(f"L values in restricted dataset: {sorted(df_restricted.index.get_level_values('L').unique())}")

# Use specified initial guess
pc_guess = 0.75
nu_guess = 0.67
delta_p = 0.08
p_range = [pc_guess - delta_p, pc_guess + delta_p]
print(f"\nUsing initial guess: p_c = {pc_guess}, nu = {nu_guess}")
print(f"Fitting range: p_proj in [{p_range[0]}, {p_range[1]}]")

try:
    dc2 = DataCollapse(df=df_restricted, p_='p', L_='L', params={}, 
                      p_range=p_range, Lmin=None, Lmax=None)
    
    # Optimize both p_c and nu (beta fixed at 0)
    print("Optimizing both p_c and nu (beta fixed at 0)...")
    result2 = dc2.datacollapse(p_c=pc_guess, nu=nu_guess, beta=0,
                              p_c_vary=True, nu_vary=True, beta_vary=False)
    print(f"SUCCESS!")
except Exception as e:
    print(f"Failed: {str(e)}")
    result2 = None

if result2 is not None:
    print("\nSTRATEGY 2 SUCCESS - BOTH PARAMETERS OPTIMIZED!")
    pc_err2 = result2.params['p_c'].stderr if result2.params['p_c'].stderr is not None else 0.0
    nu_err2 = result2.params['nu'].stderr if result2.params['nu'].stderr is not None else 0.0
    print(f"  p_c  = {result2.params['p_c'].value:.5f} ± {pc_err2:.5f}")
    print(f"  nu   = {result2.params['nu'].value:.5f} ± {nu_err2:.5f}")
    print(f"  beta = 0.000 (fixed)")
    print(f"  Reduced chi^2 = {result2.redchi:.5f}")
    
    fig2 = plot_datacollapse_result(df_restricted, result2, 'data_collapse_combined_both_optimized.png', result2.params['p_c'].value, result2.params['nu'].value)
    
    # Save results to text file
    output_dir = '/scratch/ty296/plots'
    results_path = os.path.join(output_dir, 'data_collapse_combined_full_results.txt')
    with open(results_path, 'w') as f:
        f.write("Data Collapse Analysis Results (Combined Dataset - Both Parameters Optimized)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input file: {csv_file}\n")
        f.write(f"L values: [12, 16, 20] (L=8 excluded)\n")
        f.write(f"Fitting range: p_proj in [{p_range[0]:.2f}, {p_range[1]:.2f}]\n")
        f.write(f"Initial guess: p_c={pc_guess}, nu={nu_guess}\n\n")
        f.write("Fitted Parameters:\n")
        f.write(f"  p_c  = {result2.params['p_c'].value:.6f} ± {pc_err2:.6f}\n")
        f.write(f"  nu   = {result2.params['nu'].value:.6f} ± {nu_err2:.6f}\n")
        f.write(f"  beta = 0.000 (fixed)\n\n")
        f.write("Fit Quality:\n")
        f.write(f"  Reduced chi^2 = {result2.redchi:.6f}\n")
        f.write(f"  Degrees of freedom = {result2.nfree}\n")
    print(f"  Saved results to: {results_path}")
    
else:
    print("\nSTRATEGY 2 FAILED - No valid fit obtained")

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)

