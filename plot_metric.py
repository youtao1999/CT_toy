#!/usr/bin/env python3
"""
Compare metric results across different system sizes L=8,12,16,20
"""
import sys
sys.path.append('/scratch/ty296')

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
from tqdm import tqdm
import seaborn as sns
from CT_MPS_mini.read_hdf5_func import calculate_mean_and_error, calculate_variance_and_error
def compute_entropy_from_singular_values(S, n=0, threshold=1e-15):
    """
    Compute von Neumann entropy from singular values with specified Rényi index.
    """
    S_pos = np.clip(S, threshold, None)
    eigenvalues = S_pos**2
    
    if n == 1:
        entropy = -np.sum(np.log(eigenvalues) * eigenvalues)
        return 0 if np.isnan(entropy) else entropy
    elif n == 0:
        return np.log((eigenvalues > threshold).sum())
    elif n == np.inf:
        return -np.log(np.max(eigenvalues))
    else:
        return np.log(np.sum(eigenvalues**n)) / (1-n)

def compute_tmi_from_singular_values(singular_values, n=0, threshold=1e-15):
    """
    Compute TMI from singular values using specified Rényi entropy index.
    """
    # Compute entropies for each region
    S_A = compute_entropy_from_singular_values(singular_values['A'], n, threshold)
    S_B = compute_entropy_from_singular_values(singular_values['B'], n, threshold)
    S_C = compute_entropy_from_singular_values(singular_values['C'], n, threshold)
    S_AB = compute_entropy_from_singular_values(singular_values['AB'], n, threshold)
    S_AC = compute_entropy_from_singular_values(singular_values['AC'], n, threshold)
    S_BC = compute_entropy_from_singular_values(singular_values['BC'], n, threshold)
    S_ABC = compute_entropy_from_singular_values(singular_values['ABC'], n, threshold)
    
    return S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

def load_tmi_data(filename, L, n=0, threshold=1e-15, metric = "I"):
    """
    Load TMI data from HDF5 file for a given L value.
    Returns pproj_values, metric_means, metric_sems, metric_variances, metric_sevs
    """
    print(f"Reading L={L} from: {filename}")
    
    with h5py.File(filename, 'r') as f:
        # Get the main group (should be pctrl0.400)
        main_group_name = list(f.keys())[0]
        main_group = f[main_group_name]
        
        # Extract pctrl value from group name
        pctrl_value = float(main_group_name.replace('pctrl', ''))
        
        # Get pproj values
        pproj_values = main_group['pproj'][:]
        
        # Get singular values
        sv_group = main_group['singular_values']
        
        num_pproj = len(pproj_values)
        num_samples = sv_group['A'].shape[1]
        
        print(f"  pctrl = {pctrl_value:.3f}, {num_pproj} pproj values, {num_samples} samples")
        
        # Compute TMI for each pproj and sample
        metric_means = []
        metric_sems = []

        metric_variances = []
        metric_sevs = []
        
        for pproj_idx in tqdm(range(num_pproj), desc=f"  Computing {metric} for L={L}", unit="pproj"):
            sample_metric_values = []
            for sample_idx in range(num_samples):
                singular_values = {
                    key: sv_group[key][pproj_idx, sample_idx] 
                    for key in sv_group.keys()
                }
                if metric == "I":
                    metric_value = compute_tmi_from_singular_values(singular_values, n, threshold)
                elif metric == "S":
                    metric_value = compute_entropy_from_singular_values(singular_values['AB'], n, threshold)
                sample_metric_values.append(metric_value)
            
            mean, sem = calculate_mean_and_error(sample_metric_values)
            metric_means.append(mean)
            metric_sems.append(sem)

            variance, sev = calculate_variance_and_error(sample_metric_values)
            metric_variances.append(variance)
            metric_sevs.append(sev)
        
        metric_means = np.array(metric_means)
        metric_sems = np.array(metric_sems)
        metric_variances = np.array(metric_variances)
        metric_sevs = np.array(metric_sevs)
        
        return pproj_values, metric_means, metric_sems, metric_variances, metric_sevs, pctrl_value, metric

def generate_csv_data(L_values=[8, 12, 16, 20], pctrl=0.4, n=0, threshold=1e-15, metric="I", force_recompute=False, save_folder='/scratch/ty296/plots'):
    """
    Generate CSV file with metric data for given parameters.
    
    Args:
        L_values: List of system sizes to process
        pctrl: Control measurement probability
        n: Rényi index for entropy (0 for Hartley, 1 for von Neumann)
        threshold: Threshold for singular values
        metric: Metric to compute ('I' for TMI, 'S' for entropy)
        force_recompute: If True, recompute even if CSV exists
        save_folder: Directory to save CSV files
    
    Returns:
        Path to generated CSV file
    """
    # Generate CSV filename based on parameters
    csv_filename = os.path.join(save_folder, f'{metric}_data_L{"-".join(map(str, L_values))}_pctrl{pctrl:.3f}_n{n}_threshold{threshold:.0e}.csv')
    
    # Check if file exists and skip if not forcing recompute
    if os.path.exists(csv_filename) and not force_recompute:
        print(f"\nCSV already exists: {csv_filename}")
        print(f"  Use force_recompute=True to regenerate")
        return csv_filename
    
    # Compute from HDF5 files
    if force_recompute:
        print(f"\nForce recompute enabled. Computing {metric} data from HDF5 files...")
    else:
        print(f"\nGenerating {metric} data from HDF5 files...")
    
    print(f"Processing {len(L_values)} system sizes with threshold={threshold:.1e}...\n")
    
    # List to store all data for CSV
    csv_data = []
    for idx, L in enumerate(L_values):
        filename = f'sv_L{L}_pctrl{pctrl:.3f}_p0.500-1.000/final_results_L{L}.h5'
        
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found, skipping L={L}")
            continue
        
        pproj, metric_mean, metric_sem, metric_variance, metric_sev, pctrl_val, metric_name = load_tmi_data(filename, L, n, threshold, metric)
        
        # Add data to CSV list
        for p, tm, ts, tv, tsev in zip(pproj, metric_mean, metric_sem, metric_variance, metric_sev):
            csv_data.append({
                'L': L,
                'pctrl': pctrl_val,
                'pproj': p,
                f'{metric}_mean': tm,
                f'{metric}_sem': ts,
                f'{metric}_variance': tv,
                f'{metric}_sev': tsev
            })
        
        print()  # Add blank line after each L for better readability
    
    # Save to CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_filename, index=False)
        print(f"\nSaved {metric} data to: {csv_filename}")
        print(f"  Total data points: {len(csv_data)}")
    else:
        print("\nWarning: No data to save!")
        return None
    
    return csv_filename

def batch_generate_csv(L_values=[8, 12, 16, 20], pctrl=0.4, n=0, threshold_values=None, metric="I", force_recompute=False, save_folder='/scratch/ty296/plots'):
    """
    Generate CSV files for multiple threshold values.
    
    Args:
        L_values: List of system sizes to process
        pctrl: Control measurement probability
        n: Rényi index for entropy
        threshold_values: List of threshold values to process
        metric: Metric to compute ('I' for TMI, 'S' for entropy)
        force_recompute: If True, recompute even if CSV exists
        save_folder: Directory to save CSV files
    
    Returns:
        List of generated CSV file paths
    """
    if threshold_values is None:
        threshold_values = [1e-15]
    
    csv_files = []
    for threshold in threshold_values:
        print(f"\n{'='*60}")
        print(f"Processing threshold = {threshold:.1e}")
        print(f"{'='*60}")
        
        csv_file = generate_csv_data(
            L_values=L_values,
            pctrl=pctrl,
            n=n,
            threshold=threshold,
            metric=metric,
            force_recompute=force_recompute,
            save_folder=save_folder
        )
        
        if csv_file:
            csv_files.append(csv_file)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(csv_files)} CSV files")
    print(f"{'='*60}")
    
    return csv_files

def fixed_L_threshold_comparison_plot(save_folder: str, L_list: list, n: int, pctrl: float, metric: str, threshold_values: list):
    """
    Plot the comparison of the metric for different thresholds and multiple L values.
    Each L gets a different color gradient (blue, green, red, etc.)
    
    Args:
        save_folder: Path to folder containing CSV files
        L_list: List of L values to plot
        n: Index for Rényi entropy (e.g., 0 for Hartley)
        pctrl: Control measurement probability
        metric: Metric to plot ('I' for TMI, 'S' for entropy)
        threshold_values: List of specific threshold values to plot
    """
    
    # Read from csv files matching the pattern
    L_str = "-".join(map(str, L_list))
    csv_pattern = f'{metric}_data_L{L_str}_pctrl{pctrl:.3f}_n{n}_threshold*.csv'
    csv_paths = glob.glob(os.path.join(save_folder, csv_pattern))
    
    if not csv_paths:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return
    
    # Organize data by L and threshold values
    plot_data = {}  # {L: {threshold: data_dict}}
    
    for L in L_list:
        plot_data[L] = {}
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            # Find data specific to L
            df_L = df[df['L'] == L]
            
            if len(df_L) == 0:
                continue
                
            # Extract threshold from csv_path filename
            threshold_match = re.search(r'threshold([\d\.e\-\+]+)', csv_path)
            if threshold_match:
                threshold_str = threshold_match.group(1)
                threshold_val = float(threshold_str)
            else:
                continue
            
            # Only store data for requested threshold values
            if threshold_val not in threshold_values:
                continue
                
            # Sort by pproj and get corresponding values
            sorted_indices = np.argsort(df_L['pproj'])
            plot_data[L][threshold_val] = {
                'pproj': df_L['pproj'].iloc[sorted_indices].values,
                'mean': df_L[f'{metric}_mean'].iloc[sorted_indices].values,
                'sem': df_L[f'{metric}_sem'].iloc[sorted_indices].values,
                'variance': df_L[f'{metric}_variance'].iloc[sorted_indices].values,
                'sev': df_L[f'{metric}_sev'].iloc[sorted_indices].values
            }

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Color palettes for different L values
    color_palettes = ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"]
    
    # Sort threshold values for consistent ordering
    sorted_thresholds = sorted(threshold_values)
    n_thresholds = len(sorted_thresholds)
    
    # Plot for each L value
    for L_idx, L in enumerate(L_list):
        if L not in plot_data or len(plot_data[L]) == 0:
            continue
        
        # Use different color palette for each L
        palette_name = color_palettes[L_idx % len(color_palettes)]
        colors = sns.color_palette(palette_name, n_colors=n_thresholds+2)[1:]  # Skip lightest shade

        # Plot 1: pproj vs mean ± sem
        for i, threshold in enumerate(sorted_thresholds):
            if threshold not in plot_data[L]:
                print(f"Warning: threshold {threshold} not found for L={L}")
                continue
                
            data = plot_data[L][threshold]
            ax1.errorbar(data['pproj'], data['mean'], yerr=data['sem'], 
                        label=f'L={L}, threshold={threshold:.1e}', marker='o', capsize=3, 
                        color=colors[n_thresholds-1-i], alpha=0.8, markersize=5)

            ax2.errorbar(data['pproj'], data['variance'], yerr=data['sev'],
                        label=f'L={L}, threshold={threshold:.1e}', marker='s', capsize=3, 
                        color=colors[n_thresholds-1-i], alpha=0.8, markersize=5)

    ax1.set_xlabel('pproj', fontsize=12)
    ax1.set_ylabel(f'{metric} Mean ± SEM', fontsize=12)
    ax1.set_title(f'{metric} Mean vs pproj (pctrl={pctrl:.3f}, n={n})', fontsize=13)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('pproj', fontsize=12)
    ax2.set_ylabel(f'{metric} Variance ± SEV', fontsize=12)
    ax2.set_title(f'{metric} Variance vs pproj (pctrl={pctrl:.3f}, n={n})', fontsize=13)
    ax2.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    L_str = '_'.join(map(str, L_list))
    output_file = f'{save_folder}/{metric}_threshold_comparison_L{L_str}_pctrl{pctrl:.3f}_n{n}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\nThreshold comparison saved to: {output_file}')
    
    return fig, plot_data

if __name__ == "__main__":
    # Example workflow: Generate CSV files for multiple thresholds, then plot
    
    L_values = [8, 12, 16, 20]
    pctrl = 0.4
    n = 0
    metric = "I"  # or "S" for entropy
    save_folder = '/scratch/ty296/plots'
    
    # Define threshold values to test
    threshold_values = np.logspace(-15, -10, 6)  # 6 threshold values from 1e-15 to 1e-10
    
    # Step 1: Generate CSV files for all thresholds
    print("Step 1: Generating CSV files for all thresholds...")
    csv_files = batch_generate_csv(
        L_values=L_values,
        pctrl=pctrl,
        n=n,
        threshold_values=threshold_values,
        metric=metric,
        force_recompute=True,  # Set to False to skip existing files
        save_folder=save_folder
    )
    
    # Step 2: Plot comparison across all thresholds and L values
    print("\n\nStep 2: Creating comparison plot...")
    fig, plot_data = fixed_L_threshold_comparison_plot(
        save_folder=save_folder,
        L_list=L_values,
        n=n,
        pctrl=pctrl,
        metric=metric,
        threshold_values=threshold_values
    )
    
    plt.show()

