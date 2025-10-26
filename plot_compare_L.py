#!/usr/bin/env python3
"""
Compare TMI results across different system sizes L=8,12,16,20
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm

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

def load_tmi_data(filename, L, n=0, threshold=1e-15):
    """
    Load TMI data from HDF5 file for a given L value.
    Returns pproj_values, tmi_means, tmi_sems
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
        tmi_means = []
        tmi_sems = []
        
        for pproj_idx in tqdm(range(num_pproj), desc=f"  Computing TMI for L={L}", unit="pproj"):
            sample_tmis = []
            for sample_idx in range(num_samples):
                singular_values = {
                    key: sv_group[key][pproj_idx, sample_idx] 
                    for key in sv_group.keys()
                }
                tmi = compute_tmi_from_singular_values(singular_values, n, threshold)
                sample_tmis.append(tmi)
            
            tmi_means.append(np.mean(sample_tmis))
            tmi_sems.append(np.std(sample_tmis) / np.sqrt(len(sample_tmis)))
        
        tmi_means = np.array(tmi_means)
        tmi_sems = np.array(tmi_sems)
        
        return pproj_values, tmi_means, tmi_sems, pctrl_value

def plot_comparison(L_values=[8, 12, 16, 20], pctrl=0.4, n=0, threshold=1e-15, force_recompute=False):
    """
    Plot TMI comparison for different L values at fixed pctrl.
    
    Args:
        L_values: List of system sizes to compare
        pctrl: Control measurement probability
        n: Rényi index for entropy (0 for Hartley, 1 for von Neumann)
        threshold: Threshold for singular values
        force_recompute: If True, recompute even if CSV exists
    """
    # Generate CSV filename based on parameters
    csv_filename = f'/scratch/ty296/plots/tmi_data_L{"-".join(map(str, L_values))}_pctrl{pctrl:.3f}_n{n}_threshold{threshold:.0e}.csv'
    
    all_data = {}
    
    # Try to load from CSV if it exists and force_recompute is False
    if os.path.exists(csv_filename) and not force_recompute:
        print(f"\nLoading TMI data from existing CSV: {csv_filename}")
        df = pd.read_csv(csv_filename)
        
        # Group by L and reconstruct all_data dictionary
        for L in L_values:
            L_data = df[df['L'] == L]
            if len(L_data) > 0:
                pproj = L_data['pproj'].values
                tmi_mean = L_data['tmi_mean'].values
                tmi_sem = L_data['tmi_sem'].values
                all_data[L] = (pproj, tmi_mean, tmi_sem)
                print(f"  Loaded L={L}: {len(pproj)} data points")
            else:
                print(f"  Warning: No data for L={L} in CSV")
        
        print(f"\nLoaded data for L={list(all_data.keys())}")
    else:
        # Compute from HDF5 files
        if force_recompute:
            print(f"\nForce recompute enabled. Computing TMI data from HDF5 files...")
        else:
            print(f"\nCSV file not found. Computing TMI data from HDF5 files...")
        
        print(f"Processing {len(L_values)} system sizes...\n")
        
        # List to store all data for CSV
        csv_data = []
        
        for idx, L in enumerate(L_values):
            filename = f'sv_L{L}_pctrl{pctrl:.3f}_p0.500-1.000/final_results_L{L}.h5'
            
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found, skipping L={L}")
                continue
            
            pproj, tmi_mean, tmi_sem, pctrl_val = load_tmi_data(filename, L, n, threshold)
            all_data[L] = (pproj, tmi_mean, tmi_sem)
            
            # Add data to CSV list
            for p, tm, ts in zip(pproj, tmi_mean, tmi_sem):
                csv_data.append({
                    'L': L,
                    'pctrl': pctrl_val,
                    'pproj': p,
                    'tmi_mean': tm,
                    'tmi_sem': ts
                })
            
            print()  # Add blank line after each L for better readability
        
        # Save to CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False)
            print(f"\nSaved TMI data to: {csv_filename}")
            print(f"  Total data points: {len(csv_data)}")
        else:
            print("\nWarning: No data to save!")
    
    # Now plot the data
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(L_values)))
    
    # Plot each L value's data
    for idx, L in enumerate(L_values):
        if L not in all_data:
            continue
            
        pproj, tmi_mean, tmi_sem = all_data[L]
        
        # Plot 1: TMI vs pproj with error bars
        ax = axes[0]
        ax.errorbar(pproj, tmi_mean, yerr=tmi_sem, 
                   fmt='o-', capsize=3, label=f'L={L}', 
                   color=colors[idx], alpha=0.8, markersize=5)
        
        # Plot 2: TMI vs pproj with shaded error region (no markers for clarity)
        ax = axes[1]
        ax.plot(pproj, tmi_mean, '-', label=f'L={L}', 
               color=colors[idx], linewidth=2, alpha=0.8)
        ax.fill_between(pproj, 
                        tmi_mean - tmi_sem, 
                        tmi_mean + tmi_sem,
                        alpha=0.2, color=colors[idx])
    
    # Configure first plot (with error bars)
    ax = axes[0]
    ax.set_xlabel('pproj', fontsize=12)
    ax.set_ylabel(f'TMI (n={n})', fontsize=12)
    ax.set_title(f'TMI vs pproj at pctrl={pctrl:.3f} (with error bars)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # Configure second plot (smooth lines)
    ax = axes[1]
    ax.set_xlabel('pproj', fontsize=12)
    ax.set_ylabel(f'TMI (n={n})', fontsize=12)
    ax.set_title(f'TMI vs pproj at pctrl={pctrl:.3f} (shaded SEM)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'/scratch/ty296/plots/tmi_comparison_L{"-".join(map(str, L_values))}_pctrl{pctrl:.3f}_n{n}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for L in sorted(all_data.keys()):
        pproj, tmi_mean, tmi_sem = all_data[L]
        print(f"\nL={L}:")
        print(f"  pproj range: [{pproj[0]:.3f}, {pproj[-1]:.3f}]")
        print(f"  TMI range: [{tmi_mean.min():.6f}, {tmi_mean.max():.6f}]")
        print(f"  Mean TMI: {tmi_mean.mean():.6f}")
        print(f"  Mean SEM: {tmi_sem.mean():.6f}")
    
    return fig, all_data

if __name__ == "__main__":
    # Compare L=8,12,16,20 at pctrl=0.4
    fig, data = plot_comparison(
        L_values=[8, 12, 16, 20],
        pctrl=0.4,
        n=0,
        threshold=1e-15
    )
    
    plt.show()

