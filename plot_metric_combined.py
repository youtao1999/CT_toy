#!/usr/bin/env python3
"""
Process pctrl0.4_combined.h5 and produce plots using functions from plot_metric.py
"""
import sys
sys.path.append('/scratch/ty296')
sys.path.append('/scratch/ty296/CT_toy')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot_metric import batch_generate_csv_from_combined

if __name__ == "__main__":
    # Configuration
    combined_file = '/scratch/ty296/CT_toy/pctrl0.4_combined.h5'
    L_values = [8, 12, 16, 20]
    pctrl_range = [0.6, 0.8]  # pctrl values available in combined file
    n = 0
    metric = "S"  # or "S" for entropy
    researcher = 'haining'  # or 'tao'
    save_folder = '/scratch/ty296/plots'
    
    # Define threshold values to test
    threshold_values = np.logspace(-15, -10, 6)  # 6 threshold values from 1e-15 to 1e-10
    
    print("="*60)
    print("Processing Combined HDF5 File")
    print("="*60)
    print(f"Input file: {combined_file}")
    print(f"L values: {L_values}")
    print(f"pctrl range: {pctrl_range}")
    print(f"Metric: {metric}")
    print(f"Researcher dataset: {researcher}")
    print(f"Thresholds: {len(threshold_values)} values")
    print("="*60)
    
    # Step 1: Generate CSV files for all thresholds from combined file
    print("\nStep 1: Generating CSV files from combined HDF5...")
    csv_files = batch_generate_csv_from_combined(
        combined_file=combined_file,
        L_values=L_values,
        pctrl_range=pctrl_range,
        n=n,
        threshold_values=threshold_values,
        metric=metric,
        researcher=researcher,
        force_recompute=False,  # Set to False to skip existing files
        save_folder=save_folder
    )
    
    print("\n" + "="*60)
    print("CSV Generation Complete!")
    print("="*60)
    print(f"Generated {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Step 2: Create combined plot from all CSV files
    print("\n" + "="*60)
    print("Step 2: Creating combined plot from all CSV files...")
    print("="*60)
    
    # Organize data by L and threshold
    import re
    plot_data = {}  # {L: {threshold: data_dict}}
    
    for csv_file in csv_files:
        print(f"\nLoading: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Extract threshold from filename
        threshold_match = re.search(r'threshold([\d\.e\-\+]+)', csv_file)
        if not threshold_match:
            print(f"  Warning: Could not extract threshold from {csv_file}")
            continue
        
        threshold_str = threshold_match.group(1)
        try:
            threshold_val = float(threshold_str)
        except ValueError:
            print(f"  Warning: Could not parse threshold value: {threshold_str}")
            continue
        
        # Store data for each L value
        for L in L_values:
            if L not in plot_data:
                plot_data[L] = {}
            
            L_data = df[df['L'] == L].sort_values('pproj')
            
            if len(L_data) == 0:
                continue
            
            plot_data[L][threshold_val] = {
                'pproj': L_data['pproj'].values,
                'mean': L_data[f'{metric}_mean'].values,
                'sem': L_data[f'{metric}_sem'].values,
                'variance': L_data[f'{metric}_variance'].values,
                'sev': L_data[f'{metric}_sev'].values
            }
    
    # Create combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color palettes for different L values
    color_palettes = ["Blues", "Greens", "Reds", "Purples", "Oranges", "Greys"]
    
    # Sort threshold values for consistent ordering
    sorted_thresholds = sorted(threshold_values)
    n_thresholds = len(sorted_thresholds)
    
    # Plot for each L value
    for L_idx, L in enumerate(L_values):
        if L not in plot_data or len(plot_data[L]) == 0:
            print(f"Warning: No data found for L={L}")
            continue
        
        # Use different color palette for each L
        palette_name = color_palettes[L_idx % len(color_palettes)]
        colors = sns.color_palette(palette_name, n_colors=n_thresholds+2)[1:]  # Skip lightest shade
        
        # Plot for each threshold
        for i, threshold in enumerate(sorted_thresholds):
            if threshold not in plot_data[L]:
                print(f"Warning: threshold {threshold} not found for L={L}")
                continue
            
            data = plot_data[L][threshold]
            
            # Plot mean ± SEM
            ax1.errorbar(data['pproj'], data['mean'], yerr=data['sem'], 
                        label=f'L={L}, threshold={threshold:.1e}', marker='o', capsize=3, 
                        color=colors[n_thresholds-1-i], alpha=0.8, markersize=5)
            
            # Plot variance ± SEV
            ax2.errorbar(data['pproj'], data['variance'], yerr=data['sev'],
                        label=f'L={L}, threshold={threshold:.1e}', marker='s', capsize=3, 
                        color=colors[n_thresholds-1-i], alpha=0.8, markersize=5)
    
    # Format axes
    pctrl_str = f"{min(pctrl_range):.1f}-{max(pctrl_range):.1f}"
    
    ax1.set_xlabel('pproj', fontsize=12)
    ax1.set_ylabel(f'{metric} Mean ± SEM', fontsize=12)
    ax1.set_title(f'{metric} Mean vs pproj (pctrl={pctrl_str}, n={n}, {researcher})', fontsize=13)
    # ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('pproj', fontsize=12)
    ax2.set_ylabel(f'{metric} Variance ± SEV', fontsize=12)
    ax2.set_title(f'{metric} Variance vs pproj (pctrl={pctrl_str}, n={n}, {researcher})', fontsize=13)
    ax2.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save combined plot
    L_str = '_'.join(map(str, L_values))
    output_file = f'{save_folder}/{metric}_combined_threshold_comparison_L{L_str}_pc{pctrl_str}_n{n}_{researcher}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCombined plot saved to: {output_file}")
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)

