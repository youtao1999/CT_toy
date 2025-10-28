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
    
    # Step 2: Create plots from CSV files
    print("\n" + "="*60)
    print("Step 2: Creating plots from CSV files...")
    print("="*60)
    
    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        df = pd.read_csv(csv_file)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Get unique L values
        L_vals = sorted(df['L'].unique())
        colors = sns.color_palette("Blues", n_colors=len(L_vals)+2)[2:]
        
        # Plot mean ± SEM
        for idx, L in enumerate(L_vals):
            L_data = df[df['L'] == L].sort_values('pproj')
            ax1.errorbar(L_data['pproj'], L_data[f'{metric}_mean'], 
                        yerr=L_data[f'{metric}_sem'],
                        fmt='o-', capsize=3, label=f'L={L}', 
                        color=colors[idx], alpha=0.8, markersize=5)
        
        ax1.set_xlabel('pproj', fontsize=12)
        ax1.set_ylabel(f'{metric} Mean ± SEM', fontsize=12)
        ax1.set_title(f'{metric} vs pproj (combined data, {researcher})', fontsize=13)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot variance ± SEV
        for idx, L in enumerate(L_vals):
            L_data = df[df['L'] == L].sort_values('pproj')
            ax2.errorbar(L_data['pproj'], L_data[f'{metric}_variance'], 
                        yerr=L_data[f'{metric}_sev'],
                        fmt='s-', capsize=3, label=f'L={L}', 
                        color=colors[idx], alpha=0.8, markersize=5)
        
        ax2.set_xlabel('pproj', fontsize=12)
        ax2.set_ylabel(f'{metric} Variance ± SEV', fontsize=12)
        ax2.set_title(f'{metric} Variance vs pproj (combined data, {researcher})', fontsize=13)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Extract threshold from filename and save plot
        import re
        threshold_match = re.search(r'threshold([\d\.e\-\+]+)', csv_file)
        if threshold_match:
            threshold_str = threshold_match.group(1)
            output_file = f'{save_folder}/{metric}_combined_plot_threshold{threshold_str}_{researcher}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved plot: {output_file}")
            plt.close()
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)

