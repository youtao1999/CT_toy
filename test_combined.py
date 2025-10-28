#!/usr/bin/env python3
"""
Quick test of combined file processing
"""
import sys
sys.path.append('/scratch/ty296')
sys.path.append('/scratch/ty296/CT_toy')

import numpy as np
from plot_metric import generate_csv_from_combined

# Test with one threshold
combined_file = '/scratch/ty296/CT_toy/pctrl0.4_combined.h5'
L_values = [8, 12, 16, 20]
pctrl_range = [0.6, 0.8]
n = 0
threshold = 1e-15
metric = "I"
researcher = 'haining'
save_folder = '/scratch/ty296/plots'

print("Testing combined file processing with single threshold...")
csv_file = generate_csv_from_combined(
    combined_file=combined_file,
    L_values=L_values,
    pctrl_range=pctrl_range,
    n=n,
    threshold=threshold,
    metric=metric,
    researcher=researcher,
    force_recompute=True,
    save_folder=save_folder
)

print(f"\nSuccess! Generated: {csv_file}")

