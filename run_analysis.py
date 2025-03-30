import sys
import os
import numpy as np
from read_tmi_compare_results import TMIAnalyzer

# Parameters from command line
pc_guess = 0.5
nu_guess = 1.33
p_fixed = 0.0
p_fixed_name = "pctrl"
bootstrap = False
l_min = 12
l_max = 20
p_range = (0.35, 0.65)
nu_range = (0.3, 1.5)
threshold_min = -15
threshold_max = -5
threshold_steps = 80
output_folder = "tmi_compare_results"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Run analysis for each threshold
for threshold in np.logspace(threshold_min, threshold_max, threshold_steps):
    print("\n\n" + "="*80)
    print("Running analysis with threshold = {:.1e}".format(threshold))
    print("="*80 + "\n")
    
    analyzer = TMIAnalyzer(
        pc_guess=pc_guess,
        nu_guess=nu_guess,
        p_fixed=p_fixed,
        p_fixed_name=p_fixed_name,
        threshold=threshold,
        output_folder=output_folder
    )
    
    results = analyzer.result(
        bootstrap=bootstrap,
        L_min=l_min,
        L_max=l_max,
        p_range=p_range,
        nu_range=nu_range
    )
    
    print("\nCompleted analysis for threshold = {:.1e}".format(threshold))
