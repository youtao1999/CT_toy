import h5py

# Check a file from sv_L* folder
with h5py.File('sv_L8_pctrl0.400_p0.700-0.800/final_results_L8.h5', 'r') as f:
    p_fixed_group = list(f.keys())[0]  # Get the first group
    print(f"Keys in {p_fixed_group}: {list(f[p_fixed_group].keys())}")

# Check a file from sv_comparison_L* folder
with h5py.File('sv_comparison_L8_pctrl0.400_pc0.800/final_results_L8.h5', 'r') as f:
    p_fixed_group = list(f.keys())[0]  # Get the first group
    print(f"Keys in {p_fixed_group}: {list(f[p_fixed_group].keys())}")
