import h5py
import os

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: Dataset, Shape: {obj.shape}, Type: {obj.dtype}")
    else:
        print(f"{name}: Group")

# Analyze original file
print("Structure of original final_results_L20.h5:")
try:
    with h5py.File('sv_comparison_L20_pctrl0.400_pc0.600/final_results_L20.h5', 'r') as f:
        f.visititems(print_structure)
except Exception as e:
    print(f"Error opening original file: {e}")

# Analyze the combined file
print("\nStructure of combined final_results_L20.h5:")
try:
    with h5py.File('sv_comparison_L20_pctrl0.400_pc0.800/final_results_L20.h5', 'r') as f:
        f.visititems(print_structure)
except Exception as e:
    print(f"Error opening combined file: {e}")

# Compare file sizes
original_size = os.path.getsize('sv_comparison_L20_pctrl0.400_pc0.600/final_results_L20.h5') / (1024 * 1024 * 1024)
combined_size = os.path.getsize('sv_comparison_L20_pctrl0.400_pc0.800/final_results_L20.h5') / (1024 * 1024 * 1024)

print(f"\nFile sizes:")
print(f"Original file: {original_size:.2f} GB")
print(f"Combined file: {combined_size:.2f} GB") 