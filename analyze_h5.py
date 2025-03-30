import h5py
import os

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: Dataset, Shape: {obj.shape}, Type: {obj.dtype}")
    else:
        print(f"{name}: Group")

# Analyze target file
print("Structure of final_results_L20.h5:")
try:
    with h5py.File('sv_comparison_L20_pctrl0.400_pc0.600/final_results_L20.h5', 'r') as f:
        f.visititems(print_structure)
except Exception as e:
    print(f"Error opening final_results_L20.h5: {e}")

# Analyze a chunk file
chunk_files = [f for f in os.listdir('sv_comparison_L20_pctrl0.400_pc0.800') if f.endswith('.h5')]
if chunk_files:
    chunk_file = os.path.join('sv_comparison_L20_pctrl0.400_pc0.800', chunk_files[0])
    print(f"\nStructure of chunk file: {chunk_file}")
    try:
        with h5py.File(chunk_file, 'r') as g:
            g.visititems(print_structure)
    except Exception as e:
        print(f"Error opening chunk file: {e}")
else:
    print("No chunk files found.") 