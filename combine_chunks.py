import h5py
import os
import numpy as np
from tqdm import tqdm
import re

# Input and output directories
input_dir = 'sv_comparison_L20_pctrl0.400_pc0.800'
output_dir = 'sv_comparison_L20_pctrl0.400_pc0.800'
output_file = os.path.join(output_dir, 'final_results_L20.h5')

# Get all chunk files
chunk_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.h5')])
print(f"Found {len(chunk_files)} chunk files")

# Extract chunk indices
chunk_indices = []
pattern = r'rank(\d+)_chunk(\d+)'
for filename in chunk_files:
    match = re.search(pattern, filename)
    if match:
        rank = int(match.group(1))
        chunk_idx = int(match.group(2))
        chunk_indices.append((rank, chunk_idx, filename))

# Sort by indices (rank, chunk_idx)
chunk_indices.sort()
print(f"Sorted {len(chunk_indices)} chunk files")

# Initialize variables to store dimensions and data
total_samples = 0
datasets_info = {}
pctrl_value = None

# First pass: determine dimensions and collect metadata
print("Analyzing chunks to determine dimensions...")
for _, _, filename in tqdm(chunk_indices):
    file_path = os.path.join(input_dir, filename)
    with h5py.File(file_path, 'r') as f:
        # Get pctrl value if not already set
        if pctrl_value is None and 'pctrl' in f:
            pctrl_value = f['pctrl'][()]
        
        # Get number of samples in this chunk
        # Using haining/singular_values/A as reference
        if 'haining' in f and 'singular_values' in f['haining'] and 'A' in f['haining']['singular_values']:
            chunk_samples = f['haining']['singular_values']['A'].shape[1]
            total_samples += chunk_samples
            
            # Collect dataset info if not already done
            if not datasets_info:
                for group_name in ['haining', 'tao']:
                    if group_name in f:
                        group = f[group_name]
                        if 'singular_values' in group:
                            sv_group = group['singular_values']
                            for dataset_name in sv_group.keys():
                                dataset = sv_group[dataset_name]
                                shape = dataset.shape
                                # Store info: (num_samples_dim_index, other_dims, dtype)
                                datasets_info[f"{group_name}/singular_values/{dataset_name}"] = (
                                    1,  # samples dimension index
                                    (shape[0], shape[2]),  # other dimensions
                                    dataset.dtype
                                )

# Get pproj dataset info
pproj_data = None
for _, _, filename in chunk_indices:
    file_path = os.path.join(input_dir, filename)
    with h5py.File(file_path, 'r') as f:
        if 'pproj' in f:
            pproj_data = f['pproj'][:]
            break

print(f"Total samples across all chunks: {total_samples}")
print(f"pctrl value: {pctrl_value}")
print(f"Datasets to combine: {list(datasets_info.keys())}")

# Create output file and prepare datasets
with h5py.File(output_file, 'w') as out_f:
    # Create main group
    main_group = out_f.create_group(f"pctrl{pctrl_value:.3f}")
    
    # Create pproj dataset in main group
    if pproj_data is not None:
        main_group.create_dataset('pproj', data=pproj_data)
    
    # Create datasets for singular values
    for dataset_path, (samples_dim, other_dims, dtype) in datasets_info.items():
        group_path, dataset_name = dataset_path.rsplit('/', 1)
        
        # Create groups if they don't exist
        if group_path not in main_group:
            current_path = ''
            for part in group_path.split('/'):
                current_path = current_path + '/' + part if current_path else part
                if current_path not in main_group:
                    main_group.create_group(current_path)
        
        # Create dataset with full size
        shape = list(other_dims)
        shape.insert(samples_dim, total_samples)
        main_group.create_dataset(group_path + '/' + dataset_name, 
                                 shape=tuple(shape), 
                                 dtype=dtype)

# Second pass: fill the datasets
print("Combining chunks into final file...")
sample_offset = 0

with h5py.File(output_file, 'r+') as out_f:
    main_group = out_f[f"pctrl{pctrl_value:.3f}"]
    
    # Process each chunk file
    for _, _, filename in tqdm(chunk_indices):
        file_path = os.path.join(input_dir, filename)
        with h5py.File(file_path, 'r') as f:
            # Get the number of samples in this chunk
            chunk_samples = f['haining']['singular_values']['A'].shape[1]
            
            # Copy singular values data
            for dataset_path, (samples_dim, _, _) in datasets_info.items():
                group_name, sv_group, dataset_name = dataset_path.split('/')
                
                # Source dataset in chunk
                src_dataset = f[group_name][sv_group][dataset_name]
                
                # Target dataset in output
                dst_dataset = main_group[group_name][sv_group][dataset_name]
                
                # Create slice object for the samples dimension
                slice_obj = [slice(None)] * len(dst_dataset.shape)
                slice_obj[samples_dim] = slice(sample_offset, sample_offset + chunk_samples)
                
                # Copy data
                dst_dataset[tuple(slice_obj)] = src_dataset[:]
            
            # Update offset for next chunk
            sample_offset += chunk_samples

print(f"Successfully combined {len(chunk_indices)} chunks into {output_file}")
print(f"Total samples: {total_samples}") 