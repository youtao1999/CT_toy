'''
Compare the singular values of the toy model with different p_ctrl values between Tao's and Haining's version.
Uses a fine-grained scan around a critical point, similar to tmi_fine.py.
'''

import numpy as np
import QCT as qct
from haining_correct_functions import random_control
from metric_func import tripartite_mutual_information_tao
from QCT_util import Haar_state
import os
import json
import h5py
import argparse
from mpi4py import MPI

def compute_sv_comparison(L, p_scan_values, p_fixed, chunk_size, p_fixed_name='pproj'):
    """
    Compute singular values for both Tao's and Haining's versions with a small chunk of samples.
    """
    num_time_steps = L**2*2
    
    # For each p_scan value, collect chunk_size samples
    tao_samples = []
    haining_samples = []
    
    for p_scan in p_scan_values:
        tao_chunk = []
        haining_chunk = []
        
        for _ in range(chunk_size):
            # Set up parameters based on which one is fixed
            if p_fixed_name == 'pctrl':
                p_ctrl = p_fixed
                p_proj = p_scan
            else:  # p_fixed_name == 'pproj'
                p_ctrl = p_scan
                p_proj = p_fixed
            
            # Tao's version
            qct_tao = qct.QCT(L, p_ctrl, p_proj)
            for _ in range(num_time_steps):
                qct_tao.step_evolution()
            _, tao_singular_values = tripartite_mutual_information_tao(qct_tao.state, L, n=0, return_singular_values=True)
            tao_chunk.append({'singular_values': tao_singular_values})
            
            # Haining's version
            state = Haar_state(ensemble=1, rng=np.random.default_rng(), L=L).flatten()
            state_haining = state.copy()
            for _ in range(num_time_steps):
                state_haining, _ = random_control(state_haining, p_ctrl, p_proj, L)
            _, haining_singular_values = tripartite_mutual_information_tao(state_haining, L, n=0, return_singular_values=True)
            haining_chunk.append({'singular_values': haining_singular_values})
        
        tao_samples.append(tao_chunk)
        haining_samples.append(haining_chunk)
    
    # Convert singular values to numpy arrays for efficient storage
    tao_singular_values_dict = {
        'A': np.array([[s['singular_values']['A'] for s in chunk] for chunk in tao_samples]),
        'B': np.array([[s['singular_values']['B'] for s in chunk] for chunk in tao_samples]),
        'C': np.array([[s['singular_values']['C'] for s in chunk] for chunk in tao_samples]),
        'AB': np.array([[s['singular_values']['AB'] for s in chunk] for chunk in tao_samples]),
        'AC': np.array([[s['singular_values']['AC'] for s in chunk] for chunk in tao_samples]),
        'BC': np.array([[s['singular_values']['BC'] for s in chunk] for chunk in tao_samples]),
        'ABC': np.array([[s['singular_values']['ABC'] for s in chunk] for chunk in tao_samples])
    }
    
    haining_singular_values_dict = {
        'A': np.array([[s['singular_values']['A'] for s in chunk] for chunk in haining_samples]),
        'B': np.array([[s['singular_values']['B'] for s in chunk] for chunk in haining_samples]),
        'C': np.array([[s['singular_values']['C'] for s in chunk] for chunk in haining_samples]),
        'AB': np.array([[s['singular_values']['AB'] for s in chunk] for chunk in haining_samples]),
        'AC': np.array([[s['singular_values']['AC'] for s in chunk] for chunk in haining_samples]),
        'BC': np.array([[s['singular_values']['BC'] for s in chunk] for chunk in haining_samples]),
        'ABC': np.array([[s['singular_values']['ABC'] for s in chunk] for chunk in haining_samples])
    }
    
    # Return a dictionary with the results
    return {
        'L': L,
        p_fixed_name: p_fixed,
        'p_scan_values': p_scan_values,
        'chunk_size': chunk_size,
        'tao': {
            'singular_values': tao_singular_values_dict
        },
        'haining': {
            'singular_values': haining_singular_values_dict
        }
    }

def save_chunk(result, output_dir, rank, chunk_idx, p_fixed_name):
    """Save chunk results to HDF5 file"""
    chunk_file = f'chunk_{p_fixed_name}{result[p_fixed_name]:.3f}_rank{rank}_chunk{chunk_idx}.h5'
    chunk_path = os.path.join(output_dir, chunk_file)
    
    # Initialize p_scan_name
    p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
    
    with h5py.File(chunk_path, 'w') as f:
        # Store metadata
        f.attrs['L'] = result['L']
        f.attrs[p_fixed_name] = result[p_fixed_name]
        f.attrs['chunk_size'] = result['chunk_size']
        f.create_dataset(p_fixed_name, data=result[p_fixed_name])
        
        # Add p_scan_values dataset
        f.create_dataset(p_scan_name, data=result['p_scan_values'])
        
        # Create groups for Tao and Haining results
        tao_group = f.create_group('tao')
        tao_sv_group = tao_group.create_group('singular_values')
        for key, value in result['tao']['singular_values'].items():
            tao_sv_group.create_dataset(key, data=value)
        
        haining_group = f.create_group('haining')
        haining_sv_group = haining_group.create_group('singular_values')
        for key, value in result['haining']['singular_values'].items():
            haining_sv_group.create_dataset(key, data=value)

def combine_results(output_dir, L, p_fixed, p_fixed_name):
    """Combine chunks into a single HDF5 file"""
    os.makedirs(output_dir, exist_ok=True)
    final_file = os.path.join(output_dir, f'final_results_L{L}.h5')
    
    with h5py.File(final_file, 'w') as f:
        f.attrs['L'] = L
        
        # Find all chunks for this p_fixed
        chunk_files = [f for f in os.listdir(output_dir) 
                       if f.startswith(f'chunk_{p_fixed_name}{p_fixed:.3f}_rank') and f.endswith('.h5')]
        
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {output_dir} for {p_fixed_name}={p_fixed}")

        p_fixed_group = f.create_group(f'{p_fixed_name}{p_fixed:.3f}')
        
        # Initialize string of p_scan name
        p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
        
        # Read first chunk to get structure
        with h5py.File(os.path.join(output_dir, chunk_files[0]), 'r') as chunk_f:
            p_scan_values = chunk_f[p_scan_name][:]
            p_fixed_group.create_dataset(p_scan_name, data=p_scan_values)
            
            # Initialize arrays for combined data
            tao_sv_samples = {key: [] for key in ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']}
            haining_sv_samples = {key: [] for key in ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']}
            
            # Combine all chunks
            for chunk_file in chunk_files:
                with h5py.File(os.path.join(output_dir, chunk_file), 'r') as chunk_f:
                    for key in tao_sv_samples:
                        tao_sv_samples[key].append(chunk_f['tao']['singular_values'][key][:])
                        haining_sv_samples[key].append(chunk_f['haining']['singular_values'][key][:])
            
            # Store combined data
            tao_group = p_fixed_group.create_group('tao')
            tao_sv_group = tao_group.create_group('singular_values')
            for key, value in tao_sv_samples.items():
                combined_data = np.concatenate(value, axis=1)
                tao_sv_group.create_dataset(key, data=combined_data)
            
            haining_group = p_fixed_group.create_group('haining')
            haining_sv_group = haining_group.create_group('singular_values')
            for key, value in haining_sv_samples.items():
                combined_data = np.concatenate(value, axis=1)
                haining_sv_group.create_dataset(key, data=combined_data)
            
        p_fixed_group.attrs['total_samples'] = combined_data.shape[1]
        
        # Create a summary dataset for easy reference
        summary = {
            p_scan_name: p_scan_values,
            'total_samples': combined_data.shape[1]
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Compare singular values between Tao and Haining versions')
    parser.add_argument('--L', type=int, required=True, help='System size L')
    parser.add_argument('--p_fixed_name', type=str, required=True, choices=['pproj', 'pctrl'], 
                        help='Fixed probability name (pproj or pctrl)')
    parser.add_argument('--p_fixed', type=float, required=True, help='Fixed probability value')
    parser.add_argument('--ncpu', type=int, required=True, help='Number of CPUs/MPI processes')
    parser.add_argument('--p_c', type=float, required=True, help='Critical point p_c')
    parser.add_argument('--delta_p', type=float, required=False, default=0.05, 
                        help='Range of p_scan around p_c')
    parser.add_argument('--num_p_scan', type=int, required=False, default=20, 
                        help='Number of p_scan values linearly spaced between p_c-delta_p and p_c+delta_p')
    parser.add_argument('--total_samples', type=int, required=False, default=2000, 
                        help='Total number of samples')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Verify MPI size matches requested CPUs
    if size != args.ncpu:
        if rank == 0:
            raise ValueError(f"MPI size ({size}) does not match requested number of CPUs ({args.ncpu})")

    # Fixed parameters
    chunk_size = args.total_samples // args.ncpu

    # Verify that chunks divide evenly
    if args.total_samples % args.ncpu != 0:
        if rank == 0:
            raise ValueError(
                f"Number of CPUs ({args.ncpu}) must divide total_samples ({args.total_samples}) evenly.\n"
                f"Valid CPU counts are: {[i for i in range(1, args.total_samples+1) if args.total_samples % i == 0]}"
            )

    if rank == 0:
        print(f"Running with {args.ncpu} CPUs")
        print(f"Each CPU will process chunk_size = {chunk_size} samples")
        print(f"Total samples = {args.total_samples}")
    
    # Define output directory
    output_dir = f'sv_comparison_L{args.L}_{args.p_fixed_name}{args.p_fixed:.3f}_pc{args.p_c:.3f}'
    
    # Create directory with all necessary parent directories
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Wait for directory creation to complete before proceeding
    comm.Barrier()

    # Generate p_scan values
    p_scan_values = np.linspace(args.p_c - args.delta_p, args.p_c + args.delta_p, args.num_p_scan)

    # Create all parameter combinations including chunk indices
    all_params = [(args.L, args.p_fixed, chunk_idx) 
                 for chunk_idx in range(args.ncpu)]
    
    # Distribute work across ranks
    for param_idx in range(rank, len(all_params), size):
        L, p_fixed, chunk_idx = all_params[param_idx]
        
        print(f"Rank {rank}: Starting calculation for L={L}, {args.p_fixed_name}={p_fixed}, chunk={chunk_idx}")
        
        try:
            # Compute chunk
            chunk_result = compute_sv_comparison(L, p_scan_values, p_fixed, chunk_size, p_fixed_name=args.p_fixed_name)
            save_chunk(chunk_result, output_dir, rank, chunk_idx, args.p_fixed_name)
            
            print(f"Rank {rank}: Successfully wrote chunk {chunk_idx}")
            
        except Exception as e:
            print(f"Rank {rank}: Error in chunk {chunk_idx}: {str(e)}")
            raise  # Re-raise the exception to see the full traceback

    comm.Barrier()

    # Only rank 0 combines results and plots
    if rank == 0:
        summary = combine_results(output_dir, args.L, args.p_fixed, args.p_fixed_name)
        
        print(f"Comparison completed. Results saved in '{output_dir}' folder.")

if __name__ == "__main__":
    # main()

    # mpirun -np 4 python sv_compare_fine.py --L 8 --p_fixed_name pctrl --p_fixed 0.4 --ncpu 4 --p_c 0.6 --delta_p 0.1 --num_p_scan 20 --total_samples 20

    output_dir = f'sv_comparison_L20_pctrl0.400_pc0.800'
    p_fixed = 0.400
    final_results = combine_results(output_dir, 20, p_fixed, 'pctrl')