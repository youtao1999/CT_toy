from mpi4py import MPI
import numpy as np
import QCT as qct
import argparse
import os
import json
from metric_func import tripartite_mutual_information_tao
import h5py

'''
As opposed to tmi_pctrl_course.py, this script computes TMI for a fine grid of p_ctrl values for a fixed p_proj, a range of L values,
a fixed number of CPUs, and a fixed range of p_ctrl around the critical point.
'''

def compute_tmi_single(L, p_scan_values, p_fixed, chunk_size, p_fixed_name = 'pproj'):
    """
    Compute and store singular values for a single L value with a small chunk of samples.
    Can be used for either p_ctrl or p_proj. 
    Usage: if p_fixed_name = 'pproj', then p_scan is a list of p_ctrl values.
    If p_fixed_name = 'pctrl', then p_scan is a list of p_proj values.
    p_fixed_name denotes the probability metric that is fixed while p_scan is scanned over.
    """
    num_time_steps = L**2*2
    
    # For each p_ctrl, collect chunk_size samples
    samples = []
    for p_scan in p_scan_values:
        chunk_samples = []
        for _ in range(chunk_size):
            if p_fixed_name == 'pctrl':
                qct_tao = qct.QCT(L, p_ctrl = p_fixed, p_proj = p_scan)
            elif p_fixed_name == 'pproj':
                qct_tao = qct.QCT(L, p_ctrl = p_scan, p_proj = p_fixed)
            for _ in range(num_time_steps):
                qct_tao.step_evolution()
            _, singular_values = tripartite_mutual_information_tao(qct_tao.state, L, n=0, return_singular_values=True)
            chunk_samples.append(singular_values)
        samples.append(chunk_samples)
    
    # Convert to numpy arrays for efficient storage
    singular_values_dict = {
        'A': np.array([[s['A'] for s in chunk] for chunk in samples]),
        'B': np.array([[s['B'] for s in chunk] for chunk in samples]),
        'C': np.array([[s['C'] for s in chunk] for chunk in samples]),
        'AB': np.array([[s['AB'] for s in chunk] for chunk in samples]),
        'AC': np.array([[s['AC'] for s in chunk] for chunk in samples]),
        'BC': np.array([[s['BC'] for s in chunk] for chunk in samples]),
        'ABC': np.array([[s['ABC'] for s in chunk] for chunk in samples])
    }
    
    # Return a dictionary with the results
    return {
        'L': L,
        p_fixed_name: p_fixed,
        'singular_values': singular_values_dict,
        'chunk_size': chunk_size,
        'p_scan_values': p_scan_values
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
        
        # Create group for singular values
        sv_group = f.create_group('singular_values')
        for key, value in result['singular_values'].items():
            sv_group.create_dataset(key, data=value)

def combine_results(output_dir, L, p_fixed, p_fixed_name):
    """Combine chunks into a single HDF5 file"""
    final_file = os.path.join(output_dir, f'final_results_L{L}.h5')
    
    with h5py.File(final_file, 'w') as f:
        f.attrs['L'] = L
        
        # Find all chunks for this p_proj
        chunk_files = [f for f in os.listdir(output_dir) 
                         if f.startswith(f'chunk_{p_fixed_name}{p_fixed:.3f}_rank') and f.endswith('.h5')]

        p_fixed_group = f.create_group(f'{p_fixed_name}{p_fixed:.3f}')
        
        # Initialize string of p_scan name
        p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
        # Read first chunk to get structure
        with h5py.File(os.path.join(output_dir, chunk_files[0]), 'r') as chunk_f:
            p_scan_values = chunk_f[p_scan_name][:]
            p_fixed_group.create_dataset(p_scan_name, data=p_scan_values)
            
            # Initialize arrays for combined data
            all_samples = {
                key: [] for key in ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
            }
            
            # Combine all chunks
            for chunk_file in chunk_files:
                with h5py.File(os.path.join(output_dir, chunk_file), 'r') as chunk_f:
                    for key in all_samples:
                        all_samples[key].append(chunk_f['singular_values'][key][:])
            
            # Store combined data
            sv_group = p_fixed_group.create_group('singular_values')
            for key, value in all_samples.items():
                combined_data = np.concatenate(value, axis=1)  # Combine along sample dimension
                sv_group.create_dataset(key, data=combined_data)
            
        p_fixed_group.attrs['total_samples'] = combined_data.shape[1]

def main():
    parser = argparse.ArgumentParser(description='Compute TMI for a specific system size L')
    parser.add_argument('--L', type=int, required=True, help='System size L')
    parser.add_argument('--p_fixed_name', type=str, required=True, help='Fixed probability name')
    parser.add_argument('--p_fixed', type=float, required=True, help='Fixed probability p_fixed')
    parser.add_argument('--ncpu', type=int, required=True, help='Number of CPUs/MPI processes')
    parser.add_argument('--p_c', type=float, required=True, help='Critical point p_c')
    parser.add_argument('--delta_p', type=float, required=False, default=0.05, help='Range of p_scan around p_c')
    parser.add_argument('--num_p_scan', type=int, required=False, default=20, help='Number of p_scan values linearly spaced between p_c-delta_p and p_c+delta_p')
    parser.add_argument('--total_samples', type=int, required=False, default=2000, help='Total number of samples')
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
    output_dir = f'tmi_fine_L{args.L}_{args.p_fixed_name}{args.p_fixed:.3f}_pc{args.p_c}'
    
    # Create directory with all necessary parent directories
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Wait for directory creation to complete before proceeding
    comm.Barrier()

    # Fixed parameters
    p_scan_values = np.linspace(args.p_c - args.delta_p, args.p_c + args.delta_p, args.num_p_scan)

    # Create all parameter combinations including chunk indices
    all_params = [(args.L, args.p_fixed, chunk_idx) 
                 for chunk_idx in range(args.ncpu)]
    
    # Distribute work across ranks
    for param_idx in range(rank, len(all_params), size):
        L, p_proj, chunk_idx = all_params[param_idx]
        
        print(f"Rank {rank}: Starting calculation for L={L}, p_proj={p_proj}, chunk={chunk_idx}")
        
        try:
            # Compute chunk
            chunk_result = compute_tmi_single(L, p_scan_values, args.p_fixed, chunk_size, p_fixed_name = args.p_fixed_name)
            save_chunk(chunk_result, output_dir, rank, chunk_idx, args.p_fixed_name)
            
            print(f"Rank {rank}: Successfully wrote chunk {chunk_idx}")
            
        except Exception as e:
            print(f"Rank {rank}: Error in chunk {chunk_idx}: {str(e)}")
            raise  # Re-raise the exception to see the full traceback

    comm.Barrier()

    # Only rank 0 combines results
    if rank == 0:
        final_results = combine_results(output_dir, args.L, args.p_fixed, args.p_fixed_name)
        
        # Save final results
        with open(os.path.join(output_dir, f'final_results_L{args.L}.json'), 'w') as f:
            json.dump(final_results, f)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    main()
    # output_dir = f'tmi_fine_L20_pproj0.536_pc0.49'
    # p_fixed = 0.536
    # final_results = combine_results(output_dir, 20, p_fixed, 'pproj')

    # # Save final results
    # with open(os.path.join(output_dir, f'final_results_L20_pproj{p_fixed}.json'), 'w') as f:
    #     json.dump(final_results, f)
