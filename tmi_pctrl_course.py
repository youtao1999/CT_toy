from mpi4py import MPI
import numpy as np
import QCT as qct
import argparse
import os
import json
from metric_func import tripartite_mutual_information_tao
import h5py

def compute_tmi_single(L, p_ctrl_values, p_proj, chunk_size):
    """
    Compute and store singular values for a single L value with a small chunk of samples
    """
    num_time_steps = L**2*2
    
    # For each p_ctrl, collect chunk_size samples
    samples = []
    for p_ctrl in p_ctrl_values:
        chunk_samples = []
        for _ in range(chunk_size):
            qct_tao = qct.QCT(L, p_ctrl, p_proj)
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
    
    return {
        'L': L,
        'p_proj': p_proj,
        'p_ctrl': p_ctrl_values,
        'singular_values': singular_values_dict,
        'chunk_size': chunk_size
    }

def save_chunk(result, output_dir, rank, chunk_idx):
    """Save chunk results to HDF5 file"""
    chunk_file = f'chunk_pproj{result["p_proj"]:.3f}_rank{rank}_chunk{chunk_idx}.h5'
    chunk_path = os.path.join(output_dir, chunk_file)
    
    with h5py.File(chunk_path, 'w') as f:
        # Store metadata
        f.attrs['L'] = result['L']
        f.attrs['p_proj'] = result['p_proj']
        f.attrs['chunk_size'] = result['chunk_size']
        f.create_dataset('p_ctrl', data=result['p_ctrl'])
        
        # Create group for singular values
        sv_group = f.create_group('singular_values')
        for key, value in result['singular_values'].items():
            sv_group.create_dataset(key, data=value)

def combine_results(output_dir, L, p_proj_values, total_samples=2000):
    """Combine chunks into a single HDF5 file"""
    final_file = os.path.join(output_dir, f'final_results_L{L}.h5')
    
    with h5py.File(final_file, 'w') as f:
        f.attrs['L'] = L
        
        for p_proj in p_proj_values:
            # Find all chunks for this p_proj
            chunk_files = [f for f in os.listdir(output_dir) 
                         if f.startswith(f'chunk_pproj{p_proj:.3f}_rank') and f.endswith('.h5')]
            
            if not chunk_files:
                print(f"Warning: No chunks found for p_proj={p_proj}")
                continue
            
            # Create group for this p_proj
            p_proj_group = f.create_group(f'pproj{p_proj:.3f}')
            
            # Read first chunk to get structure
            with h5py.File(os.path.join(output_dir, chunk_files[0]), 'r') as chunk_f:
                p_ctrl_values = chunk_f['p_ctrl'][:]
                p_proj_group.create_dataset('p_ctrl', data=p_ctrl_values)
            
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
            sv_group = p_proj_group.create_group('singular_values')
            for key, value in all_samples.items():
                combined_data = np.concatenate(value, axis=1)  # Combine along sample dimension
                sv_group.create_dataset(key, data=combined_data)
            
            p_proj_group.attrs['total_samples'] = combined_data.shape[1]

def main():
    parser = argparse.ArgumentParser(description='Compute TMI for a specific system size L')
    parser.add_argument('--L', type=int, required=True, help='System size L')
    parser.add_argument('--ncpu', type=int, required=True, help='Number of CPUs/MPI processes')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Verify MPI size matches requested CPUs
    if size != args.ncpu:
        if rank == 0:
            raise ValueError(f"MPI size ({size}) does not match requested number of CPUs ({args.ncpu})")

    # Fixed parameters
    total_samples = 2000
    chunk_size = total_samples // args.ncpu

    # Verify that chunks divide evenly
    if total_samples % args.ncpu != 0:
        if rank == 0:
            raise ValueError(
                f"Number of CPUs ({args.ncpu}) must divide total_samples ({total_samples}) evenly.\n"
                f"Valid CPU counts are: {[i for i in range(1, total_samples+1) if total_samples % i == 0]}"
            )

    if rank == 0:
        print(f"Running with {args.ncpu} CPUs")
        print(f"Each CPU will process chunk_size = {chunk_size} samples")
        print(f"Total samples = {total_samples}")

    # Define output directory for all ranks
    output_dir = f'tmi_pctrl_results_L{args.L}'
    
    # Only rank 0 creates the directory
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    # Wait for directory creation
    comm.Barrier()

    # Fixed parameters
    p_proj_values = np.linspace(0.5, 1.0, 2)
    p_ctrl_values = np.linspace(0, 0.6, 2)

    # Create all parameter combinations including chunk indices
    all_params = [(args.L, p_proj, chunk_idx) 
                 for p_proj in p_proj_values 
                 for chunk_idx in range(args.ncpu)]
    
    # Distribute work across ranks
    for param_idx in range(rank, len(all_params), size):
        L, p_proj, chunk_idx = all_params[param_idx]
        
        print(f"Rank {rank}: Starting calculation for L={L}, p_proj={p_proj}, chunk={chunk_idx}")
        
        try:
            # Compute chunk
            chunk_result = compute_tmi_single(L, p_ctrl_values, p_proj, chunk_size)
            save_chunk(chunk_result, output_dir, rank, chunk_idx)
            
            print(f"Rank {rank}: Successfully wrote chunk {chunk_idx}")
            
        except Exception as e:
            print(f"Rank {rank}: Error in chunk {chunk_idx}: {str(e)}")
            raise  # Re-raise the exception to see the full traceback

    comm.Barrier()

    # Only rank 0 combines results
    if rank == 0:
        final_results = combine_results(output_dir, args.L, p_proj_values, total_samples)
        
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
    # output_dir = f'tmi0_pctrl_results_L20'
    # p_proj_values = np.linspace(0.5, 1.0, 15)[:14]
    # final_results = combine_results(output_dir, 20, p_proj_values)

    # # Save final results
    # with open(os.path.join(output_dir, f'final_results_L20.json'), 'w') as f:
    #     json.dump(final_results, f)