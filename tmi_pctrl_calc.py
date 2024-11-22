from mpi4py import MPI
import numpy as np
import QCT as qct
import argparse
import os
import json
from metric_func import tripartite_mutual_information_tao

def compute_tmi_single(L, p_ctrl_values, p_proj, chunk_size):
    """
    Compute TMI for a single L value with a small chunk of samples
    
    Args:
        L: System size
        p_ctrl_values: Array of p_ctrl values to compute
        p_proj: Single p_proj value
        chunk_size: Number of samples in this chunk (small, e.g., 20)
    """
    num_time_steps = L**2*2
    results = {
        'L': L,
        'p_proj': p_proj,
        'p_ctrl': p_ctrl_values.tolist(),
        'tmi_samples': [],  # Store all samples for each p_ctrl
        'chunk_size': chunk_size
    }
    
    # For each p_ctrl, collect chunk_size samples
    for p_ctrl in p_ctrl_values:
        samples = []
        for _ in range(chunk_size):
            qct_tao = qct.QCT(L, p_ctrl, p_proj)
            for _ in range(num_time_steps):
                qct_tao.step_evolution()
            samples.append(tripartite_mutual_information_tao(qct_tao.state, L))
        results['tmi_samples'].append(samples)
    
    return results



def combine_results(output_dir, L, p_proj_values, total_samples=2000):
    """
    Combine chunks for proper ensemble averaging
    
    Args:
        output_dir: Directory containing chunk files
        L: System size
        p_proj_values: Array of p_proj values
        total_samples: Target total number of samples (e.g., 2000)
    """
    final_results = {}
    
    for p_proj in p_proj_values:
        key = f'pproj{p_proj:.3f}'
        final_results[key] = {
            'tmi': None,
            'tmi_std': None,
            'total_samples': 0
        }
        
        # Find all chunks for this p_proj
        chunk_files = [f for f in os.listdir(output_dir) 
                      if f.startswith(f'chunk_pproj{p_proj:.3f}_rank')]
        
        if not chunk_files:
            print(f"Warning: No chunks found for p_proj={p_proj}")
            continue
        
        # Read all samples from all chunks
        all_samples = []  # List of lists: [p_ctrl][sample]
        
        # Initialize with first chunk to get dimensions
        with open(os.path.join(output_dir, chunk_files[0]), 'r') as f:
            first_chunk = json.load(f)
            num_p_ctrl = len(first_chunk['tmi_samples'])
            all_samples = [[] for _ in range(num_p_ctrl)]
        
        # Collect all samples
        for chunk_file in chunk_files:
            with open(os.path.join(output_dir, chunk_file), 'r') as f:
                chunk_data = json.load(f)
                for p_ctrl_idx in range(num_p_ctrl):
                    all_samples[p_ctrl_idx].extend(chunk_data['tmi_samples'][p_ctrl_idx])
        
        # Compute ensemble averages and standard errors
        tmi_means = []
        tmi_stds = []
        n_samples = len(all_samples[0])  # Number of samples per p_ctrl
        
        for p_ctrl_samples in all_samples:
            tmi_means.append(np.mean(p_ctrl_samples))
            tmi_stds.append(np.std(p_ctrl_samples) / np.sqrt(n_samples))
        
        final_results[key]['tmi'] = tmi_means
        final_results[key]['tmi_std'] = tmi_stds
        final_results[key]['total_samples'] = n_samples
        
        print(f"Combined {n_samples} samples for p_proj={p_proj}")
        
        if n_samples < total_samples:
            print(f"Warning: Only got {n_samples} samples out of {total_samples} targeted")
    
    return final_results

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
    # p_proj_values = np.linspace(0.5, 1.0, 15)
    p_proj_values = [0.3]
    p_ctrl_values = np.linspace(0, 0.6, 21)
    chunks_needed = total_samples // chunk_size  # 20 chunks needed

    # Create all parameter combinations including chunk indices
    all_params = [(args.L, p_proj, chunk_idx) 
                 for p_proj in p_proj_values 
                 for chunk_idx in range(chunks_needed)]
    
    # Distribute work across ranks
    for param_idx in range(rank, len(all_params), size):
        L, p_proj, chunk_idx = all_params[param_idx]
        
        # Compute chunk
        chunk_result = compute_tmi_single(L, p_ctrl_values, p_proj, chunk_size)
        
        # Save chunk result with chunk index
        chunk_file = f'chunk_pproj{p_proj:.3f}_rank{rank}_chunk{chunk_idx}.json'
        with open(os.path.join(output_dir, chunk_file), 'w') as f:
            json.dump(chunk_result, f)

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