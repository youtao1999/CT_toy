'''
Compute singular values for a fine grid of p_scan values for a fixed p_fixed.
Can optionally compare Tao's and Haining's implementations.

This script combines the functionality of sv_fine.py and sv_compare_fine.py.
'''

import numpy as np
import QCT as qct
from haining_correct_functions import random_control
from metric_func import tripartite_mutual_information
from QCT_util import Haar_state
import os
import h5py
import argparse
from mpi4py import MPI
from tqdm import tqdm


class SingularValueComputer:
    """Class to compute and manage singular values computation"""
    
    def __init__(self, L, p_fixed, p_fixed_name, p_scan_values, chunk_size, comparison):
        """
        Initialize the singular value computer
        
        Parameters:
        -----------
        L : int
            System size
        p_fixed : float
            Fixed probability value
        p_fixed_name : str
            Name of fixed parameter ('p_proj' or 'p_ctrl')
        p_scan_values : array-like
            Values to scan over
        chunk_size : int
            Number of samples per p_scan value
        comparison : bool
            Whether to compare with Haining's implementation
        """
        self.L = L
        self.p_fixed = p_fixed
        self.p_fixed_name = p_fixed_name
        self.p_scan_values = p_scan_values
        self.chunk_size = chunk_size
        self.comparison = comparison
        self.num_time_steps = L**2*2
        
    def compute_chunk(self):
        """Compute singular values for a chunk of samples"""
        if self.comparison:
            return self._compute_comparison_chunk()
        else:
            return self._compute_single_chunk()
    
    def _compute_single_chunk(self):
        """Compute singular values using only Tao's implementation"""
        samples = []
        
        for p_scan in self.p_scan_values:
            chunk_samples = []
            for _ in range(self.chunk_size):
                # Set parameters based on which one is fixed
                if self.p_fixed_name == 'p_ctrl':
                    p_ctrl = self.p_fixed
                    p_proj = p_scan
                else:  # p_fixed_name == 'p_proj'
                    p_ctrl = p_scan
                    p_proj = self.p_fixed
                
                # Run Tao's implementation
                qct_tao = qct.QCT(self.L, p_ctrl, p_proj)
                for _ in tqdm(range(self.num_time_steps)):
                    qct_tao.step_evolution()
                _, singular_values = tripartite_mutual_information(
                    qct_tao.state, self.L, n=0, return_singular_values=True)
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
            'L': self.L,
            self.p_fixed_name: self.p_fixed,
            'singular_values': singular_values_dict,
            'chunk_size': self.chunk_size,
            'p_scan_values': self.p_scan_values
        }
    
    def _compute_comparison_chunk(self):
        """Compute singular values using both Tao's and Haining's implementations"""
        tao_samples = []
        haining_samples = []
        
        for p_scan in self.p_scan_values:
            tao_chunk = []
            haining_chunk = []
            
            for _ in range(self.chunk_size):
                # Set parameters based on which one is fixed
                if self.p_fixed_name == 'p_ctrl':
                    p_ctrl = self.p_fixed
                    p_proj = p_scan
                else:  # p_fixed_name == 'pproj'
                    p_ctrl = p_scan
                    p_proj = self.p_fixed
                
                # Tao's version
                qct_tao = qct.QCT(self.L, p_ctrl, p_proj)
                for _ in range(self.num_time_steps):
                    qct_tao.step_evolution()
                _, tao_singular_values = tripartite_mutual_information(
                    qct_tao.state, self.L, n=0, return_singular_values=True)
                tao_chunk.append(tao_singular_values)
                
                # Haining's version
                state = Haar_state(ensemble=1, rng=np.random.default_rng(), L=self.L).flatten()
                state_haining = state.copy()
                for _ in range(self.num_time_steps):
                    state_haining, _ = random_control(state_haining, p_ctrl, p_proj, self.L)
                _, haining_singular_values = tripartite_mutual_information(
                    state_haining, self.L, n=0, return_singular_values=True)
                haining_chunk.append(haining_singular_values)
            
            tao_samples.append(tao_chunk)
            haining_samples.append(haining_chunk)
        
        # Convert singular values to numpy arrays for efficient storage
        tao_singular_values_dict = {
            'A': np.array([[s['A'] for s in chunk] for chunk in tao_samples]),
            'B': np.array([[s['B'] for s in chunk] for chunk in tao_samples]),
            'C': np.array([[s['C'] for s in chunk] for chunk in tao_samples]),
            'AB': np.array([[s['AB'] for s in chunk] for chunk in tao_samples]),
            'AC': np.array([[s['AC'] for s in chunk] for chunk in tao_samples]),
            'BC': np.array([[s['BC'] for s in chunk] for chunk in tao_samples]),
            'ABC': np.array([[s['ABC'] for s in chunk] for chunk in tao_samples])
        }
        
        haining_singular_values_dict = {
            'A': np.array([[s['A'] for s in chunk] for chunk in haining_samples]),
            'B': np.array([[s['B'] for s in chunk] for chunk in haining_samples]),
            'C': np.array([[s['C'] for s in chunk] for chunk in haining_samples]),
            'AB': np.array([[s['AB'] for s in chunk] for chunk in haining_samples]),
            'AC': np.array([[s['AC'] for s in chunk] for chunk in haining_samples]),
            'BC': np.array([[s['BC'] for s in chunk] for chunk in haining_samples]),
            'ABC': np.array([[s['ABC'] for s in chunk] for chunk in haining_samples])
        }
        
        # Return a dictionary with the results
        return {
            'L': self.L,
            self.p_fixed_name: self.p_fixed,
            'p_scan_values': self.p_scan_values,
            'chunk_size': self.chunk_size,
            'tao': {
                'singular_values': tao_singular_values_dict
            },
            'haining': {
                'singular_values': haining_singular_values_dict
            }
        }


class DataManager:
    """Class to handle data saving and combining"""
    
    def __init__(self, output_dir, comparison):
        """
        Initialize the data manager
        
        Parameters:
        -----------
        output_dir : str
            Output directory
        comparison : bool
            Whether this is a comparison run
        """
        self.output_dir = output_dir
        self.comparison = comparison
        os.makedirs(output_dir, exist_ok=True)
    
    def save_chunk(self, result, rank, chunk_idx):
        """
        Save chunk results to HDF5 file
        
        Parameters:
        -----------
        result : dict
            Results dictionary
        rank : int
            MPI rank
        chunk_idx : int
            Chunk index
        """
        p_fixed_name = 'p_proj' if 'p_proj' in result else 'p_ctrl'
        chunk_file = f'chunk_{p_fixed_name}{result[p_fixed_name]:.3f}_rank{rank}_chunk{chunk_idx}.h5'
        chunk_path = os.path.join(self.output_dir, chunk_file)
        
        # Initialize p_scan_name
        p_scan_name = 'p_ctrl' if p_fixed_name == 'p_proj' else 'p_proj'
        
        with h5py.File(chunk_path, 'w') as f:
            # Store metadata
            f.attrs['L'] = result['L']
            f.attrs[p_fixed_name] = result[p_fixed_name]
            f.attrs['chunk_size'] = result['chunk_size']
            f.create_dataset(p_fixed_name, data=result[p_fixed_name])
            
            # Add p_scan_values dataset
            f.create_dataset(p_scan_name, data=result['p_scan_values'])
            
            if self.comparison:
                # Create groups for Tao and Haining results
                tao_group = f.create_group('tao')
                tao_sv_group = tao_group.create_group('singular_values')
                for key, value in result['tao']['singular_values'].items():
                    tao_sv_group.create_dataset(key, data=value)
                
                haining_group = f.create_group('haining')
                haining_sv_group = haining_group.create_group('singular_values')
                for key, value in result['haining']['singular_values'].items():
                    haining_sv_group.create_dataset(key, data=value)
            else:
                # Create group for singular values (Tao's implementation only)
                sv_group = f.create_group('singular_values')
                for key, value in result['singular_values'].items():
                    sv_group.create_dataset(key, data=value)
    
    def combine_results(self, L, p_fixed, p_fixed_name):
        """
        Combine chunks into a single HDF5 file
        
        Parameters:
        -----------
        L : int
            System size
        p_fixed : float
            Fixed parameter value
        p_fixed_name : str
            Name of fixed parameter
        """
        final_file = os.path.join(self.output_dir, f'final_results_L{L}.h5')
        
        with h5py.File(final_file, 'w') as f:
            f.attrs['L'] = L
            
            # Find all chunks for this p_fixed
            chunk_files = [f for f in os.listdir(self.output_dir) 
                          if f.startswith(f'chunk_{p_fixed_name}{p_fixed:.3f}_rank') and f.endswith('.h5')]
            
            if not chunk_files:
                raise FileNotFoundError(f"No chunk files found in {self.output_dir} for {p_fixed_name}={p_fixed}")

            p_fixed_group = f.create_group(f'{p_fixed_name}{p_fixed:.3f}')
            
            # Initialize string of p_scan name
            p_scan_name = 'p_ctrl' if p_fixed_name == 'p_proj' else 'p_proj'
            
            # Read first chunk to get structure and determine mode
            with h5py.File(os.path.join(self.output_dir, chunk_files[0]), 'r') as chunk_f:
                p_scan_values = chunk_f[p_scan_name][:]
                p_fixed_group.create_dataset(p_scan_name, data=p_scan_values)
                
                # Detect if this is a comparison run by checking for 'tao' group
                is_comparison = 'tao' in chunk_f
                
                # Initialize data structures based on detected mode
                if is_comparison:
                    # For comparison mode, we need two sets of arrays
                    implementations = ['tao', 'haining']
                else:
                    # For single mode, we just need one set
                    implementations = ['single']
                
                # Initialize arrays for all implementations
                sv_keys = ['A', 'B', 'C', 'AB', 'AC', 'BC', 'ABC']
                all_samples = {impl: {key: [] for key in sv_keys} for impl in implementations}
                
                # Combine all chunks
                for chunk_file in chunk_files:
                    with h5py.File(os.path.join(self.output_dir, chunk_file), 'r') as chunk_f:
                        if self.comparison:
                            # Read both implementations
                            for key in sv_keys:
                                all_samples['tao'][key].append(chunk_f['tao']['singular_values'][key][:])
                                all_samples['haining'][key].append(chunk_f['haining']['singular_values'][key][:])
                        else:
                            # Read single implementation
                            for key in sv_keys:
                                all_samples['single'][key].append(chunk_f['singular_values'][key][:])
                
                # Store combined data
                if self.comparison:
                    # Create groups for both implementations
                    for impl in implementations:
                        impl_group = p_fixed_group.create_group(impl)
                        sv_group = impl_group.create_group('singular_values')
                        for key in sv_keys:
                            combined_data = np.concatenate(all_samples[impl][key], axis=1)
                            sv_group.create_dataset(key, data=combined_data)
                else:
                    # Create group for single implementation
                    sv_group = p_fixed_group.create_group('singular_values')
                    for key in sv_keys:
                        combined_data = np.concatenate(all_samples['single'][key], axis=1)
                        sv_group.create_dataset(key, data=combined_data)
                
                # Store total samples
                p_fixed_group.attrs['total_samples'] = combined_data.shape[1]


class MPIManager:
    """Class to handle MPI operations and coordinate the computation"""
    
    def __init__(self, args):
        """
        Initialize the MPI manager
        
        Parameters:
        -----------
        args : argparse.Namespace
            Command line arguments
        """
        self.args = args
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Verify MPI size matches requested CPUs
        if self.size != args.ncpu:
            if self.rank == 0:
                raise ValueError(f"MPI size ({self.size}) does not match requested number of CPUs ({args.ncpu})")
        
        # Calculate chunk size
        self.chunk_size = args.total_samples // args.ncpu
        
        # Verify that chunks divide evenly
        if args.total_samples % args.ncpu != 0:
            if self.rank == 0:
                raise ValueError(
                    f"Number of CPUs ({args.ncpu}) must divide total_samples ({args.total_samples}) evenly.\n"
                    f"Valid CPU counts are: {[i for i in range(1, args.total_samples+1) if args.total_samples % i == 0]}"
                )
        
        # Parse p_range
        self.p_scan_values = self._parse_p_range(args.p_range)
        
        # Define output directory - use min and max of p_range for naming
        p_min = min(self.p_scan_values)
        p_max = max(self.p_scan_values)
        
        # Set output directory based on comparison flag
        if args.comparison:
            self.output_dir = f"/scratch/ty296/sv_comparison_L{args.L}_{args.p_fixed_name}{args.p_fixed:.3f}_p{p_min:.3f}-{p_max:.3f}"
        else:
            self.output_dir = f'/scratch/ty296/sv_L{args.L}_{args.p_fixed_name}{args.p_fixed:.3f}_p{p_min:.3f}-{p_max:.3f}'
        
        # Create data manager
        self.data_manager = DataManager(self.output_dir, comparison=args.comparison)
    
    def _parse_p_range(self, p_range_str):
        """
        Parse p_range string into numpy array
        
        Supports formats:
        - Comma-separated list: "0.1,0.2,0.3,0.4"
        - Linspace format: "start:stop:num" (like "0.1:0.5:5" for 5 points from 0.1 to 0.5)
        - Single value: "0.5" (will be converted to [0.5])
        
        Parameters:
        -----------
        p_range_str : str
            String specification of p_range
            
        Returns:
        --------
        numpy.ndarray
            Array of p values to scan
        """
        if ':' in p_range_str:
            # Linspace format: start:stop:num
            parts = p_range_str.split(':')
            if len(parts) != 3:
                raise ValueError("Linspace format must be 'start:stop:num'")
            start = float(parts[0])
            stop = float(parts[1])
            num = int(parts[2])
            return np.linspace(start, stop, num)
        elif ',' in p_range_str:
            # Comma-separated list
            return np.array([float(p) for p in p_range_str.split(',')])
        else:
            # Single value
            return np.array([float(p_range_str)])
    
    def run(self):
        """Run the computation"""
        if self.rank == 0:
            print(f"Running with {self.args.ncpu} CPUs")
            print(f"Each CPU will process chunk_size = {self.chunk_size} samples")
            print(f"Total samples = {self.args.total_samples}")
            if self.args.comparison:
                print("Comparing Tao's and Haining's implementations")
        
        # Create directory with all necessary parent directories
        if self.rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Wait for directory creation to complete before proceeding
        self.comm.Barrier()
        
        # Create all parameter combinations including chunk indices
        all_params = [(self.args.L, self.args.p_fixed, chunk_idx) 
                     for chunk_idx in range(self.args.ncpu)]
        
        # Distribute work across ranks
        for param_idx in range(self.rank, len(all_params), self.size):
            L, p_fixed, chunk_idx = all_params[param_idx]
            
            print(f"Rank {self.rank}: Starting calculation for L={L}, {self.args.p_fixed_name}={p_fixed}, chunk={chunk_idx}")
            
            try:
                # Create computer and compute chunk
                computer = SingularValueComputer(
                    L, p_fixed, self.args.p_fixed_name, self.p_scan_values, 
                    self.chunk_size, comparison=self.args.comparison
                )
                chunk_result = computer.compute_chunk()
                
                # Save chunk
                self.data_manager.save_chunk(chunk_result, self.rank, chunk_idx)
                
                print(f"Rank {self.rank}: Successfully wrote chunk {chunk_idx}")
                
            except Exception as e:
                print(f"Rank {self.rank}: Error in chunk {chunk_idx}: {str(e)}")
                raise  # Re-raise the exception to see the full traceback
        
        self.comm.Barrier()
        
        # Only rank 0 combines results
        if self.rank == 0:
            self.data_manager.combine_results(self.args.L, self.args.p_fixed, self.args.p_fixed_name)
            print(f"Calculation completed. Results saved in '{self.output_dir}' folder.")


def main():
    parser = argparse.ArgumentParser(description='Compute singular values for a specific system size L')
    parser.add_argument('--L', type=int, required=True, help='System size L')
    parser.add_argument('--p_fixed_name', type=str, required=True, choices=['p_proj', 'p_ctrl'], 
                        help='Fixed probability name (p_proj or p_ctrl)')
    parser.add_argument('--p_fixed', type=float, required=True, help='Fixed probability value')
    parser.add_argument('--ncpu', type=int, required=True, help='Number of CPUs/MPI processes')
    parser.add_argument('--p_range', type=str, required=True, 
                        help='Range of p values to scan. Formats: "0.1,0.2,0.3" (comma-separated list) or "0.1:0.5:5" (start:stop:num for linspace)')
    parser.add_argument('--total_samples', type=int, required=False, default=2000, 
                        help='Total number of samples')
    parser.add_argument('--comparison', action='store_true', 
                        help='Compare Tao\'s and Haining\'s implementations')
    args = parser.parse_args()
    
    # Create and run MPI manager
    mpi_manager = MPIManager(args)
    mpi_manager.run()


if __name__ == "__main__":
    main()
    # p_scan_values = np.array([0.4])  # or use np.linspace(0.0, 1.0, 2)

    # sv_computer = SingularValueComputer(L=24, p_fixed=0.4, p_fixed_name="p_ctrl", p_scan_values=p_scan_values, chunk_size=1, comparison=False)
    # result = sv_computer.compute_chunk()
    # print(f"Result keys: {result.keys()}")
    # print(f"Singular values keys: {result['singular_values'].keys()}")
    # print(f"Singular values: {np.shape(result['singular_values']['AB'])}")
