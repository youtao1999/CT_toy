# Set environment variables BEFORE importing libraries
import os

# Set thread limits before importing numerical libraries
thread_vars = {
    "NUMBA_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMPY_NUM_THREADS": "1",
    "NUMBA_DEFAULT_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1"
}

for var, value in thread_vars.items():
    os.environ[var] = value

# Now import the rest of the libraries
import numpy as np
import time
from QCT import QCT
from QCT_util import Haar_state
import psutil
from haining_correct_functions import random_control
import argparse

def set_cpu_affinity(num_cores):
    """Set CPU affinity for the current process"""
    p = psutil.Process()
    # Get list of available CPU cores
    cpu_list = list(range(psutil.cpu_count()))
    # Use only specified number of cores
    p.cpu_affinity(cpu_list[:num_cores])
    
    # Verify the setting
    current_affinity = p.cpu_affinity()
    print(f"Process restricted to CPU cores: {current_affinity}")
    if len(current_affinity) != num_cores:
        print(f"Warning: Requested {num_cores} cores but got {len(current_affinity)}")
    
    # Monitor CPU usage
    cpu_percent = p.cpu_percent(interval=1)
    print(f"Current CPU usage: {cpu_percent}%")

# Control thread/core usage for various libraries
def set_thread_environment(num_threads):
    """Set the number of threads for various libraries"""
    # Environment variables for different libraries
    thread_vars = {
        "NUMBA_NUM_THREADS": num_threads,
        "MKL_NUM_THREADS": num_threads,
        "OPENBLAS_NUM_THREADS": num_threads,
        "OMP_NUM_THREADS": num_threads,
        "VECLIB_MAXIMUM_THREADS": num_threads,
        "NUMEXPR_NUM_THREADS": num_threads,
        "NUMPY_NUM_THREADS": num_threads,  # Added this for NumPy
    }
    
    for var, value in thread_vars.items():
        os.environ[var] = str(value)
    
    print(f"Thread environment set to use {num_threads} threads")
    print("Current thread settings:")
    for var in thread_vars:
        print(f"- {var}: {os.environ.get(var, 'not set')}")

def benchmark_evolution_functions(L_values, num_iterations, num_cores=1):
    # Set CPU and thread affinity
    set_cpu_affinity(num_cores)
    set_thread_environment(num_cores)
    
    results = {}

    for L in L_values:
        # Initialize QCT instance for this L
        p_ctrl = 0.9
        p_proj = 0.9
        qct = QCT(L, p_ctrl, p_proj)
        
        num_steps = 2 * (L ** 2)  # Set number of steps to 2*L^2
        print(f"\nBenchmarking for L = {L} with {num_steps} time steps")
        
        results[L] = {
            'step_evolution_time': 0,
            'random_control_time': 0,
            'speedup': 0,
            'num_steps': num_steps
        }

        step_evolution_times = []
        random_control_times = []

        for i in range(num_iterations):
            print(f"Progress: {i:3d}/{num_iterations} [{i/num_iterations:3.0%}]", end='\r')
            # Generate a random state
            state = Haar_state(L, ensemble=1, rng=np.random.default_rng(), k=1).flatten()
            p_ctrl = 0.9
            p_proj = 0.9

            # Benchmark step_evolution
            state_copy = state.copy()
            start_time = time.perf_counter()
            for _ in range(num_steps):
                qct.step_evolution()
            step_time = time.perf_counter() - start_time
            step_evolution_times.append(step_time)

            # Benchmark random_control
            state_copy = state.copy()
            start_time = time.perf_counter()
            for _ in range(num_steps):
                state_copy, _ = random_control(state_copy, p_ctrl, p_proj, L)
            random_time = time.perf_counter() - start_time
            random_control_times.append(random_time)

        print(f"\nCompleted {num_iterations}/{num_iterations} [100%]")

        # Calculate average times
        results[L]['step_evolution_time'] = np.mean(step_evolution_times)
        results[L]['random_control_time'] = np.mean(random_control_times)
        results[L]['speedup'] = results[L]['random_control_time'] / results[L]['step_evolution_time']

    return results

def print_results(results):
    print("\nBenchmark Results:")
    print("=" * 90)
    print(f"{'L':<5}{'Steps':<8}{'step_evolution (h)':<20}{'random_control (h)':<20}{'Speedup':<10}")
    print("-" * 90)
    for L, data in results.items():
        print(f"{L:<5}{data['num_steps']:<8}{data['step_evolution_time']*2000/3600:<20.6f}"
              f"{data['random_control_time']*2000/3600:<20.6f}{data['speedup']:<10.2f}")

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark QCT evolution functions')
    parser.add_argument('--cores', type=int, default=1,
                       help='Number of CPU cores to use')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations for benchmarking')
    parser.add_argument('--L-values', type=int, nargs='+', default=[20],
                       help='List of L values to benchmark')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Running benchmark with:")
    print(f"- {args.cores} CPU cores")
    print(f"- {args.iterations} iterations")
    print(f"- L values: {args.L_values}")
    
    results = benchmark_evolution_functions(args.L_values, args.iterations, args.cores)
    print_results(results)