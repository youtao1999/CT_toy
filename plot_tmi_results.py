import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

def compute_entropy_from_singular_values(S, n, threshold):
    """
    Compute von Neumann entropy from singular values with specified Rényi index.
    """
    S_pos = np.clip(S, threshold, None)
    eigenvalues = S_pos**2
    
    if n == 1:
        entropy = -np.sum(np.log(eigenvalues) * eigenvalues)
        return 0 if np.isnan(entropy) else entropy
    elif n == 0:
        return np.log((eigenvalues > threshold).sum())
    elif n == np.inf:
        return -np.log(np.max(eigenvalues))
    else:
        return np.log(np.sum(eigenvalues**n)) / (1-n)

def compute_tmi_from_singular_values(singular_values, n, threshold):
    """
    Compute TMI from singular values using specified Rényi entropy index.
    """
    # Compute entropies for each region
    S_A = compute_entropy_from_singular_values(singular_values['A'], n, threshold)
    S_B = compute_entropy_from_singular_values(singular_values['B'], n, threshold)
    S_C = compute_entropy_from_singular_values(singular_values['C'], n, threshold)
    S_AB = compute_entropy_from_singular_values(singular_values['AB'], n, threshold)
    S_AC = compute_entropy_from_singular_values(singular_values['AC'], n, threshold)
    S_BC = compute_entropy_from_singular_values(singular_values['BC'], n, threshold)
    S_ABC = compute_entropy_from_singular_values(singular_values['ABC'], n, threshold)
    
    return S_A + S_B + S_C - S_AB - S_AC - S_BC + S_ABC

def read_tmi_results(L_values, n=0, threshold=1e-10, output_filename='tmi_results.csv'):
    """
    Read singular values from HDF5 files and compute TMI statistics, then write results to a CSV file.
    If the CSV file already exists, read results from it instead.
    """
    import csv

    # Check if the output file already exists
    if os.path.exists(output_filename):
        print(f"Output file {output_filename} already exists. Reading results from it.")
        results = {}
        with open(output_filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                L = int(row['L'])
                p_proj_key = row['p_proj']
                p_ctrl_idx = int(row['p_ctrl_index'])
                tmi_mean = float(row['tmi_mean'])
                tmi_sem = float(row['tmi_sem'])
                
                if L not in results:
                    results[L] = {}
                if p_proj_key not in results[L]:
                    results[L][p_proj_key] = {'tmi_mean': [], 'tmi_sem': []}
                
                results[L][p_proj_key]['tmi_mean'].append(tmi_mean)
                results[L][p_proj_key]['tmi_sem'].append(tmi_sem)
        
        return results

    results = {}
    
    # Prepare to write to CSV
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['L', 'p_proj', 'p_ctrl_index', 'tmi_mean', 'tmi_sem'])
        
        for L in L_values:
            filename = f'tmi0_pctrl_results_L{L}/final_results_L{L}.h5'
            if not os.path.exists(filename):
                print(f"Warning: File {filename} not found!")
                continue
                
            print(f"\nAnalyzing file: {filename}")
            with h5py.File(filename, 'r') as f:
                # Print file attributes
                print(f"File attributes: {dict(f.attrs)}")
                
                # Print structure of groups and datasets
                print("\nFile structure:")
                def print_structure(name, obj):
                    if isinstance(obj, h5py.Group):
                        print(f"GROUP: {name}/")
                    elif isinstance(obj, h5py.Dataset):
                        print(f"DATASET: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
                f.visititems(print_structure)
                
                results[L] = {}
                
                for p_proj_key in f.keys():
                    print(f"\nProcessing p_proj group: {p_proj_key}")
                    p_proj_group = f[p_proj_key]
                    print(f"p_proj group attributes: {dict(p_proj_group.attrs)}")
                    
                    tmi_means = []
                    tmi_sems = []
                    
                    sv_group = p_proj_group['singular_values']
                    print("\nSingular value datasets:")
                    for key in sv_group.keys():
                        print(f"{key}: shape {sv_group[key].shape}")
                    
                    num_p_ctrl = len(p_proj_group['p_ctrl'])
                    print(f"Number of p_ctrl values: {num_p_ctrl}")
                    print(f"p_ctrl values: {p_proj_group['p_ctrl'][:]}")

                    for p_ctrl_idx in range(num_p_ctrl):
                        # Get singular values for all samples at this p_ctrl
                        num_samples = sv_group[list(sv_group.keys())[0]].shape[1]  # Get number of samples
                        singular_values = [{
                            key: sv_group[key][p_ctrl_idx, sample_idx] 
                            for key in sv_group.keys()
                        } for sample_idx in range(num_samples)]
                        
                        # Compute TMI for each sample
                        tmi_values = [compute_tmi_from_singular_values(sv, n, threshold) 
                                    for sv in singular_values]
                        
                        tmi_mean = np.mean(tmi_values)
                        tmi_sem = stats.sem(tmi_values)
                        
                        tmi_means.append(tmi_mean)
                        tmi_sems.append(tmi_sem)
                        
                        # Write to CSV
                        writer.writerow([L, p_proj_key, p_ctrl_idx, tmi_mean, tmi_sem])
                    
                    results[L][p_proj_key] = {
                        'tmi_mean': tmi_means,
                        'tmi_sem': tmi_sems
                    }
    
    return results

def plot_tmi_vs_pctrl(L_values, n=0, threshold=1e-10):
    """
    Plot TMI vs p_ctrl with error bars for each p_proj value, comparing different L values.
    Uses specified Rényi entropy index n and threshold.
    """
    # Read and process data
    data = read_tmi_results(L_values, n, threshold)
    
    # Get p_proj values from first L
    L_first = L_values[0]
    p_proj_values = sorted([float(key.replace('pproj', '')) for key in data[L_first].keys()])
    
    # Get p_ctrl values
    p_ctrl_values = np.linspace(0, 0.6, len(data[L_first][f'pproj{p_proj_values[0]:.3f}']['tmi_mean']))
    
    # Create a plot for each p_proj value
    for p_proj in p_proj_values:
        plt.figure(figsize=(10, 6))
        
        for L in L_values:
            key = f'pproj{p_proj:.3f}'
            tmi_means = data[L][key]['tmi_mean']
            tmi_sems = data[L][key]['tmi_sem']
            
            plt.errorbar(p_ctrl_values, tmi_means, yerr=tmi_sems, 
                        fmt='o-', label=f'L={L}', capsize=3)
        
        plt.title(f'TMI vs p_ctrl (p_proj = {p_proj:.3f}, n = {n})')
        plt.xlabel('p_ctrl')
        plt.ylabel(f'TMI (n = {n})')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.savefig(f'tmi_vs_pctrl_pproj{p_proj:.3f}_n{n}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Plot results for different Rényi indices
    L_values = [8, 12, 16, 20]
    n_values = [0]
    threshold = 1e-10
    
    for n in n_values:
        plot_tmi_vs_pctrl(L_values, n=n, threshold=threshold)