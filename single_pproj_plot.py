import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json

def combine_tmi_results(folder_path, L):
    """Combine and average TMI results from chunk calculations."""
    
    # Pattern to match result files
    pattern = os.path.join(folder_path, "chunk_*.json")
    chunk_files = glob.glob(pattern)
    
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {folder_path}")
    
    # Initialize by reading first file to get p_ctrl values and structure
    with open(chunk_files[0], 'r') as f:
        first_chunk = json.load(f)
        p_ctrl_values = first_chunk['p_ctrl']
        num_p_ctrl = len(p_ctrl_values)
    
    # Initialize list to store all TMI samples for each p_ctrl
    all_samples = [[] for _ in range(num_p_ctrl)]
    
    # Read and combine all chunks
    for file in chunk_files:
        with open(file, 'r') as f:
            chunk_data = json.load(f)
            for p_ctrl_idx in range(num_p_ctrl):
                all_samples[p_ctrl_idx].extend(chunk_data['tmi_samples'][p_ctrl_idx])
    
    # Calculate means and standard errors
    tmi_means = []
    tmi_stds = []
    
    for p_ctrl_samples in all_samples:
        tmi_means.append(np.mean(p_ctrl_samples))
        tmi_stds.append(np.std(p_ctrl_samples) / np.sqrt(len(p_ctrl_samples)))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(p_ctrl_values, tmi_means, yerr=tmi_stds, fmt='o-', capsize=5)
    plt.xlabel('Control Parameter (p_ctrl)')
    plt.ylabel('TMI')
    plt.title(f'Average TMI vs Control Parameter (L={L})')
    plt.grid(True)
    
    # Save plot and data
    plt.savefig(os.path.join(folder_path, f'tmi_L{L}_combined.png'))
    np.savetxt(os.path.join(folder_path, f'tmi_L{L}_combined.txt'), 
               np.column_stack([p_ctrl_values, tmi_means, tmi_stds]),
               header='p_ctrl tmi tmi_std')
    
    return p_ctrl_values, tmi_means, tmi_stds

def plot_all_txt_results(folder_path):
    """Plot all txt results from the folder."""
    
    # Pattern to match txt files
    pattern = os.path.join(folder_path, "*.txt")
    txt_files = glob.glob(pattern)
    
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in {folder_path}")
    
    plt.figure(figsize=(10, 6))
    
    # Read and plot each file
    for file in txt_files:
        # Extract L value from filename (assuming format tmi_L{L}_combined.txt)
        L = int(file.split('L')[-1].split('_')[0])
        
        # Load data: p_ctrl, tmi, tmi_std
        data = np.loadtxt(file)
        p_ctrl = data[:, 0]
        tmi = data[:, 1]
        tmi_std = data[:, 2]
        
        # Plot with error bars
        plt.errorbar(p_ctrl, tmi, yerr=tmi_std, fmt='o-', capsize=5, label=f'L={L}')
    
    plt.xlabel('Control Parameter (p_ctrl)')
    plt.ylabel('TMI')
    plt.title('TMI vs Control Parameter for Different System Sizes')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.savefig(os.path.join(folder_path, 'all_L_combined.png'))
    plt.show()

if __name__ == "__main__":
    folder_path = 'p_proj_0.3'
    plot_all_txt_results(folder_path)