import json
import numpy as np
import matplotlib.pyplot as plt
import os

def read_tmi_results(L_values):
    """
    Read and process TMI results for specified L values.
    
    Args:
        L_values: List of L values to process (e.g., [12, 16])
    
    Returns:
        dict: Dictionary containing processed data for each L value
    """
    results = {}
    
    # Process each L value
    for L in L_values:
        # Read the JSON file
        filename = f'tmi_pctrl_results_L{L}/final_results_L{L}.json'
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found!")
            continue
            
        with open(filename, 'r') as f:
            results[L] = json.load(f)
    
    return results

def plot_tmi_vs_pctrl(L_values):
    """
    Plot TMI vs p_ctrl for each p_proj value, comparing different L values.
    """
    # Read data
    data = read_tmi_results(L_values)
    
    # Get p_proj values from first L (assuming same for all L)
    L_first = L_values[0]
    p_proj_values = sorted([float(key.replace('pproj', '')) for key in data[L_first].keys()])
    
    # Get p_ctrl values
    p_ctrl_values = np.linspace(0, 0.6, len(data[L_first][f'pproj{p_proj_values[0]:.3f}']['tmi']))
    
    # Create a plot for each p_proj value
    for p_proj in p_proj_values:
        plt.figure(figsize=(10, 6))
        
        # Plot data for each L value
        for L in L_values:
            tmi_values = data[L][f'pproj{p_proj:.3f}']['tmi']
            plt.plot(p_ctrl_values, tmi_values, 'o-', label=f'L={L}')
        
        plt.title(f'TMI vs p_ctrl (p_proj = {p_proj:.3f})')
        plt.xlabel('p_ctrl')
        plt.ylabel('TMI')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        plt.savefig(f'tmi_vs_pctrl_pproj{p_proj:.3f}.png', dpi=300, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Plot results for L=12 and L=16
    plot_tmi_vs_pctrl([8, 12, 16])