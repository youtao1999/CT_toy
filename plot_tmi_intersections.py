import json
import numpy as np
import matplotlib.pyplot as plt
import os

def read_tmi_results(L):
    """Read results for a specific L value"""
    filename = f'tmi_pctrl_results_L{L}/final_results_L{L}.json'
    with open(filename, 'r') as f:
        return json.load(f)

def plot_tmi_intersections(L_values=[8, 12, 16]):
    # Create figure directory if it doesn't exist
    os.makedirs('tmi_intersection_plots', exist_ok=True)
    
    # Read data for all L values
    results = {L: read_tmi_results(L) for L in L_values}
    
    # Get p_proj values from first L (assuming same for all L)
    p_proj_values = sorted([float(key.replace('pproj', '')) for key in results[L_values[0]].keys()])
    
    # Get p_ctrl values (assuming same for all)
    p_ctrl_values = np.linspace(0, 0.6, 21)  # 21 values as in your original code
    
    # Colors for different L values
    colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))
    
    # Create a plot for each p_proj value
    for p_proj in p_proj_values:
        plt.figure(figsize=(10, 6))
        
        # Plot data for each L value
        for idx, L in enumerate(L_values):
            key = f'pproj{p_proj:.3f}'
            tmi_values = results[L][key]['tmi']
            tmi_std = results[L][key]['tmi_std']
            
            plt.errorbar(p_ctrl_values, tmi_values, yerr=tmi_std,
                        label=f'L={L}', color=colors[idx],
                        marker='o', markersize=4, capsize=3,
                        linewidth=1.5, capthick=1)
        
        plt.title(f'TMI vs p_ctrl (p_proj = {p_proj:.3f})')
        plt.xlabel('p_ctrl')
        plt.ylabel('TMI')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add zero line for reference
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Set y-axis limits with some padding
        all_tmi = [results[L][f'pproj{p_proj:.3f}']['tmi'] for L in L_values]
        ymin = min([min(tmi) for tmi in all_tmi])
        ymax = max([max(tmi) for tmi in all_tmi])
        plt.ylim(ymin - abs(ymin)*0.1, ymax + abs(ymax)*0.1)
        
        # Save plot
        plt.savefig(f'tmi_intersection_plots/tmi_intersection_pproj{p_proj:.3f}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_tmi_intersections([8, 12, 16]) 