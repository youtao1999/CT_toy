from plot_tmi_results import read_tmi_results
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import os
# Set a specific backend if needed
# matplotlib.use('TkAgg')  # Uncomment if you need to specify a backend

# Linear interpolation function
def linear_interpolation(x_sorted, y_sorted, sigma_y_sorted, x_target):
    """
    Perform linear interpolation to estimate y' and sigma at x_target.
    Correctly computes the slope and handles endpoints.
    """
    n = len(x_sorted)
    
    for i in range(1, n - 1):
        if x_sorted[i - 1] <= x_target <= x_sorted[i + 1]:
            # Use the updated slope with three points
            slope = (y_sorted[i + 1] - y_sorted[i - 1]) / (x_sorted[i + 1] - x_sorted[i - 1])
            y_prime = y_sorted[i - 1] + slope * (x_target - x_sorted[i - 1])
            
            # Propagate errors using three points
            term1 = sigma_y_sorted[i - 1] ** 2 * ((x_sorted[i + 1] - x_target) / (x_sorted[i + 1] - x_sorted[i - 1])) ** 2
            term2 = sigma_y_sorted[i + 1] ** 2 * ((x_sorted[i - 1] - x_target) / (x_sorted[i + 1] - x_sorted[i - 1])) ** 2
            sigma_prime = np.sqrt(sigma_y_sorted[i] ** 2 + term1 + term2)
            
            return y_prime, sigma_prime
    
    # Handle endpoints with one-sided interpolation
    if x_target < x_sorted[0]:
        slope = (y_sorted[1] - y_sorted[0]) / (x_sorted[1] - x_sorted[0])
        y_prime = y_sorted[0] + slope * (x_target - x_sorted[0])
        sigma_prime = sigma_y_sorted[0]
        return y_prime, sigma_prime
    
    if x_target > x_sorted[-1]:
        slope = (y_sorted[-1] - y_sorted[-2]) / (x_sorted[-1] - x_sorted[-2])
        y_prime = y_sorted[-1] + slope * (x_target - x_sorted[-1])
        sigma_prime = sigma_y_sorted[-1]
        return y_prime, sigma_prime


# Residual function for lmfit
def residuals_lmfit(params, p_all, L_all, y_all, sigma_y_all):
    """
    Compute the residuals for lmfit, incorporating all system sizes.
    """
    pc = params['pc']
    nu = params['nu']
    residuals = []

    # Calculate x values
    x = abs(np.concatenate(p_all) - pc * np.ones(len(np.concatenate(p_all)))) * np.concatenate(L_all)**(1 / nu * np.ones(len(np.concatenate(L_all))))
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = np.concatenate(y_all)[sorted_indices]
    sigma_y_sorted = np.concatenate(sigma_y_all)[sorted_indices]

    for i, x_val in enumerate(x_sorted):
        y_prime, sigma_prime = linear_interpolation(x_sorted, y_sorted, sigma_y_sorted, x_val)
        residuals.append((y_sorted[i] - y_prime) / sigma_prime)

    return np.array(residuals)  # Convert to numpy array before returning

def data_collapse(L_values = [8, 12, 16, 20], n=0, threshold=1e-10):
    """
    Plot TMI vs p_ctrl with error bars for each p_proj value, comparing different L values.
    Uses specified Rényi entropy index n and threshold.
    """
    # Read and process data
    unarranged_data = read_tmi_results(L_values, n, threshold)
    
    # Get p_proj values from first L
    L_first = L_values[0]
    p_proj_values = sorted([float(key.replace('pproj', '')) for key in unarranged_data[L_first].keys()])
    
    # Get p_ctrl values
    p_ctrl_values = np.linspace(0, 0.6, len(unarranged_data[L_first][f'pproj{p_proj_values[0]:.3f}']['tmi_mean']))

    for p_proj_index in range(len(p_proj_values)):
        # Rearrange all data
        p_proj = p_proj_values[p_proj_index]
        p_all = [p_ctrl_values] * len(L_values)
        L_all = [np.ones(len(p_ctrl_values)) * L for L in L_values]
        y_all = [unarranged_data[L][f'pproj{p_proj:.3f}']['tmi_mean'] for L in L_values]
        sigma_y_all = [unarranged_data[L][f'pproj{p_proj:.3f}']['tmi_sem'] for L in L_values]

        # Modify parameter initialization
        params = Parameters()
        params.add('pc', value=0.5, min=0.3, max=0.7, vary=True)
        params.add('nu', value=1.0, min=0.5, max=2.0, vary=True)
        
        # Add more robust fitting options
        result = minimize(residuals_lmfit, 
                        params, 
                        args=(p_all, L_all, y_all, sigma_y_all),
                        method='leastsq',  # Try other methods like 'nelder', 'powell'
                        max_nfev=1000,
                        ftol=1e-11,
                        xtol=1e-11)
    
        # Print selective results
        print(f"\nResults for p_proj = {p_proj}:")
        print(f"Critical point (pc) = {result.params['pc'].value:.6f} ± {result.params['pc'].stderr:.6f}")
        print(f"Critical exponent (nu) = {result.params['nu'].value:.6f} ± {result.params['nu'].stderr:.6f}")
        print(f"Reduced chi-square = {result.redchi:.6f}")

        # Plot results
        # Create the scaled x values using fitted parameters
        pc = result.params['pc'].value
        nu = result.params['nu'].value
        
        plt.figure(figsize=(10, 6))
        
        # Plot data points with error bars for each L value
        for i, L in enumerate(L_values):
            x_scaled = (p_ctrl_values - pc) * L**(1/nu)
            y = unarranged_data[L][f'pproj{p_proj:.3f}']['tmi_mean']
            yerr = unarranged_data[L][f'pproj{p_proj:.3f}']['tmi_sem']
            
            plt.errorbar(x_scaled, y, yerr=yerr, fmt='o', label=f'L={L}')
        
        plt.xlabel(r'$(p - p_c)L^{1/\nu}$')
        plt.ylabel('TMI')
        plt.title(f'Data Collapse for $p_{{proj}} = {p_proj:.3f}$')
        plt.legend()
        plt.grid(True)
        # Create plots directory if it doesn't exist
        os.makedirs('plots', exist_ok=True)
        # Save plot with descriptive filename
        plt.savefig(f'plots/data_collapse_pproj_{p_proj:.3f}.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    data_collapse()

    