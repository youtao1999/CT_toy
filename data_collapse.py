from scipy.optimize import minimize
from scipy import stats
import h5py
from plot_tmi_results import compute_tmi_from_singular_values
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import os
# Set a specific backend if needed
# matplotlib.use('TkAgg')  # Uncomment if you need to specify a backend

def read_tmi_results_fine(p_fixed, p_fixed_name, p_c, L_values, n=0, threshold=1e-10):
    """
    Read singular values from HDF5 files and compute TMI statistics, then write results to a CSV file.
    If the CSV file already exists, read results from it instead.
    """
    import csv
    p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'

    output_filename = f'tmi_results_fine_{p_fixed_name}{p_fixed:.3f}_pc{p_c:.3f}.csv'
    # Check if the output file already exists
    if os.path.exists(output_filename):
        print(f"Output file {output_filename} already exists. Reading results from it.")
        results = {}
        with open(output_filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                L = int(row['L'])
                p_fixed_key = row[p_fixed_name]
                p_scan_idx = int(row[p_scan_name + '_index'])
                p_scan_value = float(row[p_scan_name + '_value'])
                tmi_mean = float(row['tmi_mean'])
                tmi_sem = float(row['tmi_sem'])
                
                if L not in results:
                    results[L] = {}
                if p_fixed_key not in results[L]:
                    results[L][p_fixed_key] = {'tmi_mean': [], 'tmi_sem': [], p_scan_name + '_values': []}
                
                results[L][p_fixed_key]['tmi_mean'].append(tmi_mean)
                results[L][p_fixed_key]['tmi_sem'].append(tmi_sem)
                results[L][p_fixed_key][p_scan_name + '_values'].append(p_scan_value)
        
        return results

    results = {}
    
    # Prepare to write to CSV
    with open(output_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(['L', p_fixed_name, p_scan_name + '_index', p_scan_name + '_value', 'tmi_mean', 'tmi_sem'])
        
        for L in L_values:
            filename = f'tmi_fine_L{L}_{p_fixed_name}{p_fixed:.3f}_pc{p_c}/final_results_L{L}.h5'
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
                
                p_fixed_key = f"{p_fixed_name}{p_fixed:.3f}"  # Format as "pproj0.500" instead of "0.5"
                p_fixed_group = f[p_fixed_key]
                print(f"\nProcessing {p_fixed_name} group: {p_fixed_key}")
                print(f"{p_fixed_name} group attributes: {dict(p_fixed_group.attrs)}")
                
                tmi_means = []
                tmi_sems = []
                p_scan_values = p_fixed_group[p_scan_name][:]  # Get p_ctrl values
                
                sv_group = p_fixed_group['singular_values']
                print("\nSingular value datasets:")
                for key in sv_group.keys():
                    print(f"{key}: shape {sv_group[key].shape}")
                
                num_p_scan = len(p_fixed_group[p_scan_name])
                print(f"Number of {p_scan_name} values: {num_p_scan}")
                print(f"{p_scan_name} values: {p_fixed_group[p_scan_name][:]}")

                for p_scan_idx in range(num_p_scan):
                    # Get singular values for all samples at this p_ctrl
                    num_samples = sv_group[list(sv_group.keys())[0]].shape[1]  # Get number of samples
                    singular_values = [{
                        key: sv_group[key][p_scan_idx, sample_idx] 
                        for key in sv_group.keys()
                    } for sample_idx in range(num_samples)]
                    
                    # Compute TMI for each sample
                    tmi_values = [compute_tmi_from_singular_values(sv, n, threshold) 
                                for sv in singular_values]
                    
                    tmi_mean = np.mean(tmi_values)
                    tmi_sem = stats.sem(tmi_values)
                    
                    tmi_means.append(tmi_mean)
                    tmi_sems.append(tmi_sem)
                    
                    # Write to CSV with p_ctrl value
                    writer.writerow([L, p_fixed_key, p_scan_idx, p_scan_values[p_scan_idx], tmi_mean, tmi_sem])
                
                results[L][p_fixed_key] = {
                    'tmi_mean': tmi_means,
                    'tmi_sem': tmi_sems,
                    p_scan_name + '_values': p_scan_values.tolist()
                }
    
    return results

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

def data_collapse(p_fixed, p_fixed_name, p_c, L_values = [8, 12], n=0, threshold=1e-10):
    """
    Plot TMI vs p_ctrl with error bars for each p_proj value, comparing different L values.
    Uses specified Rényi entropy index n and threshold.
    """
    # Read and process data
    unarranged_data = read_tmi_results_fine(p_fixed, p_fixed_name, p_c, L_values, n, threshold)
    
    # Initialize p_scan_name
    p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
    # # Get p_ctrl values
    # Extract p_ctrl values from first L value's data
    first_L = L_values[0]
    first_pfixed_key = f'{p_fixed_name}{p_fixed:.3f}'
    p_scan_values = unarranged_data[first_L][first_pfixed_key][p_scan_name + '_values']

    # Rearrange all data
    p_all = [p_scan_values] * len(L_values)
    L_all = [np.ones(len(p_scan_values)) * L for L in L_values]
    y_all = [unarranged_data[L][first_pfixed_key]['tmi_mean'] for L in L_values]
    sigma_y_all = [unarranged_data[L][first_pfixed_key]['tmi_sem'] for L in L_values]

    # Modify parameter initialization
    params = Parameters()
    params.add('pc', value=0.5, min=0.3, max=0.7, vary=True)
    params.add('nu', value=1.0, min=0.5, max=2.0, vary=True)
    
    # Add more robust fitting options
    result = minimize(residuals_lmfit, 
                    params, 
                    args=(p_all, L_all, y_all, sigma_y_all),
                    method='leastsq',  # Try other methods like 'nelder', 'powell'
                    max_nfev=10000,
                    ftol=1e-11,
                    xtol=1e-11)

    # Print selective results
    print(f"\nResults for {p_fixed_name} = {p_fixed}:")
    print(f"Critical point (pc) = {result.params['pc'].value:.6f}", end='')
    if result.params['pc'].stderr is not None:
        print(f" ± {result.params['pc'].stderr:.6f}")
    else:
        print(" (stderr not available)")
        
    print(f"Critical exponent (nu) = {result.params['nu'].value:.6f}", end='')
    if result.params['nu'].stderr is not None:
        print(f" ± {result.params['nu'].stderr:.6f}")
    else:
        print(" (stderr not available)")
    
    print(f"Reduced chi-square = {result.redchi:.6f}")

    # Plot results
    # Create the scaled x values using fitted parameters
    pc = result.params['pc'].value
    nu = result.params['nu'].value
    
    plt.figure(figsize=(10, 6))
    
    # Plot data points with error bars for each L value
    for i, L in enumerate(L_values):
        x_scaled = (p_scan_values - pc) * L**(1/nu)
        y = unarranged_data[L][first_pfixed_key]['tmi_mean']
        yerr = unarranged_data[L][first_pfixed_key]['tmi_sem']
        
        plt.errorbar(x_scaled, y, yerr=yerr, fmt='o', label=f'L={L}')
    
    plt.xlabel(f'$(p - {pc:.3f})L^{{1/{nu:.3f}}}$')
    plt.ylabel('TMI')
    plt.title(f'Data Collapse for {p_fixed_name} = {p_fixed:.3f}')
    plt.legend()
    plt.grid(True)
    # Create plots directory if it doesn't exist
    os.makedirs('_data_collapse_plots', exist_ok=True)
    # Save plot with descriptive filename
    plt.savefig(f'_data_collapse_plots/data_collapse_{p_fixed_name}{p_fixed:.3f}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    data_collapse(p_fixed=0.5, p_fixed_name='pproj', p_c=0.5)
