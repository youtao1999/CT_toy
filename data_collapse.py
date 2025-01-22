from scipy.optimize import minimize
from scipy import stats
import h5py
from plot_tmi_results import compute_tmi_from_singular_values
import numpy as np
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters
import os
import csv
import pandas as pd
# Set a specific backend if needed
# matplotlib.use('TkAgg')  # Uncomment if you need to specify a backend

# def read_tmi_results_fine(p_fixed, p_fixed_name, p_c, L_values, n=0, threshold=1e-10):
#     """
#     Read singular values from HDF5 files and compute TMI statistics, then write results to a CSV file.
#     If the CSV file already exists, read results from it instead.
#     """
#     import csv
#     p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'

#     output_filename = f'tmi_results_fine_{p_fixed_name}{p_fixed:.3f}_pc{p_c:.3f}.csv'
#     # Check if the output file already exists
#     if os.path.exists(output_filename):
#         print(f"Output file {output_filename} already exists. Reading results from it.")
#         results = {}
#         with open(output_filename, mode='r') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 L = int(row['L'])
#                 p_fixed_key = row[p_fixed_name]
#                 p_scan_idx = int(row[p_scan_name + '_index'])
#                 p_scan_value = float(row[p_scan_name + '_value'])
#                 tmi_mean = float(row['tmi_mean'])
#                 tmi_sem = float(row['tmi_sem'])
                
#                 if L not in results:
#                     results[L] = {}
#                 if p_fixed_key not in results[L]:
#                     results[L][p_fixed_key] = {'tmi_mean': [], 'tmi_sem': [], p_scan_name + '_values': []}
                
#                 results[L][p_fixed_key]['tmi_mean'].append(tmi_mean)
#                 results[L][p_fixed_key]['tmi_sem'].append(tmi_sem)
#                 results[L][p_fixed_key][p_scan_name + '_values'].append(p_scan_value)
        
#         return results

#     results = {}
    
#     # Prepare to write to CSV
#     with open(output_filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         # Write header
#         writer.writerow(['L', p_fixed_name, p_scan_name + '_index', p_scan_name + '_value', 'tmi_mean', 'tmi_sem'])
        
#         for L in L_values:
#             filename = f'tmi_fine_L{L}_{p_fixed_name}{p_fixed:.3f}_pc{p_c}/final_results_L{L}.h5'
#             if not os.path.exists(filename):
#                 print(f"Warning: File {filename} not found!")
#                 continue
                
#             print(f"\nAnalyzing file: {filename}")
#             with h5py.File(filename, 'r') as f:
#                 # Print file attributes
#                 print(f"File attributes: {dict(f.attrs)}")
                
#                 # Print structure of groups and datasets
#                 print("\nFile structure:")
#                 def print_structure(name, obj):
#                     if isinstance(obj, h5py.Group):
#                         print(f"GROUP: {name}/")
#                     elif isinstance(obj, h5py.Dataset):
#                         print(f"DATASET: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
#                 f.visititems(print_structure)
                
#                 results[L] = {}
                
#                 p_fixed_key = f"{p_fixed_name}{p_fixed:.3f}"  # Format as "pproj0.500" instead of "0.5"
#                 p_fixed_group = f[p_fixed_key]
#                 print(f"\nProcessing {p_fixed_name} group: {p_fixed_key}")
#                 print(f"{p_fixed_name} group attributes: {dict(p_fixed_group.attrs)}")
                
#                 tmi_means = []
#                 tmi_sems = []
#                 p_scan_values = p_fixed_group[p_scan_name][:]  # Get p_ctrl values
                
#                 sv_group = p_fixed_group['singular_values']
#                 print("\nSingular value datasets:")
#                 for key in sv_group.keys():
#                     print(f"{key}: shape {sv_group[key].shape}")
                
#                 num_p_scan = len(p_fixed_group[p_scan_name])
#                 print(f"Number of {p_scan_name} values: {num_p_scan}")
#                 print(f"{p_scan_name} values: {p_fixed_group[p_scan_name][:]}")

#                 for p_scan_idx in range(num_p_scan):
#                     # Get singular values for all samples at this p_ctrl
#                     num_samples = sv_group[list(sv_group.keys())[0]].shape[1]  # Get number of samples
#                     singular_values = [{
#                         key: sv_group[key][p_scan_idx, sample_idx] 
#                         for key in sv_group.keys()
#                     } for sample_idx in range(num_samples)]
                    
#                     # Compute TMI for each sample
#                     tmi_values = [compute_tmi_from_singular_values(sv, n, threshold) 
#                                 for sv in singular_values]
                    
#                     tmi_mean = np.mean(tmi_values)
#                     tmi_sem = stats.sem(tmi_values)
                    
#                     tmi_means.append(tmi_mean)
#                     tmi_sems.append(tmi_sem)
                    
#                     # Write to CSV with p_ctrl value
#                     writer.writerow([L, p_fixed_key, p_scan_idx, p_scan_values[p_scan_idx], tmi_mean, tmi_sem])
                
#                 results[L][p_fixed_key] = {
#                     'tmi_mean': tmi_means,
#                     'tmi_sem': tmi_sems,
#                     p_scan_name + '_values': p_scan_values.tolist()
#                 }
    
#     return results

def read_tmi_results_to_df(p_fixed, p_fixed_name, p_c, L_values, n=0, threshold=1e-10):
    """
    Read singular values from HDF5 files, compute TMI statistics, and return results as a DataFrame.
    If results were previously computed and saved to CSV, read directly from CSV instead.
    
    Parameters:
    -----------
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    p_c : float
        Critical point value
    L_values : list
        List of L values to process
    n : int, optional
        Parameter for TMI computation
    threshold : float, optional
        Threshold for TMI computation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with MultiIndex (p, L) containing TMI observations
    """
    output_filename = f'tmi_results_fine_{p_fixed_name}{p_fixed:.3f}_pc{p_c:.3f}.csv'
    p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
    # If CSV exists, read directly from it
    if os.path.exists(output_filename):
        print(f"Reading existing results from {output_filename}")
        data_list = []
        with open(output_filename, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data_list.append({
                    'p': float(row[f'{p_scan_name}_value']),
                    'L': int(row['L']),
                    'observations': float(row['tmi_value'])
                })
    else:
        data_list = []
        
        # Write results to CSV and collect data for DataFrame
        with open(output_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['L', p_fixed_name, p_scan_name + '_index', p_scan_name + '_value', 'sample_idx', 'tmi_value'])
            
            for L in L_values:
                filename = f'tmi_fine_L{L}_{p_fixed_name}{p_fixed:.3f}_pc{p_c}/final_results_L{L}.h5'
                if not os.path.exists(filename):
                    print(f"Warning: File {filename} not found!")
                    continue
                    
                print(f"\nAnalyzing file: {filename}")
                with h5py.File(filename, 'r') as f:
                    p_fixed_key = f"{p_fixed_name}{p_fixed:.3f}"
                    p_fixed_group = f[p_fixed_key]
                    p_scan_values = p_fixed_group[p_scan_name][:]
                    sv_group = p_fixed_group['singular_values']
                    
                    num_p_scan = len(p_fixed_group[p_scan_name])
                    
                    for p_scan_idx in range(num_p_scan):
                        num_samples = sv_group[list(sv_group.keys())[0]].shape[1]
                        singular_values = [{
                            key: sv_group[key][p_scan_idx, sample_idx] 
                            for key in sv_group.keys()
                        } for sample_idx in range(num_samples)]
                        
                        # Compute TMI for each sample
                        tmi_values = [compute_tmi_from_singular_values(sv, n, threshold) 
                                    for sv in singular_values]
                        
                        # Write to CSV and collect for DataFrame
                        for sample_idx, tmi_value in enumerate(tmi_values):
                            writer.writerow([L, p_fixed_key, p_scan_idx, p_scan_values[p_scan_idx], sample_idx, tmi_value])
                            data_list.append({
                                'p': p_scan_values[p_scan_idx],
                                'L': L,
                                'observations': tmi_value
                            })
    
    # Create DataFrame and group observations
    df = pd.DataFrame(data_list)
    df_grouped = df.groupby(['p', 'L'])['observations'].apply(list).reset_index()
    df_final = df_grouped.set_index(['p', 'L'])
    
    return df_final

# Linear interpolation function
def linear_interpolation(x_sorted, y_sorted, sigma_y_sorted, x_target):
    """
    Perform linear interpolation to estimate y' and sigma at x_target.
    x_target is guaranteed to be one of the values in x_sorted.
    """
    n = len(x_sorted)
    
    # Handle first point
    if x_target == x_sorted[0]:
        slope = (y_sorted[1] - y_sorted[0]) / (x_sorted[1] - x_sorted[0])
        y_prime = y_sorted[0]
        # Propagate errors using two points
        sigma_prime = np.sqrt(sigma_y_sorted[0]**2 + sigma_y_sorted[1]**2)
        return y_prime, sigma_prime
        
    # Handle last point
    if x_target == x_sorted[-1]:
        slope = (y_sorted[-1] - y_sorted[-2]) / (x_sorted[-1] - x_sorted[-2])
        y_prime = y_sorted[-1]
        # Propagate errors using two points
        sigma_prime = np.sqrt(sigma_y_sorted[-1]**2 + sigma_y_sorted[-2]**2)
        return y_prime, sigma_prime
    
    # Handle middle points with three-point interpolation
    for i in range(1, n - 1):
        if x_target == x_sorted[i]:
            # Use the updated slope with three points
            slope = (y_sorted[i + 1] - y_sorted[i - 1]) / (x_sorted[i + 1] - x_sorted[i - 1])
            y_prime = y_sorted[i]
            
            # Propagate errors using three points
            term1 = sigma_y_sorted[i - 1] ** 2 * ((x_sorted[i + 1] - x_target) / (x_sorted[i + 1] - x_sorted[i - 1])) ** 2
            term2 = sigma_y_sorted[i + 1] ** 2 * ((x_sorted[i - 1] - x_target) / (x_sorted[i + 1] - x_sorted[i - 1])) ** 2
            sigma_prime = np.sqrt(sigma_y_sorted[i] ** 2 + term1 + term2)
            
            return y_prime, sigma_prime
            
    raise ValueError("x_target not found in x_sorted. This should never happen.")

# Residual function for lmfit
def residuals_lmfit(params, p_all, L_all, y_all, sigma_y_all):
    """
    Compute the residuals for lmfit, incorporating all system sizes.
    """
    pc = params['pc'].value
    nu = params['nu'].value
    
    # Print parameter values during optimization
    print(f"Testing pc={pc:.6f}, nu={nu:.6f}")
    
    # Calculate x values and sort everything
    x = abs(np.concatenate(p_all) - pc) * np.concatenate(L_all)**(1/nu)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = np.concatenate(y_all)[sorted_indices]
    sigma_y_sorted = np.concatenate(sigma_y_all)[sorted_indices]

    # Let's print some intermediate values to debug
    print(f"First few x values: {x_sorted[:5]}")
    print(f"First few y values: {y_sorted[:5]}")
    
    residuals = []
    for i, x_val in enumerate(x_sorted):
        y_prime, sigma_prime = linear_interpolation(x_sorted, y_sorted, sigma_y_sorted, x_val)
        residual = (y_sorted[i] - y_prime) / sigma_prime
        
        # Print first few interpolation results
        if i < 5:
            print(f"Point {i}: x={x_val:.6f}, y={y_sorted[i]:.6f}, y_prime={y_prime:.6f}, "
                  f"sigma={sigma_prime:.6f}, residual={residual:.6f}")
        
        residuals.append(residual)
    
    residuals = np.array(residuals)
    print(f"Residuals stats: mean={np.mean(residuals):.2e}, std={np.std(residuals):.2e}, "
          f"min={np.min(residuals):.2e}, max={np.max(residuals):.2e}")
    
    return residuals

def data_collapse(p_fixed, p_fixed_name, p_c, L_values = [8, 12, 16, 20], n=0, threshold=1e-10):
    """
    Plot TMI vs p_ctrl with error bars for each p_proj value, comparing different L values.
    Uses specified Rényi entropy index n and threshold.
    """
    # Read and process data
    df = read_tmi_results_to_df(p_fixed, p_fixed_name, p_c, L_values, n, threshold)
    
    # Initialize p_scan_name
    p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
    
    # Prepare data for fitting
    p_all = []
    L_all = []
    y_all = []
    sigma_y_all = []
    
    # Process each L value
    for L in L_values:
        # Get all p values for this L
        L_data = df.xs(L, level='L')
        p_values = L_data.index.values
        observations = L_data['observations'].values
        
        # Calculate mean and standard error for each p value
        means = [np.mean(obs) for obs in observations]
        sems = [stats.sem(obs) for obs in observations]
        
        p_all.append(p_values)
        L_all.append(np.ones(len(p_values)) * L)
        y_all.append(means)
        sigma_y_all.append(sems)

    # Modify parameter initialization and fitting
    params = Parameters()
    params.add('pc', value=0.5, min=0.3, max=0.7, vary=True, brute_step=0.01)
    params.add('nu', value=1.0, min=0.5, max=2.0, vary=True, brute_step=0.01)
    
    # Try different methods if leastsq fails
    methods = ['leastsq', 'nelder', 'powell']
    best_result = None
    best_chisqr = np.inf

    for method in methods:
        try:
            # Set method-specific parameters
            fit_kwargs = {
                'calc_covar': True,
                'nan_policy': 'raise',
                'max_nfev': 10000
            }
            
            # Add method-specific parameters
            if method == 'leastsq':
                fit_kwargs.update({
                    'ftol': 1e-8,
                    'xtol': 1e-8
                })
            
            result = minimize(residuals_lmfit, 
                         params, 
                         args=(p_all, L_all, y_all, sigma_y_all),
                         method=method,
                         **fit_kwargs)
            
            if result.success and result.chisqr < best_chisqr and result.chisqr > 0:
                best_result = result
                best_chisqr = result.chisqr
                
        except Exception as e:
            print(f"Method {method} failed with error: {str(e)}")
            continue
    
    if best_result is None:
        raise ValueError("All fitting methods failed")
        
    result = best_result

    # Print more detailed fit diagnostics
    print("\nFit Diagnostics:")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Number of function evaluations: {result.nfev}")
    print(f"Number of variables: {result.nvarys}")
    print(f"Number of data points: {result.ndata}")
    print(f"Degrees of freedom: {result.nfree}")
    print(f"Chi-square: {result.chisqr}")
    print(f"Reduced chi-square: {result.redchi}")
    if result.covar is not None:
        print("Covariance matrix was calculated")
    else:
        print("Warning: Covariance matrix calculation failed")

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
        x_scaled = (p_values - pc) * L**(1/nu)
        y = y_all[i]
        yerr = sigma_y_all[i]
        
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
    data_collapse(p_fixed=0.500, p_fixed_name='pproj', p_c=0.500)