import pandas as pd
import os
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
from plot_tmi_results import compute_tmi_from_singular_values

def write_tmi_compare_results_to_csv(df_final, p_fixed, p_fixed_name, current_threshold, output_filename=None):
    """
    Write TMI comparison results to a CSV file.
    
    Parameters:
    -----------
    df_final : pd.DataFrame
        DataFrame with MultiIndex (p, L, implementation) containing TMI observations
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    current_threshold : float
        Current threshold value for TMI computation
    output_filename : str, optional
        Custom output filename
    
    Returns:
    --------
    str
        Name of the output file
    """
    if output_filename is None:
        output_filename = f'tmi_compare_results_{p_fixed_name}{p_fixed:.3f}_threshold{current_threshold:.1e}.csv'
    
    # Prepare data for CSV
    csv_data = []
    for (p, L, implementation), row in df_final.iterrows():
        for tmi_value in row['observations']:
            csv_data.append({
                'pctrl': p if p_fixed_name == 'pproj' else p_fixed,
                'pproj': p if p_fixed_name == 'pctrl' else p_fixed,
                'L': L,
                'implementation': implementation,
                'tmi': tmi_value
            })
    
    # Write to CSV
    pd.DataFrame(csv_data).to_csv(output_filename, index=False)
    print(f"Wrote results to {output_filename}")
    return output_filename

def read_and_compute_tmi_from_compare_file(filename, p_fixed_name, p_fixed, n, current_threshold):
    """
    Read singular values from a comparison HDF5 file and compute TMI values for both implementations.
    
    Parameters:
    -----------
    filename : str
        Path to the HDF5 file
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    p_fixed : float
        Fixed parameter value
    n : int
        Renyi entropy parameter
    current_threshold : float
        Threshold for singular values
        
    Returns:
    --------
    list
        List of dictionaries with TMI data
    """
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found!")
        return []
    
    data_list = []
    
    with h5py.File(filename, 'r') as f:
        p_fixed_key = f"{p_fixed_name}{p_fixed:.3f}"
        p_fixed_group = f[p_fixed_key]
        p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
        p_scan_values = p_fixed_group[p_scan_name][:]
        
        # Get L value from filename
        L = int(filename.split('_L')[-1].split('.')[0])
        
        # Process Tao's implementation
        if 'tao' in p_fixed_group:
            tao_sv_group = p_fixed_group['tao']['singular_values']
            
            for p_scan_idx in range(len(p_scan_values)):
                num_samples = tao_sv_group[list(tao_sv_group.keys())[0]].shape[1]
                
                tao_singular_values = [{
                    key: tao_sv_group[key][p_scan_idx, sample_idx] 
                    for key in tao_sv_group.keys()
                } for sample_idx in range(num_samples)]
                
                tao_tmi_values = [compute_tmi_from_singular_values(sv, n, current_threshold) 
                                for sv in tao_singular_values]
                
                data_list.append({
                    'p': p_scan_values[p_scan_idx],
                    'L': L,
                    'implementation': 'tao',
                    'observations': tao_tmi_values
                })
        
        # Process Haining's implementation
        if 'haining' in p_fixed_group:
            haining_sv_group = p_fixed_group['haining']['singular_values']
            
            for p_scan_idx in range(len(p_scan_values)):
                num_samples = haining_sv_group[list(haining_sv_group.keys())[0]].shape[1]
                
                haining_singular_values = [{
                    key: haining_sv_group[key][p_scan_idx, sample_idx] 
                    for key in haining_sv_group.keys()
                } for sample_idx in range(num_samples)]
                
                haining_tmi_values = [compute_tmi_from_singular_values(sv, n, current_threshold) 
                                    for sv in haining_singular_values]
                
                data_list.append({
                    'p': p_scan_values[p_scan_idx],
                    'L': L,
                    'implementation': 'haining',
                    'observations': haining_tmi_values
                })
    
    return data_list

def read_tmi_compare_results(p_fixed, p_fixed_name, thresholds=None, L_values=None, n=0, output_folder="tmi_compare_results"):
    """
    Read TMI comparison results, combining existing CSV data and optionally computing missing data.
    
    Parameters:
    -----------
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    thresholds : list, optional
        List of threshold values for TMI computation
    L_values : list, optional
        List of L values to include
    n : int, optional
        Renyi entropy parameter
    output_folder : str, optional
        Folder to store output files
        
    Returns:
    --------
    dict
        Dictionary with threshold values as keys and DataFrames as values
    """
    # Ensure we're in the correct directory
    current_dir = os.getcwd()
    if os.path.basename(current_dir) != 'CT_toy':
        if os.path.exists('code/CT_toy'):
            os.chdir('code/CT_toy')
        elif os.path.exists('CT_toy'):
            os.chdir('CT_toy')
        else:
            raise RuntimeError("Could not find CT_toy directory")
    
    results_dict = {}
    missing_thresholds = []

    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # Check if we are in the correct directory
    if os.path.basename(os.getcwd()) != output_folder:
        raise RuntimeError(f"Could not find {output_folder} directory")
    
    # Determine thresholds to process
    if thresholds is None:
        def extract_threshold(filename):
            """Extract threshold value from filename."""
            return float(filename.split('_threshold')[1].split('.csv')[0])

        file_pattern = f'tmi_compare_results_{p_fixed_name}{p_fixed:.3f}_threshold*.csv'
        all_files = glob.glob(file_pattern)
        thresholds = [extract_threshold(f) for f in all_files]
    
    # Check existing CSV files for each threshold
    for threshold in thresholds:
        filename = f'tmi_compare_results_{p_fixed_name}{p_fixed:.3f}_threshold{threshold:.1e}.csv'
        if os.path.exists(filename):
            print(f"Found existing data for threshold {threshold:.1e}")
            # Read and process CSV file
            df = pd.read_csv(filename)
            if L_values is not None:
                df = df[df['L'].isin(L_values)]
            
            # Group by p, L, and implementation to collect all TMI values as observations
            p_col = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
            df['p'] = df[p_col]  # Create 'p' column
            df_grouped = df.groupby(['p', 'L', 'implementation'])['tmi'].apply(list).reset_index()
            df_final = df_grouped.rename(columns={'tmi': 'observations'}).set_index(['p', 'L', 'implementation'])
            
            results_dict[threshold] = df_final
        else:
            missing_thresholds.append(threshold)
    
    if missing_thresholds:
        # Change directory back to CT_toy
        os.chdir(current_dir)

        print(f"\nMissing data for {len(missing_thresholds)} threshold values:")
        print(f"Thresholds: {[f'{t:.1e}' for t in missing_thresholds]}")
        compute = input("\nWould you like to compute these from H5 files? (yes/no): ").lower().strip()
        
        if compute in ['y', 'yes']:
            # Find all relevant HDF5 files
            file_pattern = f'sv_comparison_L*_{p_fixed_name}{p_fixed:.3f}*'
            all_files = glob.glob(file_pattern)
            
            if not all_files:
                print(f"No HDF5 files found matching pattern: {file_pattern}")
                return results_dict
                
            # Extract unique L values and p_c values from filenames
            if L_values is None:
                L_values = sorted(list(set([int(f.split('_L')[1].split('_')[0]) for f in all_files])))
            
            # Fix the p_c extraction logic for the correct filename format
            p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
            p_c_values = []
            for f in all_files:
                if f'_{p_scan_name}' in f:
                    parts = f.split(f'_{p_scan_name}')
                    print(parts)
                    if len(parts) > 1:
                        pc_part = parts[1].split('/')[0].split('_')[0]
                        try:
                            p_c_values.append(float(pc_part))
                        except ValueError:
                            print(f"Warning: Could not parse {p_scan_name} value from {f}")
            
            p_c_values = sorted(list(set(p_c_values)))
            
            if not p_c_values:
                print(f"Warning: Could not extract any {p_scan_name} values from filenames")
                return results_dict
            
            for threshold in missing_thresholds:
                print(f"\nComputing for threshold {threshold:.1e}")
                data_list = []
                
                for L in L_values:
                    for p_c in p_c_values:
                        filename = f'sv_comparison_L{L}_{p_fixed_name}{p_fixed:.3f}_pc{p_c}/final_results_L{L}.h5'
                        file_results = read_and_compute_tmi_from_compare_file(
                            filename, p_fixed_name, p_fixed, n, threshold
                        )
                        data_list.extend(file_results)
                
                if data_list:
                    # Create DataFrame and group observations
                    df = pd.DataFrame(data_list)
                    df_final = df.set_index(['p', 'L', 'implementation'])
                    
                    # Change directory to the result output directory
                    os.chdir(output_folder)
                    # Write results to CSV
                    write_tmi_compare_results_to_csv(df_final, p_fixed, p_fixed_name, threshold)
                    # Change back to the original directory
                    os.chdir(current_dir)
                    results_dict[threshold] = df_final
        else:
            print("Skipping computation for missing thresholds.")

    # Make sure we are in the original directory
    if os.path.basename(os.getcwd()) != current_dir:
        os.chdir(current_dir)

    return results_dict

def plot_tmi_comparison(results_dict, p_fixed, p_fixed_name, threshold, L_values=None, p_c=None, 
                        ylim=None, figsize=(10, 6), save_fig=True, output_dir=None):
    """
    Plot TMI comparison between Tao's and Haining's implementations.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with threshold values as keys and DataFrames as values
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    threshold : float
        Threshold value to plot
    L_values : list, optional
        List of L values to include
    p_c : float, optional
        Critical point to mark with vertical line
    ylim : tuple, optional
        Y-axis limits
    figsize : tuple, optional
        Figure size
    save_fig : bool, optional
        Whether to save the figure
    output_dir : str, optional
        Directory to save the figure
        
    Returns:
    --------
    tuple
        (fig, ax) tuple
    """
    if threshold not in results_dict:
        raise ValueError(f"Threshold {threshold} not found in results_dict")
    
    df = results_dict[threshold]
    
    # Filter by L values if provided
    if L_values is not None:
        df = df.loc[df.index.get_level_values('L').isin(L_values)]
    
    # Get unique L values and sort them
    unique_L = sorted(df.index.get_level_values('L').unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define markers and colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    colors_tao = plt.cm.Blues(np.linspace(0.5, 1.0, len(unique_L)))
    colors_haining = plt.cm.Reds(np.linspace(0.5, 1.0, len(unique_L)))
    
    # Plot data for each L value
    for i, L in enumerate(unique_L):
        # Get data for Tao's implementation
        tao_data = df.loc[(slice(None), L, 'tao'), :]
        p_values = tao_data.index.get_level_values('p')
        tao_means = [np.mean(obs) for obs in tao_data['observations']]
        tao_stds = [np.std(obs) / np.sqrt(len(obs)) for obs in tao_data['observations']]
        
        # Get data for Haining's implementation
        haining_data = df.loc[(slice(None), L, 'haining'), :]
        haining_means = [np.mean(obs) for obs in haining_data['observations']]
        haining_stds = [np.std(obs) / np.sqrt(len(obs)) for obs in haining_data['observations']]
        
        # Plot with error bars
        ax.errorbar(p_values, tao_means, yerr=tao_stds, 
                   marker=markers[i % len(markers)], linestyle='-', 
                   color=colors_tao[i], label=f'Tao L={L}')
        
        ax.errorbar(p_values, haining_means, yerr=haining_stds, 
                   marker=markers[i % len(markers)], linestyle='--', 
                   color=colors_haining[i], label=f'Haining L={L}')
    
    # Add critical point line if provided
    if p_c is not None:
        ax.axvline(x=p_c, color='k', linestyle='--', alpha=0.5, label=f'p_c = {p_c:.3f}')
    
    # Set axis labels and title
    p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
    ax.set_xlabel(f'{p_scan_name}')
    ax.set_ylabel('Tripartite Mutual Information (TMI)')
    ax.set_title(f'TMI Comparison ({p_fixed_name}={p_fixed:.3f}, threshold={threshold:.1e})')
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if output_dir is None:
            output_dir = 'tmi_compare_plots'
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f'tmi_compare_{p_fixed_name}{p_fixed:.3f}_threshold{threshold:.1e}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
    
    return fig, ax

if __name__ == "__main__":
    # Example usage
    thresholds = [1.0e-15]
    p_fixed = 0.4
    p_fixed_name = 'pctrl'
    
    # Read or compute TMI values
    results = read_tmi_compare_results(
        p_fixed=p_fixed, 
        p_fixed_name=p_fixed_name, 
        thresholds=thresholds
    )
    
    # Plot comparison
    if thresholds[0] in results:
        fig, ax = plot_tmi_comparison(
            results_dict=results,
            p_fixed=p_fixed,
            p_fixed_name=p_fixed_name,
            threshold=thresholds[0],
            p_c=0.25,  # Example critical point
            save_fig=True
        )
        plt.show()
    else:
        print(f"No results found for threshold {thresholds[0]}")
