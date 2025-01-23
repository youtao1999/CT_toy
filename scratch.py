import pandas as pd
from plot_tmi_results import compute_tmi_from_singular_values
import os
import h5py
import csv

def read_tmi_results_to_df(p_fixed, p_fixed_name, L_values=None, n=0, threshold=1e-10):
    """
    Read singular values from HDF5 files, compute TMI statistics, and return results as a DataFrame.
    Scans for all available data files with different p_c values and combines them into one CSV.
    
    Parameters:
    -----------
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    L_values : list, optional
        List of L values to process. If None, discovers all available L values.
    n : int, optional
        Parameter for TMI computation
    threshold : float, optional
        Threshold for TMI computation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with MultiIndex (p, L) containing TMI observations
    """
    import glob
    # Find all relevant data files using glob
    file_pattern = f'tmi_fine_L*_{p_fixed_name}{p_fixed:.3f}_pc*'
    all_files = glob.glob(file_pattern)
    
    # Extract unique L values and p_c values from filenames
    if L_values is None:
        L_values = sorted(list(set([int(f.split('_')[2][1:]) for f in all_files])))
    p_c_values = sorted(list(set([float(f.split('_pc')[1]) for f in all_files])))
    
    output_filename = f'tmi_results_combined_{p_fixed_name}{p_fixed:.3f}.csv'
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
            writer.writerow(['L', p_fixed_name, p_scan_name + '_index', p_scan_name + '_value', 'p_c', 'sample_idx', 'tmi_value'])
            
            for L in L_values:
                for p_c in p_c_values:
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
                                writer.writerow([L, p_fixed_key, p_scan_idx, p_scan_values[p_scan_idx], p_c, sample_idx, tmi_value])
                                data_list.append({
                                    'p': p_scan_values[p_scan_idx],
                                    'L': L,
                                    'p_c': p_c,
                                    'observations': tmi_value
                                })
    

    # Create DataFrame and group observations
    df = pd.DataFrame(data_list)
    df_grouped = df.groupby(['p', 'L'])['observations'].apply(list).reset_index()
    df_final = df_grouped.set_index(['p', 'L'])
    
    return df_final

def combine_csv_files(p_fixed, p_fixed_name):
    import glob
    file_pattern = f'tmi_results_*_{p_fixed_name}{p_fixed:.3f}*.csv'
    all_files = glob.glob(file_pattern)
    print(all_files)
    # Filter out the combined files from the list
    source_files = [f for f in all_files if not f.startswith('tmi_results_combined_')]
    combined_files = [f for f in all_files if f.startswith('tmi_results_combined_')]
    
    if not source_files:
        print("No source files found to combine")
        return None
        
    # Read all source files into a list of dataframes
    dfs = []
    for file in source_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # If a combined file exists, append to it, otherwise create new
    if combined_files:
        existing_df = pd.read_csv(combined_files[0])
        final_df = pd.concat([existing_df, combined_df], ignore_index=True)
        output_file = combined_files[0]
    else:
        final_df = combined_df
        output_file = f'tmi_results_combined_{p_fixed_name}{p_fixed:.3f}.csv'
    
    # Write the combined data
    final_df.to_csv(output_file, index=False)
    print(f"Combined data written to {output_file}")
    return None
# Usage
df = read_tmi_results_to_df(p_fixed=0.643, p_fixed_name='pproj', L_values = [8, 12, 16, 20], threshold=1e-10)
# combine_csv_files(p_fixed=0.643, p_fixed_name='pproj')