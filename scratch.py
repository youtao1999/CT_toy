import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_tmi_results import compute_tmi_from_singular_values
import os
import h5py
import csv
from scipy import stats
from FSS.DataCollapse import *

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

df = read_tmi_results_to_df(p_fixed = 0.500, p_fixed_name = 'pproj', p_c = 0.500, L_values = [8, 12, 16, 20])
dc=DataCollapse(df, p_='p',L_='L',params={},p_range=[0.45,0.55],)
dc.datacollapse(p_c=0.505,nu=1.3,beta=0.0,p_c_vary=True,beta_vary=True,nu_vary=True)
dc.plot_data_collapse()
