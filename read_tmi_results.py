import pandas as pd
from plot_tmi_results import compute_tmi_from_singular_values
import os
import numpy as np
import h5py
import csv
import matplotlib.pyplot as plt
from FSS.DataCollapse import DataCollapse

def write_tmi_results_to_csv(df_final, p_fixed, p_fixed_name, current_threshold, output_filename=None):
    """
    Write TMI results to a CSV file.
    
    Parameters:
    -----------
    df_final : pd.DataFrame
        DataFrame with MultiIndex (p, L) containing TMI observations
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    current_threshold : float
        Current threshold value for TMI computation
    
    Returns:
    --------
    str
        Name of the output file
    """
    if output_filename is None:
        output_filename = f'tmi_results_combined_{p_fixed_name}{p_fixed:.3f}_threshold{current_threshold:.1e}.csv'
    else:
        output_filename = output_filename

    # Prepare data for CSV
    csv_data = []
    for (p, L), row in df_final.iterrows():
        for tmi_value in row['observations']:
            csv_data.append({
                'pctrl': p if p_fixed_name == 'pproj' else p_fixed,
                'pproj': p if p_fixed_name == 'pctrl' else p_fixed,
                'L': L,
                'tmi': tmi_value
            })
    
    # Write to CSV
    pd.DataFrame(csv_data).to_csv(output_filename, index=False)
    print(f"Wrote results to {output_filename}")
    return output_filename

def read_and_compute_tmi_from_file(filename, p_fixed_name, p_fixed, n, current_threshold):
    """Read singular values from an HDF5 file and compute TMI values."""
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found!")
        return []
    
    data_list = []
    
    with h5py.File(filename, 'r') as f:
        p_fixed_key = f"{p_fixed_name}{p_fixed:.3f}"
        p_fixed_group = f[p_fixed_key]
        p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
        p_scan_values = p_fixed_group[p_scan_name][:]
        sv_group = p_fixed_group['singular_values']
        
        num_p_scan = len(p_fixed_group[p_scan_name])
        L = int(filename.split('_L')[-1].split('.')[0])
        
        for p_scan_idx in range(num_p_scan):
            num_samples = sv_group[list(sv_group.keys())[0]].shape[1]
            singular_values = [{
                key: sv_group[key][p_scan_idx, sample_idx] 
                for key in sv_group.keys()
            } for sample_idx in range(num_samples)]
            
            tmi_values = [compute_tmi_from_singular_values(sv, n, current_threshold) 
                         for sv in singular_values]
            
            data_list.append({
                'p': p_scan_values[p_scan_idx],
                'L': L,
                'observations': tmi_values
            })
    
    return data_list

def read_tmi_results(p_fixed, p_fixed_name, thresholds=None, L_values=None, n=0, output_folder = "tmi_results_combined"):
    """Read TMI results, combining existing CSV data and optionally computing missing data."""
    import glob
    
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

    # check to see if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # check to make sure that we are in the correct directory
    if os.path.basename(os.getcwd()) != output_folder:
        raise RuntimeError(f"Could not find {output_folder} directory")
    
    if thresholds is None:

        def extract_threshold(filename):
            """Extract threshold value from filename."""
            return float(filename.split('_threshold')[1].split('.csv')[0])

        file_pattern = f'{output_folder}_{p_fixed_name}{p_fixed:.3f}_threshold*.csv'
        all_files = glob.glob(file_pattern)
        thresholds = [extract_threshold(f) for f in all_files]
    else:
        thresholds = thresholds



    # Check existing CSV files for each threshold
    for threshold in thresholds:
        filename = f'{output_folder}_{p_fixed_name}{p_fixed:.3f}_threshold{threshold:.1e}.csv'
        if os.path.exists(filename):
            print(f"Found existing data for threshold {threshold:.1e}")
            # Read and process CSV file
            df = pd.read_csv(filename)
            if L_values is not None:
                df = df[df['L'].isin(L_values)]
            
            # Group by p and L to collect all TMI values as observations
            p_col = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
            df['p'] = df[p_col]  # Create 'p' column
            df_grouped = df.groupby(['p', 'L'])['tmi'].apply(list).reset_index()
            df_final = df_grouped.rename(columns={'tmi': 'observations'}).set_index(['p', 'L'])
            
            results_dict[threshold] = df_final
        else:
            missing_thresholds.append(threshold)
    
    if missing_thresholds:
        # change directory back to CT_toy
        os.chdir(current_dir)

        print(f"\nMissing data for {len(missing_thresholds)} threshold values:")
        print(f"Thresholds: {[f'{t:.1e}' for t in missing_thresholds]}")
        compute = input("\nWould you like to compute these from H5 files? (yes/no): ").lower().strip()
        
        if compute in ['y', 'yes']:
            # Find all relevant HDF5 files - update pattern from tmi_fine to sv_fine
            file_pattern = f'sv_fine_L*_{p_fixed_name}{p_fixed:.3f}_pc*'
            all_files = glob.glob(file_pattern)
            
            if not all_files:
                print(f"No HDF5 files found matching pattern: {file_pattern}")
                return results_dict
                
            # Extract unique L values and p_c values from filenames
            if L_values is None:
                L_values = sorted(list(set([int(f.split('_')[2][1:]) for f in all_files])))
            p_c_values = sorted(list(set([float(f.split('_pc')[1]) for f in all_files])))
            
            for threshold in missing_thresholds:
                print(f"\nComputing for threshold {threshold:.1e}")
                data_list = []
                
                for L in L_values:
                    for p_c in p_c_values:
                        # Update filename pattern from tmi_fine to sv_fine
                        filename = f'sv_fine_L{L}_{p_fixed_name}{p_fixed:.3f}_pc{p_c}/final_results_L{L}.h5'
                        file_results = read_and_compute_tmi_from_file(
                            filename, p_fixed_name, p_fixed, n, threshold
                        )
                        data_list.extend(file_results)
                
                if data_list:
                    # Create DataFrame and group observations
                    df = pd.DataFrame(data_list)
                    df_final = df.set_index(['p', 'L'])
                    
                    # change directory to the result output directory tmi_results_combined
                    os.chdir(output_folder)
                    # Write results to CSV
                    write_tmi_results_to_csv(df_final, p_fixed, p_fixed_name, threshold)
                    # change back to the original directory
                    os.chdir(current_dir)
                    results_dict[threshold] = df_final
        else:
            print("Skipping computation for missing thresholds.")

    # make sure that we are in the original directory
    if os.path.basename(os.getcwd()) != current_dir:
        os.chdir(current_dir)

    return results_dict

def combine_csv_files(p_fixed, p_fixed_name):
    import glob
    file_pattern = f'sv_results_*_{p_fixed_name}{p_fixed:.3f}*.csv'
    all_files = glob.glob(file_pattern)
    print(all_files)
    # Filter out the combined files from the list
    source_files = [f for f in all_files if not f.startswith('sv_results_combined_')]
    combined_files = [f for f in all_files if f.startswith('sv_results_combined_')]

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
        output_file = f'sv_results_combined_{p_fixed_name}{p_fixed:.3f}.csv'
    
    # Write the combined data
    final_df.to_csv(output_file, index=False)
    print(f"Combined data written to {output_file}")
    return None 

def extract_threshold(filename):
    """Extract threshold value from filename."""
    return float(filename.split('_threshold')[1].split('.csv')[0])

def bootstrap_data_collapse(df, n_samples, sample_size, p_c=0.473, nu=0.7, L_min=None, L_max=None, p_range=None, seed=None, nu_vary=True, p_c_vary=True):
    """
    Perform bootstrapped data collapse analysis on TMI data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with MultiIndex (p, L) containing TMI observations
    n_samples : int
        Number of bootstrap samples to generate
    sample_size : int
        Size of each bootstrap sample
    p_c : float, optional
        Initial guess for critical point
    nu : float, optional
        Initial guess for critical exponent
    L_min : int, optional
        Minimum system size to include
    L_max : int, optional
        Maximum system size to include
    p_range : tuple, optional
        (min, max) range of p values to include
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Contains:
        - 'nu_mean': average critical exponent
        - 'nu_std': standard deviation of critical exponent
        - 'pc_mean': average critical point
        - 'pc_std': standard deviation of critical point
        - 'redchi_mean': average reduced chi-squared
        - 'redchi_std': standard deviation of reduced chi-squared
        - 'samples': list of individual sample results
    """
    import numpy as np
    from FSS.DataCollapse import DataCollapse
    
    rng = np.random.default_rng(seed)
    results = []
    
    # Default p_range if not provided
    if p_range is None:
        p_min = df.index.get_level_values('p').min()
        p_max = df.index.get_level_values('p').max()
        p_range = [p_min, p_max]
    
    # Perform bootstrap sampling
    for i in range(n_samples):
        # Create resampled DataFrame
        resampled_data = []
        
        # Resample for each (p, L) pair
        for idx in df.index:
            p, L = idx
            observations = df.loc[idx, 'observations']
            
            # Ensure observations is a list
            if not isinstance(observations, list):
                observations = [observations]
            
            # Random sampling with replacement
            sampled_obs = rng.choice(observations, 
                                   size=min(sample_size, len(observations)), 
                                   replace=True)
            
            resampled_data.append({
                'p': p,
                'L': L,
                'observations': list(sampled_obs)
            })
        
        # Create new DataFrame from resampled data
        resampled_df = pd.DataFrame(resampled_data)
        resampled_df = resampled_df.set_index(['p', 'L'])
        
        # Perform data collapse
        dc = DataCollapse(df=resampled_df, 
                         p_='p', 
                         L_='L',
                         params={},
                         p_range=p_range,
                         Lmin=L_min,
                         Lmax=L_max)
        
        res = dc.datacollapse(p_c=p_c, 
                            nu=nu, 
                            beta=0.0,
                            p_c_vary=p_c_vary,
                            nu_vary=nu_vary,
                            beta_vary=False)
        
        # Add debugging information
        if res.params['nu'].stderr is None:
            print(f"Warning: Sample {i+1} - stderr is None")
            print(f"Fit success: {res.success}")
            print(f"Fit message: {res.message}")
            
        # Store results with fallback for None stderr values
        results.append({
            'nu': res.params['nu'].value,
            'nu_stderr': res.params['nu'].stderr if res.params['nu'].stderr is not None else 0.0,
            'pc': res.params['p_c'].value,
            'pc_stderr': res.params['p_c'].stderr if res.params['p_c'].stderr is not None else 0.0,
            'redchi': res.redchi
        })
        
        # print(f"Sample {i+1}/{n_samples}: nu = {results[-1]['nu']:.3f} ± {results[-1]['nu_stderr']:.3f}, "
        #       f"p_c = {results[-1]['pc']:.3f} ± {results[-1]['pc_stderr']:.3f}")
    
    # Calculate final results
    nu_values = [r['nu'] for r in results]
    pc_values = [r['pc'] for r in results]
    redchi_values = [r['redchi'] for r in results]
    
    # Calculate mean values
    nu_mean = np.mean(nu_values)
    pc_mean = np.mean(pc_values)
    redchi_mean = np.mean(redchi_values)
    
    # Calculate total uncertainties (combining bootstrap spread and fit uncertainties)

    nu_std = np.sqrt(np.std(nu_values)**2 + np.mean([r['nu_stderr']**2 for r in results]))
    pc_std = np.sqrt(np.std(pc_values)**2 + np.mean([r['pc_stderr']**2 for r in results]))
    redchi_std = np.std(redchi_values)
    
    return {
        'nu_mean': nu_mean,
        'nu_std': nu_std,
        'pc_mean': pc_mean,
        'pc_std': pc_std,
        'redchi_mean': redchi_mean,
        'redchi_std': redchi_std,
        'samples': results
    }

def plot_loss_manifold(df, pc_range, nu_range, n_points=50, pc = 0.473, delta_p = 0.05, L_min = 12):
    """
    Visualize the loss function manifold for different values of pc and nu
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with MultiIndex (p, L) containing TMI observations
    pc_range : tuple
        (min_pc, max_pc) range to explore
    nu_range : tuple
        (min_nu, max_nu) range to explore
    n_points : int
        Number of points to sample in each dimension
    """    
    # Initialize parameters
    p_range = [pc - delta_p, pc + delta_p]

    # Create a DataCollapse object from the DataFrame
    dc = DataCollapse(df, 
                     p_='p', 
                     L_='L',
                     params={},
                     p_range=p_range,  # Now using appropriate p_range
                     Lmin=L_min)

    
    # Create meshgrid of pc and nu values
    pc_vals = np.linspace(pc_range[0], pc_range[1], n_points)
    nu_vals = np.linspace(nu_range[0], nu_range[1], n_points)
    PC, NU = np.meshgrid(pc_vals, nu_vals)
    
    # Calculate loss for each point
    Z = np.zeros_like(PC)
    for i in range(n_points):
        for j in range(n_points):
            loss_vals = dc.loss(PC[i,j], NU[i,j], beta=0)
            Z[i,j] = np.sum(loss_vals**2) / (len(loss_vals) - 2)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Contour plot
    cont = ax1.contour(PC, NU, Z, levels=20)
    ax1.clabel(cont, inline=True, fontsize=8)
    ax1.set_xlabel('p_c')
    ax1.set_ylabel('nu')
    ax1.set_title('Loss Function Contours')
    
    # 3D surface plot
    surf = ax2.pcolormesh(PC, NU, np.log10(Z), shading='auto')
    ax2.set_xlabel('p_c')
    ax2.set_ylabel('nu')
    ax2.set_title('Log10 Loss Function (Color Map)')
    plt.colorbar(surf, ax=ax2)
    
    plt.tight_layout()
    return fig, (ax1, ax2)

if __name__ == "__main__":  
    # thresholds = np.logspace(-19, -10, 20)
    thresholds = [1.0e-15]
    results = read_tmi_results(p_fixed=0.000, p_fixed_name='pproj', thresholds=thresholds)
    df = results[thresholds[0]]
    
    fig, (ax1, ax2) = plot_loss_manifold(df, pc_range = (0., 1.), nu_range = (0.5, 1.5), pc = 0.45, delta_p = 0.1, n_points=200)
    plt.savefig('loss_manifold.png', dpi=300, bbox_inches='tight')

    # Example of bootstrapping analysis
    bootstrap_results = bootstrap_data_collapse(
        df=results[thresholds[0]],  # Use first threshold's data
        n_samples=100,
        sample_size=1000,
        p_c=0.45,
        nu=0.5,
        L_min=12,
        L_max=20
    )
    
    print("\nBootstrap Analysis Results:")
    print(f"nu = {bootstrap_results['nu_mean']:.3f} ± {bootstrap_results['nu_std']:.3f}")
    print(f"p_c = {bootstrap_results['pc_mean']:.3f} ± {bootstrap_results['pc_std']:.3f}")
    print(f"reduced chi^2 = {bootstrap_results['redchi_mean']:.3f} ± {bootstrap_results['redchi_std']:.3f}")
    
    