import pandas as pd
import os
import numpy as np
from FSS.DataCollapse import DataCollapse
import h5py
import glob
import matplotlib.pyplot as plt
from plot_tmi_results import compute_tmi_from_singular_values
from tqdm import tqdm  # Import tqdm for progress bars

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
                'observations': tmi_value  # Changed from 'tmi' to 'observations'
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
            
            # Add progress bar for p_scan_values
            for p_scan_idx in tqdm(range(len(p_scan_values)), desc="Processing Tao's implementation", leave=False):
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
            
            # Add progress bar for p_scan_values
            for p_scan_idx in tqdm(range(len(p_scan_values)), desc="Processing Haining's implementation", leave=False):
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
            p_scan_name = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
            df['p'] = df[p_scan_name]  # Create 'p' column
            
            # Group by p, L, and implementation and convert observations to lists
            df_grouped = df.groupby(['p', 'L', 'implementation'])['observations'].apply(list).reset_index()
            
            # Set MultiIndex for compatibility with plotting functions
            df_final = df_grouped.set_index(['p', 'L', 'implementation'])
            
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
            file_pattern = f'sv_comparison_L*_{p_fixed_name}{p_fixed:.3f}_p*'
            all_files = glob.glob(file_pattern)
            print(f"Found {len(all_files)} files matching pattern: {file_pattern}")

            if not all_files:
                print(f"No HDF5 files found matching pattern: {file_pattern}")
                return results_dict
            
            # Extract unique L values from filenames
            if L_values is None:
                L_values = sorted(list(set([int(f.split('_L')[1].split('_')[0]) for f in all_files])))

            # Add progress bar for thresholds
            for threshold in tqdm(missing_thresholds, desc="Processing thresholds"):
                print(f"\nComputing for threshold {threshold:.1e}")
                data_list = []
                
                # Add progress bar for L values
                for L in tqdm(L_values, desc=f"Processing L values for threshold {threshold:.1e}"):
                    # Find all files for this L value
                    L_files = [f for f in all_files if f'_L{L}_' in f]
                    
                    # Add progress bar for files
                    for file_path in tqdm(L_files, desc=f"Processing files for L={L}", leave=False):

                        # Process the file
                        filename = os.path.join(file_path, f'final_results_L{L}.h5')
                        if os.path.exists(filename):
                            file_results = read_and_compute_tmi_from_compare_file(
                                filename, p_fixed_name, p_fixed, n, threshold
                            )
                            data_list.extend(file_results)
                        else:
                            print(f"Warning: File {filename} not found!")
                
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
    
    # Ensure df has a MultiIndex with p, L, implementation
    if not isinstance(df.index, pd.MultiIndex) or 'L' not in df.index.names:
        # Try to convert to MultiIndex if possible
        if 'p' in df.columns and 'L' in df.columns and 'implementation' in df.columns:
            df = df.set_index(['p', 'L', 'implementation'])
        else:
            raise ValueError("Cannot plot: DataFrame does not have required columns (p, L, implementation)")
    
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

def plot_compare_loss_manifold(df, pc_range, nu_range, n_points=50, pc=0.473, delta_p=0.05, L_min=12, 
                              implementations=None, figsize=(15, 6), save_fig=True, output_dir=None):
    """
    Visualize and compare the loss function manifold for different implementations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with MultiIndex (p, L, implementation) containing TMI observations
    pc_range : tuple
        (min_pc, max_pc) range to explore
    nu_range : tuple
        (min_nu, max_nu) range to explore
    n_points : int
        Number of points to sample in each dimension
    pc : float
        Center point for p_range
    delta_p : float
        Range around pc to include in analysis
    L_min : int
        Minimum system size to include
    implementations : list, optional
        List of implementations to compare (default: all in df)
    figsize : tuple
        Figure size
    save_fig : bool
        Whether to save the figure
    output_dir : str, optional
        Directory to save the figure
        
    Returns:
    --------
    tuple
        (fig, axes) tuple
    """
    
    # Set p_range around pc
    p_range = [pc - delta_p, pc + delta_p]
    
    # Get unique implementations if not specified
    if implementations is None:
        implementations = df.index.get_level_values('implementation').unique()
    
    # Create figure with subplots for each implementation
    fig, axes = plt.subplots(len(implementations), 2, figsize=figsize)
    if len(implementations) == 1:
        axes = [axes]  # Make axes indexable for single implementation
    
    # Create meshgrid of pc and nu values
    pc_vals = np.linspace(pc_range[0], pc_range[1], n_points)
    nu_vals = np.linspace(nu_range[0], nu_range[1], n_points)
    PC, NU = np.meshgrid(pc_vals, nu_vals)
    
    # Process each implementation
    for i, impl in enumerate(implementations):
        # Filter data for this implementation
        impl_df = df.xs(impl, level='implementation')
        
        # Create a DataCollapse object
        dc = DataCollapse(impl_df, 
                         p_='p', 
                         L_='L',
                         params={},
                         p_range=p_range,
                         Lmin=L_min)
        
        # Calculate loss for each point
        Z = np.zeros_like(PC)
        for j in range(n_points):
            for k in range(n_points):
                loss_vals = dc.loss(PC[j,k], NU[j,k], beta=0)
                Z[j,k] = np.sum(loss_vals**2) / (len(loss_vals) - 2)
        
        # Contour plot
        cont = axes[i][0].contour(PC, NU, Z, levels=20)
        axes[i][0].clabel(cont, inline=True, fontsize=8)
        axes[i][0].set_xlabel('p_c')
        axes[i][0].set_ylabel('nu')
        axes[i][0].set_title(f'{impl.capitalize()} Implementation - Loss Contours')
        
        # Color map plot
        surf = axes[i][1].pcolormesh(PC, NU, np.log10(Z), shading='auto')
        axes[i][1].set_xlabel('p_c')
        axes[i][1].set_ylabel('nu')
        axes[i][1].set_title(f'{impl.capitalize()} Implementation - Log10 Loss')
        plt.colorbar(surf, ax=axes[i][1])
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if output_dir is None:
            output_dir = 'tmi_compare_plots'
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f'loss_manifold_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
    
    return fig, axes

def compare_bootstrap_analysis(df, n_samples=100, sample_size=1000, p_c=0.473, nu=0.7, 
                              L_min=None, L_max=None, p_range=None, seed=None, 
                              implementations=None, nu_vary=True, p_c_vary=True):
    """
    Perform and compare bootstrapped data collapse analysis for different implementations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with MultiIndex (p, L, implementation) containing TMI observations
    n_samples : int
        Number of bootstrap samples to generate
    sample_size : int
        Size of each bootstrap sample
    p_c : float
        Initial guess for critical point
    nu : float
        Initial guess for critical exponent
    L_min : int, optional
        Minimum system size to include
    L_max : int, optional
        Maximum system size to include
    p_range : tuple, optional
        (min, max) range of p values to include
    seed : int, optional
        Random seed for reproducibility
    implementations : list, optional
        List of implementations to compare (default: all in df)
    nu_vary : bool
        Whether to vary nu in the fit
    p_c_vary : bool
        Whether to vary p_c in the fit
    
    Returns:
    --------
    dict
        Dictionary with implementation names as keys and bootstrap results as values
    """
    import numpy as np
    
    # Get unique implementations if not specified
    if implementations is None:
        implementations = df.index.get_level_values('implementation').unique()
    
    results = {}
    
    # Process each implementation
    for impl in implementations:
        print(f"\nPerforming bootstrap analysis for {impl.capitalize()} implementation:")
        
        # Filter data for this implementation
        impl_df = df.xs(impl, level='implementation')
        
        # Perform bootstrap analysis
        bootstrap_results = bootstrap_data_collapse(
            df=impl_df,
            n_samples=n_samples,
            sample_size=sample_size,
            p_c=p_c,
            nu=nu,
            L_min=L_min,
            L_max=L_max,
            p_range=p_range,
            seed=seed,
            nu_vary=nu_vary,
            p_c_vary=p_c_vary
        )
        
        results[impl] = bootstrap_results
        
        # Print results
        print(f"  nu = {bootstrap_results['nu_mean']:.3f} ± {bootstrap_results['nu_std']:.3f}")
        print(f"  p_c = {bootstrap_results['pc_mean']:.3f} ± {bootstrap_results['pc_std']:.3f}")
        print(f"  reduced chi^2 = {bootstrap_results['redchi_mean']:.3f} ± {bootstrap_results['redchi_std']:.3f}")
    
    return results

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
    nu_vary : bool
        Whether to vary nu in the fit
    p_c_vary : bool
        Whether to vary p_c in the fit
    
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

def read_tmi_compare_results_from_csv(p_fixed, p_fixed_name, threshold, L_values=None):
    """
    Read TMI comparison results directly from CSV files.
    
    Parameters:
    -----------
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    threshold : float
        Threshold value for TMI computation
    L_values : list, optional
        List of L values to include
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with TMI results
    """
    # Join filename with directory name
    filename = os.path.join('tmi_compare_results', f'tmi_compare_results_{p_fixed_name}{p_fixed:.3f}_threshold{threshold:.1e}.csv')
    
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found!")
        return None
    
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Filter by L values if provided
    if L_values is not None:
        df = df[df['L'].isin(L_values)]
    
    # Ensure 'observations' column exists (for backward compatibility)
    if 'tmi' in df.columns and 'observations' not in df.columns:
        df['observations'] = df['tmi']
        df = df.drop('tmi', axis=1)
    
    return df

def compare_bootstrap_analysis_from_csv(p_fixed, p_fixed_name, threshold, 
                                       n_samples=100, sample_size=1000, 
                                       p_c=0.473, nu=0.7, L_min=None, L_max=None, 
                                       p_range=None, seed=None, implementations=None,
                                       nu_vary=True, p_c_vary=True, bootstrap=True):
    """
    Perform bootstrap analysis using data from CSV files.
    
    Parameters:
    -----------
    p_fixed : float
        Fixed parameter value
    p_fixed_name : str
        Name of fixed parameter ('pproj' or 'pctrl')
    threshold : float
        Threshold value for TMI computation
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
    implementations : list, optional
        List of implementations to compare (default: all in df)
    nu_vary : bool
        Whether to vary nu in the fit
    p_c_vary : bool
        Whether to vary p_c in the fit
    bootstrap : bool
        Whether to perform bootstrap analysis (True) or just a single data collapse (False)
    
    Returns:
    --------
    dict
        Dictionary with implementation names as keys and results as values
    """
    # Read data from CSV
    df = read_tmi_compare_results_from_csv(p_fixed, p_fixed_name, threshold)
    
    if df is None:
        print("No data available for analysis.")
        return None
    
    # Check if 'observations' column exists
    if 'observations' not in df.columns:
        print("Error: CSV file does not contain 'observations' column with TMI values")
        return None
    
    # Get unique implementations if not specified
    if implementations is None:
        implementations = df['implementation'].unique()
    
    results = {}
    
    # Process each implementation
    for impl in implementations:
        print(f"\nPerforming {'bootstrap ' if bootstrap else ''}analysis for {impl.capitalize()} implementation:")
        
        # Filter data for this implementation
        impl_df = df[df['implementation'] == impl].copy()
        
        # Filter by L values if needed
        if L_min is not None:
            impl_df = impl_df[impl_df['L'] >= L_min]
        if L_max is not None:
            impl_df = impl_df[impl_df['L'] <= L_max]
        
        # Set up for DataCollapse
        p_col = 'pctrl' if p_fixed_name == 'pproj' else 'pproj'
        impl_df['p'] = impl_df[p_col]  # Create 'p' column for DataCollapse
        
        # Determine p_range if not provided
        if p_range is None:
            p_min = impl_df['p'].min()
            p_max = impl_df['p'].max()
            local_p_range = [p_min, p_max]
        else:
            local_p_range = p_range
            
            # Filter by p range if needed
            impl_df = impl_df[(impl_df['p'] >= local_p_range[0]) & (impl_df['p'] <= local_p_range[1])]
        
        # Convert observations column from string to list of floats if needed
        if len(impl_df) > 0 and isinstance(impl_df['observations'].iloc[0], str):
            impl_df['observations'] = impl_df['observations'].apply(
                lambda x: [float(val) for val in x.strip('[]').split(',')]
            )
        
        # Group by (p, L) and collect all observations for each group
        grouped_data = []
        for (p, L), group in impl_df.groupby(['p', 'L']):
            # Combine all observations from this group
            all_obs = []
            for obs in group['observations']:
                if isinstance(obs, list):
                    all_obs.extend(obs)
                else:
                    all_obs.append(obs)
            
            grouped_data.append({
                'p': p,
                'L': L,
                'observations': all_obs
            })
        
        # Create new DataFrame for analysis
        analysis_df = pd.DataFrame(grouped_data)
        
        # Check if we have enough data
        if len(analysis_df) == 0:
            print(f"  Warning: No data available for {impl} implementation after filtering")
            continue
            
        analysis_df = analysis_df.set_index(['p', 'L'])
        
        if bootstrap:
            # Perform bootstrap analysis
            bootstrap_results = bootstrap_data_collapse(
                df=analysis_df,
                n_samples=n_samples,
                sample_size=sample_size,
                p_c=p_c,
                nu=nu,
                L_min=None,  # Already filtered
                L_max=None,  # Already filtered
                p_range=local_p_range,
                seed=seed,
                nu_vary=nu_vary,
                p_c_vary=p_c_vary
            )
            
            results[impl] = bootstrap_results
            
            # Print results
            print(f"  nu = {bootstrap_results['nu_mean']:.3f} ± {bootstrap_results['nu_std']:.3f}")
            print(f"  p_c = {bootstrap_results['pc_mean']:.3f} ± {bootstrap_results['pc_std']:.3f}")
            print(f"  reduced chi^2 = {bootstrap_results['redchi_mean']:.3f} ± {bootstrap_results['redchi_std']:.3f}")
        else:
            # Perform single data collapse on the entire dataset
            try:
                dc = DataCollapse(df=analysis_df, 
                                p_='p', 
                                L_='L',
                                params={},
                                p_range=local_p_range,
                                Lmin=L_min,
                                Lmax=L_max)
                
                res = dc.datacollapse(p_c=p_c, 
                                    nu=nu, 
                                    beta=0.0,
                                    p_c_vary=p_c_vary,
                                    nu_vary=nu_vary,
                                    beta_vary=False)
                
                # Store results
                single_results = {
                    'nu': res.params['nu'].value,
                    'nu_stderr': res.params['nu'].stderr if res.params['nu'].stderr is not None else 0.0,
                    'pc': res.params['p_c'].value,
                    'pc_stderr': res.params['p_c'].stderr if res.params['p_c'].stderr is not None else 0.0,
                    'redchi': res.redchi,
                    'result_object': res  # Store the full result object for further analysis if needed
                }
                
                results[impl] = single_results
                
                # Print results
                print(f"  nu = {single_results['nu']:.3f} ± {single_results['nu_stderr']:.3f}")
                print(f"  p_c = {single_results['pc']:.3f} ± {single_results['pc_stderr']:.3f}")
                print(f"  reduced chi^2 = {single_results['redchi']:.3f}")
            except Exception as e:
                print(f"  Error performing data collapse: {str(e)}")
                continue
    
    return results

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
    
    if thresholds[0] in results:
        # Plot comparison
        fig, ax = plot_tmi_comparison(
            results_dict=results,
            p_fixed=p_fixed,
            p_fixed_name=p_fixed_name,
            threshold=thresholds[0],
            p_c=0.75,  # Example critical point
            save_fig=True
        )
        
        # Plot loss manifold comparison
        fig_loss, axes_loss = plot_compare_loss_manifold(
            df=results[thresholds[0]],
            pc_range=(0.7, 0.8),
            nu_range=(0.3, 1.5),
            pc=0.75,
            delta_p=0.05,
            n_points=50,
            L_min=12,
            save_fig=True
        )
        
        # Perform bootstrap analysis comparison
        bootstrap_results = compare_bootstrap_analysis_from_csv(
            p_fixed=p_fixed,
            p_fixed_name=p_fixed_name,
            threshold=thresholds[0],
            n_samples=50,
            sample_size=100,
            p_c=0.75,
            nu=0.7,
            L_min=12,
            bootstrap=False
        )
        
        plt.show()
    else:
        print(f"No results found for threshold {thresholds[0]}")
