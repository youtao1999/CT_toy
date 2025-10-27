import pandas as pd
import os
import numpy as np
from FSS.DataCollapse import DataCollapse
import h5py
import glob
import matplotlib.pyplot as plt
from plot_tmi_results import compute_tmi_from_singular_values
from tqdm import tqdm

class TMIAnalyzer:
    """
    A class to analyze Tripartite Mutual Information (TMI) data.
    
    This class encapsulates functionality for reading, analyzing, and visualizing TMI data,
    including data collapse analysis and bootstrap analysis.
    """
    
    def __init__(self, pc_guess, nu_guess, p_fixed, p_fixed_name, threshold, output_folder="tmi_compare_results"):
        """
        Initialize TMIAnalyzer with basic parameters.
        
        Parameters:
        -----------
        p_fixed : float
            Fixed parameter value
        p_fixed_name : str
            Name of fixed parameter ('pproj' or 'pctrl')
        threshold : float
            Threshold value for TMI computation
        output_folder : str
            Folder to store output files
        """
        self.pc_guess = pc_guess
        self.nu_guess = nu_guess
        self.p_fixed = p_fixed
        self.p_fixed_name = p_fixed_name
        self.threshold = threshold
        self.output_folder = output_folder
        self.unscaled_df= None
        self.scaled_df= None
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Set working directory to CT_toy if needed
        self._set_working_directory()
    
    def _set_working_directory(self):
        """Set the working directory to CT_toy."""
        current_dir = os.getcwd()
        
        # If already in CT_toy, no need to change
        if os.path.basename(current_dir) == 'CT_toy':
            return
            
        # Try different possible locations
        possible_paths = [
            os.path.join(current_dir, 'CT_toy'),
            os.path.join(current_dir, 'code', 'CT_toy'),
            '/home/youtao/CT_toy'  # Absolute path as fallback
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                os.chdir(path)
                print(f"Changed working directory to: {path}")
                return
                
        raise RuntimeError(f"Could not find CT_toy directory. Searched in: {possible_paths}")
    
    def _get_csv_filename(self):
        """Get the standard CSV filename for the current parameters."""
        return os.path.join(
            self.output_folder,
            f'tmi_compare_results_{self.p_fixed_name}{self.p_fixed:.3f}_threshold{self.threshold:.1e}.csv'
        )
    
    def read_from_csv(self, L_values=None):
        """
        Read TMI comparison results from CSV file.
        
        Parameters:
        -----------
        L_values : list, optional
            List of L values to include
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with TMI results
        """
        filename = self._get_csv_filename()
        
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found!")
            return None
        
        df = pd.read_csv(filename)
        
        if L_values is None:
            L_values = df['L'].unique()
        else:
            df = df[df['L'].isin(L_values)]
        
        # Ensure 'observations' column exists
        if 'tmi' in df.columns and 'observations' not in df.columns:
            df['observations'] = df['tmi']
            df = df.drop('tmi', axis=1)
        
        # Group by p, L, and implementation to collect all TMI values as observations
        p_scan_name = 'pctrl' if self.p_fixed_name == 'pproj' else 'pproj'
        df['p'] = df[p_scan_name]  # Create 'p' column
        
        # Group observations by p, L, and implementation
        grouped_data = []
        for (p, L, impl), group in df.groupby(['p', 'L', 'implementation']):
            grouped_data.append({
                'p': p,
                'L': L,
                'implementation': impl,
                'observations': group['observations'].tolist()
            })
        
        # Create new DataFrame with MultiIndex
        df_final = pd.DataFrame(grouped_data)
        df_final = df_final.set_index(['p', 'L', 'implementation'])
        
        self.unscaled_df = df_final
        return df_final
    
    def read_and_compute_from_h5(self, n=0, L_values=None):
        """
        Read and compute TMI values from H5 files.
        
        Parameters:
        -----------
        n : int
            Renyi entropy parameter
        L_values : list, optional
            List of L values to include
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with computed TMI values
        """
        # Find all relevant HDF5 files - use a more general pattern to catch all p ranges
        base_pattern = f'sv_comparison_L*_{self.p_fixed_name}{self.p_fixed:.3f}_*'
        all_directories = glob.glob(base_pattern)
        
        if not all_directories:
            print(f"No HDF5 directories found matching pattern: {base_pattern}")
            return None
        
        # Extract unique L values from directory names if not provided
        if L_values is None:
            L_values = sorted(list(set([int(d.split('_L')[1].split('_')[0]) for d in all_directories])))
        
        data_list = []
        for L in tqdm(L_values, desc="Processing L values"):
            # Find all directories for this L value
            L_directories = [d for d in all_directories if f'_L{L}_' in d]
            print(f"\nFound {len(L_directories)} directories for L={L}:")
            for d in L_directories:
                print(f"  {d}")
            
            # Process each directory
            for directory in tqdm(L_directories, desc=f"Processing directories for L={L}", leave=False):
                filename = os.path.join(directory, f'final_results_L{L}.h5')
                if os.path.exists(filename):
                    file_results = self._read_single_h5_file(filename, n, L)
                    if file_results:  # Only extend if we got results
                        data_list.extend(file_results)
                else:
                    print(f"Warning: File {filename} not found!")
        
        if data_list:
            # Create DataFrame
            df = pd.DataFrame(data_list)
            
            # Sort by p value to ensure proper ordering
            df = df.sort_values(['L', 'implementation', 'p'])
            
            # Set index
            df_final = df.set_index(['p', 'L', 'implementation'])
            self.unscaled_df = df_final
            
            # Save to CSV
            self.save_to_csv()
            
            return df_final
        return None
    
    def _read_single_h5_file(self, filename, n, L):
        """Read and process a single H5 file."""
        data_list = []
        
        with h5py.File(filename, 'r') as f:
            p_fixed_key = f"{self.p_fixed_name}{self.p_fixed:.3f}"
            p_fixed_group = f[p_fixed_key]
            p_scan_name = 'pctrl' if self.p_fixed_name == 'pproj' else 'pproj'
            p_scan_values = p_fixed_group[p_scan_name][:]
            
            for impl in ['tao', 'haining']:
                if impl in p_fixed_group:
                    sv_group = p_fixed_group[impl]['singular_values']
                    
                    for p_scan_idx in range(len(p_scan_values)):
                        num_samples = sv_group[list(sv_group.keys())[0]].shape[1]
                        
                        singular_values = [{
                            key: sv_group[key][p_scan_idx, sample_idx] 
                            for key in sv_group.keys()
                        } for sample_idx in range(num_samples)]
                        
                        tmi_values = [compute_tmi_from_singular_values(sv, n, self.threshold) 
                                    for sv in singular_values]
                        
                        data_list.append({
                            'p': p_scan_values[p_scan_idx],
                            'L': L,
                            'implementation': impl,
                            'observations': tmi_values
                        })
        
        return data_list
    
    def save_to_csv(self):
        """Save current results to CSV file."""
        if self.unscaled_df is None:
            print("No results to save!")
            return
        
        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        csv_data = []
        for (p, L, implementation), row in self.unscaled_df.iterrows():
            for tmi_value in row['observations']:
                csv_data.append({
                    'pctrl': p if self.p_fixed_name == 'pproj' else self.p_fixed,
                    'pproj': p if self.p_fixed_name == 'pctrl' else self.p_fixed,
                    'L': L,
                    'implementation': implementation,
                    'observations': tmi_value
                })
        
        filename = self._get_csv_filename()
        pd.DataFrame(csv_data).to_csv(filename, index=False)
        print(f"Wrote results to {filename}")
    
    def plot_comparison(self, L_values=None, ylim=None, figsize=(15, 10), beta=0):
        """
        Plot comparison between unscaled raw data and scaled data based on data collapse.
        
        Parameters:
        -----------
        L_values : list, optional
            List of L values to include
        ylim : tuple, optional
            Y-axis limits as (raw_ylim, scaled_ylim)
        figsize : tuple
            Figure size
        beta : float
            Scaling dimension (usually 0 for TMI)
            
        Returns:
        --------
        tuple
            (fig, axes) tuple
        """
        if self.unscaled_df is None:
            print("No data available for plotting!")
            return None
        
        df = self.unscaled_df
        
        # Filter by L values if needed
        if L_values is not None:
            df = df.loc[df.index.get_level_values('L').isin(L_values)]
        
        # Get unique implementations and L values
        implementations = df.index.get_level_values('implementation').unique()
        unique_L = sorted(df.index.get_level_values('L').unique())
        
        # Create figure with subplots: one row per implementation, two columns (raw and scaled)
        fig, axes = plt.subplots(len(implementations), 2, figsize=figsize)
        
        # Define markers and colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_L)))
        
        # Process each implementation
        for i, impl in enumerate(implementations):
            # Get axes for this implementation
            if len(implementations) == 1:
                ax_raw = axes[0]
                ax_scaled = axes[1]
            else:
                ax_raw = axes[i, 0]
                ax_scaled = axes[i, 1]
            
            # Filter data for this implementation
            impl_df = df.xs(impl, level='implementation')
            
            # Plot raw and scaled data for each L
            for j, L in enumerate(unique_L):
                L_data = impl_df.loc[(slice(None), L), :]
                p_values = L_data.index.get_level_values('p')
                
                # Calculate statistics
                means = [np.mean(obs) for obs in L_data['observations']]
                errors = [np.std(obs) / np.sqrt(len(obs)) for obs in L_data['observations']]
                
                # Raw data plot
                ax_raw.errorbar(p_values, means, yerr=errors,
                              marker=markers[j % len(markers)], 
                              linestyle='-',
                              color=colors[j],
                              label=f'L = {L}')
                
                # Scaled data plot
                x_scaled = (p_values - self.pc_guess) * L**(1/self.nu_guess)
                y_scaled = np.array(means) * L**beta
                errors_scaled = np.array(errors) * L**beta
                
                ax_scaled.errorbar(x_scaled, y_scaled, yerr=errors_scaled,
                                 marker=markers[j % len(markers)], 
                                 linestyle='',
                                 color=colors[j],
                                 label=f'L = {L}')
            
            # Add vertical line at p_c in raw data plot
            ax_raw.axvline(x=self.pc_guess, color='k', linestyle='--', alpha=0.7,
                         label=f'p_c = {self.pc_guess:.3f}')
            
            # Set labels and titles
            ax_raw.set_xlabel('p')
            ax_raw.set_ylabel('TMI')
            ax_raw.set_title(f'Raw TMI Data - {impl.capitalize()}')
            ax_raw.legend(loc='best')
            ax_raw.grid(alpha=0.3)
            
            ax_scaled.set_xlabel(r'$(p - p_c) L^{1/\nu}$')
            ax_scaled.set_ylabel(r'$\mathrm{TMI} \cdot L^{\beta}$')
            ax_scaled.set_title(f'Scaled Data (p_c = {self.pc_guess:.3f}, ν = {self.nu_guess:.3f})')
            ax_scaled.legend(loc='best')
            ax_scaled.grid(alpha=0.3)
            
            # Set y-axis limits if provided
            if ylim is not None:
                if isinstance(ylim, tuple) and len(ylim) == 2:
                    ax_raw.set_ylim(ylim[0])
                    ax_scaled.set_ylim(ylim[1])
                else:
                    ax_raw.set_ylim(ylim)
                    ax_scaled.set_ylim(ylim)
        
        plt.tight_layout()
        
        # Note: We don't save this figure as it uses initial guesses rather than fitted parameters
        
        return fig, axes
    
    def perform_data_collapse(self, implementation, beta=0, L_min=None, L_max=None, p_range=None, 
                              nu_vary=True, p_c_vary=True):
        """
        Perform data collapse analysis for a specific implementation.
        
        Parameters:
        -----------
        implementation : str
            Name of the implementation to analyze
        beta : float
            Scaling dimension
        L_min : int, optional
            Minimum system size
        L_max : int, optional
            Maximum system size
        p_range : tuple, optional
            (min, max) range of p values
        nu_vary : bool
            Whether to vary nu in the fit
        p_c_vary : bool
            Whether to vary p_c in the fit
            
        Returns:
        --------
        tuple
            (fig, axes, result) containing the data collapse plots and fit result
        """
        if self.unscaled_df is None:
            print("No data available for analysis!")
            return None
        
        # Filter data for the specified implementation
        df = self.unscaled_df.xs(implementation, level='implementation')
        
        # Apply filters
        if L_min is not None or L_max is not None:
            L_values = df.index.get_level_values('L')
            mask = np.ones(len(df), dtype=bool)
            if L_min is not None:
                mask &= L_values >= L_min
            if L_max is not None:
                mask &= L_values <= L_max
            df = df.loc[mask]
        
        if p_range is None:
            p_min = df.index.get_level_values('p').min()
            p_max = df.index.get_level_values('p').max()
            p_range = [p_min, p_max]
        else:
            # Filter by p range
            p_values = df.index.get_level_values('p')
            mask = (p_values >= p_range[0]) & (p_values <= p_range[1])
            df = df.loc[mask]
        
        # Perform data collapse fit
        dc = DataCollapse(df=df, 
                         p_='p', 
                         L_='L',
                         params={},
                         p_range=p_range,
                         Lmin=L_min,
                         Lmax=L_max)
        
        try:
            result = dc.datacollapse(p_c=self.pc_guess, 
                                   nu=self.nu_guess, 
                                   beta=beta,
                                   p_c_vary=p_c_vary,
                                   nu_vary=nu_vary,
                                   beta_vary=False)
            
            # Extract fitted parameters
            fitted_pc = result.params['p_c'].value
            fitted_pc_err = result.params['p_c'].stderr if result.params['p_c'].stderr is not None else 0
            fitted_nu = result.params['nu'].value
            fitted_nu_err = result.params['nu'].stderr if result.params['nu'].stderr is not None else 0
            
            # Print results
            print(f"\nData Collapse Results for {implementation.capitalize()}:")
            print(f"  p_c = {fitted_pc:.5f} ± {fitted_pc_err:.5f}")
            print(f"  nu = {fitted_nu:.5f} ± {fitted_nu_err:.5f}")
            print(f"  reduced chi^2 = {result.redchi:.5f}")
            print(f"  Degrees of freedom: {result.nfree}")
            
            # Create figure with two subplots
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Get unique L values and colors
            unique_L = sorted(df.index.get_level_values('L').unique())
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_L)))
            
            # Plot raw and collapsed data with error bars for each L
            for i, L in enumerate(unique_L):
                L_data = df.loc[(slice(None), L), :]
                p_values = L_data.index.get_level_values('p')
                
                # Calculate mean and standard error for each point
                means = [np.mean(obs) for obs in L_data['observations']]
                stderrs = [np.std(obs) / np.sqrt(len(obs)) for obs in L_data['observations']]
                
                # Raw data plot with error bars
                axes[0].errorbar(p_values, means, yerr=stderrs,
                               marker='o', linestyle='-', color=colors[i],
                               label=f'L = {L}', capsize=3)
                
                # Collapsed data plot with error bars
                x_scaled = (p_values - fitted_pc) * L**(1/fitted_nu)
                y_scaled = np.array(means) * L**beta
                yerr_scaled = np.array(stderrs) * L**beta
                
                # Sort points by x_scaled for proper line connection
                sort_idx = np.argsort(x_scaled)
                x_scaled = x_scaled[sort_idx]
                y_scaled = y_scaled[sort_idx]
                yerr_scaled = yerr_scaled[sort_idx]
                
                axes[1].errorbar(x_scaled, y_scaled, yerr=yerr_scaled,
                               marker='o', linestyle='-', color=colors[i],
                               label=f'L = {L}', capsize=3)
            
            # Add vertical line at p_c in raw data plot
            axes[0].axvline(x=fitted_pc, color='k', linestyle='--', alpha=0.7,
                           label=f'p_c = {fitted_pc:.3f}')
            
            # Set labels and titles
            axes[0].set_xlabel('p')
            axes[0].set_ylabel('TMI')
            axes[0].set_title(f'Raw TMI Data - {implementation.capitalize()}')
            axes[0].legend(loc='best')
            axes[0].grid(alpha=0.3)
            
            axes[1].set_xlabel(r'$(p - p_c) L^{1/\nu}$')
            axes[1].set_ylabel(r'$\mathrm{TMI}$')
            axes[1].set_title(f'Data Collapse (p_c = {fitted_pc:.3f}, ν = {fitted_nu:.3f}, β = {beta:.3f})')
            axes[1].legend(loc='best')
            axes[1].grid(alpha=0.3)
            
            # Add uncertainty band for p_c
            pc_band_x = np.array([fitted_pc - fitted_pc_err, fitted_pc + fitted_pc_err])
            pc_band_y = axes[0].get_ylim()
            axes[0].fill_betweenx(pc_band_y, pc_band_x[0], pc_band_x[1], color='gray', alpha=0.2, label=f'p_c uncertainty')
            
            # Add uncertainty bands for the collapsed data
            for i, L in enumerate(unique_L):
                L_data = df.loc[(slice(None), L), :]
                p_values = L_data.index.get_level_values('p')
                means = [np.mean(obs) for obs in L_data['observations']]
                
                # Calculate x values for upper and lower bounds of uncertainty bands
                x_scaled_low = (p_values - (fitted_pc + fitted_pc_err)) * L**(1/(fitted_nu + fitted_nu_err))
                x_scaled_high = (p_values - (fitted_pc - fitted_pc_err)) * L**(1/(fitted_nu - fitted_nu_err))
                y_scaled = np.array(means) * L**beta
                
                # Plot uncertainty band
                x_band = np.vstack([x_scaled_low, x_scaled_high])
                y_band = np.vstack([y_scaled, y_scaled])
                axes[1].fill_between(x_band.mean(axis=0), y_band.min(axis=0), y_band.max(axis=0), 
                                   color=colors[i], alpha=0.1)
            
            plt.tight_layout()
            
            # Ensure output directory exists
            os.makedirs(self.output_folder, exist_ok=True)
            
            # Save figure
            fig_path = os.path.join(
                self.output_folder,
                f'data_collapse_{implementation}_{self.p_fixed_name}{self.p_fixed:.3f}_threshold{self.threshold:.1e}.png'
            )
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved data collapse figure to {fig_path}")
            
            return fig, axes, result
            
        except Exception as e:
            print(f"Error performing data collapse for {implementation}: {str(e)}")
            return None
    
    def bootstrap_data_collapse(self, implementation, n_samples=100, sample_size=1000, 
                               L_min=None, L_max=None, 
                               p_range=None, seed=None, nu_vary=True, p_c_vary=True):
        """
        Perform bootstrapped data collapse analysis on TMI data for a specific implementation.
        
        Parameters:
        -----------
        implementation : str
            Name of the implementation to analyze
        n_samples : int
            Number of bootstrap samples to generate
        sample_size : int
            Size of each bootstrap sample
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
        if self.unscaled_df is None:
            print("No data available for analysis!")
            return None
            
        # Filter data for the specified implementation
        df = self.unscaled_df.xs(implementation, level='implementation')
        
        # Apply filters
        if L_min is not None or L_max is not None:
            L_values = df.index.get_level_values('L')
            mask = np.ones(len(df), dtype=bool)
            if L_min is not None:
                mask &= L_values >= L_min
            if L_max is not None:
                mask &= L_values <= L_max
            df = df.loc[mask]
        
        if p_range is None:
            p_min = df.index.get_level_values('p').min()
            p_max = df.index.get_level_values('p').max()
            p_range = [p_min, p_max]
        else:
            # Filter by p range
            p_values = df.index.get_level_values('p')
            mask = (p_values >= p_range[0]) & (p_values <= p_range[1])
            df = df.loc[mask]
        
        # Set up random number generator
        rng = np.random.default_rng(seed)
        results = []
        
        # Perform bootstrap sampling
        for i in tqdm(range(n_samples), desc=f"Bootstrap samples for {implementation}"):
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
            
            try:
                res = dc.datacollapse(p_c=self.pc_guess, 
                                    nu=self.nu_guess, 
                                    beta=0.0,
                                    p_c_vary=p_c_vary,
                                    nu_vary=nu_vary,
                                    beta_vary=False)
                
                # Store results with fallback for None stderr values
                results.append({
                    'nu': res.params['nu'].value,
                    'nu_stderr': res.params['nu'].stderr if res.params['nu'].stderr is not None else 0.0,
                    'pc': res.params['p_c'].value,
                    'pc_stderr': res.params['p_c'].stderr if res.params['p_c'].stderr is not None else 0.0,
                    'redchi': res.redchi
                })
            except Exception as e:
                print(f"Warning: Bootstrap sample {i+1} failed with error: {str(e)}")
                continue
        
        if not results:
            print(f"Error: All bootstrap samples failed for {implementation}")
            return None
            
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
    
    def perform_bootstrap_analysis(self, n_samples=100, sample_size=1000, 
                                  L_min=None, L_max=None, 
                                  p_range=None, seed=None, 
                                  implementations=None, nu_vary=True, p_c_vary=True):
        """
        Perform and compare bootstrapped data collapse analysis for different implementations.
        
        Parameters:
        -----------
        n_samples : int
            Number of bootstrap samples to generate
        sample_size : int
            Size of each bootstrap sample
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
        if self.unscaled_df is None:
            print("No data available for analysis!")
            return None
        
        # Get unique implementations if not specified
        if implementations is None:
            implementations = self.unscaled_df.index.get_level_values('implementation').unique()
        
        results = {}
        
        # Process each implementation
        for impl in implementations:
            print(f"\nPerforming bootstrap analysis for {impl.capitalize()} implementation:")
            
            # Perform bootstrap analysis
            bootstrap_results = self.bootstrap_data_collapse(
                implementation=impl,
                n_samples=n_samples,
                sample_size=sample_size,
                L_min=L_min,
                L_max=L_max,
                p_range=p_range,
                seed=seed,
                nu_vary=nu_vary,
                p_c_vary=p_c_vary
            )
            
            if bootstrap_results is not None:
                results[impl] = bootstrap_results
                
                # Print results
                print(f"  nu = {bootstrap_results['nu_mean']:.3f} ± {bootstrap_results['nu_std']:.3f}")
                print(f"  p_c = {bootstrap_results['pc_mean']:.3f} ± {bootstrap_results['pc_std']:.3f}")
                print(f"  reduced chi^2 = {bootstrap_results['redchi_mean']:.3f} ± {bootstrap_results['redchi_std']:.3f}")
        
        return results

    def plot_compare_loss_manifold(self, p_range, nu_range=None, n_points=100, L_min=12, 
                                   implementations=None, figsize=(15, 6)):
        """
        Visualize and compare the loss function manifold for different implementations.
        
        Parameters:
        -----------
        p_range : tuple
            (min_p, max_p) range to explore
        nu_range : tuple, optional
            (min_nu, max_nu) range to explore. Default is (0.3, 1.5) if None
        n_points : int
            Number of points to sample in each dimension
        L_min : int
            Minimum system size to include
        implementations : list, optional
            List of implementations to compare (default: all in df)
        figsize : tuple
            Figure size
            
        Returns:
        --------
        tuple
            (fig, axes) tuple
        """
        # Add default nu_range if None
        if nu_range is None:
            print("nu_range was None, using default range (0.3, 1.5)")
            nu_range = (0.3, 1.5)
        
        # Convert string to tuple if necessary (e.g. from command line args)
        if isinstance(nu_range, str):
            try:
                nu_range = eval(nu_range)
                print(f"Converted string nu_range '{nu_range}' to tuple {nu_range}")
            except:
                print(f"Failed to convert string nu_range '{nu_range}', using default range")
                nu_range = (0.3, 1.5)
        
        if self.unscaled_df is None:
            print("No data available for analysis!")
            return None
        
        # Get unique implementations if not specified
        if implementations is None:
            implementations = self.unscaled_df.index.get_level_values('implementation').unique()
        
        # Create figure with subplots for each implementation
        fig, axes = plt.subplots(len(implementations), 2, figsize=figsize)
        if len(implementations) == 1:
            axes = [axes]  # Make axes indexable for single implementation
        
        # Create meshgrid of p and nu values
        p_vals = np.linspace(p_range[0], p_range[1], n_points)
        nu_vals = np.linspace(nu_range[0], nu_range[1], n_points)
        PC, NU = np.meshgrid(p_vals, nu_vals)
        
        # Process each implementation
        for i, impl in enumerate(implementations):
            # Filter data for this implementation
            impl_df = self.unscaled_df.xs(impl, level='implementation')
            
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
            
            # Find min and max values for better level spacing
            z_min = np.nanmin(Z)
            z_max = np.nanmax(Z)
            num_levels = 60  # Increased from 30 to 50 for more detail

            # Create contour levels with proper error handling
            try:
                # Try logarithmic spacing first (good for loss functions that vary by orders of magnitude)
                if z_min > 0:  # Ensure we don't take log of negative values
                    log_min = np.log10(max(z_min, 1e-10))  # Avoid very small values
                    log_max = np.log10(max(z_max, z_min*1.1))  # Ensure max > min
                    levels = np.logspace(log_min, log_max, num_levels)  # Logarithmic spacing

                                # Find and label all points with loss < 2
                    low_loss_mask = Z < 1.5
                    low_loss_points = np.where(low_loss_mask)
                    for point_idx in range(len(low_loss_points[0])):
                        j, k = low_loss_points[0][point_idx], low_loss_points[1][point_idx]
                        loss_val = Z[j,k]
                        p_val = PC[j,k]
                        nu_val = NU[j,k]
                        print(f"Low loss point found: p={p_val:.3f}, nu={nu_val:.3f}, loss={loss_val:.3f}")
                        # Add dot marker for low loss point
                        axes[i][0].plot(p_val, nu_val, 'o', color='blue', alpha=0.5, markersize=3)

                    # Add a horizontal line at nu = 1.5
                    axes[i][0].axhline(y=1.5, color='red', linestyle='--', alpha=0.7)
                    axes[i][0].axhline(y=self.nu_guess, color='blue', linestyle='--', alpha=0.7)
                    axes[i][0].axvline(x=self.pc_guess, color='green', linestyle='--', alpha=0.7)
                    # Add legend for reference lines
                    axes[i][0].plot([], [], '--', color='red', label='$\\nu$ = 1.5')
                    axes[i][0].plot([], [], '--', color='blue', label=f'$\\nu$ = {self.nu_guess:.3f}')
                    axes[i][0].plot([], [], '--', color='green', label=f'$p_c$ = {self.pc_guess:.3f}')
                    axes[i][0].legend(loc='upper right', fontsize=8)
                else:
                    # For linear spacing with custom step size
                    # Ensure z_max > z_min and step is reasonable
                    if np.isclose(z_max, z_min):
                        z_max = z_min * 1.1 + 1e-5  # Add a small increment if they're too close
                    
                    # Use linspace instead of arange for better numerical stability
                    levels = np.linspace(z_min, z_max, num_levels)
            except Exception as e:
                print(f"Warning: Error creating contour levels: {e}")
                # Fallback to default levels
                levels = num_levels
            
            # Contour plot with custom levels
            cont = axes[i][0].contour(PC, NU, Z, levels=levels, linewidths=0.5)
            axes[i][0].clabel(cont, inline=True, fontsize=6, fmt='%.1f')
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
        
        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Save figure
        fig_path = os.path.join(
            self.output_folder,
            f'loss_manifold_comparison_{self.p_fixed_name}{self.p_fixed:.3f}_threshold{self.threshold:.1e}.png'
        )
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved loss manifold figure to {fig_path}")
        
        return fig, axes
    
    def result(self, bootstrap=False, L_min=None, L_max=None, p_range=None, nu_range=None,
                 implementations=None, nu_vary=True, p_c_vary=True):
        '''
        Output the results of the data collapse analysis:
        This includes:
        - estimated p_c and nu for each implementation (with or without bootstrap)
        - saved figure of loss manifold for each implementation
        - saved figure of data collapse comparison for each implementation
        
        Parameters:
        -----------
        bootstrap : bool or str
            Whether to perform bootstrap analysis. Can be bool or str ('True'/'False')
        L_min : int, optional
            Minimum system size to include
        L_max : int, optional
            Maximum system size to include
        p_range : tuple, optional
            (min, max) range of p values to include
        nu_range : tuple, optional
            (min, max) range of nu values to include for loss manifold plots. Default is (0.3, 1.5)
        implementations : list, optional
            List of implementations to compare (default: all in df)
        nu_vary : bool
            Whether to vary nu in the fit
        p_c_vary : bool
            Whether to vary p_c in the fit
            
        Returns:
        --------
        dict
            Dictionary with implementation names as keys and results as values
        '''        
        # Convert string boolean to actual boolean if needed
        if isinstance(bootstrap, str):
            bootstrap = bootstrap.lower() == 'true'

        # Ensure nu_range has a value
        if nu_range is None:
            nu_range = (0.3, 1.5)
            print(f"nu_range was None, using default: {nu_range}")
        elif isinstance(nu_range, str):
            try:
                nu_range = eval(nu_range)
                print(f"Converted string nu_range to tuple: {nu_range}")
            except:
                print(f"Failed to convert string nu_range: {nu_range}, using default")
                nu_range = (0.3, 1.5)

        # Ensure data is loaded
        if self.unscaled_df is None:
            print("No data loaded yet, attempting to load from CSV...")
            self.unscaled_df = self.read_from_csv()
            
            if self.unscaled_df is None:
                print("No CSV data found, attempting to compute from H5 files...")
                self.read_and_compute_from_h5(n=0)
                
                if self.unscaled_df is None:
                    print("ERROR: Unable to load any data, cannot proceed with analysis.")
                    return None

        # Get unique implementations if not specified
        if implementations is None:
            implementations = self.unscaled_df.index.get_level_values('implementation').unique()
        
        # Determine p_range if not provided
        if p_range is None: # plot the loss manifold over the entire p range
            p_min = self.unscaled_df.index.get_level_values('p').min()
            p_max = self.unscaled_df.index.get_level_values('p').max()
            p_range = (p_min, p_max)
            print(f"Using p_range: ({p_min:.3f}, {p_max:.3f})")
        
        # Plot loss manifold to visualize the parameter space
        print("\nGenerating loss manifold plots...")
        fig_loss, axes_loss = self.plot_compare_loss_manifold(
            p_range=p_range,
            nu_range=nu_range,
            n_points=50,
            L_min=L_min,
            implementations=implementations
        )
        
        # Initialize results dictionary
        results = {}
        
        # Perform data collapse analysis
        if bootstrap:
            # # Get bootstrap parameters from user
            # n_samples = int(input("Enter the number of bootstrap samples (default: 100): ") or 100)
            # sample_size = int(input("Enter the sample size (default: 1000): ") or 1000)
            # seed = int(input("Enter random seed for reproducibility (default: 42): ") or 42)
            n_samples = 100
            sample_size = 1000
            seed = 42
            
            # Perform bootstrap data collapse analysis
            print("\nPerforming bootstrap data collapse analysis...")
            bootstrap_results = self.perform_bootstrap_analysis(
                n_samples=n_samples,
                sample_size=sample_size,
                L_min=L_min,
                L_max=L_max,
                p_range=p_range,
                seed=seed,
                implementations=implementations,
                nu_vary=nu_vary,
                p_c_vary=p_c_vary
            )
            
            # Store results
            for impl, res in bootstrap_results.items():
                results[impl] = {
                    'pc': res['pc_mean'],
                    'pc_err': res['pc_std'],
                    'nu': res['nu_mean'],
                    'nu_err': res['nu_std'],
                    'redchi': res['redchi_mean'],
                    'redchi_err': res['redchi_std'],
                    'method': 'bootstrap',
                    'bootstrap_samples': n_samples,
                    'sample_size': sample_size
                }
                
                # Generate data collapse plot using bootstrap results
                print(f"\nGenerating data collapse plot for {impl} using bootstrap results...")
                
                # Create figure with two subplots
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Filter data for this implementation
                impl_df = self.unscaled_df.xs(impl, level='implementation')
                
                # Apply filters
                if L_min is not None or L_max is not None:
                    L_values = impl_df.index.get_level_values('L')
                    mask = np.ones(len(impl_df), dtype=bool)
                    if L_min is not None:
                        mask &= L_values >= L_min
                    if L_max is not None:
                        mask &= L_values <= L_max
                    impl_df = impl_df.loc[mask]
                
                if p_range is not None:
                    p_values = impl_df.index.get_level_values('p')
                    mask = (p_values >= p_range[0]) & (p_values <= p_range[1])
                    impl_df = impl_df.loc[mask]
                
                # Get unique L values and colors
                unique_L = sorted(impl_df.index.get_level_values('L').unique())
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_L)))
                
                # Use bootstrap results for visualization
                fitted_pc = res['pc_mean']
                fitted_nu = res['nu_mean']
                beta = 0  # Usually 0 for TMI
                
                # Plot raw and collapsed data with error bars for each L
                for i, L in enumerate(unique_L):
                    L_data = impl_df.loc[(slice(None), L), :]
                    p_values = L_data.index.get_level_values('p')
                    
                    # Calculate mean and standard error for each point
                    means = [np.mean(obs) for obs in L_data['observations']]
                    stderrs = [np.std(obs) / np.sqrt(len(obs)) for obs in L_data['observations']]
                    
                    # Raw data plot with error bars
                    axes[0].errorbar(p_values, means, yerr=stderrs,
                                   marker='o', linestyle='-', color=colors[i],
                                   label=f'L = {L}', capsize=3)
                    
                    # Collapsed data plot with error bars
                    x_scaled = (p_values - fitted_pc) * L**(1/fitted_nu)
                    y_scaled = np.array(means) * L**beta
                    yerr_scaled = np.array(stderrs) * L**beta
                    
                    # Sort points by x_scaled for proper line connection
                    sort_idx = np.argsort(x_scaled)
                    x_scaled = x_scaled[sort_idx]
                    y_scaled = y_scaled[sort_idx]
                    yerr_scaled = yerr_scaled[sort_idx]
                    
                    axes[1].errorbar(x_scaled, y_scaled, yerr=yerr_scaled,
                                   marker='o', linestyle='-', color=colors[i],
                                   label=f'L = {L}', capsize=3)
                
                # Add vertical line at p_c in raw data plot
                axes[0].axvline(x=fitted_pc, color='k', linestyle='--', alpha=0.7,
                               label=f'p_c = {fitted_pc:.3f}')
                
                # Set labels and titles
                axes[0].set_xlabel('p')
                axes[0].set_ylabel('TMI')
                axes[0].set_title(f'Raw TMI Data - {impl.capitalize()}')
                axes[0].legend(loc='best')
                axes[0].grid(alpha=0.3)
                axes[0].set_xlim(p_range[0], p_range[1])
                
                axes[1].set_xlabel(r'$(p - p_c) L^{1/\nu}$')
                axes[1].set_ylabel(r'$\mathrm{TMI}$')
                axes[1].set_title(f'Bootstrap Data Collapse (p_c = {fitted_pc:.3f}, ν = {fitted_nu:.3f}, β = {beta:.3f})')
                axes[1].legend(loc='best')
                axes[1].grid(alpha=0.3)
                # Don't set fixed x-limits for the scaled plot - let matplotlib auto-scale
                plt.tight_layout()
                
                # Ensure output directory exists
                os.makedirs(self.output_folder, exist_ok=True)
                
                # Save figure
                fig_path = os.path.join(
                    self.output_folder,
                    f'bootstrap_collapse_{impl}_{self.p_fixed_name}{self.p_fixed:.3f}_threshold{self.threshold:.1e}.png'
                )
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Saved bootstrap data collapse figure to {fig_path}")
        else:
            # Perform regular data collapse for each implementation
            print("\nPerforming data collapse analysis...")
            for impl in implementations:
                result = self.perform_data_collapse(
                    implementation=impl,
                    beta=0,
                    L_min=L_min,
                    L_max=L_max,
                    p_range=p_range,
                    nu_vary=nu_vary,
                    p_c_vary=p_c_vary
                )
                
                if result:
                    fig, axes, fit_result = result
                    results[impl] = {
                        'pc': fit_result.params['p_c'].value,
                        'pc_err': fit_result.params['p_c'].stderr if fit_result.params['p_c'].stderr is not None else 0,
                        'nu': fit_result.params['nu'].value,
                        'nu_err': fit_result.params['nu'].stderr if fit_result.params['nu'].stderr is not None else 0,
                        'redchi': fit_result.redchi,
                        'method': 'direct fit',
                        'nfree': fit_result.nfree
                    }
        
        # Print summary of results
        if results:
            print("\nSummary of Data Collapse Results:")
            print("---------------------------------")
            for impl, res in results.items():
                print(f"{impl.capitalize()}:")
                print(f"  Method: {res['method']}")
                print(f"  p_c = {res['pc']:.5f} ± {res.get('pc_err', 0):.5f}")
                print(f"  nu = {res['nu']:.5f} ± {res.get('nu_err', 0):.5f}")
                print(f"  reduced chi^2 = {res['redchi']:.5f}")
                print()
            
            # Save results to CSV
            self._save_results_to_csv(results)
        
        return results
    
    def _save_results_to_csv(self, results):
        """Save analysis results to CSV file."""
        if not results:
            return
            
        # Ensure output directory exists
        os.makedirs(self.output_folder, exist_ok=True)
            
        # Prepare data for CSV
        csv_data = []
        for impl, res in results.items():
            row = {
                'implementation': impl,
                'method': res['method'],
                'p_fixed': self.p_fixed,
                'p_fixed_name': self.p_fixed_name,
                'threshold': self.threshold,
                'pc': res['pc'],
                'pc_err': res.get('pc_err', 0),
                'nu': res['nu'],
                'nu_err': res.get('nu_err', 0),
                'redchi': res['redchi']
            }
            
            # Add additional fields based on method
            if res['method'] == 'bootstrap':
                row['bootstrap_samples'] = res.get('bootstrap_samples', '')
                row['sample_size'] = res.get('sample_size', '')
                row['redchi_err'] = res.get('redchi_err', 0)
            else:
                row['nfree'] = res.get('nfree', '')
                
            csv_data.append(row)

        # Create DataFrame and save to CSV
        results_df = pd.DataFrame(csv_data)
        csv_filename = os.path.join(
            self.output_folder,
            f'data_collapse_results_{self.p_fixed_name}{self.p_fixed:.3f}_threshold{self.threshold:.1e}.csv'
        )
        results_df.to_csv(csv_filename, index=False)
        print(f"Saved results to {csv_filename}")

if __name__ == "__main__":
    # First run the analysis for different thresholds if needed
    run_analysis = False # Set to True to rerun the analysis
    pc_guess = 0.75
    delta_p = 0.2
    p_range = (pc_guess - delta_p, pc_guess + delta_p)
    nu_guess = 0.67
    delta_nu = 0.2
    nu_range = (nu_guess - delta_nu, nu_guess + delta_nu)
    p_fixed = 0.4
    p_fixed_name = 'pctrl'
    import glob
    folder = '/scratch/ty296/tmi_compare_results'
    data_collapse_file_name = f'data_collapse_results_{p_fixed_name}{p_fixed:.3f}_threshold*.csv'
    csv_file_name = f'/scratch/ty296/plots/tmi_compare_results_{p_fixed_name}{p_fixed:.3f}_threshold*.csv'
    data_collapse_files = glob.glob(os.path.join(folder, data_collapse_file_name))
    csv_files = glob.glob(os.path.join(folder, csv_file_name))
    if run_analysis:
        for file in csv_files:
            print(f"Processing file: {file}")
            threshold = float(file.split('threshold')[-1].split('.csv')[0].replace('e-', 'e-'))
            analyzer = TMIAnalyzer(pc_guess, nu_guess, p_fixed, p_fixed_name, threshold)
            results = analyzer.result(bootstrap=False, L_min=12, L_max=20, p_range=p_range, nu_range=nu_range)

    # Initialize data structures
    thresholds = []
    pc_values = {'tao': [], 'haining': []}
    pc_errors = {'tao': [], 'haining': []}
    nu_values = {'tao': [], 'haining': []}
    nu_errors = {'tao': [], 'haining': []}
    
    # Read data from each file and store with threshold
    data = []
    for file in data_collapse_files:
        # Extract threshold from filename
        threshold = float(file.split('threshold')[-1].split('.csv')[0].replace('e-', 'e-'))
        df = pd.read_csv(file)
        data.append((threshold, df))
    
    # Sort data by threshold
    data.sort(key=lambda x: x[0])
    
    # Extract sorted data
    for threshold, df in data:
        thresholds.append(threshold)
        for impl in ['tao', 'haining']:
            impl_data = df[df['implementation'] == impl]
            if not impl_data.empty:
                pc_values[impl].append(impl_data['pc'].values[0])
                pc_errors[impl].append(impl_data['pc_err'].values[0])
                nu_values[impl].append(impl_data['nu'].values[0])
                nu_errors[impl].append(impl_data['nu_err'].values[0])
    
    # Convert to numpy arrays
    thresholds = np.array(thresholds)
    for impl in ['tao', 'haining']:
        pc_values[impl] = np.array(pc_values[impl])
        pc_errors[impl] = np.array(pc_errors[impl])
        nu_values[impl] = np.array(nu_values[impl])
        nu_errors[impl] = np.array(nu_errors[impl])
    
    # Create separate figures for each implementation
    for impl, color in zip(['tao', 'haining'], ['blue', 'red']):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot pc vs threshold on left y-axis
        ax1.errorbar(thresholds, pc_values[impl], yerr=pc_errors[impl],
                    marker='o', linestyle='-', label='p_c',
                    color='blue', capsize=3)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('p_c', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Set y-axis limits for pc (left axis)
        pc_mean = np.mean(pc_values[impl])
        pc_std = np.std(pc_values[impl])
        ax1.set_ylim(pc_mean - 10*pc_std, pc_mean + 10*pc_std)
        
        # Create second y-axis for nu
        ax2 = ax1.twinx()
        ax2.errorbar(thresholds, nu_values[impl], yerr=nu_errors[impl],
                    marker='s', linestyle='-', label='ν',
                    color='red', capsize=3)
        ax2.set_ylabel('ν', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set y-axis limits for nu (right axis)
        nu_mean = np.mean(nu_values[impl])
        nu_std = np.std(nu_values[impl])
        ax2.set_ylim(nu_mean - 10*nu_std, nu_mean + 10*nu_std)
        
        # Set x-axis to log scale
        ax1.set_xscale('log')
        
        # Add title
        plt.title(f'{impl.capitalize()} Implementation: Critical Parameters vs Threshold')
        
        # Add grid (only for left y-axis to avoid cluttering)
        ax1.grid(True, alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')

        # Add horizontal line at pc_guess, nu_guess
        ax1.axhline(pc_guess, color='b', linestyle='--', alpha=0.7)
        ax2.axhline(nu_guess, color='r', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f'tmi_compare_results/threshold_dependence_{impl}.png', dpi=300, bbox_inches='tight')
        print(f"Saved threshold dependence plot for {impl} to tmi_compare_results/threshold_dependence_{impl}.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for impl in ['tao', 'haining']:
        print(f"\n{impl.capitalize()} Implementation:")
        print(f"p_c = {np.mean(pc_values[impl]):.5f} ± {np.std(pc_values[impl]):.5f} (std across thresholds)")
        print(f"ν = {np.mean(nu_values[impl]):.5f} ± {np.std(nu_values[impl]):.5f} (std across thresholds)")
        print(f"Threshold range: {min(thresholds):.1e} to {max(thresholds):.1e}")
        
        # Print values for each threshold
        print(f"\nDetailed {impl.capitalize()} results:")
        print("Threshold      p_c ± err        ν ± err")
        print("-" * 45)
        for t, pc, pc_e, nu, nu_e in zip(thresholds, pc_values[impl], pc_errors[impl], 
                                        nu_values[impl], nu_errors[impl]):
            print(f"{t:e}  {pc:.5f} ± {pc_e:.5f}  {nu:.5f} ± {nu_e:.5f}")

        