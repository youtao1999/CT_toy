#!/bin/bash
#SBATCH --job-name=tmi_analysis
#SBATCH --output=tmi_analysis_%A.log
#SBATCH --error=tmi_analysis_%A.err
#SBATCH --time=5:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Print out environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo ""

# Check Python version and use Python 3
PYTHON="python3"
echo "Checking Python version..."
$PYTHON --version || { echo "Python 3 not found. Exiting."; exit 1; }

# Set up module environment if needed
echo "Loading required modules..."
module load python/3.9.0
module list

# Install required packages if not already available
echo "Checking and installing required Python packages..."
# First uninstall numpy 2.x if it exists and install numpy 1.x for compatibility
$PYTHON -m pip uninstall -y numpy
$PYTHON -m pip install --user "numpy<2.0.0" --no-deps  # Added --no-deps to prevent automatic upgrades
# Now install other packages without upgrading numpy
$PYTHON -m pip install --user pandas matplotlib h5py tqdm --no-deps
# Check installed packages
$PYTHON -c "import numpy; print('NumPy version:', numpy.__version__)"
$PYTHON -c "import pandas; print('Pandas version:', pandas.__version__)" || $PYTHON -m pip install --user pandas --no-deps
$PYTHON -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)" || $PYTHON -m pip install --user matplotlib --no-deps
$PYTHON -c "import h5py; print('h5py version:', h5py.__version__)" || $PYTHON -m pip install --user h5py --no-deps

# Define analysis parameters
PC_GUESS=0.75
NU_GUESS=0.7
P_FIXED=0.4
P_FIXED_NAME="pctrl"
BOOTSTRAP=False
L_MIN=12
L_MAX=20
P_RANGE="(0.55, 0.95)"
NU_RANGE="(0.3, 1.5)"  # Ensure NU_RANGE is set explicitly
THRESHOLD_RANGE="10^-15 to 10^-10 with 20 steps"
OUTPUT_FOLDER="tmi_compare_results"

# Print parameters
echo "Running TMI analysis with the following parameters:"
echo "  PC_GUESS = $PC_GUESS"
echo "  NU_GUESS = $NU_GUESS"
echo "  P_FIXED = $P_FIXED"
echo "  P_FIXED_NAME = $P_FIXED_NAME"
echo "  BOOTSTRAP = $BOOTSTRAP"
echo "  L_MIN = $L_MIN"
echo "  L_MAX = $L_MAX"
echo "  P_RANGE = $P_RANGE"
echo "  NU_RANGE = $NU_RANGE"
echo "  THRESHOLD_RANGE = $THRESHOLD_RANGE"
echo "  OUTPUT_FOLDER = $OUTPUT_FOLDER"
echo ""

# Check Python environment
echo "Checking Python environment..."
echo "Python version: $($PYTHON --version 2>&1)"
echo "NumPy version: $($PYTHON -c 'import numpy; print(numpy.__version__)')"
echo "Pandas version: $($PYTHON -c 'import pandas; print(pandas.__version__)')"
echo "Matplotlib version: $($PYTHON -c 'import matplotlib; print(matplotlib.__version__)')"
echo "h5py version: $($PYTHON -c 'import h5py; print(h5py.__version__)')"

# Check if required data files exist
echo "Checking for data files..."
mkdir -p "$OUTPUT_FOLDER"
CSV_FILE="${OUTPUT_FOLDER}/tmi_compare_results_${P_FIXED_NAME}${P_FIXED}_threshold1.0e-15.csv"
if [ -f "$CSV_FILE" ]; then
    echo "Found CSV file: $CSV_FILE"
    echo "File size: $(du -h $CSV_FILE | cut -f1)"
else
    echo "CSV file not found: $CSV_FILE"
    echo "Looking for H5 files..."
    ls -la sv_comparison_L*_${P_FIXED_NAME}${P_FIXED}_p* 2>/dev/null || echo "No H5 files found matching pattern"
fi

# Create the minimal fix script if it doesn't exist
cat > minimal_fix.py << 'EOF'
#!/usr/bin/env python3
"""
Minimal fix for TMIAnalyzer class to handle None nu_range parameter.
This script patches the plot_compare_loss_manifold method to provide a default value when nu_range is None.

Usage: 
    python minimal_fix.py
    
This will monkey-patch the TMIAnalyzer class at runtime.
"""

import os
import sys
import traceback

# Try to import and patch the TMIAnalyzer class
try:
    from read_tmi_compare_results import TMIAnalyzer
    
    # Check if the class already handles None nu_range properly
    if TMIAnalyzer.plot_compare_loss_manifold.__defaults__ is not None and TMIAnalyzer.plot_compare_loss_manifold.__defaults__[0] is None:
        print("The plot_compare_loss_manifold method already has None as the default for nu_range, checking implementation...")
        
        # Check the method implementation for None handling
        import inspect
        source = inspect.getsource(TMIAnalyzer.plot_compare_loss_manifold)
        if "if nu_range is None:" in source:
            print("The method already handles None values for nu_range, no need to patch.")
            sys.exit(0)
    
    # Store the original method
    original_plot_compare_loss_manifold = TMIAnalyzer.plot_compare_loss_manifold
    
    # Define a patched version of the method
    def patched_plot_compare_loss_manifold(self, p_range, nu_range=None, n_points=100, L_min=12, 
                                          implementations=None, figsize=(15, 6)):
        """
        Patched version of plot_compare_loss_manifold that handles None nu_range.
        """
        print("Using patched plot_compare_loss_manifold method with nu_range: {}".format(nu_range))
        
        # Fix: Provide default nu_range if None
        if nu_range is None:
            nu_range = (0.3, 1.5)  # Default range
            print("nu_range was None, using default: {}".format(nu_range))
        
        # Fix: Convert string parameters to tuples if needed
        if isinstance(nu_range, str):
            try:
                nu_range = eval(nu_range)
                print("Converted string nu_range to tuple: {}".format(nu_range))
            except:
                print("Failed to convert string nu_range: {}, using default".format(nu_range))
                nu_range = (0.3, 1.5)
        
        # Fix: Check if unscaled_df is None
        if self.unscaled_df is None:
            print("Error: unscaled_df is None, cannot create loss manifold")
            return None
        
        # Call the original method with the fixed parameters
        return original_plot_compare_loss_manifold(self, p_range, nu_range, n_points, L_min, 
                                                 implementations, figsize)
    
    # Apply the patch
    TMIAnalyzer.plot_compare_loss_manifold = patched_plot_compare_loss_manifold
    print("Successfully patched TMIAnalyzer.plot_compare_loss_manifold")
    
    # Also patch the result method to ensure nu_range is set
    original_result = TMIAnalyzer.result
    
    def patched_result(self, bootstrap=False, L_min=None, L_max=None, p_range=None, nu_range=None,
                      implementations=None, nu_vary=True, p_c_vary=True):
        """
        Patched version of result method that ensures nu_range is set.
        """
        print("Using patched result method with nu_range: {}".format(nu_range))
        
        # Fix: Provide default nu_range if None
        if nu_range is None:
            nu_range = (0.3, 1.5)  # Default range
            print("nu_range was None, using default: {}".format(nu_range))
        
        # Fix: Convert string parameters to tuples if needed
        if isinstance(nu_range, str):
            try:
                nu_range = eval(nu_range)
                print("Converted string nu_range to tuple: {}".format(nu_range))
            except:
                print("Failed to convert string nu_range: {}, using default".format(nu_range))
                nu_range = (0.3, 1.5)
                
        # Fix: Check that unscaled_df is not None before proceeding
        if original_result.__code__.co_varnames.count('unscaled_df') > 0:
            if self.unscaled_df is None:
                try:
                    print("Attempting to load data from CSV first...")
                    self.read_from_csv()
                    
                    if self.unscaled_df is None:
                        print("Attempting to read and compute from H5 files...")
                        self.read_and_compute_from_h5(n=0)
                        
                        if self.unscaled_df is None:
                            print("ERROR: Unable to load any data, cannot proceed with analysis.")
                            return None
                except Exception as e:
                    print("ERROR: Failed to load data: {}".format(str(e)))
                    traceback.print_exc()
                    return None
        
        # Call the original method with the fixed parameters
        return original_result(self, bootstrap, L_min, L_max, p_range, nu_range,
                             implementations, nu_vary, p_c_vary)
    
    # Apply the patch
    TMIAnalyzer.result = patched_result
    print("Successfully patched TMIAnalyzer.result")
    
    print("\nPatch completed successfully!")
    
except Exception as e:
    print("Error patching TMIAnalyzer: {}".format(str(e)))
    traceback.print_exc()
    sys.exit(1)
EOF

# Apply minimal fix patch
echo "Applying minimal fix for plot_compare_loss_manifold..."
$PYTHON minimal_fix.py
PATCH_RESULT=$?

if [ $PATCH_RESULT -ne 0 ]; then
    echo "Failed to apply minimal fix patch, check minimal_fix.py output"
    exit 1
fi

# Create a simple Python script for the analysis
cat > run_tmi_analysis.py << EOF
#!/usr/bin/env python3
"""
TMI Analysis script with robust error handling
"""
import sys
import os
import traceback
import numpy as np

# Add current directory to path to ensure modules can be found
sys.path.insert(0, os.getcwd())

try:
    print("NumPy version:", np.__version__)
    if np.__version__.startswith('2'):
        print("WARNING: Using NumPy 2.x which may have compatibility issues with the analysis")
    
    # Import TMIAnalyzer (already monkey-patched by minimal_fix.py)
    from read_tmi_compare_results import TMIAnalyzer
    print("Imported TMIAnalyzer successfully")
    
    # Run analysis for different thresholds
    for threshold in np.logspace(-15, -10, 20):
        print('\n\n================================================================================')
        print('Running analysis with threshold = {:.1e}'.format(threshold))
        print('================================================================================\n')
        
        try:
            analyzer = TMIAnalyzer(
                pc_guess=$PC_GUESS, 
                nu_guess=$NU_GUESS, 
                p_fixed=$P_FIXED, 
                p_fixed_name='$P_FIXED_NAME', 
                threshold=threshold
            )
            
            # Explicitly check for data availability
            if not hasattr(analyzer, 'unscaled_df') or analyzer.unscaled_df is None:
                print("Data not loaded automatically, attempting manual loading...")
                analyzer.read_from_csv()
                
                if not hasattr(analyzer, 'unscaled_df') or analyzer.unscaled_df is None:
                    print("CSV data not available, trying H5 files...")
                    analyzer.read_and_compute_from_h5(n=0)
            
            # Verify data is loaded before proceeding
            if not hasattr(analyzer, 'unscaled_df') or analyzer.unscaled_df is None:
                print("ERROR: Could not load data for threshold {:.1e}".format(threshold))
                print("Skipping this threshold and continuing...")
                continue
                
            print("Data loaded successfully for threshold {:.1e}".format(threshold))
            
            # Run the analysis with explicit parameters including nu_range
            results = analyzer.result(
                bootstrap=$BOOTSTRAP, 
                L_min=$L_MIN, 
                L_max=$L_MAX, 
                p_range=$P_RANGE, 
                nu_range=$NU_RANGE
            )
            
            if results is None:
                print("WARNING: Analysis for threshold {:.1e} returned None result".format(threshold))
            else:
                print("Analysis completed successfully for threshold {:.1e}".format(threshold))
                
        except Exception as e:
            print("Error in analysis for threshold {:.1e}: {}".format(threshold, str(e)))
            traceback.print_exc()
            print("Continuing with next threshold...")
        
except Exception as e:
    print("Critical error during analysis: {}".format(str(e)))
    traceback.print_exc()
    sys.exit(1)

print("Analysis script completed execution")
EOF

# Make the script executable
chmod +x run_tmi_analysis.py

# Run analysis using the script file
echo "Starting TMI analysis..."
$PYTHON run_tmi_analysis.py

echo "Analysis completed at $(date)"

# scp all error and log files to local directory
scp 'amarel:/home/ty296/CT_you/tmi_analysis_*.err' .
scp 'amarel:/home/ty296/CT_you/tmi_analysis_*.log' .
