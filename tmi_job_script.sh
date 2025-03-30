#!/bin/bash
#SBATCH --job-name=tmi_analysis
#SBATCH --output=tmi_analysis_%j.log
#SBATCH --error=tmi_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

# Load necessary modules (using the same modules as in submit_sv.sh)
module purge
module use /projects/community/modulefiles
module load python/3.9.6-gc563
module load hdf5/1.13.3-mpi-oneapi_2022-sw1088
module load openmpi/4.1.6

# Set Python package paths
export PYTHONPATH="/home/ty296/.local/lib/python3.9/site-packages:/cache/home/ty296/.local/lib/python3.9/site-packages:"

# Print Python environment information
echo "Checking Python package locations..."
python -c "import site; print('User site packages directory:', site.USER_SITE)"
python -c "import sys; print('All Python package locations:'); [print(p) for p in sys.path]"
python -c "import lmfit; print('lmfit location:', lmfit.__file__)" || echo "lmfit not found"
echo ""

# Disable hyperthreading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Install required packages if not already available
echo "Checking and installing required Python packages..."
# First uninstall numpy 2.x if it exists and install numpy 1.x for compatibility
python -m pip uninstall -y numpy
python -m pip install --user "numpy<2.0.0"
python -m pip install --user pandas matplotlib h5py tqdm lmfit || echo "Warning: Failed to install some packages"

# Create output directory explicitly
mkdir -p tmi_compare_results

# Create a temporary Python script to run the analysis
cat > run_analysis.py << EOT
import sys
import os
import numpy as np
from read_tmi_compare_results import TMIAnalyzer

# Parameters from command line
pc_guess = 0.5
nu_guess = 1.33
p_fixed = 0.0
p_fixed_name = "pctrl"
bootstrap = False
l_min = 12
l_max = 20
p_range = (0.35, 0.65)
nu_range = (0.3, 1.5)
threshold_min = -15
threshold_max = -5
threshold_steps = 80
output_folder = "tmi_compare_results"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Run analysis for each threshold
for threshold in np.logspace(threshold_min, threshold_max, threshold_steps):
    print("\n\n" + "="*80)
    print("Running analysis with threshold = {:.1e}".format(threshold))
    print("="*80 + "\n")
    
    analyzer = TMIAnalyzer(
        pc_guess=pc_guess,
        nu_guess=nu_guess,
        p_fixed=p_fixed,
        p_fixed_name=p_fixed_name,
        threshold=threshold,
        output_folder=output_folder
    )
    
    results = analyzer.result(
        bootstrap=bootstrap,
        L_min=l_min,
        L_max=l_max,
        p_range=p_range,
        nu_range=nu_range
    )
    
    print("\nCompleted analysis for threshold = {:.1e}".format(threshold))
EOT

# Print the parameters being used
echo "Running TMI analysis with the following parameters:"
echo "  PC_GUESS = 0.5"
echo "  NU_GUESS = 1.33"
echo "  P_FIXED = 0.0"
echo "  P_FIXED_NAME = pctrl"
echo "  BOOTSTRAP = False"
echo "  L_MIN = 12"
echo "  L_MAX = 20"
echo "  P_RANGE = (0.35, 0.65)"
echo "  NU_RANGE = (0.3, 1.5)"
echo "  THRESHOLD_RANGE = 10^-15 to 10^-5 with 80 steps"
echo "  OUTPUT_FOLDER = tmi_compare_results"
echo ""

# Check if required Python packages are available
echo "Checking Python environment..."
python -c "import sys; print('Python version:', sys.version)"
python -c "import numpy; print('NumPy version:', numpy.__version__)" || echo "Warning: NumPy not found"
python -c "import pandas; print('Pandas version:', pandas.__version__)" || echo "Warning: Pandas not found"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)" || echo "Warning: Matplotlib not found"
python -c "import h5py; print('h5py version:', h5py.__version__)" || echo "Warning: h5py not found"
python -c "import lmfit; print('lmfit version:', lmfit.__version__)" || echo "Warning: lmfit not found"

# Run the analysis
echo "Starting TMI analysis..."
python run_analysis.py

# Clean up
rm run_analysis.py

echo ""
echo "Analysis completed at $(date)"
