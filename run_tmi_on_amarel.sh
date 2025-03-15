#!/bin/bash

# Combined script to run TMI analysis on Amarel
# This script handles both SSH connection and job submission

# Default parameters
PC_GUESS=0.5
NU_GUESS=1.33
P_FIXED=0.0
P_FIXED_NAME="pctrl"
BOOTSTRAP=false
L_MIN=12
L_MAX=20
P_MIN=0.35
P_MAX=0.65
THRESHOLD_MIN=-15
THRESHOLD_MAX=-10
THRESHOLD_STEPS=10
OUTPUT_FOLDER="tmi_compare_results"
PARTITION="main"
DIRECT_RUN=false
COPY_SCRIPTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pc)
            PC_GUESS="$2"
            shift 2
            ;;
        --nu)
            NU_GUESS="$2"
            shift 2
            ;;
        --p-fixed)
            P_FIXED="$2"
            shift 2
            ;;
        --p-fixed-name)
            P_FIXED_NAME="$2"
            shift 2
            ;;
        --bootstrap)
            BOOTSTRAP=true
            shift
            ;;
        --l-min)
            L_MIN="$2"
            shift 2
            ;;
        --l-max)
            L_MAX="$2"
            shift 2
            ;;
        --p-min)
            P_MIN="$2"
            shift 2
            ;;
        --p-max)
            P_MAX="$2"
            shift 2
            ;;
        --threshold-min)
            THRESHOLD_MIN="$2"
            shift 2
            ;;
        --threshold-max)
            THRESHOLD_MAX="$2"
            shift 2
            ;;
        --threshold-steps)
            THRESHOLD_STEPS="$2"
            shift 2
            ;;
        --output-folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --direct)
            DIRECT_RUN=true
            shift
            ;;
        --copy-scripts)
            COPY_SCRIPTS=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --pc VALUE               Critical point guess (default: 0.5)"
            echo "  --nu VALUE               Critical exponent guess (default: 1.33)"
            echo "  --p-fixed VALUE          Fixed parameter value (default: 0.0)"
            echo "  --p-fixed-name NAME      Name of fixed parameter ('pproj' or 'pctrl', default: 'pctrl')"
            echo "  --bootstrap              Enable bootstrap analysis (default: false)"
            echo "  --l-min VALUE            Minimum system size (default: 12)"
            echo "  --l-max VALUE            Maximum system size (default: 20)"
            echo "  --p-min VALUE            Minimum p value for range (default: 0.35)"
            echo "  --p-max VALUE            Maximum p value for range (default: 0.65)"
            echo "  --threshold-min VALUE    Minimum threshold exponent (default: -15)"
            echo "  --threshold-max VALUE    Maximum threshold exponent (default: -10)"
            echo "  --threshold-steps VALUE  Number of threshold steps (default: 10)"
            echo "  --output-folder NAME     Output folder name (default: 'tmi_compare_results')"
            echo "  --partition NAME         Slurm partition to use (default: 'main')"
            echo "  --direct                 Run directly instead of submitting to Slurm"
            echo "  --copy-scripts           Copy scripts from local machine to Amarel"
            echo "  --help                   Display this help message"
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to copy scripts to Amarel if needed
copy_scripts_to_amarel() {
    if [ "$COPY_SCRIPTS" = true ]; then
        echo "Copying scripts to Amarel..."
        
        # Check if the run_tmi_analysis.sh script exists locally
        if [ ! -f "run_tmi_analysis.sh" ]; then
            echo "Error: run_tmi_analysis.sh not found in the current directory."
            echo "Please make sure the script exists before running this command."
            exit 1
        fi
        scp run_tmi_analysis.sh ty296@amarel.rutgers.edu:~/ || {
            echo "Error: Failed to copy run_tmi_analysis.sh to Amarel."
            exit 1
        }

        # Also copy the Python script if it exists
        if [ -f "read_tmi_compare_results.py" ]; then
            echo "Copying read_tmi_compare_results.py to Amarel..."
            scp read_tmi_compare_results.py ty296@amarel.rutgers.edu:~/ || {
                echo "Warning: Failed to copy read_tmi_compare_results.py to Amarel."
            }
        fi
    fi
}

# Create the Slurm job script
create_job_script() {
    cat > tmi_job_script.sh << EOF
#!/bin/bash
#SBATCH --job-name=tmi_analysis
#SBATCH --output=tmi_analysis_%j.log
#SBATCH --error=tmi_analysis_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=${PARTITION}

# Print job information
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Start time: \$(date)"
echo ""

# Load necessary modules (using the same modules as in submit_sv.sh)
module purge
module use /projects/community/modulefiles
module load python/3.9.6-gc563
module load hdf5/1.13.3-mpi-oneapi_2022-sw1088
module load openmpi/4.1.6

# Disable hyperthreading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Install required packages if not already available
echo "Checking and installing required Python packages..."
python -m pip install --user numpy pandas matplotlib h5py tqdm || echo "Warning: Failed to install some packages"

# Create a temporary Python script to run the analysis
cat > run_analysis.py << EOT
import sys
import numpy as np
from read_tmi_compare_results import TMIAnalyzer

# Parameters from command line
pc_guess = ${PC_GUESS}
nu_guess = ${NU_GUESS}
p_fixed = ${P_FIXED}
p_fixed_name = "${P_FIXED_NAME}"
bootstrap = ${BOOTSTRAP}
l_min = ${L_MIN}
l_max = ${L_MAX}
p_range = (${P_MIN}, ${P_MAX})
threshold_min = ${THRESHOLD_MIN}
threshold_max = ${THRESHOLD_MAX}
threshold_steps = ${THRESHOLD_STEPS}
output_folder = "${OUTPUT_FOLDER}"

# Run analysis for each threshold
for threshold in np.logspace(threshold_min, threshold_max, threshold_steps):
    print("\\n\\n" + "="*80)
    print("Running analysis with threshold = {:.1e}".format(threshold))
    print("="*80 + "\\n")
    
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
        p_range=p_range
    )
    
    print("\\nCompleted analysis for threshold = {:.1e}".format(threshold))
EOT

# Print the parameters being used
echo "Running TMI analysis with the following parameters:"
echo "  PC_GUESS = ${PC_GUESS}"
echo "  NU_GUESS = ${NU_GUESS}"
echo "  P_FIXED = ${P_FIXED}"
echo "  P_FIXED_NAME = ${P_FIXED_NAME}"
echo "  BOOTSTRAP = ${BOOTSTRAP}"
echo "  L_MIN = ${L_MIN}"
echo "  L_MAX = ${L_MAX}"
echo "  P_RANGE = (${P_MIN}, ${P_MAX})"
echo "  THRESHOLD_RANGE = 10^${THRESHOLD_MIN} to 10^${THRESHOLD_MAX} with ${THRESHOLD_STEPS} steps"
echo "  OUTPUT_FOLDER = ${OUTPUT_FOLDER}"
echo ""

# Check if required Python packages are available
echo "Checking Python environment..."
python -c "import sys; print('Python version:', sys.version)"
python -c "import numpy; print('NumPy version:', numpy.__version__)" || echo "Warning: NumPy not found"
python -c "import pandas; print('Pandas version:', pandas.__version__)" || echo "Warning: Pandas not found"
python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)" || echo "Warning: Matplotlib not found"
python -c "import h5py; print('h5py version:', h5py.__version__)" || echo "Warning: h5py not found"

# Run the analysis
echo "Starting TMI analysis..."
python run_analysis.py

# Clean up
rm run_analysis.py

echo ""
echo "Analysis completed at \$(date)"
EOF
}

# Copy scripts if needed
copy_scripts_to_amarel

# Create the job script
create_job_script

# Copy the job script to Amarel and run it
echo "Copying job script to Amarel..."
scp tmi_job_script.sh ty296@amarel.rutgers.edu:~/ || {
    echo "Error: Failed to copy job script to Amarel."
    exit 1
}

# Connect to Amarel and run the job
echo "Connecting to Amarel and running the job..."
ssh ty296@amarel.rutgers.edu << EOF
    echo "Connected to Amarel"
    echo "Changing directory to /scratch/ty296/"
    cd /scratch/ty296/
    
    echo "Current directory: \$(pwd)"
    
    # Copy scripts from home directory to scratch
    cp ~/tmi_job_script.sh .
    if [ -f ~/read_tmi_compare_results.py ]; then
        cp ~/read_tmi_compare_results.py .
    fi
    
    # Make the script executable
    chmod +x tmi_job_script.sh
    
    # Create output directory if it doesn't exist
    mkdir -p ${OUTPUT_FOLDER}
    
    # Check available partitions
    echo "Available Slurm partitions:"
    sinfo -s
    
    # Run the script
    if [ "${DIRECT_RUN}" = "true" ]; then
        echo "Running analysis directly (not through Slurm)..."
        ./tmi_job_script.sh
    else
        echo "Submitting job to Slurm..."
        sbatch tmi_job_script.sh
    fi
    
    # Show job status
    echo "Current jobs in queue:"
    squeue -u ty296
EOF

# Clean up local temporary file
rm tmi_job_script.sh

echo "Job submitted to Amarel."
echo "To check job status, connect to Amarel and run: squeue -u ty296"
echo "To view output logs, connect to Amarel and run: cat tmi_analysis_*.log"