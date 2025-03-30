#!/bin/bash

# Combined script to run TMI analysis on Amarel
# This script handles both SSH connection and job submission

# Default parameters
PC_GUESS=0.5
NU_GUESS=1.33
P_FIXED=0.0
P_FIXED_NAME="pctrl"
BOOTSTRAP="False"
L_MIN=12
L_MAX=20
P_MIN=0.35
P_MAX=0.65
NU_MIN=0.3
NU_MAX=1.5
THRESHOLD_MIN=-15
THRESHOLD_MAX=-5
THRESHOLD_STEPS=20
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
            BOOTSTRAP="True"
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
        --nu-range)
            IFS=',' read -r NU_MIN NU_MAX <<< "${2//[()]/}"
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
            if [ "$2" -gt 40 ]; then
                echo "Warning: Using more than 40 threshold steps may cause the job to exceed the 24-hour time limit."
                echo "Each threshold step requires processing all L values and can take ~30 minutes."
                echo "Recommended: Use 20-30 steps for a good balance of resolution and runtime."
                read -p "Do you want to continue with $2 steps? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
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
            echo "  --nu-range VALUE         Range of nu values in format (min,max) (default: (0.3,1.5))"
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

# Function to copy back log files and clean up
copy_back_and_cleanup() {
    local job_id=$1
    local start_time=$(date +%s)
    local wait_time=0
    
    echo "Waiting for job $job_id to complete..."
    
    # Wait for job to complete by checking its status
    while true; do
        # Get job status using sacct
        job_status=$(ssh ty296@amarel.rutgers.edu "sacct -j $job_id --format=State --noheader --parsable2" | head -n1)
        
        # Check if job has completed (COMPLETED), failed (FAILED), or was cancelled (CANCELLED)
        if [[ "$job_status" == "COMPLETED" || "$job_status" == "FAILED" || "$job_status" == "CANCELLED" ]]; then
            current_time=$(date +%s)
            wait_time=$((current_time - start_time))
            echo "Job $job_id finished with status: $job_status after $((wait_time/60)) minutes"
            break
        fi
        
        # Print current status and elapsed time
        current_time=$(date +%s)
        wait_time=$((current_time - start_time))
        echo "Job $job_id status: $job_status ($(($wait_time/60)) minutes elapsed)"
        sleep 60
    done
    
    # Add a small delay to ensure files are fully written
    sleep 30
    
    echo "Attempting to copy log files..."
    
    # Create local output directory if it doesn't exist
    mkdir -p "${OUTPUT_FOLDER}/logs"
    
    # Check if files exist before copying
    if ssh ty296@amarel.rutgers.edu "test -f /scratch/ty296/CT_toy/tmi_analysis_$job_id.log"; then
        # Copy log files back
        scp "ty296@amarel.rutgers.edu:/scratch/ty296/CT_toy/tmi_analysis_$job_id.log" "${OUTPUT_FOLDER}/logs/" || echo "Warning: Failed to copy log file"
        scp "ty296@amarel.rutgers.edu:/scratch/ty296/CT_toy/tmi_analysis_$job_id.err" "${OUTPUT_FOLDER}/logs/" || echo "Warning: Failed to copy error file"
        
        # Remove log files from cluster
        ssh ty296@amarel.rutgers.edu "cd /scratch/ty296/CT_toy && rm -f tmi_analysis_$job_id.{log,err}"
        
        echo "Log files copied to ${OUTPUT_FOLDER}/logs/ and cleaned up on cluster"
    else
        echo "Warning: Log files not found on cluster. Job may have failed."
        # Get detailed job information
        echo "Final job status:"
        ssh ty296@amarel.rutgers.edu "sacct -j $job_id --format=JobID,JobName,State,ExitCode,DerivedExitCode,MaxRSS,Elapsed,NodeList%20"
    fi
    
    # If job failed, print the error file content if it exists
    if [[ "$job_status" == "FAILED" ]]; then
        echo "Job failed. Attempting to show error file content:"
        ssh ty296@amarel.rutgers.edu "cat /scratch/ty296/CT_toy/tmi_analysis_$job_id.err" || echo "No error file found"
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

# Set Python package paths
export PYTHONPATH="/home/ty296/.local/lib/python3.9/site-packages:/cache/home/ty296/.local/lib/python3.9/site-packages:$PYTHONPATH"

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
mkdir -p ${OUTPUT_FOLDER}

# Create a temporary Python script to run the analysis
cat > run_analysis.py << EOT
import sys
import os
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
nu_range = (${NU_MIN}, ${NU_MAX})
threshold_min = ${THRESHOLD_MIN}
threshold_max = ${THRESHOLD_MAX}
threshold_steps = ${THRESHOLD_STEPS}
output_folder = "${OUTPUT_FOLDER}"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Calculate expected runtime
time_per_threshold = 30  # minutes (approximate)
total_time = threshold_steps * time_per_threshold
print(f"\nEstimated runtime: {total_time//60} hours and {total_time%60} minutes")
if total_time > 1380:  # 23 hours in minutes
    print("WARNING: Estimated runtime exceeds 23 hours! Job may be killed due to time limit.")
print("\n")

# Run analysis for each threshold
for threshold in np.logspace(threshold_min, threshold_max, threshold_steps):
    print("\n\n" + "="*80)
    print("Running analysis with threshold = {:.1e}".format(threshold))
    print("Progress: {}/{} thresholds".format(
        len([t for t in np.logspace(threshold_min, threshold_max, threshold_steps) if t <= threshold]),
        threshold_steps
    ))
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
echo "  PC_GUESS = ${PC_GUESS}"
echo "  NU_GUESS = ${NU_GUESS}"
echo "  P_FIXED = ${P_FIXED}"
echo "  P_FIXED_NAME = ${P_FIXED_NAME}"
echo "  BOOTSTRAP = ${BOOTSTRAP}"
echo "  L_MIN = ${L_MIN}"
echo "  L_MAX = ${L_MAX}"
echo "  P_RANGE = (${P_MIN}, ${P_MAX})"
echo "  NU_RANGE = (${NU_MIN}, ${NU_MAX})"
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
python -c "import lmfit; print('lmfit version:', lmfit.__version__)" || echo "Warning: lmfit not found"

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
ssh ty296@amarel.rutgers.edu /bin/bash << EOF
    echo "Connected to Amarel"
    echo "Changing directory to /scratch/ty296/CT_toy"
    cd /scratch/ty296/CT_toy
    
    echo "Current directory: \$(pwd)"
    
    # Load Slurm module
    module purge
    module use /projects/community/modulefiles
    
    # Copy scripts from home directory to scratch
    cp ~/tmi_job_script.sh .
    
    # Make the script executable
    chmod +x tmi_job_script.sh
    
    # Create output directory if it doesn't exist
    mkdir -p "${OUTPUT_FOLDER}"
    
    # Check available partitions
    echo "Available Slurm partitions:"
    sinfo -s
    
    # Run the script
    if [ "${DIRECT_RUN}" = "true" ]; then
        echo "Running analysis directly (not through Slurm)..."
        ./tmi_job_script.sh
    else
        echo "Submitting job to Slurm..."
        # Submit job and capture job ID
        job_id=\$(sbatch --parsable tmi_job_script.sh)
        if [ -n "\$job_id" ]; then
            echo "Job submitted with ID: \$job_id"
            echo "\$job_id" > ~/last_job_id.txt
        else
            echo "Error: Failed to get job ID"
            exit 1
        fi
    fi
    
    # Show job status
    echo "Current jobs in queue:"
    squeue -u ty296
EOF

# Store the SSH exit status
ssh_status=$?

# Check if SSH command was successful
if [ $ssh_status -ne 0 ]; then
    echo "Error: SSH command failed with status $ssh_status"
    exit 1
fi

# Clean up local temporary file
rm tmi_job_script.sh

if [ "${DIRECT_RUN}" != "true" ]; then
    # Get the job ID from the temporary file and verify it's not empty
    job_id=$(ssh ty296@amarel.rutgers.edu "cat ~/last_job_id.txt 2>/dev/null")
    
    if [ -z "$job_id" ]; then
        echo "Error: Failed to get job ID from Amarel"
        exit 1
    fi
    
    ssh ty296@amarel.rutgers.edu "rm -f ~/last_job_id.txt"
    
    echo "Job submitted to Amarel with ID: $job_id"
    echo "Waiting for job to complete and copy back log files..."
    
    # Copy back log files and clean up after job completes
    copy_back_and_cleanup "$job_id"
else
    echo "Job ran directly (not through Slurm)."
fi

echo "To check job status, connect to Amarel and run: squeue -u ty296"
echo "Log files will be available in: ${OUTPUT_FOLDER}/logs/"