#!/bin/bash

# Default parameters
P_FIXED_NAME="pctrl"
P_FIXED="0.0"
P_RANGE="0.400:0.600:50"
TOTAL_SAMPLES=2000
COMPARISON="--comparison"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --p_fixed_name)
      P_FIXED_NAME="$2"
      shift 2
      ;;
    --p_fixed)
      P_FIXED="$2"
      shift 2
      ;;
    --p_range)
      P_RANGE="$2"
      shift 2
      ;;
    --total_samples)
      TOTAL_SAMPLES="$2"
      shift 2
      ;;
    --no_comparison)
      COMPARISON=""
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--p_fixed_name pctrl|pproj] [--p_fixed value] [--p_range range] [--total_samples count] [--no_comparison]"
      exit 1
      ;;
  esac
done

# Create a temporary Slurm job script
cat > temp_slurm_job.sh << 'EOT'
#!/bin/bash
#SBATCH --job-name=sv_compute
#SBATCH --output=sv_compute_%j.log
#SBATCH --error=sv_compute_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --partition=main

# Load necessary modules
module purge
module use /projects/community/modulefiles
module load python/3.9.6-gc563
module load hdf5/1.13.3-mpi-oneapi_2022-sw1088
module load openmpi/4.1.6

# Set Python package paths
export PYTHONPATH="/home/ty296/.local/lib/python3.9/site-packages:/cache/home/ty296/.local/lib/python3.9/site-packages:$PYTHONPATH"

# Disable hyperthreading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Change to working directory
cd /scratch/ty296/CT_toy

# Print job information
echo "Starting job at $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Parameters:"
echo "  p_fixed_name: ${p_fixed_name}"
echo "  p_fixed: ${p_fixed}"
echo "  p_range: ${p_range}"
echo "  total_samples: ${total_samples}"
echo "  comparison: ${comparison}"

# Run computations for each L value
for L in 8 12 16 20; do
    echo "Starting computation for L=$L"
    mpirun -n 4 python sv.py \
        --L $L \
        --p_fixed_name ${p_fixed_name} \
        --p_fixed ${p_fixed} \
        --p_range "${p_range}" \
        --ncpu 4 \
        --total_samples ${total_samples} \
        ${comparison}
    echo "Completed computation for L=$L"
done

echo "All computations completed at $(date)"
EOT

# Copy the script to Amarel
echo "Copying job script to Amarel..."
scp temp_slurm_job.sh ty296@amarel.rutgers.edu:/scratch/ty296/CT_toy/

# Submit the job on Amarel
echo "Submitting job to Slurm..."
ssh ty296@amarel.rutgers.edu "cd /scratch/ty296/CT_toy && \
    export p_fixed_name=${P_FIXED_NAME} && \
    export p_fixed=${P_FIXED} && \
    export p_range='${P_RANGE}' && \
    export total_samples=${TOTAL_SAMPLES} && \
    export comparison='${COMPARISON}' && \
    sbatch temp_slurm_job.sh"

# Clean up local temporary file
rm temp_slurm_job.sh

echo "Job submitted. Check status with: ssh amarel 'squeue -u ty296'" 