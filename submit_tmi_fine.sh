#!/bin/bash
#SBATCH --partition=main           # Partition name
#SBATCH --job-name=tmi_calc        # Job name
#SBATCH --output=tmi_%j.out        # Output file with job ID
#SBATCH --error=tmi_%j.err         # Error file with job ID
#SBATCH --time=20:00:00            # Time limit (20 hours)
#SBATCH --mem-per-cpu=4G           # Memory per CPU
#SBATCH --cpus-per-task=1          # Ensure one CPU per task

# Email notifications
#SBATCH --mail-user=ty296@physics.rutgers.edu
#SBATCH --mail-type=END,FAIL       # Send email when job ends or fails

# Check if L is provided
if [ -z "$1" ]; then
    echo "Error: Please provide L value"
    echo "Usage: sbatch --ntasks=<ncpu> submit_tmi_fine.sh <L> <p_proj> <p_c>"
    echo "Note: ntasks must divide 2000 evenly"
    exit 1
fi

# Set L from command line argument
L=$1
p_fixed_name=$2
p_fixed=$3
p_c=$4
delta_p=$5
num_p_scan=$6
total_samples=$7
NCPU=$SLURM_NTASKS
JOB_ID=$SLURM_JOB_ID

# Rename output and error files to include L value
mv "tmi_${JOB_ID}.out" "tmi_L${L}_${JOB_ID}.out"
mv "tmi_${JOB_ID}.err" "tmi_L${L}_${JOB_ID}.err"

# Verify ncpu is valid
if [ $((total_samples % NCPU)) -ne 0 ]; then
    echo "Error: Number of CPUs ($NCPU) must divide total_samples ($total_samples) evenly"
    echo "Valid CPU counts are: 1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000"
    exit 1
fi

# Export variables for SLURM
export SLURM_L=$L

# Unload all modules
module purge
module use /projects/community/modulefiles
module load python/3.9.6-gc563
module load hdf5/1.13.3-mpi-oneapi_2022-sw1088
module load openmpi/2.1.1

# Disable hyperthreading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create directory for this L value
mkdir -p tmi_fine_L${L}_${p_fixed_name}$(printf "%.3f" $p_fixed)_pc${p_c}

# Run the MPI program
srun python3 tmi_fine.py --L $L --ncpu $NCPU --p_c $p_c --p_fixed $p_fixed --p_fixed_name $p_fixed_name --delta_p $delta_p --num_p_scan $num_p_scan --total_samples $total_samples

