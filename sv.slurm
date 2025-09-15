#!/bin/bash
#SBATCH --partition=main           # Partition name
#SBATCH --job-name=sv         # Job name
#SBATCH --requeue
#SBATCH --output=/scratch/ty296/logs/%j.out    # Output file with job ID
#SBATCH --error=/scratch/ty296/logs/%j.err     # Error file with job ID
#SBATCH --time=20:00:00            # Time limit (20 hours)
#SBATCH --mem-per-cpu=8G           # Memory per CPU
#SBATCH --cpus-per-task=1          # Ensure one CPU per task
#SBATCH --ntasks=5                 # Number of MPI tasks


# # Set parameters from command line arguments
# L=$1
# p_fixed_name=$2
# p_fixed=$3
# p_range=$4
# total_samples=$5
# NCPU=$SLURM_NTASKS

# Unload all modules
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

# Run the MPI program
srun --mpi=pmix python3 /scratch/ty296/CT_toy/sv.py --L $L --p_fixed_name $p_fixed_name --p_fixed $p_fixed --ncpu $NCPU --p_range $p_range --total_samples $total_samples 

# srun --mpi=pmix python3 /scratch/ty296/CT_toy/sv.py --L 12 --p_fixed_name p_ctrl --p_fixed 0.0 --ncpu 1 --p_range "0.0:1.0:2" --total_samples 2