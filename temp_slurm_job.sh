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
