#!/bin/bash

# Script to submit multiple sv.slurm jobs
# Usage: /scratch/ty296/CT_toy/submit_multple_jobs_MPI.sh --L=12 --p_fixed_name=p_ctrl --p_fixed=0.5 --p_range="0.2" --samples_per_job=2 --ncpu_per_job=2 --n_jobs=2

# SLURM script
SLURM_SCRIPT="/scratch/ty296/CT_toy/sv.slurm"

# Set default values
: ${L:="16"}
: ${p_fixed_name:="p_ctrl"}
: ${p_fixed:="0.5"}
: ${p_range:="0.0:1.0:20"}
: ${samples_per_job:="10"}
: ${ncpu_per_job:="10"}
: ${comparison:="false"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --L=*)
            L="${1#*=}"
            shift
            ;;
        --p_fixed_name=*)
            p_fixed_name="${1#*=}"
            shift
            ;;
        --p_fixed=*)
            p_fixed="${1#*=}"
            shift
            ;;
        --p_range=*)
            p_range="${1#*=}"
            shift
            ;;
        --samples_per_job=*)
            samples_per_job="${1#*=}"
            shift
            ;;
        --ncpu_per_job=*)
            ncpu_per_job="${1#*=}"
            shift
            ;;
        --n_jobs=*)
            n_jobs="${1#*=}"
            shift
            ;;
        --comparison=*)
            comparison="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

for i in $(seq 1 $n_jobs); do
    sbatch --job-name="CT_toy_L${L}_${p_fixed_name}${p_fixed}_job${i}" --export=ALL,L=$L,p_fixed_name=$p_fixed_name,p_fixed=$p_fixed,p_range=$p_range,total_samples=$samples_per_job,NCPU=$ncpu_per_job,comparison=$comparison $SLURM_SCRIPT
done