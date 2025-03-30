#!/bin/bash

# Wrapper to send email with job results
# Check if all required parameters are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$6" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: ./run_sv.sh <L> <p_fixed_name> <p_fixed> <p_range> <total_samples> <ncpu> [--comparison]"
    echo "Note: ncpu must divide total_samples evenly"
    exit 1
fi

L=$1
p_fixed_name=$2
p_fixed=$3
p_range=$4
total_samples=$5
NCPU=$6

# Check if comparison flag is provided
comparison=""
if [ "$7" == "--comparison" ]; then
    comparison="--comparison"
fi

# Submit the job with proper Slurm parameters
JOB_ID=$(sbatch \
    --partition=main \
    --job-name=sv_calc_L${L} \
    --output=sv_calc_L${L}_${p_fixed_name}${p_fixed}_%j.out \
    --error=sv_calc_L${L}_${p_fixed_name}${p_fixed}_%j.err \
    --time=20:00:00 \
    --mem-per-cpu=4G \
    --ntasks=$NCPU \
    --cpus-per-task=1 \
    --mail-user=ty296@physics.rutgers.edu \
    --mail-type=END,FAIL \
    submit_sv.sh $L $p_fixed_name $p_fixed "$p_range" $total_samples $comparison | grep -o '[0-9]*')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    exit 1
fi

echo "Submitted job $JOB_ID for L=$L, ${p_fixed_name}=${p_fixed}, p_range=${p_range} with $NCPU CPUs"

# Wait for the job to complete
while squeue -j $JOB_ID > /dev/null 2>&1; do
    sleep 60  # Check every minute
done

# Create email with job results
{
    echo "Job ID: $JOB_ID"
    echo "Job Name: sv_calc (L=${L}, ${p_fixed_name}=${p_fixed}, p_range=${p_range})"
    echo "Status: Completed"
    echo -e "\nOutput Log:"
    echo "------------"
    cat "sv_calc_L${L}_${p_fixed_name}${p_fixed}_${JOB_ID}.out"
    echo -e "\nError Log:"
    echo "----------"
    cat "sv_calc_L${L}_${p_fixed_name}${p_fixed}_${JOB_ID}.err"
} | mail -s "SV Calculation Results (L=${L}, ${p_fixed_name}=${p_fixed}, JobID: ${JOB_ID})" ty296@physics.rutgers.edu

echo "Email sent with job results" 