#!/bin/bash

# Wrapper to send email with job results
# Check if all required parameters are provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$8" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: ./run_sv_compare_fine.sh <L> <p_fixed_name> <p_fixed> <p_c> <delta_p> <num_p_scan> <total_samples> <ncpu>"
    echo "Note: ncpu must divide total_samples evenly"
    exit 1
fi

L=$1
p_fixed_name=$2
p_fixed=$3
p_c=$4
delta_p=$5
num_p_scan=$6
total_samples=$7
NCPU=$8

# Submit the job and capture the job ID
JOB_ID=$(sbatch --ntasks=$NCPU submit_sv_compare_fine.sh $L $p_fixed_name $p_fixed $p_c $delta_p $num_p_scan $total_samples | grep -o '[0-9]*')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    exit 1
fi

echo "Submitted job $JOB_ID for L=$L, ${p_fixed_name}=${p_fixed}, p_c=${p_c} with $NCPU CPUs"

# Wait for the job to complete
while squeue -j $JOB_ID > /dev/null 2>&1; do
    sleep 60  # Check every minute
done

# Create email with job results
{
    echo "Job ID: $JOB_ID"
    echo "Job Name: tmi_compare (L=${L}, ${p_fixed_name}=${p_fixed}, p_c=${p_c})"
    echo "Status: Completed"
    echo -e "\nOutput Log:"
    echo "------------"
    cat "sv_compare_L${L}_${p_fixed_name}${p_fixed}_${JOB_ID}.out"
    echo -e "\nError Log:"
    echo "----------"
    cat "sv_compare_L${L}_${p_fixed_name}${p_fixed}_${JOB_ID}.err"
} | mail -s "SV Comparison Results (L=${L}, ${p_fixed_name}=${p_fixed}, p_c=${p_c}, JobID: ${JOB_ID})" ty296@physics.rutgers.edu

echo "Email sent with job results" 