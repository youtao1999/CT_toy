#!/bin/bash

# Wrapper to send email with job results
# Check if L and ncpu are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Please provide L value and number of CPUs"
    echo "Usage: ./run_tmi.sh <L> <ncpu>"
    echo "Note: ncpu must divide 2000 evenly"
    exit 1
fi

L=$1
NCPU=$2

# Submit the job and capture the job ID
JOB_ID=$(sbatch --ntasks=$NCPU submit_tmi_course.sh $L | grep -o '[0-9]*')

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to submit job"
    exit 1
fi

echo "Submitted job $JOB_ID for L=$L with $NCPU CPUs"

# Wait for the job to complete
while squeue -j $JOB_ID > /dev/null 2>&1; do
    sleep 60  # Check every minute
done

# Create email with job results
{
    echo "Job ID: $JOB_ID"
    echo "Job Name: tmi_calc (L=${L})"
    echo "Status: Completed"
    echo -e "\nOutput Log:"
    echo "------------"
    cat "tmi_L${L}_${JOB_ID}.out"
    echo -e "\nError Log:"
    echo "----------"
    cat "tmi_L${L}_${JOB_ID}.err"
} | mail -s "TMI Calculation Results (L=${L}, JobID: ${JOB_ID})" ty296@physics.rutgers.edu

echo "Email sent with job results" 