#!/bin/bash

# Make the run script executable
chmod +x run_tmi_analysis.sh

# Function to submit a job with specific parameters
submit_job() {
    local p_fixed=$1
    local p_fixed_name=$2
    local output_folder="tmi_results_${p_fixed_name}${p_fixed}"
    
    echo "Submitting job with p_fixed=$p_fixed, p_fixed_name=$p_fixed_name"
    
    # Create the command with all parameters
    cmd="sbatch run_tmi_analysis.sh --p-fixed $p_fixed --p-fixed-name $p_fixed_name --output-folder $output_folder"
    
    # Add optional parameters if provided
    if [ ! -z "$3" ]; then cmd="$cmd --pc $3"; fi
    if [ ! -z "$4" ]; then cmd="$cmd --nu $4"; fi
    if [ ! -z "$5" ]; then cmd="$cmd --bootstrap"; fi
    if [ ! -z "$6" ]; then cmd="$cmd --l-min $6"; fi
    if [ ! -z "$7" ]; then cmd="$cmd --l-max $7"; fi
    if [ ! -z "$8" ]; then cmd="$cmd --p-min $8"; fi
    if [ ! -z "$9" ]; then cmd="$cmd --p-max $9"; fi
    if [ ! -z "${10}" ]; then cmd="$cmd --threshold-min ${10}"; fi
    if [ ! -z "${11}" ]; then cmd="$cmd --threshold-max ${11}"; fi
    if [ ! -z "${12}" ]; then cmd="$cmd --threshold-steps ${12}"; fi
    
    echo "Running command: $cmd"
    eval $cmd
}

# Example usage:
# Submit jobs for different p_fixed values with pctrl fixed
for p_val in 0.0 0.1 0.2 0.3 0.4; do
    submit_job $p_val "pctrl"
done

# Submit jobs for different p_fixed values with pproj fixed
for p_val in 0.0 0.1 0.2 0.3 0.4 0.5; do
    submit_job $p_val "pproj"
done

# Example of a custom job with more parameters
# submit_job 0.0 "pctrl" 0.5 1.33 true 12 20 0.35 0.65 -15 -10 6

echo "All jobs submitted!" 