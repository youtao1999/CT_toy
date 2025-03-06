#!/bin/bash

# Script to submit multiple jobs to Amarel simultaneously with customizable parameters

# Default values (can be overridden by command line arguments)
P_FIXED_NAME="pctrl"
P_FIXED="0.0"
P_RANGE="0.55:0.65:20"
TOTAL_SAMPLES=2000
COMPARISON="--comparison"  # Empty string for no comparison

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

# Define the remote directory
REMOTE_DIR="/scratch/ty296"

# Create a temporary script to run on Amarel
cat > temp_run_commands.sh << EOF
#!/bin/bash
cd /scratch/ty296/

# Make sure run_sv.sh is executable
chmod +x /scratch/ty296/run_sv.sh

# Run all calculations simultaneously
echo "Starting L=8 calculation..."
/scratch/ty296/run_sv.sh 8 '${P_FIXED_NAME}' ${P_FIXED} "${P_RANGE}" ${TOTAL_SAMPLES} 20 ${COMPARISON} > run_sv_L8_${P_FIXED_NAME}${P_FIXED}.log 2>&1 &
L8_PID=\$!
echo "L=8 job started with PID \$L8_PID"

echo "Starting L=12 calculation..."
/scratch/ty296/run_sv.sh 12 '${P_FIXED_NAME}' ${P_FIXED} "${P_RANGE}" ${TOTAL_SAMPLES} 40 ${COMPARISON} > run_sv_L12_${P_FIXED_NAME}${P_FIXED}.log 2>&1 &
L12_PID=\$!
echo "L=12 job started with PID \$L12_PID"

echo "Starting L=16 calculation..."
/scratch/ty296/run_sv.sh 16 '${P_FIXED_NAME}' ${P_FIXED} "${P_RANGE}" ${TOTAL_SAMPLES} 100 ${COMPARISON} > run_sv_L16_${P_FIXED_NAME}${P_FIXED}.log 2>&1 &
L16_PID=\$!
echo "L=16 job started with PID \$L16_PID"

echo "Starting L=20 calculation..."
/scratch/ty296/run_sv.sh 20 '${P_FIXED_NAME}' ${P_FIXED} "${P_RANGE}" ${TOTAL_SAMPLES} 200 ${COMPARISON} > run_sv_L20_${P_FIXED_NAME}${P_FIXED}.log 2>&1 &
L20_PID=\$!
echo "L=20 job started with PID \$L20_PID"

echo "All jobs submitted simultaneously. Check individual logs for progress."
echo "PIDs: L8=\$L8_PID, L12=\$L12_PID, L16=\$L16_PID, L20=\$L20_PID"

# Optional: Wait for all jobs to complete
echo "Waiting for all jobs to complete..."
wait \$L8_PID \$L12_PID \$L16_PID \$L20_PID
echo "All calculations have completed."
EOF

# Make the temporary script executable locally (optional)
chmod +x temp_run_commands.sh

# Copy the script to Amarel
scp temp_run_commands.sh ty296@amarel.rutgers.edu:/scratch/ty296/

# Make it executable on Amarel and then run it
ssh -n amarel "chmod +x /scratch/ty296/temp_run_commands.sh && nohup /scratch/ty296/temp_run_commands.sh > /scratch/ty296/amarel_jobs_${P_FIXED_NAME}${P_FIXED}.log 2>&1 & echo 'Jobs submitted successfully!'"

# Clean up the temporary script
rm temp_run_commands.sh

echo "Jobs submitted to Amarel with parameters:"
echo "  p_fixed_name: ${P_FIXED_NAME}"
echo "  p_fixed: ${P_FIXED}"
echo "  p_range: ${P_RANGE}"
echo "  total_samples: ${TOTAL_SAMPLES}"
echo "  comparison: ${COMPARISON:-(disabled)}"
echo ""
echo "You can check progress with:"
echo "ssh amarel 'cat /scratch/ty296/amarel_jobs_${P_FIXED_NAME}${P_FIXED}.log'"
echo "or individual job logs with:"
echo "ssh amarel 'cat /scratch/ty296/run_sv_L*_${P_FIXED_NAME}${P_FIXED}.log'"
