# TMI Analysis on Cluster

This directory contains scripts for running Tripartite Mutual Information (TMI) analysis on a computing cluster using Slurm.

## Files

- `read_tmi_compare_results.py`: The main Python script that performs TMI analysis
- `run_tmi_on_amarel.sh`: Combined script that handles SSH connection to Amarel and job submission

## Requirements

The analysis requires the following Python packages:
- numpy
- pandas
- matplotlib
- h5py
- tqdm
- FSS (for DataCollapse)

## Usage

### Basic Usage

To run the analysis on Amarel with default parameters:

```bash
./run_tmi_on_amarel.sh
```

This will:
1. Create a job script with default parameters
2. Copy the job script to Amarel
3. Submit the job to the Slurm scheduler on Amarel

### Custom Parameters

You can customize the analysis by providing command-line arguments:

```bash
./run_tmi_on_amarel.sh --p-fixed 0.2 --p-fixed-name "pctrl" --bootstrap --partition main
```

### Available Parameters

- `--pc VALUE`: Critical point guess (default: 0.5)
- `--nu VALUE`: Critical exponent guess (default: 1.33)
- `--p-fixed VALUE`: Fixed parameter value (default: 0.0)
- `--p-fixed-name NAME`: Name of fixed parameter ('pproj' or 'pctrl', default: 'pctrl')
- `--bootstrap`: Enable bootstrap analysis (default: false)
- `--l-min VALUE`: Minimum system size (default: 12)
- `--l-max VALUE`: Maximum system size (default: 20)
- `--p-min VALUE`: Minimum p value for range (default: 0.35)
- `--p-max VALUE`: Maximum p value for range (default: 0.65)
- `--threshold-min VALUE`: Minimum threshold exponent (default: -15)
- `--threshold-max VALUE`: Maximum threshold exponent (default: -10)
- `--threshold-steps VALUE`: Number of threshold steps (default: 10)
- `--output-folder NAME`: Output folder name (default: 'tmi_compare_results')
- `--partition NAME`: Slurm partition to use (default: 'main')
- `--direct`: Run directly instead of submitting to Slurm
- `--copy-scripts`: Copy Python scripts from local machine to Amarel
- `--help`: Display help message

### Running Multiple Jobs

You can submit multiple jobs with different parameters by running the script multiple times:

```bash
# For different system sizes with pproj fixed
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 8 --l-max 8
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 12 --l-max 12
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 16 --l-max 16
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 20 --l-max 20
```

## Customizing the Slurm Configuration

The script uses the following default Slurm parameters:
- Job time limit: 24 hours
- Memory allocation: 16GB
- CPU cores: 4
- Partition: main

You can change the partition using the `--partition` parameter.

## Module Loading on Amarel

The script has been updated to use the correct modules on Amarel:

```bash
module purge
module use /projects/community/modulefiles
module load python/3.9.6-gc563
module load hdf5/1.13.3-mpi-oneapi_2022-sw1088
module load openmpi/4.1.6
```

These module loading commands are taken from a working Slurm script (`submit_sv.sh`) that successfully runs on Amarel.

The script also:
1. Disables hyperthreading to improve performance
2. Attempts to install required Python packages using pip if they're not already available
3. Checks for required Python packages before running the analysis

### Python Version Compatibility

The script now uses Python syntax that is compatible with older Python versions (pre-3.6) by:
- Using `.format()` instead of f-strings
- Using string concatenation instead of f-string expressions

The script loads Python 3.9.6, which should have all the necessary features for the analysis.

## Output

The analysis results will be saved in the specified output folder (default: 'tmi_compare_results'). For each threshold value, the following files will be generated:

- CSV files with TMI data
- Data collapse plots
- Loss manifold plots
- Summary CSV with fitted parameters

## Monitoring Jobs

To check the status of your jobs:

```bash
ssh ty296@amarel.rutgers.edu "squeue -u ty296"
```

To view the output of a running or completed job:

```bash
ssh ty296@amarel.rutgers.edu "cat /scratch/ty296/tmi_analysis_JOBID.log"
```

where `JOBID` is the job ID assigned by Slurm.

## Troubleshooting

If a job fails, check the error log:

```bash
ssh amarel "cat /scratch/ty296/tmi_analysis_*.err"
```

Common issues include:
- Missing Python modules
- Insufficient memory allocation
- Time limit exceeded
- Invalid partition name

### Module Loading Issues

If you encounter module loading errors:

1. Check if the modules used in the script are available on Amarel:
   ```bash
   ssh ty296@amarel.rutgers.edu "module avail python"
   ssh ty296@amarel.rutgers.edu "module avail hdf5"
   ```

2. If the specified modules are not available, you may need to modify the script to use available modules.

3. Check available partitions:
   ```bash
   ssh ty296@amarel.rutgers.edu "sinfo -s"
   ```

4. If the Python packages are still not found after loading the modules, you can try installing them manually:
   ```bash
   ssh ty296@amarel.rutgers.edu
   module purge
   module use /projects/community/modulefiles
   module load python/3.9.6-gc563
   python -m pip install --user numpy pandas matplotlib h5py tqdm
   ```

If you encounter an "invalid partition" error, try using a different partition name with the `--partition` parameter.

## Example Usage on Amarel Cluster

Below are examples of how to run similar analyses on the Amarel cluster. These examples can be adapted for the TMI analysis scripts.

### Running Jobs with nohup (Non-Slurm Method)

For running jobs directly without Slurm (useful for development or quick tests):

```bash
# Running singular value analysis for different system sizes with pproj fixed at 0.0
nohup run_sv_fine.sh 8 'pproj' 0.0 0.45 0.1 20 20 20 > run_sv_fine_L8.log 2>&1 &
nohup run_sv_fine.sh 12 'pproj' 0.0 0.45 0.1 20 2000 40 > run_sv_fine_L12.log 2>&1 &
nohup run_sv_fine.sh 16 'pproj' 0.0 0.45 0.1 20 2000 100 > run_sv_fine_L16.log 2>&1 &
nohup run_sv_fine.sh 20 'pproj' 0.0 0.45 0.1 20 2000 100 > run_sv_fine_L20.log 2>&1 &

# Copying results from the cluster to local machine
scp -r ty296@amarel:/scratch/ty296/sv_fine_L*_pproj0.000_pc0.45 .

# Running comparison analysis with pctrl fixed
nohup run_sv_compare_fine.sh 8 'pctrl' 0.4 0.6 0.1 40 2000 20 > run_sv_compare_fine_L8.log 2>&1 &
nohup run_sv_compare_fine.sh 12 'pctrl' 0.4 0.6 0.1 40 2000 40 > run_sv_compare_fine_L12.log 2>&1 &
nohup run_sv_compare_fine.sh 16 'pctrl' 0.4 0.6 0.1 40 2000 100 > run_sv_compare_fine_L16.log 2>&1 &
nohup run_sv_compare_fine.sh 20 'pctrl' 0.4 0.6 0.1 40 2000 100 > run_sv_compare_fine_L20.log 2>&1 &

# Copying comparison results
scp -r ty296@amarel:/scratch/ty296/sv_comparison_L*_pctrl0.400_pc0.800 .

# Running with parameter ranges
nohup run_sv.sh 8 'pctrl' 0.4 "0.0:1.0:20" 2000 20 > run_sv_L8.log 2>&1 &
nohup run_sv.sh 12 'pctrl' 0.4 "0.0:1.0:20" 2000 40 > run_sv_L12.log 2>&1 &
nohup run_sv.sh 16 'pctrl' 0.4 "0.0:1.0:20" 2000 100 > run_sv_L16.log 2>&1 &
nohup run_sv.sh 20 'pctrl' 0.4 "0.0:1.0:20" 2000 200 > run_sv_L20.log 2>&1 &

# Copying range results
scp -r ty296@amarel:/scratch/ty296/sv_comparison_L20_pctrl0.000_p0.550-0.650 .

# Cleaning up temporary files
find . -type f -name "*chunk*" -delete
```

### Adapting for TMI Analysis with the Combined Script

To adapt the above examples for TMI analysis using the combined script:

```bash
# For different system sizes with pproj fixed
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 8 --l-max 8
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 12 --l-max 12
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 16 --l-max 16
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --p-min 0.35 --p-max 0.45 --l-min 20 --l-max 20

# For different system sizes with pctrl fixed
./run_tmi_on_amarel.sh --p-fixed 0.4 --p-fixed-name "pctrl" --p-min 0.4 --p-max 0.6 --l-min 8 --l-max 8
./run_tmi_on_amarel.sh --p-fixed 0.4 --p-fixed-name "pctrl" --p-min 0.4 --p-max 0.6 --l-min 12 --l-max 12
./run_tmi_on_amarel.sh --p-fixed 0.4 --p-fixed-name "pctrl" --p-min 0.4 --p-max 0.6 --l-min 16 --l-max 16
./run_tmi_on_amarel.sh --p-fixed 0.4 --p-fixed-name "pctrl" --p-min 0.4 --p-max 0.6 --l-min 20 --l-max 20

# To run directly without Slurm (similar to nohup approach)
./run_tmi_on_amarel.sh --p-fixed 0.0 --p-fixed-name "pproj" --direct
```

### Parameter Explanation

In the examples above:
- First parameter: System size (L)
- Second parameter: Fixed parameter name ('pproj' or 'pctrl')
- Third parameter: Fixed parameter value
- Fourth/Fifth parameters: Range of the other parameter
- Remaining parameters: Number of steps, samples, and other configuration options

