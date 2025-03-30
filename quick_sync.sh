#!/bin/bash

# Cluster connection details - using your SSH config name
CLUSTER_NAME="amarel"
CLUSTER_PATH="/scratch/ty296/CT_toy"

# Function to display usage
function show_usage {
    echo "Quick Sync Tool for Amarel Cluster"
    echo "Usage:"
    echo "  ./quick_sync.sh [options] file1 file2 ..."
    echo "Options:"
    echo "  --all           Sync all Python files"
    echo "  --dir DIR       Sync an entire directory"
    echo "  --py            Sync all Python files in current directory"
    echo "  --delete        Delete remote files that don't exist locally (protects data files)"
    echo "  --help          Show this help message"
    exit 1
}

# Check if arguments were provided
if [ $# -eq 0 ]; then
    show_usage
fi

# Check if --delete flag is present
DELETE_FLAG=""
DATA_PROTECTION=""
ARGS=()
for arg in "$@"; do
    if [ "$arg" == "--delete" ]; then
        DELETE_FLAG="--delete"
        # Add exclusions to protect data files when using --delete
        DATA_PROTECTION="--exclude=*.data --exclude=*.csv --exclude=*.txt --exclude=*.log --exclude=*.out --exclude=*.h5 --exclude=*.hdf5 --exclude=*.pkl --exclude=*.npy --exclude=*.npz --exclude=results/ --exclude=data/ --exclude=sv_*/ --exclude=sv_L*/ --exclude=*_comparison_*/ --exclude=*_fine_*/"
    else
        ARGS+=("$arg")
    fi
done

# Process arguments
case "${ARGS[0]}" in
    --all)
        echo "Syncing all files..."
        rsync -avz $DELETE_FLAG $DATA_PROTECTION --include="*.py" --include="*.sh" --exclude="*" ./ ${CLUSTER_NAME}:${CLUSTER_PATH}/
        echo "All files synced to cluster"
        ;;
    --dir)
        if [ -z "${ARGS[1]}" ]; then
            echo "Error: Directory not specified"
            show_usage
        fi
        echo "Syncing directory ${ARGS[1]}..."
        rsync -avz $DELETE_FLAG $DATA_PROTECTION "${ARGS[1]}/" ${CLUSTER_NAME}:${CLUSTER_PATH}/"${ARGS[1]}"/
        echo "Directory ${ARGS[1]} synced to cluster"
        ;;
    --py)
        echo "Syncing all Python files in current directory..."
        rsync -avz $DELETE_FLAG $DATA_PROTECTION --include="*.py" --exclude="*" ./ ${CLUSTER_NAME}:${CLUSTER_PATH}/
        echo "All Python files synced to cluster"
        ;;
    --sh)
        echo "Syncing all shell files..."
        rsync -avz $DELETE_FLAG $DATA_PROTECTION --include="*.sh" --exclude="*" ./ ${CLUSTER_NAME}:${CLUSTER_PATH}/
        echo "All shell files synced to cluster"
        ;;
    --help)
        show_usage
        ;;
    *)
        # For individual files, --delete doesn't make sense
        if [ -n "$DELETE_FLAG" ]; then
            echo "Warning: --delete flag is ignored when syncing individual files"
        fi
        
        # Sync specific files provided as arguments
        for file in "${ARGS[@]}"; do
            if [ -f "$file" ]; then
                echo "Syncing $file to cluster..."
                rsync -avz "$file" ${CLUSTER_NAME}:${CLUSTER_PATH}/
            else
                echo "Warning: $file not found"
            fi
        done
        ;;
esac

echo "Sync complete"
