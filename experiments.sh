#!/bin/bash
#SBATCH -A sxk1942
#SBATCH -p gpu
#SBATCH -C gpul40s
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=gput069
#SBATCH --job-name=bsfa_reorder
#SBATCH --output=results_%j.log
#SBATCH --error=errors_%j.log

# Load required modules
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.8.0

# Activate virtual environment
source /home/kxt437/KLab/BlockSparseFA_Reorder/.venv/bin/activate

# Base directory for experiments
BASE_DIR="/scratch/pioneer/users/kxt437"

# Output results file
RESULTS_FILE="experiment_results_$(date +%Y%m%d_%H%M%S).csv"

# ===== CONFIGURATION PARAMETERS =====
# Filter: Only run experiments with this number of blocks (set to empty string to run all)
TARGET_NUM_BLOCKS="512"  # Change this to your desired number of blocks, or set to "" to run all

# Arrays for mode and head dimension
MODES=("BSFlexAttention" "BSFA" "BSTN" "BSTF")
HEAD_DIMS=(32 64 128)

# Only run these methods
ALLOWED_METHODS=("rcm_hpnbm" "hp_hpnbm")
# ====================================

# Create CSV header
echo "Mode,HeadDim,Experiment,ExperimentFolder,MaskType,Sparsity,NumBlocks,BlockSize,Method,AvgTimeWithReorder,AvgTimeWithoutReorder" > ${RESULTS_FILE}

# Function to extract value from log output
extract_time() {
    local pattern=$1
    local text=$2
    echo "$text" | grep "$pattern" | grep -oP '\d+\.\d+' | head -1
}

# Function to parse folder name
parse_folder_name() {
    local folder=$1
    local mask_type=""
    local sparsity=""
    local num_blocks=""
    local block_size=""
    local segment_size=""

    # Extract mask type (everything before first underscore)
    mask_type=$(echo "$folder" | cut -d'_' -f1)

    # Check if it's LongNet format (has G parameter)
    if [[ $folder =~ _G([0-9]+)_N([0-9]+)_B([0-9]+) ]]; then
        segment_size="${BASH_REMATCH[1]}"
        num_blocks="${BASH_REMATCH[2]}"
        block_size="${BASH_REMATCH[3]}"
        sparsity="G${segment_size}"
    # Otherwise it's S format
    elif [[ $folder =~ _S([0-9.]+)_N([0-9]+)_B([0-9]+) ]]; then
        sparsity="${BASH_REMATCH[1]}"
        num_blocks="${BASH_REMATCH[2]}"
        block_size="${BASH_REMATCH[3]}"
    fi

    echo "${mask_type},${sparsity},${num_blocks},${block_size}"
}

# Function to check if experiment matches target number of blocks
should_run_experiment() {
    local folder=$1

    # If TARGET_NUM_BLOCKS is empty, run all experiments
    if [ -z "$TARGET_NUM_BLOCKS" ]; then
        return 0
    fi

    # Extract number of blocks from folder name
    local num_blocks=""
    if [[ $folder =~ _G([0-9]+)_N([0-9]+)_B([0-9]+) ]]; then
        num_blocks="${BASH_REMATCH[2]}"
    elif [[ $folder =~ _S([0-9.]+)_N([0-9]+)_B([0-9]+) ]]; then
        num_blocks="${BASH_REMATCH[2]}"
    fi

    if [ "$num_blocks" == "$TARGET_NUM_BLOCKS" ]; then
        return 0
    else
        return 1
    fi
}

# Function to run experiment
run_experiment() {
    local mode=$1
    local head_dim=$2
    local exp_name=$3
    local exp_folder=$4
    local matrix_file=$5
    local row_file=$6
    local col_file=$7
    local method=$8

    # Build command
    local cmd="python main.py"

    if [ ! -z "$row_file" ] && [ -f "$row_file" ]; then
        cmd="$cmd --row-reorder $row_file"
    fi

    if [ ! -z "$col_file" ] && [ -f "$col_file" ]; then
        cmd="$cmd --col-reorder $col_file"
    fi

    cmd="$cmd --head-dim $head_dim --mode $mode $matrix_file"

    echo "Running: $cmd"

    # Run and capture output with error handling
    local output
    local exit_code
    output=$(eval $cmd 2>&1) || exit_code=$?

    # Initialize times as NA
    local time_with="NA"
    local time_without="NA"

    # Only extract times if command succeeded
    if [ -z "$exit_code" ] || [ "$exit_code" -eq 0 ]; then
        local extracted_time_with=$(extract_time "Average time with reordering:" "$output")
        local extracted_time_without=$(extract_time "Average time without reordering:" "$output")

        if [ ! -z "$extracted_time_with" ]; then
            time_with="$extracted_time_with"
        fi
        if [ ! -z "$extracted_time_without" ]; then
            time_without="$extracted_time_without"
        fi
    else
        echo "  WARNING: Command failed with exit code $exit_code"
    fi

    # Parse folder metadata
    local metadata=$(parse_folder_name "$exp_folder")

    # Write to CSV
    echo "$mode,$head_dim,$exp_name,$exp_folder,$metadata,$method,$time_with,$time_without" >> ${RESULTS_FILE}

    echo "  Time with reordering: ${time_with} ms"
    echo "  Time without reordering: ${time_without} ms"
    echo ""
}

EXP_DIRS=("Reorder_Exp_0" "Reorder_Exp_1" "Reorder_Exp_2" "Reorder_Exp_3" "Reorder_Exp_4" "Reorder_Exp_5" "Reorder_Exp_6" "Reorder_Exp_7" "Reorder_Exp_8" "Reorder_Exp_9")

for exp_dir in "${EXP_DIRS[@]}"; do
    exp_path="${BASE_DIR}/${exp_dir}"

    if [ ! -d "$exp_path" ]; then
        echo "Warning: Directory $exp_path not found, skipping..."
        continue
    fi

    echo "=========================================="
    echo "Processing experiment: $exp_dir"
    if [ ! -z "$TARGET_NUM_BLOCKS" ]; then
        echo "Filtering for NumBlocks = $TARGET_NUM_BLOCKS"
    fi
    echo "=========================================="

    # Loop through all experiment folders
    for folder in $(ls -d ${exp_path}/*/ 2>/dev/null | xargs -n 1 basename); do
        if ! should_run_experiment "$folder"; then
            echo "Skipping $folder (NumBlocks doesn't match target: $TARGET_NUM_BLOCKS)"
            continue
        fi

        folder_path="${exp_path}/${folder}"
        matrix_file="${folder_path}/original_matrix.mtx"

        if [ ! -f "$matrix_file" ]; then
            echo "Warning: Matrix file not found in $folder, skipping..."
            continue
        fi

        echo ""
        echo "Processing folder: $folder"
        echo "------------------------------------------"

        for mode in "${MODES[@]}"; do
            for head_dim in "${HEAD_DIMS[@]}"; do
                echo ""
                echo "Running with MODE=$mode, HEAD_DIM=$head_dim"
                echo ".........................................."

                # (1) baseline always
                echo "Running baseline (no reordering)..."
                run_experiment "$mode" "$head_dim" "$exp_dir" "$folder" "$matrix_file" "" "" "baseline"

                # (2) only allowed reordering methods
                for method in "${ALLOWED_METHODS[@]}"; do
                    row_file="${folder_path}/${method}_row_permutation.txt"
                    col_file="${folder_path}/${method}_col_permutation.txt"

                    # Special case: rcm_hpnbm uses rcm_hpnbm_row with rcm_col
                    if [ "$method" == "rcm_hpnbm" ] && [ -f "$row_file" ] && [ ! -f "$col_file" ]; then
                        fallback_col="${folder_path}/rcm_col_permutation.txt"
                        if [ -f "$fallback_col" ]; then
                            col_file="$fallback_col"
                        fi
                    fi

                    # If neither exists, skip
                    if [ ! -f "$row_file" ] && [ ! -f "$col_file" ]; then
                        echo "Skipping method $method (no row/col permutation files found)"
                        continue
                    fi

                    # Empty string for missing files so run_experiment doesn't add flags
                    [ -f "$row_file" ] || row_file=""
                    [ -f "$col_file" ] || col_file=""

                    if [ ! -z "$row_file" ] && [ ! -z "$col_file" ]; then
                        echo "Running with row+col reordering: $method"
                    elif [ ! -z "$row_file" ]; then
                        echo "Running with row-only reordering: $method"
                    else
                        echo "Running with col-only reordering: $method"
                    fi

                    run_experiment "$mode" "$head_dim" "$exp_dir" "$folder" "$matrix_file" "$row_file" "$col_file" "$method"
                done
            done
        done
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_FILE}"
echo "=========================================="
