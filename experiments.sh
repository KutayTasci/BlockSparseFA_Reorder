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
source .venv/bin/activate

# Base directory for experiments
BASE_DIR="/scratch/pioneer/users/kxt437"

# Output results file
RESULTS_FILE="experiment_results_$(date +%Y%m%d_%H%M%S).csv"

# Create CSV header
echo "Experiment,ExperimentFolder,MaskType,Sparsity,NumBlocks,BlockSize,Method,AvgTimeWithReorder,AvgTimeWithoutReorder" > ${RESULTS_FILE}

# Head dimension (fixed parameter)
HEAD_DIM=64
MODE="BSFA"

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

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local exp_folder=$2
    local matrix_file=$3
    local row_file=$4
    local col_file=$5
    local method=$6
    
    # Build command
    local cmd="python main.py"
    
    if [ ! -z "$row_file" ] && [ -f "$row_file" ]; then
        cmd="$cmd --row-reorder $row_file"
    fi
    
    if [ ! -z "$col_file" ] && [ -f "$col_file" ]; then
        cmd="$cmd --col-reorder $col_file"
    fi
    
    cmd="$cmd --head-dim $HEAD_DIM --mode $MODE $matrix_file"
    
    echo "Running: $cmd"
    
    # Run and capture output
    local output=$(eval $cmd 2>&1)
    
    # Extract times
    local time_with=$(extract_time "Average time with reordering:" "$output")
    local time_without=$(extract_time "Average time without reordering:" "$output")
    
    # Parse folder metadata
    local metadata=$(parse_folder_name "$exp_folder")
    
    # Write to CSV
    echo "$exp_name,$exp_folder,$metadata,$method,$time_with,$time_without" >> ${RESULTS_FILE}
    
    echo "  Time with reordering: ${time_with} ms"
    echo "  Time without reordering: ${time_without} ms"
    echo ""
}

# Main experiment loop
for exp_dir in Reorder_Exp_0 Reorder_Exp_1 Reorder_Exp_2; do
    exp_path="${BASE_DIR}/${exp_dir}"
    
    if [ ! -d "$exp_path" ]; then
        echo "Warning: Directory $exp_path not found, skipping..."
        continue
    fi
    
    echo "=========================================="
    echo "Processing experiment: $exp_dir"
    echo "=========================================="
    
    # Loop through all experiment folders
    for folder in $(ls -d ${exp_path}/*/ 2>/dev/null | xargs -n 1 basename); do
        folder_path="${exp_path}/${folder}"
        matrix_file="${folder_path}/original_matrix.mtx"
        
        if [ ! -f "$matrix_file" ]; then
            echo "Warning: Matrix file not found in $folder, skipping..."
            continue
        fi
        
        echo ""
        echo "Processing folder: $folder"
        echo "------------------------------------------"
        
        # Run baseline (no reordering)
        echo "Running baseline (no reordering)..."
        run_experiment "$exp_dir" "$folder" "$matrix_file" "" "" "baseline"
        
        # Get list of all permutation files
        row_files=$(find ${folder_path} -name "*_row_permutation.txt" 2>/dev/null || true)
        col_files=$(find ${folder_path} -name "*_col_permutation.txt" 2>/dev/null || true)
        
        # Collect all row and col methods
        declare -A row_methods
        declare -A col_methods
        declare -A all_methods
        
        # Collect row methods
        for rf in $row_files; do
            method=$(basename "$rf" | sed 's/_row_permutation\.txt$//')
            row_methods[$method]=$rf
            all_methods[$method]=1
        done
        
        # Collect col methods
        for cf in $col_files; do
            method=$(basename "$cf" | sed 's/_col_permutation\.txt$//')
            col_methods[$method]=$cf
            all_methods[$method]=1
        done
        
        # Run experiments for each unique method
        for method in "${!all_methods[@]}"; do
            row_file="${row_methods[$method]}"
            col_file="${col_methods[$method]}"
            
            # Special case: rcm_hpnbm uses rcm_hpnbm_row with rcm_col
            if [ "$method" == "rcm_hpnbm" ] && [ ! -z "$row_file" ] && [ -z "$col_file" ]; then
                col_file="${col_methods[rcm]}"
            fi
            
            # Determine method type for logging
            if [ ! -z "$row_file" ] && [ ! -z "$col_file" ]; then
                echo "Running with row+col reordering: $method"
            elif [ ! -z "$row_file" ]; then
                echo "Running with row-only reordering: $method"
            elif [ ! -z "$col_file" ]; then
                echo "Running with col-only reordering: $method"
            fi
            
            run_experiment "$exp_dir" "$folder" "$matrix_file" "$row_file" "$col_file" "$method"
        done
        
        # Clean up associative arrays
        unset row_methods
        unset col_methods
        unset all_methods
    done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: ${RESULTS_FILE}"
echo "=========================================="