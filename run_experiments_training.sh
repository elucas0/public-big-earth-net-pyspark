#!/bin/bash
#
# BigEarthNet Training Experiments
# ================================
# This script runs training experiments varying:
# - Dataset fractions: 1%, 2%, 5%, 10%
# - Number of epochs: 10
# - GPU configurations: 1 to 4 GPUs
#
# Each combination produces a separate training run.
#
# Output structure:
#  s3://ubs-homes/erasmus/elouann/experiments/data_fraction_training/
#   ├── 1/
#   │   ├── mirrored/
#   │   └── one_device/
#   ├── 2/
#   ├── 5/
#   └── 10/
#
# Usage:
#   chmod +x run_experiments_training.sh
#   ./run_experiments_training.sh
#
echo "========================================"
echo "BigEarthNet Training Experiments"
echo "========================================"
echo ""

# List of dataset fractions (in %) to be used for training
FRACTIONS=(1 2 5 10)

# List of epoch values (can be expanded for experiments)
EPOCHS_LIST=(10)

# Different GPU configurations to test:
# - single GPU
# - 2 GPUs
# - 3 GPUs
# - 4 GPUs
GPU_SETS=(
    "/gpu:0"
    "/gpu:0 /gpu:1"
    "/gpu:0 /gpu:1 /gpu:2"
    "/gpu:0 /gpu:1 /gpu:2 /gpu:3"
)

# Counter to track experiment progress
COUNT=1

# Total number of runs (product of all parameters)
TOTAL=$(( ${#FRACTIONS[@]} * ${#EPOCHS_LIST[@]} * ${#GPU_SETS[@]} ))

# Loop over all dataset fractions
for frac in "${FRACTIONS[@]}"; do
    # Loop over all epoch values
    for ep in "${EPOCHS_LIST[@]}"; do
        # Loop over all GPU configurations
        for gpu_config in "${GPU_SETS[@]}"; do
            
            # Convert the GPU configuration string into an array
            # This is to count how many GPUs are being used
            read -r -a gpu_array <<< "$gpu_config"
            
            # Select the training strategy based on the number of GPUs
            # - one_device: single GPU
            # - mirrored: multiple GPUs
            if [ ${#gpu_array[@]} -gt 1 ]; then
                STRAT="mirrored"
            else
                STRAT="one_device"
            fi

            # Print the current experiment configuration and progress
            echo "[$COUNT/$TOTAL] Running: Frac=$frac%, Ep=$ep, Strat=$STRAT, GPUs=[$gpu_config]"
            
            # Run the training script using uv with the selected parameters
            uv run training_model.py \
                --fraction="$frac" \
                --epochs="$ep" \
                --strategy="$STRAT" \
                --gpus $gpu_config
            
            # Check whether the execution was successful
            if [ $? -eq 0 ]; then
                echo "Success"
            else
                echo "Failed: Frac $frac, GPUs $gpu_config"
            fi
            
            echo "------------------------------------------"
            
            # Increment experiment counter
            COUNT=$((COUNT + 1))
        done
    done
done
