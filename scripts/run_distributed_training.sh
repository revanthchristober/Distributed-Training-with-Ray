#!/bin/bash

# ================================================
# run_distributed_training.sh - Distributed Training Script
# ================================================
# This script sets up and runs distributed training across a Ray cluster.
# It can be run either locally across multiple cores or on a cluster.

# Usage: ./run_distributed_training.sh [config_file]

# Variables
CONFIG_FILE=${1:-"config/ray_cluster_config.yaml"}
TRAINING_SCRIPT="src/training/distributed_training.py"

# Start distributed training
echo "Starting distributed training with Ray.io"
echo "Using configuration file: $CONFIG_FILE"

# Run the training script with Ray
ray exec "$CONFIG_FILE" "python $TRAINING_SCRIPT"

# Check for success/failure
if [ $? -eq 0 ]; then
    echo "Distributed training completed successfully!"
else
    echo "Distributed training failed. Check the logs for details."
    exit 1
fi
