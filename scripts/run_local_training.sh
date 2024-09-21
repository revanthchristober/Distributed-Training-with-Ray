#!/bin/bash

# ================================================
# run_local_training.sh - Local Training Script
# ================================================
# This script sets up and runs the local training for the model.
# It will automatically detect GPU availability and utilize it.

# Usage: ./run_local_training.sh

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 if available
TRAINING_SCRIPT="src/training/local_training.py"

echo "Running local training script: $TRAINING_SCRIPT"

# Check if GPU is available
if command -v nvidia-smi &> /dev/null && [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -gt 0 ]; then
    echo "GPU detected, using CUDA..."
else
    echo "No GPU detected, using CPU..."
fi

# Run the training script with Python
python "$TRAINING_SCRIPT"

# Check for success/failure
if [ $? -eq 0 ]; then
    echo "Local training completed successfully!"
else
    echo "Local training failed. Check the logs for details."
    exit 1
fi
