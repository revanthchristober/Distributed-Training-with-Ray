#!/bin/bash

# ================================================
# evaluate_model.sh - Model Evaluation Script
# ================================================
# This script evaluates the trained model using a given evaluation script.
# It also logs and outputs performance metrics (accuracy, precision, recall, etc.).

# Usage: ./evaluate_model.sh [model_checkpoint] [test_data_dir]

MODEL_CHECKPOINT=${1:-"checkpoints/best_model.pth"}
TEST_DATA_DIR=${2:-"data/processed/test"}
EVALUATION_SCRIPT="src/evaluation/evaluate_model.py"

# Check if model checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Model checkpoint not found: $MODEL_CHECKPOINT"
    exit 1
fi

# Run the evaluation script
echo "Evaluating model with checkpoint: $MODEL_CHECKPOINT"
python "$EVALUATION_SCRIPT" --checkpoint "$MODEL_CHECKPOINT" --test_data_dir "$TEST_DATA_DIR"

# Check for success/failure
if [ $? -eq 0 ]; then
    echo "Model evaluation completed successfully!"
else
    echo "Model evaluation failed. Check the logs for details."
    exit 1
fi
