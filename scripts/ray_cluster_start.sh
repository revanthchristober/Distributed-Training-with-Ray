#!/bin/bash

# ================================================
# ray_cluster_start.sh - Ray Cluster Startup Script
# ================================================
# This script starts the Ray cluster either locally or on cloud (AWS/GCP).
# Ensure the proper Ray.io cluster configuration file is passed.

# Usage: ./ray_cluster_start.sh [local|aws|gcp] [config_file]

CLUSTER_MODE=$1
CONFIG_FILE=$2

if [ "$CLUSTER_MODE" == "local" ]; then
    echo "Starting local Ray.io cluster using Docker Compose..."
    docker-compose up -d
    echo "Ray.io local cluster started!"
    echo "Access Ray Dashboard at http://localhost:8265"
elif [ "$CLUSTER_MODE" == "aws" ]; then
    echo "Starting Ray.io cluster on AWS..."
    ray up "$CONFIG_FILE" --no-config-cache
    echo "AWS Ray cluster started!"
elif [ "$CLUSTER_MODE" == "gcp" ]; then
    echo "Starting Ray.io cluster on GCP..."
    ray up "$CONFIG_FILE" --no-config-cache
    echo "GCP Ray cluster started!"
else
    echo "Usage: ./ray_cluster_start.sh [local|aws|gcp] [config_file]"
    exit 1
fi
