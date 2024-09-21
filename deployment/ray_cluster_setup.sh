#!/bin/bash

# ================================================
# ray_cluster_setup.sh - Advanced Ray Cluster Setup
# ================================================
# Supports both local and cloud (AWS, GCP) environments.
# Provides options to start, stop, and monitor the status of Ray clusters.

# Display usage guide for the script
usage() {
    echo "Usage: $0 [local|aws|gcp] [start|stop|status] [config_file (for cloud)]"
    echo ""
    echo "Commands:"
    echo "  local start                 Start a local Ray cluster using Docker Compose."
    echo "  local stop                  Stop the local Ray cluster."
    echo "  local status                Get the status of the local Ray cluster."
    echo "  aws start <config_file>     Start an AWS Ray cluster using Ray's AWS autoscaler."
    echo "  aws stop <config_file>      Stop the AWS Ray cluster."
    echo "  aws status <config_file>    Get the status of the AWS Ray cluster."
    echo "  gcp start <config_file>     Start a GCP Ray cluster using Ray's GCP autoscaler."
    echo "  gcp stop <config_file>      Stop the GCP Ray cluster."
    echo "  gcp status <config_file>    Get the status of the GCP Ray cluster."
    exit 1
}

# Check for the number of arguments
if [ "$#" -lt 2 ]; then
    usage
fi

# ================================================
# Local Cluster Management Functions (Docker)
# ================================================

start_local_cluster() {
    echo "Starting local Ray.io cluster using Docker Compose..."
    docker-compose up -d
    echo "Ray local cluster started. Ray Dashboard is available at http://localhost:8265"
}

stop_local_cluster() {
    echo "Stopping local Ray.io cluster..."
    docker-compose down
    echo "Ray local cluster stopped."
}

status_local_cluster() {
    echo "Checking status of local Ray.io cluster..."
    docker-compose ps
}

# ================================================
# AWS Cluster Management Functions
# ================================================

start_aws_cluster() {
    local config_file="$1"
    if [ -z "$config_file" ]; then
        echo "Error: Missing AWS config file."
        usage
    fi
    echo "Starting Ray.io cluster on AWS using configuration file: $config_file"
    ray up "$config_file" --no-config-cache
    ray exec "$config_file" 'ray status'
}

stop_aws_cluster() {
    local config_file="$1"
    if [ -z "$config_file" ]; then
        echo "Error: Missing AWS config file."
        usage
    fi
    echo "Stopping Ray.io cluster on AWS using configuration file: $config_file"
    ray down "$config_file" --yes
}

status_aws_cluster() {
    local config_file="$1"
    if [ -z "$config_file" ]; then
        echo "Error: Missing AWS config file."
        usage
    fi
    ray exec "$config_file" 'ray status'
}

# ================================================
# GCP Cluster Management Functions
# ================================================

start_gcp_cluster() {
    local config_file="$1"
    if [ -z "$config_file" ]; then
        echo "Error: Missing GCP config file."
        usage
    fi
    echo "Starting Ray.io cluster on GCP using configuration file: $config_file"
    ray up "$config_file" --no-config-cache
    ray exec "$config_file" 'ray status'
}

stop_gcp_cluster() {
    local config_file="$1"
    if [ -z "$config_file" ]; then
        echo "Error: Missing GCP config file."
        usage
    fi
    echo "Stopping Ray.io cluster on GCP using configuration file: $config_file"
    ray down "$config_file" --yes
}

status_gcp_cluster() {
    local config_file="$1"
    if [ -z "$config_file" ]; then
        echo "Error: Missing GCP config file."
        usage
    fi
    ray exec "$config_file" 'ray status'
}

# ================================================
# Main Logic to Handle Different Cluster Types
# ================================================

case "$1" in
    local)
        case "$2" in
            start)
                start_local_cluster
                ;;
            stop)
                stop_local_cluster
                ;;
            status)
                status_local_cluster
                ;;
            *)
                usage
                ;;
        esac
        ;;
    aws)
        case "$2" in
            start)
                start_aws_cluster "$3"
                ;;
            stop)
                stop_aws_cluster "$3"
                ;;
            status)
                status_aws_cluster "$3"
                ;;
            *)
                usage
                ;;
        esac
        ;;
    gcp)
        case "$2" in
            start)
                start_gcp_cluster "$3"
                ;;
            stop)
                stop_gcp_cluster "$3"
                ;;
            status)
                status_gcp_cluster "$3"
                ;;
            *)
                usage
                ;;
        esac
        ;;
    *)
        usage
        ;;
esac
