import os
import ray
import yaml
import logging
from ray import autoscaler
from ray.autoscaler.sdk import request_resources, get_cluster_status, get_nodes
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from typing import Dict, Optional

# Logger setup
logger = logging.getLogger("ray_cluster_utils")

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_cluster_config(cluster_config_path: str) -> Dict:
    """
    Load the Ray cluster configuration from a YAML file.

    Parameters:
    - cluster_config_path (str): Path to the Ray cluster YAML configuration file.

    Returns:
    - cluster_config (dict): The loaded cluster configuration.
    """
    if not os.path.exists(cluster_config_path):
        raise FileNotFoundError(f"Cluster configuration file not found: {cluster_config_path}")

    with open(cluster_config_path, 'r') as file:
        cluster_config = yaml.safe_load(file)

    logger.info("Cluster configuration loaded successfully.")
    return cluster_config


def start_ray_cluster(cluster_config: Dict):
    """
    Start the Ray cluster using the given configuration.

    Parameters:
    - cluster_config (dict): The Ray cluster configuration.

    Returns:
    - None
    """
    logger.info("Starting Ray cluster with the provided configuration...")
    
    ray.init(address=cluster_config["head_node"]["address"], _redis_password=cluster_config.get("redis_password", ""))
    
    logger.info("Ray cluster started successfully.")
    logger.info(f"Dashboard available at: {ray.get_dashboard_url()}")


def scale_cluster(cluster_config: Dict, min_workers: int, max_workers: int):
    """
    Scale the Ray cluster by adjusting the number of workers dynamically.

    Parameters:
    - cluster_config (dict): The Ray cluster configuration.
    - min_workers (int): Minimum number of workers to be maintained in the cluster.
    - max_workers (int): Maximum number of workers allowed in the cluster.

    Returns:
    - None
    """
    logger.info(f"Requesting resources to scale the cluster to min_workers={min_workers}, max_workers={max_workers}...")
    
    try:
        autoscaler.request_resources(num_cpus=min_workers)
        logger.info(f"Cluster scaling requested successfully.")
    except Exception as e:
        logger.error(f"Failed to scale cluster: {str(e)}")


def get_cluster_status_summary():
    """
    Get the current status of the Ray cluster.

    Returns:
    - Cluster status summary (dict)
    """
    status = get_cluster_status()
    nodes = get_nodes()

    logger.info("Cluster Status Summary:")
    logger.info(f"Total Nodes: {len(nodes)}")
    for node in nodes:
        logger.info(f"Node ID: {node['NodeID']}, Alive: {node['Alive']}, Resources: {node['Resources']}")

    return {"status": status, "nodes": nodes}


def get_node_info(node_id: str):
    """
    Get detailed information about a specific node in the Ray cluster.

    Parameters:
    - node_id (str): The unique ID of the Ray node.

    Returns:
    - node_info (dict): Information about the specified node.
    """
    nodes = get_nodes()
    node_info = next((node for node in nodes if node['NodeID'] == node_id), None)
    
    if node_info:
        logger.info(f"Node Info for {node_id}:")
        logger.info(f"Resources: {node_info['Resources']}")
        logger.info(f"Alive: {node_info['Alive']}")
    else:
        logger.warning(f"No node found with ID: {node_id}")

    return node_info


def configure_node_affinity(specific_node: str) -> NodeAffinitySchedulingStrategy:
    """
    Configure scheduling affinity for tasks to run on a specific node.

    Parameters:
    - specific_node (str): The specific node ID to pin tasks to.

    Returns:
    - NodeAffinitySchedulingStrategy: Ray scheduling strategy for node affinity.
    """
    logger.info(f"Configuring node affinity for node: {specific_node}")
    
    scheduling_strategy = NodeAffinitySchedulingStrategy(
        node_id=specific_node,
        soft=False  # Force task to be pinned to this specific node
    )
    
    return scheduling_strategy


def stop_ray_cluster():
    """
    Shut down the Ray cluster gracefully.

    Returns:
    - None
    """
    logger.info("Shutting down Ray cluster...")
    ray.shutdown()
    logger.info("Ray cluster shut down successfully.")


def create_autoscaler_config(cluster_config: Dict, new_config_path: str):
    """
    Create a new autoscaler configuration file based on existing cluster config.

    Parameters:
    - cluster_config (dict): Original Ray cluster configuration.
    - new_config_path (str): Path where the new autoscaler config will be saved.

    Returns:
    - None
    """
    logger.info("Generating a new autoscaler configuration...")
    
    autoscaler_config = {
        "cluster_name": cluster_config.get("cluster_name", "default-cluster"),
        "min_workers": cluster_config["autoscaling"]["min_workers"],
        "max_workers": cluster_config["autoscaling"]["max_workers"],
        "upscaling_speed": cluster_config["autoscaling"].get("upscaling_speed", 1.0),
        "downscaling_speed": cluster_config["autoscaling"].get("downscaling_speed", 0.8),
        "idle_timeout_minutes": cluster_config["autoscaling"].get("idle_timeout_minutes", 5)
    }

    with open(new_config_path, 'w') as file:
        yaml.dump(autoscaler_config, file)

    logger.info(f"Autoscaler configuration saved to {new_config_path}")


def monitor_cluster_health():
    """
    Monitor the health and status of the Ray cluster.

    Returns:
    - None
    """
    logger.info("Monitoring Ray cluster health...")
    nodes = get_nodes()

    unhealthy_nodes = [node for node in nodes if not node["Alive"]]
    if unhealthy_nodes:
        logger.warning(f"Detected {len(unhealthy_nodes)} unhealthy nodes.")
        for node in unhealthy_nodes:
            logger.warning(f"Unhealthy Node: {node['NodeID']} - Resources: {node['Resources']}")
    else:
        logger.info("All nodes are healthy.")


def get_node_usage_statistics():
    """
    Get resource usage statistics for each node in the cluster.

    Returns:
    - dict: Dictionary with node IDs and their respective usage statistics.
    """
    nodes = get_nodes()
    usage_statistics = {}

    for node in nodes:
        node_id = node['NodeID']
        resources = node['Resources']
        usage_statistics[node_id] = resources
        logger.info(f"Node {node_id} Resource Stats: {resources}")

    return usage_statistics


# Example usage (This block can be removed or modified when integrating with other modules):
if __name__ == "__main__":
    # Path to Ray cluster configuration file
    config_path = "/workspaces/codespaces-jupyter/Distributed-Model-Training/config/cluster.yaml"

    # Load cluster configuration
    cluster_config = load_cluster_config(config_path)

    # Start Ray cluster
    start_ray_cluster(cluster_config)

    # Scale cluster (modify min and max workers)
    scale_cluster(cluster_config, min_workers=2, max_workers=10)

    # Print cluster status
    get_cluster_status_summary()

    # Stop the cluster
    stop_ray_cluster()
