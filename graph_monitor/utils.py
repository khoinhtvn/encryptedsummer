import json
import logging
import pickle
import re

import torch

from utils import *
from datetime import datetime
import os


def save_anomalies_to_file(main_data, anomalies, processed_files_count, anomaly_log_path, nx_graph=None,
                           timestamp=None):
    """Saves detected anomalies to a JSON file with details, focusing only on node anomalies.

    Args:
        main_data (torch_geometric.data.Data): The PyG Data object containing the graph data.
        anomalies (dict): Dictionary of anomaly detection results, expected to contain:
                          'node_anomalies_recon', 'node_anomalies_mlp',
                          'node_recon_errors', 'node_scores_mlp'.
        processed_files_count (int): The sequential count of files processed.
        anomaly_log_path (str): The directory path where anomaly logs will be saved.
        nx_graph (networkx.Graph, optional): The NetworkX graph corresponding to main_data,
                                             used to retrieve node IP addresses. Defaults to None.
        timestamp (str, optional): A pre-generated timestamp string. If None, current timestamp is used.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the filename for the JSON log
    filename = os.path.join(anomaly_log_path, f"anomalies_{timestamp}_update_{processed_files_count}.json")

    # Initialize the anomaly data structure, focusing only on node anomalies
    anomaly_data = {
        "timestamp": timestamp,
        "update_count": processed_files_count,
        "nodes_in_graph": main_data.num_nodes,
        "node_anomalies": []
    }

    # Prepare a dictionary to store unique node anomalies and their detection methods
    # This helps consolidate entries if a node is anomalous by multiple criteria.
    unique_node_anomalies = {}

    # Get the list of node identifiers from the NetworkX graph if available
    node_list_from_nx = list(nx_graph.nodes()) if nx_graph is not None else []

    # Process nodes identified as anomalous by reconstruction error
    for idx_tensor in anomalies.get('node_anomalies_recon', []):
        node_index = idx_tensor.item()  # Convert tensor index to Python int

        # Create initial node info, assuming detected by reconstruction
        node_info = {
            "node_id": node_index,
            "detected_by": "reconstruction",
            "recon_error": float(anomalies.get('node_recon_errors', [])[idx_tensor].item()),
            "mlp_score": float(anomalies.get('node_scores_mlp', [])[idx_tensor].item())
        }

        # Add IP if nx_graph is available and index is valid
        if nx_graph is not None and 0 <= node_index < len(node_list_from_nx):
            node_info['ip'] = str(node_list_from_nx[node_index])

        unique_node_anomalies[node_index] = node_info

    # Process nodes identified as anomalous by MLP score
    for idx_tensor in anomalies.get('node_anomalies_mlp', []):
        node_index = idx_tensor.item()  # Convert tensor index to Python int

        # If this node was already flagged by reconstruction, update its detection method to "both"
        if node_index in unique_node_anomalies:
            unique_node_anomalies[node_index]["detected_by"] = "both"
        else:
            # Otherwise, add it as a new anomaly detected by MLP
            node_info = {
                "node_id": node_index,
                "detected_by": "mlp",
                "recon_error": float(anomalies.get('node_recon_errors', [])[idx_tensor].item()),
                "mlp_score": float(anomalies.get('node_scores_mlp', [])[idx_tensor].item())
            }
            # Add IP if nx_graph is available and index is valid
            if nx_graph is not None and 0 <= node_index < len(node_list_from_nx):
                node_info['ip'] = str(node_list_from_nx[node_index])
            unique_node_anomalies[node_index] = node_info

    # Convert the collected unique anomalies into a list for the JSON output
    for node_index in sorted(unique_node_anomalies.keys()):  # Sort for consistent output
        anomaly_data["node_anomalies"].append(unique_node_anomalies[node_index])

    # Log a warning if NetworkX graph was not provided for IP mapping
    if nx_graph is None:
        logging.warning(
            "NetworkX graph not provided to save_anomalies_to_file. Node IP addresses will not be included in anomaly details.")

    # Attempt to save the anomaly data to a JSON file
    try:
        with open(filename, 'w') as f:
            json.dump(anomaly_data, f, indent=4)
        logging.info(f"Node anomaly report saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving node anomalies to file: {e}")


def save_checkpoint(model, optimizer, scheduler, processed_files_count, model_save_path, filename_prefix="checkpoint"):
    """Saves the model checkpoint with the processed files count."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(model_save_path, f"{filename_prefix}_{timestamp}_update_{processed_files_count}.pth")
    obj = {
        'epoch': processed_files_count,  # Save processed_files_count as 'epoch'
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'node_stats': model.node_stats.__dict__ if hasattr(model, 'node_stats') else None,
        'edge_stats': model.edge_stats.__dict__ if hasattr(model, 'edge_stats') else None,
    }
    torch.save(obj, filename)

    try:
        latest_checkpoint_path = os.path.join(model_save_path, "latest_checkpoint.pth")
        if os.path.exists(latest_checkpoint_path):
            os.remove(latest_checkpoint_path)
    except OSError as e:
        logging.warning(f"Could not remove previous latest checkpoint: {e}")

    torch.save(obj, os.path.join(model_save_path, "latest_checkpoint.pth"))
    logging.info(f"Checkpoint saved to {filename} and as latest_checkpoint.pth")


def load_checkpoint(model, optimizer, scheduler, model_save_path, filename="latest_checkpoint.pth"):
    filepath = os.path.join(model_save_path, filename)
    if os.path.exists(filepath):
        try:
            checkpoint = torch.load(filepath, map_location=model.device if model is not None else 'cpu',
                                    weights_only=False)
            if model is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                if hasattr(model, 'node_stats') and 'node_stats' in checkpoint and checkpoint['node_stats'] is not None:
                    model.node_stats.__dict__.update(checkpoint['node_stats'])
                elif hasattr(model, 'node_stats'):
                    logging.warning("Node statistics not found in checkpoint.")

                if hasattr(model, 'edge_stats') and 'edge_stats' in checkpoint and checkpoint['edge_stats'] is not None:
                    model.edge_stats.__dict__.update(checkpoint['edge_stats'])
                elif hasattr(model, 'edge_stats'):
                    logging.warning("Edge statistics not found in checkpoint.")

                epoch = checkpoint.get('epoch', 0)
                logging.info(f"Checkpoint loaded from {filepath} at epoch {epoch}")
                return epoch
            else:
                logging.info(f"Checkpoint metadata loaded from {filepath}")
                return checkpoint.get('epoch', 0)
        except Exception as e:
            logging.error(f"Error loading checkpoint from {filepath}: {e}")
            return 0
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        return 0


def save_running_stats(node_stats, edge_stats, stats_save_path, filename="running_stats.pkl"):
    filepath = os.path.join(stats_save_path, filename)
    stats_dict = {
        'node_stats': node_stats.__dict__ if hasattr(node_stats, '__dict__') else None,
        'edge_stats': edge_stats.__dict__ if hasattr(edge_stats, '__dict__') else None
    }
    with open(filepath, 'wb') as f:
        pickle.dump(stats_dict, f)
    logging.info(f"Running statistics saved to {filepath}")


def load_running_stats(model, stats_save_path, filename="running_stats.pkl"):
    filepath = os.path.join(stats_save_path, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                stats_dict = pickle.load(f)
            if hasattr(model, 'node_stats') and 'node_stats' in stats_dict and stats_dict['node_stats'] is not None:
                model.node_stats.__dict__.update(stats_dict['node_stats'])
            elif hasattr(model, 'node_stats'):
                logging.warning("Node statistics not found in loaded running stats.")

            if hasattr(model, 'edge_stats') and 'edge_stats' in stats_dict and stats_dict['edge_stats'] is not None:
                model.edge_stats.__dict__.update(stats_dict['edge_stats'])
            elif hasattr(model, 'edge_stats'):
                logging.warning("Edge statistics not found in loaded running stats.")

            logging.info(f"Running statistics loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading running statistics from {filepath}: {e}")
    else:
        logging.info("No running statistics file found. Starting with new statistics.")


def extract_timestamp_from_epoch(filename):
    """
    Extracts a datetime object from a filename assuming the filename (or part of it)
    contains a Unix epoch timestamp (in seconds).

    Args:
        filename (str): The name of the file.

    Returns:
        datetime or None: The extracted datetime object, or None if no match is found
                          or if the extracted value is not a valid integer.
    """
    pattern = r"_(\d+)\."  # Matches digits between an underscore and a dot (e.g., file_1678886400.txt)

    match = re.search(pattern, filename)
    if match:
        try:
            epoch_seconds = int(match.group(1))
            return datetime.fromtimestamp(epoch_seconds)
        except ValueError:
            logging.warning(f"Could not convert epoch timestamp to integer in filename: {filename}")
            return None
    return None
