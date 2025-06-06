import json
import logging
import pickle
import re
import traceback

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from utils import *
from datetime import datetime
import os


def visualize_embeddings_3d(embeddings, node_ips, anomalous_indices_tensor,
                            save_path, timestamp=None, filename_prefix="embeddings_3d"):
    """
    Visualizes node embeddings in 3D using t-SNE, highlighting anomalous nodes,
    and includes timestamp in filename.

    Args:
        embeddings (np.ndarray): The node embeddings to visualize, shape (n_samples, n_features).
        node_ips (list): List of IP addresses (or node identifiers) corresponding
                         to the node indices in 'embeddings'. Used for total count.
        anomalous_indices_tensor (torch.Tensor or np.ndarray): Tensor/array containing
                                indices (0-based) of anomalous nodes.
        save_path (str): The directory path where the visualization image will be saved.
        timestamp (str, optional): A specific timestamp string (e.g., "20250606_173000").
                                   If None, the current UTC time will be used.
        filename_prefix (str): Prefix for the saved filename (e.g., "embeddings_3d").
    """
    logging.info("Starting 3D embedding visualization...")

    if timestamp is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_UTC")
    else:
        timestamp_str = str(timestamp) # Ensure it's a string for filename

    n_samples = embeddings.shape[0]

    # --- Perplexity Adjustment for t-SNE ---
    # t-SNE requires perplexity < n_samples. Minimum samples for t-SNE is usually 5.
    default_perplexity = 30
    perplexity_val = default_perplexity

    if n_samples < 5:
        logging.warning(f"Not enough samples ({n_samples}) for meaningful t-SNE. Skipping 3D embedding visualization.")
        return
    elif n_samples <= default_perplexity:
        # If n_samples is small, set perplexity to n_samples - 1 (must be > 0)
        perplexity_val = max(1, n_samples - 1)
        logging.warning(
            f"Number of samples ({n_samples}) is small. Adjusting t-SNE perplexity to {perplexity_val}."
        )
    # No 'else' needed, as perplexity_val is already default_perplexity

    try:
        # Initialize t-SNE. Use 'max_iter' instead of 'n_iter' to avoid FutureWarning.
        tsne = TSNE(n_components=3, random_state=42, max_iter=500, perplexity=perplexity_val)
        reduced_embeddings = tsne.fit_transform(embeddings)

        # --- Prepare Plot Labels (Normal vs. Anomalous) ---
        # Ensure anomalous_indices_tensor is a NumPy array for consistent indexing
        anomalous_indices_list = []
        if isinstance(anomalous_indices_tensor, torch.Tensor):
            anomalous_indices_list = anomalous_indices_tensor.cpu().numpy()
        elif isinstance(anomalous_indices_tensor, np.ndarray):
            anomalous_indices_list = anomalous_indices_tensor
        elif anomalous_indices_tensor is not None: # Handle cases where it might be None
            logging.warning(
                f"Unexpected type for anomalous_indices_tensor: {type(anomalous_indices_tensor)}. "
                "Expected torch.Tensor or numpy.ndarray. No anomalies will be highlighted."
            )

        # Initialize all nodes as normal (0)
        plot_labels = np.zeros(len(node_ips), dtype=int) # Use len(node_ips) for total nodes
        # Mark anomalous nodes with label 1
        for idx in anomalous_indices_list:
            if 0 <= idx < len(plot_labels):
                plot_labels[idx] = 1 # Mark as anomalous
            else:
                logging.warning(f"Anomalous index {idx} out of bounds for node_ips length {len(node_ips)}.")

        # --- Set up the 3D plot ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Define colors: Blue for normal, Red for anomalous
        colors = ['blue', 'red']
        cmap = matplotlib.colors.ListedColormap(colors)
        bounds = [0, 1, 2] # Boundaries for the colormap (0 -> blue, 1 -> red)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # Create the 3D scatter plot
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
                             c=plot_labels, cmap=cmap, norm=norm, alpha=0.7, s=50) # 's' for marker size

        # --- Set plot title and axis labels ---
        ax.set_title(f"Node Embeddings Visualization (t-SNE 3D)\nGraph Timestamp: {timestamp_str}")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")

        # --- Create Custom Legend ---
        from matplotlib.lines import Line2D # Import here to avoid potential circular dependencies if at top
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal Node',
                   markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Anomalous Node',
                   markerfacecolor='red', markersize=10)
        ]
        ax.legend(handles=legend_elements, title="Node Type", loc='upper left')

        # --- Save the plot to a file ---
        os.makedirs(save_path, exist_ok=True) # Ensure save directory exists
        filepath = os.path.join(save_path, f"{filename_prefix}_{timestamp_str}.png")

        plt.savefig(filepath, dpi=300) # Save with higher DPI for better quality
        logging.info(f"3D embeddings visualization saved to {filepath}")
        plt.close(fig) # Close the plot to free up memory

    except Exception as e:
        logging.error(f"Error during 3D t-SNE or plotting: {e}")
        logging.error(traceback.format_exc()) # Log the full traceback for detailed debugging


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
