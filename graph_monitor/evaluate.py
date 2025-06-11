import logging
import os
import re
import traceback
import argparse  # Import argparse for command-line arguments
from datetime import datetime

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn

from visualization import visualize_embeddings_3d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary functions and classes from your project files
try:
    from graph_utils import dot_to_nx, nx_to_pyg, get_sorted_node_features, get_sorted_edge_features
    from neural_net import NodeGNNAnomalyDetector
    from utils import load_checkpoint, load_running_stats
except ImportError as e:
    logging.error(
        f"Error importing modules. Please ensure graph_utils.py, neural_net.py, and utils.py are in your PYTHONPATH or the same directory.")
    logging.error(e)
    exit()

# --- Configuration (These will now be default values, overridden by command-line arguments) ---
DEFAULT_MODEL_SAVE_PATH = '/home/lu/Documents/graph_anomaly_detection/graph_monitor/model_checkpoints'
DEFAULT_STATS_SAVE_PATH = '/home/lu/Documents/graph_anomaly_detection/graph_monitor/stats'
DEFAULT_EMBEDDING_SAVE_PATH = '/home/lu/Documents/graph_anomaly_detection/graph_monitor/embeddings'
DEFAULT_VALIDATION_DATA_DIR = '/home/lu/Desktop/output_ssl_bruteforce'  # Default for main function argument

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")

# --- Analysis Function ---
def analyze_node_reconstruction(model, dataloader, device, graph_timestamp, node_ips_list,
                                anomalous_nodes_indices, embedding_save_path, visualize_3d=False):
    """
    Analyzes node reconstruction errors on a per-feature basis and optionally visualizes embeddings in 3D.
    Args:
        model (NodeGNNAnomalyDetector): The trained GNN model.
        dataloader (torch_geometric.loader.DataLoader): DataLoader for the validation data.
        device (torch.device): The device (CPU or CUDA) to run the analysis on.
        graph_timestamp (int): Timestamp of the graph being analyzed, for use in filenames.
        node_ips_list (list): List of IP addresses corresponding to node indices.
        anomalous_nodes_indices (np.ndarray): NumPy array of indices of anomalous nodes.
        embedding_save_path (str): Path to save embedding visualizations.
        visualize_3d (bool): Whether to visualize the node embeddings in 3D.
    Returns:
        numpy.ndarray: Array of mean absolute reconstruction errors for each node feature,
                        or None if no data to analyze.
    """
    model.eval()
    total_feature_errors = None
    total_embeddings = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, node_recon, embedding = model(batch)

            if batch.x is not None and node_recon is not None and batch.x.numel() > 0:
                normalized_batch = model._normalize_features(batch.clone())
                recon_error = torch.abs(node_recon - normalized_batch.x)

                num_nodes = batch.x.size(0)

                if total_feature_errors is None:
                    total_feature_errors = torch.sum(recon_error, dim=0)
                else:
                    total_feature_errors += torch.sum(recon_error, dim=0)

                total_samples += num_nodes
                total_embeddings.append(embedding.cpu().numpy())
            else:
                logging.warning("Skipping batch in analysis: node features or reconstruction are None/empty.")

    if total_feature_errors is not None and total_samples > 0:
        mean_feature_errors = total_feature_errors / total_samples
        if total_embeddings:
            all_embeddings = np.concatenate(total_embeddings, axis=0)
            if visualize_3d:
                visualize_embeddings_3d(all_embeddings, node_ips_list, anomalous_nodes_indices,
                                        save_path=embedding_save_path,
                                        timestamp=graph_timestamp,
                                        filename_prefix="validation_embeddings_3d")
        return mean_feature_errors.cpu().numpy()
    else:
        logging.warning("No node reconstruction data to analyze after processing all batches.")
        return None


# --- Main Evaluation Logic ---
def main(model_save_path: str, stats_save_path: str, embedding_save_path: str, validation_data_dir: str):
    """
    Main function to perform graph evaluation and anomaly detection.

    Args:
        model_save_path (str): Directory where the trained model checkpoint is saved.
        stats_save_path (str): Directory where running statistics are saved.
        embedding_save_path (str): Directory where embedding visualizations will be saved.
        validation_data_dir (str): Directory containing the validation graph files.
    """
    # Ensure embedding save directory exists
    os.makedirs(embedding_save_path, exist_ok=True)

    # 1. Find the most recent validation graph file
    most_recent_file = None
    most_recent_timestamp = -1

    if not os.path.exists(validation_data_dir):
        logging.error(f"Validation data directory not found: {validation_data_dir}")
        return

    for filename in os.listdir(validation_data_dir):
        if filename.startswith("nw_graph_encoded_") and filename.endswith(".dot"):
            match = re.search(r"nw_graph_encoded_(\d+)\.dot", filename)
            if match:
                timestamp = int(match.group(1))
                if timestamp > most_recent_timestamp:
                    most_recent_timestamp = timestamp
                    most_recent_file = os.path.join(validation_data_dir, filename)

    if not most_recent_file:
        logging.error(f"No .dot files found in {validation_data_dir} matching the pattern 'nw_graph_encoded_*.dot'.")
        return

    logging.info(f"Using most recent validation graph: {most_recent_file}")

    # 2. Load NetworkX graph and convert to PyG Data object
    try:
        nx_graph = dot_to_nx(most_recent_file)
        if nx_graph.number_of_nodes() == 0:
            logging.error(f"Loaded graph from {most_recent_file} has no nodes. Cannot proceed with evaluation.")
            return

        data_pyg = nx_to_pyg(nx_graph)

        if data_pyg.x is None or data_pyg.x.numel() == 0:
            logging.error(
                f"No node features (data.x) found in the PyG Data object from {most_recent_file}. Cannot proceed.")
            return

        validation_dataloader = DataLoader([data_pyg], batch_size=1)

    except Exception as e:
        logging.error(f"Error during graph loading or conversion for {most_recent_file}: {e}")
        logging.error(traceback.format_exc())
        return

    # 3. Initialize and Load the Model
    node_feature_dim = data_pyg.x.size(1)
    edge_feature_dim = data_pyg.edge_attr.size(1) if data_pyg.edge_attr is not None else 0

    model = NodeGNNAnomalyDetector(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=128,
        embedding_dim=64,
        num_gat_layers=3,
        gat_heads=4,
        recon_loss_type='mse',
        use_batch_norm=True,
        use_residual=True,
        batch_size=1,
        export_period=5,
        export_dir=embedding_save_path  # Pass export directory to the model
    )
    model.to(DEVICE)

    # Initialize optimizer and scheduler (needed for loading checkpoint, even if not training)
    # These are placeholders if load_checkpoint expects them.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    try:
        start_epoch = load_checkpoint(model, optimizer, scheduler, model_save_path)
        logging.info(f"Model loaded from checkpoint: {model_save_path} (Trained for {start_epoch} steps)")

        # Load running statistics into the model's internal stats objects
        load_running_stats(model, stats_save_path)
        logging.info(f"Running statistics loaded into model from: {stats_save_path}")

    except FileNotFoundError as fnfe:
        logging.error(f"Required file not found: {fnfe}. Please ensure model and stats files exist at specified paths.")
        return
    except Exception as e:
        logging.error(f"Error loading model or running stats: {e}")
        logging.error(traceback.format_exc())
        return

    # 4. Get Sorted Node and Edge Feature Keys for display
    sorted_node_feature_names = get_sorted_node_features(nx_graph)
    sorted_edge_feature_names = get_sorted_edge_features(nx_graph)
    node_ips = list(nx_graph.nodes())

    # --- ANOMALY DETECTION ---
    logging.info("\n--- Starting Anomaly Detection ---")
    anomalous_nodes_indices_recon = np.array([], dtype=int)

    try:
        anomaly_results = model.detect_anomalies(data_pyg)

        if anomaly_results:
            logging.info(f"Anomaly detection completed for graph: {most_recent_file}")

            anomalous_nodes_indices_recon = anomaly_results['node_anomalies_recon']
            node_recon_errors = anomaly_results['node_recon_errors']
            node_scores_mlp = anomaly_results['node_scores_mlp']
            anomalous_nodes_indices_mlp = anomaly_results['node_anomalies_mlp']

            logging.info(f"Total Nodes: {data_pyg.x.shape[0]}")
            logging.info(f"Anomalous Nodes (Reconstruction Error): {len(anomalous_nodes_indices_recon)}")
            logging.info(f"Anomalous Nodes (MLP Score): {len(anomalous_nodes_indices_mlp)}")

            if len(anomalous_nodes_indices_recon) > 0:
                logging.info("--- Details for Anomalous Nodes (by Reconstruction Error) ---")
                for node_index in anomalous_nodes_indices_recon:
                    node_ip = "N/A"
                    if 0 <= node_index < len(node_ips):
                        node_ip = node_ips[node_index]
                    recon_error_val = node_recon_errors[node_index]
                    mlp_score_val = node_scores_mlp[node_index]
                    logging.info(
                        f"Node Index {node_index} (IP: {node_ip}): Recon Error = {recon_error_val:.6f}, MLP Score = {mlp_score_val:.6f}")

            if len(anomalous_nodes_indices_mlp) > 0:
                logging.info("--- Details for Anomalous Nodes (by MLP Score) ---")
                for node_index in anomalous_nodes_indices_mlp:
                    node_ip = "N/A"
                    if 0 <= node_index < len(node_ips):
                        node_ip = node_ips[node_index]
                    recon_error_val = node_recon_errors[node_index]
                    mlp_score_val = node_scores_mlp[node_index]
                    logging.info(
                        f"Node Index {node_index} (IP: {node_ip}): Recon Error = {recon_error_val:.6f}, MLP Score = {mlp_score_val:.6f}")
        else:
            logging.info("No anomaly results returned.")

    except Exception as e:
        logging.error(f"Error during anomaly detection: {e}")
        logging.error(traceback.format_exc())

    # 5. Perform Node Reconstruction Analysis and Embedding Visualization
    logging.info("Starting node reconstruction analysis and embedding visualization...")
    feature_errors = analyze_node_reconstruction(model, validation_dataloader, DEVICE, most_recent_timestamp,
                                                 node_ips, anomalous_nodes_indices_recon,
                                                 embedding_save_path,  # Pass the path here
                                                 visualize_3d=True)

    if feature_errors is not None:
        logging.info("\n--- Node Reconstruction Error Analysis ---")
        for i, error in enumerate(feature_errors):
            if i < len(sorted_node_feature_names):
                logging.info(
                    f"Feature '{sorted_node_feature_names[i]}': Mean Absolute Reconstruction Error = {error:.6f}")
            else:
                logging.info(f"Feature {i} (Name not available): Mean Absolute Reconstruction Error = {error:.6f}")
    else:
        logging.info("Node reconstruction analysis could not be performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GNN Anomaly Detector and visualize embeddings.")
    parser.add_argument('--model_path', type=str,
                        default=DEFAULT_MODEL_SAVE_PATH,
                        help=f"Path to the directory containing the trained model checkpoint. Default: {DEFAULT_MODEL_SAVE_PATH}")
    parser.add_argument('--stats_path', type=str,
                        default=DEFAULT_STATS_SAVE_PATH,
                        help=f"Path to the directory containing the running statistics. Default: {DEFAULT_STATS_SAVE_PATH}")
    parser.add_argument('--embedding_path', type=str,
                        default=DEFAULT_EMBEDDING_SAVE_PATH,
                        help=f"Path to the directory where 3D embedding visualizations will be saved. Default: {DEFAULT_EMBEDDING_SAVE_PATH}")
    parser.add_argument('--data_dir', type=str,
                        default=DEFAULT_VALIDATION_DATA_DIR,
                        help=f"Path to the directory containing the validation graph files (*.dot). Default: {DEFAULT_VALIDATION_DATA_DIR}")

    args = parser.parse_args()

    main(model_save_path=args.model_path,
         stats_save_path=args.stats_path,
         embedding_save_path=args.embedding_path,
         validation_data_dir=args.data_dir)