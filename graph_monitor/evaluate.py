# evaluate.py

import os
import re
import logging
import traceback
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary functions and classes from your project files
# Ensure these paths are correct relative to where evaluate.py is run
try:
    from graph_utils import dot_to_nx, nx_to_pyg, get_sorted_node_features, get_sorted_edge_features
    from neural_net import HybridGNNAnomalyDetector
    from utils import load_checkpoint, load_running_stats, extract_timestamp_from_epoch
except ImportError as e:
    logging.error(
        f"Error importing modules. Please ensure graph_utils.py, neural_net.py, and utils.py are in your PYTHONPATH or the same directory.")
    logging.error(e)
    exit()

# --- Configuration ---
# Define paths for your model checkpoint, running statistics, and validation data
MODEL_SAVE_PATH = '/home/lu/Documents/graph_anomaly_detection/graph_monitor/model_checkpoints'
STATS_SAVE_PATH = '/home/lu/Documents/graph_anomaly_detection/graph_monitor/stats'
VALIDATION_DATA_DIR = '/home/lu/Desktop/output'  # Update with your actual path to .dot files

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


# --- Analysis Function (copied from previous response) ---
def analyze_node_reconstruction(model, dataloader, device):
    """
    Analyzes node reconstruction errors on a per-feature basis.
    Args:
        model (HybridGNNAnomalyDetector): The trained GNN model.
        dataloader (torch_geometric.loader.DataLoader): DataLoader for the validation data.
        device (torch.device): The device (CPU or CUDA) to run the analysis on.
    Returns:
        numpy.ndarray: Array of mean absolute reconstruction errors for each node feature,
                       or None if no data to analyze.
    """
    model.eval()  # Set the model to evaluation mode
    total_feature_errors = None
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # The forward pass returns: node_scores, edge_scores, global_score, node_recon, edge_recon, embedding, global_embedding
            # We only need node_recon and batch.x for this analysis
            _, _, _, node_recon, _, _, _ = model(batch)

            if batch.x is not None and node_recon is not None and batch.x.numel() > 0:
                recon_error = torch.abs(node_recon - batch.x)  # L1 error per element
                num_nodes = batch.x.size(0)

                if total_feature_errors is None:
                    total_feature_errors = torch.sum(recon_error, dim=0)
                else:
                    total_feature_errors += torch.sum(recon_error, dim=0)

                total_samples += num_nodes
            else:
                logging.warning("Skipping batch in analysis: node features or reconstruction are None/empty.")

    if total_feature_errors is not None and total_samples > 0:
        mean_feature_errors = total_feature_errors / total_samples
        return mean_feature_errors.cpu().numpy()
    else:
        logging.warning("No node reconstruction data to analyze after processing all batches.")
        return None


# --- Main Evaluation Logic ---
def main():
    # 1. Find the most recent validation graph file
    most_recent_file = None
    most_recent_timestamp = -1  # Use -1 to ensure any valid timestamp is greater

    if not os.path.exists(VALIDATION_DATA_DIR):
        logging.error(f"Validation data directory not found: {VALIDATION_DATA_DIR}")
        return

    for filename in os.listdir(VALIDATION_DATA_DIR):
        if filename.startswith("nw_graph_encoded_") and filename.endswith(".dot"):
            match = re.search(r"nw_graph_encoded_(\d+)\.dot", filename)
            if match:
                timestamp = int(match.group(1))
                if timestamp > most_recent_timestamp:
                    most_recent_timestamp = timestamp
                    most_recent_file = os.path.join(VALIDATION_DATA_DIR, filename)

    if not most_recent_file:
        logging.error(f"No .dot files found in {VALIDATION_DATA_DIR} matching the pattern 'nw_graph_encoded_*.dot'.")
        return

    logging.info(f"Using most recent validation graph: {most_recent_file}")

    # 2. Load NetworkX graph and convert to PyG Data object
    try:
        nx_graph = dot_to_nx(most_recent_file)
        if nx_graph.number_of_nodes() == 0:
            logging.error(f"Loaded graph from {most_recent_file} has no nodes. Cannot proceed with evaluation.")
            return

        # nx_to_pyg needs to handle feature extraction and initial scaling if any
        # It should return a Data object with .x (node features) and .edge_attr (edge features)
        # Ensure that nx_to_pyg is consistent with how training data was prepared
        data_pyg = nx_to_pyg(nx_graph, node_scaling='standard', edge_scaling='standard')  # Use same scaling as training

        if data_pyg.x is None or data_pyg.x.numel() == 0:
            logging.error(
                f"No node features (data.x) found in the PyG Data object from {most_recent_file}. Cannot proceed.")
            return

        # Create a DataLoader for the single graph
        validation_dataloader = DataLoader([data_pyg], batch_size=1)

    except Exception as e:
        logging.error(f"Error during graph loading or conversion for {most_recent_file}: {e}")
        logging.error(traceback.format_exc())
        return

    # 3. Initialize and Load the Model
    # We need to instantiate the model with correct dimensions before loading state_dict
    node_feature_dim = data_pyg.x.size(1)
    edge_feature_dim = data_pyg.edge_attr.size(1) if data_pyg.edge_attr is not None else 0

    model = HybridGNNAnomalyDetector(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=128,
        embedding_dim=64,
        num_gat_layers=3,
        gat_heads=4,
        recon_loss_type='mse',
        edge_recon_loss_type='mse',  # Set to 'mse' or 'none' if not reconstructing edges
        batch_size=8
    )
    model.to(DEVICE)

    # Create dummy optimizer and scheduler for loading checkpoint (their state won't be used for eval)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    try:
        start_epoch = load_checkpoint(model, optimizer, scheduler, MODEL_SAVE_PATH)
        logging.info(f"Model loaded from checkpoint: {MODEL_SAVE_PATH} (Trained for {start_epoch} steps)")
        load_running_stats(model, STATS_SAVE_PATH)
        logging.info(f"Running statistics loaded from: {STATS_SAVE_PATH}")
    except FileNotFoundError:
        logging.error(f"Model checkpoint not found at {MODEL_SAVE_PATH}. Please train the model first.")
        return
    except Exception as e:
        logging.error(f"Error loading model or running stats: {e}")
        logging.error(traceback.format_exc())
        return

    # 4. Get Sorted Node Feature Keys for display
    # Ensure this uses the same logic/order as during training
    sorted_node_feature_names = get_sorted_node_features(nx_graph)

    # 5. Perform Node Reconstruction Analysis
    logging.info("Starting node reconstruction analysis...")
    feature_errors = analyze_node_reconstruction(model, validation_dataloader, DEVICE)

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
    main()