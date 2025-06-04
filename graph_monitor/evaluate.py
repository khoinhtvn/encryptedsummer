# evaluate.py

import logging
import os
import re
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

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
VALIDATION_DATA_DIR = '/home/lu/Desktop/output_ssl_bruteforce'
EMBEDDING_SAVE_PATH = '/home/lu/Documents/graph_anomaly_detection/graph_monitor/embeddings'
os.makedirs(EMBEDDING_SAVE_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


def visualize_embeddings_3d(embeddings, labels, filename="embeddings_3d.png"):
    """Visualizes node embeddings in 3D using t-SNE."""
    logging.info("Starting 3D embedding visualization...")
    tsne = TSNE(n_components=3, random_state=42, n_iter=300, perplexity=30)
    try:
        reduced_embeddings = tsne.fit_transform(embeddings)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=labels,
                             cmap='viridis')
        ax.set_title("Node Embeddings Visualization (t-SNE 3D)")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
        # Add a colorbar for the labels if they are numerical
        if np.issubdtype(np.array(labels).dtype, np.number):
            cbar = fig.colorbar(scatter)
            cbar.set_label('Label')
        else:
            # Create a legend for categorical labels
            unique_labels = np.unique(labels)
            for label in unique_labels:
                ax.scatter(reduced_embeddings[labels == label, 0], reduced_embeddings[labels == label, 1],
                           reduced_embeddings[labels == label, 2], label=label)
            ax.legend()

        filepath = os.path.join(EMBEDDING_SAVE_PATH, filename)
        plt.savefig(filepath)
        logging.info(f"3D embeddings visualization saved to {filepath}")
        plt.close()
    except Exception as e:
        logging.error(f"Error during 3D t-SNE or plotting: {e}")
        logging.error(traceback.format_exc())


# --- Analysis Function (modified to call 3D visualization) ---
def analyze_node_reconstruction(model, dataloader, device, visualize_2d=False, visualize_3d=False):
    """
    Analyzes node reconstruction errors on a per-feature basis and optionally visualizes embeddings in 2D or 3D.
    Args:
        model (HybridGNNAnomalyDetector): The trained GNN model.
        dataloader (torch_geometric.loader.DataLoader): DataLoader for the validation data.
        device (torch.device): The device (CPU or CUDA) to run the analysis on.
        visualize_2d (bool): Whether to visualize the node embeddings in 2D.
        visualize_3d (bool): Whether to visualize the node embeddings in 3D.
    Returns:
        numpy.ndarray: Array of mean absolute reconstruction errors for each node feature,
                       or None if no data to analyze.
    """
    model.eval()  # Set the model to evaluation mode
    total_feature_errors = None
    total_embeddings = []
    node_labels = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # The forward pass returns: node_scores, edge_scores, global_score, node_recon, edge_recon, embedding, global_embedding
            # We need node_recon, batch.x, and embedding for this analysis
            _, _, _, node_recon, _, embedding, _ = model(batch)

            if batch.x is not None and node_recon is not None and batch.x.numel() > 0:
                recon_error = torch.abs(node_recon - batch.x)  # L1 error per element
                num_nodes = batch.x.size(0)

                if total_feature_errors is None:
                    total_feature_errors = torch.sum(recon_error, dim=0)
                else:
                    total_feature_errors += torch.sum(recon_error, dim=0)

                total_samples += num_nodes
                total_embeddings.append(embedding.cpu().numpy())
                node_labels.extend([0] * num_nodes)  # Assuming all validation data is 'normal' for visualization

            else:
                logging.warning("Skipping batch in analysis: node features or reconstruction are None/empty.")

    if total_feature_errors is not None and total_samples > 0:
        mean_feature_errors = total_feature_errors / total_samples
        if total_embeddings:
            all_embeddings = np.concatenate(total_embeddings, axis=0)
            all_labels = np.array(node_labels)
            if visualize_2d:
                visualize_embeddings(all_embeddings, all_labels, filename="validation_embeddings_2d.png")
            if visualize_3d:
                visualize_embeddings_3d(all_embeddings, all_labels, filename="validation_embeddings_3d.png")
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

    # 5. Perform Node Reconstruction Analysis and Embedding Visualization
    logging.info("Starting node reconstruction analysis and embedding visualization...")
    feature_errors = analyze_node_reconstruction(model, validation_dataloader, DEVICE, visualize_2d=False,
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
    main()
