# evaluate.py

import logging
import os
import re
import traceback

import matplotlib.colors  # Import matplotlib.colors for custom colormap
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary functions and classes from your project files
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
EMBEDDING_SAVE_PATH = '/home/lu/Documents/graph_anomaly_detection/graph_monitor/embeddings'
os.makedirs(EMBEDDING_SAVE_PATH, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {DEVICE}")


def visualize_embeddings_3d(embeddings, node_ips, anomalous_indices_tensor, timestamp, filename_prefix="embeddings_3d"):
    """
    Visualizes node embeddings in 3D using t-SNE, highlighting anomalous nodes,
    and including timestamp in filename.

    Args:
        embeddings (np.ndarray): The node embeddings to visualize.
        node_ips (list): List of IP addresses corresponding to the node indices.
        anomalous_indices_tensor (torch.Tensor): Tensor containing indices of anomalous nodes.
        timestamp (int): Timestamp of the graph, used for the plot title and filename.
        filename_prefix (str): Prefix for the saved filename.
    """
    logging.info("Starting 3D embedding visualization...")
    # Initialize t-SNE with 3 components for 3D visualization
    tsne = TSNE(n_components=3, random_state=42, n_iter=300, perplexity=30)
    try:
        # Reduce the dimensionality of embeddings using t-SNE
        reduced_embeddings = tsne.fit_transform(embeddings)

        # Ensure anomalous_indices_tensor is a NumPy array for consistent indexing
        # It might be a torch.Tensor or already a numpy.ndarray depending on detect_anomalies output
        if isinstance(anomalous_indices_tensor, torch.Tensor):
            anomalous_indices_list = anomalous_indices_tensor.cpu().numpy()
        elif isinstance(anomalous_indices_tensor, np.ndarray):
            anomalous_indices_list = anomalous_indices_tensor
        else:
            logging.error(
                f"Unexpected type for anomalous_indices_tensor: {type(anomalous_indices_tensor)}. Expected torch.Tensor or numpy.ndarray.")
            return

        # Create numerical labels for plotting: 0 for normal, 1 for anomalous
        # Initialize all nodes as normal (0)
        plot_labels = np.zeros(len(node_ips), dtype=int)
        # Mark anomalous nodes with label 1
        for idx in anomalous_indices_list:
            if 0 <= idx < len(plot_labels):
                plot_labels[idx] = 1  # Mark as anomalous

        # Set up the 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Define colors: Blue for normal, Red for anomalous
        colors = ['blue', 'red']
        # Create a colormap from the defined colors
        cmap = matplotlib.colors.ListedColormap(colors)
        # Define boundaries for the colormap to map labels (0, 1) to colors
        bounds = [0, 1, 2]
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # Create the 3D scatter plot
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
                             c=plot_labels, cmap=cmap, norm=norm, alpha=0.7)

        # Set plot title and axis labels
        ax.set_title(f"Node Embeddings Visualization (t-SNE 3D) - Timestamp: {timestamp}")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")

        # Create custom legend handles to explain the colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal Node', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Anomalous Node', markerfacecolor='red', markersize=10)
        ]
        ax.legend(handles=legend_elements, title="Node Type", loc='upper left')

        # Save the plot to a file
        filepath = os.path.join(EMBEDDING_SAVE_PATH, f"{filename_prefix}_{timestamp}.png")
        plt.savefig(filepath)
        logging.info(f"3D embeddings visualization saved to {filepath}")
        plt.close()  # Close the plot to free up memory
    except Exception as e:
        logging.error(f"Error during 3D t-SNE or plotting: {e}")
        logging.error(traceback.format_exc())


# --- Analysis Function (modified to call 3D visualization) ---
def analyze_node_reconstruction(model, dataloader, device, graph_timestamp, node_ips_list,
                                anomalous_nodes_indices_tensor, visualize_2d=False, visualize_3d=False):
    """
    Analyzes node reconstruction errors on a per-feature basis and optionally visualizes embeddings in 2D or 3D.
    Args:
        model (HybridGNNAnomalyDetector): The trained GNN model.
        dataloader (torch_geometric.loader.DataLoader): DataLoader for the validation data.
        device (torch.device): The device (CPU or CUDA) to run the analysis on.
        graph_timestamp (int): Timestamp of the graph being analyzed, for use in filenames.
        node_ips_list (list): List of IP addresses corresponding to node indices.
        anomalous_nodes_indices_tensor (torch.Tensor): Tensor of indices of anomalous nodes.
        visualize_2d (bool): Whether to visualize the node embeddings in 2D.
        visualize_3d (bool): Whether to visualize the node embeddings in 3D.
    Returns:
        numpy.ndarray: Array of mean absolute reconstruction errors for each node feature,
                       or None if no data to analyze.
    """
    model.eval()  # Set the model to evaluation mode
    total_feature_errors = None
    total_embeddings = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # Perform a forward pass through the model to get node reconstruction and embedding
            _, _, _, node_recon, _, embedding, _ = model(batch)

            if batch.x is not None and node_recon is not None and batch.x.numel() > 0:
                # Calculate L1 reconstruction error per element
                recon_error = torch.abs(node_recon - batch.x)
                num_nodes = batch.x.size(0)

                # Aggregate feature-wise errors across all nodes in the batch
                if total_feature_errors is None:
                    total_feature_errors = torch.sum(recon_error, dim=0)
                else:
                    total_feature_errors += torch.sum(recon_error, dim=0)

                total_samples += num_nodes
                # Store embeddings (move to CPU and convert to NumPy)
                total_embeddings.append(embedding.cpu().numpy())

            else:
                logging.warning("Skipping batch in analysis: node features or reconstruction are None/empty.")

    if total_feature_errors is not None and total_samples > 0:
        # Calculate mean feature errors
        mean_feature_errors = total_feature_errors / total_samples
        if total_embeddings:
            # Concatenate all collected embeddings into a single array
            all_embeddings = np.concatenate(total_embeddings, axis=0)
            # If 3D visualization is requested, call the visualization function
            if visualize_3d:
                visualize_embeddings_3d(all_embeddings, node_ips_list, anomalous_nodes_indices_tensor, graph_timestamp,
                                        filename_prefix="validation_embeddings_3d")
        return mean_feature_errors.cpu().numpy()
    else:
        logging.warning("No node reconstruction data to analyze after processing all batches.")
        return None


# --- Main Evaluation Logic ---
def main(validation_data_dir: str):  # VALIDATION_DATA_DIR is now a parameter
    # 1. Find the most recent validation graph file
    most_recent_file = None
    most_recent_timestamp = -1

    if not os.path.exists(validation_data_dir):
        logging.error(f"Validation data directory not found: {validation_data_dir}")
        return

    # Iterate through files to find the latest graph by timestamp
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
        nx_graph = dot_to_nx(most_recent_file)  # Keep nx_graph here!
        if nx_graph.number_of_nodes() == 0:
            logging.error(f"Loaded graph from {most_recent_file} has no nodes. Cannot proceed with evaluation.")
            return

        # Convert NetworkX graph to PyTorch Geometric Data object
        data_pyg = nx_to_pyg(nx_graph, node_scaling='standard', edge_scaling='standard')

        if data_pyg.x is None or data_pyg.x.numel() == 0:
            logging.error(
                f"No node features (data.x) found in the PyG Data object from {most_recent_file}. Cannot proceed.")
            return

        # Create a DataLoader for the PyG Data object (batch size 1 for single graph)
        validation_dataloader = DataLoader([data_pyg], batch_size=1)

    except Exception as e:
        logging.error(f"Error during graph loading or conversion for {most_recent_file}: {e}")
        logging.error(traceback.format_exc())
        return

    # 3. Initialize and Load the Model
    node_feature_dim = data_pyg.x.size(1)
    edge_feature_dim = data_pyg.edge_attr.size(1) if data_pyg.edge_attr is not None else 0

    # Initialize the HybridGNNAnomalyDetector model
    model = HybridGNNAnomalyDetector(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        hidden_dim=128,
        embedding_dim=64,
        num_gat_layers=3,
        gat_heads=4,
        recon_loss_type='mse',
        edge_recon_loss_type='bce',
        batch_size=8
    )
    model.to(DEVICE)  # Move model to the specified device (CPU/CUDA)

    # Initialize optimizer and scheduler (even if not training, needed for loading checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    # Load the trained model checkpoint and running statistics
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

    # 4. Get Sorted Node and Edge Feature Keys for display
    sorted_node_feature_names = get_sorted_node_features(nx_graph)
    sorted_edge_feature_names = get_sorted_edge_features(nx_graph)
    # Get the list of node IPs from the NetworkX graph. This order matches PyG node indexing.
    node_ips = list(nx_graph.nodes())

    # --- ANOMALY DETECTION ---
    logging.info("\n--- Starting Anomaly Detection ---")
    ANOMALY_THRESHOLD_SIGMAS = 4.5  # Tune this based on your dataset and desired sensitivity
    anomalous_nodes_indices = torch.tensor([], dtype=torch.long, device=DEVICE)  # Initialize as empty tensor

    try:
        # Perform anomaly detection using the loaded model
        anomaly_results = model.detect_anomalies(data_pyg, threshold=ANOMALY_THRESHOLD_SIGMAS)

        if anomaly_results:
            logging.info(f"Anomaly detection completed for graph: {most_recent_file}")

            # Log Node Anomaly Results
            anomalous_nodes_indices = anomaly_results['node_anomalies_recon']
            node_recon_errors = anomaly_results['node_recon_errors']
            node_scores_mlp = anomaly_results['node_scores_mlp']

            logging.info(f"Total Nodes: {data_pyg.x.shape[0]}")
            logging.info(
                f"Anomalous Nodes (Reconstruction Error > {ANOMALY_THRESHOLD_SIGMAS} std dev): {len(anomalous_nodes_indices)}")

            if len(anomalous_nodes_indices) > 0:
                logging.info("--- Details for Anomalous Nodes (by Reconstruction Error) ---")
                for i in anomalous_nodes_indices:
                    node_index = i.item()  # Convert tensor to Python int
                    node_ip = "N/A"
                    if 0 <= node_index < len(node_ips):
                        node_ip = node_ips[node_index]  # Get the IP
                    logging.info(
                        f"Node Index {node_index} (IP: {node_ip}): Recon Error = {node_recon_errors[i]:.6f}, MLP Score = {node_scores_mlp[i]:.6f}")

            # Log Edge Anomaly Results
            anomalous_edges_indices = anomaly_results['edge_anomalies_recon']
            edge_recon_errors = anomaly_results['edge_recon_errors']
            edge_scores_mlp = anomaly_results['edge_scores_mlp']

            logging.info(f"Total Edges: {data_pyg.edge_index.shape[1]}")
            logging.info(
                f"Anomalous Edges (Reconstruction Error > {ANOMALY_THRESHOLD_SIGMAS} std dev): {len(anomalous_edges_indices)}")

            if len(anomalous_edges_indices) > 0:
                logging.info("--- Details for Anomalous Edges (by Reconstruction Error) ---")
                for i in anomalous_edges_indices:
                    edge_idx = i.item()  # Convert tensor to Python int
                    src_node_idx = data_pyg.edge_index[0, edge_idx].item()
                    dst_node_idx = data_pyg.edge_index[1, edge_idx].item()

                    src_ip = "N/A"
                    dst_ip = "N/A"
                    if 0 <= src_node_idx < len(node_ips):
                        src_ip = node_ips[src_node_idx]
                    if 0 <= dst_node_idx < len(node_ips):
                        dst_ip = node_ips[dst_node_idx]

                    logging.info(
                        f"Edge ({src_node_idx} -> {dst_node_idx}) (IPs: {src_ip} -> {dst_ip}) Index {edge_idx}: Recon Error = {edge_recon_errors[i]:.6f}, MLP Score = {edge_scores_mlp[i]:.6f}")

            # Log Global Anomaly Score
            global_anomaly_score_mlp = anomaly_results['global_anomaly_mlp']
            logging.info(f"Global Anomaly Score (from MLP): {global_anomaly_score_mlp:.6f}")

            if global_anomaly_score_mlp > 0.5:  # Example threshold for global score
                logging.warning("Global anomaly score is high, indicating potential overall anomalous graph behavior.")

        else:
            logging.info("No anomaly results returned.")

    except Exception as e:
        logging.error(f"Error during anomaly detection: {e}")
        logging.error(traceback.format_exc())

    # 5. Perform Node Reconstruction Analysis and Embedding Visualization
    logging.info("Starting node reconstruction analysis and embedding visualization...")
    feature_errors = analyze_node_reconstruction(model, validation_dataloader, DEVICE, most_recent_timestamp,
                                                 node_ips, anomalous_nodes_indices,
                                                 visualize_2d=False,
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
    # Example usage: Pass the directory containing your graph files
    # This line should be adjusted when you run the script
    main(validation_data_dir='/home/lu/Desktop/output_ssl_bruteforce')
