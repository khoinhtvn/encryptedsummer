import logging

from visualization import visualize_node_features, visualize_edge_features, visualize_all_edge_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary functions and classes from your project files
# Ensure these paths are correct relative to where this script is run
try:
    from graph_utils import dot_to_nx, nx_to_pyg, get_sorted_node_features, get_sorted_edge_features
    from neural_net import HybridGNNAnomalyDetector
    from utils import load_checkpoint, save_checkpoint, load_running_stats, save_running_stats, \
        extract_timestamp_from_epoch, save_anomalies_to_file
except ImportError as e:
    logging.error(
        f"Error importing modules. Please ensure graph_utils.py, neural_net.py, and utils.py are in your PYTHONPATH or the same directory.")
    logging.error(e)
    exit()

# Initialize global encoders (if not already initialized)
if 'edge_categorical_encoders' not in globals():
    edge_categorical_encoders = {}


def visualize_embeddings_3d(embeddings, labels, save_path, filename="embeddings_3d.png"):
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

        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath)
        logging.info(f"3D embeddings visualization saved to {filepath}")
        plt.close()
    except Exception as e:
        logging.error(f"Error during 3D t-SNE or plotting: {e}")
        logging.error(traceback.format_exc())


def process_single_update(filepath):
    """Processes a single .dot file, converts it to NetworkX and PyTorch Geometric Data."""
    logging.info(f"Processing update from: {filepath}")
    graph = dot_to_nx(filepath)
    pytorch_data = nx_to_pyg(graph, node_scaling='standard', edge_scaling='standard')
    return graph, pytorch_data


def process_existing_directory(directory, model_save_path, stats_save_path, anomaly_log_path,
                               export_period_updates=50, visualization_path=None, train_mode=False):
    """
    Processes all .dot files in the specified directory, updating the graph and training the model
    sequentially for each file, and visualizing embeddings periodically in 3D.

    Args:
        directory (str): Path to the directory containing .dot files.
        model_save_path (str): Path to save the model checkpoint.
        stats_save_path (str): Path to save the running statistics.
        anomaly_log_path (str): Path to save the anomaly logs.
        export_period_updates (int): Frequency (in number of updates) to export embeddings.
        visualization_path (str, optional): Path to save visualizations. Defaults to None.
        train_mode (bool): If True, performs more extensive initial training. Defaults to False.
    """
    all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.dot')]
    files_with_timestamps = []

    for filename in all_files:
        timestamp_dt = extract_timestamp_from_epoch(filename)
        if timestamp_dt:
            files_with_timestamps.append((timestamp_dt, os.path.join(directory, filename), filename))
        else:
            logging.warning(f"Could not extract timestamp from filename: {filename}. Processing with current order.")
            files_with_timestamps.append((datetime.now(), os.path.join(directory, filename), filename))

    files_with_timestamps.sort(key=lambda item: item[0])

    main_graph = None
    main_data = None
    gnn_model = None
    processed_count = 0
    export_counter = 0

    # Training parameters
    initial_training_epochs = 100 if train_mode else 50  # More epochs if train_mode is True
    online_update_steps = 50 if train_mode else 25  # More steps if train_mode is True

    logging.info(f"Starting processing of existing directory: {directory}, Train Mode: {train_mode}")

    for timestamp_dt, filepath, filename in files_with_timestamps:
        logging.info(f"\n--- Processing file: {filename} (Timestamp: {timestamp_dt}) ---")

        # Process the current update file
        main_graph, main_data = process_single_update(filepath)
        processed_count += 1

        if main_data is not None:
            # Initialize model if needed
            if gnn_model is None:
                node_feature_dim = main_data.x.size(1)
                edge_feature_dim = main_data.edge_attr.size(1) if main_data.edge_attr is not None else 0
                gnn_model = HybridGNNAnomalyDetector(
                    node_feature_dim=node_feature_dim,
                    edge_feature_dim=edge_feature_dim,
                    hidden_dim=128,
                    embedding_dim=64,
                    num_gat_layers=3,
                    gat_heads=4,
                    recon_loss_type='mse',  # Or 'l1'
                    edge_recon_loss_type='bce',  # Or 'mse', 'l1'
                    batch_size=16
                )
                logging.info(f"Initialized Hybrid GNN (node_dim={node_feature_dim}, edge_dim={edge_feature_dim})")

                # Initial training
                logging.info(f"Performing initial training for {initial_training_epochs} epochs...")
                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
                for epoch in range(initial_training_epochs):
                    optimizer.zero_grad()
                    node_scores, edge_scores, _, node_recon, edge_recon, _, _ = gnn_model(main_data)
                    recon_loss_node = F.mse_loss(node_recon, main_data.x)
                    edge_recon_target = torch.ones_like(edge_recon) * 0.5
                    recon_loss_edge = F.binary_cross_entropy_with_logits(edge_recon, edge_recon_target)
                    anomaly_loss = (node_scores.abs().mean() + edge_scores.abs().mean()) / 2
                    loss = 0.8 * (recon_loss_node + recon_loss_edge) + 0.2 * anomaly_loss
                    loss.backward()
                    optimizer.step()
                    logging.info(f"Epoch {epoch + 1}/{initial_training_epochs}, Loss: {loss.item():.4f}")
                logging.info("Initial training complete.")

                # Load latest checkpoint if available AFTER initial training
                start_epoch = load_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler, model_save_path)
                logging.info(f"Starting or resuming online updates from step: {start_epoch}")
                load_running_stats(gnn_model, stats_save_path)  # Load stats again for the actual model
            else:
                # Online update
                logging.info("Performing online update...")
                loss = gnn_model.update_online(main_data, n_steps=online_update_steps)
                logging.info(f"Online update complete. Loss: {loss:.4f}")

            # Detect anomalies
            logging.info("Running anomaly detection...")
            anomalies = gnn_model.detect_anomalies(main_data)

            # Save anomalies to file
            save_anomalies_to_file(main_data, anomalies, processed_count, anomaly_log_path, nx_graph=main_graph)

            # Save running statistics
            if gnn_model:
                save_running_stats(gnn_model.node_stats, gnn_model.edge_stats, stats_save_path)

            # Save checkpoint
            if gnn_model:
                save_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler, processed_count,
                                model_save_path)

            # Report results
            logging.info(f"\nGlobal anomaly score (after {filename}): {anomalies['global_anomaly_mlp']:.4f}")

            # Node anomalies
            if len(anomalies['node_anomalies_recon']) > 0:
                logging.warning(
                    f"Detected {len(anomalies['node_anomalies_recon'])} anomalous nodes (reconstruction-based):")
                for idx in anomalies['node_anomalies_recon'][:5]:  # Show top 5
                    node_id = list(main_graph.nodes)[idx.item()] if main_graph else str(idx.item())
                    logging.warning(
                        f"  Node ID: {node_id}, Recon Error: {anomalies['node_recon_errors'][idx]:.4f}, MLP Score: {anomalies['node_scores_mlp'][idx]:.4f}")
            else:
                logging.info("No significant node anomalies detected (reconstruction-based)")

            # Edge anomalies
            if len(anomalies['edge_anomalies_recon']) > 0:
                logging.warning(
                    f"Detected {len(anomalies['edge_anomalies_recon'])} anomalous edges (reconstruction-based):")
                for idx in anomalies['edge_anomalies_recon'][:5]:  # Show top 5
                    if main_graph:
                        src_idx = main_data.edge_index[0][idx].item()
                        dst_idx = main_data.edge_index[1][idx].item()
                        src_node = list(main_graph.nodes)[src_idx]
                        dst_node = list(main_graph.nodes)[dst_idx]
                        logging.warning(
                            f"  Source: {src_node}, Target: {dst_node}, Recon Error: {anomalies['edge_recon_errors'][idx].item():.4f}, MLP Score: {anomalies['edge_scores_mlp'][idx].item():.4f}")
                    else:
                        logging.warning(
                            f"  Edge Index: {idx.item()}, Recon Error: {anomalies['edge_recon_errors'][idx].item():.4f}, MLP Score: {anomalies['edge_scores_mlp'][idx].item():.4f}")
            else:
                logging.info("No significant edge anomalies detected (reconstruction-based)")

            # Visualization (optional)
            if visualization_path is not None:
                # Periodic Embedding Visualization (3D)
                if gnn_model and export_period_updates > 0:
                    if export_counter % export_period_updates == 0:
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename_3d = os.path.join(visualization_path,
                                                       f"embeddings_3d_{processed_count}_updates_{timestamp}.png")
                            # Ensure main_data is on the correct device for encoding
                            embeddings = gnn_model.autoencoder.encode(
                                main_data.x.to(gnn_model.device),
                                main_data.edge_index.to(gnn_model.device),
                                main_data.edge_attr.to(gnn_model.device) if main_data.edge_attr is not None else None
                            ).cpu().numpy()
                            labels = [0] * main_data.num_nodes  # Assuming all nodes in a single update are of the same type for visualization
                            visualize_embeddings_3d(embeddings, labels, visualization_path, filename=filename_3d)
                        except Exception as e:
                            logging.error(f"Error during periodic 3D embedding visualization: {e}")
                            logging.error(traceback.format_exc())  # Print full traceback for debugging

                        # Existing 2D visualizations (ensure these functions are defined in visualization.py)
                        if main_data.x is not None and main_data.edge_attr is not None and main_graph is not None:
                            logging.info("Performing visualization of node and edge features.")
                            try:
                                # Assuming visualize_node_features, visualize_edge_features, visualize_all_edge_features exist in visualization.py
                                # and accept save_path and feature_names arguments correctly.
                                # You might need to pass main_data.x.cpu().numpy() for visualization functions if they expect numpy.
                                visualize_node_features(main_data, save_path=visualization_path,
                                                        feature_names=get_sorted_node_features(main_graph))
                                visualize_edge_features(main_data, save_path=visualization_path,
                                                        edge_feature_names=get_sorted_edge_features(main_graph))
                                visualize_all_edge_features(main_data, save_path=visualization_path,
                                                            edge_feature_names=get_sorted_edge_features(main_graph))
                            except Exception as e:
                                logging.error(f"Error during 2D feature visualization: {e}")
                                logging.error(traceback.format_exc())
                        elif main_data.x is None:
                            logging.warning("Skipping node feature visualization: No node features available.")
                        elif main_data.edge_attr is None:
                            logging.warning("Skipping edge feature visualization: No edge features available.")
                    export_counter += 1

    logging.info("Finished processing all files in the directory.")


import time
import os
import logging
import traceback
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary functions and classes from your project files
# Ensure these paths are correct relative to where this script is run
try:
    from graph_utils import dot_to_nx, nx_to_pyg, get_sorted_node_features, get_sorted_edge_features
    from neural_net import HybridGNNAnomalyDetector
    from utils import load_checkpoint, save_checkpoint, load_running_stats, save_running_stats, \
        extract_timestamp_from_epoch, save_anomalies_to_file
except ImportError as e:
    logging.error(
        f"Error importing modules. Please ensure graph_utils.py, neural_net.py, and utils.py are in your PYTHONPATH or the same directory.")
    logging.error(e)
    exit()

# Initialize global encoders (if not already initialized)
if 'edge_categorical_encoders' not in globals():
    edge_categorical_encoders = {}


def process_single_update(filepath):
    """Processes a single .dot file, converts it to NetworkX and PyTorch Geometric Data."""
    logging.info(f"Processing update from: {filepath}")
    graph = dot_to_nx(filepath)
    pytorch_data = nx_to_pyg(graph, node_scaling='standard', edge_scaling='standard')
    return graph, pytorch_data


def monitor_new_files(directory, model_save_path, stats_save_path, anomaly_log_path,
                      update_interval_seconds=30, export_period_updates=50, visualization_path=None, train_mode=False):
    """
    Monitors a directory for new .dot files and processes all .dot files found, including those present at startup.

    Args:
        directory (str): Path to the directory to monitor for .dot files.
        model_save_path (str): Path to save the model checkpoint.
        stats_save_path (str): Path to save the running statistics.
        anomaly_log_path (str): Path to save the anomaly logs.
        update_interval_seconds (int): How often (in seconds) to check for new files.
        export_period_updates (int): Frequency (in number of updates) to export embeddings.
        visualization_path (str, optional): Path to save visualizations. Defaults to None.
        train_mode (bool): If True, performs more extensive initial training. Defaults to False.
    """
    logging.info(
        f"Starting to monitor directory: {directory} for new and existing files every {update_interval_seconds} seconds.")
    logging.info(f"Train Mode: {train_mode}, Visualization Path: {visualization_path}")

    processed_files = set()
    gnn_model = None
    processed_count = 0
    export_counter = 0  # Counter for visualization exports

    # Training parameters (same as process_existing_directory)
    initial_training_epochs = 100 if train_mode else 50
    online_update_steps = 50 if train_mode else 25

    while True:
        if os.path.exists(directory):
            current_files = set(
                [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.dot')])
            files_to_process = sorted(
                list(current_files - processed_files))  # Process only new files in subsequent checks
            initial_files = sorted(
                list(current_files - processed_files)) if processed_count == 0 else []  # Process all at start

            files_to_process = initial_files + files_to_process  # Process initial files first

            if files_to_process:
                logging.info(f"Found {len(files_to_process)} files to process.")
                for filename in files_to_process:
                    filepath = os.path.join(directory, filename)
                    logging.info(f"\n--- Processing file: {filename} ---")

                    try:
                        main_graph, main_data = process_single_update(filepath)
                        processed_count += 1

                        if main_data is None:
                            logging.warning(f"Skipping processing of {filename}: No valid data generated.")
                            continue

                        # Initialize model if needed (or load from checkpoint if it exists)
                        if gnn_model is None:
                            node_feature_dim = main_data.x.size(1)
                            edge_feature_dim = main_data.edge_attr.size(1) if main_data.edge_attr is not None else 0
                            gnn_model = HybridGNNAnomalyDetector(
                                node_feature_dim=node_feature_dim,
                                edge_feature_dim=edge_feature_dim,
                                hidden_dim=128,
                                embedding_dim=64,
                                num_gat_layers=3,
                                gat_heads=4,
                                recon_loss_type='mse',
                                edge_recon_loss_type='bce',
                                batch_size=16
                            )
                            logging.info(
                                f"Initialized Hybrid GNN (node_dim={node_feature_dim}, edge_dim={edge_feature_dim})")

                            # Attempt to load checkpoint and stats
                            try:
                                start_epoch = load_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler,
                                                              model_save_path)
                                logging.info(f"Resuming online updates from step: {start_epoch}")
                                load_running_stats(gnn_model, stats_save_path)
                            except FileNotFoundError:
                                logging.warning(f"No checkpoint found at {model_save_path}. Starting initial training.")
                                # Perform initial training if no checkpoint is found
                                logging.info(f"Performing initial training for {initial_training_epochs} epochs...")
                                optimizer = torch.optim.Adam(gnn_model.parameters(),
                                                             lr=0.001)  # Use a temporary optimizer for initial training
                                for epoch in range(initial_training_epochs):
                                    optimizer.zero_grad()
                                    node_scores, edge_scores, _, node_recon, edge_recon, _, _ = gnn_model(main_data)
                                    recon_loss_node = F.mse_loss(node_recon, main_data.x)
                                    edge_recon_target = torch.ones_like(edge_recon) * 0.5
                                    recon_loss_edge = F.binary_cross_entropy_with_logits(edge_recon, edge_recon_target)
                                    anomaly_loss = (node_scores.abs().mean() + edge_scores.abs().mean()) / 2
                                    loss = 0.8 * (recon_loss_node + recon_loss_edge) + 0.2 * anomaly_loss
                                    loss.backward()
                                    optimizer.step()
                                    logging.info(
                                        f"Epoch {epoch + 1}/{initial_training_epochs}, Loss: {loss.item():.4f}")
                                logging.info("Initial training complete.")
                                # After initial training, ensure the model's internal optimizer/scheduler are used for subsequent updates
                                # This might involve re-initializing gnn_model.optimizer and gnn_model.scheduler if they were temporary for initial training
                                # For simplicity here, we assume gnn_model.optimizer and gnn_model.scheduler are already set up for online updates.
                            except Exception as e:
                                logging.error(f"Error loading model or running stats during initial setup: {e}")
                                logging.error(traceback.format_exc())
                                continue  # Skip this file if model/stats loading fails

                        # Online update
                        logging.info("Performing online update...")
                        loss = gnn_model.update_online(main_data, n_steps=online_update_steps)
                        logging.info(f"Online update complete. Loss: {loss:.4f}")

                        # Detect anomalies
                        logging.info("Running anomaly detection...")
                        anomalies = gnn_model.detect_anomalies(main_data)

                        # Save anomalies to file
                        save_anomalies_to_file(main_data, anomalies, processed_count, anomaly_log_path,
                                               nx_graph=main_graph)

                        # Save running statistics
                        if gnn_model:
                            save_running_stats(gnn_model.node_stats, gnn_model.edge_stats, stats_save_path)

                        # Save checkpoint
                        if gnn_model:
                            save_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler, processed_count,
                                            model_save_path)

                        # Report results
                        logging.info(
                            f"\nGlobal anomaly score (after {filename}): {anomalies['global_anomaly_mlp']:.4f}")

                        # Node anomalies
                        if len(anomalies['node_anomalies_recon']) > 0:
                            logging.warning(
                                f"Detected {len(anomalies['node_anomalies_recon'])} anomalous nodes (reconstruction-based):")
                            for idx in anomalies['node_anomalies_recon'][:5]:  # Show top 5
                                node_id = list(main_graph.nodes)[idx.item()] if main_graph else str(idx.item())
                                logging.warning(
                                    f"  Node ID: {node_id}, Recon Error: {anomalies['node_recon_errors'][idx]:.4f}, MLP Score: {anomalies['node_scores_mlp'][idx]:.4f}")
                        else:
                            logging.info("No significant node anomalies detected (reconstruction-based)")

                        # Edge anomalies
                        if len(anomalies['edge_anomalies_recon']) > 0:
                            logging.warning(
                                f"Detected {len(anomalies['edge_anomalies_recon'])} anomalous edges (reconstruction-based):")
                            for idx in anomalies['edge_anomalies_recon'][:5]:  # Show top 5
                                if main_graph:
                                    src_idx = main_data.edge_index[0][idx].item()
                                    dst_idx = main_data.edge_index[1][idx].item()
                                    src_node = list(main_graph.nodes)[src_idx]
                                    dst_node = list(main_graph.nodes)[dst_idx]
                                    logging.warning(
                                        f"  Source: {src_node}, Target: {dst_node}, Recon Error: {anomalies['edge_recon_errors'][idx].item():.4f}, MLP Score: {anomalies['edge_scores_mlp'][idx].item():.4f}")
                                else:
                                    logging.warning(
                                        f"  Edge Index: {idx.item()}, Recon Error: {anomalies['edge_recon_errors'][idx].item():.4f}, MLP Score: {anomalies['edge_scores_mlp'][idx].item():.4f}")
                        else:
                            logging.info("No significant edge anomalies detected (reconstruction-based)")

                        # Visualization (optional)
                        if visualization_path is not None:
                            # Periodic Embedding Visualization (3D)
                            if gnn_model and export_period_updates > 0:
                                if export_counter % export_period_updates == 0:
                                    try:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        filename_3d = f"embeddings_3d_{processed_count}_updates_{timestamp}.png"
                                        # Ensure main_data is on the correct device for encoding
                                        embeddings = gnn_model.autoencoder.encode(
                                            main_data.x.to(gnn_model.device),
                                            main_data.edge_index.to(gnn_model.device),
                                            main_data.edge_attr.to(
                                                gnn_model.device) if main_data.edge_attr is not None else None
                                        ).detach().cpu().numpy()
                                        labels = [0] * main_data.num_nodes  # Assuming all nodes in a single update are of the same type for visualization
                                        visualize_embeddings_3d(embeddings, labels, visualization_path,
                                                                filename=filename_3d)
                                    except Exception as e:
                                        logging.error(f"Error during periodic 3D embedding visualization: {e}")
                                        logging.error(traceback.format_exc())  # Print full traceback for debugging

                                    # Existing 2D visualizations (ensure these functions are defined in visualization.py)
                                    if main_data.x is not None and main_data.edge_attr is not None and main_graph is not None:
                                        logging.info("Performing visualization of node and edge features.")
                                        try:
                                            # Assuming visualize_node_features, visualize_edge_features, visualize_all_edge_features exist in visualization.py
                                            # and accept save_path and feature_names arguments correctly.
                                            # You might need to pass main_data.x.cpu().numpy() for visualization functions if they expect numpy.
                                            visualize_node_features(main_data, save_path=visualization_path,
                                                                    feature_names=get_sorted_node_features(main_graph))
                                            visualize_edge_features(main_data, save_path=visualization_path,
                                                                    edge_feature_names=get_sorted_edge_features(
                                                                        main_graph))
                                            visualize_all_edge_features(main_data, save_path=visualization_path,
                                                                        edge_feature_names=get_sorted_edge_features(
                                                                            main_graph))
                                        except Exception as e:
                                            logging.error(f"Error during 2D feature visualization: {e}")
                                            logging.error(traceback.format_exc())
                                    elif main_data.x is None:
                                        logging.warning(
                                            "Skipping node feature visualization: No node features available.")
                                    elif main_data.edge_attr is None:
                                        logging.warning(
                                            "Skipping edge feature visualization: No edge features available.")
                                export_counter += 1

                        processed_files.add(filename)  # Mark file as processed after successful handling
                    except Exception as e:
                        logging.error(f"An error occurred while processing file {filename}: {e}")
                        logging.error(traceback.format_exc())
            else:
                logging.info(
                    f"No new .dot files found in {directory}. Waiting for {update_interval_seconds} seconds...")
        else:
            logging.info(f"Directory {directory} does not exist. Waiting for {update_interval_seconds} seconds...")

        time.sleep(update_interval_seconds)
