import logging
import os
import time
import traceback
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F

from visualization import visualize_node_features, visualize_embeddings_3d

# Import necessary functions and classes from your project files
# Ensure these paths are correct relative to where this script is run
try:
    from graph_utils import dot_to_nx, nx_to_pyg, get_sorted_node_features, get_sorted_edge_features
    from neural_net import NodeGNNAnomalyDetector
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
    # node_scaling and edge_scaling should be handled consistently.
    # If using running stats for normalization, ensure they are passed or handled internally.
    pytorch_data = nx_to_pyg(graph, node_scaling='standard', edge_scaling='standard')
    return graph, pytorch_data



def monitor_new_files(directory, model_save_path, stats_save_path, anomaly_log_path,
                      update_interval_seconds=30, export_period_updates=50, visualization_path=None, train_mode=False):
    """
    Monitors a directory for new .dot files and processes all .dot files found, including those present at startup.
    In 'detect mode' (train_mode=False), it loads the model and performs detection without online updates.

    Args:
        directory (str): Path to the directory to monitor for .dot files.
        model_save_path (str): Path to save/load the model checkpoint.
        stats_save_path (str): Path to save/load the running statistics.
        anomaly_log_path (str): Path to save the anomaly logs.
        update_interval_seconds (int): How often (in seconds) to check for new files.
        export_period_updates (int): Frequency (in number of updates) to export embeddings.
        visualization_path (str, optional): Path to save visualizations. Defaults to None.
        train_mode (bool): If True, performs initial training and online updates.
                            If False (detect mode), loads model and performs detection only.
    """
    logging.info(
        f"Starting to monitor directory: {directory} for new and existing files every {update_interval_seconds} seconds.")
    logging.info(f"Train Mode: {train_mode}, Visualization Path: {visualization_path}")

    processed_files = set()
    gnn_model = None
    processed_count = 0
    export_counter = 0  # Counter for visualization exports

    # Training parameters
    initial_training_epochs = 100 if train_mode else 50
    online_update_steps = 50 if train_mode else 25

    while True:
        if os.path.exists(directory):
            current_files = set(
                [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.dot')])
            files_to_process = sorted(list(current_files - processed_files)) if processed_count > 0 else sorted(
                list(current_files))

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

                        # Initialize model if needed and load checkpoint/stats
                        if gnn_model is None:
                            node_feature_dim = main_data.x.size(1)
                            # Ensure edge_feature_dim is handled if edge_attr is None
                            edge_feature_dim = main_data.edge_attr.size(1) if hasattr(main_data,
                                                                                       'edge_attr') and main_data.edge_attr is not None else 0

                            gnn_model = NodeGNNAnomalyDetector(
                                node_feature_dim=node_feature_dim,
                                edge_feature_dim=edge_feature_dim,
                                hidden_dim=128,
                                embedding_dim=64,
                                num_gat_layers=3,
                                gat_heads=4,
                                recon_loss_type='mse',
                                batch_size=16
                            )
                            logging.info(
                                f"Initialized Node GNN Anomaly Detector (node_dim={node_feature_dim}, edge_dim={edge_feature_dim})")

                            try:
                                start_epoch = load_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler,
                                                              model_save_path)
                                logging.info(f"Model loaded successfully. Resuming from step: {start_epoch}")
                                load_running_stats(gnn_model, stats_save_path)
                                logging.info("Running statistics loaded.")
                            except FileNotFoundError:
                                logging.warning(f"No checkpoint found at {model_save_path}.")
                                if train_mode:
                                    logging.info("Performing initial training for new model.")
                                    initial_optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
                                    # For initial training, we add data to buffer first, then train from buffer
                                    gnn_model._add_to_replay_buffer(main_data.cpu())  # Add first data point
                                    for epoch in range(initial_training_epochs):
                                        batch_for_init_train = gnn_model._get_training_batch()
                                        if batch_for_init_train is None:
                                            logging.warning(
                                                "Not enough data for initial training batch. Adding more data...")
                                            # If initial batch is too small, add the current data and try again
                                            gnn_model._add_to_replay_buffer(main_data.cpu())
                                            if len(gnn_model.replay_buffer) < gnn_model.batch_size:
                                                logging.warning(
                                                    "Skipping initial training due to insufficient data in buffer.")
                                                break  # Cannot train if no batch can be formed
                                            batch_for_init_train = gnn_model._get_training_batch()  # Try again
                                            if batch_for_init_train is None:
                                                logging.warning(
                                                    "Still not enough data after adding. Skipping initial training.")
                                                break

                                        initial_optimizer.zero_grad()
                                        loss = gnn_model._calculate_online_loss(
                                            batch=batch_for_init_train.to(gnn_model.device),
                                            recon_weight=0.8,
                                            anomaly_weight=0.2,
                                            use_focal_loss=True
                                        )
                                        if loss is not None:
                                            loss.backward()
                                            initial_optimizer.step()
                                            logging.info(
                                                f"Initial Training Epoch {epoch + 1}/{initial_training_epochs}, Loss: {loss.item():.4f}")
                                            # Update recon error stats during initial training
                                            with torch.no_grad():
                                                normalized_batch_x_init = gnn_model._normalize_features(
                                                    batch_for_init_train.clone()).x
                                                _, node_recon_eval_init, _ = gnn_model(
                                                    batch_for_init_train.to(gnn_model.device))
                                                node_recon_error_init = F.mse_loss(node_recon_eval_init,
                                                                                   normalized_batch_x_init,
                                                                                   reduction='none').mean(dim=1)
                                                gnn_model.node_recon_error_stats.update(node_recon_error_init)

                                        else:
                                            logging.error("Loss calculation failed during initial training.")
                                            break
                                    logging.info("Initial training complete.")
                                    save_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler,
                                                    processed_count, model_save_path)
                                    # Ensure all necessary stats are saved
                                    save_running_stats(gnn_model.node_stats, gnn_model.edge_stats, stats_save_path)
                                else:
                                    logging.error(
                                        "No model checkpoint found and 'train_mode' is False. Cannot perform detection without a trained model. Skipping this file.")
                                    processed_files.add(filename)
                                    continue

                            except Exception as e:
                                logging.error(f"Error loading model or running stats during initial setup: {e}")
                                logging.error(traceback.format_exc())
                                logging.error("Cannot proceed without a working model. Skipping this file.")
                                processed_files.add(filename)
                                continue

                        # --- Online Update / Detection Logic ---
                        if train_mode:
                            logging.info("Performing online update...")
                            loss = gnn_model.update_online(main_data, n_steps=online_update_steps)
                            logging.info(f"Online update complete. Loss: {loss:.4f}")
                        else:
                            logging.info("Skipping online update: Running in detect mode (train_mode=False).")

                        # Detect anomalies
                        logging.info("Running anomaly detection...")
                        # The detect_anomalies method now only returns node-related anomaly info
                        anomalies = gnn_model.detect_anomalies(main_data)

                        # Save anomalies to file
                        save_anomalies_to_file(main_data, anomalies, processed_count, anomaly_log_path,
                                               nx_graph=main_graph)

                        # Save running statistics and checkpoint (only if in training mode)
                        if train_mode:
                            save_running_stats(gnn_model.node_stats, gnn_model.edge_stats, stats_save_path)
                            save_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler, processed_count,
                                            model_save_path)

                        # Report results (only for node anomalies)
                        logging.info(f"\nNode anomaly detection results for file: {filename}")

                        if len(anomalies['node_anomalies_recon']) > 0:
                            logging.warning(
                                f"Detected {len(anomalies['node_anomalies_recon'])} anomalous nodes (reconstruction-based):")
                            for idx in anomalies['node_anomalies_recon'][:5]:
                                node_id = list(main_graph.nodes)[idx.item()] if main_graph else str(idx.item())
                                logging.warning(
                                    f"  Node ID: {node_id}, Recon Error: {anomalies['node_recon_errors'][idx]:.4f}, MLP Score: {anomalies['node_scores_mlp'][idx]:.4f}")
                        else:
                            logging.info("No significant node anomalies detected (reconstruction-based)")

                        if len(anomalies['node_anomalies_mlp']) > 0:
                            logging.warning(
                                f"Detected {len(anomalies['node_anomalies_mlp'])} anomalous nodes (MLP-based):")
                            for idx in anomalies['node_anomalies_mlp'][:5]:
                                node_id = list(main_graph.nodes)[idx.item()] if main_graph else str(idx.item())
                                logging.warning(
                                    f"  Node ID: {node_id}, Recon Error: {anomalies['node_recon_errors'][idx]:.4f}, MLP Score: {anomalies['node_scores_mlp'][idx]:.4f}")
                        else:
                            logging.info("No significant node anomalies detected (MLP-based)")

                        # --- Visualization (optional) ---
                        if visualization_path is not None:
                            if gnn_model and export_period_updates > 0:
                                if export_counter % export_period_updates == 0:
                                    try:
                                        # Get current node embeddings from the model
                                        embeddings = gnn_model.autoencoder.encode(
                                            main_data.x.to(gnn_model.device),
                                            main_data.edge_index.to(gnn_model.device),
                                            main_data.edge_attr.to(
                                                gnn_model.device) if main_data.edge_attr is not None else None
                                        ).detach().cpu().numpy()

                                        # Get the list of all node IPs from the current graph
                                        node_ips = list(main_graph.nodes) if main_graph else []

                                        # Use the reconstruction-based anomalies for visualization highlighting
                                        anomalous_indices_for_viz = anomalies.get('node_anomalies_recon', torch.tensor([]))

                                        # Use the processed_count as a timestamp for the filename
                                        # Or if you have a real timestamp for the graph update, use that instead.
                                        current_timestamp_for_viz = processed_count # Using processed_count as a unique identifier

                                        # Call the NEW visualize_embeddings_3d function
                                        visualize_embeddings_3d(
                                            embeddings=embeddings,
                                            node_ips=node_ips,
                                            anomalous_indices_tensor=anomalous_indices_for_viz,
                                            save_path=visualization_path,
                                            timestamp=current_timestamp_for_viz,
                                            filename_prefix="gcn_tsne_anomalies" # Custom prefix
                                        )
                                        logging.info("3D embeddings visualization triggered.")

                                    except Exception as e:
                                        logging.error(f"Error during periodic 3D embedding visualization: {e}")
                                        logging.error(traceback.format_exc())

                                if main_data.x is not None and main_graph is not None:
                                    logging.info("Performing visualization of node features.")
                                    try:
                                        visualize_node_features(main_data, save_path=visualization_path,
                                                                feature_names=get_sorted_node_features(main_graph))
                                    except Exception as e:
                                        logging.error(f"Error during 2D feature visualization: {e}")
                                        logging.error(traceback.format_exc())
                                elif main_data.x is None:
                                    logging.warning(
                                        "Skipping node feature visualization: No node features available.")
                                export_counter += 1

                        processed_files.add(filename)
                    except Exception as e:
                        logging.error(f"An error occurred while processing file {filename}: {e}")
                        logging.error(traceback.format_exc())
            else:
                logging.info(
                    f"No new .dot files found in {directory}. Waiting for {update_interval_seconds} seconds...")
        else:
            logging.info(f"Directory {directory} does not exist. Waiting for {update_interval_seconds} seconds...")

        time.sleep(update_interval_seconds)
