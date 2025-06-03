import json
import pickle
import time
import os
import logging
from datetime import datetime

import torch.nn.functional as F

from graph_utils import *
from neural_net import HybridGNNAnomalyDetector
from visualization import *
from utils import *


def process_single_update(filepath):
    logging.info(f"Processing update from: {filepath}")
    graph = dot_to_nx(filepath)

    # Initialize the global dictionary if it hasn't been already (though the import should handle this)
    if not 'edge_categorical_encoders' in globals():
        global edge_categorical_encoders
        edge_categorical_encoders = {}

    pytorch_data = nx_to_pyg(graph, node_scaling='standard', edge_scaling='standard')
    return graph, pytorch_data


def process_existing_directory(directory, model_save_path, stats_save_path, anomaly_log_path,
                               export_period_updates=50, visualization_path=None, train_mode=False):
    """
    Processes all .dot files in the specified directory, updating the graph and training the model
    sequentially for each file.

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
                        edge_recon_loss_type='bce', # Or 'mse', 'l1'
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
                # Periodic Embedding Export
                if gnn_model and export_period_updates > 0:
                    if export_counter % export_period_updates == 0:
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(gnn_model.export_dir,
                                                    f"embeddings_{processed_count}_updates_{timestamp}.png")
                            gnn_model.export_embeddings(main_data, filename=filename)
                        except Exception as e:
                            logging.error(f"Error during periodic embedding export: {e}")
                        if main_data.x is not None and main_data.edge_attr is not None and main_graph is not None:
                            logging.info("Performing visualization of node and edge features.")
                            try:
                                visualize_node_features(main_data, save_path=visualization_path,
                                                        feature_names=get_sorted_node_features(main_graph))
                                visualize_edge_features(main_data, save_path=visualization_path,
                                                        edge_feature_names=get_sorted_edge_features(main_graph))
                                visualize_all_edge_features(main_data, save_path=visualization_path,
                                                            edge_feature_names=get_sorted_edge_features(main_graph))
                            except Exception as e:
                                logging.error(f"Error during visualization: {e}")
                        elif main_data.x is None:
                            logging.warning("Skipping node feature visualization: No node features available.")
                        elif main_data.edge_attr is None:
                            logging.warning("Skipping edge feature visualization: No edge features available.")
                    export_counter += 1

    logging.info("Finished processing all files in the directory.")