import json
import pickle
import time

import torch.nn.functional as F

from graph_utils import *
from neural_net import HybridGNNAnomalyDetector
from visualization import *


def save_anomalies_to_file(main_data, anomalies, processed_files_count, anomaly_log_path, nx_graph=None,
                           timestamp=None):
    """Saves detected anomalies to a JSON file with more details, including IP addresses."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(anomaly_log_path, f"anomalies_{timestamp}_update_{processed_files_count}.json")
    anomaly_data = {
        "timestamp": timestamp,
        "update_count": processed_files_count,
        "global_anomaly_score": anomalies.get('global_anomaly_mlp', None),  # Use the MLP global score
        "node_anomalies": [],
        "edge_anomalies": []
    }

    if nx_graph is not None and main_data is not None:
        node_list = list(nx_graph.nodes())  # Get the list of node IPs from the NetworkX graph

        for idx in anomalies.get('node_anomalies_recon', []):  # Use reconstruction-based anomalies
            node_index = idx.item()
            node_info = {"node_id": node_index, "recon_error": anomalies.get('node_recon_errors', [])[idx].item(),
                         "mlp_score": anomalies.get('node_scores_mlp', [])[idx].item()}
            if 0 <= node_index < len(node_list):
                node_info['ip'] = str(node_list[node_index])  # Use the index to get the IP
            anomaly_data["node_anomalies"].append(node_info)

        if main_data.edge_index is not None:
            for idx in anomalies.get('edge_anomalies_recon', []):  # Use reconstruction-based anomalies
                src_index = main_data.edge_index[0][idx].item()
                dst_index = main_data.edge_index[1][idx].item()
                edge_info = {
                    "source_node_id": src_index,
                    "target_node_id": dst_index,
                    "recon_error": anomalies.get('edge_recon_errors', [])[idx].item(),
                    "mlp_score": anomalies.get('edge_scores_mlp', [])[idx].item()
                }
                if 0 <= src_index < len(node_list):
                    edge_info['source_ip'] = str(node_list[src_index])  # Source IP
                if 0 <= dst_index < len(node_list):
                    edge_info['target_ip'] = str(node_list[dst_index])  # Target IP
                anomaly_data["edge_anomalies"].append(edge_info)
    else:
        logging.warning("NetworkX graph or main_data not provided, cannot include IP addresses in anomaly details.")
        for idx in anomalies.get('node_anomalies_recon', []):
            anomaly_data["node_anomalies"].append(
                {"node_id": idx.item(), "recon_error": anomalies.get('node_recon_errors', [])[idx].item(),
                 "mlp_score": anomalies.get('node_scores_mlp', [])[idx].item()})
        if main_data is not None and main_data.edge_index is not None:
            for idx in anomalies.get('edge_anomalies_recon', []):
                anomaly_data["edge_anomalies"].append({
                    "source_node_id": main_data.edge_index[0][idx].item(),
                    "target_node_id": main_data.edge_index[1][idx].item(),
                    "recon_error": anomalies.get('edge_recon_errors', [])[idx].item(),
                    "mlp_score": anomalies.get('edge_scores_mlp', [])[idx].item()
                })

    try:
        with open(filename, 'w') as f:
            json.dump(anomaly_data, f, indent=4)
        logging.info(f"Anomalies saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving anomalies to file: {e}")


def save_checkpoint(model, optimizer, scheduler, processed_files_count, model_save_path, filename_prefix="checkpoint"):
    """Saves the model checkpoint with the processed files count."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(model_save_path, f"{filename_prefix}_{timestamp}_update_{processed_files_count}.pth")
    obj = {
        'epoch': processed_files_count,  # Save processed_files_count as 'epoch'
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'node_stats': model.node_stats,
        'edge_stats': model.edge_stats,
    }
    torch.save(obj, filename)

    try:
        os.remove(os.path.join(model_save_path, "latest_checkpoint.pth"))
    except OSError:
        pass

    torch.save(obj, os.path.join(model_save_path, "latest_checkpoint.pth"))
    logging.info(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, scheduler, model_save_path, filename="latest_checkpoint.pth"):
    filepath = os.path.join(model_save_path, filename)
    if os.path.exists(filepath):
        try:
            checkpoint = torch.load(filepath, weights_only=False)
            if model is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                model.node_stats = checkpoint.get('node_stats', model.node_stats)
                model.edge_stats = checkpoint.get('edge_stats', model.edge_stats)
                epoch = checkpoint['epoch']
                logging.info(f"Checkpoint loaded from {filepath} at epoch {epoch}")
                return epoch
            else:
                logging.info(f"Checkpoint metadata loaded from {filepath}")
            return checkpoint.get('epoch', 0)
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            return 0
    else:
        logging.info("No checkpoint found. Starting from scratch.")
        return 0


def save_running_stats(node_stats, edge_stats, stats_save_path, filename="running_stats.pkl"):
    filepath = os.path.join(stats_save_path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'node_stats': node_stats, 'edge_stats': edge_stats}, f)
    logging.info(f"Running statistics saved to {filepath}")


def load_running_stats(model, stats_save_path, filename="running_stats.pkl"):
    filepath = os.path.join(stats_save_path, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
            model.node_stats = stats.get('node_stats', model.node_stats)
            model.edge_stats = stats.get('edge_stats', model.edge_stats)
        logging.info(f"Running statistics loaded from {filepath}")
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


def process_file(filepath, main_graph):
    """Processes a single file, parses graph updates, and updates the main graph."""
    logging.info(f"Processing file: {filepath}")
    if main_graph is None:
        initial_graph = dot_to_nx(filepath)
        pytorch_data = nx_to_pyg(initial_graph, node_scaling='standard', edge_scaling='none')
        logging.info(
            f"Initialized main graph with {initial_graph.number_of_nodes()} nodes and {initial_graph.number_of_edges()} edges.")
        return initial_graph, pytorch_data
    else:
        updated_graph = update_nx_graph(main_graph, filepath)
        pytorch_data = nx_to_pyg(updated_graph, node_scaling='standard', edge_scaling='none')
        logging.info(
            f"Updated main graph to {updated_graph.number_of_nodes()} nodes and {updated_graph.number_of_edges()} edges.")
        return updated_graph, pytorch_data


def process_and_learn(directory, model_save_path, stats_save_path, anomaly_log_path, update_interval_seconds=60,
                      export_period_updates=50, visualization_path=None):
    """
    Monitors the directory for files, processes them, updates the graph,
    triggers online learning at fixed intervals, and exports embeddings periodically.
    """
    processed_files = set()
    main_graph = None
    main_data = None
    last_update_time = time.time()
    gnn_model = None
    processed_count = 0
    export_counter = 0

    # Training parameters
    initial_training_epochs = 50  # Only for first training
    online_update_steps = 25  # For subsequent updates

    logging.info(
        f"Starting process and learn in directory: {directory} with update interval: {update_interval_seconds} seconds, "
        f"visualization every {export_period_updates} updates.")

    while True:
        # Check for new files
        all_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files_with_timestamps = []

        for filename in all_files:
            timestamp_dt = extract_timestamp_from_epoch(filename)
            if timestamp_dt:
                files_with_timestamps.append((timestamp_dt, os.path.join(directory, filename), filename))

        files_with_timestamps.sort(key=lambda item: item[0])

        # Process new files
        for timestamp_dt, filepath, filename in files_with_timestamps:
            if filename not in processed_files:
                logging.info(f"Found new file: {filename} (Timestamp: {timestamp_dt})")
                main_graph, main_data = process_file(filepath, main_graph)
                processed_files.add(filename)
                processed_count += 1

        # Periodic graph update and learning
        current_time = time.time()
        if current_time - last_update_time >= update_interval_seconds:
            logging.info(
                f"\n--- Graph Update Cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Processed {processed_count} files) ---")

            if main_data is not None:
                # Initialize model if needed
                if gnn_model is None:
                    node_feature_dim = main_data.x.size(1)
                    edge_feature_dim = main_data.edge_attr.size(1) if main_data.edge_attr is not None else 0
                    gnn_model = HybridGNNAnomalyDetector(node_feature_dim, edge_feature_dim,
                                                         export_dir=visualization_path)
                    logging.info(f"Initialized Hybrid GNN (node_dim={node_feature_dim}, edge_dim={edge_feature_dim})")

                    # Initial training (optional, can be removed if purely online)
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

                # Save anomalies to file, providing the NetworkX graph
                save_anomalies_to_file(main_data, anomalies, processed_count, anomaly_log_path, nx_graph=main_graph)

                # Save running statistics
                if gnn_model:
                    save_running_stats(gnn_model.node_stats, gnn_model.edge_stats, stats_save_path)

                # Save checkpoint periodically
                if gnn_model:
                    save_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler, processed_count,
                                    model_save_path)

                # Report results
                logging.info(f"\nGlobal anomaly score: {anomalies['global_anomaly_mlp']:.4f}")

                # Node anomalies
                if len(anomalies['node_anomalies_recon']) > 0:
                    logging.warning(
                        f"Detected {len(anomalies['node_anomalies_recon'])} anomalous nodes (reconstruction-based):")
                    for idx in anomalies['node_anomalies_recon'][:5]:  # Show top 5
                        logging.warning(
                            f"  IP: {main_graph.nodes.get(idx.item(), {}).get('ip', 'N/A')}, Recon Error: {anomalies['node_recon_errors'][idx]:.4f}, MLP Score: {anomalies['node_scores_mlp'][idx]:.4f}")
                else:
                    logging.info("No significant node anomalies detected (reconstruction-based)")

                # Edge anomalies
                if len(anomalies['edge_anomalies_recon']) > 0:
                    logging.warning(
                        f"Detected {len(anomalies['edge_anomalies_recon'])} anomalous edges (reconstruction-based):")
                    for idx in anomalies['edge_anomalies_recon'][:5]:  # Show top 5
                        src = main_data.edge_index[0][idx].item()
                        dst = main_data.edge_index[1][idx].item()
                        logging.debug(f"Value of idx: {idx}")
                        logging.debug(
                            f"Shape of anomalies['edge_recon_errors']: {anomalies['edge_recon_errors'].shape}")
                        logging.debug(f"Shape of anomalies['edge_scores_mlp']: {anomalies['edge_scores_mlp'].shape}")
                        logging.warning(
                            f"  Source IP: {main_graph.nodes.get(src, {}).get('ip', 'N/A')}, Target IP: {main_graph.nodes.get(dst, {}).get('ip', 'N/A')}, Recon Error: {anomalies['edge_recon_errors'][idx].item():.4f}, MLP Score: {anomalies['edge_scores_mlp'][idx].item():.4f}")
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
                            if main_data.x is not None and main_data.edge_attr is not None:
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

            last_update_time = current_time

        # Wait before next check
        sleep_time = min(5, update_interval_seconds / 2)  # Check frequently but not too fast
        time.sleep(sleep_time)
