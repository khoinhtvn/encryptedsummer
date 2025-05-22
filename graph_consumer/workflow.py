import json
import logging
import os
import pickle
import re
import time
from datetime import datetime

from graph_utils import *
from neural_net import HybridGNNAnomalyDetector
from visualization import *

MODEL_SAVE_PATH = "model_checkpoints"
STATS_SAVE_PATH = "stats"
ANOMALY_LOG_PATH = "anomaly_logs"

if not os.path.exists(ANOMALY_LOG_PATH):
    os.makedirs(ANOMALY_LOG_PATH)
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
if not os.path.exists(STATS_SAVE_PATH):
    os.makedirs(STATS_SAVE_PATH)


def save_anomalies_to_file(main_data, anomalies, processed_files_count, nx_graph=None, timestamp=None):
    """Saves detected anomalies to a JSON file with more details."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(ANOMALY_LOG_PATH, f"anomalies_{timestamp}_update_{processed_files_count}.json")
    anomaly_data = {
        "timestamp": timestamp,
        "update_count": processed_files_count,
        "global_anomaly_score": anomalies.get('global_anomaly', None),
        "node_anomalies": [],
        "edge_anomalies": []
    }

    if nx_graph is not None:
        for idx in anomalies.get('node_anomalies', []):
            node_id = idx.item()
            node_info = {"node_id": node_id, "score": anomalies['node_scores'][idx].item()}
            if node_id in nx_graph.nodes:
                node_info['ip'] = node_id  # Assuming node_id IS the IP address
            anomaly_data["node_anomalies"].append(node_info)

        if main_data is not None and main_data.edge_index is not None:
            for idx in anomalies.get('edge_anomalies', []):
                src = main_data.edge_index[0][idx].item()
                dst = main_data.edge_index[1][idx].item()
                edge_info = {
                    "source_node_id": src,
                    "target_node_id": dst,
                    "score": anomalies['edge_scores'][idx].item()
                }
                if src in nx_graph.nodes:
                    edge_info['source_ip'] = src  # Assuming node_id IS the IP address
                if dst in nx_graph.nodes:
                    edge_info['target_ip'] = dst  # Assuming node_id IS the IP address
                anomaly_data["edge_anomalies"].append(edge_info)
    else:
        logging.warning("NetworkX graph not provided, cannot include IP addresses in anomaly details.")
        for idx in anomalies.get('node_anomalies', []):
            anomaly_data["node_anomalies"].append(
                {"node_id": idx.item(), "score": anomalies['node_scores'][idx].item()})
        if main_data is not None and main_data.edge_index is not None:
            for idx in anomalies.get('edge_anomalies', []):
                anomaly_data["edge_anomalies"].append({
                    "source_node_id": main_data.edge_index[0][idx].item(),
                    "target_node_id": main_data.edge_index[1][idx].item(),
                    "score": anomalies['edge_scores'][idx].item()
                })

    try:
        with open(filename, 'w') as f:
            json.dump(anomaly_data, f, indent=4)
        logging.info(f"Anomalies saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving anomalies to file: {e}")


def save_checkpoint(model, optimizer, scheduler, epoch, processed_files_count, filename_prefix="checkpoint"):
    """Saves the model checkpoint with the processed files count."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(MODEL_SAVE_PATH, f"{filename_prefix}_{timestamp}_update_{processed_files_count}.pth")
    obj = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'node_stats': model.node_stats,
        'edge_stats': model.edge_stats,
    }
    torch.save(obj, filename)
    os.remove(os.path.join(MODEL_SAVE_PATH,"latest_checkpoint.pth"))
    torch.save(obj, os.path.join(MODEL_SAVE_PATH,"latest_checkpoint.pth"))
    logging.info(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, scheduler, filename="latest_checkpoint.pth"):
    filepath = os.path.join(MODEL_SAVE_PATH, filename)
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


def save_running_stats(node_stats, edge_stats, filename="running_stats.pkl"):
    filepath = os.path.join(STATS_SAVE_PATH, filename)
    with open(filepath, 'wb') as f:
        pickle.dump({'node_stats': node_stats, 'edge_stats': edge_stats}, f)
    logging.info(f"Running statistics saved to {filepath}")


def load_running_stats(model, filename="running_stats.pkl"):
    filepath = os.path.join(STATS_SAVE_PATH, filename)
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


def process_file(filepath, main_graph, main_data):
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


def process_and_learn(directory, update_interval_seconds=60, visualize=False):
    """
    Monitors the directory for files, processes them, updates the graph,
    and triggers online learning at fixed intervals.
    """
    processed_files = set()
    main_graph = None
    main_data = None
    last_update_time = time.time()
    gnn_model = None
    processed_count = 0

    # Training parameters
    initial_training_epochs = 10  # Only for first training
    online_update_steps = 5  # For subsequent updates

    logging.info(
        f"Starting process and learn in directory: {directory} with update interval: {update_interval_seconds} seconds.")

    # Initialize a dummy model for loading running stats
    temp_model_for_stats = HybridGNNAnomalyDetector(1, 1)
    load_running_stats(temp_model_for_stats)
    del temp_model_for_stats


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
                main_graph, main_data = process_file(filepath, main_graph, main_data)
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
                    gnn_model = HybridGNNAnomalyDetector(node_feature_dim, edge_feature_dim)
                    logging.info(f"Initialized Hybrid GNN (node_dim={node_feature_dim}, edge_dim={edge_feature_dim})")

                    # Load latest checkpoint if available AFTER model initialization
                    start_epoch = load_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler)
                    logging.info(f"Starting or resuming training from epoch: {start_epoch}")
                    load_running_stats(gnn_model)  # Load stats again for the actual model
                else:
                    # Online update
                    logging.info("Performing online update...")
                    loss = gnn_model.update_online(main_data, n_steps=online_update_steps)
                    logging.info(f"Online update complete. Loss: {loss:.4f}")

                # Detect anomalies
                logging.info("Running anomaly detection...")
                anomalies = gnn_model.detect_anomalies(main_data)

                # Save anomalies to file, providing the NetworkX graph
                save_anomalies_to_file(main_data, anomalies, processed_count, main_graph)

                # Save running statistics
                if gnn_model:
                    save_running_stats(gnn_model.node_stats, gnn_model.edge_stats)

                # Save checkpoint periodically
                if gnn_model:
                    save_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler, processed_count,
                                    processed_count)

                # Report results
                logging.info(f"\nGlobal anomaly score: {anomalies['global_anomaly']:.4f}")

                # Node anomalies
                if len(anomalies['node_anomalies']) > 0:
                    logging.warning(f"Detected {len(anomalies['node_anomalies'])} anomalous nodes:")
                    for idx in anomalies['node_anomalies'][:5]:  # Show top 5
                        logging.warning(
                            f"  IP: {main_graph.nodes.get(idx.item(), {}).get('ip', 'N/A')}, Score: {anomalies['node_scores'][idx]:.4f}")
                else:
                    logging.info("No significant node anomalies detected")

                # Edge anomalies
                if len(anomalies['edge_anomalies']) > 0:
                    logging.warning(f"Detected {len(anomalies['edge_anomalies'])} anomalous edges:")
                    for idx in anomalies['edge_anomalies'][:5]:  # Show top 5
                        src = main_data.edge_index[0][idx].item()
                        dst = main_data.edge_index[1][idx].item()
                        logging.warning(
                            f"  Source IP: {main_graph.nodes.get(src, {}).get('ip', 'N/A')}, Target IP: {main_graph.nodes.get(dst, {}).get('ip', 'N/A')}, Score: {anomalies['edge_scores'][idx]:.4f}")
                else:
                    logging.info("No significant edge anomalies detected")

                # Visualization (optional)
                if visualize and (
                        len(processed_files) % 5 == 0 and main_data.x is not None and main_data.edge_attr is not None):
                    logging.info("Performing visualization of node and edge features.")
                    try:
                        visualize_node_features(main_data)
                        visualize_edge_features(main_data)
                    except Exception as e:
                        logging.error(f"Error during visualization: {e}")
                elif main_data.x is None:
                    logging.warning("Skipping node feature visualization: No node features available.")
                elif main_data.edge_attr is None:
                    logging.warning("Skipping edge feature visualization: No edge features available.")

            last_update_time = current_time

        # Wait before next check
        sleep_time = min(5, update_interval_seconds / 2)  # Check frequently but not too fast
        time.sleep(sleep_time)
