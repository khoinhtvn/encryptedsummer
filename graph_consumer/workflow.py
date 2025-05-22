import time
from neural_net import HybridGNNAnomalyDetector
from graph_utils import *
from visualization import *
import logging
import os
import re
from datetime import datetime
import pickle
import json

MODEL_SAVE_PATH = "model_checkpoints"
STATS_SAVE_PATH = "stats"
ANOMALY_LOG_PATH = "anomaly_logs"

if not os.path.exists(ANOMALY_LOG_PATH):
    os.makedirs(ANOMALY_LOG_PATH)
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)
if not os.path.exists(STATS_SAVE_PATH):
    os.makedirs(STATS_SAVE_PATH)

def save_anomalies_to_file(main_data, anomalies, processed_files_count, timestamp=None):
    """Saves detected anomalies to a JSON file."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(ANOMALY_LOG_PATH, f"anomalies_{timestamp}_update_{processed_files_count}.json")
    anomaly_data = {
        "timestamp": timestamp,
        "update_count": processed_files_count,
        "global_anomaly_score": anomalies.get('global_anomaly', None),
        "node_anomalies": [
            {"node_id": idx.item(), "score": anomalies['node_scores'][idx].item()}
            for idx in anomalies.get('node_anomalies', [])
        ],
        "edge_anomalies": [
            {"source": main_data.edge_index[0][idx].item(),
             "target": main_data.edge_index[1][idx].item(),
             "score": anomalies['edge_scores'][idx].item()}
            for idx in anomalies.get('edge_anomalies', []) if main_data is not None and main_data.edge_index is not None
        ]
    }
    try:
        with open(filename, 'w') as f:
            json.dump(anomaly_data, f, indent=4)
        logging.info(f"Anomalies saved to: {filename}")
    except Exception as e:
        logging.error(f"Error saving anomalies to file: {e}")

def save_checkpoint(model, optimizer, scheduler, epoch, filename="latest_checkpoint.pth"):
    filepath = os.path.join(MODEL_SAVE_PATH, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'node_stats': model.node_stats,
        'edge_stats': model.edge_stats,
    }, filepath)
    logging.info(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, scheduler, filename="latest_checkpoint.pth"):
    filepath = os.path.join(MODEL_SAVE_PATH, filename)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        model.node_stats = checkpoint['node_stats']
        model.edge_stats = checkpoint['edge_stats']
        epoch = checkpoint['epoch']
        logging.info(f"Checkpoint loaded from {filepath} at epoch {epoch}")
        return epoch
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
            model.node_stats = stats['node_stats']
            model.edge_stats = stats['edge_stats']
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
    pattern = r"_(\d+)\."    # Matches digits between an underscore and a dot (e.g., file_1678886400.txt)

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
        logging.info(f"Initialized main graph with {initial_graph.number_of_nodes()} nodes and {initial_graph.number_of_edges()} edges.")
        return initial_graph, pytorch_data
    else:
        updated_graph = update_nx_graph(main_graph, filepath)
        pytorch_data = nx_to_pyg(updated_graph, node_scaling='standard', edge_scaling='none')
        logging.info(f"Updated main graph to {updated_graph.number_of_nodes()} nodes and {updated_graph.number_of_edges()} edges.")
        return updated_graph, pytorch_data


def process_and_learn(directory, update_interval_seconds=60, visualize = False):
    """
    Monitors the directory for files, processes them, updates the graph,
    and triggers online learning at fixed intervals.
    """
    processed_files = set()
    main_graph = None
    main_data = None
    last_update_time = time.time()
    gnn_model = None

    # Training parameters
    initial_training_epochs = 10  # Only for first training
    online_update_steps = 5       # For subsequent updates

    logging.info(f"Starting process and learn in directory: {directory} with update interval: {update_interval_seconds} seconds.")

    # Load checkpoint if available
    start_epoch = load_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler)

    # Load running statistics if available
    load_running_stats(gnn_model)

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

        # Periodic graph update and learning
        current_time = time.time()
        if current_time - last_update_time >= update_interval_seconds:
            logging.info(f"\n--- Graph Update Cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            if main_data is not None:
                # Initialize model if needed
                if gnn_model is None:
                    node_feature_dim = main_data.x.size(1)
                    edge_feature_dim = main_data.edge_attr.size(1) if main_data.edge_attr is not None else 0
                    gnn_model = HybridGNNAnomalyDetector(node_feature_dim, edge_feature_dim)
                    logging.info(f"Initialized Hybrid GNN (node_dim={node_feature_dim}, edge_dim={edge_feature_dim})")

                    # Initial training
                    logging.info("Performing initial training...")
                    for epoch in range(initial_training_epochs):
                        loss = gnn_model.update_online(main_data, n_steps=5)
                        logging.info(f"Epoch {epoch+1}/{initial_training_epochs}, Loss: {loss:.4f}")
                else:
                    # Online update
                    logging.info("Performing online update...")
                    loss = gnn_model.update_online(main_data, n_steps=online_update_steps)
                    logging.info(f"Online update complete. Loss: {loss:.4f}")

                # Detect and report anomalies
                logging.info("Running anomaly detection...")
                anomalies = gnn_model.detect_anomalies(main_data)

                # Save anomalies to file
                save_anomalies_to_file(main_data, anomalies, processed_files)
                # Report results
                logging.info(f"\nGlobal anomaly score: {anomalies['global_anomaly']:.4f}")

                # Node anomalies
                if len(anomalies['node_anomalies']) > 0:
                    logging.warning(f"Detected {len(anomalies['node_anomalies'])} anomalous nodes:")
                    for idx in anomalies['node_anomalies'][:5]:  # Show top 5
                        logging.warning(f"  Node {idx}: score={anomalies['node_scores'][idx]:.4f}")
                else:
                    logging.info("No significant node anomalies detected")

                # Edge anomalies
                if len(anomalies['edge_anomalies']) > 0:
                    logging.warning(f"Detected {len(anomalies['edge_anomalies'])} anomalous edges:")
                    for idx in anomalies['edge_anomalies'][:5]:  # Show top 5
                        src = main_data.edge_index[0][idx].item()
                        dst = main_data.edge_index[1][idx].item()
                        logging.warning(f"  Edge {src}->{dst}: score={anomalies['edge_scores'][idx]:.4f}")
                else:
                    logging.info("No significant edge anomalies detected")

                # Visualization (optional)
                if visualize and (len(processed_files) % 5 == 0 and main_data.x is not None and main_data.edge_attr is not None):
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

                # Save checkpoint periodically
                if len(processed_files) % 10 == 0:
                    save_checkpoint(gnn_model, gnn_model.optimizer, gnn_model.scheduler, len(processed_files))

                # Save running statistics periodically or at the end
                save_running_stats(gnn_model.node_stats, gnn_model.edge_stats)
            last_update_time = current_time

        # Wait before next check
        sleep_time = min(5, update_interval_seconds/2)  # Check frequently but not too fast
        time.sleep(sleep_time)