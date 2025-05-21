import time
from neural_net import HybridGNNAnomalyDetector
from graph_utils import *
from visualization import *

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
    # Example patterns: adjust as needed to match how the epoch is in your filenames
    pattern = r"_(\d+)\."         # Matches digits between an underscore and a dot (e.g., file_1678886400.txt)

    match = re.search(pattern, filename)
    if match:
        try:
            epoch_seconds = int(match.group(1))
            return datetime.fromtimestamp(epoch_seconds)
        except ValueError:
            return None
    return None

def process_file(filepath, main_graph, main_data):
    """Processes a single file, parses graph updates, and updates the main graph."""
    print(f"Processing file: {filepath}")
    if main_graph is None:
        initial_graph = dot_to_nx(filepath)
        pytorch_data = nx_to_pyg(initial_graph, node_scaling='standard', edge_scaling='none')
        return initial_graph, pytorch_data
    else:
        updated_graph = update_nx_graph(main_graph, filepath)
        pytorch_data = nx_to_pyg(updated_graph, node_scaling='standard', edge_scaling='none')
        return updated_graph, pytorch_data


def process_and_learn(directory, update_interval_seconds=60):
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
                print(f"Found new file: {filename} (Timestamp: {timestamp_dt})")
                main_graph, main_data = process_file(filepath, main_graph, main_data)
                processed_files.add(filename)

        # Periodic graph update and learning
        current_time = time.time()
        if current_time - last_update_time >= update_interval_seconds:
            print(f"\n--- Graph Update Cycle at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            if main_data is not None:
                # Initialize model if needed
                if gnn_model is None:
                    node_feature_dim = main_data.x.size(1)
                    edge_feature_dim = main_data.edge_attr.size(1)
                    gnn_model = HybridGNNAnomalyDetector(node_feature_dim, edge_feature_dim)
                    print(f"Initialized Hybrid GNN (node_dim={node_feature_dim}, edge_dim={edge_feature_dim})")

                    # Initial training
                    print("Performing initial training...")
                    for epoch in range(initial_training_epochs):
                        loss = gnn_model.update_online(main_data, n_steps=5)
                        print(f"Epoch {epoch+1}/{initial_training_epochs}, Loss: {loss:.4f}")
                else:
                    # Online update
                    print("Performing online update...")
                    loss = gnn_model.update_online(main_data, n_steps=online_update_steps)
                    print(f"Online update complete. Loss: {loss:.4f}")

                # Detect and report anomalies
                print("\nRunning anomaly detection...")
                anomalies = gnn_model.detect_anomalies(main_data)

                # Report results
                print(f"\nGlobal anomaly score: {anomalies['global_anomaly']:.4f}")

                # Node anomalies
                if len(anomalies['node_anomalies']) > 0:
                    print(f"\nDetected {len(anomalies['node_anomalies'])} anomalous nodes:")
                    for idx in anomalies['node_anomalies'][:5]:  # Show top 5
                        print(f"  Node {idx}: score={anomalies['node_scores'][idx]:.4f}")
                else:
                    print("\nNo significant node anomalies detected")

                # Edge anomalies
                if len(anomalies['edge_anomalies']) > 0:
                    print(f"\nDetected {len(anomalies['edge_anomalies'])} anomalous edges:")
                    for idx in anomalies['edge_anomalies'][:5]:  # Show top 5
                        src = main_data.edge_index[0][idx].item()
                        dst = main_data.edge_index[1][idx].item()
                        print(f"  Edge {src}->{dst}: score={anomalies['edge_scores'][idx]:.4f}")
                else:
                    print("\nNo significant edge anomalies detected")

                # Visualization (optional)
                if len(processed_files) % 5 == 0:  # Visualize every 5 updates
                    visualize_node_features(main_data)
                    visualize_edge_features(main_data)

            last_update_time = current_time

        # Wait before next check
        sleep_time = min(5, update_interval_seconds/2)  # Check frequently but not too fast
        time.sleep(sleep_time)