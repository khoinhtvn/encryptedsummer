import logging
import os
import re
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np  # Ensure numpy is imported
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.data import Data

# Global variables
node_scaler = None
edge_scaler = None
last_update_time = None
update_interval = 3600  # seconds (e.g., 1 hour)
NODE_FLOAT_FEATURES = [
    "first_seen_hour_minute_sin",
    "first_seen_hour_minute_cos",
    "last_seen_hour_minute_sin",
    "last_seen_hour_minute_cos",
    "outgoing_connection_ratio",
    "incoming_connection_ratio",
    "unique_remote_ports_connected_to",
    "unique_local_ports_used",
    "unique_remote_ports_connected_from",
    "unique_local_ports_listening_on",
    "unique_http_versions_used",
    "unique_http_status_codes_seen",
    "unique_ssl_versions_used",
    "unique_ssl_ciphers_used",
    "has_ssl_resumption",
    "has_ssl_server_name",
    "has_ssl_history",
    "ever_connected_to_privileged_port",
    "ever_listened_on_privileged_port",
]
EDGE_FLOAT_FEATURES = []  # Add edge feature names if applicable


def dot_to_nx(dot_file):
    """
    Parses a .dot file representing a graph in the ZeekTraffic format
    and returns a NetworkX MultiDiGraph. Handles node and edge attributes.

    Args:
        dot_file (str): Path to the .dot file.

    Returns:
        networkx.MultiDiGraph: A NetworkX MultiDiGraph object.
    """
    graph = nx.MultiDiGraph()
    logging.info(f"Processing .dot file: {dot_file}")
    try:
        with open(dot_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"Error: .dot file not found at '{dot_file}'")
        return graph  # Return an empty graph or raise an exception as needed
    except Exception as e:
        logging.error(f"Error reading .dot file '{dot_file}': {e}")
        logging.error(traceback.format_exc())
        return graph

    # Extract node information
    node_pattern = re.compile(r'^\s*"([^"]+)"\s*\[(.*?)\];\s*$', re.MULTILINE)
    for match in node_pattern.finditer(content):
        node_id = match.group(1)
        attributes_str = match.group(2).strip()
        attributes = {}
        for attr in attributes_str.split(','):
            if '=' in attr:
                key, value = attr.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"')
                try:
                    attributes[key] = float(value)
                except ValueError:
                    attributes[key] = value
            else:
                logging.warning(f"Skipping malformed node attribute: '{attr}' in node '{node_id}'")
        graph.add_node(node_id, **attributes)
    logging.info(f"Parsed {len(graph.nodes)} nodes from '{dot_file}'.")

    # Extract edge information
    edge_pattern = re.compile(r'"([^"]+)" -> "([^"]+)" \[(.*?)\];')
    edge_count = 0
    for match in edge_pattern.finditer(content):
        source, target = match.group(1), match.group(2)
        attributes_str = match.group(3).strip()
        attributes = {}
        if attributes_str:
            for attr in attributes_str.split(','):
                if '=' in attr:
                    key, value = attr.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')
                    try:
                        attributes[key] = float(value)
                    except ValueError:
                        attributes[key] = value
                else:
                    logging.warning(f"Skipping malformed edge attribute: '{attr}' between '{source}' and '{target}'")
        graph.add_edge(source, target, **attributes)
        edge_count += 1
    logging.info(f"Parsed {edge_count} edges from '{dot_file}'.")

    return graph


def visualize_nx_graph(nx_graph, layout_algorithm='spring', output_path=None):
    """
    Visualizes a NetworkX graph, handling multiple edges and allowing saving to a file.

    Args:
        nx_graph (networkx.MultiDiGraph): The NetworkX graph to visualize.
        layout_algorithm (str, optional): The layout algorithm to use.
            Defaults to 'spring'.
        output_path (str, optional): Path to save the visualization.
    """
    logging.info(
        f"Visualizing NetworkX graph with {len(nx_graph.nodes)} nodes and {nx_graph.number_of_edges()} edges using '{layout_algorithm}' layout.")
    plt.figure(figsize=(12, 10))

    if layout_algorithm == 'spring':
        pos = nx.spring_layout(nx_graph, k=0.3, iterations=100, seed=42)
    elif layout_algorithm == 'circular':
        pos = nx.circular_layout(nx_graph)
    elif layout_algorithm == 'spectral':
        pos = nx.spectral_layout(nx_graph)
    elif layout_algorithm == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(nx_graph)
    elif layout_algorithm == 'random':
        pos = nx.random_layout(nx_graph)
    else:
        pos = nx.spring_layout(nx_graph)
        logging.warning(f"Invalid layout algorithm '{layout_algorithm}'. Using default 'spring' layout.")

    # Draw nodes
    nx.draw_networkx_nodes(nx_graph, pos, node_size=1000, node_color="lightblue")
    nx.draw_networkx_labels(nx_graph, pos, font_size=8, font_weight="bold")

    # Draw edges with different styles for multiple edges
    for u, v, k, data in nx_graph.edges(data=True, keys=True):
        if nx_graph.number_of_edges(u, v) > 1:
            # Distinguish parallel edges visually
            if k == 0:
                edge_color = "red"
                style = "solid"
            elif k == 1:
                edge_color = "green"
                style = "dashed"
            elif k == 2:
                edge_color = "blue"
                style = "dotted"
            else:
                edge_color = "gray"
                style = "solid"
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], edge_color=edge_color, style=style, arrows=True)
        else:
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], edge_color="black", style="solid", arrows=True)

    # Add edge labels (show attributes)
    edge_labels = {(u, v): str(d) for u, v, d in nx_graph.edges(data=True)}  # Get all edge attributes
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=6)

    plt.title(f"NetworkX Graph Visualization ({layout_algorithm} Layout)")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        logging.info(f"Graph visualization saved to '{output_path}'.")
    else:
        plt.show()

    plt.close()


def save_nx_graph(nx_graph, output_path):
    """
    Saves a NetworkX graph to a file. Uses a format that preserves multiple edges
    and their attributes.

    Args:
        nx_graph (networkx.MultiDiGraph): The NetworkX graph to save.
        output_path (str): Path to save the graph.
    """
    logging.info(f"Saving NetworkX graph to '{output_path}'.")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nx.write_gpickle(nx_graph, output_path)
        logging.info(f"Graph saved to '{output_path}' in Pickle format.")
    except Exception as e:
        logging.error(f"Error saving graph to '{output_path}': {e}")
        logging.error(traceback.format_exc())


def get_sorted_node_features(nx_graph):
    all_node_attr_keys = set()
    for node in nx_graph.nodes():
        all_node_attr_keys.update(nx_graph.nodes[node].keys())
    sorted_node_attr_keys = sorted(list(all_node_attr_keys))
    return sorted_node_attr_keys


def get_sorted_edge_features(nx_graph):
    # Edge attributes
    edge_attr_keys = set()
    for u, v, data in nx_graph.edges(data=True):
        edge_attr_keys.update(data.keys())
    sorted_edge_attr_keys = sorted(list(edge_attr_keys))
    return sorted_edge_attr_keys


def nx_to_pyg(nx_graph, node_scaling='none', edge_scaling='none', fit_scaler=True):
    global node_scaler, edge_scaler, last_update_time, update_interval

    current_time = datetime.now().timestamp()
    logging.info("Starting NetworkX to PyG conversion.")

    sorted_node_attr_keys = get_sorted_node_features(nx_graph)
    logging.debug(f"Sorted node attribute keys: {sorted_node_attr_keys}")

    node_features = []
    node_to_index = {node: i for i, node in enumerate(nx_graph.nodes())}
    for node in nx_graph.nodes():
        node_feature_list = []
        for key in sorted_node_attr_keys:
            value = nx_graph.nodes[node].get(key)
            if isinstance(value, (int, float)):
                node_feature_list.append(value)
            else:
                node_feature_list.append(0.0)
        node_features.append(node_feature_list)
    x = torch.tensor(node_features, dtype=torch.float)
    logging.debug(f"Initial node features tensor shape: {x.shape}")

    # 3. Apply Node Feature Scaling
    if node_scaling == 'standard':
        if x.numel() > 0 and x.size(1) > 0:  # Check if there are features to scale
            # Identify numerical features for scaling
            numerical_node_indices = [sorted_node_attr_keys.index(f) for f in NODE_FLOAT_FEATURES if
                                      f in sorted_node_attr_keys]
            if numerical_node_indices:
                x_numerical = x[:, numerical_node_indices].numpy()
                if np.std(x_numerical, axis=0).any():  # Check if any std is non-zero
                    if fit_scaler or node_scaler is None or (
                            last_update_time is not None and (current_time - last_update_time) > update_interval):
                        node_scaler = StandardScaler()
                        x_scaled_numerical = node_scaler.fit_transform(x_numerical)
                        if fit_scaler:
                            logging.info("Numerical Node Features (x) after initial standard scaling.")
                        else:
                            logging.info("Numerical Node Features (x) after periodic standard scaling.")
                        last_update_time = current_time
                    elif node_scaler is not None:
                        x_scaled_numerical = node_scaler.transform(x_numerical)
                        logging.info("Numerical Node Features (x) after online standard scaling.")
                    else:
                        logging.warning("Skipping standard scaling for node features: Scaler not initialized.")
                    x_scaled_tensor = torch.tensor(x_scaled_numerical, dtype=torch.float)
                    x[:, numerical_node_indices] = x_scaled_tensor
                else:
                    logging.warning(
                        "Skipping standard scaling for node features: Zero standard deviation in numerical features.")
            else:
                logging.info("No numerical node features found for standard scaling.")
        else:
            logging.warning("Skipping standard scaling for node features: No features to scale.")
    elif node_scaling == 'minmax':
        if x.numel() > 0 and x.size(1) > 0:
            numerical_node_indices = [sorted_node_attr_keys.index(f) for f in NODE_FLOAT_FEATURES if
                                      f in sorted_node_attr_keys]
            if numerical_node_indices:
                x_numerical = x[:, numerical_node_indices].numpy()
                x_min = np.min(x_numerical, axis=0)
                x_max = np.max(x_numerical, axis=0)
                # Avoid division by zero
                if np.any(x_max > x_min):
                    if fit_scaler or node_scaler is None or (
                            last_update_time is not None and (current_time - last_update_time) > update_interval):
                        node_scaler = MinMaxScaler()
                        x_scaled_numerical = node_scaler.fit_transform(x_numerical)
                        if fit_scaler:
                            logging.info("Numerical Node Features (x) after initial min-max scaling.")
                        else:
                            logging.info("Numerical Node Features (x) after periodic min-max scaling.")
                        last_update_time = current_time
                    elif node_scaler is not None:
                        x_scaled_numerical = node_scaler.transform(x_numerical)
                        logging.info("Numerical Node Features (x) after online min-max scaling.")
                    else:
                        logging.warning("Skipping min-max scaling for node features: Scaler not initialized.")
                    x_scaled_tensor = torch.tensor(x_scaled_numerical, dtype=torch.float)
                    x[:, numerical_node_indices] = x_scaled_tensor
                else:
                    logging.warning("Skipping min-max scaling for node features: All values are the same.")
            else:
                logging.info("No numerical node features found for min-max scaling.")
        else:
            logging.warning("Skipping min-max scaling for node features: No features to scale.")
    elif node_scaling == 'none':
        logging.info("No node feature scaling applied.")
    else:
        raise ValueError(f"Invalid node_scaling method: {node_scaling}.  Choose 'none', 'standard', or 'minmax'.")
    logging.debug(f"Node features tensor shape after scaling: {x.shape}")

    # Edge indices
    edge_index_list = []
    for u, v in nx_graph.edges():
        edge_index_list.append([node_to_index[u], node_to_index[v]])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    logging.debug(f"Edge index tensor shape: {edge_index.shape}")

    edge_features = []
    sorted_edge_attr_keys = get_sorted_edge_features(nx_graph)
    logging.debug(f"Sorted edge attribute keys: {sorted_edge_attr_keys}")

    for u, v, data in nx_graph.edges(data=True):
        edge_feature_list = []
        for key in sorted_edge_attr_keys:
            value = data.get(key)
            if isinstance(value, (int, float)):
                edge_feature_list.append(value)
            else:
                edge_feature_list.append(0.0)
        edge_features.append(edge_feature_list)

    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None
    if edge_attr is not None:
        logging.debug(f"Initial edge attributes tensor shape: {edge_attr.shape}")
        # 4. Apply Edge Feature Scaling
        if edge_scaling == 'standard':
            if edge_attr.numel() > 0 and edge_attr.size(1) > 0:
                numerical_edge_indices = [sorted_edge_attr_keys.index(f) for f in EDGE_FLOAT_FEATURES if
                                          f in sorted_edge_attr_keys]
                if numerical_edge_indices:
                    edge_attr_numerical = edge_attr[:, numerical_edge_indices].numpy()
                    if np.std(edge_attr_numerical, axis=0).any():
                        if fit_scaler or edge_scaler is None or (
                                last_update_time is not None and (current_time - last_update_time) > update_interval):
                            edge_scaler = StandardScaler()
                            edge_attr_scaled_numerical = edge_scaler.fit_transform(edge_attr_numerical)
                            if fit_scaler:
                                logging.info("Numerical Edge Attributes (edge_attr) after initial standard scaling.")
                            else:
                                logging.info("Numerical Edge Attributes (edge_attr) after periodic standard scaling.")
                            last_update_time = current_time
                        elif edge_scaler is not None:
                            edge_attr_scaled_numerical = edge_scaler.transform(edge_attr_numerical)
                            logging.info("Numerical Edge Attributes (edge_attr) after online standard scaling.")
                        else:
                            logging.warning("Skipping standard scaling for edge features: Scaler not initialized.")
                        edge_attr_scaled_tensor = torch.tensor(edge_attr_scaled_numerical, dtype=torch.float)
                        edge_attr[:, numerical_edge_indices] = edge_attr_scaled_tensor
                    else:
                        logging.warning(
                            "Skipping standard scaling for edge features: Zero standard deviation in numerical features.")
                else:
                    logging.warning("Skipping standard scaling for edge features: No features to scale.")
            elif edge_scaling == 'minmax':
                if edge_attr.numel() > 0 and edge_attr.size(1) > 0:
                    numerical_edge_indices = [sorted_edge_attr_keys.index(f) for f in EDGE_FLOAT_FEATURES if
                                              f in sorted_edge_attr_keys]
                    if numerical_edge_indices:
                        edge_attr_numerical = edge_attr[:, numerical_edge_indices].numpy()
                        edge_attr_min = np.min(edge_attr_numerical, axis=0)
                        edge_attr_max = np.max(edge_attr_numerical, axis=0)
                        if np.any(edge_attr_max > edge_attr_min):
                            if fit_scaler or edge_scaler is None or (
                                    last_update_time is not None and (
                                    current_time - last_update_time) > update_interval):
                                edge_scaler = MinMaxScaler()
                                edge_attr_scaled_numerical = edge_scaler.fit_transform(edge_attr_numerical)
                                if fit_scaler:
                                    logging.info(
                                        "Numerical Edge Attributes (edge_attr) after initial min-max scaling.")
                                else:
                                    logging.info(
                                        "Numerical Edge Attributes (edge_attr) after periodic min-max scaling.")
                                last_update_time = current_time
                            elif edge_scaler is not None:
                                edge_attr_scaled_numerical = edge_scaler.transform(edge_attr_numerical)
                                logging.info("Numerical Edge Attributes (edge_attr) after online min-max scaling.")
                            else:
                                logging.warning(
                                    "Skipping min-max scaling for edge features: Scaler not initialized.")
                            edge_attr_scaled_tensor = torch.tensor(edge_attr_scaled_numerical, dtype=torch.float)
                            edge_attr[:, numerical_edge_indices] = edge_attr_scaled_tensor
                        else:
                            logging.warning("Skipping min-max scaling for edge features: All values are the same.")
                    else:
                        logging.info("No numerical edge features found for min-max scaling.")
                else:
                    logging.warning("Skipping min-max scaling for edge features: No features to scale.")
            elif edge_scaling == 'none':
                logging.info("No edge feature scaling applied.")
            else:
                raise ValueError(
                    f"Invalid edge_scaling method: {edge_scaling}. Choose 'none', 'standard', or 'minmax'.")
            logging.debug(f"Edge attributes tensor shape after scaling: {edge_attr.shape}")
        else:
            logging.info("No edge attributes found.")

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            logging.info("Successfully converted NetworkX graph to PyG Data object.")
            return data


def update_nx_graph(nx_graph, update_dot_file):
    """
    Updates a NetworkX graph with the changes from an update .dot file in the
    ZeekTraffic format. Handles adding, modifying, and deleting nodes and edges.
    Assumes the update_dot_file contains the full updated graph.

    Args:
        nx_graphnx_graph (networkx.MultiDiGraph): The original NetworkX graph to update.
        update_dot_file (str): Path to the .dot file containing the updated graph.
    """
    logging.info(f"Updating NetworkX graph from '{update_dot_file}'.")
    update_graph = dot_to_nx(update_dot_file)  # Parse the update file

    # 1. Update or add nodes and their attributes
    logging.info("Processing node updates/additions.")
    for node_id, attrs in update_graph.nodes(data=True):
        if nx_graph.has_node(node_id):
            # Update existing node attributes
            for key, value in attrs.items():
                nx_graph.nodes[node_id][key] = value
            logging.debug(f"Updated node: {node_id} with attributes {attrs}.")
        else:
            # Add new node with attributes
            nx_graph.add_node(node_id, **attrs)
            logging.debug(f"Added node: {node_id} with attributes {attrs}.")

    # 2. Update or add edges and their attributes
    logging.info("Processing edge updates/additions.")
    for u, v, key, attrs in update_graph.edges(data=True, keys=True):
        if not nx_graph.has_node(u):
            nx_graph.add_node(u)
            logging.warning(f"Added missing node '{u}' during edge update.")
        if not nx_graph.has_node(v):
            nx_graph.add_node(v)
            logging.warning(f"Added missing node '{v}' during edge update.")
        if nx_graph.has_edge(u, v, key=key):
            # Update existing edge attributes
            for k, val in attrs.items():
                nx_graph[u][v][key][k] = val
            logging.debug(f"Updated edge: '{u}' -> '{v}' with key={key} and attributes {attrs}.")
        else:
            # Add new edge with attributes
            nx_graph.add_edge(u, v, key=key, **attrs)
            logging.debug(f"Added edge: '{u}' -> '{v}' with key={key} and attributes {attrs}.")

    # 3. Handle Deletions (Nodes and Edges)
    logging.info("Processing deletions (nodes and edges).")

    # Identify nodes to delete (in original but not in update)
    nodes_to_delete = [node for node in nx_graph.nodes() if node not in update_graph.nodes()]
    for node in nodes_to_delete:
        nx_graph.remove_node(node)
        logging.debug(f"Deleted node: {node}.")

    # Identify edges to delete (in original but not in update)
    edges_to_delete = []
    for u, v, key in nx_graph.edges(keys=True):
        if not update_graph.has_edge(u, v, key=key):
            edges_to_delete.append((u, v, key))
    for u, v, key in edges_to_delete:
        try:
            nx_graph.remove_edge(u, v, key=key)
            logging.debug(f"Deleted edge: '{u}' -> '{v}' with key={key}.")
        except nx.NetworkXError as e:
            logging.warning(f"Error deleting edge '{u}' -> '{v}' with key={key}: {e}")

    logging.info("Finished updating NetworkX graph.")
    return nx_graph
