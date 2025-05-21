import re
import networkx as nx
import traceback
import matplotlib.pyplot as plt
import os
import torch
from torch_geometric.data import Data
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Global variables
node_scaler = None
edge_scaler = None
last_update_time = None
update_interval = 3600  # seconds (e.g., 1 hour)

def dot_to_nx(dot_file):
    """
    Parses a .dot file representing a  graph and returns a NetworkX MultiDiGraph.
    Handles node and edge attributes.

    Args:
        dot_file (str): Path to the .dot file.

    Returns:
        networkx.MultiDiGraph: A NetworkX MultiDiGraph object.
    """
    graph = nx.MultiDiGraph()

    with open(dot_file, 'r') as f:
        content = f.read()

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
                print(f"Warning: Skipping malformed node attribute: '{attr}' in node '{node_id}'")
        graph.add_node(node_id, **attributes)

    # Extract edge information
    edge_pattern = re.compile(r'"([^"]+)" -> "([^"]+)" \[(.*?)\];')
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
                    print(f"Warning: Skipping malformed edge attribute: '{attr}' between '{source}' and '{target}'")
        graph.add_edge(source, target, **attributes)

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
        print(f"Graph saved to {output_path}")
    else:
        plt.show()

    plt.close()

def save_nx_graph(nx_graph, output_path):
    """
    Saves a NetworkX graph to a file.  Uses a format that preserves multiple edges
    and their attributes.

    Args:
        nx_graph (networkx.MultiDiGraph): The NetworkX graph to save.
        output_path (str): Path to save the graph.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nx.write_gpickle(nx_graph, output_path)
        print(f"Graph saved to {output_path} in Pickle format")
    except Exception as e:
        print(f"Error saving graph: {e}")
        traceback.print_exc()

def nx_to_pyg(nx_graph, node_scaling='none', edge_scaling='none', fit_scaler=True):
    """
    Converts a NetworkX MultiDiGraph to a PyTorch Geometric Data object,
    extracting features and applying preprocessing, handling online scaling.

    Args:
        nx_graph (networkx.MultiDiGraph): The input NetworkX graph.
        node_scaling (str, optional): Scaling method for node features ('none', 'standard', 'minmax').
            Defaults to 'none'.
        edge_scaling (str, optional): Scaling method for edge features ('none', 'standard', 'minmax').
            Defaults to 'none'.
        fit_scaler (bool, optional): Whether to fit the scalers on the input data.
            Defaults to True.  Set to False for online scaling after initial fitting.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object.
    """
    global node_scaler, edge_scaler, last_update_time, update_interval

    current_time = datetime.now().timestamp()

    # 1. Determine the complete set of node attributes.
    all_node_attr_keys = set()
    for node in nx_graph.nodes():
        all_node_attr_keys.update(nx_graph.nodes[node].keys())
    sorted_node_attr_keys = sorted(list(all_node_attr_keys))
    print("Sorted node attribute keys:", sorted_node_attr_keys)
    # 2.  Create node features
    node_features = []
    node_ip_addresses = []
    for node in nx_graph.nodes():
        node_ip_addresses.append(node)
        node_feature_list = []
        for key in sorted_node_attr_keys:
            value = nx_graph.nodes[node].get(key)
            if isinstance(value, (int, float)):
                node_feature_list.append(value)
            else:
                node_feature_list.append(0.0)
        node_features.append(node_feature_list)
    x = torch.tensor(node_features, dtype=torch.float)

    # 3. Apply Node Feature Scaling
    if node_scaling == 'standard':
        if fit_scaler or node_scaler is None or (last_update_time is not None and (current_time - last_update_time) > update_interval):
            node_scaler = StandardScaler()
            x = torch.tensor(node_scaler.fit_transform(x), dtype=torch.float)
            if fit_scaler:
                print("Node Features (x) after initial standard scaling:", x)
            else:
                print("Node Features (x) after periodic standard scaling:", x)
            last_update_time = current_time  # update last_update_time
        elif node_scaler is not None:
            x = torch.tensor(node_scaler.transform(x), dtype=torch.float)
            print("Node Features (x) after online standard scaling:", x)
        else:
            print("No node scaler fitted. Skipping scaling.")
    elif node_scaling == 'minmax':
        if fit_scaler or node_scaler is None or (last_update_time is not None and (current_time - last_update_time) > update_interval):
            node_scaler = MinMaxScaler()
            x = torch.tensor(node_scaler.fit_transform(x), dtype=torch.float)
            if fit_scaler:
                print("Node Features (x) after initial min-max scaling:", x)
            else:
                print("Node Features (x) after periodic min-max scaling:", x)
            last_update_time = current_time
        elif node_scaler is not None:
            x = torch.tensor(node_scaler.transform(x), dtype=torch.float)
            print("Node Features (x) after online min-max scaling:", x)
        else:
            print("No node scaler fitted. Skipping scaling.")
    elif node_scaling == 'none':
        print("No node feature scaling applied.")
    else:
        raise ValueError(f"Invalid node_scaling method: {node_scaling}.  Choose 'none', 'standard', or 'minmax'.")

    # Edge indices
    edge_index = torch.tensor(
        [[list(nx_graph.nodes).index(u), list(nx_graph.nodes).index(v)] for u, v in nx_graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Edge attributes
    edge_features = []
    edge_attr_keys = set()
    for u, v, data in nx_graph.edges(data=True):
        edge_attr_keys.update(data.keys())
    sorted_edge_attr_keys = sorted(list(edge_attr_keys))
    print("Sorted edge attribute keys:", sorted_edge_attr_keys) # print the sorted keys
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

    # 4. Apply Edge Feature Scaling
    if edge_attr is not None:
        if edge_scaling == 'standard':
            if fit_scaler or edge_scaler is None or (last_update_time is not None and (current_time - last_update_time) > update_interval):
                edge_scaler = StandardScaler()
                edge_attr = torch.tensor(edge_scaler.fit_transform(edge_attr), dtype=torch.float)
                if fit_scaler:
                    print("Edge Attributes (edge_attr) after initial standard scaling:", edge_attr)
                else:
                    print("Edge Attributes (edge_attr) after periodic standard scaling:", edge_attr)
            elif edge_scaler is not None:
                edge_attr = torch.tensor(edge_scaler.transform(edge_attr), dtype=torch.float)
                print("Edge Attributes (edge_attr) after online standard scaling:", edge_attr)
            else:
                print("No edge scaler fitted. Skipping scaling")
        elif edge_scaling == 'minmax':
            if fit_scaler or edge_scaler is None or (last_update_time is not None and (current_time - last_update_time) > update_interval):
                edge_scaler = MinMaxScaler()
                edge_attr = torch.tensor(edge_scaler.fit_transform(edge_attr), dtype=torch.float)
                if fit_scaler:
                    print("Edge Attributes (edge_attr) after initial min-max scaling:", edge_attr)
                else:
                    print("Edge Attributes (edge_attr) after periodic min-max scaling:", edge_attr)
            elif edge_scaler is not None:
                edge_attr = torch.tensor(edge_scaler.transform(edge_attr), dtype=torch.float)
                print("Edge Attributes (edge_attr) after online min-max scaling:", edge_attr)
            else:
                print("No edge scaler fitted. Skipping scaling.")
        elif edge_scaling == 'none':
            print("No edge feature scaling applied.")
        else:
            raise ValueError(f"Invalid edge_scaling method: {edge_scaling}.  Choose 'none', 'standard', or 'minmax'.")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def update_nx_graph(nx_graph, update_dot_file):
    """
    Updates a NetworkX graph with the changes from an update .dot file.
    Handles adding, modifying, and deleting nodes and edges.  Assumes the
    update_dot_file contains only the *changes* to the graph.

    Args:
        nx_graph (networkx.MultiDiGraph): The original NetworkX graph to update.
        update_dot_file (str): Path to the .dot file containing the updates.
    """
    update_graph = dot_to_nx(update_dot_file)  # Use the existing dot_to_nx function

    # 1. Update or add nodes
    for node_id in update_graph.nodes():
        if nx_graph.has_node(node_id):
            # Update existing node attributes
            for key, value in update_graph.nodes[node_id].items():
                nx_graph.nodes[node_id][key] = value
            print(f"Updated node: {node_id}")
        else:
            # Add new node with attributes
            nx_graph.add_node(node_id, **update_graph.nodes[node_id])
            print(f"Added node: {node_id}")

    # 2. Update or add edges
    for u, v, key, data in update_graph.edges(data=True, keys=True):
        """
        if nx_graph.has_edge(u, v, key):
            # Update existing edge attributes
            for key_attr, value in data.items():
                nx_graph.edges[u, v, key][key_attr] = value
            print(f"Updated edge: {u} -> {v}, key={key}")
        else:
        """
        # Add new edge with attributes.  Important:  Check if the nodes exist.
        if not nx_graph.has_node(u):
            nx_graph.add_node(u)  # Add the node if it doesn't exist
            print(f"Added missing node {u} during edge update.")
        if not nx_graph.has_node(v):
            nx_graph.add_node(v)  # Add the node if it doesn't exist
            print(f"Added missing node {v} during edge update.")
        nx_graph.add_edge(u, v, key=key, **data)
        print(f"Added edge: {u} -> {v}, key={key}")
    """
    # 3. Handle Deletions (Nodes and Edges) - This is CRUCIAL for correct updates
    # Identify nodes to delete (nodes in original graph but not in update graph)
    nodes_to_delete = [node_id for node_id in nx_graph.nodes() if not update_graph.has_node(node_id)]
    for node_id in nodes_to_delete:
        nx_graph.remove_node(node_id)
        print(f"Deleted node: {node_id}")

    # Identify edges to delete (more complex because of keys)
    edges_to_delete = []
    for u, v, key in nx_graph.edges(keys=True):
        if not update_graph.has_edge(u, v, key):
            edges_to_delete.append((u, v, key))
    for u, v, key in edges_to_delete:
        nx_graph.remove_edge(u, v, key=key)
        print(f"Deleted edge: {u} -> {v}, key={key}")
    """
    return nx_graph
