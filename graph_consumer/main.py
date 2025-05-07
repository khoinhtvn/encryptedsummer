import re
import networkx as nx
import argparse
import traceback
import matplotlib.pyplot as plt
import os
import torch
from torch_geometric.data import Data
from datetime import datetime  # For timestamp conversion

def dot_to_nx(dot_file):
    """
    Parses a .dot file representing a network graph and returns a NetworkX MultiDiGraph.
    Handles node and edge attributes, including the specific Zeek log attributes.

    Args:
        dot_file (str): Path to the .dot file.

    Returns:
        networkx.MultiDiGraph: A NetworkX MultiDiGraph object.
    """
    graph = nx.MultiDiGraph()

    with open(dot_file, 'r') as f:
        content = f.read()

    # Extract node information
    # Corrected node pattern:  Match only lines starting with a node ID and attributes within brackets
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
            If None, displays the plot. Defaults to None.
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
        # print(f"Edge from {u} to {v} with key {k} and data {data}")
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
                edge_color = "gray"  # For more than 3 parallel edges
                style = "solid"
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], edge_color=edge_color, style=style, arrows=True)
        else:
            nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], edge_color="black", style="solid", arrows=True)

    # Add edge labels (show attributes)
    edge_labels = {(u, v): str(d) for u, v, d in nx_graph.edges(data=True)}  # Get all edge attributes
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=6)

    plt.title(f"NetworkX Graph Visualization ({layout_algorithm} Layout)")
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off

    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)  # Save the plot
        print(f"Graph saved to {output_path}")
    else:
        plt.show()  # Display the plot

    plt.close()  # Close the figure to prevent it from consuming memory.


def save_nx_graph(nx_graph, output_path):
    """
    Saves a NetworkX graph to a file.  Uses a format that preserves multiple edges
    and their attributes.  Pickle is a simple option, but other formats like GraphML
    can also be used.

    Args:
        nx_graph (networkx.MultiDiGraph): The NetworkX graph to save.
        output_path (str): Path to save the graph.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nx.write_gpickle(nx_graph, output_path)  # Use pickle for simplicity
        print(f"Graph saved to {output_path} in Pickle format")
    except Exception as e:
        print(f"Error saving graph: {e}")
        traceback.print_exc()



def nx_to_pyg(nx_graph):
    """
    Converts a NetworkX MultiDiGraph to a PyTorch Geometric Data object,
    extracting the specified Zeek log features from edge attributes.  Ensures
    consistent node feature extraction.

    Args:
        nx_graph (networkx.MultiDiGraph): The input NetworkX graph.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object.
    """
    # 1. Determine the complete set of node attributes.
    all_node_attr_keys = set()
    for node in nx_graph.nodes():
        #  Use .nodes[node].keys() to get only node attributes
        all_node_attr_keys.update(nx_graph.nodes[node].keys())
    sorted_node_attr_keys = sorted(list(all_node_attr_keys))

    # 2.  Create node features, ensuring consistent order and handling missing attributes.
    node_features = []
    for node in nx_graph.nodes():
        node_feature_list = []
        for key in sorted_node_attr_keys:
            value = nx_graph.nodes[node].get(key)  # Use .get() to handle missing attributes
            if isinstance(value, (int, float)):
                node_feature_list.append(value)
            else:
                node_feature_list.append(0.0)  # Default value for non-numeric or missing
        node_features.append(node_feature_list)
    x = torch.tensor(node_features, dtype=torch.float)

    # Edge indices
    edge_index = torch.tensor(
        [[list(nx_graph.nodes).index(u), list(nx_graph.nodes).index(v)] for u, v in nx_graph.edges()],
        dtype=torch.long
    ).t().contiguous()

    # Edge attributes
    edge_features = []
    for u, v, data in nx_graph.edges(data=True):
        # Initialize all features to 0.0
        src_port = 0
        dst_port = 0
        request_body_len = 0
        response_body_len = 0
        status_code = 0
        timestamp = 0.0
        hour_of_day = 0
        day_of_week = 0
        is_http_1_1 = 0
        is_http_1_0 = 0
        method_GET = 0
        method_POST = 0
        method_PUT = 0
        is_status_4xx = 0
        is_status_5xx = 0
        is_status_2xx = 0
        is_status_3xx = 0
        is_firefox = 0
        is_chrome = 0
        browser_version = 0.0
        uri_length = 0
        has_query_params = 0
        num_path_segments = 0
        mime_text_html = 0
        mime_application_json = 0

        # Extract available features from edge attributes
        if 'src_port' in data:
            src_port = int(data['src_port'])
        if 'dst_port' in data:
            dst_port = int(data['dst_port'])
        if 'request_body_len' in data:
            request_body_len = int(data['request_body_len'])
        if 'response_body_len' in data:
            response_body_len = int(data['response_body_len'])
        if 'status_code' in data:
            status_code = int(data['status_code'])
        if 'timestamp' in data:
            timestamp = float(data['timestamp'])
            dt = datetime.fromtimestamp(timestamp)
            hour_of_day = dt.hour
            day_of_week = dt.weekday()  # 0 for Monday, 6 for Sunday
        if 'http_version' in data:  #  'http_version' instead of 'is_http_1.1' and 'is_http_1.0'
            if data['http_version'] == '1.1':
                is_http_1_1 = 1
                is_http_1_0 = 0
            elif data['http_version'] == '1.0':
                is_http_1_1 = 0
                is_http_1_0 = 1
            else:
                is_http_1_1 = 0
                is_http_1_0 = 0

        if 'method' in data:
            if data['method'] == 'GET':
                method_GET = 1
            elif data['method'] == 'POST':
                method_POST = 1
            elif data['method'] == 'PUT':
                method_PUT = 1
            # Add other methods as needed

        if 'status_code' in data:
            if 400 <= status_code < 500:
                is_status_4xx = 1
            elif 500 <= status_code < 600:
                is_status_5xx = 1
            elif 200 <= status_code < 300:
                is_status_2xx = 1
            elif 300 <= status_code < 400:
                is_status_3xx = 1

        if 'user_agent' in data:
            user_agent = data['user_agent'].lower()
            if 'firefox' in user_agent:
                is_firefox = 1
                # Extract browser version (simplified)
                version_match = re.search(r'firefox/(\d+\.\d+)', user_agent)
                if version_match:
                    browser_version = float(version_match.group(1))
            elif 'chrome' in user_agent:
                is_chrome = 1
            elif 'safari' in user_agent:
                is_safari = 1
            if 'mobile' in user_agent:
                is_mobile = 1

        if 'uri' in data:
            uri_length = len(data['uri'])
            has_query_params = 1 if '?' in data['uri'] else 0
            num_path_segments = data['uri'].count('/') - 1  # Count slashes, adjust as needed

        if 'mime_type' in data:
            if data['mime_type'] == 'text/html':
                mime_text_html = 1
            elif data['mime_type'] == 'application/json':
                mime_application_json = 1

        edge_features.append([
            src_port, dst_port, request_body_len, response_body_len, status_code,
            timestamp, hour_of_day, day_of_week, is_http_1_1, is_http_1_0,
            method_GET, method_POST, method_PUT, is_status_4xx, is_status_5xx,
            is_status_2xx, is_status_3xx, is_firefox, is_chrome, browser_version,
            uri_length, has_query_params, num_path_segments, mime_text_html, mime_application_json
        ])
    edge_attr = torch.tensor(edge_features, dtype=torch.float) if edge_features else None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert a .dot file to a NetworkX graph, visualize it, and save it, then convert to PyG.'
    )
    parser.add_argument('dot_file_path', type=str, help='Path to the input .dot file.')
    parser.add_argument(
        '--layout', type=str, default='spring',
        choices=['spring', 'circular', 'spectral', 'kamada_kawai', 'random'],
        help='Layout algorithm to use for visualization.'
    )
    parser.add_argument(
        '--output_plot', type=str, default=None,
        help='Path to save the plot (e.g., "graph.png", "graph.pdf"). If not provided, the plot is displayed.'
    )
    parser.add_argument(
        '--output_graph', type=str, default=None,
        help='Path to save the NetworkX graph (e.g., "graph.pkl"). If not provided, the graph is not saved.'
    )
    parser.add_argument(
        '--to_pyg', action='store_true',
        help='Convert the NetworkX graph to a PyTorch Geometric Data object and print it.'
    )
    args = parser.parse_args()

    dot_file_path = args.dot_file_path
    layout_algorithm = args.layout
    output_plot_path = args.output_plot
    output_graph_path = args.output_graph
    to_pyg = args.to_pyg

    try:
        nx_graph = dot_to_nx(dot_file_path)

        # Visualize the graph
        visualize_nx_graph(nx_graph, layout_algorithm, output_plot_path)

        # Save the graph, if an output path is provided
        if output_graph_path:
            save_nx_graph(nx_graph, output_graph_path)

        # Convert to PyG
        if to_pyg:
            data = nx_to_pyg(nx_graph)
            print("\nPyTorch Geometric Data Object:")
            print(data)

            # Add this code to explore the Data object:
            print("\n--- Node Features (x) ---")
            print(data.x)
            print(f"Shape of node features: {data.x.shape}")

            print("\n--- Edge Indices (edge_index) ---")
            print(data.edge_index)
            print(f"Shape of edge indices: {data.edge_index.shape}")

            print("\n--- Edge Attributes (edge_attr) ---")
            if data.edge_attr is not None:
                print(data.edge_attr)
                print(f"Shape of edge attributes: {data.edge_attr.shape}")
            else:
                print("No edge attributes.")

            print("\n--- Graph Structure ---")
            print(data)


    except FileNotFoundError:
        print(f"Error: The file '{dot_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred:")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("Traceback:")
        traceback.print_exc()
