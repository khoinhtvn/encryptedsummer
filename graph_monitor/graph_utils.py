import logging
import math
import re
import traceback
from collections import Counter
from datetime import datetime

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from torch_geometric.data import Data

# Define feature lists
EDGE_CATEGORICAL_FEATURES = ["protocol", "service"]
EDGE_FLOAT_FEATURES = ["dst_port", "count", "total_orig_bytes", "total_resp_bytes",
                       "total_orig_pkts", "total_resp_pkts", "total_orig_ip_bytes",
                       "total_resp_ip_bytes"]

# Global scalers and encoders
node_scaler = None
edge_scaler = None
edge_categorical_encoders = {}
last_update_time = None
update_interval = 3600


def get_sorted_edge_features(nx_graph):
    """Returns sorted edge feature keys for consistency."""
    return EDGE_FLOAT_FEATURES + EDGE_CATEGORICAL_FEATURES


def parse_dot_nodes(dot_file_content):
    """Parses node information from the .dot file content."""
    graph = nx.MultiDiGraph()
    node_pattern = re.compile(r'^\s*"([^"]+)"\s*\[(.*?)\];\s*$', re.MULTILINE)
    for match in node_pattern.finditer(dot_file_content):
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
    logging.info(f"Parsed {len(graph.nodes)} nodes.")
    return graph


def parse_dot_edges(dot_file_content, graph):
    """Parses edge information from the .dot file content and adds to the graph."""
    edge_pattern = re.compile(r'"([^"]+)" -> "([^"]+)" \[(.*?)\];')
    edge_count = 0
    for match in edge_pattern.finditer(dot_file_content):
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
    logging.info(f"Parsed {edge_count} edges.")
    return graph


def aggregate_node_data(graph):
    """
    Aggregates connection-level data into comprehensive node attributes.
    """
    for node in graph.nodes():
        # Initialize counters
        outgoing_connections = 0
        incoming_connections = 0
        protocols_seen = set()
        protocol_counts = Counter()
        services_seen = set()
        service_counts = Counter()
        remote_ports_connected_to = set()
        local_ports_used = set()
        privileged_remote_connections = 0
        privileged_local_connections = 0

        # HTTP/HTTPS specific
        http_connections = 0
        https_connections = 0
        # user_agents_seen = set()
        # http_4xx_count = 0
        # http_5xx_count = 0

        # SSL/TLS specific
        ssl_connections = 0
        # ssl_versions_seen = set()
        # ssl_ciphers_seen = set()
        # ssl_resumptions = 0
        # ssl_sni_connections = 0

        # Traffic volume aggregation
        total_outgoing_bytes = 0
        total_incoming_bytes = 0
        total_outgoing_packets = 0
        total_incoming_packets = 0

        # Temporal tracking (if available)
        connection_times = []

        # Process all edges involving this node
        for source, target, data in graph.edges(data=True):
            protocol = data.get('protocol', 'unknown')
            service = data.get('service', 'unknown')
            dst_port = int(data.get('dst_port', 0))

            if source == node:  # Outgoing connection
                outgoing_connections += 1
                protocols_seen.add(protocol)
                protocol_counts[protocol] += 1
                services_seen.add(service)
                service_counts[service] += 1
                remote_ports_connected_to.add(dst_port)

                if dst_port < 1024:
                    privileged_remote_connections += 1

                # Traffic volume
                total_outgoing_bytes += data.get('total_orig_bytes', 0)
                total_outgoing_packets += data.get('total_orig_pkts', 0)

                # Application-specific tracking
                if service.lower() in ['http', 'https'] or dst_port in [80, 443, 8080, 8443]:
                    http_connections += 1
                    if dst_port == 443 or service.lower() == 'https':
                        https_connections += 1
                        ssl_connections += 1

                # SSL tracking
                if protocol.lower() == 'ssl' or service.lower() == 'https' or dst_port == 443:
                    ssl_connections += 1

            elif target == node:  # Incoming connection
                incoming_connections += 1
                protocols_seen.add(protocol)
                protocol_counts[protocol] += 1
                services_seen.add(service)
                service_counts[service] += 1
                local_ports_used.add(dst_port)

                if dst_port < 1024:
                    privileged_local_connections += 1

                # Traffic volume
                total_incoming_bytes += data.get('total_resp_bytes', 0)
                total_incoming_packets += data.get('total_resp_pkts', 0)

                # Application-specific tracking
                if service.lower() in ['http', 'https'] or dst_port in [80, 443, 8080, 8443]:
                    http_connections += 1
                    if dst_port == 443 or service.lower() == 'https':
                        https_connections += 1
                        ssl_connections += 1

        # Store aggregated data
        total_connections = outgoing_connections + incoming_connections
        graph.nodes[node].update({
            'total_connections': total_connections,
            'outgoing_connections': outgoing_connections,
            'incoming_connections': incoming_connections,
            'protocols_seen': list(protocols_seen),
            'protocol_counts': dict(protocol_counts),
            'services_seen': list(services_seen),
            'service_counts': dict(service_counts),
            'remote_ports_connected_to': list(remote_ports_connected_to),
            'local_ports_used': list(local_ports_used),
            'privileged_remote_connections': privileged_remote_connections,
            'privileged_local_connections': privileged_local_connections,

            # HTTP/HTTPS features
            'http_connections': http_connections,
            'https_connections': https_connections,
            # 'user_agents_seen': list(user_agents_seen),
            # 'http_4xx_count': http_4xx_count,
            # 'http_5xx_count': http_5xx_count,

            # SSL/TLS features
            'ssl_connections': ssl_connections,
            # 'ssl_versions_seen': list(ssl_versions_seen),
            # 'ssl_ciphers_seen': list(ssl_ciphers_seen),
            # 'ssl_resumptions': ssl_resumptions,
            # 'ssl_sni_connections': ssl_sni_connections,

            # Traffic volume
            'total_outgoing_bytes': total_outgoing_bytes,
            'total_incoming_bytes': total_incoming_bytes,
            'total_outgoing_packets': total_outgoing_packets,
            'total_incoming_packets': total_incoming_packets,

            # Placeholder for temporal features
            #'hourly_connections': [0] * 24,
            #'activity_span_hours': 0,
        })

    return graph


def calculate_entropy(count_dict):
    """Calculate Shannon entropy for diversity measurement"""
    if not count_dict:
        return 0
    total = sum(count_dict.values())
    if total == 0:
        return 0
    return -sum((count / total) * math.log2(count / total) for count in count_dict.values() if count > 0)


def calculate_connection_consistency(hourly_counts):
    """Measure how consistent connection patterns are over time"""
    if not hourly_counts or len(hourly_counts) < 2:
        return 0
    std_dev = np.std(hourly_counts)
    return 1.0 / (1.0 + std_dev)  # Higher = more consistent


def extract_node_features_improved(node_data, time_window_stats=None):
    """Extracts improved node features with robust error handling."""
    features = []

    # 1. BEHAVIORAL ROLE FEATURES
    total_connections = node_data.get('total_connections', 0)
    outgoing_conns = node_data.get('outgoing_connections', 0)
    incoming_conns = node_data.get('incoming_connections', 0)

    # Normalize ratios properly
    outgoing_ratio = outgoing_conns / (total_connections + 1e-9)
    incoming_ratio = incoming_conns / (total_connections + 1e-9)

    # Role classification score (server vs client)
    server_score = incoming_ratio * np.log1p(incoming_conns)
    client_score = outgoing_ratio * np.log1p(outgoing_conns)

    features.extend([outgoing_ratio, incoming_ratio, server_score, client_score])

    # 2. PROTOCOL DIVERSITY FEATURES
    protocol_counts = node_data.get('protocol_counts', {})
    unique_protocols = len(protocol_counts)
    protocol_entropy = calculate_entropy(protocol_counts)

    # Most frequent protocol as ratio, not just ID
    most_freq_proto_ratio = max(
        protocol_counts.values()) / total_connections if total_connections > 0 and protocol_counts else 0

    features.extend([unique_protocols, protocol_entropy, most_freq_proto_ratio])

    # 3. PORT USAGE PATTERNS
    remote_ports = node_data.get('remote_ports_connected_to', [])
    local_ports = node_data.get('local_ports_used', [])
    unique_remote_ports = len(set(remote_ports))
    unique_local_ports = len(set(local_ports))

    # Port diversity ratios
    remote_port_diversity = unique_remote_ports / (outgoing_conns + 1e-9)
    local_port_diversity = unique_local_ports / (incoming_conns + 1e-9)

    # Privileged port usage ratios
    privileged_remote_ratio = node_data.get('privileged_remote_connections', 0) / (outgoing_conns + 1e-9)
    privileged_local_ratio = node_data.get('privileged_local_connections', 0) / (incoming_conns + 1e-9)

    features.extend([
        unique_remote_ports, unique_local_ports,
        remote_port_diversity, local_port_diversity,
        privileged_remote_ratio, privileged_local_ratio
    ])

    # 4. TEMPORAL PATTERNS
    features.extend([
        node_data.get('first_seen_hour_minute_sin', 0),
        node_data.get('first_seen_hour_minute_cos', 0),
        node_data.get('last_seen_hour_minute_sin', 0),
        node_data.get('last_seen_hour_minute_cos', 0)
    ])

    # Add activity duration and consistency
    # activity_span_hours = node_data.get('activity_span_hours', 0) # Needs 'activity_span_hours' in node_data
    # connection_consistency = calculate_connection_consistency(node_data.get('hourly_connections', [])) # Needs 'hourly_connections' and 'calculate_connection_consistency'
    # features.extend([activity_span_hours, connection_consistency])

    # 5. APPLICATION LAYER FEATURES
    http_connections = node_data.get('http_connections', 0)
    http_ratio = http_connections / (total_connections + 1e-9)

    # user_agents = node_data.get('user_agents_seen', []) # Needs 'user_agents_seen' in node_data
    # unique_user_agents = len(set(user_agents)) # Needs 'user_agents_seen' in node_data
    # user_agent_diversity = unique_user_agents / (http_connections + 1e-9) # Needs 'user_agents_seen' in node_data

    # http_error_ratio = (node_data.get('http_4xx_count', 0) + node_data.get('http_5xx_count', 0)) / ( # Needs 'http_4xx_count', 'http_5xx_count' in node_data
    #     http_connections + 1e-9) # Needs 'http_4xx_count', 'http_5xx_count' in node_data

    # features.extend([http_ratio, user_agent_diversity, http_error_ratio])
    features.extend([http_ratio])

    # SSL/TLS features
    ssl_connections = node_data.get('ssl_connections', 0)
    ssl_ratio = ssl_connections / (total_connections + 1e-9)

    # ssl_versions = node_data.get('ssl_versions_seen', []) # Needs 'ssl_versions_seen' in node_data
    # ssl_ciphers = node_data.get('ssl_ciphers_seen', []) # Needs 'ssl_ciphers_seen' in node_data
    # ssl_version_diversity = len(set(ssl_versions)) / (ssl_connections + 1e-9) if ssl_connections > 0 else 0 # Needs 'ssl_versions_seen' in node_data
    # ssl_cipher_diversity = len(set(ssl_ciphers)) / (ssl_connections + 1e-9) if ssl_connections > 0 else 0 # Needs 'ssl_ciphers_seen' in node_data

    # ssl_resumption_ratio = node_data.get('ssl_resumptions', 0) / (ssl_connections + 1e-9) # Needs 'ssl_resumptions' in node_data
    # sni_usage_ratio = node_data.get('ssl_sni_connections', 0) / (ssl_connections + 1e-9) # Needs 'ssl_sni_connections' in node_data

    # features.extend([
    #     ssl_ratio, ssl_version_diversity, ssl_cipher_diversity,
    #     ssl_resumption_ratio, sni_usage_ratio
    # ])
    features.extend([ssl_ratio])

    # 6. NETWORK TOPOLOGY FEATURES
    # degree_centrality = node_data.get('degree_centrality', 0) # Needs 'degree_centrality' in node_data
    # betweenness_centrality = node_data.get('betweenness_centrality', 0) # Needs 'betweenness_centrality' in node_data
    # clustering_coefficient = node_data.get('clustering_coefficient', 0) # Needs 'clustering_coefficient' in node_data

    # features.extend([degree_centrality, betweenness_centrality, clustering_coefficient])

    return np.array(features, dtype=float)


def get_sorted_node_features(nx_graph):
    """Returns a sorted list of all unique node attribute keys used by extract_node_features_improved."""
    return [
        'outgoing_ratio',  # Behavioral role feature: Ratio of outgoing connections to total connections.
        'incoming_ratio',  # Behavioral role feature: Ratio of incoming connections to total connections.
        'server_score',  # Behavioral role feature: Score indicating server-like behavior.
        'client_score',  # Behavioral role feature: Score indicating client-like behavior.
        'unique_protocols',  # Protocol diversity feature: Number of unique protocols used by the node.
        'protocol_entropy',  # Protocol diversity feature: Entropy of the protocol distribution.
        'most_freq_proto_ratio',
        # Protocol diversity feature: Ratio of the most frequent protocol to total connections.
        'unique_remote_ports',  # Port usage pattern feature: Number of unique remote ports the node connected to.
        'unique_local_ports',  # Port usage pattern feature: Number of unique local ports used by the node.
        'remote_port_diversity',  # Port usage pattern feature: Ratio of unique remote ports to outgoing connections.
        'local_port_diversity',  # Port usage pattern feature: Ratio of unique local ports to incoming connections.
        'privileged_remote_ratio',
        # Port usage pattern feature: Ratio of privileged remote connections to outgoing connections.
        'privileged_local_ratio',
        # Port usage pattern feature: Ratio of privileged local connections to incoming connections.
        'first_seen_hour_minute_sin',
        # Temporal pattern feature: Sine of the hour and minute when the node was first seen.
        'first_seen_hour_minute_cos',
        # Temporal pattern feature: Cosine of the hour and minute when the node was first seen.
        'last_seen_hour_minute_sin',
        # Temporal pattern feature: Sine of the hour and minute when the node was last seen.
        'last_seen_hour_minute_cos',
        # Temporal pattern feature: Cosine of the hour and minute when the node was last seen.
        # 'activity_span_hours', # Temporal pattern feature: Duration of the node's activity (needs 'activity_span_hours' in node_data).
        # 'connection_consistency', # Temporal pattern feature: Consistency of the node's connection activity over time (needs 'hourly_connections' and 'calculate_connection_consistency').
        'http_ratio',  # Application layer feature: Ratio of HTTP connections to total connections.
        # 'user_agent_diversity', # Application layer feature: Diversity of user agents seen in HTTP traffic (needs 'user_agents_seen' in node_data).
        # 'http_error_ratio', # Application layer feature: Ratio of HTTP error responses (4xx and 5xx) to total HTTP connections (needs 'http_4xx_count', 'http_5xx_count' in node_data).
        'ssl_ratio',  # Application layer feature: Ratio of SSL/TLS connections to total connections.
        # 'ssl_version_diversity', # Application layer feature: Diversity of SSL/TLS versions used (needs 'ssl_versions_seen' in node_data).
        # 'ssl_cipher_diversity', # Application layer feature: Diversity of SSL/TLS ciphers used (needs 'ssl_ciphers_seen' in node_data).
        # 'ssl_resumption_ratio', # Application layer feature: Ratio of SSL/TLS session resumptions to total SSL/TLS connections (needs 'ssl_resumptions' in node_data).
        # 'sni_usage_ratio', # Application layer feature: Ratio of SSL/TLS connections with SNI to total SSL/TLS connections (needs 'ssl_sni_connections' in node_data).
        # 'degree_centrality', # Network topology feature: Degree centrality of the node (needs 'degree_centrality' in node_data).
        # 'betweenness_centrality', # Network topology feature: Betweenness centrality of the node (needs 'betweenness_centrality' in node_data).
        # 'clustering_coefficient' # Network topology feature: Clustering coefficient of the node (needs 'clustering_coefficient' in node_data).
    ]


def extract_node_features(nx_graph, sorted_node_attr_keys):
    """Extracts node features as a PyTorch tensor using the improved method."""
    node_features = []
    for node in nx_graph.nodes():
        node_data = nx_graph.nodes[node]
        features = extract_node_features_improved(node_data)
        node_features.append(features)

    if not node_features:
        # Return empty tensor if no nodes
        return torch.empty((0, len(sorted_node_attr_keys)), dtype=torch.float)

    return torch.tensor(np.array(node_features), dtype=torch.float)


def dot_to_nx(dot_file_name):
    """Parses .dot file content and returns a NetworkX MultiDiGraph."""
    logging.info(f"Processing {dot_file_name} file.")
    graph = nx.MultiDiGraph()
    try:
        with open(dot_file_name, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"Error: .dot file not found at '{dot_file_name}'")
        return graph  # Return an empty graph or raise an exception as needed
    except Exception as e:
        logging.error(f"Error reading .dot file '{dot_file_name}': {e}")
        logging.error(traceback.format_exc())
        return graph
    graph = parse_dot_nodes(content)
    graph = parse_dot_edges(content, graph)
    graph = aggregate_node_data(graph)
    return graph


def scale_node_features(x, sorted_node_attr_keys, node_scaling, fit_scaler):
    """Scales numerical node features."""
    global node_scaler, last_update_time, update_interval
    current_time = datetime.now().timestamp()

    if node_scaling in ['standard', 'minmax']:
        if x.numel() > 0 and x.size(1) > 0:
            numerical_node_indices = list(range(x.size(1)))  # Scale all the new features
            if numerical_node_indices:
                x_numerical = x[:, numerical_node_indices].numpy()
                scaler = None
                scaler_type = None
                if node_scaling == 'standard':
                    scaler = StandardScaler()
                    scaler_type = "standard"
                elif node_scaling == 'minmax':
                    scaler = MinMaxScaler()
                    scaler_type = "min-max"

                if scaler is not None:
                    if np.std(x_numerical, axis=0).any() if scaler_type == 'standard' else (
                            np.max(x_numerical, axis=0) > np.min(x_numerical, axis=0)).any():
                        if fit_scaler or node_scaler is None or (
                                last_update_time is not None and (current_time - last_update_time) > update_interval):
                            node_scaler = scaler.fit(x_numerical)
                            x_scaled_numerical = node_scaler.transform(x_numerical)
                            log_message = f"Numerical Node Features (x) after initial {scaler_type} scaling." if fit_scaler else f"Numerical Node Features (x) after periodic {scaler_type} scaling."
                            logging.info(log_message)
                            last_update_time = current_time
                        elif node_scaler is not None:
                            x_scaled_numerical = node_scaler.transform(x_numerical)
                            logging.info(f"Numerical Node Features (x) after online {scaler_type} scaling.")
                        else:
                            logging.warning(
                                f"Skipping {scaler_type} scaling for node features: Scaler not initialized.")
                        x[:, numerical_node_indices] = torch.tensor(x_scaled_numerical, dtype=torch.float)
                    else:
                        logging.warning(
                            f"Skipping {scaler_type} scaling for node features: Zero standard deviation or all values are the same.")
            else:
                logging.info(f"No numerical node features found for {node_scaling} scaling.")
        else:
            logging.warning(f"Skipping {node_scaling} scaling for node features: No features to scale.")
    elif node_scaling == 'none':
        logging.info("No node feature scaling applied.")
    else:
        raise ValueError(f"Invalid node_scaling method: {node_scaling}. Choose 'none', 'standard', or 'minmax'.")
    logging.debug(f"Node features tensor shape after scaling: {x.shape}")
    return x


def create_edge_index(nx_graph, node_to_index):
    """Creates the edge index tensor."""
    edge_index = torch.tensor([[node_to_index[u], node_to_index[v]] for u, v in nx_graph.edges()],
                              dtype=torch.long).t().contiguous() if nx_graph.edges else torch.empty((2, 0),
                                                                                                    dtype=torch.long)
    logging.debug(f"Edge index tensor shape: {edge_index.shape}")
    return edge_index


def process_edge_features(nx_graph, sorted_edge_attr_keys, fit_scaler, current_time):
    """Extracts, calculates meaningful traffic features, and encodes edge features."""
    global edge_categorical_encoders, edge_scaler, last_update_time, update_interval
    edge_features_list = []
    for _, _, data in nx_graph.edges(data=True):
        count = data.get('count', 1.0)
        total_orig_bytes = data.get('total_orig_bytes', 0.0)
        total_resp_bytes = data.get('total_resp_bytes', 0.0)
        total_orig_pkts = data.get('total_orig_pkts', 0.0)
        total_resp_pkts = data.get('total_resp_pkts', 0.0)

        feature_vector = []

        # 1. Meaningful averages (size characteristics)
        avg_orig_bytes_per_conn = total_orig_bytes / count if count > 0 else 0.0
        avg_resp_bytes_per_conn = total_resp_bytes / count if count > 0 else 0.0
        avg_orig_pkts_per_conn = total_orig_pkts / count if count > 0 else 0.0
        avg_resp_pkts_per_conn = total_resp_pkts / count if count > 0 else 0.0

        # 2. Traffic direction ratios (communication patterns)
        total_bytes = total_orig_bytes + total_resp_bytes
        total_pkts = total_orig_pkts + total_resp_pkts

        orig_bytes_ratio = total_orig_bytes / (total_bytes + 1e-9)
        orig_pkts_ratio = total_orig_pkts / (total_pkts + 1e-9)

        # 3. Packet size characteristics
        avg_orig_packet_size = total_orig_bytes / (total_orig_pkts + 1e-9)
        avg_resp_packet_size = total_resp_bytes / (total_resp_pkts + 1e-9)

        # 4. Communication efficiency
        bytes_per_packet_ratio = avg_orig_packet_size / (avg_resp_packet_size + 1e-9)

        # 5. Volume features (log-scaled to handle wide ranges)
        log_total_bytes = np.log1p(total_bytes)  # log(1 + x) handles zeros
        log_total_pkts = np.log1p(total_pkts)
        log_count = np.log1p(count)

        feature_vector.extend([
            avg_orig_bytes_per_conn, avg_resp_bytes_per_conn,
            avg_orig_pkts_per_conn, avg_resp_pkts_per_conn,
            orig_bytes_ratio, orig_pkts_ratio,
            avg_orig_packet_size, avg_resp_packet_size,
            bytes_per_packet_ratio,
            log_total_bytes, log_total_pkts, log_count
        ])

        # Add other original edge features (excluding those already used)
        for key in sorted_edge_attr_keys:
            if key not in ['count', 'total_orig_bytes', 'total_resp_bytes', 'total_orig_pkts', 'total_resp_pkts']:
                feature_vector.append(data.get(key))

        edge_features_list.append(feature_vector)

    if edge_features_list:
        edge_attr_raw = np.array(edge_features_list, dtype=object)
        encoded_categorical = []
        numerical_edge_features = []
        processed_edge_feature_names = [
            'avg_orig_bytes_per_conn', 'avg_resp_bytes_per_conn',
            'avg_orig_pkts_per_conn', 'avg_resp_pkts_per_conn',
            'orig_bytes_ratio', 'orig_pkts_ratio',
            'avg_orig_packet_size', 'avg_resp_packet_size',
            'bytes_per_packet_ratio',
            'log_total_bytes', 'log_total_pkts', 'log_count'
        ]

        for i, feature_name in enumerate(processed_edge_feature_names):
            feature_values = edge_attr_raw[:, i]
            numerical_edge_features.append(
                np.array([float(val) if isinstance(val, (int, float)) else 0.0 for val in feature_values]).reshape(-1,
                                                                                                                   1))

        original_categorical_indices = [i for i, key in enumerate(sorted_edge_attr_keys) if
                                        key in EDGE_CATEGORICAL_FEATURES]
        for original_index in original_categorical_indices:
            feature_name = sorted_edge_attr_keys[original_index]
            categorical_data_index = len(processed_edge_feature_names) + original_index - sum(
                1 for k in sorted_edge_attr_keys[:original_index] if
                k in ['count', 'total_orig_bytes', 'total_resp_bytes', 'total_orig_pkts', 'total_resp_pkts'])

            if categorical_data_index < edge_attr_raw.shape[1]:
                feature_values = edge_attr_raw[:, categorical_data_index]
                encoder = edge_categorical_encoders.get(feature_name)
                if encoder is None or fit_scaler or (
                        last_update_time is not None and (current_time - last_update_time) > update_interval):
                    encoder = LabelEncoder()
                    encoder.fit(feature_values.astype(str))
                    edge_categorical_encoders[feature_name] = encoder
                    encoded = encoder.transform(feature_values.astype(str)).reshape(-1, 1)
                    logging.info(
                        f"Fitted and transformed categorical edge feature: {feature_name}. Unique values: {len(encoder.classes_)}")
                    if fit_scaler or last_update_time is None or (current_time - last_update_time) > update_interval:
                        last_update_time = current_time
                else:
                    encoded = encoder.transform(feature_values.astype(str)).reshape(-1, 1)
                    logging.info(f"Transformed categorical edge feature: {feature_name}.")
                encoded_categorical.append(torch.tensor(encoded, dtype=torch.float))

        original_numerical_indices = [i for i, key in enumerate(sorted_edge_attr_keys) if key in EDGE_FLOAT_FEATURES]
        for original_index in original_numerical_indices:
            feature_name = sorted_edge_attr_keys[original_index]
            numerical_data_index = len(processed_edge_feature_names) + original_index - sum(
                1 for k in sorted_edge_attr_keys[:original_index] if
                k in ['count', 'total_orig_bytes', 'total_resp_bytes', 'total_orig_pkts', 'total_resp_pkts'])
            if numerical_data_index < edge_attr_raw.shape[1]:
                feature_values = edge_attr_raw[:, numerical_data_index]
                numerical_values = [float(val) if isinstance(val, (int, float)) else 0.0 for val in feature_values]
                numerical_edge_features.append(np.array(numerical_values).reshape(-1, 1))

        edge_attr_categorical = torch.cat(encoded_categorical, dim=1) if encoded_categorical else None
        logging.debug(
            f"Shape of encoded categorical edge features: {edge_attr_categorical.shape if edge_attr_categorical is not None else (0,)}")

        edge_attr_numerical_concat = np.concatenate(numerical_edge_features,
                                                    axis=1) if numerical_edge_features else None
        return edge_attr_categorical, edge_attr_numerical_concat
    else:
        return None, None


def scale_edge_features(edge_attr_numerical_concat, edge_scaling, fit_scaler):
    """Scales numerical edge features."""
    global edge_scaler, last_update_time, update_interval
    current_time = datetime.now().timestamp()
    edge_attr_numerical_scaled = None
    if edge_attr_numerical_concat is not None:
        if edge_scaling in ['standard', 'minmax']:
            scaler = None
            scaler_type = None
            if edge_scaling == 'standard':
                scaler = StandardScaler()
                scaler_type = "standard"
            elif edge_scaling == 'minmax':
                scaler = MinMaxScaler()
                scaler_type = "min-max"

            if scaler is not None:
                if np.std(edge_attr_numerical_concat, axis=0).any() if scaler_type == 'standard' else (
                        np.max(edge_attr_numerical_concat, axis=0) > np.min(edge_attr_numerical_concat,
                                                                            axis=0)).any():
                    if fit_scaler or edge_scaler is None or (last_update_time is not None and (
                            current_time - last_update_time) > update_interval):
                        edge_scaler = scaler.fit(edge_attr_numerical_concat)
                        edge_attr_numerical_scaled = edge_scaler.transform(edge_attr_numerical_concat)
                        log_message = f"Numerical Edge Attributes after initial {scaler_type} scaling." if fit_scaler else f"Numerical Edge Attributes after periodic {scaler_type} scaling."
                        logging.info(log_message)
                        last_update_time = current_time
                    elif edge_scaler is not None:
                        edge_attr_numerical_scaled = edge_scaler.transform(edge_attr_numerical_concat)
                        logging.info(f"Numerical Edge Attributes after online {scaler_type} scaling.")
                    else:
                        logging.warning(
                            f"Skipping {scaler_type} scaling for numerical edge features: Scaler not initialized.")
                else:
                    logging.warning(
                        f"Skipping {scaler_type} scaling for numerical edge features: Zero standard deviation or all values are the same.")
            edge_attr_numerical_scaled = torch.tensor(edge_attr_numerical_scaled,
                                                      dtype=torch.float) if edge_attr_numerical_scaled is not None else None
            logging.debug(
                f"Shape of scaled numerical edge features: {edge_attr_numerical_scaled.shape if edge_attr_numerical_scaled is not None else (0,)}")
        elif edge_scaling == 'none':
            edge_attr_numerical_scaled = torch.tensor(edge_attr_numerical_concat,
                                                      dtype=torch.float) if edge_attr_numerical_concat is not None else None
            logging.info("No scaling applied to numerical edge features.")
        else:
            raise ValueError(
                f"Invalid edge_scaling method: {edge_scaling}. Choose 'none', 'standard', or 'minmax'.")
    return edge_attr_numerical_scaled


def nx_to_pyg(nx_graph, node_scaling='none', edge_scaling='none', fit_scaler=True):
    """Converts a NetworkX graph to a PyG Data object."""
    global node_scaler, edge_scaler, edge_categorical_encoders, last_update_time, update_interval

    current_time = datetime.now().timestamp()
    logging.info("Starting NetworkX to PyG conversion.")

    # 1. Create node index mapping and extract node features
    sorted_node_attr_keys = get_sorted_node_features(nx_graph)
    logging.debug(f"Sorted node attribute keys: {sorted_node_attr_keys}")
    node_to_index = {node: i for i, node in enumerate(nx_graph.nodes())}
    x = extract_node_features(nx_graph, sorted_node_attr_keys)
    logging.debug(f"Initial node features tensor shape: {x.shape}")

    # 2. Apply Node Feature Scaling
    x = scale_node_features(x, sorted_node_attr_keys, node_scaling, fit_scaler)

    # 3. Create edge indices
    edge_index = create_edge_index(nx_graph, node_to_index)

    # 4. Extract and process edge features
    sorted_edge_attr_keys = get_sorted_edge_features(nx_graph)
    logging.debug(f"Sorted edge attribute keys: {sorted_edge_attr_keys}")
    edge_attr_categorical, edge_attr_numerical_concat = process_edge_features(nx_graph, sorted_edge_attr_keys,
                                                                              fit_scaler, current_time)

    # 5. Scale numerical edge features
    edge_attr_numerical_scaled = scale_edge_features(edge_attr_numerical_concat, edge_scaling, fit_scaler)

    # 6. Concatenate categorical and numerical edge features
    edge_attr = None
    if edge_attr_categorical is not None and edge_attr_numerical_scaled is not None:
        edge_attr = torch.cat([edge_attr_categorical, edge_attr_numerical_scaled], dim=1)
    elif edge_attr_categorical is not None:
        edge_attr = edge_attr_categorical
    elif edge_attr_numerical_scaled is not None:
        edge_attr = edge_attr_numerical_scaled

    if edge_attr is not None:
        logging.debug(f"Final edge attributes tensor shape: {edge_attr.shape}")
    else:
        logging.info("No edge attributes (numerical or categorical) found.")

    # 7. Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    logging.info("Successfully converted NetworkX graph to PyG Data object.")
    return data
