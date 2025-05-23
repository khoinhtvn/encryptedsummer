import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool


class RunningStats:
    """Utility for tracking feature statistics"""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        logging.info("RunningStats initialized.")

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        x = x.astype(np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_n = x.shape[0]

        if self.mean is None:
            self.mean = batch_mean
            self.M2 = batch_var
            self.n = batch_n
            logging.debug(f"Initialized RunningStats with first batch: mean={self.mean}, var={self.M2}, n={self.n}")
        else:
            delta = batch_mean - self.mean
            total_n = self.n + batch_n

            new_mean = self.mean + delta * batch_n / total_n
            safe_M2 = np.clip(self.M2, a_min=1e-10, a_max=1e10)
            safe_batch_var = np.clip(batch_var, a_min=1e-10, a_max=1e10)

            new_M2 = (safe_M2 * self.n + safe_batch_var * batch_n +
                      np.square(delta) * self.n * batch_n / total_n)

            self.mean = new_mean
            self.M2 = new_M2
            self.n = total_n
            logging.debug(f"Updated RunningStats: new_mean={self.mean}, new_var={self.M2}, new_n={self.n}")

    def get_std(self):
        std = np.sqrt(self.M2 / self.n) if self.n > 0 else None
        logging.debug(f"Calculated standard deviation: {std}")
        return std


class HybridGNNAnomalyDetector(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, heads=4):
        super(HybridGNNAnomalyDetector, self).__init__()
        logging.info(
            f"Initializing HybridGNNAnomalyDetector with node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, hidden_dim={hidden_dim}, heads={heads}")
        # Graph Attention Network layers
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=heads, edge_dim=edge_feature_dim)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_feature_dim)
        logging.debug(
            f"GATConv layers initialized: conv1 out_channels={hidden_dim * heads}, conv2 out_channels={hidden_dim}")

        # Anomaly scoring heads
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        logging.debug("MLP layers for anomaly scoring initialized.")

        # Online learning components
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        logging.info("Optimizer and scheduler initialized.")

        # Experience replay buffer (optional)
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32
        logging.info(f"Replay buffer initialized with maxlen={self.replay_buffer.maxlen}, batch_size={self.batch_size}")

        # Statistical controls
        self.node_stats = RunningStats()
        self.edge_stats = RunningStats()
        self.feature_mean = None
        self.feature_std = None
        self.drift_threshold = 0.15
        self.consecutive_drifts = 0
        logging.info(f"Statistical controls initialized: drift_threshold={self.drift_threshold}")

    def forward(self, data):
        # Process graph through GAT layers
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        logging.debug(
            f"Forward pass: node_features shape={x.shape}, edge_index shape={edge_index.shape}, edge_attr shape={edge_attr.shape}")
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        logging.debug(f"GAT layers output shape: {x.shape}")

        # Node-level anomaly scores
        node_scores = self.node_mlp(x)
        logging.debug(f"Node scores shape: {node_scores.shape}")

        # Edge-level anomaly scores
        src, dst = edge_index[0], edge_index[1]
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=1)
        edge_scores = self.edge_mlp(edge_features)
        logging.debug(f"Edge features shape: {edge_features.shape}, edge scores shape: {edge_scores.shape}")

        # Global anomaly score
        global_score = self.global_mlp(
            global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device)))
        logging.debug(f"Global score: {global_score.item()}")

        return node_scores, edge_scores, global_score

    def update_online(self, data, n_steps=5):
        """Hybrid online update with statistical controls"""
        self.train()
        logging.info(f"Starting online update for {n_steps} steps.")

        if data is None:
            logging.error("`data` is None. Skipping update_online.")
            return 0.0

        # Update running statistics
        self.node_stats.update(data.x)
        self.edge_stats.update(data.edge_attr)

        # Add current data to replay buffer
        self._add_to_replay_buffer(data)

        total_loss = 0.0
        valid_steps = 0

        for step in range(n_steps):
            logging.debug(f"Online update step: {step + 1}/{n_steps}")
            if len(self.replay_buffer) >= self.batch_size:
                batch = self._sample_from_replay_buffer()
                logging.debug("Sampled batch from replay buffer.")
            else:
                batch = data  # fallback: usa data direttamente se la buffer è vuota
                logging.debug("Replay buffer too small, using current data.")

            if batch is None:
                logging.warning("No batch available. Skipping this step.")
                continue

            self.optimizer.zero_grad()
            try:
                node_scores, edge_scores, _ = self(batch)
                loss = self._compute_loss(node_scores, edge_scores)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                valid_steps += 1
                logging.debug(f"Step {step + 1}: loss={loss.item()}")
            except AttributeError as e:
                logging.error(f"Error during forward pass: {e}")
                continue

        # Update learning rate solo se almeno un passo è valido
        if valid_steps > 0:
            avg_loss = total_loss / valid_steps
            self.scheduler.step(avg_loss)
            logging.info(
                f"Online update finished. Average loss: {avg_loss}, Learning rate: {self.optimizer.param_groups[0]['lr']}")
        else:
            avg_loss = 0.0
            logging.warning("No valid steps during online update.")

        # Drift detection
        self.detect_feature_drift(data)

        return avg_loss

    def _compute_loss(self, node_scores, edge_scores):
        """Enhanced loss with statistical regularization"""
        base_loss = (node_scores.abs().mean() + edge_scores.abs().mean()) / 2
        logging.debug(f"Base loss: {base_loss.item()}")

        # Add regularization based on running statistics
        if self.node_stats.n > 10 and self.edge_stats.n > 10:
            node_std = torch.tensor(self.node_stats.get_std(), device=node_scores.device)
            edge_std = torch.tensor(self.edge_stats.get_std(), device=edge_scores.device)
            reg_loss = (node_scores.pow(2) / (node_std + 1e-6).pow(2)).mean() + \
                       (edge_scores.pow(2) / (edge_std + 1e-6).pow(2)).mean()
            loss = 0.7 * base_loss + 0.3 * reg_loss
            logging.debug(f"Regularization loss: {reg_loss.item()}, Total loss: {loss.item()}")
            return loss
        return base_loss

    def detect_feature_drift(self, data):
        """Statistical drift detection"""
        current_node_mean = data.x.mean(dim=0)
        current_edge_mean = data.edge_attr.mean(dim=0)
        logging.debug(
            f"Current node mean: {current_node_mean.cpu().numpy()}, current edge mean: {current_edge_mean.cpu().numpy()}")

        if self.feature_mean is not None:
            node_diff = (current_node_mean - self.feature_mean[0]).abs() / (self.feature_std[0] + 1e-6)
            edge_diff = (current_edge_mean - self.feature_mean[1]).abs() / (self.feature_std[1] + 1e-6)
            logging.debug(
                f"Node drift difference: {node_diff.max().item()}, Edge drift difference: {edge_diff.max().item()}")

            if node_diff.max() > self.drift_threshold or edge_diff.max() > self.drift_threshold:
                self.consecutive_drifts += 1
                logging.warning(f"Feature drift detected ({self.consecutive_drifts} consecutive)")

                if self.consecutive_drifts >= 3:
                    self._adaptive_reset()
            else:
                self.consecutive_drifts = 0

        # Update reference statistics
        self.feature_mean = (current_node_mean, current_edge_mean)
        self.feature_std = (data.x.std(dim=0), data.edge_attr.std(dim=0))
        logging.debug(
            f"Updated reference mean: node={self.feature_mean[0].cpu().numpy()}, edge={self.feature_mean[1].cpu().numpy()}")
        logging.debug(
            f"Updated reference std: node={self.feature_std[0].cpu().numpy()}, edge={self.feature_std[1].cpu().numpy()}")

    def _adaptive_reset(self):
        """Partial model reset for major behavior changes"""
        logging.warning("Major behavior change detected - performing adaptive reset of scoring heads.")

        # Reset only the anomaly scoring heads
        for name, module in self.named_modules():
            if name.endswith('mlp'):
                logging.info(f"Resetting parameters of module: {name}")
                module.apply(self._reset_weights)

        # Reset optimizer state
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        self.consecutive_drifts = 0
        logging.info("Optimizer and scheduler reset.")

    def _reset_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _add_to_replay_buffer(self, data):
        """Optional: Add subsampled graph to replay buffer"""
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)
        max_nodes = 100
        max_edges = 200

        sampled_nodes_indices = torch.randperm(num_nodes)[:min(max_nodes, num_nodes)]
        node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        node_mask[sampled_nodes_indices] = True

        edge_mask = (node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]])
        sampled_edges_indices = torch.where(edge_mask)[0][:min(max_edges, edge_mask.sum())]

        if sampled_edges_indices.numel() > 0 and sampled_nodes_indices.numel() > 0:
            subgraph_edge_index = data.edge_index[:, sampled_edges_indices]

            # Create mapping from original node indices to subgraph node indices
            unique_nodes_subgraph = torch.unique(subgraph_edge_index.flatten()).sort().values
            subgraph_node_map = {orig_idx.item(): new_idx for new_idx, orig_idx in enumerate(unique_nodes_subgraph)}

            # Remap edge indices
            remapped_edge_index = torch.tensor(
                [[subgraph_node_map[i.item()] for i in row] for row in subgraph_edge_index], dtype=torch.long)

            subgraph = Data(
                x=data.x[unique_nodes_subgraph],
                edge_index=remapped_edge_index,
                edge_attr=data.edge_attr[sampled_edges_indices] if data.edge_attr is not None else None
            )
            self.replay_buffer.append(subgraph)
            logging.debug(f"Added subsampled graph to replay buffer: {subgraph}")
        elif num_nodes > 0:
            # If no edges are left after sampling, add a node-only subgraph
            subgraph = Data(x=data.x[sampled_nodes_indices])
            self.replay_buffer.append(subgraph)
            logging.debug(f"Added node-only subgraph to replay buffer: {subgraph}")
        else:
            logging.debug("Skipped adding to replay buffer: no nodes in data.")

    def _sample_from_replay_buffer(self):
        """Sample a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = Batch.from_data_list([self.replay_buffer[i] for i in indices])
        logging.debug(f"Sampled batch from replay buffer: {batch}")
        return batch

    def detect_anomalies(self, data, threshold=2.5):
        """Detect anomalies with statistical normalization"""
        self.eval()
        logging.info("Starting anomaly detection.")
        with torch.no_grad():
            node_scores, edge_scores, global_score = self(data)

            # Convert to probabilities
            node_probs = torch.sigmoid(node_scores).squeeze()
            edge_probs = torch.sigmoid(edge_scores).squeeze()
            global_prob = torch.sigmoid(global_score).item()
            logging.debug(
                f"Node probabilities: {node_probs.cpu().numpy()}, Edge probabilities: {edge_probs.cpu().numpy()}, Global probability: {global_prob}")
            logging.debug(
                f"Node scores: {node_scores.cpu().numpy()}, Edge scores: {edge_scores.cpu().numpy()}")
            # Get statistically significant anomalies
            node_mean = node_probs.mean()
            node_std = node_probs.std()
            edge_mean = edge_probs.mean()
            edge_std = edge_probs.std()

            anomalous_nodes = (node_probs > node_mean + threshold * node_std).nonzero(as_tuple=True)[0]
            anomalous_edges = (edge_probs > edge_mean + threshold * edge_std).nonzero(as_tuple=True)[0]

            return {
                'node_anomalies': anomalous_nodes,
                'edge_anomalies': anomalous_edges,
                'global_anomaly': global_prob,
                'node_scores': node_probs,
                'edge_scores': edge_probs
            }
