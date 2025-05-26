import logging
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
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


class GraphAutoencoder(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, embedding_dim=32):
        super(GraphAutoencoder, self).__init__()
        logging.info(
            f"Initializing GraphAutoencoder with node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, hidden_dim={hidden_dim}, embedding_dim={embedding_dim}")

        # Encoder GAT layers
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=4, edge_dim=edge_feature_dim)
        self.conv2 = GATConv(hidden_dim * 4, embedding_dim, heads=1, edge_dim=edge_feature_dim)
        logging.debug(
            f"Encoder GATConv layers initialized: conv1 out_channels={hidden_dim * 4}, conv2 out_channels={embedding_dim}")

        # Decoder MLPs for node and edge reconstruction (using embeddings)
        self.node_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a score for edge existence/anomaly
        )
        logging.debug("Decoder MLP layers initialized.")

    def encode(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        embedding = F.relu(self.conv2(x, edge_index, edge_attr))
        return embedding

    def decode_node(self, z):
        return self.node_decoder(z)

    def decode_edge(self, z, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        edge_features = torch.cat([z[src], z[dst], edge_attr], dim=1)
        return torch.sigmoid(self.edge_decoder(edge_features)).squeeze()  # Sigmoid for probability

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        embedding = self.encode(x, edge_index, edge_attr)
        node_recon = self.decode_node(embedding)
        edge_recon = self.decode_edge(embedding, edge_index, edge_attr)
        return embedding, node_recon, edge_recon


class HybridGNNAnomalyDetector(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, embedding_dim=32, heads=4, export_period=5,
                 export_dir=None):
        super(HybridGNNAnomalyDetector, self).__init__()
        logging.info(
            f"Initializing HybridGNNAnomalyDetector with node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, hidden_dim={hidden_dim}, embedding_dim={embedding_dim}, heads={heads}")

        self.autoencoder = GraphAutoencoder(node_feature_dim, edge_feature_dim, hidden_dim, embedding_dim)

        # Anomaly scoring MLPs on the embeddings
        self.node_anomaly_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.edge_anomaly_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + edge_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.global_anomaly_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        logging.debug("Anomaly scoring MLPs initialized.")

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

        # Embedding export parameters
        self.export_period = export_period
        self.update_count = 0
        self.export_dir = export_dir  # Initialize export_dir here
        if export_dir is not None:
            os.makedirs(self.export_dir, exist_ok=True)
            logging.info(f"Embedding export will occur every {self.export_period} updates, saving to {self.export_dir}")

    def forward(self, data):
        embedding, node_recon, edge_recon = self.autoencoder(data)

        # Node-level anomaly scores based on embeddings
        node_scores = self.node_anomaly_mlp(embedding)
        logging.debug(f"Node anomaly scores shape: {node_scores.shape}")

        # Edge-level anomaly scores based on embeddings and edge features
        src, dst = data.edge_index[0], data.edge_index[1]
        edge_features = torch.cat([embedding[src], embedding[dst], data.edge_attr], dim=1)
        edge_scores = self.edge_anomaly_mlp(edge_features)
        logging.debug(f"Edge anomaly scores shape: {edge_scores.shape}")

        # Global anomaly score based on global mean pooled embedding
        global_embedding = global_mean_pool(embedding, batch=torch.zeros(embedding.size(0), dtype=torch.long,
                                                                         device=embedding.device))
        global_score = self.global_anomaly_mlp(global_embedding)
        logging.debug(f"Global anomaly score shape: {global_score.shape}")

        return node_scores, edge_scores, global_score, node_recon, edge_recon, embedding, global_embedding

    def update_online(self, data, n_steps=15, recon_weight=0.8, anomaly_weight=0.2):
        """Hybrid online update with reconstruction and anomaly scoring loss."""
        self.train()
        logging.info(f"Starting online update for {n_steps} steps.")

        if data is None:
            logging.error("`data` is None. Skipping update_online.")
            return 0.0

        # Update running statistics
        if data.x is not None and data.x.numel() > 0:
            self._update_running_stats(self.node_stats, data.x)
        if data.edge_attr is not None and data.edge_attr.numel() > 0:
            self._update_running_stats(self.edge_stats, data.edge_attr)

        # Add current data to replay buffer
        self._add_to_replay_buffer(data)

        total_loss = 0.0
        successful_steps = 0

        for step in range(n_steps):
            logging.debug(f"Online update step: {step + 1}/{n_steps}")
            batch = self._get_training_batch(data)
            if batch is None:
                logging.warning("No training batch available. Skipping this step.")
                continue

            self.optimizer.zero_grad()
            loss = self._calculate_online_loss(batch, recon_weight, anomaly_weight)

            if loss is not None:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                successful_steps += 1
                logging.debug(f"Step {step + 1}: Loss={loss.item():.4f}")

        avg_loss = total_loss / successful_steps if successful_steps > 0 else 0.0
        if successful_steps > 0:
            self.scheduler.step(avg_loss)
            logging.info(
                f"Online update finished. Average loss: {avg_loss:.4f}, Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        else:
            logging.warning("No successful steps during online update.")

        # Drift detection
        self.detect_feature_drift(data)

        return avg_loss

    def _update_running_stats(self, running_stats, current_data, alpha=0.1):
        """Updates the running mean and std of the features."""
        if current_data is not None and current_data.numel() > 0:
            mean = current_data.mean(dim=0)
            std = current_data.std(dim=0)
            running_stats.update(mean.detach().cpu().numpy())
            running_stats.update(std.detach().cpu().numpy())

    def _get_training_batch(self, current_data):
        """Retrieves a training batch from the replay buffer or uses current data."""
        if len(self.replay_buffer) >= self.batch_size:
            batch = self._sample_from_replay_buffer()
            logging.debug(f"Sampled batch from replay buffer (size: {batch.num_graphs} graphs).")
            return batch
        elif current_data is not None:
            logging.debug(f"Replay buffer too small (size: {len(self.replay_buffer)}), using current data.")
            return current_data
        else:
            logging.warning("No data available for training batch.")
            return None

    def _calculate_online_loss(self, batch, recon_weight, anomaly_weight):
        """Calculates the combined reconstruction and anomaly loss."""
        try:
            node_scores, edge_scores, _, node_recon, edge_recon, _, _ = self(batch)

            recon_loss_node = F.mse_loss(node_recon,
                                         batch.x) if batch.x is not None and batch.x.numel() > 0 else torch.tensor(0.0,
                                                                                                                   device=self.device)
            edge_recon_target = torch.ones_like(
                edge_recon) * 0.5 if edge_recon is not None and edge_recon.numel() > 0 else torch.tensor(0.0,
                                                                                                         device=self.device)
            recon_loss_edge = F.binary_cross_entropy_with_logits(edge_recon,
                                                                 edge_recon_target) if edge_recon is not None and edge_recon.numel() > 0 else torch.tensor(
                0.0, device=self.device)

            anomaly_loss_node = node_scores.abs().mean() if node_scores is not None and node_scores.numel() > 0 else torch.tensor(
                0.0, device=self.device)
            anomaly_loss_edge = edge_scores.abs().mean() if edge_scores is not None and edge_scores.numel() > 0 else torch.tensor(
                0.0, device=self.device)
            anomaly_loss = (anomaly_loss_node + anomaly_loss_edge) / 2

            loss = recon_weight * (recon_loss_node + recon_loss_edge) + anomaly_weight * anomaly_loss
            return loss
        except AttributeError as e:
            logging.error(f"Error during forward pass: {e}")
            return None

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
                f"Node drift difference: {node_diff.max().item():.4f}, Edge drift difference: {edge_diff.max().item():.4f}")

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
        logging.warning("Major behavior change detected - performing adaptive reset of scoring heads and decoder.")

        # Reset only the anomaly scoring heads and the autoencoder's decoder
        for name, module in self.named_modules():
            if name.endswith('mlp') or 'decoder' in name:
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
        """Detect anomalies with statistical normalization on reconstruction error"""
        self.eval()
        logging.info("Starting anomaly detection.")
        with torch.no_grad():
            embedding, node_recon, edge_recon = self.autoencoder(data)

            # Reconstruction errors
            node_recon_error = F.mse_loss(node_recon, data.x, reduction='none').mean(dim=1)
            if data.edge_attr is not None and data.edge_attr.numel() > 0:
                edge_recon_target_mean = data.edge_attr.mean(dim=1, keepdim=True)
                edge_recon_error = torch.abs(edge_recon - edge_recon_target_mean.squeeze())
            else:
                edge_recon_error = torch.abs(edge_recon - 0.5)

            logging.debug(f"Node reconstruction errors: {node_recon_error.cpu().numpy()}")
            logging.debug(f"Edge reconstruction errors: {edge_recon_error.cpu().numpy()}")

            # Statistical normalization of reconstruction errors
            node_mean_recon = node_recon_error.mean()
            node_std_recon = node_recon_error.std()
            edge_mean_recon = edge_recon_error.mean()
            edge_std_recon = edge_recon_error.std()

            anomalous_nodes = (node_recon_error > node_mean_recon + threshold * node_std_recon).nonzero(as_tuple=True)[
                0]
            anomalous_edges = (edge_recon_error > edge_mean_recon + threshold * edge_std_recon).nonzero(as_tuple=True)[
                0]

            # Also get anomaly scores from the dedicated MLPs (you might want to combine these)
            node_scores_mlp = torch.sigmoid(self.node_anomaly_mlp(embedding)).squeeze()
            edge_scores_mlp = torch.sigmoid(self.edge_anomaly_mlp(
                torch.cat([embedding[data.edge_index[0]], embedding[data.edge_index[1]], data.edge_attr],
                          dim=1))).squeeze()
            global_score_mlp = torch.sigmoid(self.global_anomaly_mlp(global_mean_pool(embedding, batch=torch.zeros(
                embedding.size(0), dtype=torch.long, device=embedding.device)))).item()

            return {
                'node_anomalies_recon': anomalous_nodes,
                'edge_anomalies_recon': anomalous_edges,
                'node_recon_errors': node_recon_error,
                'edge_recon_errors': edge_recon_error,
                'node_scores_mlp': node_scores_mlp,
                'edge_scores_mlp': edge_scores_mlp,
                'global_anomaly_mlp': global_score_mlp,
                'embedding': embedding.cpu().numpy()
            }

    def export_embeddings(self, data, filename="embeddings.png", n_components=2, perplexity=30, n_iter=300):
        self.eval()
        with torch.no_grad():
            embedding, _, _ = self.autoencoder(data)
            embedding_np = embedding.cpu().numpy()

            n_samples = embedding_np.shape[0]
            safe_perplexity = min(perplexity, max(5, n_samples - 1))  # Ensure it's within bounds

            if safe_perplexity < 1:
                logging.error(
                    f"Error during embedding export: n_samples ({n_samples}) is too small for meaningful t-SNE.")
                return

            tsne = TSNE(n_components=n_components, random_state=42, perplexity=safe_perplexity, n_iter=n_iter)
            reduced_embeddings = tsne.fit_transform(embedding_np)

            plt.figure(figsize=(8, 8))
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
            plt.title(f"Node Embeddings Visualization (t-SNE, Perplexity={safe_perplexity})")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.savefig(filename)
            plt.close()
            logging.info(f"Embeddings visualization saved to {filename}")
