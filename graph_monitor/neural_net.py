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
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm


class RunningStats:
    """Utility for tracking feature statistics."""

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
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, embedding_dim=32, num_gat_layers=2,
                 gat_heads=4, use_batch_norm=True, use_residual=True):
        super().__init__()
        logging.info(
            f"Initializing GraphAutoencoder with node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, "
            f"hidden_dim={hidden_dim}, embedding_dim={embedding_dim}, num_gat_layers={num_gat_layers}, "
            f"gat_heads={gat_heads}, use_batch_norm={use_batch_norm}, use_residual={use_residual}")

        self.num_gat_layers = num_gat_layers
        self.gat_heads = gat_heads
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(0.2)  # Increased dropout for better regularization

        # Encoder GAT layers with better initialization
        current_dim = node_feature_dim
        for i in range(num_gat_layers):
            if i == num_gat_layers - 1:  # Last layer
                out_dim = embedding_dim
                heads = 1
            else:
                out_dim = hidden_dim
                heads = gat_heads

            self.gat_layers.append(GATConv(current_dim, out_dim, heads=heads, edge_dim=edge_feature_dim,
                                           dropout=0.1, add_self_loops=True, bias=True))

            if use_batch_norm and i < num_gat_layers - 1:
                self.batch_norms.append(BatchNorm(out_dim * heads))

            current_dim = out_dim * heads

        logging.debug(f"Encoder GATConv layers initialized with final embedding dimension: {embedding_dim}")

        # Improved decoder MLPs with skip connections
        self.node_decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )

        self.edge_decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2 + edge_feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights properly
        self.apply(self._init_weights)
        logging.debug("Decoder MLP layers initialized with proper weight initialization.")

    def _init_weights(self, module):
        """Proper weight initialization for faster convergence"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def encode(self, x, edge_index, edge_attr):
        residual = None

        for i, conv in enumerate(self.gat_layers):
            x_prev = x
            x = conv(x, edge_index, edge_attr)

            if i < self.num_gat_layers - 1:
                # Apply batch normalization
                if self.use_batch_norm and self.batch_norms is not None:
                    x = self.batch_norms[i](x)

                # Apply activation
                x = F.elu(x)  # ELU instead of ReLU for better gradients

                # Apply residual connection if dimensions match
                if self.use_residual and x.shape == x_prev.shape:
                    x = x + x_prev

                # Apply dropout
                x = self.dropout(x)

        return x

    def decode_node(self, z):
        return self.node_decoder(z)

    def decode_edge(self, z, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        edge_features = torch.cat([z[src], z[dst], edge_attr], dim=1)
        return torch.sigmoid(self.edge_decoder(edge_features)).squeeze()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        embedding = self.encode(x, edge_index, edge_attr)
        node_recon = self.decode_node(embedding)
        edge_recon = self.decode_edge(embedding, edge_index, edge_attr)
        return embedding, node_recon, edge_recon


class HybridGNNAnomalyDetector(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, embedding_dim=32, heads=4, export_period=5,
                 export_dir: str = None, num_gat_layers=2, gat_heads=4,
                 recon_loss_type='mse', edge_recon_loss_type='bce', use_batch_norm=True, use_residual=True):
        super().__init__()
        logging.info(
            f"Initializing HybridGNNAnomalyDetector with node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, "
            f"hidden_dim={hidden_dim}, embedding_dim={embedding_dim}, heads={heads}, num_gat_layers={num_gat_layers}, "
            f"gat_heads={gat_heads}, recon_loss_type={recon_loss_type}, edge_recon_loss_type={edge_recon_loss_type}")

        self.autoencoder = GraphAutoencoder(node_feature_dim, edge_feature_dim, hidden_dim, embedding_dim,
                                            num_gat_layers, gat_heads, use_batch_norm, use_residual)
        self.recon_loss_type = recon_loss_type
        self.edge_recon_loss_type = edge_recon_loss_type

        # Improved anomaly scoring MLPs
        self.node_anomaly_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.edge_anomaly_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.global_anomaly_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Apply proper initialization
        self.apply(self._init_weights)
        logging.debug("Anomaly scoring MLPs initialized.")

        # Improved optimizer with better hyperparameters
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.003, weight_decay=1e-4,
                                           betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        logging.info("Optimizer (AdamW) and scheduler (CosineAnnealingWarmRestarts) initialized.")

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=2000)  # Increased buffer size
        self.batch_size = 64  # Increased batch size
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
        self.export_dir = export_dir
        if export_dir is not None:
            os.makedirs(self.export_dir, exist_ok=True)
            logging.info(f"Embedding export will occur every {self.export_period} updates, saving to {self.export_dir}")

        # Set device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logging.info(f"Using device: {self.device}")

    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, BatchNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, data):
        embedding, node_recon, edge_recon = self.autoencoder(data)

        # Node-level anomaly scores
        node_scores = self.node_anomaly_mlp(embedding)
        logging.debug(f"Node anomaly scores shape: {node_scores.shape}")

        # Edge-level anomaly scores
        src, dst = data.edge_index[0], data.edge_index[1]
        edge_features = torch.cat([embedding[src], embedding[dst], data.edge_attr], dim=1)
        edge_scores = self.edge_anomaly_mlp(edge_features)
        logging.debug(f"Edge anomaly scores shape: {edge_scores.shape}")

        # Global anomaly score
        global_embedding = global_mean_pool(embedding, batch=torch.zeros(embedding.size(0), dtype=torch.long,
                                                                         device=embedding.device))
        global_score = self.global_anomaly_mlp(global_embedding)
        logging.debug(f"Global anomaly score shape: {global_score.shape}")

        return node_scores, edge_scores, global_score, node_recon, edge_recon, embedding, global_embedding

    def update_online(self, data: Data, n_steps=15, recon_weight=0.9, anomaly_weight=0.1,
                      use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        """Improved online update with better loss balancing and focal loss"""
        self.train()
        logging.info(f"Starting online update for {n_steps} steps.")

        if data is None:
            logging.error("`data` is None. Skipping update_online.")
            return 0.0

        # Move data to the device
        data = data.to(self.device)

        # Update running statistics
        if data.x is not None and data.x.numel() > 0:
            self._update_running_stats(self.node_stats, data.x)
        if data.edge_attr is not None and data.edge_attr.numel() > 0:
            self._update_running_stats(self.edge_stats, data.edge_attr)

        # Add current data to replay buffer
        self._add_to_replay_buffer(data.cpu())

        total_loss = 0.0
        successful_steps = 0

        for step in range(n_steps):
            logging.debug(f"Online update step: {step + 1}/{n_steps}")
            batch = self._get_training_batch()
            if batch is None:
                logging.warning("No training batch available in replay buffer. Skipping this step.")
                continue

            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            loss = self._calculate_online_loss(batch, recon_weight, anomaly_weight,
                                               use_focal_loss, focal_alpha, focal_gamma)

            if loss is not None:
                loss.backward()
                # Gradient clipping with adaptive norm
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                successful_steps += 1
                logging.debug(f"Step {step + 1}: Loss={loss.item():.4f}")

        avg_loss = total_loss / successful_steps if successful_steps > 0 else 0.0
        if successful_steps > 0:
            self.scheduler.step()  # CosineAnnealingWarmRestarts doesn't need loss
            logging.info(
                f"Online update finished. Average loss: {avg_loss:.4f}, "
                f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        else:
            logging.warning("No successful steps during online update.")

        # Drift detection
        self.detect_feature_drift(data)

        return avg_loss

    def _calculate_online_loss(self, batch, recon_weight, anomaly_weight,
                               use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        """Enhanced loss calculation with focal loss and better reconstruction targets"""
        try:
            node_scores, edge_scores, _, node_recon, edge_recon, _, _ = self(batch)

            # Reconstruction losses with better targets
            if self.recon_loss_type == 'mse':
                recon_loss_node = F.mse_loss(node_recon,
                                             batch.x) if batch.x is not None and batch.x.numel() > 0 else torch.tensor(
                    0.0, device=self.device)
            elif self.recon_loss_type == 'l1':
                recon_loss_node = F.l1_loss(node_recon,
                                            batch.x) if batch.x is not None and batch.x.numel() > 0 else torch.tensor(
                    0.0, device=self.device)
            elif self.recon_loss_type == 'huber':
                recon_loss_node = F.huber_loss(node_recon,
                                               batch.x) if batch.x is not None and batch.x.numel() > 0 else torch.tensor(
                    0.0, device=self.device)
            else:
                raise ValueError(f"Unsupported node reconstruction loss type: {self.recon_loss_type}")

            # Edge reconstruction with adaptive targets
            if edge_recon is not None and edge_recon.numel() > 0:
                if self.edge_recon_loss_type == 'bce':
                    # Use more realistic edge existence probability
                    edge_recon_target = torch.ones_like(edge_recon) * 0.8  # Higher probability for existing edges
                    recon_loss_edge = F.binary_cross_entropy(edge_recon, edge_recon_target)
                elif self.edge_recon_loss_type == 'mse':
                    edge_recon_target = torch.ones_like(edge_recon) * 0.8
                    recon_loss_edge = F.mse_loss(edge_recon, edge_recon_target)
                elif self.edge_recon_loss_type == 'l1':
                    edge_recon_target = torch.ones_like(edge_recon) * 0.8
                    recon_loss_edge = F.l1_loss(edge_recon, edge_recon_target)
                else:
                    raise ValueError(f"Unsupported edge reconstruction loss type: {self.edge_recon_loss_type}")
            else:
                recon_loss_edge = torch.tensor(0.0, device=self.device)

            # Anomaly loss with regularization
            if use_focal_loss:
                # Focal loss for anomaly scores (assuming normal data should have low scores)
                node_targets = torch.zeros_like(node_scores)
                edge_targets = torch.zeros_like(edge_scores)

                node_probs = torch.sigmoid(node_scores)
                edge_probs = torch.sigmoid(edge_scores)

                # Focal loss computation
                node_focal_loss = -focal_alpha * (1 - node_probs) ** focal_gamma * torch.log(node_probs + 1e-8)
                edge_focal_loss = -focal_alpha * (1 - edge_probs) ** focal_gamma * torch.log(edge_probs + 1e-8)

                anomaly_loss = (node_focal_loss.mean() + edge_focal_loss.mean()) / 2
            else:
                # Standard L1 regularization for normal behavior
                anomaly_loss_node = node_scores.abs().mean() if node_scores is not None and node_scores.numel() > 0 else torch.tensor(
                    0.0, device=self.device)
                anomaly_loss_edge = edge_scores.abs().mean() if edge_scores is not None and edge_scores.numel() > 0 else torch.tensor(
                    0.0, device=self.device)
                anomaly_loss = (anomaly_loss_node + anomaly_loss_edge) / 2

            # Combined loss with better weighting
            total_recon_loss = recon_loss_node + recon_loss_edge
            loss = recon_weight * total_recon_loss + anomaly_weight * anomaly_loss

            return loss
        except Exception as e:
            logging.error(f"Error during loss calculation: {e}")
            return None

    # ... (rest of the methods remain the same with minor improvements)

    def _update_running_stats(self, running_stats, current_data, alpha=0.1):
        """Updates the running mean and std of the features."""
        if current_data is not None and current_data.numel() > 0:
            mean = current_data.mean(dim=0)
            std = current_data.std(dim=0)
            running_stats.update(mean.detach().cpu().numpy())
            running_stats.update(std.detach().cpu().numpy())

    def _get_training_batch(self):
        """Retrieves a training batch from the replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = Batch.from_data_list([self.replay_buffer[i] for i in indices])
        logging.debug(f"Sampled batch from replay buffer (size: {batch.num_graphs} graphs).")
        return batch

    def detect_feature_drift(self, data):
        """Statistical drift detection."""
        if data.x is None or data.edge_attr is None:
            logging.warning("Skipping drift detection: Node or edge features are None.")
            return

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
        self.feature_mean = (current_node_mean.clone().detach(), current_edge_mean.clone().detach())
        self.feature_std = (data.x.std(dim=0).clone().detach(), data.edge_attr.std(dim=0).clone().detach())

    def _adaptive_reset(self):
        """Partial model reset for major behavior changes."""
        logging.warning("Major behavior change detected - performing adaptive reset of scoring heads and decoder.")

        # Reset only the anomaly scoring heads and the autoencoder's decoder
        for name, module in self.named_modules():
            if name.endswith('mlp') or 'decoder' in name:
                logging.info(f"Resetting parameters of module: {name}")
                module.apply(self._init_weights)

        # Reset optimizer state
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.003, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        self.consecutive_drifts = 0
        logging.info("Optimizer and scheduler reset.")

    def _add_to_replay_buffer(self, data: Data):
        """Adds a Data object to the replay buffer."""
        logging.debug(f"Adding data to replay buffer. Has edge_index: {hasattr(data, 'edge_index')}")
        self.replay_buffer.append(data)
        if len(self.replay_buffer) > self.replay_buffer.maxlen:
            self.replay_buffer.popleft()

    def detect_anomalies(self, data: Data, threshold=2.5):
        """Detect anomalies with statistical normalization on reconstruction error."""
        self.eval()
        logging.info("Starting anomaly detection.")
        data = data.to(self.device)
        with torch.no_grad():
            embedding, node_recon, edge_recon = self.autoencoder(data)

            # Reconstruction errors
            node_recon_error = F.mse_loss(node_recon, data.x, reduction='none').mean(dim=1)
            if data.edge_attr is not None and data.edge_attr.numel() > 0:
                edge_recon_target_mean = data.edge_attr.mean(dim=1, keepdim=True)
                edge_recon_error = torch.abs(edge_recon - edge_recon_target_mean.squeeze())
            else:
                edge_recon_error = torch.abs(edge_recon - 0.5)

            # Statistical normalization of reconstruction errors
            node_mean_recon = node_recon_error.mean()
            node_std_recon = node_recon_error.std()
            edge_mean_recon = edge_recon_error.mean()
            edge_std_recon = edge_recon_error.std()

            anomalous_nodes = (node_recon_error > node_mean_recon + threshold * node_std_recon).nonzero(as_tuple=True)[
                0]
            anomalous_edges = (edge_recon_error > edge_mean_recon + threshold * edge_std_recon).nonzero(as_tuple=True)[
                0]

            # Get anomaly scores from the dedicated MLPs
            node_scores_mlp = torch.sigmoid(self.node_anomaly_mlp(embedding)).squeeze()
            edge_scores_mlp = torch.sigmoid(self.edge_anomaly_mlp(
                torch.cat([embedding[data.edge_index[0]], embedding[data.edge_index[1]], data.edge_attr],
                          dim=1))).squeeze()
            global_score_mlp = torch.sigmoid(self.global_anomaly_mlp(global_mean_pool(embedding, batch=torch.zeros(
                embedding.size(0), dtype=torch.long, device=embedding.device)))).item()

            return {
                'node_anomalies_recon': anomalous_nodes.cpu().numpy(),
                'edge_anomalies_recon': anomalous_edges.cpu().numpy(),
                'node_recon_errors': node_recon_error.cpu().numpy(),
                'edge_recon_errors': edge_recon_error.cpu().numpy(),
                'node_scores_mlp': node_scores_mlp.cpu().numpy(),
                'edge_scores_mlp': edge_scores_mlp.cpu().numpy(),
                'global_anomaly_mlp': global_score_mlp,
                'embedding': embedding.cpu().numpy()
            }

    def export_embeddings(self, data: Data, filename="embeddings.png", n_components=2, perplexity=30, n_iter=300):
        self.eval()
        data = data.to(self.device)
        with torch.no_grad():
            embedding, _, _ = self.autoencoder(data)
            embedding_np = embedding.cpu().numpy()

            n_samples = embedding_np.shape[0]
            safe_perplexity = min(perplexity, max(5, n_samples - 1))

            if safe_perplexity < 1:
                logging.error(
                    f"Error during embedding export: n_samples ({n_samples}) is too small for meaningful t-SNE.")
                return

            tsne = TSNE(n_components=n_components, random_state=42, perplexity=safe_perplexity, max_iter=n_iter)
            reduced_embeddings = tsne.fit_transform(embedding_np)

            plt.figure(figsize=(8, 8))
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
            plt.title(f"Node Embeddings Visualization (t-SNE, Perplexity={safe_perplexity})")
            plt.xlabel("t-SNE Dimension 1")
            plt.ylabel("t-SNE Dimension 2")
            plt.savefig(filename)
            plt.close()
            logging.info(f"Embeddings visualization saved to {filename}")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self.to(value)