import logging
import os
import traceback
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

# Configure logging for better insights into training
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RunningStats:
    """Utility for tracking feature statistics using Welford's algorithm."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None  # Sum of squared differences from the mean
        logging.info("RunningStats initialized.")

    def update(self, x):
        """
        Updates the running statistics with a new batch of data.
        x: A numpy array or torch.Tensor of shape (batch_size, feature_dim)
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = x.astype(np.float64)  # Ensure float64 for precision

        batch_n = x.shape[0]
        batch_mean = np.mean(x, axis=0)

        if self.n == 0:
            self.mean = batch_mean
            self.M2 = np.sum(np.square(x - batch_mean), axis=0)  # Initial M2
            self.n = batch_n
            logging.debug(f"Initialized RunningStats with first batch: mean={self.mean}, M2={self.M2}, n={self.n}")
        else:
            # Welford's online algorithm for combining batches
            delta = batch_mean - self.mean
            total_n = self.n + batch_n

            new_mean = self.mean + delta * batch_n / total_n

            # Calculate M2 for the current batch
            batch_M2 = np.sum(np.square(x - batch_mean), axis=0)

            # Combine M2 values
            new_M2 = self.M2 + batch_M2 + np.square(delta) * self.n * batch_n / total_n

            self.mean = new_mean
            self.M2 = new_M2
            self.n = total_n
            logging.debug(f"Updated RunningStats: new_mean={self.mean}, new_M2={self.M2}, new_n={self.n}")

    def get_mean(self):
        """Returns the current running mean."""
        return self.mean

    def get_std(self):
        """Returns the current running standard deviation."""
        # Add a small epsilon to prevent division by zero for features with zero variance
        std = np.sqrt(self.M2 / self.n) if self.n > 0 else None
        if std is not None:
            std = np.maximum(std, 1e-6)  # Ensure std is never zero
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

        # Store edge feature dimension for handling None cases
        self.edge_feature_dim = edge_feature_dim if edge_feature_dim is not None else 0

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(0.2)

        # Encoder GAT layers with better initialization
        current_dim = node_feature_dim
        for i in range(num_gat_layers):
            if i == num_gat_layers - 1:  # Last layer
                out_dim = embedding_dim
                heads = 1
            else:
                out_dim = hidden_dim
                heads = gat_heads

            # Handle None edge_feature_dim for GAT layers
            gat_edge_dim = edge_feature_dim if edge_feature_dim is not None else None
            self.gat_layers.append(GATConv(current_dim, out_dim, heads=heads, edge_dim=gat_edge_dim,
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

        # Initialize weights properly
        self.apply(self._init_weights)
        logging.debug("Decoder MLP layers initialized with proper weight initialization.")

    def _init_weights(self, module):
        """Proper weight initialization for faster convergence"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Standard PyTorch BatchNorm layers
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif hasattr(module, '__class__') and 'BatchNorm' in module.__class__.__name__:
            # PyTorch Geometric BatchNorm or other BatchNorm variants
            try:
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            except Exception as e:
                # If initialization fails for any reason, just skip it
                logging.debug(f"Skipping initialization for {module.__class__.__name__}: {e}")
                pass
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def encode(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.gat_layers):
            x_prev = x  # Store x before transformation for residual connection
            x = conv(x, edge_index, edge_attr)

            if i < self.num_gat_layers - 1:
                # Apply batch normalization
                if self.use_batch_norm and self.batch_norms is not None:
                    x = self.batch_norms[i](x)

                # Apply activation
                x = F.elu(x)  # ELU instead of ReLU for better gradients

                # Apply residual connection if dimensions match
                # Only apply if input and output dimensions are the same
                if self.use_residual and x.shape == x_prev.shape:
                    x = x + x_prev

                # Apply dropout
                x = self.dropout(x)
        return x

    def decode_node(self, z):
        return self.node_decoder(z)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        embedding = self.encode(x, edge_index, edge_attr)
        node_recon = self.decode_node(embedding)
        return embedding, node_recon


class NodeGNNAnomalyDetector(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, embedding_dim=32, num_gat_layers=2,
                 gat_heads=4, recon_loss_type='mse', use_batch_norm=True, use_residual=True,
                 batch_size=64, export_period=5, export_dir: str = None):
        super().__init__()
        logging.info(
            f"Initializing NodeGNNAnomalyDetector with node_dim={node_feature_dim}, edge_dim={edge_feature_dim}, "
            f"hidden_dim={hidden_dim}, embedding_dim={embedding_dim}, num_gat_layers={num_gat_layers}, "
            f"gat_heads={gat_heads}, recon_loss_type={recon_loss_type}")

        self.autoencoder = GraphAutoencoder(node_feature_dim, edge_feature_dim, hidden_dim, embedding_dim,
                                            num_gat_layers, gat_heads, use_batch_norm, use_residual)
        self.recon_loss_type = recon_loss_type

        # Node anomaly MLP
        self.node_anomaly_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)
        logging.debug("Node anomaly scoring MLP initialized.")

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.003, weight_decay=1e-4,
                                           betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        logging.info("Optimizer (AdamW) and scheduler (CosineAnnealingWarmRestarts) initialized.")

        self.replay_buffer = deque(maxlen=2000)
        self.batch_size = batch_size
        logging.info(f"Replay buffer initialized with maxlen={self.replay_buffer.maxlen}, batch_size={self.batch_size}")

        # Statistical controls for features
        self.node_stats = RunningStats()
        self.edge_stats = RunningStats()
        self.feature_mean = None
        self.feature_std = None
        self.drift_threshold = 0.15
        self.consecutive_drifts = 0
        logging.info(f"Statistical controls initialized: drift_threshold={self.drift_threshold}")

        # Anomaly detection thresholds for nodes
        self.node_recon_error_threshold_factor = 2.0  # Multiplier for std dev above mean
        self.node_mlp_score_threshold = 0.8  # Direct sigmoid score threshold
        logging.info(
            f"Anomaly thresholds initialized: node_recon_error_factor={self.node_recon_error_threshold_factor}, "
            f"node_mlp_score_threshold={self.node_mlp_score_threshold}")

        # Statistics for node anomaly thresholds (e.g., mean/std of reconstruction errors for normal data)
        self.node_recon_error_stats = RunningStats()
        logging.info("RunningStats for node reconstruction errors initialized.")

        self.export_period = export_period
        self.update_count = 0
        self.export_dir = export_dir
        if export_dir is not None:
            os.makedirs(self.export_dir, exist_ok=True)
            logging.info(f"Embedding export will occur every {self.export_period} updates, saving to {self.export_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        logging.info(f"Using device: {self.device}")

    def _init_weights(self, module):
        """Proper weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif hasattr(module, '__class__') and 'BatchNorm' in module.__class__.__name__:
            try:
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            except Exception as e:
                logging.debug(f"Skipping initialization for {module.__class__.__name__}: {e}")
                pass
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, 1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _normalize_features(self, data: Data):
        """Normalizes node and edge features using current running statistics."""
        if self.feature_mean is None or self.feature_std is None:
            logging.warning("Feature statistics not yet available for normalization. Skipping normalization.")
            return data

        node_mean = torch.tensor(self.feature_mean[0], dtype=torch.float32, device=self.device)
        node_std = torch.tensor(self.feature_std[0], dtype=torch.float32, device=self.device)

        if data.x is not None and data.x.numel() > 0:
            data.x = (data.x - node_mean) / (node_std + 1e-6)

        if data.edge_attr is not None and data.edge_attr.numel() > 0 and self.feature_mean[1] is not None:
            edge_mean = torch.tensor(self.feature_mean[1], dtype=torch.float32, device=self.device)
            edge_std = torch.tensor(self.feature_std[1], dtype=torch.float32, device=self.device)
            data.edge_attr = (data.edge_attr - edge_mean) / (edge_std + 1e-6)

        return data

    def forward(self, data):
        normalized_data = self._normalize_features(data.clone())
        embedding, node_recon = self.autoencoder(normalized_data)
        node_scores = self.node_anomaly_mlp(embedding)
        logging.debug(f"Node anomaly scores shape: {node_scores.shape}")
        return node_scores, node_recon, embedding

    def update_online(self, data: Data, n_steps=30, recon_weight=1.0, anomaly_weight=0.1,
                      use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        self.train()
        logging.info(f"Starting online update for {n_steps} steps.")

        if data is None:
            logging.error("`data` is None. Skipping update_online.")
            return 0.0

        data_on_device = data.to(self.device)

        if data_on_device.x is not None and data_on_device.x.numel() > 0:
            self.node_stats.update(data_on_device.x)
        if data_on_device.edge_attr is not None and data_on_device.edge_attr.numel() > 0:
            self.edge_stats.update(data_on_device.edge_attr)

        node_mean = self.node_stats.get_mean()
        node_std = self.node_stats.get_std()
        edge_mean = self.edge_stats.get_mean() if self.edge_stats.n > 0 else None
        edge_std = self.edge_stats.get_std() if self.edge_stats.n > 0 else None

        self.feature_mean = (node_mean, edge_mean)
        self.feature_std = (node_std, edge_std)

        self._add_to_replay_buffer(data.cpu())

        total_loss = 0.0
        successful_steps = 0

        for step in range(n_steps):
            logging.debug(f"Online update step: {step + 1}/{n_steps}")
            batch = self._get_training_batch()
            if batch is None:
                logging.warning("Not enough data in replay buffer for a full batch. Skipping this step.")
                continue

            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            loss = self._calculate_online_loss(batch, recon_weight, anomaly_weight,
                                               use_focal_loss, focal_alpha, focal_gamma)

            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                grad_norms = [p.grad.norm().item() for p in self.parameters() if p.grad is not None]
                if grad_norms:
                    logging.debug(f"Gradient norms - Min: {min(grad_norms):.4f}, Max: {max(grad_norms):.4f}")

                self.optimizer.step()
                total_loss += loss.item()
                successful_steps += 1
                logging.debug(f"Step {step + 1}: Loss={loss.item():.4f}")

                with torch.no_grad():
                    # Get reconstruction for updating stats (against normalized original)
                    normalized_batch_x = self._normalize_features(batch.clone()).x
                    _, node_recon_eval, _ = self(batch)
                    node_recon_error = F.mse_loss(node_recon_eval, normalized_batch_x, reduction='none').mean(dim=1)
                    self.node_recon_error_stats.update(node_recon_error)

        avg_loss = total_loss / successful_steps if successful_steps > 0 else 0.0
        if successful_steps > 0:
            self.scheduler.step()
            logging.info(
                f"Online update finished. Average loss: {avg_loss:.4f}, "
                f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        else:
            logging.warning("No successful steps during online update.")

        self.detect_feature_drift(data)
        self.update_count += 1
        return avg_loss

    def _augment_graph(self, data: Data, edge_dropout_rate=0.1):
        """
        Performs graph augmentation (e.g., random edge dropout).
        """
        if data.edge_index is not None and data.edge_index.numel() > 0:
            num_edges = data.edge_index.size(1)
            mask = torch.rand(num_edges, device=data.edge_index.device) > edge_dropout_rate
            data.edge_index = data.edge_index[:, mask]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]
            logging.debug(f"Applied edge dropout: {num_edges - data.edge_index.size(1)} edges dropped.")
        return data

    def _add_to_replay_buffer(self, data: Data):
        """Adds a Data object to the replay buffer."""
        logging.debug(f"Adding data to replay buffer. Has edge_index: {hasattr(data, 'edge_index')}")
        self.replay_buffer.append(data)

    def _get_training_batch(self):
        """Retrieves a training batch from the replay buffer, normalizes it, and moves to device."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)

        batch_list = []
        for i in indices:
            data_item = self.replay_buffer[i].clone().to(self.device)
            data_item = self._normalize_features(data_item)
            data_item = self._augment_graph(data_item)
            batch_list.append(data_item)

        batch = Batch.from_data_list(batch_list)
        logging.debug(f"Sampled batch from replay buffer (size: {batch.num_graphs} graphs).")
        return batch

    def _calculate_online_loss(self, batch, recon_weight, anomaly_weight,
                               use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        """
        Enhanced loss calculation with focal loss and better reconstruction targets.
        """
        try:
            node_scores, node_recon, _ = self(batch)

            recon_loss_node = torch.tensor(0.0, device=self.device)
            if batch.x is not None and batch.x.numel() > 0:
                if self.recon_loss_type == 'mse':
                    recon_loss_node = F.mse_loss(node_recon, batch.x)
                elif self.recon_loss_type == 'l1':
                    recon_loss_node = F.l1_loss(node_recon, batch.x)
                elif self.recon_loss_type == 'huber':
                    recon_loss_node = F.huber_loss(node_recon, batch.x, delta=1.0)
                elif self.recon_loss_type == 'log_cosh':
                    recon_loss_node = torch.log(torch.cosh(node_recon - batch.x)).mean()
                else:
                    raise ValueError(f"Unsupported node reconstruction loss type: {self.recon_loss_type}")
            logging.debug(f"Node reconstruction loss: {recon_loss_node.item():.4f}")

            if use_focal_loss:
                node_probs = torch.sigmoid(node_scores)
                anomaly_loss = -focal_alpha * (node_probs) ** focal_gamma * torch.log(1 - node_probs + 1e-8)
                anomaly_loss = anomaly_loss.mean()
            else:
                anomaly_loss = node_scores.abs().mean() if node_scores is not None and node_scores.numel() > 0 else torch.tensor(
                    0.0, device=self.device)

            logging.debug(f"Anomaly regularization loss: {anomaly_loss.item():.4f}")

            loss = recon_weight * recon_loss_node + anomaly_weight * anomaly_loss
            return loss
        except Exception as e:
            logging.error(f"Error during loss calculation: {e}")
            traceback.print_exc()
            return None

    def detect_feature_drift(self, data):
        """Statistical drift detection on unnormalized features."""
        if data.x is None:
            logging.warning("Skipping drift detection: Node features are None.")
            return

        current_node_mean = data.x.mean(dim=0).cpu().numpy()
        current_node_std = data.x.std(dim=0).cpu().numpy()

        # Edge stats are still gathered but not used for drift detection as per requirement
        current_edge_mean = data.edge_attr.mean(
            dim=0).cpu().numpy() if data.edge_attr is not None and data.edge_attr.numel() > 0 else None
        current_edge_std = data.edge_attr.std(
            dim=0).cpu().numpy() if data.edge_attr is not None and data.edge_attr.numel() > 0 else None

        logging.debug(f"Current node mean: {current_node_mean}")

        if self.feature_mean is None or self.feature_std is None:
            self.feature_mean = (current_node_mean, current_edge_mean)
            self.feature_std = (current_node_std, current_edge_std)
            logging.info("Initialized reference feature statistics for drift detection.")
            return

        node_diff = np.abs(current_node_mean - self.feature_mean[0]) / (self.feature_std[0] + 1e-6)
        logging.debug(f"Node drift difference (max): {node_diff.max():.4f}")

        if node_diff.max() > self.drift_threshold:
            self.consecutive_drifts += 1
            logging.warning(f"Feature drift detected ({self.consecutive_drifts} consecutive)")

            if self.consecutive_drifts >= 3:
                logging.warning("Triggering model adaptation due to persistent feature drift.")
                self._adapt_to_drift()
                self.consecutive_drifts = 0
        else:
            self.consecutive_drifts = 0

        self.feature_mean = (current_node_mean, current_edge_mean)
        self.feature_std = (current_node_std, current_edge_std)
        logging.debug(
            f"Updated reference mean: node={self.feature_mean[0]}, edge={self.feature_mean[1] if self.feature_mean[1] is not None else 'N/A'}")
        logging.debug(
            f"Updated reference std: node={self.feature_std[0]}, edge={self.feature_std[1] if self.feature_std[1] is not None else 'N/A'}")

    def _adapt_to_drift(self):
        """Adapts the model to feature drift by reinitializing optimizer and some layers."""
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.003, weight_decay=1e-4,
                                           betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        self.node_anomaly_mlp[-1].apply(self._init_weights)
        self.autoencoder.node_decoder.apply(self._init_weights)

        buffer_size = len(self.replay_buffer)
        keep_size = buffer_size // 2
        self.replay_buffer = deque(list(self.replay_buffer)[-keep_size:], maxlen=self.replay_buffer.maxlen)
        logging.info("Model adaptation completed: optimizer reset, selected layers reinitialized, buffer trimmed.")

    def get_anomaly_scores(self, data: Data):
        """
        Computes node anomaly scores and reconstruction errors for given data without updating the model.
        Returns:
            dict: Contains 'node_scores_mlp' (sigmoid-normalized MLP scores 0-1)
                  and 'node_recon_errors' (raw MSE reconstruction errors).
        """
        self.eval()
        with torch.no_grad():
            data_on_device = data.to(self.device)
            normalized_data = self._normalize_features(data_on_device.clone())

            node_scores_raw, node_recon, _ = self(normalized_data)

            node_recon_error = F.mse_loss(node_recon, normalized_data.x, reduction='none').mean(dim=1)

            node_scores_mlp = torch.sigmoid(node_scores_raw).squeeze().cpu().numpy()

            return {
                'node_scores_mlp': node_scores_mlp,
                'node_recon_errors': node_recon_error.cpu().numpy()
            }

    def detect_anomalies(self, data: Data):
        """
        Detects anomalies based on computed scores and predefined thresholds, focusing only on nodes.

        Args:
            data (Data): The input PyG Data object for which to detect anomalies.

        Returns:
            dict: A dictionary containing:
                - 'node_scores_mlp': NumPy array of MLP-based anomaly scores for nodes.
                - 'node_recon_errors': NumPy array of reconstruction errors for nodes.
                - 'node_anomalies_mlp': Indices of nodes detected as anomalous by MLP.
                - 'node_anomalies_recon': Indices of nodes detected as anomalous by reconstruction error.
        """
        logging.info("Detecting anomalies...")
        scores = self.get_anomaly_scores(data)
        node_scores_mlp = scores['node_scores_mlp']
        node_recon_errors = scores['node_recon_errors']

        # Node Anomaly Detection based on MLP scores
        node_anomalies_mlp_indices = np.where(node_scores_mlp > self.node_mlp_score_threshold)[0]
        logging.debug(
            f"Nodes exceeding MLP score threshold ({self.node_mlp_score_threshold}): {len(node_anomalies_mlp_indices)}")

        # Node Anomaly Detection based on Reconstruction error
        recon_mean = self.node_recon_error_stats.get_mean()
        recon_std = self.node_recon_error_stats.get_std()

        if self.node_recon_error_stats.n < 2:
            logging.warning(
                "Not enough reconstruction error statistics for robust thresholding. Using percentile-based fallback.")
            node_anomalies_recon_indices = np.where(node_recon_errors > np.percentile(node_recon_errors, 95))[0]
        else:
            recon_error_threshold = recon_mean + self.node_recon_error_threshold_factor * recon_std
            node_anomalies_recon_indices = np.where(node_recon_errors > recon_error_threshold)[0]
            logging.debug(
                f"Nodes exceeding reconstruction error threshold ({recon_error_threshold.item():.4f}): {len(node_anomalies_recon_indices)}")

        return {
            'node_scores_mlp': node_scores_mlp,
            'node_recon_errors': node_recon_errors,
            'node_anomalies_mlp': node_anomalies_mlp_indices,
            'node_anomalies_recon': node_anomalies_recon_indices,
        }

    def export_embeddings(self, data: Data, filename_suffix=""):
        """Exports node embeddings to a file for visualization or analysis."""
        if self.export_dir is None:
            logging.warning("Export directory not set. Skipping embedding export.")
            return

        self.eval()
        with torch.no_grad():
            data_on_device = data.to(self.device)
            normalized_data = self._normalize_features(data_on_device.clone())
            embeddings = self.autoencoder.encode(
                normalized_data.x,
                normalized_data.edge_index,
                normalized_data.edge_attr
            ).detach().cpu().numpy()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            node_emb_path = os.path.join(self.export_dir, f"node_embeddings_{timestamp}{filename_suffix}.npy")
            np.save(node_emb_path, embeddings)
            logging.info(f"Node embeddings exported: {node_emb_path}")

    def get_model_statistics(self):
        """Returns current model statistics and state information."""
        stats = {
            'update_count': self.update_count,
            'replay_buffer_size': len(self.replay_buffer),
            'consecutive_drifts': self.consecutive_drifts,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'device': str(self.device),
            'node_stats': {
                'mean': self.node_stats.get_mean().tolist() if self.node_stats.n > 0 else None,
                'std': self.node_stats.get_std().tolist() if self.node_stats.n > 0 else None,
                'count': self.node_stats.n
            },
            'edge_stats': {  # Still included for completeness of stats, but not used for anomaly detection
                'mean': self.edge_stats.get_mean().tolist() if self.edge_stats.n > 0 else None,
                'std': self.edge_stats.get_std().tolist() if self.edge_stats.n > 0 else None,
                'count': self.edge_stats.n
            },
            'node_recon_error_stats': {
                'mean': self.node_recon_error_stats.get_mean().tolist() if self.node_recon_error_stats.n > 0 else None,
                'std': self.node_recon_error_stats.get_std().tolist() if self.node_recon_error_stats.n > 0 else None,
                'count': self.node_recon_error_stats.n
            }
        }
        return stats

    def reset_model_state(self):
        """Resets the model to initial state (useful for retraining scenarios)."""
        logging.info("Resetting model state...")

        self.replay_buffer.clear()
        logging.debug("Replay buffer cleared.")

        self.node_stats = RunningStats()
        self.edge_stats = RunningStats()
        self.node_recon_error_stats = RunningStats()  # Reset recon error stats too
        self.feature_mean = None
        self.feature_std = None
        logging.debug("RunningStats and feature_mean/std reset.")

        self.consecutive_drifts = 0
        self.update_count = 0
        logging.debug("Drift and update counters reset.")

        self.apply(self._init_weights)
        logging.debug("All model weights reinitialized.")

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=0.003, weight_decay=1e-4,
                                           betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        logging.debug("Optimizer and scheduler reset to initial configurations.")

        self.to(self.device)
        logging.info("Model state completely reset.")