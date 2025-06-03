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

        # Edge decoder input dimension: embedding_dim * 2 + edge_feature_dim
        # Use max(1, edge_feature_dim) to handle None case for the linear layer input
        edge_decoder_input_dim = embedding_dim * 2 + max(1, self.edge_feature_dim)
        self.edge_decoder = nn.Sequential(
            nn.Linear(edge_decoder_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a score for edge existence/anomaly
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

    def decode_edge(self, z, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]

        # Handle case where edge_attr is None
        if edge_attr is not None:
            edge_features = torch.cat([z[src], z[dst], edge_attr], dim=1)
        else:
            # If no edge attributes, create zero padding matching the expected dimension
            device = z.device
            num_edges = edge_index.shape[1]
            # Use the stored edge_feature_dim from __init__
            if self.edge_feature_dim > 0:
                zero_edge_attr = torch.zeros(num_edges, self.edge_feature_dim, device=device)
                edge_features = torch.cat([z[src], z[dst], zero_edge_attr], dim=1)
            else:
                # If edge_feature_dim was 0, just concatenate node embeddings
                edge_features = torch.cat([z[src], z[dst]], dim=1)
                # Adjust edge_decoder_input_dim in __init__ if this path is taken without a dummy feature

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
                 recon_loss_type='mse', edge_recon_loss_type='bce', use_batch_norm=True, use_residual=True,
                 batch_size=64):
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

        # Edge anomaly MLP input dimension: embedding_dim * 2 + edge_feature_dim
        # Use max(1, edge_feature_dim) for the linear layer input if edge_feature_dim can be 0 or None
        edge_anomaly_input_dim = embedding_dim * 2 + (edge_feature_dim if edge_feature_dim is not None else 0)
        if edge_anomaly_input_dim == embedding_dim * 2:  # If no edge features, ensure it's still > 0
            edge_anomaly_input_dim += 1  # Add a dummy dimension if edge_feature_dim was effectively zero

        self.edge_anomaly_mlp = nn.Sequential(
            nn.Linear(edge_anomaly_input_dim, hidden_dim),
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
            nn.LayerNorm(hidden_dim),  # Using LayerNorm here as per your suggestion
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
        self.batch_size = batch_size  # Increased batch size
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

    def _normalize_features(self, data: Data):
        """Normalizes node and edge features using current running statistics."""
        if self.feature_mean is None or self.feature_std is None:
            logging.warning("Feature statistics not yet available for normalization. Skipping normalization.")
            return data

        # Ensure feature_mean and feature_std are on the correct device and are tensors
        node_mean = torch.tensor(self.feature_mean[0], dtype=torch.float32, device=self.device)
        node_std = torch.tensor(self.feature_std[0], dtype=torch.float32, device=self.device)
        edge_mean = torch.tensor(self.feature_mean[1], dtype=torch.float32, device=self.device)
        edge_std = torch.tensor(self.feature_std[1], dtype=torch.float32, device=self.device)

        # Apply normalization, ensuring no division by zero
        if data.x is not None and data.x.numel() > 0:
            data.x = (data.x - node_mean) / (node_std + 1e-6)
        # Only normalize edge_attr if it exists and has elements
        if data.edge_attr is not None and data.edge_attr.numel() > 0:
            data.edge_attr = (data.edge_attr - edge_mean) / (edge_std + 1e-6)
        return data

    def forward(self, data):
        # Normalize features before passing to autoencoder
        # Clone data to avoid modifying the original `data` object passed in.
        normalized_data = self._normalize_features(data.clone())
        x, edge_index, edge_attr = normalized_data.x, normalized_data.edge_index, normalized_data.edge_attr

        embedding, node_recon, edge_recon = self.autoencoder(normalized_data)

        # Node-level anomaly scores based on embeddings
        node_scores = self.node_anomaly_mlp(embedding)
        logging.debug(f"Node anomaly scores shape: {node_scores.shape}")

        # Edge-level anomaly scores based on embeddings and edge features
        src, dst = edge_index[0], edge_index[1]

        # Handle None edge attributes for anomaly MLP input
        if edge_attr is not None:
            edge_features_for_anomaly_mlp = torch.cat([embedding[src], embedding[dst], edge_attr], dim=1)
        else:
            # If no edge attributes, create zero padding matching the expected dimension
            device = embedding.device
            num_edges = edge_index.shape[1]
            # Use the stored edge_feature_dim from autoencoder
            edge_feature_dim_ae = getattr(self.autoencoder, 'edge_feature_dim', 0)
            if edge_feature_dim_ae > 0:
                zero_edge_attr = torch.zeros(num_edges, edge_feature_dim_ae, device=device)
                edge_features_for_anomaly_mlp = torch.cat([embedding[src], embedding[dst], zero_edge_attr], dim=1)
            else:
                # If edge_feature_dim was 0, just concatenate node embeddings
                edge_features_for_anomaly_mlp = torch.cat([embedding[src], embedding[dst]], dim=1)
                # If edge_anomaly_mlp expects a dummy feature, add it here
                if self.edge_anomaly_mlp[0].in_features == embedding.shape[1] * 2 + 1:
                    edge_features_for_anomaly_mlp = torch.cat(
                        [edge_features_for_anomaly_mlp, torch.zeros(num_edges, 1, device=device)], dim=1)

        edge_scores = self.edge_anomaly_mlp(edge_features_for_anomaly_mlp)
        logging.debug(f"Edge anomaly scores shape: {edge_scores.shape}")

        # Global anomaly score based on global mean pooled embedding
        global_embedding = global_mean_pool(embedding, batch=torch.zeros(embedding.size(0), dtype=torch.long,
                                                                         device=embedding.device))
        global_score = self.global_anomaly_mlp(global_embedding)
        logging.debug(f"Global anomaly score shape: {global_score.shape}")

        return node_scores, edge_scores, global_score, node_recon, edge_recon, embedding, global_embedding

    def update_online(self, data: Data, n_steps=30, recon_weight=0.9, anomaly_weight=0.1,
                      use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        """Improved online update with better loss balancing and focal loss"""
        self.train()
        logging.info(f"Starting online update for {n_steps} steps.")

        if data is None:
            logging.error("`data` is None. Skipping update_online.")
            return 0.0

        # Move data to the device temporarily for stat update if it's not already there
        data_on_device = data.to(self.device)

        # Update running statistics with raw data (before normalization)
        if data_on_device.x is not None and data_on_device.x.numel() > 0:
            self.node_stats.update(data_on_device.x)
        if data_on_device.edge_attr is not None and data_on_device.edge_attr.numel() > 0:
            self.edge_stats.update(data_on_device.edge_attr)

        # Update the feature_mean and feature_std for normalization
        # Ensure these are numpy arrays for consistency with RunningStats output
        self.feature_mean = (self.node_stats.get_mean(), self.edge_stats.get_mean())
        self.feature_std = (self.node_stats.get_std(), self.edge_stats.get_std())

        # Add current data to replay buffer (store raw data on CPU to save GPU memory)
        self._add_to_replay_buffer(data.cpu())

        total_loss = 0.0
        successful_steps = 0

        for step in range(n_steps):
            logging.debug(f"Online update step: {step + 1}/{n_steps}")
            batch = self._get_training_batch()  # Sample batch from buffer, normalizes and moves to device
            if batch is None:
                logging.warning("Not enough data in replay buffer for a full batch. Skipping this step.")
                continue

            self.optimizer.zero_grad()
            batch = batch.to(self.device)  # Ensure batch is on device for forward pass
            loss = self._calculate_online_loss(batch, recon_weight, anomaly_weight,
                                               use_focal_loss, focal_alpha, focal_gamma)

            if loss is not None:
                loss.backward()
                # Gradient clipping with adaptive norm
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # Gradient Monitoring
                grad_norms = [p.grad.norm().item() for p in self.parameters() if p.grad is not None]
                if grad_norms:
                    logging.debug(f"Gradient norms - Min: {min(grad_norms):.4f}, Max: {max(grad_norms):.4f}")

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

        # Drift detection (using the original, unnormalized data)
        self.detect_feature_drift(data)
        self.update_count += 1  # Increment update count for curriculum learning if used

        return avg_loss

    def _augment_graph(self, data: Data, edge_dropout_rate=0.1):
        """
        Performs graph augmentation (e.g., random edge dropout).
        data: A Data object to be augmented.
        edge_dropout_rate: Probability of dropping an edge.
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
        # deque automatically handles maxlen, no need for popleft explicitly

    def _get_training_batch(self):
        """Retrieves a training batch from the replay buffer, normalizes it, and moves to device."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)

        batch_list = []
        for i in indices:
            # Clone and move to device first, then normalize and augment
            data_item = self.replay_buffer[i].clone().to(self.device)
            data_item = self._normalize_features(data_item)
            data_item = self._augment_graph(data_item)  # Apply graph augmentation
            batch_list.append(data_item)

        batch = Batch.from_data_list(batch_list)
        logging.debug(f"Sampled batch from replay buffer (size: {batch.num_graphs} graphs).")
        return batch

    def _calculate_online_loss(self, batch, recon_weight, anomaly_weight,
                               use_focal_loss=True, focal_alpha=0.25, focal_gamma=2.0):
        """Enhanced loss calculation with focal loss and better reconstruction targets"""
        try:
            # Note: `batch` here is already normalized and augmented by _get_training_batch
            node_scores, edge_scores, _, node_recon, edge_recon, _, _ = self(batch)

            # Node Reconstruction Loss
            recon_loss_node = torch.tensor(0.0, device=self.device)
            if batch.x is not None and batch.x.numel() > 0:
                if self.recon_loss_type == 'mse':
                    recon_loss_node = F.mse_loss(node_recon, batch.x)
                elif self.recon_loss_type == 'l1':
                    recon_loss_node = F.l1_loss(node_recon, batch.x)
                elif self.recon_loss_type == 'huber':
                    recon_loss_node = F.huber_loss(node_recon, batch.x, delta=1.0)  # Delta parameter for Huber loss
                elif self.recon_loss_type == 'log_cosh':
                    recon_loss_node = torch.log(torch.cosh(node_recon - batch.x)).mean()
                else:
                    raise ValueError(f"Unsupported node reconstruction loss type: {self.recon_loss_type}")
            logging.debug(f"Node reconstruction loss: {recon_loss_node.item():.4f}")

            # Edge Reconstruction Loss
            recon_loss_edge = torch.tensor(0.0, device=self.device)
            if batch.edge_attr is not None and batch.edge_attr.numel() > 0:
                if self.edge_recon_loss_type == 'bce':
                    # Use more realistic edge existence probability for existing edges
                    # Assuming edge_recon is output of sigmoid, target should be 0-1
                    edge_recon_target = torch.ones_like(edge_recon) * 0.8  # Higher probability for existing edges
                    recon_loss_edge = F.binary_cross_entropy(edge_recon, edge_recon_target)
                elif self.edge_recon_loss_type == 'mse':
                    # Target should be the actual normalized edge attributes
                    recon_loss_edge = F.mse_loss(edge_recon, batch.edge_attr)
                elif self.edge_recon_loss_type == 'l1':
                    # Target should be the actual normalized edge attributes
                    recon_loss_edge = F.l1_loss(edge_recon, batch.edge_attr)
                else:
                    raise ValueError(f"Unsupported edge reconstruction loss type: {self.edge_recon_loss_type}")
            logging.debug(f"Edge reconstruction loss: {recon_loss_edge.item():.4f}")

            # Anomaly loss with regularization
            if use_focal_loss:
                # Focal loss for anomaly scores (assuming normal data should have low scores, target 0)
                node_probs = torch.sigmoid(node_scores)
                edge_probs = torch.sigmoid(edge_scores)

                # Focal loss computation for target 0 (normal data)
                # FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)
                # Here, p_t is the probability of being the target class (normal, which is 0)
                # So p_t = 1 - node_probs (prob of being normal)
                node_focal_loss = -focal_alpha * (node_probs) ** focal_gamma * torch.log(1 - node_probs + 1e-8)
                edge_focal_loss = -focal_alpha * (edge_probs) ** focal_gamma * torch.log(1 - edge_probs + 1e-8)

                anomaly_loss = (node_focal_loss.mean() + edge_focal_loss.mean()) / 2
            else:
                # Standard L1 regularization for normal behavior (push scores towards zero)
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

    def detect_feature_drift(self, data):
        """Statistical drift detection on unnormalized features."""
        if data.x is None or data.edge_attr is None:
            logging.warning("Skipping drift detection: Node or edge features are None.")
            return

        current_node_mean = data.x.mean(dim=0).cpu().numpy()
        current_edge_mean = data.edge_attr.mean(dim=0).cpu().numpy()
        current_node_std = data.x.std(dim=0).cpu().numpy()
        current_edge_std = data.edge_attr.std(dim=0).cpu().numpy()

        logging.debug(
            f"Current node mean: {current_node_mean}, current edge mean: {current_edge_mean}")

        # Initialize feature_mean and feature_std if they are None (first call)
        if self.feature_mean is None or self.feature_std is None:
            self.feature_mean = (current_node_mean, current_edge_mean)
            self.feature_std = (current_node_std, current_edge_std)
            logging.info("Initialized reference feature statistics for drift detection.")
            return

        # Calculate difference relative to historical standard deviation (robust to scale)
        node_diff = np.abs(current_node_mean - self.feature_mean[0]) / (self.feature_std[0] + 1e-6)
        edge_diff = np.abs(current_edge_mean - self.feature_mean[1]) / (self.feature_std[1] + 1e-6)
        logging.debug(
            f"Node drift difference (max): {node_diff.max():.4f}, Edge drift difference (max): {edge_diff.max():.4f}")

        if node_diff.max() > self.drift_threshold or edge_diff.max() > self.drift_threshold:
            self.consecutive_drifts += 1
            logging.warning(f"Feature drift detected ({self.consecutive_drifts} consecutive)")

            if self.consecutive_drifts >= 3:  # Number of consecutive drifts to trigger reset
                self._adaptive_reset()
        else:
            self.consecutive_drifts = 0

        # Update reference statistics for drift detection (using current batch stats)
        self.feature_mean = (current_node_mean, current_edge_mean)
        self.feature_std = (current_node_std, current_edge_std)
        logging.debug(
            f"Updated reference mean: node={self.feature_mean[0]}, edge={self.feature_mean[1]}")
        logging.debug(
            f"Updated reference std: node={self.feature_std[0]}, edge={self.feature_std[1]}")

    def _adaptive_reset(self, new_lr=0.003, new_weight_decay=1e-4):
        """Partial model reset for major behavior changes."""
        logging.warning("Major behavior change detected - performing adaptive reset of scoring heads and decoder.")

        # Reset only the anomaly scoring heads and the autoencoder's decoder
        for name, module in self.named_modules():
            if name.endswith('mlp') or 'decoder' in name:
                logging.info(f"Resetting parameters of module: {name}")
                module.apply(self._init_weights)

        # Reset optimizer state with potentially new hyperparameters
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=new_lr, weight_decay=new_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        self.consecutive_drifts = 0
        logging.info("Optimizer and scheduler reset.")

    def _reset_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def detect_anomalies(self, data: Data, threshold=2.5):
        """Detect anomalies with statistical normalization on reconstruction error."""
        self.eval()
        logging.info("Starting anomaly detection.")
        data = data.to(self.device)
        # Normalize the input data for inference as well
        normalized_data = self._normalize_features(data.clone())

        with torch.no_grad():
            embedding, node_recon, edge_recon = self.autoencoder(normalized_data)

            # Reconstruction errors for normalized data
            node_recon_error = F.mse_loss(node_recon, normalized_data.x, reduction='none').mean(dim=1)
            # Edge reconstruction error. If edge_attr is continuous, MSE is appropriate.
            # If edge_attr was binary and BCE was used for training, then use BCE for error.
            # Assuming MSE for consistency with general pattern learning.
            # Ensure edge_recon has the correct shape for comparison with normalized_data.edge_attr
            if normalized_data.edge_attr is not None and normalized_data.edge_attr.numel() > 0:
                if self.edge_recon_loss_type == 'bce':
                    # For BCE, compare probability output (edge_recon) with a binary target (e.g., 0.8 for existing)
                    # This is for anomaly detection, so we want to see how far it is from the 'normal' target.
                    # Using abs difference from target 0.8 for anomaly scoring, not BCE loss.
                    edge_recon_error = torch.abs(edge_recon - 0.8)  # How far is it from the 'normal' probability
                else:  # mse or l1
                    edge_recon_error = F.mse_loss(edge_recon, normalized_data.edge_attr, reduction='none').mean(dim=1)
            else:
                edge_recon_error = torch.abs(edge_recon - 0.5)  # Default if no edge_attr

            logging.debug(f"Node reconstruction errors: {node_recon_error.cpu().numpy()}")
            logging.debug(f"Edge reconstruction errors: {edge_recon_error.cpu().numpy()}")

            # Statistical normalization of reconstruction errors
            # Use a small epsilon to prevent division by zero if std is zero
            node_mean_recon = node_recon_error.mean()
            node_std_recon = node_recon_error.std() + 1e-6
            edge_mean_recon = edge_recon_error.mean()
            edge_std_recon = edge_recon_error.std() + 1e-6

            # Identify anomalous nodes/edges based on reconstruction error
            anomalous_nodes = (node_recon_error > node_mean_recon + threshold * node_std_recon).nonzero(as_tuple=True)[
                0]
            anomalous_edges = (edge_recon_error > edge_mean_recon + threshold * edge_std_recon).nonzero(as_tuple=True)[
                0]

            # Also get anomaly scores from the dedicated MLPs (these scores are trained to be low for normal data)
            # Apply sigmoid to make them probability-like, though the training pushes them to zero.
            node_scores_mlp = torch.sigmoid(self.node_anomaly_mlp(embedding)).squeeze()

            # Handle None edge attributes for anomaly MLP input during inference
            src, dst = normalized_data.edge_index[0], normalized_data.edge_index[1]
            if normalized_data.edge_attr is not None:
                edge_features_for_anomaly_mlp_inference = torch.cat(
                    [embedding[src], embedding[dst], normalized_data.edge_attr], dim=1)
            else:
                device = embedding.device
                num_edges = normalized_data.edge_index.shape[1]
                edge_feature_dim_ae = getattr(self.autoencoder, 'edge_feature_dim', 0)
                if edge_feature_dim_ae > 0:
                    zero_edge_attr = torch.zeros(num_edges, edge_feature_dim_ae, device=device)
                    edge_features_for_anomaly_mlp_inference = torch.cat(
                        [embedding[src], embedding[dst], zero_edge_attr], dim=1)
                else:
                    edge_features_for_anomaly_mlp_inference = torch.cat([embedding[src], embedding[dst]], dim=1)
                    if self.edge_anomaly_mlp[0].in_features == embedding.shape[1] * 2 + 1:
                        edge_features_for_anomaly_mlp_inference = torch.cat(
                            [edge_features_for_anomaly_mlp_inference, torch.zeros(num_edges, 1, device=device)], dim=1)

            edge_scores_mlp = torch.sigmoid(self.edge_anomaly_mlp(edge_features_for_anomaly_mlp_inference)).squeeze()
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
        # Normalize the input data for embedding export
        normalized_data = self._normalize_features(data.clone())

        with torch.no_grad():
            embedding, _, _ = self.autoencoder(normalized_data)
            embedding_np = embedding.cpu().numpy()

            n_samples = embedding_np.shape[0]
            # Ensure perplexity is valid for t-SNE
            safe_perplexity = min(perplexity, max(5, n_samples - 1))
            if n_samples <= 1:
                logging.error(f"Error during embedding export: Not enough samples ({n_samples}) for t-SNE.")
                return

            logging.debug(f"Embedding standard deviations before t-SNE: {np.std(embedding_np, axis=0)}")
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

    def plot_reconstruction_heatmaps(self, data: Data, filename="reconstruction_heatmaps.png"):
        """
        Generates heatmaps of original and reconstructed node features.
        data: A single Data object.
        filename: Name of the file to save the heatmap.
        """
        self.eval()
        data = data.to(self.device)
        normalized_data = self._normalize_features(data.clone())

        with torch.no_grad():
            _, node_recon, edge_recon = self.autoencoder(normalized_data)

            # For node features
            original_nodes_np = normalized_data.x.cpu().numpy()
            reconstructed_nodes_np = node_recon.cpu().numpy()

            plt.figure(figsize=(16, 8))  # Increased figure size for better visibility

            plt.subplot(1, 2, 1)
            plt.imshow(original_nodes_np.T, aspect='auto', cmap='viridis')  # Transpose for features on Y-axis
            plt.colorbar(label='Feature Value')
            plt.title("Original Node Features")
            plt.xlabel("Node Index")
            plt.ylabel("Feature Dimension")

            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_nodes_np.T, aspect='auto', cmap='viridis')  # Transpose for features on Y-axis
            plt.colorbar(label='Feature Value')
            plt.title("Reconstructed Node Features")
            plt.xlabel("Node Index")
            plt.ylabel("Feature Dimension")

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            logging.info(f"Node reconstruction heatmaps saved to {filename}")

            # Optional: Add edge reconstruction heatmaps if relevant and edge_attr is not None
            if normalized_data.edge_attr is not None and normalized_data.edge_attr.numel() > 0:
                original_edges_np = normalized_data.edge_attr.cpu().numpy()
                # Ensure edge_recon matches shape for plotting
                if edge_recon.dim() == 1:  # If edge_recon is squeezed to 1D
                    reconstructed_edges_np = edge_recon.unsqueeze(1).cpu().numpy()
                else:
                    reconstructed_edges_np = edge_recon.cpu().numpy()

                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(original_edges_np.T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Feature Value')
                plt.title("Original Edge Features")
                plt.xlabel("Edge Index")
                plt.ylabel("Feature Dimension")

                plt.subplot(1, 2, 2)
                plt.imshow(reconstructed_edges_np.T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Feature Value')
                plt.title("Reconstructed Edge Features")
                plt.xlabel("Edge Index")
                plt.ylabel("Feature Dimension")

                plt.tight_layout()
                edge_filename = filename.replace(".png", "_edges.png")
                plt.savefig(edge_filename)
                plt.close()
                logging.info(f"Edge reconstruction heatmaps saved to {edge_filename}")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self.to(value)
