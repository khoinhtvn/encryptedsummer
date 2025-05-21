import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from collections import deque
import numpy as np

class RunningStats:
    """Utility for tracking feature statistics"""
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None

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

    def get_std(self):
        return np.sqrt(self.M2 / self.n) if self.n > 0 else None

class HybridGNNAnomalyDetector(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, heads=4):
        super(HybridGNNAnomalyDetector, self).__init__()
        # Graph Attention Network layers
        self.conv1 = GATConv(node_feature_dim, hidden_dim, heads=heads, edge_dim=edge_feature_dim)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_feature_dim)

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

        # Online learning components
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

        # Experience replay buffer (optional)
        self.replay_buffer = deque(maxlen=1000)
        self.batch_size = 32

        # Statistical controls
        self.node_stats = RunningStats()
        self.edge_stats = RunningStats()
        self.feature_mean = None
        self.feature_std = None
        self.drift_threshold = 0.15
        self.consecutive_drifts = 0

    def forward(self, data):
        # Process graph through GAT layers
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Node-level anomaly scores
        node_scores = self.node_mlp(x)

        # Edge-level anomaly scores
        src, dst = edge_index[0], edge_index[1]
        edge_features = torch.cat([x[src], x[dst], edge_attr], dim=1)
        edge_scores = self.edge_mlp(edge_features)

        # Global anomaly score
        global_score = self.global_mlp(global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device)))

        return node_scores, edge_scores, global_score

    def update_online(self, data, n_steps=5):
        """Hybrid online update with statistical controls"""
        self.train()

        if data is None:
            print("⚠️ ERRORE: `data` è None. Salto update_online.")
            return 0.0

        # Update running statistics
        self.node_stats.update(data.x)
        self.edge_stats.update(data.edge_attr)

        # Add current data to replay buffer
        self._add_to_replay_buffer(data)

        total_loss = 0.0
        valid_steps = 0

        for _ in range(n_steps):
            if len(self.replay_buffer) >= self.batch_size:
                batch = self._sample_from_replay_buffer()
            else:
                batch = data  # fallback: usa data direttamente se la buffer è vuota

            if batch is None:
                print("⚠️ Nessun batch disponibile. Salto questo step.")
                continue

            self.optimizer.zero_grad()
            try:
                node_scores, edge_scores, _ = self(batch)
            except AttributeError as e:
                print(f"⚠️ Errore durante il forward: {e}")
                continue

            loss = self._compute_loss(node_scores, edge_scores)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            valid_steps += 1

        # Update learning rate solo se almeno un passo è valido
        if valid_steps > 0:
            avg_loss = total_loss / valid_steps
            self.scheduler.step(avg_loss)
        else:
            avg_loss = 0.0

        # Drift detection
        self.detect_feature_drift(data)

        return avg_loss


    def _compute_loss(self, node_scores, edge_scores):
        """Enhanced loss with statistical regularization"""
        base_loss = (node_scores.abs().mean() + edge_scores.abs().mean()) / 2

        # Add regularization based on running statistics
        if self.node_stats.n > 10 and self.edge_stats.n > 10:
            node_std = torch.tensor(self.node_stats.get_std(), device=node_scores.device)
            edge_std = torch.tensor(self.edge_stats.get_std(), device=edge_scores.device)

            # Penalize scores that deviate significantly from historical norms
            reg_loss = (node_scores.pow(2) / (node_std + 1e-6).pow(2)).mean() + \
                      (edge_scores.pow(2) / (edge_std + 1e-6).pow(2)).mean()

            return 0.7 * base_loss + 0.3 * reg_loss
        return base_loss

    def detect_feature_drift(self, data):
        """Statistical drift detection"""
        current_node_mean = data.x.mean(dim=0)
        current_edge_mean = data.edge_attr.mean(dim=0)

        if self.feature_mean is not None:
            node_diff = (current_node_mean - self.feature_mean[0]).abs() / (self.feature_std[0] + 1e-6)
            edge_diff = (current_edge_mean - self.feature_mean[1]).abs() / (self.feature_std[1] + 1e-6)

            if node_diff.max() > self.drift_threshold or edge_diff.max() > self.drift_threshold:
                self.consecutive_drifts += 1
                print(f"Feature drift detected ({self.consecutive_drifts} consecutive)")

                if self.consecutive_drifts >= 3:
                    self._adaptive_reset()
            else:
                self.consecutive_drifts = 0

        # Update reference statistics
        self.feature_mean = (current_node_mean, current_edge_mean)
        self.feature_std = (data.x.std(dim=0), data.edge_attr.std(dim=0))

    def _adaptive_reset(self):
        """Partial model reset for major behavior changes"""
        print("Major behavior change - performing adaptive reset")

        # Reset only the anomaly scoring heads
        for layer in [self.node_mlp, self.edge_mlp, self.global_mlp]:
            for p in layer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        # Reset optimizer state
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.consecutive_drifts = 0

    def _add_to_replay_buffer(self, data):
        """Optional: Add subsampled graph to replay buffer"""
        sampled_nodes = torch.randperm(data.x.size(0))[:min(100, data.x.size(0))]
        sampled_edges = torch.randperm(data.edge_index.size(1))[:min(200, data.edge_index.size(1))]

        subgraph = Data(
            x=data.x[sampled_nodes],
            edge_index=data.edge_index[:, sampled_edges],
            edge_attr=data.edge_attr[sampled_edges]
        )
        self.replay_buffer.append(subgraph)

    def _sample_from_replay_buffer(self):
        """Sample a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        return Batch.from_data_list([self.replay_buffer[i] for i in indices])

    def detect_anomalies(self, data, threshold=2.5):
        """Detect anomalies with statistical normalization"""
        self.eval()
        with torch.no_grad():
            node_scores, edge_scores, global_score = self(data)

            # Convert to probabilities
            node_probs = torch.sigmoid(node_scores).squeeze()
            edge_probs = torch.sigmoid(edge_scores).squeeze()
            global_prob = torch.sigmoid(global_score).item()

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