import logging
import os
import traceback
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


def visualize_node_features(data: Data, feature_names: list = None, save_path: str = None):
    """
    Visualizes node features with appropriate scales and optimized layout

    Args:
        data: PyTorch Geometric Data object
        feature_names: List of names for the features
    """
    logging.info("Starting visualization of node features.")

    # Data extraction and preparation
    if not hasattr(data, 'x') or data.x is None:
        logging.warning("No features found in the graph.")
        return

    node_features = data.x.numpy() if data.x.is_cuda else data.x.detach().numpy()
    num_nodes, num_features = node_features.shape
    logging.info(f"Graph has {num_nodes} nodes and {num_features} features.")

    # Feature names
    if feature_names is not None:
        if len(feature_names) != num_features:
            logging.error(f"Expected {num_features} feature names, but got {len(feature_names)}.")
            raise ValueError(f"Expected {num_features} names, got {len(feature_names)}")
        use_names = feature_names
    else:
        use_names = [f'Feature {i}' for i in range(num_features)]
    logging.info(f"Feature names: {', '.join(use_names)}")

    # Quick statistics calculation
    logging.info("Calculating feature statistics:")
    stats = []
    for i in range(num_features):
        unique_vals = np.unique(node_features[:, i])
        stat = {
            'name': use_names[i],
            'unique': len(unique_vals),
            'mean': np.mean(node_features[:, i]),
            'std': np.std(node_features[:, i]),
            'min': np.min(node_features[:, i]),
            'max': np.max(node_features[:, i])
        }
        stats.append(stat)
        logging.info(f"- {use_names[i]}: {stat}")

    # Creating figure with optimized layout
    plt.figure(figsize=(18, 12))
    plt.suptitle('Node Feature Analysis', y=1.02, fontsize=16)

    # 1. Distribution of activity_score (if present)
    if 'activity_score' in use_names:
        idx = use_names.index('activity_score')
        plt.subplot(2, 3, 1)
        sns.kdeplot(node_features[:, idx], label='activity_score', bw_method=0.15, fill=True)
        plt.title('Activity Score Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        logging.debug("Plotted distribution of activity_score.")

    # 2. Degree distributions (logarithmic scale)
    degree_features = []
    for name in ['degree', 'in_degree', 'out_degree', 'total_connections']:
        if name in use_names:
            degree_features.append(use_names.index(name))

    if degree_features:
        plt.subplot(2, 3, 2)
        for i in degree_features:
            log_vals = np.log1p(node_features[:, i])
            sns.kdeplot(log_vals, label=use_names[i], bw_method=0.3, fill=True)

        plt.title('Degree Metric Distributions\n(logarithmic scale)')
        plt.xlabel('log(value + 1)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        logging.debug("Plotted distributions of degree metrics (log scale).")

    # 3. Boxplot of degree features (log)
    if degree_features:
        plt.subplot(2, 3, 3)
        log_data = np.log1p(node_features[:, degree_features])
        sns.boxplot(data=pd.DataFrame(log_data, columns=[use_names[i] for i in degree_features]))
        plt.title('Comparison of Degree Metrics')
        plt.xticks(rotation=45)
        plt.ylabel('log(value + 1)')
        logging.debug("Plotted boxplot of degree metrics (log scale).")

    # 4. Correlation matrix
    plt.subplot(2, 3, 4)
    corr = np.corrcoef(node_features.T)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                xticklabels=use_names, yticklabels=use_names,
                fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation Matrix')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    logging.debug("Plotted correlation matrix of node features.")

    # 5. Feature value ranges
    plt.subplot(2, 3, 5)
    for i, feat in enumerate(stats):
        plt.errorbar(i, feat['mean'], yerr=feat['std'], fmt='o', color='blue')
        plt.plot(i, feat['min'], 'v', color='red')
        plt.plot(i, feat['max'], '^', color='green')

    plt.xticks(range(num_features), use_names, rotation=45)
    plt.title('Feature Value Ranges')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    logging.debug("Plotted ranges of node feature values.")

    # 6. Feature sparsity
    plt.subplot(2, 3, 6)
    sparsity = np.mean(node_features == 0, axis=0)
    plt.bar(range(num_features), sparsity)
    plt.xticks(range(num_features), use_names, rotation=45)
    plt.title('Feature Sparsity (Fraction of Zeros)')
    plt.ylabel('Fraction of zeros')
    plt.axhline(np.mean(sparsity), color='r', linestyle='--',
                label=f'Mean: {np.mean(sparsity):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    logging.debug("Plotted sparsity of node features.")

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"node_features_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300)
        logging.info(f"Node feature visualization saved to {filepath}")
        plt.close()
    else:
        plt.show()

    logging.info("Finished visualization of node features.")

    # Additional analysis
    logging.info("Performing additional feature analysis:")
    if degree_features:
        logging.info("- Degree metrics are visualized on a logarithmic scale.")

    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    max_corr = np.max(abs_corr)
    logging.info(f"- Maximum correlation between different features: {max_corr:.2f}")

    if 'activity_score' in use_names and degree_features:
        activity_idx = use_names.index('activity_score')
        corrs_with_activity = corr[activity_idx, degree_features]
        logging.info("- Correlations between activity_score and degree metrics:")
        for i, val in enumerate(corrs_with_activity):
            logging.info(f"  {use_names[degree_features[i]]}: {val:.2f}")


def visualize_edge_features(data: Data, edge_feature_names: list = None, max_features_to_plot=12,
                            save_path: str = None):
    """
    Visualizes edge features with techniques to handle many features

    Args:
        data: PyTorch Geometric Data object
        edge_feature_names: List of names for the edge features
        max_features_to_plot: Maximum number of features to visualize
        save_path: Path to save the visualization plot(s). If None, shows the plot.
    """
    logging.info("Starting visualization of edge features.")

    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        logging.warning("No edge features found in the graph.")
        return

    edge_features = data.edge_attr.numpy() if data.edge_attr.is_cuda else data.edge_attr.detach().numpy()
    num_edges, num_features = edge_features.shape
    logging.info(f"Graph has {num_edges} edges and {num_features} edge features.")

    # Feature names
    if edge_feature_names is not None:
        if len(edge_feature_names) != num_features:
            logging.error(f"Expected {num_features} edge feature names, but got {len(edge_feature_names)}.")
            raise ValueError(f"Expected {num_features} names, got {len(edge_feature_names)}")
        use_names = edge_feature_names
    else:
        use_names = [f'EdgeFeat_{i}' for i in range(num_features)]
    logging.info(f"Edge feature names: {', '.join(use_names)}")

    # Quick statistics calculation
    logging.info("Calculating edge feature statistics:")
    stats = []
    for i in range(num_features):
        unique_vals = np.unique(edge_features[:, i])
        stat = {
            'name': use_names[i],
            'unique': len(unique_vals),
            'mean': np.mean(edge_features[:, i]),
            'std': np.std(edge_features[:, i]),
            'min': np.min(edge_features[:, i]),
            'max': np.max(edge_features[:, i])
        }
        stats.append(stat)

    # Print summary statistics for a subset of features
    num_stats_to_show = 5
    logging.info(f"Showing statistics for the first {num_stats_to_show} and last {num_stats_to_show} edge features:")
    for stat in stats[:num_stats_to_show]:
        logging.info(f"- {stat['name']}: {stat}")
    if num_features > 2 * num_stats_to_show:
        logging.info(f"... omitted {num_features - 2 * num_stats_to_show} intermediate features ...")
    for stat in stats[-num_stats_to_show:]:
        logging.info(f"- {stat['name']}: {stat}")

    # --- Visualization ---
    num_rows = 3
    num_cols = 2
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'Edge Feature Analysis ({num_features} total)', y=1.02, fontsize=16)

    # 1. Heatmap of a subset of features
    plt.subplot(num_rows, num_cols, 1)
    features_to_show_heatmap = min(max_features_to_plot, num_features)
    sns.heatmap(edge_features[:, :features_to_show_heatmap].T,
                cmap='viridis',
                yticklabels=use_names[:features_to_show_heatmap],
                cbar_kws={'label': 'Feature Value'})
    plt.title(f'Heatmap of First {features_to_show_heatmap} Edge Features')
    plt.xlabel('Edges')
    plt.ylabel('Feature')
    logging.debug(f"Plotted heatmap of the first {features_to_show_heatmap} edge features.")

    # 2. Correlation matrix of non-constant features
    plt.subplot(num_rows, num_cols, 2)
    non_constant_indices = [i for i, stat in enumerate(stats) if stat['unique'] > 1]
    if len(non_constant_indices) > 1:
        corr = np.corrcoef(edge_features[:, non_constant_indices].T)
        non_constant_names = [use_names[i] for i in non_constant_indices]
        sns.heatmap(corr, cmap='coolwarm', center=0,
                    xticklabels=non_constant_names, yticklabels=non_constant_names,
                    annot=False)
        plt.title('Correlation Matrix\n(Non-constant features)')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        logging.debug("Plotted correlation matrix of non-constant edge features.")
    else:
        plt.text(0.5, 0.5, 'Too few non-constant features\nfor correlation matrix',
                 ha='center', va='center')
        plt.axis('off')
        logging.warning("Too few non-constant edge features for correlation matrix.")

    # 3. Distributions of most variable features
    plt.subplot(num_rows, num_cols, 3)
    stds = [stat['std'] for stat in stats]
    most_variable_indices = np.argsort(stds)[::-1][:max_features_to_plot]
    for idx in most_variable_indices:
        if stats[idx]['unique'] > 1:
            sns.kdeplot(edge_features[:, idx], label=use_names[idx], fill=True, alpha=0.5)
    plt.title(f'Distributions of Top {min(max_features_to_plot, num_features)} Variable Features')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    logging.debug(f"Plotted distributions of the top {min(max_features_to_plot, num_features)} variable edge features.")

    # 4. Boxplots of a subset of non-constant features
    plt.subplot(num_rows, num_cols, 4)
    non_constant_to_boxplot = non_constant_indices[:max_features_to_plot]
    if non_constant_to_boxplot:
        boxplot_data = pd.DataFrame(edge_features[:, non_constant_to_boxplot],
                                    columns=[use_names[i] for i in non_constant_to_boxplot])
        sns.boxplot(data=boxplot_data)
        plt.title(f'Boxplots of First {len(non_constant_to_boxplot)} Non-Constant Features')
        plt.xticks(rotation=90)
        plt.ylabel('Value')
        logging.debug(f"Plotted boxplots of the first {len(non_constant_to_boxplot)} non-constant edge features.")
    else:
        plt.text(0.5, 0.5, 'No non-constant features\nfor boxplots',
                 ha='center', va='center')
        plt.axis('off')
        logging.warning("No non-constant edge features for boxplots.")

    # 5. Sparsity analysis
    plt.subplot(num_rows, num_cols, 5)
    sparsity = np.mean(edge_features == 0, axis=0)
    plt.bar(range(num_features), sparsity)
    plt.xticks(range(num_features), use_names, rotation=90)
    plt.title('Edge Feature Sparsity\n(Fraction of Zeros)')
    plt.ylabel('Fraction of Zeros')
    plt.axhline(np.mean(sparsity), color='r', linestyle='--',
                label=f'Mean: {np.mean(sparsity):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    logging.debug("Plotted sparsity of edge features.")

    # 6. PCA for dimensionality reduction (if enough non-constant features)
    plt.subplot(num_rows, num_cols, 6)
    if len(non_constant_indices) >= 2:
        X = StandardScaler().fit_transform(edge_features[:, non_constant_indices])
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)
        explained_variance_ratio = pca.explained_variance_ratio_
        plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
        plt.title(f'PCA (2 Components)\nExplained: {np.sum(explained_variance_ratio):.2f}')
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2f})')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2f})')
        plt.grid(True, alpha=0.3)
        logging.debug("Applied PCA for dimensionality reduction of non-constant edge features.")
    else:
        plt.text(0.5, 0.5, 'Not enough non-constant features\nfor PCA (at least 2 required)',
                 ha='center', va='center')
        plt.axis('off')
        logging.warning("Not enough non-constant edge features for PCA.")

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"edge_features_{timestamp}.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300)
        logging.info(f"Edge feature visualization saved to {filepath}")
        plt.close()
    else:
        plt.show()

    logging.info("Finished visualization of edge features.")

    # --- Additional Analysis ---
    logging.info("Performing additional edge feature analysis:")
    num_constant = num_features - len(non_constant_indices)
    if num_constant > 0:
        logging.info(f"- {num_constant} out of {num_features} edge features are constant.")

    if len(non_constant_indices) > 1:
        corr_non_constant = np.corrcoef(edge_features[:, non_constant_indices].T)
        abs_corr_non_constant = np.abs(corr_non_constant)
        np.fill_diagonal(abs_corr_non_constant, 0)
        max_corr_val = np.max(abs_corr_non_constant)
        max_corr_idx = np.unravel_index(np.argmax(abs_corr_non_constant), abs_corr_non_constant.shape)
        if max_corr_val > 0.8:
            feature1_name = use_names[non_constant_indices[max_corr_idx[0]]]
            feature2_name = use_names[non_constant_indices[max_corr_idx[1]]]
            logging.info(
                f"- Highest correlation ({max_corr_val:.2f}) between features: '{feature1_name}' and '{feature2_name}'.")
        else:
            logging.info(f"- Maximum correlation between different non-constant edge features: {max_corr_val:.2f}.")
    else:
        logging.info("- Less than two non-constant edge features, skipping correlation analysis.")


def visualize_all_edge_features(data: Data, edge_feature_names: list = None, save_path: str = None):
    """
    Visualizes all edge features using summary statistics and 3D PCA.

    Args:
        data: PyTorch Geometric Data object
        edge_feature_names: List of names for the edge features
        save_path: Path to save the visualization plot(s). If None, shows the plot.
    """
    logging.info("Starting visualization of all edge features with 3D PCA.")

    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        logging.warning("No edge features found in the graph.")
        return

    edge_features = data.edge_attr.numpy() if data.edge_attr.is_cuda else data.edge_attr.detach().numpy()
    num_edges, num_features = edge_features.shape
    logging.info(f"Graph has {num_edges} edges and {num_features} edge features.")

    # Feature names
    if edge_feature_names is not None:
        if len(edge_feature_names) != num_features:
            logging.error(f"Expected {num_features} edge feature names, but got {len(edge_feature_names)}.")
            raise ValueError(f"Expected {num_features} names, got {len(edge_feature_names)}")
        use_names = edge_feature_names
    else:
        use_names = [f'EdgeFeat_{i}' for i in range(num_features)]
    logging.info(f"Edge feature names: {', '.join(use_names)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"all_edge_features_3d_pca_{timestamp}"
    os.makedirs(save_path, exist_ok=True) if save_path else None

    # 1. Summary Statistics (remains the same)
    plt.figure(figsize=(12, 6))
    stats_df = pd.DataFrame({
        'Feature': use_names,
        'Mean': np.mean(edge_features, axis=0),
        'Std': np.std(edge_features, axis=0),
        'Min': np.min(edge_features, axis=0),
        'Max': np.max(edge_features, axis=0),
        'Sparsity (%)': np.mean(edge_features == 0, axis=0) * 100
    })
    stats_df_melted = stats_df.melt(id_vars='Feature', var_name='Statistic', value_name='Value')
    sns.barplot(data=stats_df_melted, x='Feature', y='Value', hue='Statistic')
    plt.xticks([])  # Remove x-axis labels if too many features
    plt.title('Summary Statistics of All Edge Features')
    plt.ylabel('Value / Sparsity (%)')
    plt.tight_layout()
    if save_path:
        filepath = os.path.join(save_path, f"{base_filename}_summary.png")
        plt.savefig(filepath, dpi=300)
        logging.info(f"Summary statistics plot saved to {filepath}")
        plt.close()
    else:
        plt.show()

    # 2. Dimensionality Reduction (PCA to 3D)
    if num_features > 3:
        plt.figure(figsize=(10, 8))
        ax = plt.figure().add_subplot(111, projection='3d')
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(edge_features)
        pca = PCA(n_components=3)
        principal_components = pca.fit_transform(scaled_features)

        ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], alpha=0.5)
        ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f})')
        ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f})')
        ax.set_zlabel(f'Principal Component 3 ({pca.explained_variance_ratio_[2]:.2f})')
        ax.set_title('PCA of Edge Features (3 Components)')
        plt.tight_layout()
        if save_path:
            filepath = os.path.join(save_path, f"{base_filename}_pca_3d.png")
            plt.savefig(filepath, dpi=300)
            plt.close()
            logging.info(f"PCA (3D) plot saved to {filepath}")
        else:
            plt.show()

        # Optional: PCA Explained Variance Ratio (remains the same)
        plt.figure(figsize=(10, 6))
        explained_variance_ratio = pca.explained_variance_ratio_
        plt.bar(range(pca.n_components_), explained_variance_ratio)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.xticks(range(pca.n_components_))
        plt.tight_layout()
        if save_path:
            filepath = os.path.join(save_path, f"{base_filename}_pca_variance.png")
            plt.savefig(filepath, dpi=300)
            plt.close()
            logging.info(f"PCA explained variance ratio plot saved to {filepath}")
        else:
            plt.show()

    elif num_features <= 3:
        logging.warning("Number of edge features is too small for PCA to 3D.")

    logging.info("Finished visualization of all edge features with 3D PCA.")

# Example usage (assuming you have a PyTorch Geometric Data object named 'data'):
# visualize_all_edge_features_3d_pca(data, save_path='./edge_feature_plots')
def visualize_embeddings_3d(embeddings, node_ips, anomalous_indices_tensor,
                            save_path, timestamp=None, filename_prefix="embeddings_3d"):
    """
    Visualizes node embeddings in 3D using t-SNE, highlighting anomalous nodes,
    and includes timestamp in filename.

    Args:
        embeddings (np.ndarray): The node embeddings to visualize, shape (n_samples, n_features).
        node_ips (list): List of IP addresses (or node identifiers) corresponding
                         to the node indices in 'embeddings'. Used for total count.
        anomalous_indices_tensor (torch.Tensor or np.ndarray): Tensor/array containing
                                indices (0-based) of anomalous nodes.
        save_path (str): The directory path where the visualization image will be saved.
        timestamp (str, optional): A specific timestamp string (e.g., "20250606_173000").
                                   If None, the current UTC time will be used.
        filename_prefix (str): Prefix for the saved filename (e.g., "embeddings_3d").
    """
    logging.info("Starting 3D embedding visualization...")

    if timestamp is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_UTC")
    else:
        timestamp_str = str(timestamp) # Ensure it's a string for filename

    n_samples = embeddings.shape[0]

    # --- Perplexity Adjustment for t-SNE ---
    # t-SNE requires perplexity < n_samples. Minimum samples for t-SNE is usually 5.
    default_perplexity = 30
    perplexity_val = default_perplexity

    if n_samples < 5:
        logging.warning(f"Not enough samples ({n_samples}) for meaningful t-SNE. Skipping 3D embedding visualization.")
        return
    elif n_samples <= default_perplexity:
        # If n_samples is small, set perplexity to n_samples - 1 (must be > 0)
        perplexity_val = max(1, n_samples - 1)
        logging.warning(
            f"Number of samples ({n_samples}) is small. Adjusting t-SNE perplexity to {perplexity_val}."
        )
    # No 'else' needed, as perplexity_val is already default_perplexity

    try:
        # Initialize t-SNE. Use 'max_iter' instead of 'n_iter' to avoid FutureWarning.
        tsne = TSNE(n_components=3, random_state=42, max_iter=500, perplexity=perplexity_val)
        reduced_embeddings = tsne.fit_transform(embeddings)

        # --- Prepare Plot Labels (Normal vs. Anomalous) ---
        # Ensure anomalous_indices_tensor is a NumPy array for consistent indexing
        anomalous_indices_list = []
        if isinstance(anomalous_indices_tensor, torch.Tensor):
            anomalous_indices_list = anomalous_indices_tensor.cpu().numpy()
        elif isinstance(anomalous_indices_tensor, np.ndarray):
            anomalous_indices_list = anomalous_indices_tensor
        elif anomalous_indices_tensor is not None: # Handle cases where it might be None
            logging.warning(
                f"Unexpected type for anomalous_indices_tensor: {type(anomalous_indices_tensor)}. "
                "Expected torch.Tensor or numpy.ndarray. No anomalies will be highlighted."
            )

        # Initialize all nodes as normal (0)
        plot_labels = np.zeros(len(node_ips), dtype=int) # Use len(node_ips) for total nodes
        # Mark anomalous nodes with label 1
        for idx in anomalous_indices_list:
            if 0 <= idx < len(plot_labels):
                plot_labels[idx] = 1 # Mark as anomalous
            else:
                logging.warning(f"Anomalous index {idx} out of bounds for node_ips length {len(node_ips)}.")

        # --- Set up the 3D plot ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Define colors: Blue for normal, Red for anomalous
        colors = ['blue', 'red']
        cmap = matplotlib.colors.ListedColormap(colors)
        bounds = [0, 1, 2] # Boundaries for the colormap (0 -> blue, 1 -> red)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # Create the 3D scatter plot
        scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2],
                             c=plot_labels, cmap=cmap, norm=norm, alpha=0.7, s=50) # 's' for marker size

        # --- Set plot title and axis labels ---
        ax.set_title(f"Node Embeddings Visualization (t-SNE 3D)\nGraph Timestamp: {timestamp_str}")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")

        # --- Create Custom Legend ---
        from matplotlib.lines import Line2D # Import here to avoid potential circular dependencies if at top
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal Node',
                   markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Anomalous Node',
                   markerfacecolor='red', markersize=10)
        ]
        ax.legend(handles=legend_elements, title="Node Type", loc='upper left')

        # --- Save the plot to a file ---
        os.makedirs(save_path, exist_ok=True) # Ensure save directory exists
        filepath = os.path.join(save_path, f"{filename_prefix}_{timestamp_str}.png")

        plt.savefig(filepath, dpi=300) # Save with higher DPI for better quality
        logging.info(f"3D embeddings visualization saved to {filepath}")
        plt.close(fig) # Close the plot to free up memory

    except Exception as e:
        logging.error(f"Error during 3D t-SNE or plotting: {e}")
        logging.error(traceback.format_exc()) # Log the full traceback for detailed debugging
