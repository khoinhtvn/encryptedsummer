import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from torch_geometric.data import Data


def visualize_node_features(data: Data, feature_names: list = None):
    """
    Visualizza le feature dei nodi con scale appropriate e layout ottimizzato

    Args:
        data: PyTorch Geometric Data object
        feature_names: Lista di nomi per le feature
    """
    logging.info("Starting visualization of node features.")

    # Estrazione e preparazione dati
    if not hasattr(data, 'x') or data.x is None:
        logging.warning("Nessuna feature trovata nel grafo.")
        return

    node_features = data.x.numpy() if data.x.is_cuda else data.x.detach().numpy()
    num_nodes, num_features = node_features.shape
    logging.info(f"Graph has {num_nodes} nodes and {num_features} features.")

    # Nomi delle feature
    if feature_names is not None:
        if len(feature_names) != num_features:
            logging.error(f"Expected {num_features} feature names, but got {len(feature_names)}.")
            raise ValueError(f"Attesi {num_features} nomi, ottenuti {len(feature_names)}")
        use_names = feature_names
    else:
        use_names = [f'Feature {i}' for i in range(num_features)]
    logging.info(f"Feature names: {', '.join(use_names)}")

    # Calcolo statistiche rapide
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

    # Creazione figura con layout ottimizzato
    plt.figure(figsize=(18, 12))
    plt.suptitle('Analisi Feature dei Nodi', y=1.02, fontsize=16)

    # 1. Distribuzione activity_score (se presente)
    if 'activity_score' in use_names:
        idx = use_names.index('activity_score')
        plt.subplot(2, 3, 1)
        sns.kdeplot(node_features[:, idx], label='activity_score', bw_method=0.15, fill=True)
        plt.title('Distribuzione Activity Score')
        plt.xlabel('Valore')
        plt.ylabel('Densità')
        plt.grid(True, alpha=0.3)
        logging.debug("Plotted distribution of activity_score.")

    # 2. Distribuzioni di grado (scala logaritmica)
    degree_features = []
    for name in ['degree', 'in_degree', 'out_degree', 'total_connections']:
        if name in use_names:
            degree_features.append(use_names.index(name))

    if degree_features:
        plt.subplot(2, 3, 2)
        for i in degree_features:
            log_vals = np.log1p(node_features[:, i])
            sns.kdeplot(log_vals, label=use_names[i], bw_method=0.3, fill=True)

        plt.title('Distribuzioni Metriche di Grado\n(scala logaritmica)')
        plt.xlabel('log(valore + 1)')
        plt.ylabel('Densità')
        plt.legend()
        plt.grid(True, alpha=0.3)
        logging.debug("Plotted distributions of degree metrics (log scale).")

    # 3. Boxplot features di grado (log)
    if degree_features:
        plt.subplot(2, 3, 3)
        log_data = np.log1p(node_features[:, degree_features])
        sns.boxplot(data=pd.DataFrame(log_data, columns=[use_names[i] for i in degree_features]))
        plt.title('Confronto Metriche di Grado')
        plt.xticks(rotation=45)
        plt.ylabel('log(valore + 1)')
        logging.debug("Plotted boxplot of degree metrics (log scale).")

    # 4. Matrice di correlazione
    plt.subplot(2, 3, 4)
    corr = np.corrcoef(node_features.T)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                xticklabels=use_names, yticklabels=use_names,
                fmt=".2f", annot_kws={"size": 10})
    plt.title('Matrice di Correlazione')
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
    plt.title('Range Valori Features')
    plt.ylabel('Valore')
    plt.grid(True, alpha=0.3)
    logging.debug("Plotted ranges of node feature values.")

    # 6. Feature sparsity
    plt.subplot(2, 3, 6)
    sparsity = np.mean(node_features == 0, axis=0)
    plt.bar(range(num_features), sparsity)
    plt.xticks(range(num_features), use_names, rotation=45)
    plt.title('Sparsità Features (Frazione Zeri)')
    plt.ylabel('Frazione di zeri')
    plt.axhline(np.mean(sparsity), color='r', linestyle='--',
                label=f'Media: {np.mean(sparsity):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    logging.debug("Plotted sparsity of node features.")

    plt.tight_layout()
    plt.show()
    logging.info("Finished visualization of node features.")

    # Analisi aggiuntiva
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


def visualize_edge_features(data: Data, edge_feature_names: list = None, max_features_to_plot=12):
    """
    Visualizza le edge features con tecniche per gestire molte feature

    Args:
        data: PyTorch Geometric Data object
        edge_feature_names: Lista di nomi per le edge features
        max_features_to_plot: Numero massimo di feature da visualizzare
    """
    logging.info("Starting visualization of edge features.")

    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        logging.warning("Nessuna edge feature trovata nel grafo.")
        return

    edge_features = data.edge_attr.numpy() if data.edge_attr.is_cuda else data.edge_attr.detach().numpy()
    num_edges, num_features = edge_features.shape
    logging.info(f"Graph has {num_edges} edges and {num_features} edge features.")

    # Nomi delle feature
    if edge_feature_names is not None:
        if len(edge_feature_names) != num_features:
            logging.error(f"Expected {num_features} edge feature names, but got {len(edge_feature_names)}.")
            raise ValueError(f"Attesi {num_features} nomi, ottenuti {len(edge_feature_names)}")
        use_names = edge_feature_names
    else:
        use_names = [f'EdgeFeat_{i}' for i in range(num_features)]
    logging.info(f"Edge feature names: {', '.join(use_names)}")

    # Calcolo statistiche rapide
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

    # Stampiamo solo le prime e ultime 5 feature per non inondare l'output
    for stat in stats[:5] + stats[-5:]:
        logging.info(f"- {stat['name']}: {stat}")
    if num_features > 10:
        logging.info(f"... omesse {num_features - 10} features intermedie ...")

    # Creazione figura con layout ottimizzato per molte feature
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'Analisi Edge Features ({num_features} totali)', y=1.02, fontsize=16)

    # 1. Heatmap delle prime N feature (per vedere pattern)
    plt.subplot(3, 2, 1)
    features_to_show = min(max_features_to_plot, num_features)
    sns.heatmap(edge_features[:, :features_to_show].T,
                cmap='viridis',
                yticklabels=use_names[:features_to_show],
                cbar_kws={'label': 'Valore feature'})
    plt.title(f'Heatmap Prime {features_to_show} Edge Features')
    plt.xlabel('Archi')
    plt.ylabel('Feature')
    logging.debug(f"Plotted heatmap of the first {features_to_show} edge features.")

    # 2. Matrice di correlazione (solo per feature non costanti)
    plt.subplot(3, 2, 2)
    non_constant_features = [i for i in range(num_features) if stats[i]['unique'] > 1]
    if len(non_constant_features) > 1:
        corr = np.corrcoef(edge_features[:, non_constant_features].T)
        sns.heatmap(corr,
                    cmap='coolwarm',
                    center=0,
                    xticklabels=[use_names[i] for i in non_constant_features],
                    yticklabels=[use_names[i] for i in non_constant_features],
                    annot=False)
        plt.title('Matrice di Correlazione\n(Solo feature non costanti)')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        logging.debug("Plotted correlation matrix of non-constant edge features.")
    else:
        plt.text(0.5, 0.5, 'Troppe feature costanti\nper la matrice di correlazione',
                 ha='center', va='center')
        plt.axis('off')
        logging.warning("Too many constant edge features for correlation matrix.")

    # 3. Distribuzioni delle feature più interessanti
    plt.subplot(3, 2, 3)

    # Seleziona le feature più informative (basato su deviazione standard)
    stds = [s['std'] for s in stats]
    if np.sum(stds) > 0:  # Se almeno una feature non è costante
        # Prendiamo le feature con deviazione standard più alta
        most_variable = np.argsort(stds)[-max_features_to_plot:]

        for i in most_variable:
            if stats[i]['unique'] > 1:  # Solo feature non costanti
                # Normalizziamo per visualizzare insieme
                feat = edge_features[:, i]
                if stats[i]['unique'] > 10:  # Feature continua
                    sns.kdeplot(feat, label=use_names[i], bw_method=0.3)
                else:  # Feature discreta
                    sns.histplot(feat, label=use_names[i], discrete=True)

        plt.title(f'Distribuzioni Feature più Variabili\n({max_features_to_plot} su {num_features})')
        plt.xlabel('Valore')
        plt.ylabel('Densità/Conteggio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        logging.debug(f"Plotted distributions of the {max_features_to_plot} most variable edge features.")
    else:
        plt.text(0.5, 0.5, 'Tutte le feature sono costanti!',
                 ha='center', va='center')
        plt.axis('off')
        logging.warning("All edge features are constant.")

    # 4. Boxplot delle feature più importanti
    plt.subplot(3, 2, 4)
    if len(non_constant_features) > 0:
        features_to_boxplot = non_constant_features[:max_features_to_plot]
        boxplot_data = edge_features[:, features_to_boxplot]

        sns.boxplot(data=pd.DataFrame(boxplot_data,
                                      columns=[use_names[i] for i in features_to_boxplot]))
        plt.title(f'Boxplot Feature non Costanti\n(Prime {len(features_to_boxplot)})')
        plt.xticks(rotation=90)
        plt.ylabel('Valore')
        logging.debug(f"Plotted boxplot of the first {len(features_to_boxplot)} non-constant edge features.")
    else:
        plt.text(0.5, 0.5, 'Nessuna feature non costante\nper il boxplot',
                 ha='center', va='center')
        plt.axis('off')
        logging.warning("No non-constant edge features for boxplot.")

    # 5. Analisi sparsità
    plt.subplot(3, 2, 5)
    sparsity = np.mean(edge_features == 0, axis=0)
    plt.bar(range(num_features), sparsity)
    plt.xticks(range(num_features), use_names, rotation=90)
    plt.title('Sparsità Edge Features\n(Frazione di zeri)')
    plt.ylabel('Frazione di zeri')
    plt.axhline(np.mean(sparsity), color='r', linestyle='--',
                label=f'Media: {np.mean(sparsity):.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    logging.debug("Plotted sparsity of edge features.")

    # 6. PCA per riduzione dimensionalità (se molte feature non costanti)
    plt.subplot(3, 2, 6)
    if len(non_constant_features) > 3:
        # Prendiamo solo feature non costanti
        X = edge_features[:, non_constant_features]
        X = StandardScaler().fit_transform(X)

        # PCA con 2 componenti
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)

        plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
        plt.title('Riduzione Dimensionalità (PCA)\nSpiegato: {:.1f}%'.format(100 * pca.explained_variance_ratio_.sum()))
        plt.xlabel('Prima Componente ({:.1f}%)'.format(100 * pca.explained_variance_ratio_[0]))
        plt.ylabel('Seconda Componente ({:.1f}%)'.format(100 * pca.explained_variance_ratio_[1]))
        plt.grid(True, alpha=0.3)
        logging.debug("Applied PCA for dimensionality reduction of non-constant edge features.")
    else:
        plt.text(0.5, 0.5, 'Troppe poche feature non costanti\nper PCA',
                 ha='center', va='center')
        plt.axis('off')
        logging.warning("Too few non-constant edge features for PCA.")

    plt.tight_layout()
    plt.show()
    logging.info("Finished visualization of edge features.")

    # Analisi aggiuntiva
    logging.info("Performing additional edge feature analysis:")
    if len(non_constant_features) < num_features:
        logging.info(f"- {num_features - len(non_constant_features)} edge features are constant.")

    if len(non_constant_features) > 1:
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0)
        max_corr = np.max(abs_corr)
        logging.info(f"- Maximum correlation between different edge features: {max_corr:.2f}")

        # Trova le feature più correlate
        if max_corr > 0.8:  # Soglia per correlazione alta
            idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
            logging.info(f"- Most correlated edge features: {use_names[non_constant_features[idx[0]]]} and "
                         f"{use_names[non_constant_features[idx[1]]]} ({max_corr:.2f})")
