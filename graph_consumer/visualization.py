import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np

def visualize_node_features(data: Data, feature_names: list = None):
    """
    Visualizza le feature dei nodi con scale appropriate e layout ottimizzato

    Args:
        data: PyTorch Geometric Data object
        feature_names: Lista di nomi per le feature
    """
    # Estrazione e preparazione dati
    if not hasattr(data, 'x') or data.x is None:
        print("Nessuna feature trovata nel grafo.")
        return

    node_features = data.x.numpy() if data.x.is_cuda else data.x.detach().numpy()
    num_nodes, num_features = node_features.shape

    # Nomi delle feature
    if feature_names is not None:
        if len(feature_names) != num_features:
            raise ValueError(f"Attesi {num_features} nomi, ottenuti {len(feature_names)}")
        use_names = feature_names
    else:
        use_names = [f'Feature {i}' for i in range(num_features)]

    print(f"Grafo con {num_nodes} nodi e {num_features} features:")
    print("Nomi features:", ", ".join(use_names))

    # Calcolo statistiche rapide
    print("\nStatistiche features:")
    stats = []
    for i in range(num_features):
        unique_vals = np.unique(node_features[:, i])
        stats.append({
            'name': use_names[i],
            'unique': len(unique_vals),
            'mean': np.mean(node_features[:, i]),
            'std': np.std(node_features[:, i]),
            'min': np.min(node_features[:, i]),
            'max': np.max(node_features[:, i])
        })
        print(f"- {use_names[i]}: {stats[-1]}")

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

    # 3. Boxplot features di grado (log)
    if degree_features:
        plt.subplot(2, 3, 3)
        log_data = np.log1p(node_features[:, degree_features])
        sns.boxplot(data=pd.DataFrame(log_data, columns=[use_names[i] for i in degree_features]))
        plt.title('Confronto Metriche di Grado')
        plt.xticks(rotation=45)
        plt.ylabel('log(valore + 1)')

    # 4. Matrice di correlazione
    plt.subplot(2, 3, 4)
    corr = np.corrcoef(node_features.T)
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                xticklabels=use_names, yticklabels=use_names,
                fmt=".2f", annot_kws={"size": 10})
    plt.title('Matrice di Correlazione')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 5. Feature value ranges
    plt.subplot(2, 3, 5)
    for i, feat in enumerate(stats):
        plt.errorbar(i, feat['mean'], yerr=feat['std'], fmt='o', color='blue')
        #plt.plot(i, feat['mean'], '.', color='red')
        plt.plot(i, feat['min'], 'v', color='red')
        plt.plot(i, feat['max'], '^', color='green')

    plt.xticks(range(num_features), use_names, rotation=45)
    plt.title('Range Valori Features')
    plt.ylabel('Valore')
    #plt.legend(['Media ± Std', 'Minimo', 'Massimo'])
    plt.grid(True, alpha=0.3)

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

    plt.tight_layout()
    plt.show()

    # Analisi aggiuntiva
    print("\nAnalisi aggiuntiva:")
    if degree_features:
        print("- Le metriche di grado sono visualizzate in scala logaritmica")

    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0)
    max_corr = np.max(abs_corr)
    print(f"- Correlazione massima tra feature diverse: {max_corr:.2f}")

    if 'activity_score' in use_names and degree_features:
        activity_idx = use_names.index('activity_score')
        corrs_with_activity = corr[activity_idx, degree_features]
        print("- Correlazioni tra activity_score e metriche di grado:")
        for i, val in enumerate(corrs_with_activity):
            print(f"  {use_names[degree_features[i]]}: {val:.2f}")

def visualize_edge_features(data: Data, edge_feature_names: list = None, max_features_to_plot=12):
    """
    Visualizza le edge features con tecniche per gestire molte feature

    Args:
        data: PyTorch Geometric Data object
        edge_feature_names: Lista di nomi per le edge features
        max_features_to_plot: Numero massimo di feature da visualizzare
    """
    if not hasattr(data, 'edge_attr') or data.edge_attr is None:
        print("Nessuna edge feature trovata nel grafo.")
        return

    edge_features = data.edge_attr.numpy() if data.edge_attr.is_cuda else data.edge_attr.detach().numpy()
    num_edges, num_features = edge_features.shape

    # Nomi delle feature
    if edge_feature_names is not None:
        if len(edge_feature_names) != num_features:
            raise ValueError(f"Attesi {num_features} nomi, ottenuti {len(edge_feature_names)}")
        use_names = edge_feature_names
    else:
        use_names = [f'EdgeFeat_{i}' for i in range(num_features)]

    print(f"Grafo con {num_edges} archi e {num_features} edge features:")

    # Calcolo statistiche rapide
    print("\nStatistiche edge features:")
    stats = []
    for i in range(num_features):
        unique_vals = np.unique(edge_features[:, i])
        stats.append({
            'name': use_names[i],
            'unique': len(unique_vals),
            'mean': np.mean(edge_features[:, i]),
            'std': np.std(edge_features[:, i]),
            'min': np.min(edge_features[:, i]),
            'max': np.max(edge_features[:, i])
        })

    # Stampiamo solo le prime e ultime 5 feature per non inondare l'output
    for stat in stats[:5] + stats[-5:]:
        print(f"- {stat['name']}: {stat}")
    if num_features > 10:
        print(f"\n... omesse {num_features-10} features intermedie ...")

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
    else:
        plt.text(0.5, 0.5, 'Troppe feature costanti\nper la matrice di correlazione',
               ha='center', va='center')
        plt.axis('off')

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
        plt.xlabel('Valore normalizzato')
        plt.ylabel('Densità')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Tutte le feature sono costanti!',
               ha='center', va='center')
        plt.axis('off')

    # 4. Boxplot delle feature più importanti
    plt.subplot(3, 2, 4)
    if len(non_constant_features) > 0:
        features_to_boxplot = non_constant_features[:max_features_to_plot]
        boxplot_data = edge_features[:, features_to_boxplot]

        # Normalizzazione per rendere i boxplot comparabili
        boxplot_data = (boxplot_data - np.mean(boxplot_data, axis=0)) / (np.std(boxplot_data, axis=0) + 1e-8)

        sns.boxplot(data=pd.DataFrame(boxplot_data,
                                    columns=[use_names[i] for i in features_to_boxplot]))
        plt.title(f'Boxplot Feature non Costanti\n(Normalizzate, prime {len(features_to_boxplot)})')
        plt.xticks(rotation=90)
        plt.ylabel('Valori normalizzati')
    else:
        plt.text(0.5, 0.5, 'Nessuna feature non costante\nper il boxplot',
               ha='center', va='center')
        plt.axis('off')

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

    # 6. PCA per riduzione dimensionalità (se molte feature)
    plt.subplot(3, 2, 6)
    if len(non_constant_features) > 3:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Prendiamo solo feature non costanti
        X = edge_features[:, non_constant_features]
        X = StandardScaler().fit_transform(X)

        # PCA con 2 componenti
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X)

        plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
        plt.title('Riduzione Dimensionalità (PCA)\nSpiegato: {:.1f}%'.format(100*pca.explained_variance_ratio_.sum()))
        plt.xlabel('Prima Componente ({:.1f}%)'.format(100*pca.explained_variance_ratio_[0]))
        plt.ylabel('Seconda Componente ({:.1f}%)'.format(100*pca.explained_variance_ratio_[1]))
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Troppe poche feature non costanti\nper PCA',
               ha='center', va='center')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Analisi aggiuntiva
    print("\nAnalisi aggiuntiva:")
    if len(non_constant_features) < num_features:
        print(f"- {num_features - len(non_constant_features)} feature sono costanti")

    if len(non_constant_features) > 1:
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 0)
        max_corr = np.max(abs_corr)
        print(f"- Massima correlazione tra feature diverse: {max_corr:.2f}")

        # Trova le feature più correlate
        if max_corr > 0.8:  # Soglia per correlazione alta
            idx = np.unravel_index(np.argmax(abs_corr), abs_corr.shape)
            print(f"- Feature più correlate: {use_names[non_constant_features[idx[0]]]} e "
                 f"{use_names[non_constant_features[idx[1]]]} ({max_corr:.2f})")