### Packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

### Optimise a K-Means algorithm for the silhouette score and the Davies-Bouldin index
def KM_opt_RNA(hit_count_matrix, gene_feat, max_n_clust, random_state=42):
    X = hit_count_matrix[hit_count_matrix.columns[hit_count_matrix.columns.isin(gene_feat['Gene stable ID'])]]
    cluster_range = range(2, max_n_clust + 1)
    results = []

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(X)

        labels = kmeans.labels_
        silhouette = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        max_cluster_size = Counter(kmeans.labels_).most_common(1)
        results.append((n_clusters, silhouette, dbi, max_cluster_size[0][1]))

    results_df = pd.DataFrame(results, columns=['n_clusters', 'silhouette_score', 'DBI_score', 'max_cluster_size'])
    return results_df

def KM_RNA(hit_count_matrix, gene_feat, n_clust, random_state=42):
    X = hit_count_matrix[hit_count_matrix.columns[hit_count_matrix.columns.isin(gene_feat['Gene stable ID'])]]

    kmeans = KMeans(n_clusters=n_clust, random_state=random_state)
    kmeans.fit(X)
    km_out = pd.DataFrame({
        'Sample': X.index,
        'Cluster': kmeans.labels_
    })
    km_out = km_out.set_index("Sample")
    return km_out, X