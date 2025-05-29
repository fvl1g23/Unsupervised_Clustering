### Import packages
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter

### Optimise DBSCAN algorithm for silhouette score and DBI
def DBSCAN_opt(cohort, gene_feat, cohort_name='SPARC'):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]
    eps_range = np.arange(0.1, 1.1, 0.1)
    results = []

    for eps in eps_range:
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=eps)
        labels = dbscan.fit_predict(X)
        max_cluster_size = Counter(labels).most_common(1)
        uniq_labels = np.unique(labels)
        n_clusters = len(uniq_labels)

        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
        else:
            silhouette = None
            dbi = None
            print(f'eps: {eps}, Silhouette {silhouette}: Undefined (single cluster)')

        results.append((eps, silhouette, dbi, n_clusters, max_cluster_size[0][1]))

    results_df = pd.DataFrame(results, columns=['eps', 'silhouette_score', 'DBI', 'n_clusters', 'max_cluster_size'])
    return results_df

### Implement DBSCAN
def dbscan(cohort, gene_feat, eps, cohort_name='SPARC'):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]

    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=eps)
    labels = dbscan.fit_predict(X)
    dbscan_out = pd.DataFrame({
        'Samid': X.index,
        'Cluster': labels
    })
    dbscan_out = dbscan_out.set_index("Samid")
    return dbscan_out, X