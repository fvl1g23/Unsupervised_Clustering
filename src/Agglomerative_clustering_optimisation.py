### Import packages
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from collections import Counter
from scipy.cluster.hierarchy import linkage


### Optimise agglomerative clustering algorithm for the silhouette score and the DBI
#Default ward linkage and Euclidean distance
def AggC_opt(cohort, gene_feat, cohort_name='SPARC', max_n_cluster=50):
    results = []
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]
    n_clusters = range(2, max_n_cluster, 1)

    for n_cluster in n_clusters:
        model = AgglomerativeClustering(linkage="ward", metric="euclidean", n_clusters=n_cluster)
        model.fit(X)
        labels = model.labels_
        silhouette = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        max_cluster_size = Counter(labels).most_common(1)
        results.append((n_cluster, silhouette, dbi, max_cluster_size[0][1]))

    results_df = pd.DataFrame(results, columns=['n_clusters', 'silhouette_score', 'DBI_score', 'max_cluster_size'])
    return results_df

### Implement agglomerative clustering algorithm
def AggC(cohort, gene_feat, cohort_name='SPARC', threshold=None, n_cluster=None, **kwargs):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[
            cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]

    # Compute the linkage matrix
    linkage_matrix = linkage(X, method="ward", metric="euclidean")
    model = AgglomerativeClustering(linkage="ward", metric="euclidean",
                                    n_clusters=n_cluster, distance_threshold=threshold)
    model.fit(X)
    aggc_out = pd.DataFrame({
        'Samid': X.index,
        'Cluster': model.labels_
    })
    aggc_out = aggc_out.set_index("Samid")

    return aggc_out, X, linkage_matrix