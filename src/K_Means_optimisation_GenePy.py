### Packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from collections import Counter
import itertools
from joblib import Parallel, delayed
from scipy.stats import sem, t

### Optimise a K-Means algorithm for the silhouette score and the Davies-Bouldin index
def KM_opt(cohort, gene_feat, max_n_clust, cohort_name='SPARC', random_state=42):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]
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

### Optimise a K-Means algorithm based on clustering stability
def KM_opt_stabl(cohort, gene_feat, max_n_clust, cohort_name='SPARC', n_iter=50, sample_fraction=0.8, random_state=None):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]
    cluster_range = range(2, max_n_clust + 1)
    rng = np.random.default_rng(random_state)
    stability_scores = {k: [] for k in cluster_range}

    for k in cluster_range:
        for _ in range(n_iter):
            # Step 1: Subsample the data twice
            idx1 = rng.choice(X.index, size=int(0.8 * len(X)), replace=False)
            idx2 = rng.choice(X.index, size=int(0.8 * len(X)), replace=False)
            X1 = X.loc[idx1]
            X2 = X.loc[idx2]

            km1 = KMeans(n_clusters=k, n_init='auto', random_state=None).fit(X1)
            km2 = KMeans(n_clusters=k, n_init='auto', random_state=None).fit(X2)

            # Step 2: Find overlapping samples
            overlap_idx = np.intersect1d(idx1, idx2)
            if len(overlap_idx) < 2:
                continue

            # Step 3: Map overlap indices to their positions in idx1 and idx2
            idx1_map = {sample_idx: pos for pos, sample_idx in enumerate(idx1)}
            idx2_map = {sample_idx: pos for pos, sample_idx in enumerate(idx2)}

            # Step 4: Get cluster labels for overlapping samples
            labels1 = [km1.labels_[idx1_map[i]] for i in overlap_idx]
            labels2 = [km2.labels_[idx2_map[i]] for i in overlap_idx]

            # Step 5: Compute similarity (e.g., ARI)
            ari = adjusted_rand_score(labels1, labels2)
            stability_scores[k].append(ari)
    return stability_scores

def _run_single_kmeans(X, k, rs):
    model = KMeans(n_clusters=k, n_init='auto', random_state=rs)
    labels = model.fit_predict(X)
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    return labels, sil, dbi

def compute_clustering_metrics(cohort, gene_feat, max_n_clust, cohort_name="SPARC", n_iter=50, random_state_seed=42, n_jobs=-1, confidence=0.95):
    """
        Compute mean, std, and 95% CI for ARI, Silhouette, and DBI for each k over multiple K-Means runs.

        Parameters:
        - X : array-like or DataFrame of shape (n_samples, n_features)
        - k_range : iterable of int
            List or range of cluster counts to test.
        - n_iter : int
            Number of K-Means runs per k (each with a different random_state).
        - random_state_seed : int or None
            Seed for reproducibility of random states.

        Returns:
        - metrics_df : pd.DataFrame
            DataFrame with columns:
            ['k', 'mean_ari', 'std_ari', 'ci_ari',
                   'mean_silhouette', 'std_silhouette', 'ci_silhouette',
                   'mean_dbi', 'std_dbi', 'ci_dbi']
    """
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('_').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]
    cluster_range = range(2, max_n_clust + 1)
    results = []
    rng = np.random.default_rng(random_state_seed)

    for k in cluster_range:
        seeds = rng.integers(0, 1_000_000, size=n_iter)

        # Parallel execution of clustering runs for this k
        out = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_kmeans)(X, k, rs) for rs in seeds
        )

        labels_list, sils, dbis = zip(*out)

        # Compute pairwise ARIs
        aris = []
        for i in range(n_iter):
            for j in range(i + 1, n_iter):
                ari = adjusted_rand_score(labels_list[i], labels_list[j])
                aris.append(ari)

        def ci(series):
            if len(series) < 2:
                return 0
            s = np.array(series)
            return sem(s) * t.ppf((1 + confidence) / 2., len(s) - 1)

        results.append({
            'k': k,
            'mean_ari': np.mean(aris),
            'std_ari': np.std(aris),
            'ci_ari': ci(aris),
            'mean_silhouette': np.mean(sils),
            'std_silhouette': np.std(sils),
            'ci_silhouette': ci(sils),
            'mean_dbi': np.mean(dbis),
            'std_dbi': np.std(dbis),
            'ci_dbi': ci(dbis)
        })

    return pd.DataFrame(results)

### Assess clustering stability for a given k
def KM_random(cohort, gene_feat, n_clust, cohort_name='SPARC', n_runs=50, sample_fraction=0.8, random_state=None):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]

    rng = np.random.default_rng(random_state)
    clusterings = []

    for _ in range(n_runs):
        sampled_idx = rng.choice(X.index, size=int(sample_fraction * len(X)), replace=False)
        X_sample = X.loc[sampled_idx]
        km = KMeans(n_clusters=n_clust, n_init='auto', random_state=None).fit(X_sample)
        labels = pd.Series(km.labels_, index=X_sample.index)
        clusterings.append(labels)

    return clusterings

def compute_ari_matrix(clusterings):
    n = len(clusterings)
    ari_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            overlap = clusterings[i].index.intersection(clusterings[j].index)
            if len(overlap) >= 2:
                ari = adjusted_rand_score(
                    clusterings[i].loc[overlap],
                    clusterings[j].loc[overlap]
                )
                ari_matrix[i, j] = ari
                ari_matrix[j, i] = ari

    np.fill_diagonal(ari_matrix, 1.0)
    return ari_matrix

### Iterate through n principal components in PCA to analyse explained variance
def PCA_opt(cohort, gene_feat, max_n_comp, cohort_name='SPARC'):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]
    component_range = range(2, max_n_comp + 1)
    results = []

    for n_components in component_range:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        explained_variance_ratio = pca.explained_variance_ratio_
        total_explained_variance = np.sum(explained_variance_ratio)
        results.append((n_components, explained_variance_ratio, total_explained_variance))

    results_df = pd.DataFrame(results, columns=['n_components', 'explained_variance_ratio', 'total_explained_variance'])
    return results_df

### K-Means implementation
def KM(cohort, gene_feat, n_clust, cohort_name='SPARC', random_state=42):
    if cohort_name == 'SPARC':
        X = cohort[cohort.columns[cohort.columns.isin(gene_feat.iloc[:, 2])]]
    elif cohort_name == 'Soton':
        X = cohort[cohort.columns[cohort.columns.str.split('_').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]

    kmeans = KMeans(n_clusters=n_clust, random_state=random_state)
    kmeans.fit(X)
    km_out = pd.DataFrame({
        'Samid': X.index,
        'Cluster': kmeans.labels_
    })
    km_out = km_out.set_index("Samid")

    return km_out, X

### Calculate per cluster feature importances from a Random Forest Classifier
def KM_RF(km_out, X, random_state=1):
    results_list = []
    clf = RandomForestClassifier(random_state=random_state)

    for i in np.arange(0, max(km_out['Cluster'].unique()) + 1):
        # Binarise cluster labels
        km_in = km_out.copy()
        km_in[f'Binary Cluster {i}'] = (km_out['Cluster'] == i).astype(int)

        # Make dataframe with samid, GenePy features and binarised cluster labels
        df = pd.merge(X, km_in, left_index=True, right_index=True)

        clf.fit(X.values, df[f'Binary Cluster {i}'].values)
        sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1]  # Reverse sort
        features = np.take_along_axis(
            np.array(df.iloc[:, 0:X.shape[1]].columns.tolist()),
            sorted_feature_weight_idxes, axis=0)
        importances = np.take_along_axis(
            np.array(clf.feature_importances_),
            sorted_feature_weight_idxes, axis=0)

        # Create a list of dictionaries for each feature and its importance
        cluster_results = [{'Feature': feature, 'Importance': importance, 'Cluster': i}
                           for feature, importance in zip(features, importances)]

        # Append the list of dictionaries to the results list
        results_list.extend(cluster_results)
    results_df = pd.DataFrame(results_list)
    return results_df

def merge_phen_genepy(km_out,cohort, X, cohort_name='SPARC', phen=False):
    #Merge cluster labels with clinical features
    if phen==False:
        if cohort_name == 'SPARC':
            km_out_phen = pd.merge(km_out, cohort.loc[:, ['SEX', 'DIAGNOSIS', 'Age.at.diagnosis', "CROHNS.DISEASE.PHENOTYPE", "IBD.SURGERY.FINAL"]],
                                   left_index=True, right_index=True)

        elif cohort_name == 'Soton':
            km_out_phen = pd.merge(km_out, cohort.loc[:,
                                           ['Gender', 'Age.at.diagnosis', 'Diagnosis', "Stricturing", "Fistulating", "Granuloma", "IBD.Surgery", "IBD.phenotype"]],
                                   left_index=True, right_index=True)
    else:
        if cohort_name == 'SPARC':
            km_out_phen = pd.merge(km_out, cohort.loc[X.index, ['SEX', 'Age at diagnosis']],
                                   left_index=True, right_index=True)
        elif cohort_name == 'Soton':
            km_out_phen = pd.merge(km_out, cohort.loc[X.index, ['Gender', 'Age at diagnosis', "Stricturing", "Fistulating"]],
                                   left_index=True, right_index=True)

    #Merge GenePy
    km_out_phen_GenePy = pd.merge(km_out_phen, X, left_index=True, right_index=True)
    km_out_phen_GenePy = km_out_phen_GenePy.reset_index()
    km_out_phen_GenePy = km_out_phen_GenePy.rename(columns={"index": "Samid", "SEX": "Sex", "DIAGNOSIS": "Diagnosis",
                                                            "CROHNS.DISEASE.PHENOTYPE": "Crohn s disease phenotype",
                                                            "IBD.SURGERY.FINAL": "IBD surgery final"})

    return km_out_phen_GenePy
