### Packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from collections import Counter

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
        X = cohort[cohort.columns[cohort.columns.str.split('.').str[0].isin(gene_feat.iloc[:, 2].str.split('_').str[0])]]

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
            km_out_phen = pd.merge(km_out, cohort.loc[:, ['SEX', 'DIAGNOSIS', 'Age at diagnosis', "CROHNS.DISEASE.PHENOTYPE", "IBD.SURGERY.FINAL"]],
                                   left_index=True, right_index=True)

        elif cohort_name == 'Soton':
            km_out_phen = pd.merge(km_out, cohort.loc[:,
                                           ['Gender', 'Age at diagnosis', 'Diagnosis', "Stricturing", "Fistulating", "Granuloma", "IBD Surgery", "IBD phenotype"]],
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
