import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

def plotk(results_df, vlines=None):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set(xlabel='k', ylabel='Silhouette score')
    ax1.plot(results_df['n_clusters'], results_df['silhouette_score'], color=color, label="Silhouette score")

    # Add dashed vertical lines with labels if provided
    if vlines is not None:
        y_max = ax1.get_ylim()[1]
        for k in vlines:
            ax1.axvline(x=k, color='gray', linestyle='--', linewidth=1)
            ax1.text(k, y_max * 1.02, f'k={k}', rotation=0, verticalalignment='bottom',
                     horizontalalignment='center', fontsize=10, color='gray')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set(xlabel='k', ylabel='DBI score')
    ax2.plot(results_df['n_clusters'], results_df['DBI_score'], color=color, label="DBI score")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", bbox_to_anchor=(1.5, 1))
    plt.show()

### Plot dendrogram
def AggC_dend(linkage_matrix, **kwargs):
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix, color_threshold=0,  # Turn off color variation
               **kwargs)
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

### Plot ARI confidence intervals
def plot_ari_conf_intervals(stability_scores,  title="K-Means Stability: ARI with 95% CI"):
    """
        Plots mean Adjusted Rand Index (ARI) with 95% confidence intervals for each k.

        Parameters:
        - stability_scores: dict[int, list[float]]
            Dictionary where keys are values of k and values are lists of ARI scores across runs.
        - title: str
            Title for the plot.
        """
    k_vals = sorted(stability_scores.keys())
    means = [np.mean(stability_scores[k]) for k in k_vals]
    stds = [np.std(stability_scores[k], ddof=1) for k in k_vals]
    ns = [len(stability_scores[k]) for k in k_vals]
    sems = [std / np.sqrt(n) for std, n in zip(stds, ns)]
    cis = [1.96 * sem for sem in sems]  # 95% CI assuming normal distribution

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(k_vals, means, yerr=cis, fmt='-o', capsize=4)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Mean Adjusted Rand Index (± 95% CI)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

### Plot ARI heatmap
def plot_ari_heatmap(ari_matrix, k):
    plt.figure(figsize=(10, 8))
    sns.heatmap(ari_matrix, cmap='viridis', square=True, linewidths=0.5,
                cbar_kws={"label": "Adjusted Rand Index"}, annot=False)
    plt.title(f'Pairwise ARI Heatmap for k = {k}')
    plt.xlabel("Run")
    plt.ylabel("Run")
    plt.tight_layout()
    plt.show()

### Plot all clustering metrics
def plot_clustering_metrics(metrics_df, invert_dbi=True, show_ci=True, vlines=None):
    """
        Plot ARI (with error bars), silhouette score, and DBI from a metrics_df.

        Parameters:
        - metrics_df : DataFrame
            Output of compute_clustering_metrics.
        - invert_dbi : bool
            If True, plot -DBI so all metrics follow 'higher is better'.
        - show_ci : bool
            If True, adds shaded 95% confidence intervals for Silhouette and DBI.
        - vlines : list of int, optional
        List of k values to mark with vertical dashed lines and labels.
        """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ks = metrics_df['k']

    # ARI bar plot with error bars
    ax1.bar(ks, metrics_df['mean_ari'], yerr=metrics_df['ci_ari'],
            align='center', alpha=0.6, label='ARI (±95% CI)', color='steelblue', capsize=3)
    ax1.set_ylabel("Adjusted Rand Index (ARI)", color='steelblue')
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # Twin axis for silhouette and DBI
    ax2 = ax1.twinx()

    sil = metrics_df['mean_silhouette']
    dbi = -metrics_df['mean_dbi'] if invert_dbi else metrics_df['mean_dbi']
    ci_sil = metrics_df['ci_silhouette']
    ci_dbi = metrics_df['ci_dbi']

    ax2.plot(ks, sil, label='Silhouette Score', color='darkgreen', linewidth=2)
    ax2.plot(ks, dbi, label='-DBI' if invert_dbi else 'DBI', color='darkred', linewidth=2)

    if show_ci:
        ax2.fill_between(ks, sil - ci_sil, sil + ci_sil, color='darkgreen', alpha=0.2)
        dbi_low = dbi - ci_dbi if invert_dbi else dbi - ci_dbi
        dbi_high = dbi + ci_dbi if invert_dbi else dbi + ci_dbi
        ax2.fill_between(ks, dbi_low, dbi_high, color='darkred', alpha=0.2)

    ax2.set_ylabel("Silhouette / (Inverted) DBI", color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add dashed vertical lines with labels if provided
    if vlines is not None:
        y_max = ax1.get_ylim()[1]
        for k in vlines:
            ax1.axvline(x=k, color='gray', linestyle='--', linewidth=1)
            ax1.text(k, y_max * 1.02, f'k={k}', rotation=0, verticalalignment='bottom',
                     horizontalalignment='center', fontsize=10, color='gray')

    # Legends and title
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    #plt.title("K-Means Clustering Metrics vs Number of Clusters (k)")
    plt.tight_layout()
    plt.show()