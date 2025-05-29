import matplotlib.pyplot as plt
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