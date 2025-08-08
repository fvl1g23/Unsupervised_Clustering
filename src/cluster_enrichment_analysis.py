#Import libraries
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm, chi2
from statsmodels.stats.multitest import multipletests

def compute_standardized_residuals(cont_table):
    """
    Compute standardized residuals and expected frequencies for a contingency table.
    """
    chi2_stat, p_value, dof, expected = chi2_contingency(cont_table)
    residuals = (cont_table.values - expected) / np.sqrt(expected)
    return residuals, expected, chi2_stat, p_value

def correct_p_values(p_values_matrix, alpha, method):
    """
    Flatten p-values, apply multiple testing correction, then reshape.
    """
    flat_p_values = p_values_matrix.flatten()
    _, corrected_p_values, _, _ = multipletests(flat_p_values, alpha=alpha, method=method)
    return corrected_p_values.reshape(p_values_matrix.shape)

def create_significance_matrix(residuals, corrected_p_values, alpha, col_names, col):
    """
    Generate significance matrix indicating which clusters significantly deviate.
    """
    significance_matrix = pd.DataFrame(
        np.where(corrected_p_values < alpha, np.sign(residuals), 0),
        columns=[f"{col}_{category}" for category in col_names]
    )
    return significance_matrix

def format_residuals(residuals, p_values_matrix, alpha, cont_table):
    """
    Create a formatted DataFrame with residuals and significance indicators.
    """
    # Initialize an empty DataFrame for the formatted results
    formatted_results = pd.DataFrame(index=cont_table.index, columns=cont_table.columns, data='')

    for i in range(residuals.shape[0]):
        for j in range(residuals.shape[1]):
            value = residuals[i, j]
            p_val = p_values_matrix[i, j]
            if p_val < alpha:
                if value > 0:
                    formatted_results.iloc[i, j] = f'↑ {value:.2f} (p={p_val:.4f})'
                else:
                    formatted_results.iloc[i, j] = f'↓ {value:.2f} (p={p_val:.4f})'
            else:
                formatted_results.iloc[i, j] = f'{value:.2f} (p={p_val:.4f})'

    return formatted_results

def analyze_cluster_deviations(df, categorical_cols, cluster_col="Cluster", alpha=0.05, correction_method='fdr_bh'):
    """
    Analyze which clusters significantly deviate from expected frequencies for categorical variables.
    """
    results = {}
    significance_indicators = []

    for col in categorical_cols:
        cont_table = pd.crosstab(df[cluster_col], df[col])
        if col == "Diagnosis":
            cont_table = cont_table.loc[:,["CD", "UC"]]
        elif col == "IBD_diagnosis":
            cont_table = cont_table.loc[:,["Crohn's Disease", "Ulcerative Colitis"]]
        elif col == "Crohn s disease phenotype":
            cont_table = cont_table.loc[:,["Inflammatory non-penetrating, non-stricturing (B1)",
            "Stricturing (B2)", "Penetrating (B3)", "Both stricturing and penetrating (B2B3)"]]
        elif col == "IBD phenotype":
            cont_table = cont_table.loc[:, ["B1", "B2", "B3", "B2B3"]]

        mask = (cont_table < 5).any(axis=1)

        if mask.any():
            # Sum the rows that need to be merged into 'Miscellaneous'
            misc_cluster = pd.DataFrame(cont_table.loc[mask].sum()).T
            misc_cluster.index = ['Misc']  # Set new index name

            # Keep rows where all values are >= 5 and append "Miscellaneous" row
            cont_table_filtered = cont_table.loc[~mask]
            cont_table = pd.concat([cont_table_filtered, misc_cluster])

        residuals, expected, chi2_stat, overall_p_value = compute_standardized_residuals(cont_table)

        # Compute raw p-values for each residual
        p_values_matrix = 2 * (1 - norm.cdf(abs(residuals)))
        corrected_p_values_matrix = correct_p_values(p_values_matrix, alpha, correction_method)

        # Create residuals, p-values, and corrected p-values DataFrames
        residuals_df = pd.DataFrame(residuals, index=cont_table.index, columns=cont_table.columns)
        p_values_df = pd.DataFrame(p_values_matrix, index=cont_table.index, columns=cont_table.columns)
        corr_p_values_df = pd.DataFrame(corrected_p_values_matrix, index=cont_table.index, columns=cont_table.columns)

        # Create significance matrix
        significance_matrix = create_significance_matrix(residuals, corrected_p_values_matrix, alpha,
                                                         cont_table.columns, col)
        significance_matrix['Index'] = [df.index[df[cluster_col] == cluster].tolist() for cluster in cont_table.index]
        significance_indicators.append(significance_matrix)

        # Format residuals
        formatted_results = format_residuals(residuals, p_values_matrix, alpha, cont_table)

        #Compute X2 contributions
        cell_contributions = (cont_table - expected) ** 2 / expected

        # Compute raw p-values for the chi-square contributions
        p_values_chi2 = 1 - chi2.cdf(cell_contributions, df=1)
        p_values_chi2_df = pd.DataFrame(p_values_chi2, index=cont_table.index, columns=cont_table.columns)

        #Correct chi2 p values
        corrected_p_values_chi2 = correct_p_values(p_values_chi2, alpha, correction_method)

        corr_p_values_chi2_df = pd.DataFrame(corrected_p_values_chi2,
                                             index=cont_table.index, columns=cont_table.columns)

        # Store results
        results[col] = {
            'observed': cont_table,
            'expected': pd.DataFrame(expected, index=cont_table.index, columns=cont_table.columns),
            'chi2_stat': chi2_stat,
            'overall_p_value': overall_p_value,
            'residuals': residuals_df,
            'p_values': p_values_df,
            'corr_p_values': corr_p_values_df,
            'formatted': formatted_results,
            'chi2_contributions': cell_contributions,
            'p_values_chi2': p_values_chi2_df,
            'corr_p_values_chi2': corr_p_values_chi2_df
        }

    # Merge significance indicators across all categorical variables
    merged_significance = pd.concat(significance_indicators, axis=1)
    merged_significance = merged_significance.loc[:, ~merged_significance.columns.duplicated(keep='first')]

    # Ensure 'Index' is the first column
    columns_order = ['Index'] + [col for col in merged_significance.columns if col != 'Index']
    merged_significance = merged_significance[columns_order]
    results['merged_significance'] = merged_significance

    return results

def analyze_cluster_deviations_2(cont_table, col, cluster_col="Group", alpha=0.05, correction_method='fdr_bh'):
    results = {}
    significance_indicators = []

    residuals, expected, chi2_stat, overall_p_value = compute_standardized_residuals(cont_table)

    # Compute raw p-values for each residual
    p_values_matrix = 2 * (1 - norm.cdf(abs(residuals)))
    corrected_p_values_matrix = correct_p_values(p_values_matrix, alpha, correction_method)

    # Create residuals, p-values, and corrected p-values DataFrames
    residuals_df = pd.DataFrame(residuals, index=cont_table.index, columns=cont_table.columns)
    p_values_df = pd.DataFrame(p_values_matrix, index=cont_table.index, columns=cont_table.columns)
    corr_p_values_df = pd.DataFrame(corrected_p_values_matrix, index=cont_table.index, columns=cont_table.columns)

    # Format residuals
    formatted_results = format_residuals(residuals, p_values_matrix, alpha, cont_table)

    # Compute X2 contributions
    cell_contributions = (cont_table - expected) ** 2 / expected

    # Compute raw p-values for the chi-square contributions
    p_values_chi2 = 1 - chi2.cdf(cell_contributions, df=1)
    p_values_chi2_df = pd.DataFrame(p_values_chi2, index=cont_table.index, columns=cont_table.columns)

    # Correct chi2 p values
    corrected_p_values_chi2 = correct_p_values(p_values_chi2, alpha, correction_method)

    corr_p_values_chi2_df = pd.DataFrame(corrected_p_values_chi2,
                                         index=cont_table.index, columns=cont_table.columns)

    # Store results
    results[col] = {
        'observed': cont_table,
        'expected': pd.DataFrame(expected, index=cont_table.index, columns=cont_table.columns),
        'chi2_stat': chi2_stat,
        'overall_p_value': overall_p_value,
        'residuals': residuals_df,
        'p_values': p_values_df,
        'corr_p_values': corr_p_values_df,
        'formatted': formatted_results,
        'chi2_contributions': cell_contributions,
        'p_values_chi2': p_values_chi2_df,
        'corr_p_values_chi2': corr_p_values_chi2_df
    }

    return results

