import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import scipy as sp
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

def intercell_interactions_analysis(
    directory,
    interactions, 
    ):
    """
    Computes interaction presence/absence in intercellular
    interactions networks of pairs of tissues

    Parameters:
    -----------
    directory : str
        os directory where which contains intercellular interactions
        networks.

    interactions : DataFrame
        Contains all possible intercellular interactions.

    Returns:
    --------
    inters_pairs : 2D-array
        Array filled with zeros and ones where a one represents presence
        of a interaction (rows) in a tissue pair intercellular network
        (columns)
    pairs : DataFrame
        
    """
    files = sorted(os.listdir(directory))
    inters_pairs = np.zeros((interactions.shape[0], len(files)))
    pairs = []
    for i, file in enumerate(tqdm(files)):

        graph = pd.read_csv(directory+'/'+file)

        ct = graph.columns[0]
        mt = graph.columns[1]
        pairs.append((ct, mt))
        
        # create graph
        simp_graph = graph.drop_duplicates(subset=[ct, mt])

        # return the graph to the orginal order of source, target
        # this way, if an interaction happens both ways it becames
        # duplicated in the new graph
        x = simp_graph[simp_graph.direction=='>']\
            .rename({ct:'source', mt:'target'}, axis=1)\
                .drop('direction', axis=1)
        y = simp_graph[simp_graph.direction=='<']\
            .rename({ct:'target', mt:'source'}, axis=1)\
                .drop('direction', axis=1)

        # drop interactions that happen both ways
        unique_inters = pd.concat([x, y])\
            .drop_duplicates(ignore_index=True)
        unique_inters[0] = [1 for i in range(unique_inters.shape[0])]

        unique_inters_array = pd.merge(
            interactions, unique_inters, how='left')\
                .fillna(0)
                
        unique_inters_array.sort_values(by=['source', 'target'], inplace=True)   
        
        inters_pairs[:,i] = unique_inters_array[0].to_numpy()
        
    pairs = pd.DataFrame(
        pairs, columns=['cancer', 'metastasis'])
        
    return inters_pairs, pairs


def interaction_stats(inter_array, labels, interaction, extra_labels=None):
    """
    Function.
    """
    table = pd.crosstab(inter_array, labels)
    table.sort_index(ascending=False, inplace=True) # sort rows
    table.sort_index(ascending=False, axis=1, inplace=True) # sort columns

    # some interactions might appear in all pairs or none
    # when generating a 2x2 contingency table, this will mean the
    # table will have a (1, 2) shape
    if table.shape[0] == 1:
        inter_stats = {
        'source': interaction[0],
        'target': interaction[1],
        'OR': np.nan,
        'logOR': np.nan,
        'OR_pvalue': np.nan,
        'RR': np.nan,
        'logRR': np.nan,
        'RR_pvalue': np.nan,
        'fisher_exact': 1
    }

    # when some condition has no counts only the fisher's exact test
    # might be relevant
    elif 0 in table.values:
        inter_stats = {
        'source': interaction[0],
        'target': interaction[1],
        'OR': np.nan,
        'logOR': np.nan,
        'OR_pvalue': np.nan,
        'RR': np.nan,
        'logRR': np.nan,
        'RR_pvalue': np.nan,
        'jaccard': \
            table.iloc[0,0]/(table.iloc[0,0]+table.iloc[0,1]+table.iloc[1,0]),
        'fisher_exact': sp.stats.fisher_exact(table)[1]
    }

    else:
        cont_table = sm.stats.Table2x2(table.to_numpy(), shift_zeros=False)
        inter_stats = {
            'source': interaction[0],
            'target': interaction[1],
            'OR': cont_table.oddsratio,
            'logOR': cont_table.log_oddsratio,
            'OR_pvalue': cont_table.oddsratio_pvalue(),
            'RR': cont_table.riskratio,
            'logRR': cont_table.log_riskratio,
            'RR_pvalue': cont_table.riskratio_pvalue(),
            'jaccard': \
                table.iloc[0,0]/(table.iloc[0,0]+table.iloc[0,1]+table.iloc[1,0]),
            'fisher_exact': sp.stats.fisher_exact(table)[1]
        }

    if extra_labels is not None:
        for k, v in extra_labels:
            inter_stats[k] = v

    return inter_stats


def freq_interaction_stats(frequency, labels, interaction, extra_labels=None):
    """
    Tests if the tissue pairs that have an interaction have a significantly larger or
    smaller frequency of ocurrence than pairs without the interaction. Computes the
    Wilcoxon-Mann-Whitney U and the Mood's Median Test.
    
    Parameters:
    -----------
    frequency : array-like 1D
        1D array with observed frequency for each tissue pair.
        
    labels : array-like 1D
        1D array of zeros and ones representing interaction presence, absence in the
        tissue pairs.
        
    interaction : array-like
        Array with len(2): first element is the name of the source gene and second
        element is the name of the target gene.
        
    extra_labels : dict-like, default None
        Identifiers to add to records
        
    """
    inter = frequency[labels==1]
    n_inter = frequency[labels==0]
    
    if (len(inter)*len(n_inter)) == 0:
        inter_stats = {
            'source': interaction[0],
            'target': interaction[1],
            'inter_median': np.nan,
            'n_inter_median': np.nan,
            'MWU_inter_stat': np.nan,
            'MWU_n_inter_stat': np.nan,
            'MannWhitneyU': np.nan,
            'MoodsMedianTest': np.nan,
            'GrandMedian': np.nan,
        }
    else:
        xstat, pval_mwu = sp.stats.mannwhitneyu(inter, n_inter)
        
        # compute n_inter stat (ystat)
        nx, ny = len(inter), len(n_inter)
        ystat = nx*ny - xstat

        _, pval_mmt, gmedian, _ = sp.stats.median_test(inter, n_inter)

        inter_stats = {
            'source': interaction[0],
            'target': interaction[1],
            'inter_median': np.median(inter),
            'n_inter_median': np.median(n_inter),
            'MWU_inter_stat': xstat,
            'MWU_n_inter_stat': ystat,
            'MannWhitneyU': pval_mwu,
            'MoodsMedianTest': pval_mmt,
            'GrandMedian': gmedian,
        }
        
    if extra_labels is not None:
        for k, v in extra_labels:
            inter_stats[k] = v
            
    return inter_stats


def weighted_intercell_interactions_analysis(
    pair,
    weights,
    interactions,
    ):
    """
    Computes a weighted network. The weight of an interaction is the product of the two gene weights

    Parameters:
    -----------
    pair : list-like with len(2)
        pair of tissues.
    
    weights : DataFrame
        DataFrame where indices are the gene symbols and the columns are 
        the tissues. Each value is a weight for a gene in a tissue.

    interactions : DataFrame
        DataFrame with two columns named 'source' and 'target'. Each row 
        represents an interaction between gene in 'source' and gene in
        'target'.

    Returns:
    --------
    simp_graph : DataFrame
        DataFrame of a weighted network where each row represents an interaction with a source column, a target column and a weight column.
    """

    t1 = pair[0]
    t2 = pair[1]
    t1_weights = weights[t1]
    t2_weights = weights[t2]
    
    t1_to_t2 = pd.merge(
        interactions,
        t1_weights,
        left_on='source',
        right_on='gene_id')
    t1_to_t2 = pd.merge(
        t1_to_t2,
        t2_weights,
        left_on='target',
        right_on='gene_id')

    t2_to_t1 = pd.merge(
        interactions,
        t2_weights,
        left_on='source',
        right_on='gene_id')
    t2_to_t1 = pd.merge(
        t2_to_t1,
        t1_weights,
        left_on='target',
        right_on='gene_id')

    # since we only want 1 copy of each interaction in this
    # analysis, we don't need to reverse the graph in one direction to 
    # simplify the graph. We'll drop the copy of each interaction
    # that has lowest weight
    
    # build graph
    graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)
    
    # compute interaction weight
    graph['weight'] = graph[t1]*graph[t2]
    graph.sort_values(by='weight', ascending=False, inplace=True)

    # Drop copy of interaction with lowest weight
    simp_graph = graph.drop_duplicates(subset=['source', 'target'], keep='first')
    simp_graph.drop([t1, t2], axis=1, inplace=True)

    return simp_graph
            

def weighted_interaction_stats(weights, labels, interaction, extra_labels=None):
    """
    Tests if the organotropism pairs have a significantly larger or
    smaller weight than non-organotropism pairs. Computes the
    Wilcoxon-Mann-Whithney U and the Mood's Median Test.
    
    Parameters:
    -----------
    frequency : array-like 1D
        1D array with observed frequency for each tissue pair.
        
    labels : array-like 1D
        1D array of zeros and ones representing interaction presence, absence in the
        tissue pairs.
        
    interaction : array-like
        Array with len(2): first element is the name of the source gene and second
        element is the name of the target gene.
        
    extra_labels : dict-like, default None
        Identifiers to add to records
        
    """
    org = weights[labels=='org']
    n_org = weights[labels=='n_org']
    
    if (len(org)*len(n_org)) == 0:
        # we cannot perform the tests if only one group exists
        inter_stats = {
            'source': interaction[0],
            'target': interaction[1],
            'org_median': np.nan,
            'n_org_median': np.nan,
            'MWU_org_stat': np.nan,
            'MWU_n_org_stat': np.nan,
            'MannWhitneyU': np.nan,
            'MoodsMedianTest': np.nan,
            'GrandMedian': np.nan,
        }
    else:
        xstat, pval_mwu = sp.stats.mannwhitneyu(org, n_org)
        
        nx, ny = len(org), len(n_org)
        ystat = nx*ny - xstat

        try:
            stat_mmt, pval_mmt, gmedian, table = sp.stats.median_test(org, n_org)
        except ValueError:
            # Since we are considering that values equal to the grand median
            # are below the grand median, when all weights are the same (usually 0), 
            # no value will be above the grand median, and an Value error is raised
            # we will assign a gmedian of zero and a p-value of 1 in these cases
            # since they are not significant   
            pval_mmt = 1
            gmedian = 0

        inter_stats = {
            'source': interaction[0],
            'target': interaction[1],
            'org_median': np.median(org),
            'n_org_median': np.median(n_org),
            'MWU_org_stat': xstat,
            'MWU_n_org_stat': ystat,
            'MannWhitneyU': pval_mwu,
            'MoodsMedianTest': pval_mmt,
            'GrandMedian': gmedian,
        }
        
    if extra_labels is not None:
        for k, v in extra_labels:
            inter_stats[k] = v
            
    return inter_stats


def weighted_freq_interaction_stats(
    weights,
    frequencies,
    interaction,
    extra_labels=None):

    """
    Tests if there is a correlation between metastasis frequency
    and network weights. Computes the Spearman Rank Correlation coefficient.
    
    Parameters:
    -----------
    weights : array-like 1D
        1D array with interaction weights for each tissue pair graph.

    frequency : array-like 1D
        1D array with observed frequency for each tissue pair graph.
        
    interaction : array-like
        Array with len(2): first element is the name of the source gene and second
        element is the name of the target gene.
        
    extra_labels : dict-like, default None
        Identifiers to add to records
        
    """
    rho, pval = sp.stats.spearmanr(weights, frequencies)

   
    inter_stats = {
        'source': interaction[0],
        'target': interaction[1],
        'spearman': rho,
        'pvalue': pval
    }
            
    if extra_labels is not None:
        for k, v in extra_labels:
            inter_stats[k] = v
            
    return inter_stats


def inter_shannon_index(row, inter_array, pairs):
    """
    Computes the Shannon Index (Entropy) of an interaction.
    """
    
    inter_id = int(row['interaction'])
    nt = row['network_type']
    td = row['tissue_dataset']
    mt = row['metastasis_dataset']

    values = pd.Series(inter_array[nt][td][inter_id], name='value')
    pv = pd.merge(pairs[mt][td], values, left_on='id', right_index=True)
    
    new_row = row.copy()
    for t in ['cancer', 'metastasis']:
        if nt == 'genecalls':
            inter = pv[pv.value==1]
            freq = [inter[inter[t]==c].shape[0] for c in inter[t].unique()]
            probs = np.array(freq)/inter.shape[0]

        elif nt == 'weighted_network':
            
            freq = [pv[pv[t]==c].value.sum() for c in pv[t].unique()]
            probs = np.array(freq)/sum(freq)

        new_row[f'{t}_entropy'] = sp.stats.entropy(probs)
    
    return new_row

