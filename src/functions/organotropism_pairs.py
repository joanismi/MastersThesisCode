from scipy import stats
import pandas as pd
import numpy as np
from statsmodels.stats import multitest
from functions.utility_functions import tuckeys_fences

pd.set_option('mode.chained_assignment', None)


def filter_df(df, col, row, drop=None, keep=None):
    """
    Filters columns and row in a count data DataFrame.
    Sums counts rows/columns with the same name and drops the
    unwanted ones. 

    Parameters
    ----------
    df : DataFrame
        DataFrame to be filtered

    col : list-like
        New names for the columns of df. Must have the same length
        as df.columns.

    row : list-like
        New names for the index of df. Must have the same length
        as df.index.

    keep : str or list-like
        Columns (metastasis tissues) to keep. If None, all columns 
        are kept

    drop : str or list-like
        Columns (metastasis tissues) to drop. If None, no columns 
        are dropped

    Returns
    -------
    filtered : DataFrame
        Filtered DataFrame.
    """
    
    # rename
    rename_col = dict(zip(df.columns, col))
    rename_row = dict(zip(df.index, row))
    renamed = df.rename(index=rename_row, columns=rename_col)

    # sum counts of rows/columns with the same 'name' 
    filtered = renamed.groupby(level=0, axis=1).sum()\
        .groupby(level=0, axis=0).sum()
    
    if drop is not None:
        if drop in filtered.index:
            filtered.drop(index=drop, inplace=True)
        if drop in filtered.columns:
            filtered.drop(columns=drop, inplace=True)
    
    if keep is not None:
        
        filtered = filtered.loc[
            filtered.index.isin(keep),
            filtered.columns.isin(keep)
        ]

    return filtered


def hypergeometric_test(df, alternative='greater'):
    """
    Computes the hypergeometric test for each cell in a dataframe.
    This is similar to a fisher's exact test but without contigency table.
    Computes the probability of observing a value as or more extreme than
    the observed value.

    Parameters
    ----------
    df : DataFrame
        Dataframe of count data

    alternative : {'less', 'greater'}, default 'greater'
        Defines the alternative  hypothesis:
            * 'greater': computing P(X>=k)
            * 'less': computing P(X<=k)

    Returns
    -------
    p_values : DataFrame
        DataFrame with same shape as df with p-values for the test.

    """
    N = df.sum().sum()  # population size N
    K = df.sum(axis=1)  # population successes K (column sum)
    # sample size n (column_sum) and sample successes (k) will be
    # defined in the apply instance. This means the hypergeom.sf()/cdf()
    # method is applied columnwise.
    if alternative == 'greater':
        # To compute P(X>=k) we have to set k'=k-1 since the sf function
        # computes P(X>k)
        pvalues = df.apply(lambda k: stats.hypergeom.sf(k - 1, N, K, k.sum()))
        
    elif alternative == 'less':
        pvalues = df.apply(lambda k: stats.hypergeom.cdf(k - 1, N, K, k.sum()))
    
    else:
        raise ValueError(f'{alternative} is not a tail value')
    
    return pvalues


def fdr_correction(pvalues, alpha=0.05, method='indep'):
    """
    p-value correction for false discovery rate.
    
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.

    Parameters
    ----------
    pvalues : DataFrame
        Set of p-values of the individual tests.

    alpha : float, default 0.05
        Family-wise error rate.

    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests).
        ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).

    Returns
    -------
    pvalues_corr : DataFrame
        Same shape as pvalues
        pvalues adjusted for multiple hypothesis testing to limit FDR
        """

    pvalues_flat = pvalues.to_numpy().flatten()
    pvalues_corr = multitest.fdrcorrection(
                    pvalues_flat,
                    alpha=alpha,
                    method=method
                    )
    pvalues_corr = pd.DataFrame(
                    pvalues_corr[1].reshape(pvalues.shape),
                    index=pvalues.index,
                    columns=pvalues.columns
                    ) 
    return pvalues_corr
    

def match_tissues(org_pairs, drop_met=True):
    """
    Removes organotropism pairs when the metastasis tissue is the
    same as the primary cancer tissue.
    Drops 'empty tissues', i.e.,rows and/or columns when they lack
    organotropism pairs.
    
    Parameters
    ----------
    org_pairs : DataFrame
        DataFrame of organotropism pairs comprised of zeros and ones,
        where ones represent organotropism pairs.
    
    drop_met : bool, default True
        If True, columns (cancer tissues) and rows (metastasis tissues)
        without organotropism pairs are removed. If False only rows
        (cancers) without organotropism pairs are removed

    Returns
    -------
    new_org_pairs : DataFrame
        Updated organotropism pairs with "empty" tissues removed.
    
    match : DataFrame
        DataFrame filled with ones except in the positions where
        tissues match.

    """
    # Find tissues that appear both in cancer and in metastasis organs
    tissues = [i for i in org_pairs.index if i in org_pairs.columns]
    match = org_pairs.copy()
    match[match==0] = 1
    for i in tissues:
        match.loc[i, i] = 0
    
    # remove organotropism pairs whose tissues match 
    new_org_pairs = org_pairs*match

    # remove tissues without organotropism pairs
    if drop_met:
        new_org_pairs = new_org_pairs.loc[
            new_org_pairs.sum(axis=1)>0,
            new_org_pairs.sum(axis=0)>0
            ]
    
        # update match dataframe
        match = match.loc[
            match.index.isin(new_org_pairs.index), 
            match.columns.isin(new_org_pairs.columns)
            ]
    else:
        new_org_pairs = new_org_pairs.loc[
            new_org_pairs.sum(axis=1)>0]
    
        # update match dataframe
        match = match.loc[
            match.index.isin(new_org_pairs.index)]

    return new_org_pairs, match


def organotropism_pairs_hyper_test(
    counts,
    tissues=[],
    alpha=0.05,
    fdr_corr=False,
    method='indep'):
    """
    Determines organotropism pairs by applying the hypergeometric test.

    Parameters
    ----------
    counts : DataFrame
        DataFrame of count frequencies. Rows represent primary cancer
        tissues and columns represent metastasis tissues.
    
    tissues : list-like, default empty list
        List of tissues to filter dataframe. Tissues not present in this
        list are removed from the final organotropism pairs DataFrame
    
    alpha : int or float, default 0.05
        p-value cutoff and family-wise error rate.
        Pairs below cutoff are considered organotropic.

    fdr_corr : bool, default False
        False Discovery Rate (FDR) correction for multiple tests.

    method : {'i', 'indep', 'p', 'poscorr', 'n', 'negcorr'}, optional
        Which method to use for FDR correction.
        ``{'i', 'indep', 'p', 'poscorr'}`` all refer to ``fdr_bh``
        (Benjamini/Hochberg for independent or positively
        correlated tests).
        ``{'n', 'negcorr'}`` both refer to ``fdr_by``
        (Benjamini/Yekutieli for general or negatively correlated tests).

    Returns
    -------
    org_pairs : DataFrame
        DataFrame of organotropism pairs filled with ones and zeros,
        where ones are organotropism pairs. Rows represent primary cancer 
        tissues and columns represent metastasis tissues.

    match : DataFrame
        DataFrame filled with ones except in the positions where
        tissues match.

    """
    # compute hypergeometric test and determine all organotropism pairs
    pvalues = hypergeometric_test(counts)
    if fdr_corr:
        pvalues = fdr_correction(pvalues, alpha, method)

    pvalues[pvalues < alpha] = '1' # organotropism pairs = 1
    pvalues = pvalues.where(pvalues=='1', '0')
    org_pairs = pvalues.astype('int64')

    # drop tissues not in tissues list
    if tissues:
        org_pairs = org_pairs.loc[
            org_pairs.index.isin(tissues), org_pairs.columns.isin(tissues)]

    # when the cancer tissue and metastasis tissue are the same they not
    # define organotropism or control pairs
    # remove organotropism pairs in matched tissues
    # drop rows/columns without organotropism pairs
    org_pairs, match = match_tissues(org_pairs)

    return org_pairs, match


def organotropism_pairs_frequency(
    counts,
    tissues=[],
    method='outlier_detection',
    k=1.5,
    quantile=0.75,
    filter_tissues='first',
    drop_met=True
    ):
    """
    Determines organotropism pairs.

    Parameters
    ----------
    counts : DataFrame
        DataFrame of count frequencies. Rows represent primary cancer
        tissues and columns represent metastasis tissues.
    
    tissues : list-like, default empty list
        List of tissues to filter dataframe. Tissues not present in this
        list are removed from the final organotropism pairs DataFrame.

    method : {'quantile', 'outlier_detection}, default upper_quartil
        Which method to use for determine organotropism pairs.

    quantile : float or int, default
    Returns
    -------
    org_pairs : DataFrame
        DataFrame of organotropism pairs filled with ones and zeros,
        where ones are organotropism pairs. Rows represent primary cancer 
        tissues and columns represent metastasis tissues.

    match : DataFrame
        DataFrame filled with ones (organotropism pairs), minus ones 
        (control pairs) and zeors (positions where tissues match)

    """
    df = counts.copy()

    # drop tissues not in tissues list
    if filter_tissues == 'first':
        if tissues is not None:
            df = df.loc[
                df.index.isin(tissues), df.columns.isin(tissues)]
    
    # remove cancers without metastasis site
    df = df.loc[df.sum(axis=1)>0]

    # determine all organotropism pairs
    if method=='quantil':
        
        def freq_pairs(x):
            q = x.quantile(quantile)
            x[x<q] = 0
            x[x>0] = 1
            return x

        org_pairs = df.transform(freq_pairs, axis=1)
    
    elif method=='outlier_detection':
        
        org_pairs = df.transform(
            tuckeys_fences,
            axis=1,
            **(dict(k=k)))

    # drop tissues not in tissues list
    if filter_tissues == 'last':
        if tissues:
            org_pairs = org_pairs.loc[
                org_pairs.index.isin(tissues), org_pairs.columns.isin(tissues)]

    # when the cancer tissue and metastasis tissue are the same they not
    # define organotropism or control pairs
    # remove organotropism pairs in matched tissues
    # drop rows/columns without organotropism pairs
    org_pairs, match = match_tissues(org_pairs, drop_met=drop_met)
    
    # since we are doing a z-score analysis with these pairs,
    # the control pairs will be all pairs that are not organotropism 
    org_pairs.where(org_pairs==1, -1, inplace=True)

    # remove control pairs with the same cancer and metastasis tissue
    org_pairs = org_pairs*match

    return org_pairs, match


def control_pairs(
    org_pairs, match, random_start=False, max_iterations=10000, random_state=None):
    """
    Determines control pairs, i.e., pairs not considered organotropic using
    the organotropism_pairs() method.
    Tries to converge to a perfect proportion of control pairs/organotropism
    pair for each tissue.

    Parameters
    ----------
    org_pairs : DataFrame
        DataFrame of organotropism pairs filled with ones and zeros, where
        ones are organotropism pairs. Rows represent primary cancer 
        tissues and columns represent metastasis tissues.
    
    match : DataFrame
        DataFrame filled with ones except in the positions where
        tissues match.
    
    random_start : bool, default False
        If True iterates rows randomly.
    
    max_iterations : int, default 10000
        Maximum number of iterations if algorithm does not find a perfect
        proportion of control pairs/organotropism pairs for each tissue.

    Returns
    -------
    best_pairs : DataFrame
        DataFrame with same shape as org_pairs. Proportion of pairs with
        best score.

    """
    not_pairs = match.to_numpy()
    rng = np.random.default_rng(random_state)
    
    # row indices
    rows = np.arange(org_pairs.shape[0])

    best_score = np.inf
    count = 0
    
    while (count < max_iterations) and (best_score > 0):
        if random_start:
            # Randomize the order of row iteration
            rng.shuffle(rows)

        pairs = org_pairs.copy().to_numpy()
        for i in rows:
            row_sum = pairs.sum(axis=1)
            weights = pairs.sum(axis=0)
            row = pairs[i]

            index = np.indices(row.shape).flatten()
            # select columns with organotropism pairs 
            index_0 = index[row != 0]
            # columns with organotropism pairs will have a probability of 
            # being choosed == 0
            weights[index_0] = 0
            # columns with a negative sum will also not be modified since the goal is to
            # have all columns with sum=0. Choosing this columns would only worsen the final score
            weights[weights < 0] = 0

            # Set the probabilities of choosing a position to insert a control pair 
            if abs(weights).sum() > 0:
                probs = weights**2/(weights**2).sum()
            else:
                # when all columns have sum=0. The probabilities of choosing are the same
                # for every position, except where there are organotropism pairs, where
                # the probability is 0
                weights = pairs.sum(axis=0)
                weights = np.where(weights==1, weights, 1)
                weights[index_0] = 0
                probs = weights/weights.sum()

            # randomly choose rows with weights to insert control pairs 
            if len(probs[probs>0]) < abs(row_sum[i]):
                # when n_org_pairs > (n_probabilities > 0)
                # it means that the control pairs arrangement
                # not ideal and some rows will end with more 
                # org_pairs than control
                sample = rng.choice(
                    index, len(probs[probs>0]), p=probs, replace=False)
            else:
                sample = rng.choice(
                    index, abs(row_sum[i]), p=probs, replace=False)

            # insert control pairs
            pairs[i, sample] = -1

        # remove control pairs with the same tissue
        pairs = pairs*not_pairs

        # calculate score
        row_score = abs(pairs.sum(axis=1))
        
        col_score = abs(pairs.sum(axis=0))
        
        score = row_score.sum() + col_score.sum()
        if score == 0:
            best_score = score
            best_pairs = pairs

        elif score < best_score:
            best_score = score
            best_pairs = pairs

        count += 1
            
    best_pairs = pd.DataFrame(
        best_pairs, index=org_pairs.index, columns=org_pairs.columns
    )
    return best_pairs


def compute_pairs(
    pairs,
    tissues,
    tissue_label,
    extra_labels=[],
    ):
    """
    Computes organotropism and control pairs, given a DataFrame.
    Returns pairs in a records format.
    """
    tissue_pairs = []
    for row in pairs.iterrows():
        cancer = row[0]
        pairs_ = row[1]
        org_pairs = pairs_[pairs_==1].index.to_list()
        cont_pairs = pairs_[pairs_==-1].index.to_list()
        
        tissue_dataset = tissues[tissue_label].dropna()
        cancer_tissues = tissue_dataset.loc[[cancer]].to_list()

        met_tissues = {
            'organotropism': tissue_dataset.loc[org_pairs].to_list(),
            'control': tissue_dataset.loc[cont_pairs].to_list(),
        }
        
        for pair_type, met_tissue in met_tissues.items():
            for mt in met_tissue:
                for ct in cancer_tissues:

                    pair = {
                        'cancer': cancer,
                        'metastasis': tissues[
                            tissues[tissue_label]==mt].index[0],
                        'cancer_tissue': ct,
                        'metastasis_tissue': mt,
                        'type': pair_type,
                    }

                    if extra_labels:
                            for k, v in extra_labels:
                                pair[k] = v
                    tissue_pairs.append(pair)
    return tissue_pairs

