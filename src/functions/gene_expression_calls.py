from scipy import stats
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, calinski_harabasz_score,\
    davies_bouldin_score
from functions.utility_functions import tuckeys_fences

pd.set_option('mode.chained_assignment', None)


def apply_tau(s):
    """
    Applies tau algorithm to each row (gene) in a DataFrame

    """
    tau = round((1-s/max(s)).sum()/(len(s) - 1), 3)
    return tau


def outlier_expression_calls(
    df, 
    fence='upper', 
    k=1.5,
    lower_threshold=None,
    upper_threshold=None,
    require_upper_outlier=True):
    
    """
    Performs gene expression calls using the tuckey's fences outlier
    method.
    
    Returns a DataFrame filled with zeros & ones, representing gene
    absence and presence, respectively, in each tissue.
    
    Parameters
    ----------
    df : DataFrame
        Index represents genes and columns represent tissues.
        
    fence : {'upper', 'lower'}, default 'upper'
        Type of fence to compute.
        
    k : int or float, default 1.5
        k determines the reach of the fence. The higher is k, the
        more strigent the detection method is.
        k=1.5 usually defines the inner fence and k=3 defines the
        outer fence.
        
    lower_threshold : int or float
        Gene is considered absent in tissues where its expression
        value is below this threshold.
    
    upper_threshold : int or float
        Gene is considered present in tissues where its expression
        value is above or equal to this threshold.
    
    require_upper_outlier : bool, default True
        if True, when no outlier is detected while computing the
        'upper' fence, the max value will be considered the only
        outlier.

    Returns
    -------
    expression_calls_df : DataFrame
        Dataframe where zeros represent gene absence and ones
        represent gene presence in each tissue.

    """
    
    try:
        assert lower_threshold < upper_threshold,\
            "lower_threshold > upper_threshold"
    except TypeError: # when both thresholds are None
        pass
    
    # using a numpy 2D array is faster
    index = df.index
    columns = df.columns
    expression_array = df.to_numpy()
    
    expression_calls =\
        np.apply_along_axis(tuckeys_fences, 1, expression_array, fence, k, require_upper_outlier)
        
    # compute thresholds
    if upper_threshold is not None:
        expression_calls[expression_array >= upper_threshold] = 1
        
    if lower_threshold is not None:
        expression_calls[expression_array < lower_threshold] = 0
        
    expression_calls_df =\
        pd.DataFrame(expression_calls, index, columns, dtype='int')
    
    return expression_calls_df


def mean_shift_clustering(s, quantiles, metric='silhouette'):
    """
    Computes the Mean Shift clustering algorithm.
    
    This implementation is specially tailored to detect in which
    tissues a gene is expressed, given an array-like object (s)
    with the expression values in each tissue. Gene is considered
    not expressed in tissues in the lowest expression cluster.
    
    It returns an array filled with zeros & ones, representing gene
    absence and presence, respectively, in each tissue. 
    
    Parameters
    ----------
    s : numpy 1D array or Series
        Sample where the Mean Shift clustering algorithm is applied.
        
    quantiles : list-like,
        Quantile values to iterate over. These values are used to
        compute the estimate_bandwidth function.
        
    metric : str {'silhouette', 'calinski_harabasz', 'davies_bouldin'},
            default silhouette
        Scoring function to used to determine the best clustering results.
        
    Returns
    -------
    X : same type as s
        array-like filled with zeros & ones where 1 represent tissues
        where gene is present.

    """
    
    X = s.copy()
    X_2D = X.reshape(-1, 1)

    if metric == 'silhouette' or metric == 'calinski_harabasz':
        best_score = 0
    elif metric == 'davies_bouldin':
        best_score = np.inf
    else:
        raise ValueError(f'{metric} is not a valid scoring function')

    for n in quantiles:
    # The bandwidth can be automatically detected using estimate bandwidth
        bandwidth = estimate_bandwidth(X_2D, quantile=n, random_state=42)

        ms = MeanShift(bandwidth=bandwidth)
        labels = ms.fit_predict(X_2D)
        if metric == 'silhouette':
            score = (silhouette_score(X_2D, labels)\
                if len(set(labels)) > 1 else 0)
            if score > best_score:
                best_score = score
                best_ms_labels = labels

        elif metric == 'calinski_harabasz':
            score = (calinski_harabasz_score(X_2D, labels)\
                 if len(set(labels)) > 1 else 0)
            if score > best_score:
                best_score = score
                best_ms_labels = labels

        elif metric == 'davies_bouldin':
            score = (davies_bouldin_score(X_2D, labels)\
                 if len(set(labels)) > 1 else np.inf)
            if score < best_score:
                best_score = score
                best_ms_labels = labels

    # index of lowest expression cluster
    index_array = best_ms_labels[X == min(X)][0]
    # bool array for tissues in the lowest expression cluster
    lower_cluster = (best_ms_labels == index_array)
    
    X[lower_cluster] = 0
    X[X > 0] = 1
    
    return X


def clustering_expression_calls(
    df,
    quantiles,
    metric='silhouette',
    lower_threshold=None,
    upper_threshold=None):
    
    """
    Performs gene expression calls using the Mean Shift Clustering
    algorithm.
    
    Returns a DataFrame filled with zeros & ones, representing gene
    absence and presence, respectively, in each tissue.
    
    Parameters
    ----------
    df : DataFrame
        Index represents genes and columns represent tissues.
        
    quantiles : list-like,
        Quantile values to iterate over. These values are used to
        compute the estimate_bandwidth function.
        
    metric : str {'silhouette', 'calinski_harabasz', 'davies_bouldin'},
            default silhouette
        Scoring function used to determine the best clustering results.
        
    lower_threshold : int or float
        Gene is considered absent in tissues where its expression value
        is below this threshold.
    
    upper_threshold : int or float
        Gene is considered present in tissues where its expression value
        is above or equal to this threshold.
        
    Returns
    -------
    expression_calls_df : DataFrame
        Dataframe where zeros represent gene absence and ones represent
        gene presence in each tissue.

    """
    
    try:
        assert lower_threshold < upper_threshold,\
            "lower_threshold > upper_threshold"
    except TypeError: # when both thresholds are None
        pass
    
    # using a numpy 2D array is faster
    index = df.index
    columns = df.columns
    expression_array = df.copy().to_numpy()
    
    args = (quantiles, metric)
    expression_calls =\
        np.apply_along_axis(mean_shift_clustering, 1, expression_array, *args)
        
    # compute thresholds
    if upper_threshold is not None:
        expression_calls[expression_array >= upper_threshold] = 1
        
    if lower_threshold is not None:
        expression_calls[expression_array < lower_threshold] = 0
        
    expression_calls_df =\
        pd.DataFrame(expression_calls, index, columns, dtype='int')
    
    return expression_calls_df


def outlier_results(calls_df, fence):
    
    dist = calls_df.sum(axis=1).rename('n_tissues').reset_index()
    dist['fence'] = [fence]*dist.shape[0]
    
    corr_methods = {
        "spearman": stats.spearmanr,
        "pearson": stats.pearsonr
        }
    corr_results = []
    for corr, func in corr_methods.items():
        
        stat, pval = func(dist['tau'], dist['n_tissues'])

        corr_results.append({
            "correlation": stat,
            "pvalue": pval,
            "method": corr,
            "fence": fence
            })
        
    return dist, pd.DataFrame(corr_results)

