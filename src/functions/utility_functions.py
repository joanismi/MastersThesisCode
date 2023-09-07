import numpy as np
import os
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

def check_dir(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        pass
    else:
        os.makedirs(dir)


def tuckeys_fences(s, fence='upper', k=1.5, require_upper_outlier=True):
    """
    Computes the upper or lower Tukey's fences. This method is commonly
    used to detect outliers and to define the whiskers in box plots.
    
    This implementation is specially tailored to detect in which tissues
    a gene is expressed, given an array-like object (s) with the
    expression values in each tissue. The upper fence is used for tissue
    specific genes and the lower fence for housekeeping gene. It returns 
    an array filled with zeros & ones, representing gene absence and
    presence, respectively, in each tissue.
    
    Parameters
    ----------
    s : numpy 1D array or Series
        Sample where tuckey's fences method is applied.
        
    fence : {'upper', 'lower'}, default 'upper'
        Type of fence to compute.
        
    k : int or float, default 1.5
        k determines the reach of the fence. The higher is k, the more
        strigent the detection method is. 
        k=1.5 usually defines the inner fence and k=3 defines the
        outer fence.
        
    require_upper_outlier : bool, default True
        if True, when no outlier is detected while computing the 'upper'
        fence, the max value will be considered the only outlier.
        
    Returns
    -------
    outliers : same type as s
        array-like filled with zeros & ones:
        fence='upper' : ones are outliers
        fence='lower' : zeros are outliers

    """
    outliers = s.copy()
    
    IQR = np.quantile(s, .75) - np.quantile(s, .25)
    if fence == 'lower':
        fence_val = np.quantile(s, .25) - k*IQR
        outliers[outliers < fence_val] = 0
        outliers[outliers >= fence_val] = 1

    elif fence == 'upper':
        fence_val = np.quantile(s, .75) + k*IQR
        outliers[outliers <= fence_val] = 0
        outliers[outliers > fence_val] = 1
        if len(outliers[outliers==1]) == 0 and\
             require_upper_outlier is True:
            # if the Tukey's fences algorithm does not find outliers,
            # we will assume that the gene is present only in the
            # tissue with max expression
            # Every gene in the dataframe has at least 1 tissue with
            # TPM >= 1
            outliers = s.copy()
            outliers[outliers < outliers.max()] = 0
            outliers[outliers == outliers.max()] = 1
    else:
        raise ValueError("not a tuckey's fence")
    
    return outliers


def spearman_corr(df, corr_cols, alternative='two-sided'):
    """
    Computes the Spearman Rank Correlation Coefficient (SRCC).
    
    Parameters:
    -----------
    df : DataFrame
        Dataframe with numeric columns representing the used
        to compute the (SRCC).
        
    group_cols : list-like
        columns in df used to determine the groups to compute
        the (SRCC).
    
    y_cols : list_like
        y variables used to compute the SRCC against the x_col
        variable. This variables will be returned with the
        correlation coefficient and p-value.

    x_col : str
        x variable used to compute the SRCC against the y variable.
        This variable will not be returned in the columns.
    
    alternative : {"two-sided", "less", "greater"}, default "greater"

    Defines the alternative hypothesis. The following options are available:
        - "two-sided": the correlation is nonzero
        - "less": the correlation is negative (less than zero)
        - "greater": the correlation is positive (greater than zero)

    Returns:
    --------
    spearman : DataFrame
        DataFrame in records format where the numeric columns
        have the SRCC and p-values for that variable.
    """
    
    rho, pval = stats.spearmanr(df[corr_cols], alternative=alternative)

    return pd.Series({'coefficient': rho, 'pvalue': pval})
        

def perm_test(x, y, alternative= 'greater', n_permutations=1000):
    """
    Computes a permutation test on the difference of means
    """
    def statistic(x, y, axis=None):
        mean_diff = np.mean(x, axis=axis) - np.mean(y, axis=axis)
        return mean_diff
        
    res = stats.permutation_test(
        [x, y],
        statistic,
        alternative=alternative,
        n_resamples=n_permutations
    )

    return res.statistic, res.pvalue


def mannwhitney_test(array, x_index, y_index, fdr_corr=True, perm_test=False, n_jobs=8):
    """
    Performs a Mann-Whitney U test on an array.
    """
    
    x = array[x_index,:]
    y = array[y_index,:]
    
    mannwhitney = stats.mannwhitneyu(x, y, axis=0)[1]
    if fdr_corr is True:
        result = multitest.fdrcorrection(mannwhitney)[1]
    else:
        result = mannwhitney
        
    def permutation_test(array, x_index, y_index):
        rng = np.random.default_rng()
        a = rng.permutation(array, axis=0)
        x = a[x_index,:]
        y = a[y_index,:]
    
        mannwhitney = stats.mannwhitneyu(x, y, axis=0)[1]
        mannwhitney_corr = multitest.fdrcorrection(mannwhitney)[1]

        return mannwhitney_corr
        
    if perm_test is False:
        return result
    else:
        perm_list = Parallel(
            n_jobs=n_jobs
            )(delayed(permutation_test)(
                array,
                x_index,
                y_index) for i in tqdm(range(perm_test))
                )
        
        perm_array = np.array(perm_list)
        
        return result, perm_array