import numpy as np
import os

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