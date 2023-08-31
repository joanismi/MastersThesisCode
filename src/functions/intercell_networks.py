import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os
from joblib import Parallel, delayed
import itertools

pd.set_option('mode.chained_assignment', None)

def build_intercell_networks(
    tissues,
    calls,
    interactions,
    key_column='index',
    tissue_column='tissues',
    directory='intercell_networks',
    sep=''
    ):
    """
    Builds intercellular networks and exports to file as a table

    Parameters:
    -----------
    tissues :

    calls :

    interactions :

    key_column : str, default 'index'

    tissue_column : str, default 'tissues'

    directory : str, default 'intercell_networks'

    Return:
    -------

    """
    tissues_copy = tissues.copy()
    unique_tiss = tissues_copy.index.unique()
    index = 0

    while tissues_copy.shape[0] > 1:
        t = unique_tiss[index]
        t_pairs = tissues_copy.index.drop(t).unique()

        # using an iterable to index guarantees that
        # it returns always an iterable
        t1 = tissues_copy.loc[[t], tissue_column]
        t2 = tissues_copy.loc[t_pairs, tissue_column]
    
        for t1_ in t1:
            # key based on index to name file
            t1_key = tissues_copy.loc[
                tissues_copy[tissue_column]==t1_, key_column].values[0]

            t1_genes = calls[t1_].dropna().index.to_list()
            
            for t2_ in t2:
                # key based on index to name file
                t2_key = tissues_copy.loc[
                    tissues_copy[tissue_column]==t2_, key_column].values[0]

                t2_genes = calls[t2_].dropna().index.to_list()
                
                t1_to_t2 = interactions[
                            interactions.source.isin(t1_genes)
                             & interactions.target.isin(t2_genes)
                        ].reset_index(drop=True)
                
                t1_to_t2['direction'] =\
                    ['>' for i in range(t1_to_t2.shape[0])]

                t2_to_t1 = interactions[
                    interactions.source.isin(t2_genes)
                    & interactions.target.isin(t1_genes)
                ].reset_index(drop=True)

                t2_to_t1['direction'] =\
                    ['<' for i in range(t2_to_t1.shape[0])]

                t2_to_t1.rename(dict(
                    source='target',
                    target='source'), axis=1, inplace=True)

                graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)
                
                graph.rename(dict(
                    source=t1_,
                    target=t2_,
                    ), axis=1, inplace=True)

                graph.to_csv(
                    f'{directory}/{t1_key}{sep}{t2_key}.csv',
                    index=False)
                
        tissues_copy.drop(t, inplace=True)
        index += 1


def build_grouped_intercell_networks(
    calls,
    interactions,
    directory='intercell_networks',
    sep=''
    ):
    """
    Builds intercellular networks and exports to file as a table

    Parameters:
    -----------
    calls :

    interactions :

    directory : str, default 'intercell_networks'

    Return:
    -------

    """
    tissues = calls.columns
    pairs = itertools.combinations(tissues, 2)
    
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    for p in pairs:
        t1=p[0]
        t2=p[1]
       
        t1_genes = calls[t1].dropna().index.to_list()
        t2_genes = calls[t2].dropna().index.to_list()
                
        t1_to_t2 = interactions[
                    interactions.source.isin(t1_genes)
                        & interactions.target.isin(t2_genes)
                ].reset_index(drop=True)
        
        t1_to_t2['direction'] =\
            ['>' for i in range(t1_to_t2.shape[0])]

        t2_to_t1 = interactions[
            interactions.source.isin(t2_genes)
            & interactions.target.isin(t1_genes)
        ].reset_index(drop=True)

        t2_to_t1['direction'] =\
            ['<' for i in range(t2_to_t1.shape[0])]

        t2_to_t1.rename(dict(
            source='target',
            target='source'), axis=1, inplace=True)

        graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)
        
        graph.rename(dict(
            source=t1,
            target=t2,
            ), axis=1, inplace=True)

        graph.to_csv(
            f'{directory}/{t1}{sep}{t2}.csv',
            index=False)


def build_random_intercell_networks(
    tissues,
    calls,
    interactions,
    directed_graph=False,
    weights=False,
    extra_labels=None,
    iterations=1000,
    random_state=None,
    n_jobs=1
    ):
    """
    Builds random intercellular networks.

    Parameters:
    -----------
    tissues :

    calls :

    interactions :

    weights: bool, default False

    extra_labels : str, 

    iterations : int, default 1000

    n_jobs : int, default 1

    Return:
    -------

    """
    # interactions that might exist based on the gene calls
    inter_calls = interactions[
        (interactions.source.isin(calls.index)) &
        (interactions.target.isin(calls.index))
    ]
    # we want to keep the same proportion of source/target genes
    # build a list with 3 distinct intercellular gene pools
    # the first entry will have the genes that can be both source and
    # target
    
    gene_pools = [
        inter_calls.source[
            inter_calls.source.isin(inter_calls.target)
        ].unique()
    ]
    
    # add the gene pools with the genes that can be only source or target
    # source only genes
    gene_pools.append(inter_calls.source[
        ~inter_calls.source.isin(gene_pools[0])
        ].unique())
    # target only genes
    gene_pools.append(inter_calls.target[
        ~inter_calls.target.isin(gene_pools[0])
        ].unique())

    # create a list with gene probablilities for each gene pool
    # the gene probablilities are the number of times each gene is
    # expressed in all tissues divided by the sum of counts of all genes
    if weights:
        gene_weights = []
        for pool in gene_pools:
            g = calls[calls.index.isin(pool)]
            w = (g.sum(axis=1)/g.sum(axis=1).sum()).values
            gene_weights.append(w)
    else:
        gene_weights = None
    
    tissues_copy = tissues.copy()
    unique_tiss = tissues_copy.index.unique()
    index = 0
    records = []

    while tissues_copy.shape[0] > 1:
        t = unique_tiss[index]
        t_pairs = tissues_copy.index.drop(t).unique()

        # using an iterable to index guarantees that
        # it returns always an iterable
        t1 = tissues_copy.loc[[t]]
        t2 = tissues_copy.loc[t_pairs]
        
        for t1_ in t1:
            
            t1_genes = calls[t1_].dropna().index.to_series()
            t1_ngenes = []
            for pool in gene_pools:
                n = t1_genes[t1_genes.isin(pool)].shape[0]
                t1_ngenes.append(n)

            for t2_ in t2:
                
                t2_genes = calls[t2_].dropna().index.to_series()
                t2_ngenes = []
                for pool in gene_pools:
                    n = t2_genes[t2_genes.isin(pool)].shape[0]
                    t2_ngenes.append(n)

                distribution = Parallel(
                    n_jobs=n_jobs
                    )(delayed(random_inters)(
                        inter_calls,
                        gene_pools,
                        t1_ngenes,
                        t2_ngenes,
                        gene_weights,
                        directed_graph,
                        random_state) for i in range(iterations))

                if directed_graph is True:

                    directions = {'c_to_m': [], 'm_to_c': []}

                    for d1, d2 in distribution:
                        directions['c_to_m'].append(d1)
                        directions['m_to_c'].append(d2)
                    
                    for dir, dist in directions.items():
                        pair = {
                            'cancer_tissue': t1_,
                            'metastasis_tissue': t2_,
                            'direction': dir,
                            'dist': dist,
                            'mean': np.mean(dist),
                            'std': np.std(dist)
                        }

                        if extra_labels is not None:
                            for k, v in extra_labels:
                                pair[k] = v
                        
                        records.append(pair)

                else:
                    pair = {
                        'cancer_tissue': t1_,
                        'metastasis_tissue': t2_,
                        'dist': distribution,
                        'mean': np.mean(distribution),
                        'std': np.std(distribution)
                    }
                    
                    if extra_labels is not None:
                        for k, v in extra_labels:
                            pair[k] = v
                
                    records.append(pair)
        
        tissues_copy.drop(t, inplace=True)
        index += 1
    
    return records


def build_random_grouped_intercell_networks(
    tissues,
    calls,
    interactions,
    directed_graph=False,
    weights=False,
    extra_labels=None,
    iterations=1000,
    random_state=None,
    n_jobs=1
    ):
    """
    Builds random intercellular networks with grouped tissues

    Parameters:
    -----------
    tissues :

    calls :

    interactions :

    weights: bool, default False

    extra_labels : str, 

    iterations : int, default 1000

    random_state : int, default None

    n_jobs : int, default 1

    Return:
    -------

    """
    # interactions that might exist based on the gene calls
    inter_calls = interactions[
        (interactions.source.isin(calls.index)) &
        (interactions.target.isin(calls.index))
    ]
    # we want to keep the same proportion of source/target genes
    # build a list with 3 distinct intercellular gene pools
    # the first entry will have the genes that can be both source and
    # target
    
    gene_pools = [
        inter_calls.source[
            inter_calls.source.isin(inter_calls.target)
        ].unique()
    ]
    
    # add the gene pools with the genes that can be only source or target
    # source only genes
    gene_pools.append(inter_calls.source[
        ~inter_calls.source.isin(gene_pools[0])
        ].unique())
    # target only genes
    gene_pools.append(inter_calls.target[
        ~inter_calls.target.isin(gene_pools[0])
        ].unique())

    # create a list with gene probablilities for each gene pool
    # the gene probablilities are the number of times each gene is
    # expressed in all tissues divided by the sum of counts of all genes
    if weights:
        gene_weights = []
        for pool in gene_pools:
            g = calls[calls.index.isin(pool)]
            w = (g.sum(axis=1)/g.sum(axis=1).sum()).values
            gene_weights.append(w)
    else:
        gene_weights = None
    

    tissues = calls.columns
    pairs = itertools.combinations(tissues, 2)
    
    records = []
    for p in pairs:
        t1=p[0]
        t2=p[1]

        # using an iterable to index guarantees that
        # it always returns an iterable
                
        t1_genes = calls[t1].dropna().index.to_series()
        t2_genes = calls[t2].dropna().index.to_series()

        t1_ngenes = []
        t2_ngenes = []
        for pool in gene_pools:
            n1 = t1_genes[t1_genes.isin(pool)].shape[0]
            n2 = t2_genes[t2_genes.isin(pool)].shape[0]
            t1_ngenes.append(n1)
            t2_ngenes.append(n2)

        distribution = Parallel(
            n_jobs=n_jobs
            )(delayed(random_inters)(
            inter_calls,
            gene_pools,
            t1_ngenes,
            t2_ngenes,
            gene_weights,
            directed_graph,
            random_state) for _ in range(iterations))
        
        if directed_graph:

            directions = {'c_to_m': [], 'm_to_c': []}

            for d1, d2 in distribution:
                directions['c_to_m'].append(d1)
                directions['m_to_c'].append(d2)
                    
            for dir, dist in directions.items():
                pair = {
                    'cancer_tissue': t1,
                    'metastasis_tissue': t2,
                    'direction': dir,
                    'dist': dist,
                    'mean': np.mean(dist),
                    'std': np.std(dist)
                }

                if extra_labels is not None:
                    for k, v in extra_labels:
                        pair[k] = v
                
                records.append(pair)

        else:
            pair = {
                'cancer_tissue': t1,
                'metastasis_tissue': t2,
                'dist': distribution,
                'mean': np.mean(distribution),
                'std': np.std(distribution)
            }
            
            if extra_labels is not None:
                for k, v in extra_labels:
                    pair[k] = v
        
            records.append(pair)
    
    return records


def random_inters(
    interactions,
    gene_pools,
    t1_ngenes,
    t2_ngenes,
    weights=None,
    directed_graph=False,
    random_state=None
    ):
    """
    Generates a undirected random intercellular interaction
    network between 2 pairs of tissues and counts the number of
    established interactions. 
    """
    rng = np.random.default_rng(random_state)
    t1_genes = []
    t2_genes = []
    
    if weights is None:
        weights = [None for _ in range(len(gene_pools))]
    
    for pool, w, t1, t2 in zip(gene_pools, weights, t1_ngenes, t2_ngenes):
        t1_genes.extend(rng.choice(pool, t1, replace=False, p=w))
        t2_genes.extend(rng.choice(pool, t2, replace=False, p=w))
    
    t1_to_t2 = interactions[
        interactions.source.isin(t1_genes) &
        interactions.target.isin(t2_genes)
    ]
    t1_to_t2['direction'] = ['>'] * t1_to_t2.shape[0]

    t2_to_t1 = interactions[
        interactions.source.isin(t2_genes) &
        interactions.target.isin(t1_genes)
    ]
    t2_to_t1['direction'] = ['<'] * t2_to_t1.shape[0]

    t2_to_t1 = t2_to_t1.rename(dict(
                    source='target',
                    target='source'), axis=1)

    graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)
                
    if directed_graph:
        # remove bidirectional interactions.
        # This interactions will be considered
        # as having no direction
        dir_graph = graph.drop_duplicates(
            subset=['source', 'target'], keep=False)
        
        # number of directed interactions
        n_inter = (
            # cancer to metastasis interactions
            dir_graph[
                dir_graph.direction=='>'].shape[0],
            # metastasis to cancer interactions
            dir_graph[
                dir_graph.direction=='<'].shape[0]
        )
        
    else:

        # simplify graph and compute the number of intercell
        # interactions
        n_inter = graph.drop_duplicates(
            subset=['source', 'target']).shape[0]

    return n_inter


def compute_intercell_interactions(
    directory,
    extra_labels=None
    ):
    """
    Computes the number of intercellular interactions.

    Parameters:
    -----------
    directory : str

    extra_labels : dict-like, default None
        Identifiers to add to records

    Returns:
    --------
    records : list
        List of dictionaries where each entry has the score for a
        weighted network with defined conditions.

    """
    records = []
    files = os.listdir(directory)
    for file in tqdm(files, desc='networks'):
        
        graph = pd.read_csv(directory+'/'+file)
        #c_index, m_index = re.findall(r'(\d+)', file)
        #ct = tissues.loc[tissues[key_column]==int(c_index), tissue_column][0]
        #mt = tissues.loc[tissues[key_column]==int(m_index), tissue_column][0]
        ct = graph.columns[0]
        mt = graph.columns[1]

        # simplify graph and count number of interactions
        n_inter = graph.drop_duplicates(
            subset=[ct, mt]).shape[0]
        
        # remove bidirectional interactions.
        # This interactions will be considered
        # as having no direction
        dir_graph = graph.drop_duplicates(
            subset=[ct, mt], keep=False)
        
        # number of directed interactions
        n_dir_inter = dict(
            # cancer to metastasis interactions
            c_to_m = dir_graph[
                dir_graph.direction=='>'].shape[0],
            # metastasis to cancer interactions
            m_to_c = dir_graph[
                dir_graph.direction=='<'].shape[0]
        )
            
        # add labels                                               
        for dir_, n_dir in n_dir_inter.items():
            pair = {
                'cancer_tissue': ct,
                'metastasis_tissue': mt,
                'simple_interactions': n_inter,
                'directed_interactions': n_dir,
                'direction': dir_
            }
        
            if extra_labels is not None:
                for k, v in extra_labels:
                    pair[k] = v
        
            records.append(pair)

    return records


def jaccard_index(pair, calls, interactions, intersection, direction=False):
    """
    Computes the jaccard index.
    """
    if direction is False:
        A_plus_B = 0
        for tissue in pair:
            genes = calls[tissue].dropna().index.to_list()

            source_inter = interactions[
                interactions.source.isin(genes)]
            target_inter = interactions[
                interactions.target.isin(genes)]

            target_inter.rename(dict(
                source='target',
                target='source'), axis=1, inplace=True)

            graph = pd.concat(
                [source_inter, target_inter], ignore_index=True)

            simp_graph = graph.drop_duplicates().shape[0]

            A_plus_B += simp_graph

        jaccard = intersection/(A_plus_B-intersection)

    else:
        A_plus_B = 0
        if direction == 'c_to_m':
            x = dict(source=0, target=1)
        elif direction == 'm_to_c':
            x = dict(source=1, target=0)    

        for k, v in x.items():
            
            tissue = pair[v]
            genes_source = calls[tissue].dropna().index.to_list()

            forward = interactions[
                interactions[k].isin(genes_source)]
            
            # this step is used to remove bidireccional interactions
            reverse = forward.rename(dict(
                    source='target',
                    target='source'), axis=1)

            forward['keep'] = ['x' for i in range(forward.shape[0])]

            # the "keep" column is used to guarantee that the forward 
            # interactions are not dropped when we use dropna()
            # since the reverse doesn't have this column when we
            # concatenate the dataframes, the values will be fill with
            # nan.
            graph = pd.concat(
                [forward, reverse], ignore_index=True)

            simp_graph = graph.drop_duplicates(
                subset=['source', 'target'], keep=False).dropna()
            
            A_plus_B += simp_graph.shape[0]
        
        jaccard = intersection/(A_plus_B-intersection)

    return jaccard


def weighted_intercell_network(
    pair,
    weights,
    interactions,
    direction=False,
    extra_labels=None
    ):
    """
    Computes the weighted value of a intercellular interactions network.

    Parameters:
    -----------
    pair : list-like with len(2)
        pair of tissues
    
    weights : DataFrame
        DataFrame where indices are the gene symbols and the columns are 
        the tissues. Each value is a weight for a gene in a tissue.

    interactions : DataFrame
        DataFrame with two columns named 'source' and 'target'. Each row 
        represents an interaction between gene in 'source' and gene in
        'target'.

    direction : {False, list_like}, default False
        If interactions are considered as having direction. Name of the
        two directions is required when directed interactions are desired

    extra_labels : dict-like, default None
        Identifiers to add to records

    Returns:
    --------
    records : list
        List of dictionaries where each entry has the score for a
        weighted network with defined conditions.

    """
    records = []
    t1 = pair[0]
    t2 = pair[1]
    
    t1_weights = weights[t1]
    t2_weights = weights[t2]
    
    if direction is False:
        t1_to_t2 = pd.merge(
            interactions,
            t1_weights,
            left_on='source',
            right_on='Gene name')
        t1_to_t2 = pd.merge(
            t1_to_t2,
            t2_weights,
            left_on='target',
            right_on='Gene name')

        t2_to_t1 = pd.merge(
            interactions,
            t2_weights,
            left_on='source',
            right_on='Gene name')
        t2_to_t1 = pd.merge(
            t2_to_t1,
            t1_weights,
            left_on='target',
            right_on='Gene name')

        t2_to_t1.rename(dict(
                source='target',
                target='source'), axis=1, inplace=True)
        
        # build graph
        graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)

        # compute interaction weight
        graph['product'] = graph[t1]*graph[t2]
        graph['min'] = graph[[t1, t2]].agg(func='min', axis=1)
    
        # simplify graph
        simp_graph = graph.drop_duplicates(subset=['source', 'target'])
        for w in ['product', 'min']:
            record = dict(
                tissue1=t1,
                tissue2=t2,
                interaction_weight=w,
                value=simp_graph[w].sum(),
            )

            if extra_labels is not None:

                for k, v in extra_labels:
                    record[k] = v

            records.append(record)
            
    else:
        t1_to_t2 = pd.merge(
            interactions,
            t1_weights,
            left_on='source',
            right_on='Gene name')
        t1_to_t2 = pd.merge(
            t1_to_t2,
            t2_weights,
            left_on='target',
            right_on='Gene name')

        t1_to_t2['direction'] = [
            direction[0] for i in range(t1_to_t2.shape[0])]
    
        t2_to_t1 = pd.merge(
            interactions,
            t2_weights,
            left_on='source',
            right_on='Gene name')
        t2_to_t1 = pd.merge(
            t2_to_t1,
            t1_weights,
            left_on='target',
            right_on='Gene name')

        t2_to_t1['direction'] = [
            direction[1] for i in range(t2_to_t1.shape[0])]
        t2_to_t1.rename(dict(
            source='target',
            target='source'), axis=1, inplace=True)
        
        # build graph
        graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)
        
        # compute interaction weight
        graph['product'] = graph[t1]*graph[t2]
        graph['min'] = graph[[t1, t2]].agg(func='min', axis=1)
        
        # directed graph
        dir_graph = graph.drop_duplicates(
            subset=['source', 'target'], keep=False)

        for w in ['product', 'min']:
            for d in direction:
                
                record = dict(
                    tissue1=t1,
                    tissue2=t2,
                    interaction_weight=w,
                    direction=d,
                    value=dir_graph[dir_graph.direction==d][w].sum(),
                )
                
                if extra_labels is not None:
                    
                    for k, v in extra_labels:
                        record[k] = v

                records.append(record)
    return records


def random_grouped_weighted_intercell_networks(
    weights,
    interactions,
    direction=False,
    iterations=1000,
    n_jobs=1,
    extra_labels=None
):
    
    rng = np.random.default_rng()
    tissues = weights.columns
    pairs = itertools.combinations(tissues, 2)

    records = []
    for p in pairs:

        def compute_rand_dist(graph):
            rand = rng.permuted(graph[[t1, t2]], axis=0)
            rand_weight = rand[:,0]*rand[:,1]
            return np.sum(rand_weight)

        t1=p[0]
        t2=p[1]

        t1_weights = weights[t1]
        t2_weights = weights[t2]

        if direction is False:

            t1_to_t2 = pd.merge(
                interactions,
                t1_weights,
                left_on='source',
                right_on='Gene name')
            t1_to_t2 = pd.merge(
                t1_to_t2,
                t2_weights,
                left_on='target',
                right_on='Gene name')

            t2_to_t1 = pd.merge(
                interactions,
                t2_weights,
                left_on='source',
                right_on='Gene name')
            t2_to_t1 = pd.merge(
                t2_to_t1,
                t1_weights,
                left_on='target',
                right_on='Gene name')

            t2_to_t1.rename(dict(
                    source='target',
                    target='source'), axis=1, inplace=True)

            # build graph
            graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)

            # simplify graph
            simp_graph = graph.drop_duplicates(subset=['source', 'target'])

            # compute network score
            value = (simp_graph[t1]*simp_graph[t2]).sum()

            distribution = Parallel(n_jobs=n_jobs)(
                    delayed(compute_rand_dist)(simp_graph) for i in range(iterations))

            mean = np.mean(distribution)
            std = np.std(distribution)

            record = {
                'cancer_tissue': t1,
                'metastasis_tissue': t2,
                'value': value,
                'dist': distribution,
                'mean': mean,
                'std': std,
                'z_score': (value-mean)/std
            }
            if extra_labels is not None:

                for k, v in extra_labels:
                    record[k] = v

            records.append(record)

        else:
            
            t1_to_t2 = pd.merge(
                interactions,
                t1_weights,
                left_on='source',
                right_on='Gene name')
            t1_to_t2 = pd.merge(
                t1_to_t2,
                t2_weights,
                left_on='target',
                right_on='Gene name')

            t1_to_t2['direction'] = [
                direction[0] for i in range(t1_to_t2.shape[0])]

            t2_to_t1 = pd.merge(
                interactions,
                t2_weights,
                left_on='source',
                right_on='Gene name')
            t2_to_t1 = pd.merge(
                t2_to_t1,
                t1_weights,
                left_on='target',
                right_on='Gene name')

            t2_to_t1['direction'] = [
                direction[1] for i in range(t2_to_t1.shape[0])]
            t2_to_t1.rename(dict(
                source='target',
                target='source'), axis=1, inplace=True)

            # build graph
            graph = pd.concat([t1_to_t2, t2_to_t1], ignore_index=True)

            # directed graph
            dir_graph = graph.drop_duplicates(
                subset=['source', 'target'], keep=False)

            for d in direction:

                dir_graph_d = dir_graph[dir_graph.direction==d]

                # compute network score
                value = (dir_graph_d[t1]*dir_graph_d[t2]).sum()

                distribution = Parallel(n_jobs=n_jobs)(
                    delayed(compute_rand_dist)(dir_graph_d) for i in range(iterations))

                mean = np.mean(distribution)
                std = np.std(distribution)

                record = {
                    'cancer_tissue': t1,
                    'metastasis_tissue': t2,
                    'direction': d,
                    'value': value,
                    'dist': distribution,
                    'mean': mean,
                    'std': std,
                    'z_score': (value-mean)/std
                }
                if extra_labels is not None:

                    for k, v in extra_labels:
                        record[k] = v

                records.append(record)
    return records


def add_controls(df, network):
    """
    Adds all possible control pairs in "network" to the
    organotropism pairs in "df" to perform cancer-wise analysis.
    
    Function to use with the apply method to a DataFrame. Apply to
    a GroupBy object.

    Parameters:
    -----------
    df : DataFrame
        DataFrame of organotropism pairs with the following columns:
            - cancer_tissue
            - metastasis_tissue
            - interactions
            - metastasis_dataset
            - type
            - fdr
    network : DataFrame
        DataFrame of intercellular networks with the following columns:
            - cancer_tissue
            - metastasis_tissue
            - interactions
            - metastasis_dataset
            - type
            - fdr
    
    Returns:
    --------
    all_pairs : DataFrame
        DataFrame with all possible controls
    """
    x = network.loc[
        (network['cancer_tissue']==df['cancer_tissue'].values[0]) &
        (network['interactions']==df['interactions'].values[0])
    ]
    org_pairs = df['metastasis_tissue'].unique()

    # control pairs are the ones not present in organotropism pairs
    cont_pairs = x[~x['metastasis_tissue'].isin(org_pairs)]
    
    cont_pairs['type'] = ['control' for i in range(cont_pairs.shape[0])]
    cont_pairs['fdr'] = ['not_sign' for i in range(cont_pairs.shape[0])]

    metastasis_label = df['metastasis_dataset'].values[0]
    cont_pairs['metastasis_dataset'] =\
        [metastasis_label for i in range(cont_pairs.shape[0])]
    
    all_pairs = pd.concat([df, cont_pairs])
    
    return all_pairs


