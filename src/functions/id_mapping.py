import pandas as pd
from tqdm import notebook
import numpy as np
import re

def unfold_complexes(
    interactions,
    columns = ['source', 'target'], 
    drop_dups='all',
    ):

    """
    Unfolds complexes in an interactions network. Matches each protein/gene
    in a complex to the interaction partner, be it a single gene/protein or 
    an unfolded complex. It assumes all genes/proteins in a complex interact
    with their partner in a specific interaction.
    
    Parameters
    ----------
    interactions : Pandas DataFrame
        DataFrame with interactions. Each row represents an interaction
        
    columns : list-like, default ["source", "target"]
        columns in "interactions" where to find the interactions. It has to 
        have exactly 2 columns
    
    drop_dups : {'all', 'columns', None}, default 'all'
        Which columns to account when removing duplicates.

    Returns
    -------
    new_interactions : Pandas DataFrame
        DataFrame with all new possible interactions the proteins/genes in an
        unfolded complex can make. Also includes all the interactions previously
        present in the "interactions" DataFrame with duplicated interactions 
        removed.
    
    """

    interactions_dict = interactions.to_dict('records')
    row_list = [] # list(intercell.columns)
    
    for row in interactions_dict:
        for column in columns:
            # regex to find complexes
            comp = re.findall(r'(?<=[:_])([a-zA-Z0-9]*)', row[column])

            if len(comp) > 0:
                row[column] = comp
            else:
                row[column] = [row[column]] # list to iterate in the next step

        for source in row[columns[0]]:
            for target in row[columns[1]]:
                new_row = {i:j for i,j in row.items()}
                new_row[columns[0]] = source
                new_row[columns[1]] = target
                row_list.append(new_row)

    # create new dataframe with all interactions
    new_interactions = pd.DataFrame(row_list)
    if drop_dups == 'all':
        new_interactions = new_interactions.drop_duplicates()
    elif drop_dups == 'columns':
        new_interactions = new_interactions.drop_duplicates(subset=columns)

    return new_interactions


def hgnc_id_mapping(ids_df, hgnc_df, columns=['ensembl_id', 'Description']):
    """
    
    """
    old_id = columns[0]
    new_id = columns[1]

    ids = ids_df[[old_id, new_id]].to_dict('records')
    hgnc_list = hgnc_df['symbol'].tolist()
    ensembl_list = hgnc_df['ensembl_gene_id'].dropna().tolist()
    count_hgnc = 0
    count_ensembl = 0
    count_other = 0
    solve_rows = []
    for row in notebook.tqdm(ids):
        # row = {'Ensembl_id': 'ENSGXXXXXXXXXXX', 'Description': 'YYYYYYY'} 
        # map Ensembl gene id to HGNC id   
        if row[old_id] in ensembl_list: # check ensembl ids
            if ensembl_list.count(row[old_id]) == 1:
                row[new_id] = hgnc_df.loc[hgnc_df['ensembl_gene_id'] ==\
                                            row[old_id], 'symbol'].iloc[0]
                count_ensembl += 1
            else:
            # In the HGNC table the same HGNC symbol can be mapped to more than 
            # 1 ensembl_gene_id
                print(f'id {row[old_id]} is duplicated')
                count_ensembl += 1
        else:
            # this resolves issues with old ensembl_id versions that do not appear in
            # the table
            if row[new_id] in hgnc_list:
                # check hgnc ids
                solve_rows.append(row.copy())
                row[new_id] = np.nan
                count_hgnc += 1
                
            else: 
                # check other ids
                for column in ['alias_symbol', 'prev_symbol']:
                    loop = True
                    for i in hgnc_df[column].dropna():
                        # some values have many ids separated by '|'
                        if row[new_id] in i.split('|'):
                            row[new_id] = hgnc_df.loc[hgnc_df[column] == i, 
                                                                    'symbol'].iloc[0]
                            solve_rows.append(row.copy())
                            row[new_id] = np.nan
                            count_other += 1
                            loop = False
                            break
                    if loop is False:
                        break
                
                else:
                    row[new_id] = np.nan

    print(f'Ensembl ids mapped: {count_ensembl}')
    print(f'HGNC ids to resolve: {count_hgnc}')
    print(f'Other ids to resolve: {count_other}')
    
    new_ids_df = pd.merge(ids_df, pd.DataFrame(ids), how='left', on=old_id)\
        .rename({f'{new_id}_y': new_id}, axis=1)\
            .drop(f'{new_id}_x', axis=1) # drop old ids column
    
    # assert that there are not any duplicates
    assert new_ids_df[new_id].dropna().drop_duplicates().shape[0] ==\
        new_ids_df[new_id].dropna().shape[0], print('duplicates found!!!!')
    
    new_ids_df = solve_id_mapping(new_ids_df, solve_rows, columns)
     
    return new_ids_df


def solve_id_mapping(ids_df, solve_rows, columns):
    """
    
    """
    
    old_id = columns[0]
    new_id = columns[1]
    solved = ids_df[new_id].to_list()
    to_solve = [i[new_id] for i in solve_rows]
    for row in solve_rows:
        # to avoid duplicates, the id must be found only once in the solve_row list and
        # should not be present in ids_rec (solved ids) 
        if to_solve.count(row[new_id]) == 1 and solved.count(row[new_id]) == 0:
            continue
        else:
            print(f'id {row[new_id]} is duplicated')
            row.clear()

    new_ids_df = ids_df.set_index(old_id).combine_first(
        pd.DataFrame(solve_rows).dropna().set_index(old_id)).dropna().reset_index()
    
    # assert that there are not any duplicates
    assert new_ids_df[new_id].dropna().drop_duplicates().shape[0] ==\
        new_ids_df[new_id].dropna().shape[0], print('duplicates found!!!!')
    
    return new_ids_df


def id_conversion(id_list, hgnc_table, column='entrez_id'):
    """
    
    """
    conversion_list = hgnc_table[column].dropna().tolist()
    converted_ids = []
    count_mapped = 0
    for id in notebook.tqdm(id_list):
        if id in conversion_list:
            if conversion_list.count(id) == 1:
                converted_ids.append(
                    hgnc_table.loc[hgnc_table[column] == id, 'symbol'].iloc[0])
                count_mapped += 1
            else:
            # In the HGNC table the same HGNC symbol can be mapped to more than 
            # 1 id from other column
                print(f'id {id} is duplicated')
                count_mapped += 1

    print(f'Mapped ids: {count_mapped}')
    return converted_ids