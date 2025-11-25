#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:35:45 2025

@author: chandrima.modak
"""
import os
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import crispat as cr
import matplotlib.pyplot as plt
import anndata as ad
import scipy as sp
from functools import reduce
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from tqdm.auto import tqdm
import re
import glob


# =============================================================================
# Checking for background to signal ratio
# =============================================================================
def get_long_sgrna_umi_counts(grna_ad):
    """
    Convert sgRNA AnnData object to long format DataFrame with UMI counts.
    
    Parameters
    ----------
    grna_ad : anndata.AnnData
        AnnData object containing sgRNA UMI counts
        
    Returns
    -------
    pandas.DataFrame
        Long format DataFrame with columns:
        - cell: cell barcode
        - guide: guide ID
        - UMI_counts: number of UMIs
    """
    # Get the UMI count matrix
    X = grna_ad.X

    # Convert sparse matrix to coordinate format if needed
    if hasattr(X, 'tocoo'):
        X_coo = X.tocoo()
    else:
        # If X is dense, convert to sparse first
        X_coo = sp.csr_matrix(X).tocoo()

    long_df = pd.DataFrame({
        'cell': grna_ad.obs_names[X_coo.row],
        'gRNA': grna_ad.var_names[X_coo.col], 
        'UMI_counts': X_coo.data
    })
    
    # Filter out zero counts and reset index
    long_df = long_df[long_df['UMI_counts'] > 0].reset_index(drop=True)
    
    return long_df

def get_background_vs_signal_guide_counts(crispr_a, assigned_guides_csv):
    '''
    Analyze sgRNA counts to distinguish between background and signal for all samples in a directory.
    
    Args:
        datadir: crispr adata
        
        
    Returns:
        summary of background to signal ratio
    '''
    
    all_umi_counts = get_long_sgrna_umi_counts(crispr_a)
    assigned_guides = pd.read_csv(assigned_guides_csv)

    all_guides = pd.merge(all_umi_counts, assigned_guides, on=['cell','gRNA'], how='left', suffixes=['_all', '_assigned'])
    all_guides['signal_vs_bg'] = np.where(all_guides['UMI_counts_assigned'].isna(), 'background', 'signal')

    return all_guides

# =============================================================================
# Checking guide efficiency
# =============================================================================

def guide_type_efficiency(gex_adata, sample_name, remove_per_type):
    """
    Analyze the efficiency of guides and ranking them
    
    Parameters
    ----------
    e
    
    Returns
    --------
    
    """
    adata_obj = gex_adata[gex_adata.obs.guide_id.notna(), :]
    adata_obj_single_guide = adata_obj[adata_obj.obs.guide_id != 'multi_sgRNA', :]
    target_genes = adata_obj_single_guide.var_names.to_list()
    
    # Initializing
    pseudocount = 5e-2
    effect_sizes = []
    guides = []
    pvalues = []
    fcs = []
    basal_exp = []
    gene_exp = []
    num_cells = []
    
    for i in tqdm(adata_obj_single_guide.obs.guide_id.astype(str).unique()):
        if i.split("_")[0] not in target_genes or i.replace("-","_").split("_")[0] == 'NTC':
            continue
        else:
            # print(i)
            perturb_counts = adata_obj_single_guide[adata_obj_single_guide.obs['guide_id'] == i, i.split("_")[0]].X.toarray().flatten()
            ncells = perturb_counts.size
            mean_perturb = np.mean(perturb_counts)
            ntc_counts = adata_obj_single_guide[adata_obj_single_guide.obs['guide_id'].str.startswith("NTC"), i.split("_")[0]].X.toarray().flatten()
            mean_ntc = np.mean(ntc_counts)
            effect_size = pg.compute_effsize(perturb_counts, ntc_counts, paired=False, eftype='cohen')
            if 'a' in sample_name.split("_")[1]:
                pvalue = sp.stats.mannwhitneyu(perturb_counts, ntc_counts, alternative='greater').pvalue
                fc = (np.mean(perturb_counts)+pseudocount)/(np.mean(ntc_counts)+pseudocount)
                fc = (mean_perturb+pseudocount)/(mean_ntc+pseudocount)
            elif 'i' in sample_name.split("_")[1] :
                pvalue = sp.stats.mannwhitneyu(perturb_counts, ntc_counts, alternative='less').pvalue
                fc = (mean_ntc+pseudocount)/(mean_perturb+pseudocount)
    
            # Appending all the values
            effect_sizes.append(effect_size)
            guides.append(i)
            pvalues.append(pvalue)
            fcs.append(fc)
            gene_exp.append(mean_perturb)
            basal_exp.append(mean_ntc)
            num_cells.append(ncells)
    
    results_df = pd.DataFrame({
        f'{sample_name}_guide': guides,
        f'{sample_name}_pvalue': pvalues,
        f'{sample_name}_fold_change': fcs,
        f'{sample_name}_effect_size': effect_sizes,
        f'{sample_name}_gene_exp': gene_exp,
        f'{sample_name}_basal_exp': basal_exp,
        f'{sample_name}_num_of_cells': num_cells
    })
    results_df[f'{sample_name}_fdr'] = multipletests(results_df[f'{sample_name}_pvalue'].fillna(1.0), method='fdr_bh')[1]
    return results_df