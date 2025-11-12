#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:50:38 2025

@author: chandrima.modak
"""

'''
Preprocess and merge Perturb-seq data.
'''
import os,sys
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from pathlib import Path
import warnings
from tqdm import tqdm
import xarray
import pytest

warnings.filterwarnings('ignore')

def _split_assay(xdata, exp, sample_name, lane):
    '''
        Parameters
        ----------
        xdata : Here the input data is 10x cellranger sample filtered assay matrix .h5/h5ad data.
        exp: Here input has to crispr
        sample_name: str experiment construct name
        lane_id: str lane id 
        Returns gene expression and or crispr assay adata
        -------
        None.

    '''
    
    try:
        if exp.lower() == 'crispr':
            a = sc.read_10x_h5(xdata, gex_only = False)
        else:
            a = sc.read_10x_h5(xdata)
    except ValueError:
        a = sc.read_h5ad(xdata)
    # Labeling the construct and lane name    
    a.obs['library_id'] = sample_name
    a.obs['lane_id'] = lane
    a.obs_names = a.obs_names + '_' + a.obs['lane_id']+'_' + a.obs['library_id']
    # split by modality (for Perturb-seq)
    if not all(a.var['feature_types'] == 'Gene Expression'):
        gex_a = a[:, a.var['feature_types'] == 'Gene Expression'].copy()
        gex_a.var.drop(['pattern', 'read', 'sequence'], axis=1, inplace=True, errors='ignore')
        gex_a.var['gene_name'] = gex_a.var_names.values
        gex_a.var_names = gex_a.var['gene_ids'].values
        crispr_a = a[:, a.var['feature_types'] != 'Gene Expression'].copy()
        return(gex_a, crispr_a)
    else:
        a.var.drop(['pattern', 'read', 'sequence'], axis=1, inplace=True, errors='ignore')
        a.var['gene_name'] = a.var_names.values
        a.var_names = a.var['gene_ids'].values
        return(a)
    
def _basic_qc_gex(adata, filter_cells=True):
    """
    Perform basic QC on gene expression data
    input: Processed gex_a  
    """
    ## Basic QC metrics
    adata.var["mt"] = adata.var['gene_name'].str.startswith("MT-")  # "MT-" for human, "Mt-" for mouse
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=True, inplace=True)

    if filter_cells:
        # print(f"Cells before filtering: {adata.n_obs}")
        pre_filter_cells = adata.n_obs
        adata = adata[adata.obs['pct_counts_mt'] < 20].copy()
        sc.pp.filter_cells(adata, min_genes=200)
        post_filter_cells = adata.n_obs
        # print(f"Cells after filtering: {adata.n_obs}")
    
    return adata, pre_filter_cells, post_filter_cells

def _compute_nonzero_means_v1(X_mat):
    if X_mat.format == 'csc':
        nnz_per_col = np.diff(X_mat.indptr)
    else:
        # Convert to CSC temporarily for column operations
        X_csc = X_mat.tocsc()
        nnz_per_col = np.diff(X_csc.indptr)
    
    col_sums = np.asarray(X_mat.sum(axis=0)).ravel()
    
    return np.divide(col_sums, nnz_per_col, 
                    out=np.zeros_like(col_sums), 
                    where=nnz_per_col != 0)


def get_sgrna_qc_metrics(crispr_a, min_sgrna_counts=3, q=0.05):
    var_cols = ['sgrna_id','perturbed_gene_name', 'perturbation_type', 'feature_types', 'genome', 'pattern', 'read', 'sequence',
       'n_cells', 'mean_counts', 'total_counts', 'nonz_means']
    # Sanitize excel problems
    crispr_a.var_names = np.where(crispr_a.var_names == '1-Jun', 'JUN-1', crispr_a.var_names)
    crispr_a.var_names = np.where(crispr_a.var_names == '2-Jun', 'JUN-2', crispr_a.var_names)
    # Compute mean of non-zero UMIs
    crispr_a.var['nonz_means'] = _compute_nonzero_means_v1(crispr_a.X)
    sc.pp.calculate_qc_metrics(crispr_a, inplace=True)
    perturb_metadata = crispr_a.var.copy()
    # Annotate perturbed gene
    perturb_metadata['perturbation_type'] = perturb_metadata.index.str.extract(r'(CRISPRi|CRISPRa|NTC)', expand=False)
    perturb_metadata['perturbed_gene_name'] = perturb_metadata.index.str.replace('-', '_').str.split('_').str[0]
    perturb_metadata['sgrna_id'] = perturb_metadata.index
    perturb_metadata = perturb_metadata.rename(
        {'n_cells_by_counts':'n_cells'}, 
        axis=1) 
    crispr_a.var = perturb_metadata[var_cols].copy()

def test_get_sgrna_qc_metrics(crispr_a):
    expected_cols = ['sgrna_id','perturbed_gene_name', 'perturbation_type', 'feature_types', 'genome', 'pattern', 'read', 'sequence',
       'n_cells', 'mean_counts', 'total_counts', 'nonz_means']
    assert list(crispr_a.var.columns) == expected_cols


def process_cellranger_h5(xdata, exp, sample_name, lane):
    '''
    Process single cellranger file
    file_path: h5 or h5ad cellranger filtered data
    exp: experiment name
    sample_name: experiment run
    '''
# =============================================================================
#     file_path = f'{datadir}/CRISPRia_Cellanome_{lane}/per_sample_outs/{sample_name}/count/sample_filtered_feature_bc_matrix.h5'    
#     gex_a, crispr_a = _split_assay(file_path, exp)
#     gex_a.obs['library_id'] = sample_name
#     gex_a.obs['lane_id'] = lane
#     crispr_a.obs['library_id'] = sample_name
#     crispr_a.obs['lane_id'] = lane
#     gex_a.obs_names = gex_a.obs_names + "_" + gex_a.obs['lane_id'] + "_" + gex_a.obs['library_id'] 
#     crispr_a.obs_names = crispr_a.obs_names + "_" + crispr_a.obs['lane_id'] + "_" + crispr_a.obs['library_id']
#     
# =============================================================================
    gex_a, crispr_a = _split_assay(xdata, exp, sample_name, lane)
    # Process sgRNA adata
    get_sgrna_qc_metrics(crispr_a, min_sgrna_counts=3, q=0.05)
    if 'i' in sample_name:
        crispr_a = crispr_a[:, crispr_a.var['perturbation_type'] != 'CRISPRa']
    elif 'a' in sample_name:
        crispr_a = crispr_a[:, crispr_a.var['perturbation_type'] != 'CRISPRi']
    

    # Process scRNA adata
    gex_a.layers['counts'] = gex_a.X.copy()
    # rsc.get.anndata_to_GPU(gex_a)
    gex_a, pre_count, post_count = _basic_qc_gex(gex_a)
    sc.pp.normalize_total(gex_a)
    sc.pp.log1p(gex_a)
    return gex_a, crispr_a, pre_count, post_count

