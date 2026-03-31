#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:54:57 2026

@author: chandrima.modak
"""

from functools import reduce
from typing import Optional
import scanpy as sc
import anndata
import pytest
import numpy as np
import pandas as pd
import warnings
import os, sys
import scipy.sparse as sp

warnings.filterwarnings('ignore')
class Adata_Qc_preprocessing:

    # 1. Fetch h5d file loc
    def __init__(self, path : str, 
                 exp: str, 
                 library_id : str, 
                 lane: str):
        """
        Perform basic QC on gene expression data
        
        Parameters:
        ----------------------------------------------
        path: Experiment cellranger path
        exp: CRISPR perturb seq
        libray_id: Experiment details
        lane_id : lane id
        
        Returns:
        -----------------------------------------------
        Class objects
        """
        self.path = path
        self.exp = exp
        self.library_id = library_id
        self.lane = lane
        
        # Will be populated 
        self.adata = None
        self.gex_a = None
        self.crispr_a = None
        self.pre_filter_cells: int = 0
        self.post_filter_cells: int = 0
     
    def get_assay(self):
        try:
            # For perturb seq data
            if self.exp.lower() == 'crispr':
                self.adata = sc.read_10x_h5(self.path, gex_only=False)
            else:
                self.adata = sc.read_10x_h5(self.path)
                
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"Cannot read {self.path}: {exc}") from exc

        self.adata.obs['library_id'] = self.library_id
        self.adata.obs['lane_id'] = self.lane
        self.adata.obs_names = self.adata.obs_names + '_' + self.adata.obs['lane_id']+'_' + self.adata.obs['library_id']
            
        return self.adata
        


    @classmethod
    def clean_prefix(cls, adata, prefix):
        """
        Removes prefix if present
        """
        adata.var_names = adata.var_names.str.replace(prefix, '', regex=True)
        return adata

    @classmethod
    def columns_in_adata(cls, adata):
        """
        Unit test to cell the expected cols exists on the raw anndata h5 object
        """
        expected_cols = ['gene_ids', 'feature_types', 'genome', 'pattern', 'read', 'sequence']
        assert list(adata.var.columns) == expected_cols
    
    # 2. Split assay
    @classmethod
    def split_assay(cls, adata):
        """
        Parameters:
        ----------------------------------------------
        Adata: anndata object
        Returns:
        -----------------------------------------------
        CRISPR anndata 
        Gene Expression anndata
        """
        if not all(adata.var['feature_types'] == 'Gene Expression'):
            gex_a = adata[:, adata.var['feature_types'] == 'Gene Expression'].copy()
            gex_a.var.drop(['pattern', 'read', 'sequence'], axis=1, inplace=True, errors='ignore')
            gex_a.var['gene_name'] = gex_a.var_names.values
            crispr_a = adata[:, adata.var['feature_types'] != 'Gene Expression'].copy()
            return(gex_a, crispr_a)
        else:
            adata.var.drop(['pattern', 'read', 'sequence'], axis=1, inplace=True, errors='ignore')
            adata.var['gene_name'] = adata.var_names.values
            return adata, None
            
    # 3. Basic Qc of gene expression
    def _basic_qc_gex(self, 
                      mt_pct: float,
                      filter_cells: bool):
        """
        Perform basic QC on gene expression data
        
        Parameters:
        ----------------------------------------------
        Gene Expression anndata object
        mt_pct: mitonchondrial % threshold
        filter_cells: True/False
        
        Returns:
        -----------------------------------------------
        Post filtered  Gene Expression anndata object
        Pre-Filter and Post-Filter Stats objects
        """
        ## Basic QC metrics
        self.gex_a.var["mt"] = self.gex_a.var['gene_name'].str.startswith("MT-")  # "MT-" for human, "Mt-" for mouse
        sc.pp.calculate_qc_metrics(self.gex_a, qc_vars=["mt"], log1p=True, inplace=True)
    
        
        self.pre_filter_cells = self.gex_a.n_obs 
        if filter_cells:
            self.gex_a = self.gex_a[self.gex_a.obs['pct_counts_mt'] < mt_pct].copy()
            sc.pp.filter_cells(self.gex_a, min_genes=200, inplace= filter_cells)
        self.post_filter_cells = self.gex_a.n_obs
        
        return self.gex_a, self.pre_filter_cells, self.post_filter_cells

        
    # 4.1. mean sgRNA umi counts
    @staticmethod
    def _compute_nonzero_means_v1(X_mat):
        X_csc = X_mat.tocsc() if not sp.issparse(X_mat) or X_mat.format != "csc" else X_mat
        nnz_per_col = np.diff(X_csc.indptr)
        col_sums = np.asarray(X_mat.sum(axis=0)).ravel()
        
        return np.divide(col_sums, nnz_per_col, 
                        out=np.zeros_like(col_sums), 
                        where=nnz_per_col != 0)
        
    # 4. Basic Qc of cripsr assay
    def get_sgrna_qc_metrics(self):
        var_cols = ['sgrna_id','perturbed_gene_name','feature_types', 'genome', 'pattern', 'read', 'sequence',
           'n_cells', 'mean_counts', 'total_counts', 'nonz_means']
        # Sanitize excel problems
        self.crispr_a.var_names = np.where(self.crispr_a.var_names == '1-Jun', 'JUN-1', self.crispr_a.var_names)
        self.crispr_a.var_names = np.where(self.crispr_a.var_names == '2-Jun', 'JUN-2', self.crispr_a.var_names)
        
        # 4.2 Compute mean of non-zero UMIs
        self.crispr_a.var['nonz_means'] = Adata_Qc_preprocessing._compute_nonzero_means_v1(self.crispr_a.X)
        sc.pp.calculate_qc_metrics(self.crispr_a, inplace=True)
        perturb_metadata = self.crispr_a.var.copy()
        
        # Annotate perturbed gene
        # perturb_metadata['perturbation_type'] = perturb_metadata.index.str.extract(r'(CRISPRi|CRISPRa|NTC)', expand=False)
        perturb_metadata['perturbed_gene_name'] = perturb_metadata.index.str.replace('-', '_').str.split('_').str[0]
        perturb_metadata['sgrna_id'] = perturb_metadata.index
        perturb_metadata = perturb_metadata.rename(
            {'n_cells_by_counts':'n_cells'}, 
            axis=1) 
        self.crispr_a.var = perturb_metadata[var_cols].copy()
        return self.crispr_a


# =============================================================================
# Execution point for the code
# =============================================================================
    
# Executing the pre-processing
def process_cellranger_h5(path: str, 
                          exp: str, 
                          sample_name:str, 
                          lane:str,
                         prefix: str | None = None,
                         mt_pct: float = 20.0, filter_cells = True):
    '''
    Process single cellranger file
    file_path: h5 or h5ad cellranger filtered data
    exp: experiment name
    sample_name: experiment run
    '''
    
    anndata_obj = Adata_Qc_preprocessing(path, exp, sample_name, lane)
    anndata_obj.get_assay()
    anndata_obj.columns_in_adata(anndata_obj.adata)
    try:
        if prefix is not None:
            if anndata_obj.adata.var_names.str.startswith(prefix).any():
                anndata_obj.clean_prefix(anndata_obj.adata, prefix)
        
        anndata_obj.gex_a, anndata_obj.crispr_a = anndata_obj.split_assay(anndata_obj.adata)
        anndata_obj._basic_qc_gex(mt_pct, filter_cells)
        anndata_obj.get_sgrna_qc_metrics()
    except Exception as e:
        print(f"Problem: {e}")
    
    # Process scRNA adata
    anndata_obj.gex_a.layers['counts'] = anndata_obj.gex_a.X.copy()
    sc.pp.normalize_total(anndata_obj.gex_a)
    sc.pp.log1p(anndata_obj.gex_a)
    return anndata_obj.gex_a, anndata_obj.crispr_a, anndata_obj.pre_filter_cells, anndata_obj.post_filter_cells

    
    
    