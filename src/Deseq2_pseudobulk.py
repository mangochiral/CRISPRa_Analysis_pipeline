#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:36:16 2026

@author: chandrima.modak
"""

import os,sys
import scanpy as sc
import numpy as np
import pandas as pd
import anndata 
from plotnine import *
from pathlib import Path
from tqdm.auto import tqdm
import re
import glob
from scipy import stats, sparse
import pertpy as ptp
import multiprocessing as mp
import argparse  


def _get_features(directory_path):
    """Selecting highly variable genes for DGE""" 
    
    pattern = glob.glob(os.path.join(directory_path, 'DE_feature_selection_vars.csv'))[0]
    
    features_selected = pd.read_csv(pattern)
    
    is_target = features_selected[features_selected['highly_variable']]
    
    gene_list =  is_target.gene_name.unique().tolist()
    
    return gene_list

def _make_model(pbulk_adata, design = '~ log10_n_cells  + target_gene', fit_cpus = 8):
    
    ## Define design of test
    # we use log10_n_cells as a confounder to regress out
    # no need to include condition here, since we already subset to one condition at a time
    design_formula = design
    
    ## Fit Negative binomial GLM model on all targets at once
    model = ptp.tl.PyDESeq2(pbulk_adata, design=design_formula)
    
    model.fit(n_cpus = fit_cpus, quiet=False)

    return model

def _do_contrast(args):
    
    '''Uses chunk adata to make DGE negative binomial GLM  model and does pairwise between targeting vs non-targeting'''
    
    chunk_adata, chunk_targets, control, fit_cpus = args
    
    model = _make_model(chunk_adata, fit_cpus=fit_cpus)
    
    # Do pairwise comparison with the NTCs
    paircomp = {t:(model.cond(target_gene = t) - model.cond(target_gene = control)) 
                for t in chunk_targets
                if t != control
                }
    
    # Wald test and FDR correction
    dge_df_chunks = model.test_contrasts(paircomp)
    
    return dge_df_chunks

class run_DEseq2:
    def __init__(self, path, pbulk_adata, condition):
        self.path = path
        self.pbulk_adata = pbulk_adata
        self.all_dges = pd.DataFrame()
        self.condition = condition
        

    def _batch_dge_jobs(self,min_counts_per_gene = 10, end_idx=None, batch_size=50, num_cores = None, fit_cpus = None ):
        
        """
        Parameters
        ----------
        directory_path : 
            Path to parent directory
        min_counts_per_gene:
            minimum counts required for each gene
        num_cores : int, optional
            Number of cores for parallel processing.
        n_iter : int
            Number of iterations.
        end_idx : int, optional
            End index for gRNA list.
        batch_size : int
            Size of batches for processing.
    
        Returns
        -------
        all_dge : concated dataframe for DGEs
        """
        
        
        
        # pbulk_adata = sc.read_h5ad(pattern)
        self.pbulk_adata = self.pbulk_adata[:, self.pbulk_adata.var['gene_name'].isin(gene_list)].copy()
        
        all_targets = self.pbulk_adata.obs['target_gene'].unique().tolist()
        
        gene_list = _get_features(self.path)
        
        # Making chunks of DGE csv outputs
        output_dir = os.path.join(self.path, 'DE_outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        max_cpus = min(mp.cpu_count(), slurm_cpus)
    
        if num_cores is None:
            num_cores = max_cpus
        else:
            num_cores = min(int(num_cores), max_cpus)
        
        if fit_cpus is None:
            fit_cpus = max(1, max_cpus // num_cores)
        
        # Determine chunk boundaries
        if end_idx is None:
            end_idx = len(all_targets)
        else:
            end_idx = min(end_idx, len(all_targets))
            
        all_dge = []
        runs_args =[]
        for i in range(0, end_idx, batch_size): 
            chunk_targets = all_targets[i:i+batch_size]
            if 'NTC' not in chunk_targets:
                chunk_targets.append('NTC')
                
            chunk_adata =  self.pbulk_adata[self.pbulk_adata.obs.target_gene.isin(chunk_targets)].copy()
            
            runs_args.append((chunk_adata, chunk_targets))
                
        ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
        with ctx.Pool(processes=num_cores) as pool:
            for indices, dge_df_chunks in enumerate(pool.imap(_do_contrast, runs_args)):
                dge_df_chunks.to_csv(os.path.join(output_dir, f"DGE_chunk_{indices}.csv"))
                all_dge.append(dge_df_chunks)
        
        self.all_dges = pd.concat(all_dge)
        
        return self.all_dges

    def parse_DE_results_2_adata(self):
        all_dfs ={}
        for stat in ['baseMean', 'log_fc', 'lfcSE', 'p_value','adj_p_value']:
            stat_df = self.all_dges.pivot(values=stat, columns='variable', index='contrast')
            all_dfs[stat] = stat_df
    
        DE_anndata = anndata.AnnData(
            layers = all_dfs.copy()
        )
    
        DE_anndata.obs_names = all_dfs['log_fc'].index.tolist()
        DE_anndata.var_names = all_dfs['log_fc'].columns.tolist()
        DE_anndata.obs['contrast'] = DE_anndata.obs_names.values
        DE_anndata.obs['condition'] = self.condition
        DE_anndata.obs_names =  DE_anndata.obs['condition'] + '_' + DE_anndata.obs['contrast']
        return(DE_anndata)

    
def main():
    parser = argparse.ArgumentParser(description="Get Pbulk adata path")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to processed directory",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=mp.cpu_count(),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Array task ID — processes a single run from the list",
    )
    args = parser.parse_args()

    
    
    pattern = glob.glob(os.path.join(args.processed_dir, '*_DE_pseudobulk_for_test.h5ad'))[0]
    
    pbulk_adata = sc.read_h5ad(pattern)
    
    conditions = pbulk_adata.obs['condition'].unique().tolist()
    
    for c in conditions:
        adata_c = pbulk_adata[pbulk_adata.obs['condition'] == c].copy()
        adata_run = run_DEseq2(args.processed_dir, adata_c, c)
        
        adata_run._batch_dge_jobs(num_cores=args.nprocs)
        
        adata_run.parse_DE_results_2_adata().write_h5ad(
            os.path.join(args.processed_dir, f"DE_{c}_anndata.h5ad")       )

    

if __name__ == "__main__":
    main()  
    
    
