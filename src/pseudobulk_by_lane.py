#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:46:58 2026

@author: chandrima.modak
"""

import os,sys
import anndata 
import scanpy as sc
import pandas as pd
import numpy as np
import glob
import argparse
from scipy import sparse
from tqdm import tqdm
import multiprocessing as mp

def make_pseudobulk(adata, condition_col='lane_id', sgrna_col='gRNA'):
    """
    Create pseudobulk data from single-cell RNA-seq data.
    
    Parameters:
    -----------
    adata : 
        Anndata single-cell 
    condition_col : str, optional
        Column name for condition information, default is 'culture_condition'
    sgrna_col : str, optional
        Column name for sgRNA information, default is 'guide_id'
    """
    
    sample_cols = [sgrna_col,condition_col]
    adata.obs["sample_id"] = adata.obs[sample_cols].apply(lambda x: "_".join(x), axis=1)
    n_cells_obs = adata.obs.value_counts(['sample_id'] + ['library_id', 'target_gene', 'sgRNA_type']
                                         + sample_cols).reset_index()
    n_cells_obs = n_cells_obs.set_index('sample_id').rename({'count':'n_cells'}, axis=1)
    pbulk_adata = sc.get.aggregate(adata, by=['sample_id'], func=['sum'])
    pbulk_adata.obs = n_cells_obs.loc[pbulk_adata.obs_names].copy()
    pbulk_adata.layers['sum'] = sparse.csr_matrix(pbulk_adata.layers['sum'])
    return pbulk_adata

def run_pbulk_jobs(run):
    # Worker for a single (colname, value) job.
    processed_dir, colname, value, condition_col, sgrna_col = run

    directory_path = os.path.join(processed_dir, f"{colname}_{value}")
    if not os.path.isdir(directory_path):
        print(f"{value}_{colname} not found")
        return (colname, value)

    expected_out = os.path.join(directory_path, f"{colname}_DE_pseudobulk.h5ad")
    if os.path.isfile(expected_out):
        print(f"{expected_out} exists, skipping...")
        return (colname, value)

    
    gex_singlets_pattern = glob.glob(os.path.join(directory_path, "*_gex_singlets.h5ad"))

    if len(gex_singlets_pattern) == 0:
        print(f"Missing preprocessed h5ad in {directory_path}, skipping.")
        return (colname, value)

    gex_singlets = sc.read_h5ad(gex_singlets_pattern[0])
    
    pbulk_adata = make_pseudobulk(gex_singlets, condition_col, sgrna_col)
    
    pbulk_adata.write_h5ad(
        os.path.join(directory_path, f"{colname}_DE_pseudobulk.h5ad")
        )
    return (colname, value)
    
def main():
    parser = argparse.ArgumentParser(description="Processing QC of guide assignment")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to directory that processed dir",
    )
    parser.add_argument(
        "--expmeta",
        type=str,
        required=True,
        help="Experiment metadata CSV filename",
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
        help="Array task ID - processes single run from list",
    )
    parser.add_argument(
        '--condition_col', 
        type=str, 
        default='lane_id',
        help='Column name for condition information'
        )
    parser.add_argument(
        '--sgrna_col', type=str, 
        default='guide_id',
        help='Column name for sgRNA information'
        )
    args = parser.parse_args()
    exp_meta = pd.read_csv(os.path.join(args.processed_dir, args.expmeta))

    # Drop any Unnamed index-like columns
    exp_meta = exp_meta.loc[:, ~exp_meta.columns.str.startswith("Unnamed")]
    
    # Build list of jobs: one per (colname, value)
    runs_args = [(args.processed_dir, sample_name, lane_id, args.condition_col, args.sgrna_col)for lane_id in exp_meta
        for sample_name in exp_meta[lane_id].dropna()]
    
    # If task_id is provided, process only that single run
    if args.task_id is not None:
        if args.task_id >= len(runs_args):
            print(f"Task ID {args.task_id} exceeds number of runs ({len(runs_args)}). Exiting.")
            return
        
        print(f"Task {args.task_id}: processing single run (total runs: {len(runs_args)})")
        my_runs = [runs_args[args.task_id]]
    else:
        # Process all runs with multiprocessing
        print(f"Processing all {len(runs_args)} runs with {args.nprocs} workers")
        my_runs = runs_args

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(processes=args.nprocs) as pool:
        # FIX: Changed variable names from (colname, value) to (sample_name, lane_id)
        for sample_name, lane_id in pool.imap_unordered(run_pbulk_jobs, my_runs):
            print(f"completed {sample_name}_{lane_id}")
    
    

if __name__ == "__main__":
    main()
