#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 14:15:58 2026

@author: chandrima.modak
"""
import os,sys
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
from functools import reduce
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm
import re
import glob
from scipy import stats
import argparse
import multiprocessing as mp

def run_guide_stats(run_procs):
    processed_dir, sample_name, lane_id = run_procs

    directory_path = os.path.join(processed_dir, f"{sample_name}_{lane_id}")
    if not os.path.isdir(directory_path):
        print(f"[MISSING DIR] {directory_path}")
        return (sample_name, lane_id)

    # Be a bit more flexible with naming
    patterns = [
        os.path.join(directory_path, "*_gex_guide.h5ad"),
        os.path.join(directory_path, "*gex*guide*.h5ad"),
    ]

    gex_files = []
    for pat in patterns:
        gex_files.extend(glob.glob(pat))

    gex_files = sorted(set(gex_files))

    if len(gex_files) == 0:
        print(f"[MISSING FILE] No gex guide h5ad in: {directory_path}")
        return (sample_name, lane_id)

    gex_path = gex_files[0]  # if multiple, pick first deterministically
    gex_adata = sc.read_h5ad(gex_path)

    gex_a_df = guide_efficiency_record(gex_adata)

    out_csv = os.path.join(directory_path, f"{sample_name}_guide_count_info.csv")
    gex_a_df.to_csv(out_csv, index=False)

    return (sample_name, lane_id)


def guide_efficiency_record(adata):
    
    rows = []
    
    # NTC mask calculated once
    mask_ntc = adata.obs["guide_id"].astype(str).str.startswith('NTC').to_numpy()

    gRNA_list = adata.obs["guide_id"]
    guide_to_rows = gRNA_list.groupby(gRNA_list).indices
    
    guide_to_gene = {gene: gene.split("_")[0] for gene in guide_to_rows.keys()}


    #Run loop on guide list
    
    for guide, cell_idx in tqdm(guide_to_rows.items(), desc = 'Calculating guide stats'):
        gene = guide_to_gene[guide]
    
    
        # gene must exist in var_names to index
        if gene not in adata.var_names:
            continue
    
    
        # target-guide cells expression
        X = adata[cell_idx, gene].X
        if hasattr(X, "toarray"):
            expr = X.toarray().ravel()
        else:
            expr = np.asarray(X).ravel()
        sum_target = float(expr.sum())
        sumsq_target = float(np.dot(expr, expr))
        n_cells = expr.size
    
        # NTC expression for the same gene
        ntc_X = adata[mask_ntc, gene].X
        if hasattr(ntc_X, "toarray"):
            ntc_expr = ntc_X.toarray().ravel()
        else:
            ntc_expr = np.asarray(ntc_X).ravel()
         
        sum_ntc = float(ntc_expr.sum())
        sumsq_ntc = float(np.dot(ntc_expr, ntc_expr))
        ntc_cells = ntc_expr.size
    
        rows.append({
            "guide_id": guide,
            "target_gene": gene,

            # guide sufficient stats
            "n_cells": n_cells,
            "sum_guide": sum_target,
            "sumsq_guide": sumsq_target,

            # NTC sufficient stats (same guide)
            "ntc_cells": ntc_cells,
            "sum_ntc": sum_ntc,
            "sumsq_ntc": sumsq_ntc,
        })

    
    df = pd.DataFrame(rows)
    

    return df

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
    
    
    args = parser.parse_args()
    exp_meta = pd.read_csv(os.path.join(args.processed_dir, args.expmeta))

    # Drop any Unnamed index-like columns
    exp_meta = exp_meta.loc[:, ~exp_meta.columns.str.startswith("Unnamed")]
    
    # Build list of jobs: one per (colname, value)
    runs_args = [(args.processed_dir, sample_name, lane_id)for lane_id in exp_meta
        for sample_name in exp_meta[lane_id].dropna()]
    
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(processes=args.nprocs) as pool:
        for lane_id, sample_name in pool.imap_unordered(run_guide_stats, runs_args):
            print(f"completed {sample_name}_{lane_id}")
    
    

if __name__ == "__main__":
    main()
