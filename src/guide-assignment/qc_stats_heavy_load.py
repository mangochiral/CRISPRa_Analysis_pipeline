
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

def parse_bool(x):
    """Accept True/False/1/0/yes/no from CLI."""
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {x}")

def run_guide_stats(run_procs):
    processed_dir, sample_name, lane_id, multiguide = run_procs
    multiguide = parse_bool(multiguide)

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
    
    if multiguide:
        gex_adata = clean_adata_for_multiguide(gex_adata)
    
        gex_adata.write_h5ad(os.path.join(directory_path, f"{sample_name}_gex_multiguide.h5ad"))
    
    gex_a_df, gex_a_df_guide_cells_eff = guide_efficiency_record(gex_adata, multiguide=multiguide)

    out_csv_1 = os.path.join(directory_path, f"{sample_name}_guide_count_info.csv")
    out_csv_2 = os.path.join(directory_path, f"{sample_name}_cells_based_guide_efficiency.csv")
    gex_a_df.to_csv(out_csv_1, index=False)
    gex_a_df_guide_cells_eff.to_csv(out_csv_2, index=False)

    return (sample_name, lane_id)

def clean_adata_for_multiguide(adata):
    adata.var_names = adata.var.gene_name
    
    adata.obs['sgRNA_type'] = 'Single sgRNA'
    
    adata.obs.loc[adata.obs['guide_id'].isna(), 'sgRNA_type'] = 'no sgRNA'
    
    adata.obs.loc[adata.obs['guide_id'].str.startswith('NTC', na=False), 'sgRNA_type'] = 'single NTC sgRNA'
    
    adata.obs.loc[adata.obs['guide_id'].str.startswith('multi', na=False), 'sgRNA_type'] = 'multi sgRNA'
    
    # Order the rows
    type_order = ['Single sgRNA','single NTC sgRNA','multi sgRNA','no sgRNA']
    adata.obs['sgRNA_type'] = pd.Categorical(adata.obs['sgRNA_type'],categories=type_order,ordered=True)
    
    adata = adata[adata.obs.sgRNA_type != 'no sgRNA'].copy()
    max_sgRNA = adata.obs.groupby('sgRNA_type')['total_counts'].max()
    cut_off = max_sgRNA.loc['Single sgRNA',]
    adata = adata[~((adata.obs["sgRNA_type"] == "multi sgRNA") & (adata.obs["total_counts"] > cut_off))].copy()
    return adata

def guide_efficiency_record(adata, multiguide = False):
    
    rows = []
    
    if multiguide:
        
        if "guide_matrix" not in adata.obsm:
            raise KeyError("multiguide=True but adata.obsm['guide_matrix'] is missing.")

        
        # For NTC cells with multi guides
        guide_matrix = adata.obsm["guide_matrix"] 
        
        # Expect pandas DataFrame for this logic
        if not hasattr(guide_matrix, "columns"):
            raise TypeError("adata.obsm['guide_matrix'] must be a pandas DataFrame with .columns for multiguide mode.")
            
            
        find_ntc_columns = guide_matrix.columns.str.startswith('NTC').astype(int)
        
        # Find cells with any NTC
        cells_with_ntc = np.dot(guide_matrix, find_ntc_columns)
        
        # Sum all NTC present in a cell
        guide_cell_ntc = guide_matrix.sum(axis=1).to_numpy()

        # Find true NTC cells irrespective of multiguide or single NTC guide
        true_ntc_mask = (guide_cell_ntc - cells_with_ntc) == 0

        # Replace guide_id of multiguide values and NTC- with NTC
        adata.obs['guide_id'] = adata.obs['guide_id'].astype(str)
        adata.obs.loc[true_ntc_mask, 'guide_id'] = 'NTC'

        # For Cells with single and multiguides 
        guide_list = guide_matrix.columns.to_list()
        guide_to_gene = {gene: gene.split("-")[0] for gene in guide_list if not gene.startswith('NTC-')} 
        
        guide_to_rows = {guide_id:guide_matrix.index[guide_matrix[guide_id] == 1].tolist() for guide_id in guide_matrix.columns if not guide_id.startswith('NTC-')}
        
    
    else:
        gRNA_list = adata.obs["guide_id"]
        guide_to_rows = gRNA_list.groupby(gRNA_list).indices
    
        guide_to_gene = {gene: gene.split("_")[0] for gene in guide_to_rows.keys()}
    
    # NTC mask calculated once
    mask_ntc = adata.obs["guide_id"].astype(str).str.startswith('NTC').to_numpy()
    
    # For targeted gene NTC Expression to calculate the pseudo p value
    ntc_expr_sorted_cache = {}
    guide_dfs = []


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
        
        # Pseudo P value to check cells was KO efficiency
        if gene  not in ntc_expr_sorted_cache:
            ntc_expr_sorted_cache[gene] = np.sort(ntc_expr)
        
        ntc_expr_sorted = ntc_expr_sorted_cache[gene]

        if len(ntc_expr_sorted) == 0:
            fractions = np.full(expr.shape, np.nan, dtype=float)
        else:
            fractions = np.searchsorted(ntc_expr_sorted, expr, side="left") / len(ntc_expr_sorted)

        
        guide_dfs.append(
            pd.DataFrame({
                "guide_id": guide,
                "cell_id": cell_idx,
                "expr": expr,
                "ntc_fraction": fractions,
            })
        )
    
    df = pd.DataFrame(rows)
    
    df_guide_cells_eff = pd.concat(guide_dfs, ignore_index=True) if guide_dfs else pd.DataFrame(columns=["guide_id", "cell_id", "expr", "ntc_fraction"])
    
    
    
    

    return df, df_guide_cells_eff

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
    parser.add_argument("--multiguide", 
                        type=parse_bool, 
                        default=False, 
                        required=False,
                        help="Run multiguide cleanup + guide_matrix mode (true/false)."
                        )

    
    args = parser.parse_args()
    exp_meta = pd.read_csv(os.path.join(args.processed_dir, args.expmeta))

    # Drop any Unnamed index-like columns
    exp_meta = exp_meta.loc[:, ~exp_meta.columns.str.startswith("Unnamed")]
    
    # Build list of jobs: one per (colname, value)
    runs_args = [(args.processed_dir, sample_name, lane_id, args.multiguide)for lane_id in exp_meta
        for sample_name in exp_meta[lane_id].dropna()]
    
    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(processes=args.nprocs) as pool:
        for lane_id, sample_name in pool.imap_unordered(run_guide_stats, runs_args):
            print(f"completed {sample_name}_{lane_id}")
    
    

if __name__ == "__main__":
    main()
