#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:35:45 2025

@author: chandrima.modak
"""


import os,sys
import glob
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import pingouin as pg
import crispat as cr
import matplotlib.pyplot as plt
import anndata as ad
import scipy as sp
from plotnine import *
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import re
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm.auto import tqdm


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



def get_background_vs_signal_guide_counts(adata_cr, csv_file, return_summary=False):
    '''
    Analyze sgRNA counts to distinguish between background and signal for all samples in a directory.
    '''
    sample = '_'.join(glob.glob(os.path.join(csv_file, "*_processed_guide.csv"))[0].split('/')[-1].split('_')[:3])
    
    all_guides_combined = pd.DataFrame()
    
    # 1) Read UMI counts from AnnData
    
    all_umi_counts = get_long_sgrna_umi_counts(adata_cr)
    
    # 2) Read assigned guides
    clean_assignment = glob.glob(os.path.join(csv_file, "*_processed_guide.csv"))[0]
    assigned_guides = pd.read_csv(clean_assignment)
    assigned_guides = pd.read_csv(clean_assignment, index_col=0)
    assigned_guides = assigned_guides.reset_index().rename(columns={"index": "cell"})

    # 2a) Get multi-sgRNA cells from guide_id label
    all_multi_sgrna_cells = assigned_guides.loc[assigned_guides['guide_id'] == 'multi_sgRNA', 'cell'].unique().tolist()

    # 3) Merge: all UMI counts (left) with assignments (right)
    all_guides = pd.merge(all_umi_counts, assigned_guides, on= ['cell', 'gRNA'], how='left', suffixes=['_all', '_assigned'])
    all_guides['signal_vs_bg'] = np.where(all_guides['UMI_counts_assigned'].isna(), 'background', 'signal')
    all_guides['sample'] = sample
    all_guides_combined = pd.concat([all_guides_combined, all_guides], ignore_index=True)
    if return_summary:
        # Exclude multi-sgRNA cells from summary
        all_guides_combined_sum = all_guides_combined[~all_guides_combined['cell'].isin(all_multi_sgrna_cells)]
        all_guides_combined_sum = all_guides_combined_sum[['cell', 'gRNA', 'UMI_counts_all', 'signal_vs_bg']].copy()
        summary_df = all_guides_combined_sum.groupby(['signal_vs_bg', 'gRNA'])['UMI_counts_all'].median().reset_index()
        summary_df = summary_df.pivot_table(columns='signal_vs_bg', index=['gRNA'], values='UMI_counts_all')
        summary_df = summary_df.fillna(0)
        summary_df.to_csv(os.path.join(csv_file, f'{sample}_guide_sigvsbg_stats.csv'))
        
        return summary_df
    else:
        return all_guides_combined

# =============================================================================
# Checking guide efficiency
# =============================================================================

def guide_type_efficiency(outdir, gex_adata):
    """
    Analyze the efficiency of guides and ranking them
    
    Parameters
    ----------
    e
    
    Returns
    --------
    stats report pval fdr effective size
    
    """
    sample_name = os.path.basename(outdir)
    
    # If var names look like ENSG..., swap to gene_name
    if gex_adata.var.index.str.startswith("ENSG").any():
        if "gene_name" not in gex_adata.var.columns:
            raise ValueError("gene_name column is missing in gex_adata.var")
        gex_adata.var.index = gex_adata.var["gene_name"]
        gex_adata.var_names = gex_adata.var.index
    
    adata_obj = gex_adata[gex_adata.obs.guide_id.notna(), :]
    adata_obj_single_guide = adata_obj[adata_obj.obs.guide_id != 'multi_sgRNA', :]
    target_genes = adata_obj_single_guide.var_names.to_list()
    target_cells = adata_obj_single_guide.shape[0]
    non_target_cells = gex_adata.shape[0] - target_cells
    
    with open(os.path.join(outdir, 'stats_report.txt'), 'a', encoding='utf-8') as file:
        file.write(f"Targeted perturbed: {target_cells}\n")
        file.write(f"Non Targeted perturbed cells: {non_target_cells}\n")
    
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
    results_df.to_csv(os.path.join(outdir, f'{sample_name}_guide_efficiency.csv'))
    return results_df

def run_qc(run_args):
    processed_dir, colname, value, adhoc = run_args
    
    directory_path = os.path.join(processed_dir, f"{value}_{colname}")
    if not os.path.isdir(directory_path):
        print(f"{value}_{colname} not found")
        return (colname, value)
    
    gex_pattern = glob.glob(os.path.join(directory_path, "*_gex_guide.h5ad"))
    crispr_pattern = glob.glob(os.path.join(directory_path, "*_crispr_preprocessed.h5ad"))
    
    # Read inputs
    crispr_a = sc.read_h5ad(crispr_pattern[0])
    gex_adata = sc.read(gex_pattern[0])   # or sc.read_h5ad 
    
    if adhoc == 'yes':
        get_background_vs_signal_guide_counts(crispr_a, directory_path, True)
    
    guide_type_efficiency(directory_path, gex_adata)
    
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
	"--lane",
        type=str,
        default=None,
        help="Lane column name to process, e.g. lane1. "
             "If not given, process all lanes.",
    )
    parser.add_argument(
	"--adhoc",
        type=str,
        default=None,
        help="choice between yes to generate signal to background report",
    )
    args = parser.parse_args()
    exp_meta = pd.read_csv(os.path.join(args.processed_dir, args.expmeta))

    # Drop any Unnamed index-like columns
    exp_meta = exp_meta.loc[:, ~exp_meta.columns.str.startswith("Unnamed")]
    # Decide which columns (lanes) to use
    if args.lane is not None:
        if args.lane not in exp_meta.columns:
            raise ValueError(f"Lane {args.lane} not found in columns {list(exp_meta.columns)}")
        lane_cols = [args.lane]
    else:
        lane_cols = list(exp_meta.columns)
    # Build list of jobs: one per (colname, value)
    runs_args = [
        (args.processed_dir, colname, value, args.adhoc)for colname in lane_cols
        for value in exp_meta[colname].dropna()]

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(processes=args.nprocs) as pool:
        for colname, value in pool.imap_unordered(run_qc, runs_args):
            print(f"completed {value}_{colname}")


if __name__ == "__main__":
    main()

                                         
    
    