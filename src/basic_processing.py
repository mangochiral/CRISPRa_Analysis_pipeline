#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:31:31 2025

@author: chandrima.modak
"""
import re
import sys
import os
import pandas as pd
import argparse
from  preprocess import *
import multiprocessing as mp

def _run_one(run):
    """Worker for a single (lane, sample_name) job."""
    datadir, output_dir, exp, lane, sample_name = run
    try:
        path_sample_dir = os.path.join(
            datadir, f'CRISPRia_Cellanome_{lane}', 'per_sample_outs', sample_name
        )
        
        xdata = os.path.join(path_sample_dir, 'count', 'sample_filtered_feature_bc_matrix.h5')

        if not os.path.exists(xdata):
            return (f"Missing input: {xdata}")

        gex_a, crispr_a, pre_filter_counts, post_filter_counts = process_cellranger_h5(
            xdata, exp, sample_name, lane
        )
        # perturb_info  = crispr_a.var.groupby('perturbation_type')['n_cells'].sum()
        outdir = os.path.join(output_dir, f'{sample_name}_{lane}')
        os.makedirs(outdir, exist_ok=True)

        gex_out    = os.path.join(outdir, f"{sample_name}_gex_preprocessed.h5ad")
        crispr_out = os.path.join(outdir, f"{sample_name}_crispr_preprocessed.h5ad")

        gex_a.write_h5ad(gex_out)
        crispr_a.write_h5ad(crispr_out)

        with open(os.path.join(outdir, 'stats_report.txt'), 'w', encoding='utf-8') as file:
            file.write(f"Pre filter cells count: {pre_filter_counts}\n")
            file.write(f"Post filter cells count: {post_filter_counts}\n")
            # file.write(f'Perturbed cells counts info {perturb_info}')
        
        return lane, sample_name

    except Exception as e:
        return f"Error: {e}"
    

def main():
    parser = argparse.ArgumentParser(description='Processing CRISPR experiment')
    parser.add_argument('--datadir',type=str,required=True,help='Path to directory that contains CRISPRia_Cellanome_<lane> folders')
    parser.add_argument('--exp',type=str,default='crispr',help="Experiment type; for Perturb-seq keep as 'crispr'")
    parser.add_argument('--output_dir',type=str,required=True,help='Path to directory that processed data to be saved')
    parser.add_argument('--nprocs', type=int, help='Number of worker processes')
    args = parser.parse_args()
    
    # All lane folders like CRISPRia_Cellanome_lane1, lane2, ...
    folder_path = [d for d in os.listdir(args.datadir)if d.startswith('CRISPRia_Cellanome_')
                   and os.path.isdir(os.path.join(args.datadir, d))]
    if not folder_path:
        raise RuntimeError(f"No CRISPRia_Cellanome_* folders found in {args.datadir}")
    
    lanes = sorted(d.split('_')[2] for d in folder_path)
    
    # Sample names from the first lane’s per_sample_outs
    per_sample_dir = os.path.join(args.datadir, folder_path[0], 'per_sample_outs')
    sample_names = [
        s for s in os.listdir(per_sample_dir)
        if s != '.DS_Store' and os.path.isdir(os.path.join(per_sample_dir, s))
    ]
    if not sample_names:
        raise RuntimeError(f"No sample folders found in {per_sample_dir}")
    
    # Make sure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    num_cores = min(args.nprocs, mp.cpu_count())
    
    # Running lanes as one job
    runs = [(args.datadir, args.output_dir, args.exp, lane, sample) for lane in lanes for sample in sample_names]
    ctx =  mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
    with ctx.Pool(processes= num_cores)as pool:
        for lane, sample in pool.imap(_run_one, runs):
            print(f'completed {sample},{lane}')

if __name__ == "__main__":
    main()

