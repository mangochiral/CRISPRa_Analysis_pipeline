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
import multiprocessing as mp
import preprocess_adata as ppr

def run_preprocessing_job(cellranger_dir: str, 
                          output_dir: str, 
                          exp:str, 
                          lane:str, 
                          sample_name:str, 
                          prefix:str| None, 
                          mt_pct: float, 
                          filter_cells:bool)->dict:
    
    path = os.path.join(cellranger_dir, 'count', 'sample_filtered_feature_bc_matrix.h5')
    try:
        if not os.path.exists(path):
            return (f"Missing input: {path}")
        
        gex_a, crispr_a, pre_count, post_count = ppr.process_cellranger_h5(
            path, exp, sample_name, lane, prefix, mt_pct, filter_cells
        )
        # perturb_info  = crispr_a.var.groupby('perturbation_type')['n_cells'].sum()
        outdir = os.path.join(output_dir, f'{sample_name}_{lane}')
        os.makedirs(outdir, exist_ok=True)
    
        gex_a.write_h5ad(os.path.join(outdir, f"{sample_name}_gex_preprocessed.h5ad"))
        if crispr_a is not None:
            crispr_a.write_h5ad(os.path.join(outdir, f"{sample_name}_crispr_preprocessed.h5ad"))
    
    
        with open(os.path.join(outdir, 'stats_report.txt'), 'w', encoding='utf-8') as file:
            file.write(f"Pre filter cells count: {pre_count}\n")
            file.write(f"Post filter cells count: {post_count}\n")
            # file.write(f'Perturbed cells counts info {perturb_info}')
        
        return lane, sample_name

    except Exception as e:
        return f"Error: {e}"
    

def _unpack_and_run(args_tuple):
    return run_preprocessing_job(*args_tuple)
    
def main():
    parser = argparse.ArgumentParser(description='Processing CRISPR experiment')
    parser.add_argument('--cellranger_dir',type=str,required=True,help='Path to directory that contains CRISPR h5 data')
    parser.add_argument('--experiment_info', type=str,required=True,help="experiment metainfo csv file")
    parser.add_argument('--mt_pct', type=float,default= 20, required=True,help="Threshold for mitochondrial percent")
    parser.add_argument('--prefix', type=str, default= None, required=True,help="Name of the prefix on the guides")
    parser.add_argument('--exp',type=str,default='crispr',help="Experiment type; for Perturb-seq keep as 'crispr'")
    parser.add_argument('--filter_cells', action='store_true', default=True,help='Apply cell-level QC filtering (default: True)')
    parser.add_argument('--output_dir',type=str,required=True,help='Path to directory that processed data to be saved')
    parser.add_argument('--nprocs', type=int, help='Number of worker processes')
    args = parser.parse_args()
    
    info_path = os.path.join(args.cellranger_dir, args.experiment_info)
    experiment_info = pd.read_csv(info_path)
    experiment_info = experiment_info.loc[:, ~experiment_info.columns.str.startswith("Unnamed")]
    
    jobs = []
    for lane in experiment_info.columns:
        for sample_name in experiment_info[lane].dropna().unique():
            sample_dir = os.path.join(
                args.cellranger_dir,
                f'CRISPRia_Cellanome_{lane}',
                'per_sample_outs',
                sample_name,
            )
            jobs.append((sample_dir,args.output_dir, args.exp, lane, sample_name, args.prefix, args.mt_pct,args.filter_cells ))
    
    # Make sure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    num_cores = min(args.nprocs, mp.cpu_count())
    
    ctx =  mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
    with ctx.Pool(processes= num_cores)as pool:
        for result in pool.imap_unordered(_unpack_and_run, jobs):
            if "error" in result:
                print(f"FAILED — {result['error']}")
                print(f"Completed {result['sample_name']}, {result['lane']}")
 

if __name__ == "__main__":
    main()
