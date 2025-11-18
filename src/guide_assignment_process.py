#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:10:18 2025

@author: chandrima.modak
"""

import os, sys
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
import glob
import time
import argparse
import multiprocessing as mp

# =============================================================================
# folders = [d for d in os.listdir('/Users/chandrima.modak/Gladstone Dropbox/Chandrima Modak/gw-CRISPRa/Diff053/cellranger') if d.startswith('CRISPRia_Cellanome_')]
# folders = sorted(folders)
# files = {}
# for folder in folders:
#     folder_path = os.path.join('/Users/chandrima.modak/Gladstone Dropbox/Chandrima Modak/gw-CRISPRa/Diff053/cellranger', folder, 'per_sample_outs')
#     lane = folder.split('_')[2]
#     t = [d for d in os.listdir(folder_path) if d != '.DS_Store']
#     files[lane] = t
#     
# files_pd = pd.DataFrame.from_dict(files)
# 
# files_pd.to_csv(os.path.join('/Users/chandrima.modak/Gladstone Dropbox/Chandrima Modak/gw-CRISPRa/Diff053', 'expirements_meta.csv'), ignore_index = True)
# 
# =============================================================================



def process_batch(args):
    """
    Process a batch of gRNAs - redesigned to take a single argument for better pickle compatibility
    """
    gRNA_list, adata_crispr, output_dir, n_iter, start_idx, step, batch_id = args
    
    # Add debugging information
    print(f"[Worker {batch_id}] Starting batch with {min(step, len(gRNA_list) - start_idx)} gRNAs")
    sys.stdout.flush()  # Force output to be displayed immediately
    
    batch_perturbations = pd.DataFrame()
    batch_thresholds = pd.DataFrame()
    
    end_idx = min(start_idx + step, len(gRNA_list))
    for i in range(start_idx, end_idx):
        gRNA = gRNA_list[i]
        # Removed tqdm from inside the worker function
        try:
            if i % 5 == 0:  # Print progress every few gRNAs
                print(f"[Worker {batch_id}] Processing gRNA {i-start_idx+1}/{end_idx-start_idx}: {gRNA}")
                sys.stdout.flush()
                
            perturbed_cells, threshold, loss, map_estimates = cr.fit_PGMM(
                gRNA, adata_crispr, output_dir, 2024, n_iter
            )
            if len(perturbed_cells) != 0:
                # get UMI_counts of assigned cells, handle sparse & dense
                X_slice = adata_crispr[perturbed_cells, [gRNA]].X
                if sp.sparse.issparse(X_slice):
                    UMI_counts = X_slice.toarray().ravel()
                else:
                    UMI_counts = np.asarray(X_slice).ravel()
                df = pd.DataFrame({'cell': perturbed_cells, 'gRNA': gRNA, 'UMI_counts': UMI_counts})
                batch_perturbations = pd.concat([batch_perturbations, df], ignore_index=True)
                batch_thresholds = pd.concat([batch_thresholds, pd.DataFrame({'gRNA': [gRNA], 'threshold': [threshold]})])
        except Exception as e:
            print(f"[Worker {batch_id}] Error processing gRNA {gRNA}: {str(e)}")
            sys.stdout.flush()
    
    print(f"[Worker {batch_id}] Finished batch with {batch_perturbations.shape[0]} perturbations")
    sys.stdout.flush()
    
    return batch_perturbations, batch_thresholds

def assign_sgrna_crispat(adata_crispr, output_dir, start_idx=0, end_idx=None, UMI_threshold=3, n_iter=500, n_guides_parallel=4, num_cores=5):
    """
    Assign sgRNAs to cells using the CRISPAT Poisson-Gaussian Mixture Model with parallel processing.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timer to measure performance
    start_time = time.time()
    
    gRNA_list = adata_crispr.var_names.tolist()
    
    # Set up parallel processing
    if num_cores is None:
        num_cores = min(mp.cpu_count(), n_guides_parallel)
    else:
        num_cores = min(num_cores, mp.cpu_count(), len(gRNA_list))

    # Determine chunk boundaries
    if end_idx is None:
        end_idx = len(gRNA_list)
    else:
        end_idx = min(end_idx, len(gRNA_list))
    
    # Extract the subset of guides we'll process
    chunk_gRNAs = gRNA_list[start_idx:end_idx]
    chunk_adata = adata_crispr[:, chunk_gRNAs].copy()

    # Create batches for parallel processing
    batch_size = max(1, len(chunk_gRNAs) // num_cores)
    batch_indices = list(range(0, len(chunk_gRNAs), batch_size))
    
    # Prepare arguments for multiprocessing
    process_args = [(chunk_gRNAs, chunk_adata, output_dir, n_iter, start_batch, batch_size, idx) 
                   for idx, start_batch in enumerate(batch_indices)]
    
    # Process batches in parallel with better progress reporting
    print(f'Fitting Poisson-Gaussian Mixture Model for {len(gRNA_list)} gRNAs using {num_cores} cores')
    print(f'Each core will process approximately {batch_size} gRNAs')
    
    # Use 'fork' method instead of 'spawn' for Linux/macOS to avoid module import overhead
    ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
    with ctx.Pool(processes=num_cores) as pool:
        results = []
        for i, res in enumerate(pool.imap(process_batch, process_args)):
            print(f"Completed batch {i+1}/{len(process_args)}")
            results.append(res)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"All batches completed in {elapsed_time:.2f} seconds")
    
    # Combine results
    perturbations = pd.DataFrame()
    thresholds = pd.DataFrame()
    for batch_perturbations, batch_thresholds in results:
        perturbations = pd.concat([perturbations, batch_perturbations], ignore_index=True)
        thresholds = pd.concat([thresholds, batch_thresholds], ignore_index=True)

    # Filter by UMI threshold
    perturbations = perturbations[perturbations['UMI_counts'] >= UMI_threshold]

    # Make unique cell assignment - handle empty DataFrame case
    if len(perturbations) == 0:
        print("Warning: No perturbations passed the UMI threshold filter")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    assignment_size = perturbations.groupby('cell').size() 
    # Use drop=True to avoid duplicate 'cell' column
    assignment_crispat = perturbations.groupby('cell').apply(lambda x: x.loc[x['UMI_counts'].idxmax()]).reset_index(drop=True)
    # Add cell column back
    assignment_crispat['cell'] = perturbations.groupby('cell').apply(lambda x: x.name).values
    assignment_crispat['guide_id'] = np.where(assignment_size[assignment_crispat['cell']].values > 1, 
                                             'multi_sgRNA', 
                                             assignment_crispat['gRNA'])
    assignment_crispat.set_index('cell', inplace=True)
    assert assignment_crispat.index.is_unique
    
    return assignment_crispat, perturbations, thresholds

def main():
    parser = argparse.ArgumentParser(description='Processing CRISPR assay')
    parser.add_argument('--processed_dir',type=str,required=True,help='Path to directory that processed dir')
    parser.add_argument('--main_dir',type=str,required=True,help='Path to directory that main dir')
    parser.add_argument('--expmeta',type=str,required=True,help='Path to directory that Experiment metadata')
    parser.add_argument('--nprocs', type=int, help='Number of worker processes')
    args = parser.parse_args()
    
    exp_meta = pd.read_csv(os.path.join(args.main_dir, args.expmeta))
    
    for colname, coldata in exp_meta.items():
        for value in coldata:
            directory_path = os.path.join(args.processed_dir, f'{value}_{colname}')
            if os.path.isdir(directory_path):
                crispr_a = sc.read_h5ad(glob.glob1(directory_path, '*_crispr_preprocessed.h5ad'))
                assignment_crispat, perturbations, thresholds = assign_sgrna_crispat(crispr_a, num_cores=args.nprocs,)
                
                if assignment_crispat.empty:
                    print(f"No assignments for {value_str}_{colname}, skipping write.")
                    continue
                
                assignment_crispat.to_csv(os.path.join(directory_path, f'{value}_{colname}_processed_guide.csv'))
                metadata = crispr_a.obs.copy()
                metadata =  pd.merge(metadata, assignment_crispat[['gRNA', 'guide_id', 'UMI_counts']], left_index=True, 
                                          right_index=True, how='left').copy()
                crispr_a.obs = metadata.copy()
                crispr_a.write_h5ad(os.path.join(directory_path, f'{value}_{colname}_crispr_guide.h5ad'))
                
            else:
                print(f'{value}_{colname} not found')
                continue
                
            
   

if __name__ == "__main__":
    main()
