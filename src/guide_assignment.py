import os
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
from scipy import sparse

class assign_sgrna:
    def __init__(self, path, crispr_adata):
        self.path = path
        self.adata_crispr_file = crispr_adata
        # self.gex_adata =  gex_adata
        # Store results as instance variables for access across methods
        self.perturbations = pd.DataFrame()
        self.all_thresholds = pd.DataFrame()
    
    def run_guide_assign(self):
        output_dir = self.path
        
        # Create directory structure
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("Creating subdirectories for model plots...")
        directories_to_create = ["loss_plots", "fitted_model_plots"]
        for dir_name in directories_to_create:
            dir_path = os.path.join(output_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Directory created: {dir_path}")
            
        # IMPORTANT: crispat seems to use string concatenation, so force a trailing slash
        if output_dir.endswith(os.sep):
            output_dir_for_crispat = output_dir
        else:
            output_dir_for_crispat = output_dir + os.sep
        
        # Get list of gRNAs 
        gRNA_list = self.adata_crispr_file.var_names.tolist()
        
        # Reset storage for results
        self.perturbations = pd.DataFrame()
        self.all_thresholds = pd.DataFrame()
        
        # Process each gRNA individually
        for gRNA in gRNA_list:
            try:
                # Call fit_PGMM with a single gRNA string
                perturbed_cells, threshold, loss, map_estimates = cr.fit_PGMM(
                    gRNA, self.adata_crispr_file, output_dir_for_crispat, 2025, n_iter=500
                )
                
                if len(perturbed_cells) != 0:
                    # Get UMI_counts of assigned cells
                    UMI_counts = self.adata_crispr_file[perturbed_cells, [gRNA]].X.toarray().ravel()
                    df = pd.DataFrame({"cell": perturbed_cells, "gRNA": gRNA, "UMI_counts": UMI_counts} )
                    
                    # Append to running results
                    self.perturbations = pd.concat([self.perturbations, df], ignore_index=True)
                    self.all_thresholds = pd.concat([self.all_thresholds, pd.DataFrame({"gRNA": [gRNA], "threshold": [threshold]}),],ignore_index=True)
                    
                    # Save as csv for other calculation
                    self.perturbations.to_csv(os.path.join(self.path, "guide_assigned.csv"),index=False)
                    self.all_thresholds.to_csv(os.path.join(self.path, "guide_threshold.csv"),index=False)
            except Exception as e:
                print(f"  Failed for {gRNA}: {e}")
                continue
        
        return self.perturbations, self.all_thresholds
    
    def merge_guide_assign(self):
        # Check if perturbations exist
        if self.perturbations.empty:
            raise ValueError("No perturbations found. Run 'run_guide_assign' first.")
        
        # Calculate assignment size for each cell
        assignment_size = self.perturbations.groupby("cell").size()
        
        # Get the gRNA with maximum UMI counts for each cell
        assignment_crispat = (
            self.perturbations.groupby("cell")
            .apply(lambda x: x.loc[x["UMI_counts"].idxmax()])
            .reset_index(drop=True)
        )
        
        # Mark cells with multiple sgRNAs
        assignment_crispat["guide_id"] = np.where(
            assignment_size[assignment_crispat["cell"]].values > 1,
            "multi_sgRNA",
            assignment_crispat["gRNA"],
        )
        
        assignment_crispat.set_index("cell", inplace=True)
        
        return assignment_crispat

    def binary_obsm(self):
        df_dummy = pd.get_dummies(self.perturbations[["cell", "gRNA"]], columns=["gRNA"], 
                                  dtype="int",prefix="", prefix_sep="")
    
        binary_matrix = (df_dummy.groupby("cell", sort=False).max())
    
    
        return binary_matrix

def run_guide_jobs(run):
    # Worker for a single (colname, value) job.
    processed_dir, colname, value = run

    directory_path = os.path.join(processed_dir, f"{value}_{colname}")
    if not os.path.isdir(directory_path):
        print(f"{value}_{colname} not found")
        return (colname, value)

    expected_out = os.path.join(directory_path, f"{value}_gex_guide.h5ad")
    if os.path.isfile(expected_out):
        print(f"{expected_out} exists, skipping...")
        return (colname, value)

    crispr_pattern = glob.glob(os.path.join(directory_path, "*_crispr_preprocessed.h5ad"))
    gex_pattern = glob.glob(os.path.join(directory_path, "*_gex_preprocessed.h5ad"))

    if len(crispr_pattern) == 0 or len(gex_pattern) == 0:
        print(f"Missing preprocessed h5ad in {directory_path}, skipping.")
        return (colname, value)

    # Read inputs
    crispr_a = sc.read_h5ad(crispr_pattern[0])
    gex_adata = sc.read(gex_pattern[0])   # or sc.read_h5ad if you prefer

    guide_assign = assign_sgrna(directory_path, crispr_a)
    guide_assign.run_guide_assign()
    
    binary_matrix = guide_assign.binary_obsm()

    # assign guides to cells
    assignment_crispat = guide_assign.merge_guide_assign()

    if assignment_crispat.empty:
        print(f"No assignments for {value}_{colname}, skipping write.")
        return (colname, value)

    # --- align assignment_crispat to gex_adata.obs ---
    common_cells = assignment_crispat.index.intersection(gex_adata.obs.index)
    assignment_crispat = assignment_crispat.loc[common_cells].copy()

    if assignment_crispat.index.has_duplicates:
        assignment_crispat = assignment_crispat[
            ~assignment_crispat.index.duplicated(keep="first")
        ]

    # --- target_gene parsing ---
    is_multi = assignment_crispat["guide_id"].astype(str).eq("multi_sgRNA")

    assignment_crispat.loc[~is_multi, "target_gene"] = (
        assignment_crispat.loc[~is_multi, "guide_id"]
        .astype(str)
        .str.replace("-", "_")
        .str.split("_")
        .str[0]
    )
    assignment_crispat.loc[is_multi, "target_gene"] = "multi_sgRNA"

    # --- join onto obs ---
    gex_adata.obs = gex_adata.obs.join(
        assignment_crispat[["gRNA", "guide_id", "target_gene", "UMI_counts"]],
        how="left",
    )
    # Align to adata.obs_names
    binary_matrix = binary_matrix.reindex(index=gex_adata.obs_names, fill_value=0)

    # Store sparse
    gex_adata.obsm['guide_matrix'] = sparse.csr_matrix(binary_matrix.to_numpy(dtype="int8"))
    gex_adata.uns["guide_matrix_cols"] = binary_matrix.columns.to_list()
    
    # Save outputs
    assignment_crispat.to_csv(
        os.path.join(directory_path, f"{value}_processed_guide.csv")
    )
    gex_adata.write_h5ad(
        os.path.join(directory_path, f"{value}_gex_guide.h5ad")
    )

    return (colname, value)
   
     
     
def main():
    parser = argparse.ArgumentParser(description="Processing CRISPR assay")
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
    runs = [
        (args.processed_dir, colname, value)
        for colname, coldata in exp_meta.items()
        for value in coldata
    ]

    ctx = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    with ctx.Pool(processes=args.nprocs) as pool:
        for colname, value in pool.imap_unordered(run_guide_jobs, runs):
            print(f"completed {value}_{colname}")


if __name__ == "__main__":
    main()
