#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:12:01 2026

@author: chandrima.modak

PRODUCTION VERSION - Bug fixes + robust resume validation
"""
import os
import sys
import glob
import scanpy as sc
import numpy as np
import pandas as pd
import crispat
import anndata as ad
from pathlib import Path
import argparse
from scipy import sparse
import multiprocessing as mp
import tempfile
import shutil


# =============================================================================
# Batch guide Assignment Helper
# =============================================================================

def _batch_guide_assign(args):
    """
    Parameters
    ----------
    args : tuple
        (chunk_adata, chunk_gRNA, output_dir, n_iter, seed)

    Returns
    -------
    batch_perturb_df : pd.DataFrame
        Cells with their assigned guides and UMI counts.
    batch_threshold_df : pd.DataFrame
        Threshold per guide.
    """
    chunk_adata, chunk_gRNA, output_dir, n_iter, seed = args

    # Ensure trailing separator for crispat path handling
    if not output_dir.endswith(os.sep):
        output_dir_for_crispat = output_dir + os.sep
    else:
        output_dir_for_crispat = output_dir

    batch_perturb_list = []
    batch_threshold_list = []
    errors = []

    for gRNA in chunk_gRNA:
        try:
            perturbed_cells, threshold, loss, map_estimates = crispat.fit_PGMM(
                gRNA, chunk_adata, output_dir_for_crispat, seed, n_iter
            )

            # Always record the threshold for this guide
            batch_threshold_list.append({"gRNA": gRNA, "Threshold": threshold})

            if len(perturbed_cells) != 0:
                UMI_counts = chunk_adata[perturbed_cells, [gRNA]].X.toarray().ravel()
                batch_perturb_list.append(
                    pd.DataFrame(
                        {
                            "cell": perturbed_cells,
                            "gRNA": gRNA,
                            "UMI_counts": UMI_counts,
                        }
                    )
                )

        except Exception as e:
            error_msg = f"Failed for {gRNA}: {e}"
            print(error_msg)
            errors.append(error_msg)
            continue

    if errors:
        print(f"WARNING: Encountered {len(errors)} errors in this batch")

    if batch_perturb_list:
        batch_perturb_df = pd.concat(batch_perturb_list, ignore_index=True)
    else:
        batch_perturb_df = pd.DataFrame(columns=["cell", "gRNA", "UMI_counts"])

    if batch_threshold_list:
        batch_threshold_df = pd.DataFrame(batch_threshold_list)
    else:
        batch_threshold_df = pd.DataFrame(columns=["gRNA", "Threshold"])

    return batch_perturb_df, batch_threshold_df


# =============================================================================
# File validation helpers
# =============================================================================

def _validate_batch_files(perturb_file, thresh_file):
    """
    Validate that batch files exist and contain valid data.
    
    Parameters
    ----------
    perturb_file : str
        Path to perturbation CSV file
    thresh_file : str
        Path to threshold CSV file
        
    Returns
    -------
    bool
        True if both files exist and are valid, False otherwise
    """
    # Check both files exist
    if not (os.path.exists(perturb_file) and os.path.exists(thresh_file)):
        return False
    
    try:
        # Attempt to read perturbation file
        perturb_df = pd.read_csv(perturb_file)
        
        # Check expected columns exist
        expected_perturb_cols = ["cell", "gRNA", "UMI_counts"]
        if not all(col in perturb_df.columns for col in expected_perturb_cols):
            print(f"WARNING: Invalid columns in {perturb_file}")
            return False
        
        # Check data types are reasonable (not all NaN)
        if perturb_df[expected_perturb_cols].isna().all().any():
            print(f"WARNING: All NaN columns in {perturb_file}")
            return False
            
        # Attempt to read threshold file
        thresh_df = pd.read_csv(thresh_file)
        
        # Check expected columns exist
        expected_thresh_cols = ["gRNA", "Threshold"]
        if not all(col in thresh_df.columns for col in expected_thresh_cols):
            print(f"WARNING: Invalid columns in {thresh_file}")
            return False
        
        # Threshold file should not be empty (even if perturb is)
        if len(thresh_df) == 0:
            print(f"WARNING: Empty threshold file {thresh_file}")
            return False
        
        # Files are valid
        return True
        
    except pd.errors.EmptyDataError:
        print(f"WARNING: Empty CSV file detected")
        return False
    except Exception as e:
        print(f"WARNING: Error validating batch files: {e}")
        return False


def _atomic_csv_write(df, filepath, output_dir):
    """
    Write CSV atomically using temp file + rename.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write
    filepath : str
        Final destination path
    output_dir : str
        Directory for temp file (same filesystem as filepath)
    """
    # Create temp file in same directory to ensure same filesystem
    with tempfile.NamedTemporaryFile(
        mode='w', 
        delete=False, 
        dir=output_dir, 
        suffix='.tmp',
        prefix='.batch_'
    ) as tmp:
        tmp_path = tmp.name
    
    try:
        # Write to temp file
        df.to_csv(tmp_path, index=False)
        
        # Atomic rename (on most filesystems)
        shutil.move(tmp_path, filepath)
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        raise e


class assign_sgrna:
    def __init__(self, path, crispr_adata):
        self.path = path
        self.adata_crispr_file = crispr_adata
        self.perturbations = pd.DataFrame()
        self.all_thresholds = pd.DataFrame()

    # =============================================================================
    # Guide level parallel processing
    # =============================================================================

    def run_guide_assign(self, num_cores=None, n_iter=500, seed=2025, end_idx=None, 
                        batch_size=1000, resume=True):
        """
        Parameters
        ----------
        num_cores : int, optional
            Number of cores for parallel processing.
        n_iter : int
            Number of iterations.
        seed : int
            Random seed.
        end_idx : int, optional
            End index for gRNA list.
        batch_size : int
            Size of batches for processing.
        resume : bool
            Whether to resume from existing batch files.

        Returns
        -------
        perturbations : pd.DataFrame
            All cells with their assigned guides and UMI counts.
        all_thresholds : pd.DataFrame
            Threshold per guide.
        """
        slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        max_cpus = min(mp.cpu_count(), slurm_cpus)
        
        if num_cores is None:
            num_cores = max_cpus
        else:
            num_cores = min(int(num_cores), max_cpus)

        output_dir = self.path

        # --- Create plot subdirectories ONCE, before spawning workers ---
        try:
            for dir_name in ("loss_plots", "fitted_model_plots"):
                os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
        except OSError as e:
            print(f"WARNING: Directory creation issue: {e}")

        gRNA_list = self.adata_crispr_file.var_names.tolist()
        
        # Where per-batch outputs go
        batch_out = os.path.join(output_dir, "batch_outputs")
        if resume:
            os.makedirs(batch_out, exist_ok=True)

        # Determine chunk boundaries
        if end_idx is None:
            end_idx = len(gRNA_list)
        else:
            end_idx = min(end_idx, len(gRNA_list))

        # Build arguments for parallel processing
        runs_args = []
        batch_ids = []
        skipped_count = 0
        reprocessed_count = 0
        
        for i in range(0, end_idx, batch_size):
            batch_id = i // batch_size
            
            if resume:
                # FIX: Use zero-padded batch IDs for correct sorting
                perturb_file = os.path.join(batch_out, f"perturb_batch_{batch_id:04d}.csv")
                thresh_file = os.path.join(batch_out, f"threshold_batch_{batch_id:04d}.csv")

                # ROBUST: Validate files before skipping
                if _validate_batch_files(perturb_file, thresh_file):
                    skipped_count += 1
                    continue
                
                # If files exist but are invalid, delete them
                for filepath in [perturb_file, thresh_file]:
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                            reprocessed_count += 1
                        except OSError as e:
                            print(f"WARNING: Could not remove {filepath}: {e}")
                
            chunk_gRNA = gRNA_list[i : i + batch_size]
            chunk_adata = self.adata_crispr_file[:, chunk_gRNA].copy()
            runs_args.append((chunk_adata, chunk_gRNA, output_dir, n_iter, seed))
            batch_ids.append(batch_id)
        
        if resume:
            print(f"[resume] Skipped {skipped_count} completed batches")
            if reprocessed_count > 0:
                print(f"[resume] Reprocessing {reprocessed_count // 2} batches with invalid files")
            
        # Nothing to do (everything already complete)
        if resume and len(runs_args) == 0:
            print("[resume] No remaining batches to run; merging existing outputs.")
            return self._merge_from_disk(batch_out)

        # FIX: Use spawn for both Windows and macOS
        ctx = mp.get_context("spawn" if sys.platform in ["win32", "darwin"] else "fork")
        
        print(f"Processing {len(runs_args)} batches with {num_cores} cores")
        
        # OWhen resume=True, don't collect in memory
        if resume:
            with ctx.Pool(processes=num_cores) as pool:
                for batch_id, (batch_perturb_df, batch_threshold_df) in zip(
                    batch_ids, pool.imap(_batch_guide_assign, runs_args)
                ):
                    # FIX: Use zero-padded batch IDs + atomic writes
                    perturb_file = os.path.join(batch_out, f"perturb_batch_{batch_id:04d}.csv")
                    thresh_file = os.path.join(batch_out, f"threshold_batch_{batch_id:04d}.csv")
                    
                    try:
                        _atomic_csv_write(batch_perturb_df, perturb_file, batch_out)
                        _atomic_csv_write(batch_threshold_df, thresh_file, batch_out)
                        print(f"Completed batch {batch_id}")
                    except Exception as e:
                        print(f"ERROR: Failed to write batch {batch_id}: {e}")
                        # Clean up any partial files
                        for filepath in [perturb_file, thresh_file]:
                            if os.path.exists(filepath):
                                try:
                                    os.remove(filepath)
                                except:
                                    pass
                        raise
            
            return self._merge_from_disk(batch_out)
        
        else:
            # Non-resume mode: collect in memory for backward compatibility
            all_perturb = []
            all_threshold = []
            
            with ctx.Pool(processes=num_cores) as pool:
                for batch_id, (batch_perturb_df, batch_threshold_df) in zip(
                    batch_ids, pool.imap(_batch_guide_assign, runs_args)
                ):
                    print(f"Completed batch {batch_id}")
                    all_perturb.append(batch_perturb_df)
                    all_threshold.append(batch_threshold_df)

            if all_perturb:
                self.perturbations = pd.concat(all_perturb, ignore_index=True)
                self.all_thresholds = pd.concat(all_threshold, ignore_index=True)
            else:
                self.perturbations = pd.DataFrame(columns=["cell", "gRNA", "UMI_counts"])
                self.all_thresholds = pd.DataFrame(columns=["gRNA", "Threshold"])

            return self.perturbations, self.all_thresholds
    
    def _merge_from_disk(self, batch_out):
        """Merge all batch files from disk."""
        perturb_files = sorted(glob.glob(os.path.join(batch_out, "perturb_batch_*.csv")))
        thresh_files = sorted(glob.glob(os.path.join(batch_out, "threshold_batch_*.csv")))

        print(f"Merging {len(perturb_files)} perturbation files and {len(thresh_files)} threshold files")

        # Validate before merging
        valid_perturb = []
        valid_thresh = []
        
        for pf, tf in zip(perturb_files, thresh_files):
            if _validate_batch_files(pf, tf):
                valid_perturb.append(pf)
                valid_thresh.append(tf)
            else:
                print(f"WARNING: Skipping invalid batch files: {pf}, {tf}")
        
        if valid_perturb:
            try:
                self.perturbations = pd.concat(
                    [pd.read_csv(f) for f in valid_perturb], 
                    ignore_index=True
                )
            except Exception as e:
                print(f"ERROR: Failed to merge perturbation files: {e}")
                raise
        else:
            self.perturbations = pd.DataFrame(columns=["cell", "gRNA", "UMI_counts"])

        if valid_thresh:
            try:
                self.all_thresholds = pd.concat(
                    [pd.read_csv(f) for f in valid_thresh], 
                    ignore_index=True
                )
            except Exception as e:
                print(f"ERROR: Failed to merge threshold files: {e}")
                raise
        else:
            self.all_thresholds = pd.DataFrame(columns=["gRNA", "Threshold"])

        return self.perturbations, self.all_thresholds

    # =============================================================================
    # Extracting cell level guide assignment processing
    # =============================================================================

    def merge_guide_assign(self):
        """
        Returns
        -------
        assignment_crispat : pd.DataFrame
            Cell-level guide labels; multi-guide cells keep the top-UMI guide.
        """
        if self.perturbations.empty:
            raise ValueError("No perturbations found. Run 'run_guide_assign' first.")

        # Number of guides assigned to each cell
        assignment_size = self.perturbations.groupby("cell").size()

        # Keep the gRNA with the highest UMI count per cell
        idx_max = self.perturbations.groupby("cell")["UMI_counts"].idxmax()
        assignment_crispat = self.perturbations.loc[idx_max].copy()

        # FIX: Use reindex for safer access
        # Mark cells with multiple sgRNAs
        assignment_crispat["guide_id"] = np.where(
            assignment_size.reindex(assignment_crispat["cell"].values, fill_value=1).values > 1,
            "multi_sgRNA",
            assignment_crispat["gRNA"],
        )

        assignment_crispat.set_index("cell", inplace=True)

        return assignment_crispat

    def make_binary_obsm(self):
        """
        Returns
        -------
        binary_matrix : pd.DataFrame
            Binary cell × guide matrix.
        """
        df_dummy = pd.get_dummies(
            self.perturbations[["cell", "gRNA"]],
            columns=["gRNA"],
            prefix="",
            prefix_sep="",
            dtype="int",
        )
        binary_matrix = df_dummy.groupby("cell", sort=False).max()
        return binary_matrix


# =============================================================================
# Per-sample job runner
# =============================================================================

def run_guide_jobs(run):
    """Worker for a single (processed_dir, colname, value) job."""
    processed_dir, colname, value = run

    directory_path = os.path.join(processed_dir, f"{value}_{colname}")
    if not os.path.isdir(directory_path):
        print(f"{value}_{colname} not found")
        return (colname, value, "not_found")

    expected_out = os.path.join(directory_path, f"{value}_gex_guide.h5ad")
    if os.path.isfile(expected_out):
        print(f"{expected_out} exists, skipping...")
        return (colname, value, "skipped")

    crispr_pattern = glob.glob(
        os.path.join(directory_path, "*_crispr_preprocessed.h5ad")
    )
    gex_pattern = glob.glob(
        os.path.join(directory_path, "*_gex_preprocessed.h5ad")
    )

    if len(crispr_pattern) == 0 or len(gex_pattern) == 0:
        print(f"Missing preprocessed h5ad in {directory_path}, skipping.")
        return (colname, value, "missing_input")

    # Read inputs
    crispr_a = sc.read_h5ad(crispr_pattern[0])
    gex_adata = sc.read_h5ad(gex_pattern[0])

    guide_assign = assign_sgrna(directory_path, crispr_a)
    guide_assign.run_guide_assign(num_cores=None, resume=True)

    binary_matrix = guide_assign.make_binary_obsm()
    assignment_crispat = guide_assign.merge_guide_assign()

    if assignment_crispat.empty:
        print(f"No assignments for {value}_{colname}, skipping write.")
        return (colname, value, "no_assignments")

    # --- align assignment_crispat to gex_adata.obs ---
    common_cells = assignment_crispat.index.intersection(gex_adata.obs.index)
    assignment_crispat = assignment_crispat.loc[common_cells].copy()

    if assignment_crispat.index.has_duplicates:
        print(f"WARNING: Duplicate indices found for {value}_{colname}, keeping first occurrence")
        assignment_crispat = assignment_crispat[
            ~assignment_crispat.index.duplicated(keep="first")
        ]

    # --- target_gene parsing with error handling ---
    is_multi = assignment_crispat["guide_id"].astype(str).eq("multi_sgRNA")

    try:
        assignment_crispat.loc[~is_multi, "target_gene"] = (
            assignment_crispat.loc[~is_multi, "guide_id"]
            .astype(str)
            .str.replace("-", "_")
            .str.split("_")
            .str[0]
        )
    except Exception as e:
        print(f"ERROR: Error parsing target_gene for {value}_{colname}: {e}")
        # Fallback: use guide_id as target_gene
        assignment_crispat.loc[~is_multi, "target_gene"] = (
            assignment_crispat.loc[~is_multi, "guide_id"].astype(str)
        )
    
    assignment_crispat.loc[is_multi, "target_gene"] = "multi_sgRNA"

    # --- join onto obs ---
    gex_adata.obs = gex_adata.obs.join(
        assignment_crispat[["gRNA", "guide_id", "target_gene", "UMI_counts"]],
        how="left",
    )

    # Align binary matrix to adata.obs_names and store as sparse
    binary_matrix = binary_matrix.reindex(index=gex_adata.obs_names, fill_value=0)
    gex_adata.obsm["guide_matrix"] = sparse.csr_matrix(
        binary_matrix.to_numpy(dtype="int8")
    )
    gex_adata.uns["guide_matrix_cols"] = binary_matrix.columns.to_list()

    # Save outputs
    assignment_crispat.to_csv(
        os.path.join(directory_path, f"{value}_processed_guide.csv")
    )
    gex_adata.write_h5ad(
        os.path.join(directory_path, f"{value}_gex_guide.h5ad")
    )

    print(f"Successfully processed {value}_{colname}")
    return (colname, value, "done")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Processing CRISPR assay")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="Path to processed directory",
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
        help="Array task ID — processes a single run from the list",
    )
    args = parser.parse_args()

    exp_meta = pd.read_csv(os.path.join(args.processed_dir, args.expmeta))

    # Drop any unnamed index-like columns
    exp_meta = exp_meta.loc[:, ~exp_meta.columns.str.startswith("Unnamed")]

    # Build list of jobs: one per (colname, value)
    runs = [
        (args.processed_dir, colname, value)
        for colname, coldata in exp_meta.items()
        for value in coldata
    ]

    if args.task_id is not None:
        if args.task_id >= len(runs):
            print(
                f"ERROR: Task ID {args.task_id} exceeds number of runs ({len(runs)}). Exiting."
            )
            return
        print(f"Task {args.task_id}: processing single run (total runs: {len(runs)})")
        my_runs = [runs[args.task_id]]
    else:
        print(f"Processing all {len(runs)} runs with {args.nprocs} workers")
        my_runs = runs

    # FIX: Use spawn for both Windows and macOS
    ctx = mp.get_context("spawn" if sys.platform in ["win32", "darwin"] else "fork")
    
    with ctx.Pool(processes=args.nprocs) as pool:
        for colname, value, status in pool.imap_unordered(run_guide_jobs, my_runs):
            print(f"{status}: {value}_{colname}")


if __name__ == "__main__":
    main()