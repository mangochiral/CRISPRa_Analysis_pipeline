#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:12:01 2026

@author: chandrima.modak
"""
import os
import sys
import glob
import scanpy as sc
import numpy as np
import pandas as pd
import crispat
import anndata
import argparse
from scipy import sparse
import multiprocessing as mp
from typing import List


# =============================================================================
# Non-daemonic Pool — allows child processes to spawn grandchildren
# =============================================================================

class _NoDaemonProcess(mp.Process):
    """A Process subclass that is never daemonic, so it can spawn children."""
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass  # ignore attempts to set daemon=True


class _NoDaemonContext(type(mp.get_context())):
    """A multiprocessing context that uses non-daemon processes."""
    Process = _NoDaemonProcess


class NoDaemonPool(mp.pool.Pool):
    """A Pool that uses non-daemon workers, allowing nested Pool inside workers."""
    def __init__(self, *args, **kwargs):
        kwargs['context'] = _NoDaemonContext()
        super().__init__(*args, **kwargs)


class assign_sgrna:
    def __init__(self, path, crispr_adata):
        self.path = path
        self.adata_crispr_file = crispr_adata
        self.perturbations = pd.DataFrame()
        self.all_thresholds = pd.DataFrame()

    # =============================================================================
    # Batch guide Assignment Helper
    # =============================================================================
    @staticmethod
    def _batch_guide_assign(
            chunk_adata: anndata.AnnData,
            chunk_gRNA: List[str],
            output_dir: str,
            n_iter: int,
            seed=2025
            ) -> dict:
        """
        Process a batch of gRNAs through crispat.fit_PGMM.

        Returns
        -------
        batch_perturb_df : pd.DataFrame
            Cells with their assigned guides and UMI counts.
        batch_threshold_df : pd.DataFrame
            Threshold per guide.
        """

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
    # Guide level parallel processing
    # =============================================================================

    def run_guide_assign(self, num_cores=None, n_iter=500, seed=2025, end_idx=None,
                         batch_size=1000):
        """
        Parameters
        ----------
        num_cores : int, optional
            Number of cores for the inner guide-batch pool.
        n_iter : int
            Number of iterations.
        seed : int
            Random seed.
        end_idx : int, optional
            End index for gRNA list.
        batch_size : int
            Size of batches for processing.

        Returns
        -------
        perturbations : pd.DataFrame
            All cells with their assigned guides and UMI counts.
        all_thresholds : pd.DataFrame
            Threshold per guide.
        """
        if num_cores is None:
            slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            num_cores = min(mp.cpu_count(), slurm_cpus)
        else:
            num_cores = max(1, int(num_cores))

        output_dir = self.path

        # --- Create plot subdirectories ONCE, before spawning workers ---
        try:
            for dir_name in ("loss_plots", "fitted_model_plots"):
                os.makedirs(os.path.join(output_dir, dir_name), exist_ok=True)
        except OSError as e:
            print(f"WARNING: Directory creation issue: {e}")

        gRNA_list = self.adata_crispr_file.var_names.tolist()

        # Determine chunk boundaries
        if end_idx is None:
            end_idx = len(gRNA_list)
        else:
            end_idx = min(end_idx, len(gRNA_list))

        # Build arguments for parallel processing
        runs_args = []
        batch_ids = []

        for i in range(0, end_idx, batch_size):
            batch_id = i // batch_size
            chunk_gRNA = gRNA_list[i: i + batch_size]
            chunk_adata = self.adata_crispr_file[:, chunk_gRNA].copy()
            runs_args.append((chunk_adata, chunk_gRNA, output_dir, n_iter, seed))
            batch_ids.append(batch_id)

        all_perturb = []
        all_threshold = []

        print(f"  Guide-level pool: {num_cores} cores, {len(batch_ids)} batches")

        # Inner pool for guide batches (grandchild processes)
        ctx = mp.get_context("spawn" if sys.platform in ["win32", "darwin"] else "fork")
        with ctx.Pool(processes=num_cores) as pool:
            for batch_id, (batch_perturb_df, batch_threshold_df) in zip(
                batch_ids, pool.starmap(assign_sgrna._batch_guide_assign, runs_args)):
                print(f"  Completed batch {batch_id}")
                all_perturb.append(batch_perturb_df)
                all_threshold.append(batch_threshold_df)

        if all_perturb:
            self.perturbations = pd.concat(all_perturb, ignore_index=True)
            self.all_thresholds = pd.concat(all_threshold, ignore_index=True)
        else:
            self.perturbations = pd.DataFrame(columns=["cell", "gRNA", "UMI_counts"])
            self.all_thresholds = pd.DataFrame(columns=["gRNA", "Threshold"])

        return self.perturbations, self.all_thresholds

    # =============================================================================
    # Extracting cell level guide assignment processing
    # =============================================================================
    @property
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

        # Mark cells with multiple sgRNAs
        assignment_crispat["guide_id"] = np.where(
            assignment_size.reindex(assignment_crispat["cell"].values, fill_value=1).values > 1,
            "multi_sgRNA",
            assignment_crispat["gRNA"],
        )

        assignment_crispat.set_index("cell", inplace=True)

        return assignment_crispat
    @property
    def make_binary_obsm(self):
        """
        Returns
        -------
        binary_matrix : pd.DataFrame
            Binary cell x guide matrix.
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
# Per-sample job runner (child process)
# =============================================================================

def run_guide_jobs(run):
    """
    Worker for a single (processed_dir, colname, value, inner_cores) job.

    Called as a NON-DAEMON child process from main()'s NoDaemonPool.
    Spawns grandchild processes via run_guide_assign's inner Pool.
    """
    processed_dir, colname, value, inner_cores = run

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
    # This child is non-daemonic, so it CAN spawn grandchildren
    # inner_cores is pre-computed by main() to avoid oversubscription
    guide_assign.run_guide_assign(num_cores=inner_cores)

    binary_matrix = guide_assign.make_binary_obsm
    assignment_crispat = guide_assign.merge_guide_assign

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
# CLI entry point (parent process)
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
        "--cellranger_dir",
        type=str,
        required=True,
        help="Path to cellranger directory",
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
        default=None,
        help="Number of outer sample-level workers (default: number of samples)",
    )
    args = parser.parse_args()

    exp_meta = pd.read_csv(os.path.join(args.cellranger_dir, args.expmeta))

    # Drop any unnamed index-like columns
    exp_meta = exp_meta.loc[:, ~exp_meta.columns.str.startswith("Unnamed")]

    # Build list of jobs: one per (colname, value)
    jobs = [
        (args.processed_dir, colname, value)
        for colname, coldata in exp_meta.items()
        for value in coldata
    ]

    n_jobs = len(jobs)
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    total_cpus = min(mp.cpu_count(), slurm_cpus)

    # Outer workers = number of samples (or user override), capped at total CPUs
    nprocs = min(args.nprocs or n_jobs, n_jobs, total_cpus)

    # Split remaining CPUs evenly across outer workers for inner guide parallelism
    inner_cores = max(1, total_cpus // nprocs)

    print(f"Processing {n_jobs} samples: {nprocs} outer workers x {inner_cores} inner cores "
          f"({total_cpus} total CPUs)")

    # Attach inner_cores to each job tuple so child workers know their CPU budget
    runs = [
        (processed_dir, colname, value, inner_cores)
        for processed_dir, colname, value in jobs
    ]

    # NoDaemonPool: children are non-daemonic, so each can spawn its own
    # inner Pool of grandchildren for guide-level parallelism
    with NoDaemonPool(processes=nprocs) as pool:
        for colname, value, status in pool.imap_unordered(run_guide_jobs, runs):
            print(f"{status}: {value}_{colname}")


if __name__ == "__main__":
    main()