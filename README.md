# CRISPRa Analysis Pipeline

## Overview

This repository provides tools and scripts for preprocessing, analyzing, and visualizing data from CRISPR experiments.

## Directory Structure

```
src/
├── basic_processing.py
│   └── preprocess_adata.py          # Data preprocessing utilities
├── guide_assignment_parallel.py     # Assign guides in parallel
├── qc_stats.py                     # Quality control statistics (standard)
├── qc_stats_heavy_load.py          # QC for large datasets (summary, t-test)
├── guide_efficiency_qc_stats.ipynb # Guide efficiency QC with t-test
├── qc_plots/
│   ├── qc_guide_type_distribution.ipynb
│   └── qc_plot_cumulative_perturbation_fraction_of_genes.ipynb
├── pseudobulk_by_lane.py           # Pseudobulk analysis per lane
│   └── prep_DE_merge_pseudobulk.ipynb # Prepare DE and merge pseudobulk results
├── Deseq2_pseudobulk.py            # Run DESeq2 analysis on pseudobulked data
```

## Getting Started

1. **Preprocess your data** using `basic_processing.py` and `preprocess.py`.
2. **Assign guides** with `guide_assignment_parallel.py`.
3. **Quality control**: Use `qc_stats.py`, `qc_stats_heavy_load.py`, or the notebooks in `qc_plots/` for QC and visualization.
4. **Pseudobulk analysis**: See `pseudobulk_by_lane.py` and related notebooks.
5. **Differential expression analysis**: Use `Deseq2_pseudobulk.py`.

## Submitting Preprocessing as a SLURM Job

You can use the provided `submit_preprocessing.sh` script to submit preprocessing tasks on a SLURM cluster.

```bash
#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --time=1:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=N         # <-- N CPUs per job
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err

export OMP_NUM_THREADS=1

# Run the guide assignment
python3 basic_processing.py \
  --cellranger_dir <PATH TO CELLRANGER DIRECTORY> \
  --experiment_info <EXPERIMENT META INFO CSV> \
  --mt_pct <MITOCHONDRIAL THRESHOLD> \
  --prefix None \
  --exp crispr \
  --filter_cells <FALSE if no filter must be executed> \
  --output_dir <PATH TO OUTPUT DIRECTORY> \
  --nprocs "${SLURM_CPUS_PER_TASK}"

echo "Completed!"
```

## Getting Started

1. **Preprocess your data** using `basic_processing.py` and `preprocess.py`.
2. **Assign guides** with `guide_assignment_parallel.py`.
3. **Quality control**: Use `qc_stats.py`, `qc_stats_heavy_load.py`, or the notebooks in `qc_plots/` for QC and visualization.
4. **Pseudobulk analysis**: See `pseudobulk_by_lane.py` and related notebooks.
5. **Differential expression analysis**: Use `Deseq2_pseudobulk.py`.

## Notes

- Jupyter notebooks (`.ipynb`) provide interactive analysis and plotting.
- Each script is modular—run the steps relevant for your experiment.
