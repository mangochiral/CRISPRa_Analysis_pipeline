# Guide-assignment

Part of the [CRISPRa Analysis Pipeline](https://github.com/mangochiral/CRISPRa_Analysis_pipeline), this module provides tools for assigning guide RNAs (gRNAs) to cells in CRISPR screening experiments.

## Features

- Assigns candidate gRNAs to cells based on sequencing data
- Produces assignment tables compatible with downstream analysis and visualization
- Generates QC statistics for each targeting
- Includes summary notebooks for QC and data distribution insights
- Easily integrates into the larger CRISPRa pipeline

## Usage

### 1. Input Preparation

- Requires Anndata objects from CRISPR assays containing gRNA UMIs.
- Supported input format: `.h5ad` files as described in the code.

### 2. Running Guide Assignment

You can run assignments via a standalone Python script (recommended for batch/cluster environments) or interactively in a Jupyter notebook.

#### Example: SLURM Batch Script (Guide Assignment)

```bash
#!/bin/bash
#SBATCH --job-name=guide_assignment
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/guide_assignment_%j.out
#SBATCH --error=logs/guide_assignment_%j.err

echo "Starting guide assignment on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs"

# Restrict BLAS/OpenMP threading; Python script manages its own parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 guide_assignment_parallel.py \
    --processed_dir <PATH TO PROCESSED .h5ad DIRECTORY> \
    --cellranger_dir <PATH TO CELLRANGER DIRECTORY WITH EXPERIMENT METADATA CSV> \
    --expmeta experiments_meta.csv \
    --nprocs 8

echo "Completed all samples"
```

Alternatively, open `guide_assignment_parallel.ipynb` in JupyterLab and follow the documented cells.

#### Example: SLURM Batch Script (Run QC Stats)

Use this script to run QC statistics with `qc_stats_heavy_load.py` in array mode for parallel sample processing:

```bash
#!/bin/bash
#SBATCH --job-name=guide_stats
#SBATCH --array=0-7
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/guide_stats%A_%a.out
#SBATCH --error=logs/guide_stats%A_%a.err

# Safer multiprocessing with numpy/BLAS
export OMP_NUM_THREADS=1

python3 qc_stats_heavy_load.py \
  --processed_dir /groups/marson/chandrima/reanalysis_gw_crispr \
  --cellranger_dir /groups/marson/chandrima/ron_data_gw_crispr/Diff053/cellranger \
  --expmeta expirements_meta.csv \
  --nprocs "${SLURM_CPUS_PER_TASK}" \
  --multiguide False
```

---

### 3. Parameters and Options

- Filtering options (e.g., minimum on-target score)
- Aggregation and summary of guide statistics

For detailed parameterization and available options, refer to the provided notebooks or run scripts with the `--help` flag.

---

## Outputs

### Guide Assignment Outputs

For each experiment/lane, the following files are generated:

- `guide_assignment.csv`:  
  Main table assigning candidate guides to cells.

- `guide_threshold`:  
  Thresholding information used for guide calls.

- `<expr_lane>_processed_guide.csv`:  
  Per-lane processed guide assignments, with labels per cell/sample.

- `<expr_lane>_gex_guide.h5ad`:  
  AnnData `.h5ad` file with all guide-assigned metadata (assignment columns inside `.obs`), and gene expression (`.X`) for the assigned cells.  
  **This file is input to the QC statistics step (`qc_stats_heavy_load.py`).**

### QC Stats Outputs

Each run of `qc_stats_heavy_load.py` produces, for each experiment/lane:

- `<expr_lane>_guide_count_info.csv`:  
  For each guide:
    - Number of cells assigned that guide
    - Number of NTC (non-targeting control) cells
    - mRNA expression levels of all cells assigned to that guide
    - mRNA expression levels of corresponding NTC cells

---

## Data Exploration & Visualization

Several Jupyter notebooks are provided for further exploration and QC visualization:

- **Guide_type_distribuition.ipynb** — Explore the distribution of guide types across your data (e.g., targeting, non-targeting).
- **Guides_per_cell_distribution_plots.ipynb** — Visualize the distribution of the number of guides assigned per cell.

Use these notebooks to gain insights into the assignment quality and experimental design.
