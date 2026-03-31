# Guide Assignment

This module is part of the [CRISPRa Analysis Pipeline](https://github.com/mangochiral/CRISPRa_Analysis_pipeline), providing tools for guide RNA (gRNA) assignment, quality control, and visualization for CRISPR-based screening experiments. The workflow supports CRISPRa, CRISPRi, and KO screens.

---

## Overview

- Assign candidate gRNAs to cells from sequencing data.
- Produce assignment tables for downstream analysis and visualization.
- Generate QC statistics and efficiency metrics for guides.
- Provide summary notebooks for data exploration.
- Integrate seamlessly with the larger CRISPR pipeline.

---

## Recommended Workflow

### Step 1: Prepare CRISPR AnnData

**(Optional for KO: Knockout screens)**  
- If both CRISPRa and CRISPRi guides are present, and the main experiment is CRISPRi, remove all CRISPRa guides from the dataset.  
- Use the notebook: **Prep_for_guide_assignment.ipynb**

### Step 2: Guide Assignment

- Assign guides to cells using:
  - `guide_assignment_parallel.py`

### Step 3: Guide Efficiency Statistics

- Compute guide efficiency statistics with `qc_stats_heavy_load.py`.

### Step 4: Guide Type Distribution Plot

- Explore guide type distribution (e.g., targeting, non-targeting) using `Guide_type_distribuition.ipynb`.

### Step 5: Guide Efficiency Testing & Plots

- **Guide_efficiency_test_stats.ipynb** — Performs t-tests to statistically evaluate the efficiency of each targeting guide.
- **Guide_efficiency_plots.ipynb** — Generates plots to visualize individual guide efficiency and assess guide performance.

---

## Usage Instructions

### Input Preparation

- Requires AnnData objects (usually `.h5ad`) containing gRNA UMI counts from CRISPR assays.
- Supported format: `.h5ad` files as defined in the pipeline.

### Running Guide Assignment

- Assignment can be run via batch script:

#### Example: SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=guide_assignment
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/guide_assignment_%j.out
#SBATCH --error=logs/guide_assignment_%j.err

# Restrict BLAS/OpenMP threading; Python script manages parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 guide_assignment_parallel.py \
    --processed_dir <PATH TO PROCESSED .h5ad DIRECTORY> \
    --cellranger_dir <PATH TO CELLRANGER DIRECTORY WITH EXPERIMENT METADATA CSV> \
    --expmeta experiments_meta.csv \
    --nprocs 8
```



#### Running QC Statistics

Use `qc_stats_heavy_load.py` to calculate QC metrics in array (parallel) mode:

```bash
#!/bin/bash
#SBATCH --job-name=guide_stats
#SBATCH --array=0-7
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/guide_stats%A_%a.out
#SBATCH --error=logs/guide_stats%A_%a.err

export OMP_NUM_THREADS=1

python3 qc_stats_heavy_load.py \
  --processed_dir <PROCESSED_DATA_DIR> \
  --cellranger_dir <CELLRANGER_DIR> \
  --expmeta experiments_meta.csv \
  --nprocs "${SLURM_CPUS_PER_TASK}" \
  --multiguide False
```

---

## Outputs

### Guide Assignment

For each experiment or lane, the following files are produced:
- `guide_assignment.csv`: Table assigning candidate guides to cells.
- `guide_threshold`: Thresholding info for guide calls.
- `<expr_lane>_processed_guide.csv`: Per-lane processed assignment table.
- `<expr_lane>_gex_guide.h5ad`: AnnData object with guide-assigned metadata and gene expression.  
  _**Input to QC statistics step (`qc_stats_heavy_load.py`)**_

### QC Statistics

Each run of `qc_stats_heavy_load.py` produces (per experiment/lane):
- `<expr_lane>_guide_count_info.csv`:  
  - Number of cells per guide  
  - Number of non-targeting control (NTC) cells  
  - mRNA expression for assigned and NTC cells

---

## Data Exploration & Visualization

Use the included notebooks for further QC and exploration:

- **Guide_type_distribuition.ipynb** — Visualizes distribution of guide types (e.g., targeting, non-targeting).
- **Guides_per_cell_distribution_plots.ipynb** — Plots the number of guides assigned per cell.
- **Guide_efficiency_test_stats.ipynb** — Performs t-tests to evaluate the efficiency of each targeting guide.
- **Guide_efficiency_plots.ipynb** — Visualizes individual guide efficiency results.

