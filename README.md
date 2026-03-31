# Perturb-seq Analysis Pipeline Template

## General Note

This repository serves as a **template** for processing and analyzing Perturb-seq data. The intent is for this codebase to act as a general starting point:  
- **Do not edit scripts in this repository for project-specific needs.**  
- Instead, **copy or fork this repository** into a new, project-specific folder or repo, and make any project-specific edits there.  
This ensures that the template remains general-purpose and reusable for anyone starting a new Perturb-seq analysis project.

## Purpose

The scripts and workflows provided here are designed to cover the full pipeline for preprocessing, analyzing, and visualizing data from CRISPR-based Perturb-seq experiments. The focus is on modularity and generalizability:
- **No hardcoding of experiment-specific details** (e.g., whether experiments are CRISPRi, CRISPRa, etc.).
- The type of perturbation should be specified as metadata (e.g., in the AnnData object) during initial basic preprocessing.
- All downstream scripts should handle experiment-specific behavior internally, based on this metadata.

## Directory Structure

```
src/
├── preprocessing/
│   ├── README.md
│   ├── Sanity_Check.ipynb
│   ├── basic_processing.py
│   ├── preprocess_adata.py
│   └── preprocess_tutorial.ipynb
├── guide-assignment/
│   ├── Guide_efficiency_plots.ipynb
│   ├── Guide_efficiency_test_stats.ipynb
│   ├── Guide_type_distribuition.ipynb
│   ├── Guides_per_cell_distribution_plots.ipynb
│   ├── Prep_for_guide_assignment.ipynb
│   ├── README.md
│   ├── Sanity_check.ipynb
│   ├── guide_assignment_parallel.py
│   └── qc_stats_heavy_load.py
├── pseudobulk/
│   ├── Keep_singlets_prep_adata.ipynb
│   ├── prep_DE_merge_pseudobulk.ipynb
│   └── pseudobulk_by_lane.py
└── DGE_analysis/
    └── Deseq2_pseudobulk.py
```

## Best Practices

- **Keep scripts general:** Avoid project-specific assumptions or hard-coded parameters.
- **Use metadata:** Specify experiment type and other details in metadata files or AnnData objects, not in scripts.
- **Fork, then customize:** For a new project, fork or copy this repo, then make modifications in your project copy only.
- **Multiple script versions:** Provide both notebook (.ipynb) and script (.py) versions of analyses where possible, to support different user needs and computational environments. Examples:
  - Notebooks for interactive or exploratory analysis.
  - SLURM job scripts for high-throughput or large-scale processing.

## Getting Started (Template)

### 1. Preprocessing

Use `basic_processing.py` and utilities in `preprocessing/` to clean and preprocess your raw data.  
Choose between:
- `.py` scripts for command-line/SLURM processing
- `.ipynb` notebooks for interactive analysis (see `qc_plots/` and related files)

### 2. Guide Assignment

Assign guides in parallel with `guide_assignment_parallel.py`.

### 3. Quality Control

Run QC using:
- `qc_stats.py`
- `qc_stats_heavy_load.py` (for large datasets)
- Notebooks in `qc_plots/` for interactive QC and visualization

### 4. Pseudobulk & Differential Expression

- Perform pseudobulk analysis with `pseudobulk_by_lane.py`.
- Run DESeq2 with `Deseq2_pseudobulk.py`.

### 5. SLURM Example

A SLURM submission script (`submit_preprocessing.sh`) is provided for batch processing.
Update the placeholders for your project as needed.

```bash
#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --time=1:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=N
#SBATCH --output=logs/preprocess_%A_%a.out
#SBATCH --error=logs/preprocess_%A_%a.err

export OMP_NUM_THREADS=1

python3 basic_processing.py \
  --cellranger_dir <PATH TO CELLRANGER DIRECTORY> \
  --experiment_info <EXPERIMENT META INFO CSV> \
  --mt_pct <MITOCHONDRIAL THRESHOLD> \
  --prefix None \
  --exp <crisper/perturbation_type> \
  --filter_cells <TRUE/FALSE> \
  --output_dir <PATH TO OUTPUT DIRECTORY> \
  --nprocs "${SLURM_CPUS_PER_TASK}"

echo "Completed!"
```

## Notes

- **Jupyter notebooks (`.ipynb`)** are provided for interactive analyses and visualizations.
- **Scripts are designed to be modular.** Run only the steps relevant for your experiment.
- **Metadata-driven:** Each step should rely on experiment information provided via files/AnnData, not hardcoded.

---

**To contribute improvements:**  
Generalize scripts further, improve modularity, or add new template scripts/noebooks—but avoid project-specific edits here.

---

**Remember: For project-specific pipelines, fork or copy this repo and apply your changes to your copy only.**
