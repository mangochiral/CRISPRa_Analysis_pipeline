# Perturb-seq Analysis Pipeline Template

## General Note

This repository serves as a **template** for processing and analyzing Perturb-seq data. The intent is for this codebase to act as a general starting point:  
- **Do not edit scripts in this repository for project-specific needs.**  
- Instead, **copy or fork this repository** into a new, project-specific folder or repo, and make any project-specific edits there.  
This ensures that the template remains general-purpose and reusable for anyone starting a new Perturb-seq analysis project.

## Purpose

The scripts and workflows provided here are designed to cover the full pipeline for preprocessing, analyzing, and visualizing data from CRISPR-based Perturb-seq experiments. The focus is on modularity and generalizability:
- Fundamentally there are 4 general steps:
1. Preprocessing of raw 10x h5ad and splitting them to  gene expression Anndata objects and CRISPR Anndata objects.
2. Guide Assignment: This workflow only used mixture model of Poisson-Guassian distribution for the assignment to the cells. 
**Note** Threshold of guide UMI counts for filtering out cells should be according to experiment design. Additional processing the guide assigned anndata object will be required before downstream steps.
3. Pseudobulking: The current workflow is a template, on a confounders, technical and biological replicate should be considered and pseudobulking should be done accordingly. Before the next steps prior prep is required for DGE analysis.
4. DGE analysis: This workflow uses DESeq2 for differential analysis caused by perturbation

- 
- 

## Directory Structure

```
src/
├── 1_preprocessing/
│   ├── README.md
│   ├── Sanity_Check.ipynb
│   ├── basic_processing.py
│   ├── preprocess_adata.py
│   └── preprocess_tutorial.ipynb
├── 2_guide-assignment/
│   ├── Guide_efficiency_plots.ipynb
│   ├── Guide_efficiency_test_stats.ipynb
│   ├── Guide_type_distribuition.ipynb
│   ├── Guides_per_cell_distribution_plots.ipynb
│   ├── Prep_for_guide_assignment.ipynb
│   ├── README.md
│   ├── Sanity_check.ipynb
│   ├── guide_assignment_parallel.py
│   └── qc_stats_heavy_load.py
├── 3_pseudobulk/
│   ├── Keep_singlets_prep_adata.ipynb
│   ├── prep_DE_merge_pseudobulk.ipynb
│   └── pseudobulk_by_lane.py
└── 4_DGE_analysis/
    └── Deseq2_pseudobulk.py
```

## Best Practices

- **Use metadata:** Specify experiment type and other details in metadata files or AnnData objects, not in scripts.
- **Fork, then customize:** For a new project, fork or copy this repo, then make modifications in your project copy only.
- **Multiple script versions:** Provide both notebook (.ipynb) and script (.py) versions of analyses where possible, to support different user needs and computational environments. Examples:
  - Notebooks for interactive or exploratory analysis.
  - SLURM job scripts for high-throughput or large-scale processing.

## Notes

- **Jupyter notebooks (`.ipynb`)** are provided for interactive analyses and visualizations.
- **Scripts are designed to be modular.** Run only the steps relevant for your experiment.
- **Metadata-driven:** Each step should rely on experiment information provided via files/AnnData.

---

**To contribute improvements:**  
Generalize scripts further, improve modularity, or add new template scripts/noebooks—but avoid project-specific edits here.
