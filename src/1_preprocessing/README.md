# Preprocessing with SLURM & Interactive Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../../LICENSE)
![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

---

**Preprocessing** module of the CRISPR Perturb-Seq Analysis! This section covers SLURM job submission, running interactive Jupyter-based analysis, and expected output.

<details>
<summary> Table of Contents</summary>

- [How to Run with SLURM](#how-to-run-with-slurm)
- [Expected outputs](#expected-outputs)
- [Interactive Usage](#interactive-usage)
- [Parameters & Tips](#parameters--tips)
</details>

---

## How to Run with SLURM

You can use the provided `submit_preprocessing.sh` script to submit preprocessing tasks on a cluster. Example job config:

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

---
## Expected Outputs

Initial QC was done to remove low quality cells i.e, high mitochondrial percent, and cells have less than 200 genes.
```
  Output files: {sample_name_lane}_gex_preprocessed.h5ad
		        {sample_name_lane}_crispr_preprocessed.h5ad
```
---

## 📔 Interactive Usage

Prefer a notebook experience? You can run preprocessing directly with Jupyter Notebooks:

```bash
jupyter notebook preprocess_tutorial.ipynb
```
---

## ⚙️ Parameters & Tips

- Main Python script: `basic_processing.py`
- Notebooks live here for demonstration and stepwise analysis
- Check for any prefix you might use for guide names which might need some clean up
- Use the `--help` flag for CLI options:

```bash
python3 basic_processing.py --help
```
---
