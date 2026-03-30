# guide-assignment

This module is part of the [CRISPRa Analysis Pipeline](https://github.com/mangochiral/CRISPRa_Analysis_pipeline) and provides tools for assigning guide RNAs (gRNAs) to cells for CRISPR screening experiments.

## Features

- Assigns candidate guide RNAs to cells
- Outputs assignment tables compatible with downstream analysis and visualization.
- QC stats of each targeting
- Easily integrates with the overall CRISPRa pipeline.

## Usage

1. **Input Preparation**
   - CRISPR assay Anndata objects containing guide RNA UMIs.
   - Supported formats: `.h5ad` as described in the code.

2. **Running guide-assignment**
   - Launch the corresponding Python script.
   - Example command:

     ```bash
	#!/bin/bash
#SBATCH --job-name=guide_assignment
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/guide_assignment_%j.out
#SBATCH --error=logs/guide_assignment_%j.err

echo "Starting guide assignment on $(hostname) with ${SLURM_CPUS_PER_TASK} CPUs"

# Pin BLAS/OpenMP to 1 thread — the Python script manages its own parallelism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Single job processes all samples in parallel:
#   main (parent) → NoDaemonPool with 8 children (one per sample)
#   each child   → inner Pool with 8 grandchildren (64 / 8 = 8 cores each)
#
# --nprocs controls outer workers (default = number of samples in metadata)
python3 guide_assignment_parallel.py \
    --processed_dir /groups/marson/chandrima/reanalysis_gw_crispr \
    --cellranger_dir /groups/marson/chandrima/ron_data_gw_crispr/Diff053/cellranger \
    --expmeta expirements_meta.csv \
    --nprocs 8

echo "Completed all samples"
     
     ```

   - Or open the notebook in JupyterLab and proceed through the documented cells.

3. **Parameters and Options**
   - Filtering options (e.g., minimum on-target score).
   - Aggregated guide statistics.

Refer to the provided notebooks or scripts for in-depth usage and parameter settings.

## Example

```python
# Example function call
from assign_guides import assign_guides_to_targets

assign_guides_to_targets('guides.csv', 'targets.csv', output='assigned_guides.csv')
```

Check the `example/` subdirectory for demonstration files and outputs.



## Acknowledgements

Inspired by published CRISPRa libraries and community-developed analysis pipelines.

---

