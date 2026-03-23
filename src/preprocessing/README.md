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
