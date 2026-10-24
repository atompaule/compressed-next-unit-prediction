#!/bin/bash
#SBATCH --job-name=dolma-smoke
#SBATCH --account=sci-demelo
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --nodes=1
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100&GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
source_dir=$(realpath "$1")
data_dir=$(realpath "$2")
output_dir=$(realpath -m "$3")
# Preserve the venv/bin/python symlink: resolving it bypasses the environment.
python_bin=$(realpath -s "$4")
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
exec "$python_bin" -u "$source_dir/smoke.py" --data "$data_dir" --out "$output_dir" --device cuda
