#!/bin/bash
#SBATCH --job-name=dolma-prepare
#SBATCH --account=sci-demelo
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --nodes=1
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100&GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail
source_dir=$(realpath "$1")
output_dir=$(realpath -m "$2")
environment=$(realpath -m "$3")
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export DOLMA_MIN_FREE_GIB=15
mkdir -p "$output_dir"
export HF_HOME="$output_dir/hf_home"
bash "$source_dir/hpi/setup_environment.sh" "$source_dir" "$environment"
python_bin="$environment/venv/bin/python"
exec "$python_bin" -u "$source_dir/prepare.py" --out "$output_dir" --device cuda
