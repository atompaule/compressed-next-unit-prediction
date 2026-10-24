#!/bin/bash
#SBATCH --job-name=dolma-evaluate
#SBATCH --account=sci-demelo
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --nodes=1
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100&GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
set -euo pipefail
source_dir=$(realpath "$1")
data_dir=$(realpath "$2")
campaign_dir=$(realpath "$3")
# Preserve the venv/bin/python symlink: resolving it bypasses the environment.
python_bin=$(realpath -s "$4")
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
if [[ $# -ge 5 ]]; then
    "$python_bin" "$source_dir/hpi/check_campaign.py" \
        --source "$source_dir" --data "$data_dir" --smoke "$5" --phase evaluation
fi
exec "$python_bin" -u "$source_dir/evaluate.py" --data "$data_dir" --campaign "$campaign_dir" --out "$campaign_dir/evaluation" --device cuda
