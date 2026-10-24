#!/bin/bash
#SBATCH --job-name=dolma-train
#SBATCH --account=sci-demelo
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --nodes=1
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100&GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
set -euo pipefail
source_dir=$(realpath "$1")
data_dir=$(realpath "$2")
campaign_dir=$(realpath -m "$3")
# Preserve the venv/bin/python symlink: resolving it bypasses the environment.
python_bin=$(realpath -s "$4")
condition="$5"
seed="$6"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
# Signal forwarding allows an atomic recovery checkpoint before the time limit.
child=""
trap '[[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
"$python_bin" -u "$source_dir/train.py" --data "$data_dir" --out "$campaign_dir/$condition/$seed" --condition "$condition" --seed "$seed" --device cuda &
child=$!
set +e
wait "$child"
result=$?
if kill -0 "$child" 2>/dev/null; then wait "$child"; result=$?; fi
exit "$result"
