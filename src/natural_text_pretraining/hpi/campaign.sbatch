#!/bin/bash
#SBATCH --job-name=dolma-a100
#SBATCH --account=sci-demelo
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --nodes=1
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100&GPU_MEM:40GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --array=0-14%3
set -euo pipefail
source_dir=$(realpath "$1")
data_dir=$(realpath "$2")
campaign_dir=$(realpath "$3")
python_bin=$(realpath -s "$4")
smoke=$(realpath "$5")
conditions=(discrete learned_6000 learned_12000 learned_18000 fixed_18000)
seeds=(101 102 103)
index=${SLURM_ARRAY_TASK_ID:?Submit as an array}
[[ "$index" =~ ^[0-9]+$ && "$index" -lt 15 ]] || exit 2
condition=${conditions[$((index % 5))]}
seed=${seeds[$((index / 5))]}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
"$python_bin" "$source_dir/hpi/check_campaign.py" \
    --source "$source_dir" --data "$data_dir" --smoke "$smoke" --phase training
printf 'Training condition=%s seed=%s array_task=%s node=%s\n' \
    "$condition" "$seed" "$index" "${SLURMD_NODENAME:?}"
# exec preserves train.sbatch's signal-forwarding handler for the batch shell.
exec bash "$source_dir/hpi/train.sbatch" \
    "$source_dir" "$data_dir" "$campaign_dir" "$python_bin" "$condition" "$seed"
