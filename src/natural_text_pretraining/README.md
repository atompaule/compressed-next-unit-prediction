# Chapter 6: pretraining on natural text at a small scale

This self-contained directory compares discrete GPT-2 and CoNUP on the same
Dolma subset. `sentence`/`SentenceDot` are internal CoNUP names.
[experiment.json](experiment.json) defines the pinned assets, data, model and
training settings; [PHRASE_RULES.md](PHRASE_RULES.md) specifies grammatical chunking.
Use the chapter's `requirements-linux.txt`, not the root environment.

| Condition | Rotations | Negative chunks/update |
|---|---|---:|
| `discrete` | — | Full token vocabulary |
| `learned_6000` | Learned | 6,000 |
| `learned_12000` | Learned | 12,000 |
| `learned_18000` | Learned | 18,000 |
| `fixed_18000` | Fixed | 18,000 |

Each condition uses seeds 101–103: 15 runs. Models train from scratch for one epoch
with batch 128, microbatch 32 and BF16. Evaluation uses the end-of-budget checkpoint.
Native token/chunk losses have different units; continuation selection provides
the shared comparison.

## 1. Freeze source and prepare data

Run these commands in **Bash** from a clean, committed checkout on the cluster.
Submit jobs manually. The Slurm files target HPI's `sci-demelo` account and
`pot-hpi-gpu-batch` partition; adapt account/partition settings before freezing
source if needed. The campaign requires `NVIDIA A100-SXM4-40GB` and up to 96 GiB
host RAM for preparation. Install `uv` at `$HOME/.local/bin/uv` or set `UV_BIN`.

```bash
set -euo pipefail
campaign="$HOME/dolma_sentence_dot_campaign"
source_dir="$campaign/source"
environment="$HOME/dolma_sentence_dot_env"
python_bin="$environment/venv/bin/python"
preparation="$HOME/dolma_sentence_dot_data"
mkdir "$campaign"
mkdir "$campaign/logs"
cp -R src/natural_text_pretraining "$source_dir"
git rev-parse HEAD > "$campaign/repository_commit.txt"
sbatch --output="$campaign/logs/prepare_%j.log" \
  "$source_dir/hpi/prepare.sbatch" "$source_dir" "$preparation" "$environment"
```

The preparation job installs Python 3.11 and pinned Linux dependencies on the
compute node, downloads inputs, calibrates a balanced-document quota across web,
encyclopedia and science, and creates a 98%/1%/1% document split. The target is
approximately 2.1M training samples; inspect the actual sample/token counts.

Wait for completion, then review `$preparation/calibration.json`,
`$preparation/data/verification.json` and `$preparation/data/review.html`.
The verified token arrays and catalogues are under `$preparation/data`.
Preparation resumes its SQLite cache after interruption; use the same source,
configuration and paths. A changed protocol requires a fresh output directory.

Plan roughly 75 GiB for the environment, downloads, corpus and campaign, plus
headroom for temporary catalogue vectors and backups. This is an estimate, not
an upper bound. Catalogue vectors require `4 × 512 × catalogue_size` bytes
(about 16.5 GiB for the reported corpus). Preparation stops resumably below 15 GiB
free; campaign startup requires 18 GiB for training and 20 GiB for evaluation.

## 2. Check the production setup

Using the same shell variables, submit the bounded GPU check after preparation:

```bash
sbatch --output="$campaign/logs/smoke_%j.log" \
  "$source_dir/hpi/smoke.sbatch" "$source_dir" "$preparation/data" \
  "$campaign/smoke" "$python_bin"
```

Wait for `$campaign/smoke/report.json` to report `passed` for all five conditions.
This checks production shapes, resume, sample order, learning rates and a long-input
stress batch. It uses a reduced validation monitor and creates diagnostic weights;
production starts fresh. Allow about 4 GiB for these outputs. Keep the source and
configuration unchanged through training, resume and evaluation.

## 3. Train and evaluate

```bash
smoke="$campaign/smoke/report.json"
train_job=$(sbatch --parsable --output="$campaign/logs/train_%A_%a.log" \
  "$source_dir/hpi/campaign.sbatch" "$source_dir" "$preparation/data" \
  "$campaign" "$python_bin" "$smoke")
train_job=${train_job%%;*}
printf '%s\n' "$train_job" > "$campaign/train_job.txt"
eval_job=$(sbatch --parsable --dependency="afterok:$train_job" \
  --output="$campaign/logs/evaluation_%j.log" \
  "$source_dir/hpi/evaluate.sbatch" "$source_dir" "$preparation/data" \
  "$campaign" "$python_bin" "$smoke")
printf '%s\n' "${eval_job%%;*}" > "$campaign/evaluation_job.txt"
```

The array runs all 15 combinations, at most three concurrently. Indices 0–4 use
seed 101, 5–9 seed 102, and 10–14 seed 103, in the condition order above.
Training requests 24 hours per run; evaluation requests 48 hours. These are limits,
not runtime estimates. Inspect jobs with `squeue -r -j "$train_job"` and `sacct`.

An interrupted run saves `last.pt` and exits 75. Resubmit only unfinished array
indices by adding `--array=INDEX[,INDEX...]%3` to the same training command; it
restores optimizer, RNG, data position and schedule. Keep completed runs. Recreate
the evaluation dependency using all outstanding replacement/original job IDs
(`afterok:ID1:ID2`); a failed original dependency will not release evaluation.
If only evaluation submission fails, submit evaluation again without restarting
training.

## 4. Results and backup

Each `$campaign/<condition>/<seed>/` contains `config.json`, training records,
`last.pt` for recovery and `final.pt` for evaluation. Under `$campaign/evaluation/`:

- `summary.json`: continuation-selection metrics, coverage, seed summaries and
  paired document-bootstrap intervals; `summary.tsv`: accuracy, time and FLOPs.
- `<condition>/<seed>/result.json`: per-question records and provenance.
- `learning_curves.png` / `.pdf`: native validation losses.
- `qualitative.html`: matched greedy/sampled examples using the training catalogue.

Inspect `summary.json` for `status: complete`. Continuation selection uses eight
matched candidates; it does not measure unrestricted generation. Training and
validation time/FLOPs are recorded separately; FLOPs use operation accounting.

After evaluation, create a portable backup from an allocated compute node:

```bash
"$python_bin" "$source_dir/export_results.py" --campaign "$campaign" \
  --data "$preparation/data" --out "$HOME/dolma_sentence_dot_backup" --include-data
```

The backup contains final weights, results, source and canonical data, with hashes.
Recovery checkpoints and download caches are excluded. Omitting `--include-data`
keeps only data provenance and the tokenizer. Use a fresh backup destination with
enough free space, then copy it off the cluster.
