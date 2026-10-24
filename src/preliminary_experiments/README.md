# Pretrained-model experiments

This guide covers constructed vocabulary mixtures in Chapter 3, reinforcement
learning in Appendix A, Soft Thinking versus random perturbations in Appendix B,
and the confidence interval methods described in Appendix C.

Run from the repository root after the [shared setup](../../README.md#setup).
These experiments use pretrained models. Descriptions and convergence require
Apple Silicon/MPS; benchmarks need enough accelerator memory for the chosen model.

## Model access and downloads

Accept any required Llama/Gemma access terms and authenticate, then download each
model before running benchmarks or convergence. For example:

```bash
.venv/bin/hf auth login
.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("meta-llama/Llama-3.1-8B-Instruct")'
```

Benchmark datasets download automatically to
`data/preliminary_experiments/huggingface/`. Use fresh result directories.

## Chapter 3: constructed vocabulary mixtures

These commands use Llama 3.1 8B and the seven cases in `concept_descriptions.py`:

```bash
.venv/bin/python src/preliminary_experiments/mixture_experiments/concept_descriptions.py
.venv/bin/python src/preliminary_experiments/mixture_experiments/representational_convergence.py \
  --output-dir results/preliminary_experiments/convergence
```

Descriptions and nearest-neighbor comparisons go to `results/preliminary_experiments/mixtures/`.
Convergence writes JSON, text and figures; `--plot-only-json PATH` redraws saved
measurements without loading the model.

## Appendix A: reinforcement learning

The GSM8K preset in `reinforcement_learning/config.py` uses Llama 3.1 8B or
Llama 3.2 3B, four questions × four rollouts, LoRA rank 32/alpha 64, learning rate
1e-5 and temperature one. Policies: `discrete`, `gaussian_noise`, `token_sampling`,
`dirichlet_sampling`. Repeat training and evaluation for both models and all four
policies, choosing separate run and result directories.

```bash
.venv/bin/python -m src.preliminary_experiments.reinforcement_learning.train \
  --model-id meta-llama/Llama-3.1-8B-Instruct --approach dirichlet_sampling \
  --output-dir checkpoints/preliminary_experiments/rl/llama8b_dirichlet

CKPT_DIR=checkpoints/preliminary_experiments/rl/llama8b_dirichlet \
EVAL_LOG_DIR=results/preliminary_experiments/rl/llama8b_dirichlet_evaluation \
.venv/bin/python -m src.preliminary_experiments.reinforcement_learning.evaluate

.venv/bin/python -m src.preliminary_experiments.reinforcement_learning.plot_training \
  --run-dir checkpoints/preliminary_experiments/rl/llama8b_dirichlet \
  --output-dir results/preliminary_experiments/rl/llama8b_dirichlet_training
```

The corrected loop uses 1,000 batches and 250 optimizer updates, with checkpoints
every 100 batches. Add `--resume` to the same training command to restore optimizer
and random states; configuration changes are rejected. Original trajectories are
not exactly reconstructable. `--print-config` previews training settings.

Evaluation infers the model and canonical deterministic policy from `run_config.json`
and uses the full 1,319-question test split (`N` limits diagnostic runs). It writes
JSON and text with accuracy and individual answers. For base-model comparisons,
set `CKPT=base`, `MODEL_ID=...` and `METHOD=discrete`, `soft_full` or `top_n_soft`;
`CKPT=base` overrides `CKPT_DIR`.

The plotter writes `training.png`, `training.pdf`, `metrics.csv` and `summary.json`.
It accepts partial/resumed logs and records missing batches.

## Appendix B: Soft Thinking versus random perturbations

Each runner evaluates Soft Thinking, matched random perturbations, greedy tokens
and sampled tokens. Settings are defined in `soft_thinking/mixture_protocol.py`.

| Benchmark | Llama/Qwen runner | Gemma runner |
|---|---|---|
| GSM8K | `eval_gsm8k_search.py` | `eval_gsm8k_search_gemma.py` |
| ARC-Challenge | `eval_arc_search.py` | `eval_arc_search_gemma.py` |
| WinoGrande | `eval_winogrande_search.py` | `eval_winogrande_search_gemma.py` |

Run all three benchmarks for Llama 3.2 3B Instruct, Llama 3.1 8B Instruct,
Qwen 3.5 9B and Gemma 2 9B IT. Use the corresponding Hugging Face model ID and
a separate output directory for each pair. The Gemma runners default to
`google/gemma-2-9b-it`. Example:

```bash
.venv/bin/python src/preliminary_experiments/soft_thinking/eval_gsm8k_search.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --output-dir results/preliminary_experiments/soft_thinking/gsm8k_llama8b
```

`--print-config` previews runner settings without loading a model. Runners write
text and question-level JSON. Use these outputs for the confidence intervals below.
Model revisions are not pinned here, so later downloads need not reproduce the
historical weights.

## Appendix C: confidence intervals for benchmark evaluation

Appendix C defines the Wilson and paired accuracy-difference intervals used in
Appendices A and B. For the Soft Thinking benchmark outputs from Appendix B, run:

```bash
.venv/bin/python src/preliminary_experiments/soft_thinking/compute_confidence_intervals.py \
  --inputs results/preliminary_experiments/soft_thinking/gsm8k_llama8b/*.json \
  --output-dir results/preliminary_experiments/soft_thinking/intervals
```

The command accepts one or more result files, validates counts/pairing, and writes
`results.json` plus `report.md` with Wilson accuracy intervals and paired
Soft-minus-Noisy differences. This is postprocessing of benchmark outputs; no
additional model run is required.
