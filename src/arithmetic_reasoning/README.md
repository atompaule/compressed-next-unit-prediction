# Chapter 4: arithmetic state tracking via positional compression

Run from the repository root after the [shared setup](../../README.md#setup).
Models train from scratch; `--device cpu`, `mps` and `cuda` are supported.

## 1. Three-register comparison

```bash
.venv/bin/python -m src.arithmetic_reasoning.generate_data \
  --num-registers 3 --output-dir data/arithmetic_reasoning/register \
  --train-samples 65536 --validation-samples 4096 --test-samples 4096
.venv/bin/python -m src.arithmetic_reasoning.train \
  --data-dir data/arithmetic_reasoning/register \
  --output-root checkpoints/arithmetic_reasoning/baseline \
  --condition all --d-model 96 --layers 3 --heads 4 --epochs 40 \
  --seeds 101 102 103 104 105
```

Keep the generator defaults, including its data seed. It writes three splits,
`manifest.json` with configuration/hashes, and `example.md`.

| Condition | Feedback |
|---|---|
| `continuous` | Composed values with positional binding |
| `discrete_conventional` | Separate tokens |
| `continuous_unbound` | Composition without positional binding |
| `cache_only_full_head` | Constant token |
| `continuous_cot2` | CoT2-style distributions |

`all` runs all five. Validation state loss selects checkpoints; final validation
and test evaluation use free-running and gold feedback. Each
`baseline/seed_<seed>/<condition>/` contains `best.pt`, `history.jsonl` and
`final_metrics.json`; seed directories also contain `manifest.json` and
`comparison.json`. Accuracy evaluation is part of training.

## 2. Five- and seven-register comparisons

```bash
for width in 5 7; do
  .venv/bin/python -m src.arithmetic_reasoning.generate_data \
    --num-registers "$width" \
    --output-dir "data/arithmetic_reasoning/width_sweep/registers_$width" \
    --train-samples 65536 --validation-samples 4096 --test-samples 4096
done

for width in 5 7; do
  if [ "$width" = 5 ]; then epochs=80; else epochs=160; fi
  for condition in continuous discrete_conventional; do
    .venv/bin/python -m src.arithmetic_reasoning.train \
      --data-dir "data/arithmetic_reasoning/width_sweep/registers_$width" \
      --output-root "checkpoints/arithmetic_reasoning/capacity/registers_$width" \
      --num-registers "$width" --condition "$condition" \
      --d-model 192 --layers 3 --heads 4 --epochs "$epochs" \
      --seeds 101 102 103 104 105 \
      --rollout-eval-every 5 --rollout-eval-samples 256
  done
done
```

Intermediate rollout probes are diagnostic; they do not select checkpoints.
The three-register comparison leaves them disabled.

## 3. Non-orthogonal control

For each register width, repeat the continuous condition with
`--variant nonorthogonal`, the same data, architecture, budget, probes and seeds,
and a separate output root. For three registers:

```bash
.venv/bin/python -m src.arithmetic_reasoning.train \
  --variant nonorthogonal --condition continuous \
  --data-dir data/arithmetic_reasoning/register \
  --output-root checkpoints/arithmetic_reasoning/nonorthogonal/registers_3 \
  --d-model 96 --layers 3 --heads 4 --epochs 40 \
  --seeds 101 102 103 104 105
```

Use `nonorthogonal/registers_5` and `nonorthogonal/registers_7` for the wider controls.

## 4. Summarize and benchmark

```bash
.venv/bin/python -m src.common.summarize \
  --experiment arithmetic_reasoning \
  --input-root checkpoints/arithmetic_reasoning \
  --output-dir results/arithmetic_reasoning/summary
```

This reads completed runs and writes `per_seed.csv`, `summary.csv`, `summary.json`
and PNG/PDF learning curves. Data, model settings, source versions and budgets
remain separate; incomplete runs are listed. Accuracy remains a fraction, and
sample SD is empty for one seed. Use a narrower input root to report one comparison.

Metrics include free-running/gold-feedback state accuracy by step and query
accuracy. `query_value_in_final_state_accuracy` ignores register assignment;
`mean_register_query_accuracy` requires the correct register.

For latency, benchmark each seed directory separately:

```bash
.venv/bin/python -m src.arithmetic_reasoning.benchmark \
  checkpoints/arithmetic_reasoning/baseline/seed_101
```

`latency_comparison.json` records timings and hardware. Repeat for the other seeds
and comparisons on the same hardware; `--conditions` restricts the models.
