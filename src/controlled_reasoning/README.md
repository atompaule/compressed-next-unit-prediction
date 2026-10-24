# Chapter 5: compositional next-unit prediction on a controlled inference task

Run from the repository root after the [shared setup](../../README.md#setup).
Full training requires Apple Silicon/MPS; data preparation and diagnostics support
CPU. Models train from scratch.

## 1. Prepare data

```bash
.venv/bin/python src/controlled_reasoning/create_data.py \
  --output data/controlled_reasoning/full --conclusions 3 6 \
  --epochs 120 --per-epoch 1024 --validation 512 --test 1024 \
  --seed 170917 --allow-repeats-after 0
.venv/bin/python src/controlled_reasoning/extend_data.py
```

This creates vocabulary, catalogues, held-out splits and epoch files for both task
variants. The extension adds six-inference epochs 121–240 under
`data/controlled_reasoning/extended`, preserving the evaluation split.

## 2. Train

| Condition | Prediction | Premises | Rotations |
|---|---|---|---|
| `discrete` | Token by token | Tokenized | — |
| `sentence` | CoNUP | Tokenized | Learned |
| `chunked_prompt` | CoNUP | Compressed | Learned |
| `nope` | CoNUP | Tokenized | None |
| `fixed_rates` | CoNUP | Tokenized | Fixed |
| `chunked_prompt_fixed_rates` | CoNUP | Compressed | Fixed |

All six conditions use two/four layers and seeds 101–105. Run the initial stage
before the extension; omit `--execute` to preview commands:

```bash
.venv/bin/python src/controlled_reasoning/run_campaign.py \
  --conditions discrete sentence nope chunked_prompt fixed_rates chunked_prompt_fixed_rates \
  --conclusions 3 6 --layers 2 4 --seeds 101 102 103 104 105 --execute
.venv/bin/python src/controlled_reasoning/run_campaign.py --stage extension \
  --conditions discrete sentence nope chunked_prompt fixed_rates chunked_prompt_fixed_rates \
  --conclusions 6 --layers 2 4 --seeds 101 102 103 104 105 --execute
```

Three-inference models train for 120 epochs; six-inference models continue to 240
with restored optimizer and batch RNG state. Fixed rotations retain the matched
random initialization. Existing run directories are protected.

Validation selects checkpoints; final evaluation includes likelihood, greedy
generation and two temperature-one samples per test world. Outputs are under
`checkpoints/controlled_reasoning/full/<run>/` and `extended_full/<run>/`:
`config.json`, `history.jsonl`, `best.pt`, `final_metrics.json`, `test_generations.json`.
Run names start with `sentence_3_seed101` or `discrete_3_seed101`; suffixes identify
`_4l`, `_nope`, `_fixed_rates` and `_chunked_prompt` where applicable.

## 3. Summarize results

```bash
.venv/bin/python -m src.common.summarize \
  --experiment controlled_reasoning \
  --input-root checkpoints/controlled_reasoning \
  --output-dir results/controlled_reasoning/summary
```

Outputs: `per_seed.csv`, `summary.csv`, `summary.json` and PNG/PDF learning curves.
Any completed subset works; incomplete runs are listed. Data, architecture,
source versions and training budgets remain separate. Use the **240-epoch** group
for final six-inference results. Metrics retain native units; accuracy is a
fraction, and sample SD is empty for one seed. Token and statement losses differ.

## 4. Decoding diagnostics

After the learned-rotation tokenized/compressed-premise runs finish, run:

```bash
.venv/bin/python src/controlled_reasoning/analysis/evaluate_no_repeat.py --device cpu --threads 1
.venv/bin/python src/controlled_reasoning/analysis/evaluate_no_repeat_sampled.py --workers 2 --threads 1
.venv/bin/python src/controlled_reasoning/analysis/verify_no_repeat_sampled.py
.venv/bin/python src/controlled_reasoning/analysis/evaluate_no_repeat_three.py --workers 2 --threads 1
.venv/bin/python src/controlled_reasoning/analysis/verify_no_repeat_three.py
```

These cover both premise modes, depths and all five seeds, preserving checkpoints
and sampling seeds. They verify standard outputs before masking repeated catalogue
entries. Protocols, generations and summaries go to
`results/controlled_reasoning/compressed_premises/{no_repeat,no_repeat_sampled,no_repeat_three}/`.
Fixed-rate and discrete/no-position models are outside this diagnostic grid.

Rescore the no-position model's saved outputs as token multisets:

```bash
.venv/bin/python src/controlled_reasoning/analysis/rescore_permutations.py \
  --run-dirs checkpoints/controlled_reasoning/full/sentence_3_seed101_nope \
  --output-dir results/controlled_reasoning/permutations_3_seed101
```

Pass additional run directories for other seeds, depths and six-inference
extensions. `results.json` contains adjusted greedy/sampled scores. Reserved-target
membership remains unchanged; matching a known permutation does not establish
novel-statement generation. Neither diagnostic retrains or reselects models.

## 5. Timing

After training, benchmark the discrete, tokenized CoNUP and no-position models on MPS:

```bash
.venv/bin/python src/controlled_reasoning/benchmark_steps.py
```

This writes `checkpoints/controlled_reasoning/step_efficiency.json` for 60 runs.
It measures optimizer updates and decoding steps, excluding prefill and catalogue
construction; require `complete: true` before using the report.
