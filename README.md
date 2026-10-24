# Compositional Next-Unit Prediction

Experiments for the master's thesis on compressed continuous representations in
language models. The guides cover inputs, training, evaluation and results for the
corresponding chapters and appendices.

## Setup

Run commands from the repository root. Chapters 3–5 and Appendices A–C use Python
3.11 and the shared [uv](https://docs.astral.sh/uv/) environment:

```bash
uv sync --locked --python 3.11
```

Chapter 6 has a separate Linux/CUDA environment. Pretrained-model experiments
require downloaded weights and, for gated Llama/Gemma models, Hugging Face access.

## Reproduce a chapter or appendix

| Thesis location | Guide | Hardware |
|---|---|---|
| Chapter 3 | [Constructed vocabulary mixtures](src/preliminary_experiments/README.md#chapter-3-constructed-vocabulary-mixtures) | Apple Silicon/MPS for descriptions and convergence |
| Chapter 4 | [Arithmetic state tracking](src/arithmetic_reasoning/README.md) | CPU, MPS or CUDA |
| Chapter 5 | [Controlled inference with CoNUP](src/controlled_reasoning/README.md) | Apple Silicon/MPS for full training |
| Chapter 6 | [Natural-text pretraining](src/natural_text_pretraining/README.md) | Linux/CUDA, Slurm and A100 40 GB |
| Appendix A | [Reinforcement learning](src/preliminary_experiments/README.md#appendix-a-reinforcement-learning) | Accelerator memory for Llama 3B/8B |
| Appendix B | [Soft Thinking versus random perturbations](src/preliminary_experiments/README.md#appendix-b-soft-thinking-versus-random-perturbations) | Sufficient accelerator memory for the chosen model |
| Appendix C | [Confidence intervals for benchmark evaluation](src/preliminary_experiments/README.md#appendix-c-confidence-intervals-for-benchmark-evaluation) | CPU; benchmark outputs required |

Experiment groups run independently. Follow each guide for the applicable
preparation, training, evaluation and analysis steps.
Use the documented budgets and seeds for the full experiments. Small runs validate
the pipelines; a complete rerun of every experiment is not verified. Numerical
results can vary with hardware, model revisions and random sampling.

## Files and outputs

| Directory | Contents | Default-root override |
|---|---|---|
| `src/` | Experiment code and guides | — |
| `data/` | Generated datasets and downloads | `THESIS_DATA_ROOT` |
| `checkpoints/` | Weights, configurations and training records | `THESIS_EXPERIMENT_ROOT` |
| `results/` | Analyses, tables and figures | `THESIS_RESULTS_ROOT` |

Explicit path arguments take precedence over these defaults. Chapter 6 uses the
campaign paths in its guide. Use fresh output directories unless resuming a run.
Generated files, development tests and local archives are excluded from Git;
new experiments require no existing results or sibling thesis-document repository.
