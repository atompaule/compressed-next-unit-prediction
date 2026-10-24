"""Fixed mixture settings for the preliminary Soft Thinking/noise benchmarks.

The April 2026 benchmark used temperature 0.6 and top-p 0.95 (see commit
eba154b). Later changes to the general-purpose composition helper must not
silently change that reference distribution in the benchmark runners.
"""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from src.preliminary_experiments.methods.soft_thinking import soft_thinking


def benchmark_soft_thinking(logits, vocab_embs):
    return soft_thinking(logits, vocab_embs, temperature=0.6, top_p=0.95)


def thesis_approach_configs():
    """The four conditions reported in Appendix B, in their original order."""
    return [
        {"name": "soft_thinking", "sampling_temp": 0.0, "top_p": 0.95},
        {"name": "noisy_discrete", "sampling_temp": 0.0, "top_p": 0.95},
        {"name": "discrete_top1", "sampling_temp": 0.0, "top_p": 0.95},
        {"name": "default_baseline", "sampling_temp": 0.6, "top_p": 0.90},
    ]
