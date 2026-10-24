
# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import torch

from src.preliminary_experiments.methods.utils import select_targets


def soft_thinking(
    logits: torch.Tensor,  # [V] — raw logits from LLM
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    Soft thinking: probability-weighted sum of token embeddings over the top-p nucleus.

    Selects tokens via top-p, computes softmax(logits/temperature) over the selected
    subset, and returns the weighted sum of their embeddings.

    Args:
        logits:      [V] raw logits from the LLM
        vocab_embs:  [V, d] token embedding matrix (unnormalized)
        temperature: softmax temperature applied before top-p selection
        top_p:       nucleus threshold; 1.0 keeps the full distribution

    Returns: [1, d] tensor — the soft thinking embedding.
    """
    # Select tokens via top-p
    target_probs, target_ids = select_targets(logits, temperature, top_p)
    # if top_p == 1.0:
    #     assert len(target_probs) == len(logits)

    # Weighted sum of embeddings
    target_embs = vocab_embs[target_ids]  # [k, d]
    result = (target_probs.unsqueeze(1) * target_embs).sum(
        dim=0, keepdim=True
    )  # [1, d]

    return result
