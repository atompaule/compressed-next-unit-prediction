"""Fresh-candidate, inclusion-corrected approximation to catalogue CE.

All distinct batch targets A enter exactly once. K entries are drawn uniformly
    without replacement from C minus A, independently on each update. Their partition
contributions receive weight (|C|-|A|)/K. Conditional on the batch, Z_hat is
unbiased; log(Z_hat), the loss and its gradient generally are NOT unbiased.
Increasing K to the complement size recovers the existing exact objective.

The candidate embeddings, rotations and norm scale remain fully trainable.
The training runner uses this sampled objective.
"""

import math

import numpy as np
import torch

from objective import encode_chunks, exact_catalogue_backward
from protocol import NEGATIVE_CANDIDATES


def draw_complement(size, included, count, rng):
    """Uniform distinct IDs outside included, via ranks in its complement.

    No full complement array is built here. NumPy's choice implementation may
    itself allocate O(size) workspace for sampling without replacement.
    """
    included = np.unique(np.asarray(included, dtype=np.int64))
    if len(included) and (included[0] < 0 or included[-1] >= size):
        raise ValueError("Included candidate ID outside catalogue")
    remaining = size - len(included)
    if count < 0 or count > remaining:
        raise ValueError("Invalid complement sample size")
    ranks = rng.choice(remaining, count, replace=False)
    return ranks + np.searchsorted(
        included - np.arange(len(included)), ranks, side="right"
    )


class SelectedBank:
    """CPU token sequences only; never caches parameter-dependent vectors."""

    def __init__(self, bank, ids):
        sequences = [bank.sequence(int(i)) for i in ids]
        self.offsets = np.r_[0, np.cumsum([len(s) for s in sequences])]
        self.tokens = np.concatenate(sequences).astype(np.int64)

    def __len__(self):
        return len(self.offsets) - 1

    def blocks(self, size, device):
        for lo in range(0, len(self), size):
            hi = min(lo + size, len(self))
            start, stop = self.offsets[lo], self.offsets[hi]
            tokens = torch.from_numpy(self.tokens[start:stop]).to(device)
            offsets = torch.from_numpy(self.offsets[lo : hi + 1] - start).to(device)
            yield lo, hi, tokens, offsets


def sampled_catalogue_backward(
    h,
    targets,
    coefficients,
    bank,
    weight,
    alpha,
    scale,
    rng,
    negatives=NEGATIVE_CANDIDATES,
    block_size=4096,
):
    """Two streamed passes, including exact gradients through sampled vectors."""
    if negatives <= 0:
        raise ValueError("At least one randomly sampled negative is required")
    global_targets = targets.detach().cpu().numpy()
    included = np.unique(global_targets)
    remaining = len(bank) - len(included)
    if negatives >= remaining:
        result = exact_catalogue_backward(
            h, targets, coefficients, bank, weight, alpha, scale, block_size
        )
        return result | {"candidates": len(bank), "partition_is_exact": True}
    sampled = draw_complement(len(bank), included, negatives, rng)
    selected = SelectedBank(bank, np.r_[included, sampled])
    local_targets = torch.as_tensor(
        np.searchsorted(included, global_targets), device=h.device
    )
    log_weight = torch.zeros(len(selected), device=h.device, dtype=torch.float32)
    log_weight[len(included) :] = math.log(remaining / negatives)
    hd = h.detach().float()
    with torch.no_grad():
        logz = torch.full((len(h),), -torch.inf, device=h.device)
        positive = torch.empty(len(h), device=h.device)
        for lo, hi, tokens, offsets in selected.blocks(block_size, h.device):
            vectors = encode_chunks(
                weight.detach(), alpha.detach(), tokens, offsets, scale.detach()
            )
            raw = hd @ vectors.T
            logz = torch.logaddexp(
                logz, torch.logsumexp(raw + log_weight[None, lo:hi], dim=-1)
            )
            rows = torch.where((local_targets >= lo) & (local_targets < hi))[0]
            positive[rows] = raw[rows, local_targets[rows] - lo]
        losses = logz - positive
        if not torch.isfinite(losses).all():
            raise FloatingPointError("Nonfinite sampled catalogue loss")
    dh = torch.zeros_like(hd)
    for lo, hi, tokens, offsets in selected.blocks(block_size, h.device):
        vectors = encode_chunks(weight, alpha, tokens, offsets, scale)
        with torch.no_grad():
            raw = hd @ vectors.detach().T
            delta = torch.exp(raw + log_weight[None, lo:hi] - logz[:, None])
            rows = torch.where((local_targets >= lo) & (local_targets < hi))[0]
            delta[rows, local_targets[rows] - lo] -= 1
            delta *= coefficients[:, None]
            dh += delta @ vectors.detach()
            dv = delta.T @ hd
        vectors.backward(dv)
    h.backward(dh.to(h.dtype))
    return {
        "loss": float((losses * coefficients).sum()),
        "raw_nll_sum": float(losses.sum()),
        "queries": len(targets),
        "candidates": len(selected),
        "partition_is_exact": False,
    }
