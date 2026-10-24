"""Exact, streamed SentenceDot catalogue cross entropy.

The catalogue is recomputed from current embeddings on every update. Two
passes trade candidate compute for bounded accelerator memory: a no-grad
log-partition pass and an exact vector-Jacobian pass. No stale bank, sampled
negative, or detached candidate gradient enters the objective.
"""

from __future__ import annotations


import torch
from torch.nn import functional as F


def rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (
            angles.cos() * even - angles.sin() * odd,
            angles.sin() * even + angles.cos() * odd,
        ),
        -1,
    ).flatten(-2)


def encode_chunks(
    weight: torch.Tensor,
    alpha: torch.Tensor,
    tokens: torch.Tensor,
    offsets: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Position-rotated normalized sum, used for both input and catalogue readout."""
    counts = offsets[1:] - offsets[:-1]
    assert bool((counts > 0).all())
    group = torch.repeat_interleave(
        torch.arange(len(counts), device=tokens.device), counts
    )
    position = torch.arange(len(tokens), device=tokens.device) - offsets[group] + 1
    embedded = F.embedding(tokens, weight)
    vector = rotate(embedded, position[:, None] * alpha[None]).to(torch.float32)
    summed = torch.zeros(
        (len(counts), weight.shape[1]), device=weight.device, dtype=torch.float32
    )
    summed.index_add_(0, group, vector)
    return F.normalize(summed, dim=-1, eps=1e-8) * scale


def positive_vectors(
    bank,
    ids: torch.Tensor,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Encode positives once; duplicate IDs are harmless and preserve order."""
    return bank.encode_ids(ids.detach().cpu().tolist(), weight, alpha, scale)


def exact_catalogue_backward(
    h: torch.Tensor,
    targets: torch.Tensor,
    coefficients: torch.Tensor,
    bank,
    weight: torch.Tensor,
    alpha: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = 4096,
) -> dict:
    """Add exact CE gradients to h, weight, alpha and scale; return detached loss.

    ``coefficients`` specifies the optimization reduction. The training runner
    uses 1 / total_target_chunks_in_batch for every target chunk.
    ``bank.blocks`` yields disjoint contiguous catalogue ID ranges.
    """
    assert targets.ndim == coefficients.ndim == 1 and len(targets) == len(h)
    assert len(targets) > 0 and bool(((targets >= 0) & (targets < len(bank))).all())
    hd = h.detach().float()
    with torch.no_grad():
        logz = torch.full((len(h),), -torch.inf, device=h.device, dtype=torch.float32)
        for lo, hi, tokens, offsets in bank.blocks(block_size, h.device):
            vectors = encode_chunks(
                weight.detach(), alpha.detach(), tokens, offsets, scale.detach()
            )
            scores = hd @ vectors.T
            logz = torch.logaddexp(logz, torch.logsumexp(scores, dim=-1))
        pv = positive_vectors(
            bank, targets, weight.detach(), alpha.detach(), scale.detach()
        )
        positive = (hd * pv).sum(-1)
        losses = logz - positive
        if not bool(torch.isfinite(losses).all()):
            raise FloatingPointError("Nonfinite exact catalogue loss")

    dh = torch.zeros_like(hd)
    for lo, hi, tokens, offsets in bank.blocks(block_size, h.device):
        vectors = encode_chunks(weight, alpha, tokens, offsets, scale)
        with torch.no_grad():
            scores = hd @ vectors.detach().T
            delta = torch.exp(scores - logz[:, None])
            selected = (targets >= lo) & (targets < hi)
            rows = torch.nonzero(selected, as_tuple=True)[0]
            if len(rows):
                delta[rows, targets[rows] - lo] -= 1
            delta *= coefficients[:, None]
            dh += delta @ vectors.detach()
            dv = delta.T @ hd
        vectors.backward(dv)
    h.backward(dh.to(h.dtype))
    return {
        "loss": float((losses * coefficients).sum()),
        "raw_nll_sum": float(losses.sum()),
        "queries": len(targets),
    }
