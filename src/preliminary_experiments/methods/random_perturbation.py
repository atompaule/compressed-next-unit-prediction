"""Matched random angular perturbation used in the Soft Thinking comparison."""

import torch
import torch.nn.functional as F


def noisy_target_sim(
    e: torch.Tensor,
    target_sim: float = 0.90,
    reference: torch.Tensor | None = None,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Perturb a token embedding with random noise to achieve target_sim cosine similarity.

    By default, similarity is measured against e itself (i.e. the result is a
    noisy version of e at a fixed angular distance). If reference is provided,
    the result is instead slerped to have cosine similarity target_sim to
    reference, while preserving the magnitude of e.

    Args:
        e:           Input embedding [d], need not be unit norm.
        target_sim:  Desired cosine similarity to the reference vector.
        reference:   Reference vector [d] to target similarity against.
                     Defaults to e itself if None.
        tol:         Tolerance below which target_sim is treated as 1.0.

    Returns:
        Vector [d] with cosine similarity ~target_sim to reference, magnitude ||e||.
    """
    mag = e.norm(p=2).clamp_min(1e-8)
    e_hat = e / mag  # [d], unit norm

    ref_hat = (
        F.normalize(reference.unsqueeze(0), dim=1).squeeze(0)
        if reference is not None
        else e_hat
    )

    if target_sim >= 1.0 - tol:
        return mag * ref_hat

    noise = torch.randn_like(ref_hat)
    noise = noise - (noise @ ref_hat) * ref_hat
    noise = F.normalize(noise.unsqueeze(0), dim=1).squeeze(
        0
    )  # unit norm, perp to ref_hat

    theta = torch.arccos(torch.tensor(target_sim, dtype=e.dtype, device=e.device))
    result = torch.cos(theta) * ref_hat + torch.sin(theta) * noise

    return mag * F.normalize(result.unsqueeze(0), dim=1).squeeze(0)
