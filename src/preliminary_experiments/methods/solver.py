
# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.preliminary_experiments.methods.config import CFG
from src.preliminary_experiments.methods.utils import select_targets


def geometric_solver(
    logits: torch.Tensor,  # [V] — raw logits from LLM
    vocab_embs: torch.Tensor,  # [V, d] — token embedding matrix (unnormalized)
    use_cosine: bool = False,  # if True, use cosine similarity; if False, use dot product
) -> torch.Tensor:
    """
    Geometric solver that optimizes a latent embedding to satisfy four loss terms:
      1. Target ranking loss: preserve pairwise ordering of target similarities
      2. Target margin loss: enforce similarity gaps proportional to probability gaps (adjacent pairs)
      3. Interloper similarity loss: push interloper similarities below weakest target similarity
      4. Target similarity loss: pull embedding toward all target tokens

    All losses are averaged by their number of elements, making them invariant
    to the dynamic k from top-p selection.

    Args:
        logits:      [V] raw logits from the LLM at a given time step
        vocab_embs:  [V, d] token embedding matrix (unnormalized)
        use_cosine:  if True, compute cosine similarity (normalize both sides);
                     if False, compute raw dot products (unnormalized vocab embeddings,
                     latent embedding optimized in full vector space)

    Returns: [1, d] tensor — the optimized latent embedding scaled to target magnitude.
    """
    device = logits.device
    d = vocab_embs.size(1)

    vocab_embs_norm = F.normalize(vocab_embs, dim=1)  # [V, d]

    target_probs, target_ids = select_targets(logits, CFG.temperature, CFG.top_p)
    k = len(target_ids)

    target_embs = vocab_embs[target_ids]
    target_embs_norm = F.normalize(target_embs, dim=1)  # [k, d]

    target_magnitude = (target_probs * target_embs.norm(dim=1)).sum().item()

    vocab_embs_sim = vocab_embs_norm if use_cosine else vocab_embs
    target_embs_sim = target_embs_norm if use_cosine else target_embs

    with torch.no_grad():
        pairwise_mask = (
            target_probs.unsqueeze(1) > target_probs.unsqueeze(0)
        ).float()  # [k, k]
        n_pairs = pairwise_mask.sum().clamp(min=1)

        target_probs_margins = CFG.solver_config.margin_scale * (
            target_probs[:-1] - target_probs[1:]
        )  # [k-1]
        n_adjacent = max(k - 1, 1)

    with torch.no_grad():
        centroid_emb_norm = F.normalize(
            target_embs_norm.sum(dim=0, keepdim=True), dim=1
        )
        sims_to_centroid = (centroid_emb_norm @ vocab_embs_norm.T).squeeze(0)  # [V]

        interloper_mask = torch.ones(
            vocab_embs_norm.size(0), dtype=torch.bool, device=device
        )
        interloper_mask[target_ids] = False

        interloper_sims_masked = sims_to_centroid.clone()
        interloper_sims_masked[~interloper_mask] = -float("inf")
        n_interlopers = min(CFG.num_interlopers, interloper_mask.sum().item())
        _, interloper_ids = torch.topk(interloper_sims_masked, n_interlopers)
        interloper_embs_sim = vocab_embs_sim[interloper_ids]  # [n_interlopers, d]

    latent_emb = torch.nn.Parameter(F.normalize(torch.randn(d, device=device), dim=0))

    opt = optim.Adam([latent_emb], lr=CFG.solver_config.lr)

    for _ in range(CFG.solver_config.steps):
        opt.zero_grad()

        latent_emb_q = F.normalize(latent_emb.unsqueeze(0), dim=1)  # [1, d]

        target_sims = (latent_emb_q @ target_embs_sim.T).squeeze(0)  # [k]
        interloper_sims = (latent_emb_q @ interloper_embs_sim.T).squeeze(
            0
        )  # [n_interlopers]

        # Loss 1: Target ranking loss (averaged over ordered pairs)
        # L_rank = (1/n_pairs) * sum_{p_i > p_j} max(0, s_j - s_i)
        target_sims_diffs = target_sims.unsqueeze(1) - target_sims.unsqueeze(
            0
        )  # [k, k]
        ranking_violations = F.relu(-target_sims_diffs)
        loss_rank = (
            CFG.solver_config.ranking_weight
            * (pairwise_mask * ranking_violations).sum()
            / n_pairs
        )

        # Loss 2: Target margin loss (averaged over adjacent pairs)
        # L_margin = (1/(k-1)) * sum_i max(0, m_min + m_i - (s_i - s_{i+1}))
        target_sims_margins = target_sims[:-1] - target_sims[1:]  # [k-1]
        loss_margin = (
            CFG.solver_config.margin_weight
            * F.relu(
                CFG.solver_config.margin_min
                + target_probs_margins
                - target_sims_margins
            ).sum()
            / n_adjacent
        )

        # Loss 3: Interloper similarity loss (averaged over interlopers)
        # L_interloper = (1/n_interlopers) * sum_l max(0, m_l + s_l - s_k)
        loss_interloper = (
            CFG.solver_config.interloper_weight
            * F.relu(
                interloper_sims - CFG.solver_config.interloper_margin_threshold
            ).sum()
            / n_interlopers
        )

        # Loss 4: Target similarity loss (averaged over targets)
        # L_target = -(lambda_target / k) * sum_i s_i
        loss_target = -CFG.solver_config.target_sim_weight * target_sims.sum() / k

        # Total loss
        loss = loss_rank + loss_margin + loss_target + loss_interloper
        loss.backward()
        opt.step()

    # Scale to desired magnitude only at the end
    with torch.no_grad():
        result = F.normalize(latent_emb.unsqueeze(0), dim=1) * target_magnitude

    return result
