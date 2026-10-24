import torch
import torch.nn.functional as F


def select_targets(
    logits: torch.Tensor,  # [V] or [B, L, V] — raw logits
    temperature: float,
    top_p: float,
    min_k: int = 1,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Select target tokens via top-p (nucleus) filtering:
      1. Apply temperature, then softmax
      2. Sort descending by probability
      3. Keep tokens until cumulative probability first reaches top_p (nucleus)
      4. Keep at least min_k tokens
      5. Renormalize selected probabilities to sum to 1

    Returns:
        Scalar (logits is [V]):
            target_probs: [k]           — renormalized probabilities, sorted descending
            target_ids:   [k]           — token indices of selected tokens

        Batched (logits is [B, L, V]):
            target_probs: [B, L, k_max] — renormalized probabilities, 0-padded
            target_ids:   [B, L, k_max] — token indices, -1-padded
            k_per_pos:    [B, L]        — number of valid entries per position
    """
    # Temperature-scaled probability distribution
    probs = F.softmax(logits / temperature, dim=-1)  # [V] or [B, L, V]

    # Sort descending
    sorted_probs, sorted_ids = torch.sort(probs, descending=True, dim=-1)

    # Top-p filtering
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cumsum_shifted = torch.roll(cumsum, 1, dims=-1)
    cumsum_shifted[..., 0] = 0.0
    mask = cumsum_shifted < top_p  # [V] or [B, L, V]

    if logits.dim() == 1:
        k = max(int(mask.sum().item()), min_k)

        target_ids = sorted_ids[:k]
        target_probs = sorted_probs[:k] / sorted_probs[:k].sum()

        return target_probs, target_ids
    else:
        k_per_pos = mask.sum(dim=-1).clamp(min=min_k)  # [B, L]
        k_max = int(k_per_pos.max().item())

        target_ids = sorted_ids[..., :k_max].clone()  # [B, L, k_max]
        target_probs = sorted_probs[..., :k_max].clone()  # [B, L, k_max]

        pos_idx = torch.arange(k_max, device=logits.device)  # [k_max]
        pad_mask = pos_idx.unsqueeze(0).unsqueeze(0) >= k_per_pos.unsqueeze(
            -1
        )  # [B, L, k_max]

        target_ids[pad_mask] = -1
        target_probs[pad_mask] = 0.0

        # Renormalize over valid entries; pads contribute 0 to the sum.
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)

        return target_probs, target_ids, k_per_pos


def emit(text, file_handle=None):
    print(text)
    if file_handle:
        file_handle.write(text + "\n")
