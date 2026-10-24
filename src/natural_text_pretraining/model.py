"""Matched token and chunk prediction with a shared GPT-2 backbone."""

from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.nn import functional as F

TRAINING_REDUCTION = "mean_over_native_target_positions_v1"


def mean_embedding_norm(weight: torch.Tensor, lexical: torch.Tensor) -> torch.Tensor:
    return weight[lexical].float().norm(dim=-1).mean()


def precision_context(device: str, use_bf16: bool):
    if use_bf16 and device in ("mps", "cuda"):
        return torch.autocast(device, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def queries(
    model,
    alpha,
    bank,
    rows,
    scale,
    kind: str,
    use_bf16: bool,
    input_dropout: float = 0.0,
    sentence_prompt_mode: str = "chunk",
):
    """Teacher-forced target positions; one update consumes the same windows.

    Average across all valid target positions in the batch: chunks for
    SentenceDot, content tokens for discrete. Prompt and padding positions
    are excluded. No additional per-window or underlying-token weighting.
    """
    device = model.embed_tokens.weight.device
    sequences, positions, targets = [], [], []
    for context_ids, target_ids in rows:
        assert context_ids and target_ids
        if kind == "sentence":
            if sentence_prompt_mode == "chunk":
                prefix = bank.encode_ids(
                    context_ids, model.embed_tokens.weight, alpha, scale
                )
            elif sentence_prompt_mode == "token":
                prompt = [t for i in context_ids for t in bank.sequence(i)]
                prefix = model.embed_tokens(torch.tensor(prompt, device=device))
            else:
                raise ValueError(
                    f"Unknown SentenceDot prompt mode: {sentence_prompt_mode}"
                )
            earlier = (
                bank.encode_ids(
                    target_ids[:-1], model.embed_tokens.weight, alpha, scale
                )
                if len(target_ids) > 1
                else prefix[:0]
            )
            sequences.append(torch.cat((prefix, earlier), dim=0))
            positions.append(
                list(range(len(prefix) - 1, len(prefix) + len(target_ids) - 1))
            )
            targets.extend(target_ids)
        elif kind == "discrete":
            prompt = [t for i in context_ids for t in bank.sequence(i)]
            goal = [t for i in target_ids for t in bank.sequence(i)]
            sequences.append(torch.tensor((prompt + goal)[:-1], device=device))
            positions.append(list(range(len(prompt) - 1, len(prompt) + len(goal) - 1)))
            targets.extend(goal)
        else:
            raise ValueError(kind)
    lengths = [len(s) for s in sequences]
    x = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    mask = (
        torch.arange(x.shape[1], device=device)[None]
        < torch.tensor(lengths, device=device)[:, None]
    )
    if kind == "discrete":
        x = model.embed_tokens(x)
    if input_dropout and model.training:
        x = F.dropout(x, p=input_dropout, training=True)
    with precision_context(device.type, use_bf16):
        out = model(
            inputs_embeds=x, attention_mask=mask, use_cache=False
        ).last_hidden_state
    batch_ids = torch.tensor(
        [i for i, pos in enumerate(positions) for _ in pos], device=device
    )
    query_positions = torch.tensor([j for pos in positions for j in pos], device=device)
    h = out[batch_ids, query_positions]
    coefficients = torch.full(
        (len(targets),), 1 / len(targets), device=device, dtype=torch.float32
    )
    return h, torch.tensor(targets, device=device, dtype=torch.long), coefficients


def batched_queries(
    model,
    alpha,
    bank,
    rows,
    scale,
    kind,
    use_bf16,
    input_dropout=0.0,
    sentence_prompt_mode="chunk",
    microbatch_windows=8,
):
    """Checkpoint backbone microbatches, then share one readout across the batch.

    Non-reentrant checkpointing preserves CPU/CUDA RNG on replay. MPS requires
    an explicit RNG context: its module has no CUDA-style _initialized/device
    interface that checkpoint's automatic device-state capture relies on.
    Coefficients are normalized globally, never per microbatch.
    No optimizer step or catalogue traversal occurs inside this function.
    """
    from torch.utils.checkpoint import checkpoint

    if microbatch_windows <= 0:
        raise ValueError("microbatch_windows must be positive")
    hidden, targets = [], []
    device = model.embed_tokens.weight.device

    @contextlib.contextmanager
    def restore_mps_rng(state):
        later_state = torch.mps.get_rng_state()
        torch.mps.set_rng_state(state)
        try:
            yield
        finally:
            torch.mps.set_rng_state(later_state)

    for lo in range(0, len(rows), microbatch_windows):
        part = rows[lo : lo + microbatch_windows]

        def forward(_device_anchor, part=part):
            h, t, _ = queries(
                model,
                alpha,
                bank,
                part,
                scale,
                kind,
                use_bf16,
                input_dropout,
                sentence_prompt_mode,
            )
            return h, t

        options = {}
        if device.type == "mps":
            state = torch.mps.get_rng_state()

            def context_fn(state=state):
                return contextlib.nullcontext(), restore_mps_rng(state)

            options["context_fn"] = context_fn
        # Explicit tensor argument lets CUDA checkpointing find the device RNG;
        # tensors captured only in the closure are invisible to that mechanism.
        h, t = checkpoint(
            forward,
            model.embed_tokens.weight,
            use_reentrant=False,
            preserve_rng_state=True,
            **options,
        )
        hidden.append(h)
        targets.append(t)
    h, t = torch.cat(hidden), torch.cat(targets)
    coefficients = torch.full(
        (len(t),), 1 / len(t), device=h.device, dtype=torch.float32
    )
    return h, t, coefficients


def token_objective(h, targets, coefficients, weight, block: int = 128):
    # Release each vocabulary-logit block immediately; retain only dL/dh.
    proxy = h.detach().requires_grad_()
    loss, nll = 0.0, 0.0
    for lo in range(0, len(h), block):
        hi = min(lo + block, len(h))
        logits = proxy[lo:hi].float() @ weight.T
        values = F.cross_entropy(logits, targets[lo:hi], reduction="none")
        total = (values * coefficients[lo:hi]).sum()
        loss += float(total.detach())
        nll += float(values.detach().sum())
        total.backward()
    h.backward(proxy.grad)
    return {"loss": loss, "raw_nll_sum": nll, "queries": len(targets)}
