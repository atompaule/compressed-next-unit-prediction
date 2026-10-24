"""Fixed-checkpoint generation helpers; no training-catalogue recomposition cache."""

import hashlib
import json
from collections import defaultdict
import torch
from model import mean_embedding_norm

SEED = 314159


def announce(stage, **details):
    print(json.dumps(dict(stage=stage, **details)), flush=True)


def select_windows(metadata, count=50, seed=SEED):
    """Round-robin source quotas; selection never sees predictions or references."""
    groups = defaultdict(list)
    for i, m in enumerate(metadata):
        key = hashlib.sha256(f"{seed}:{m['window_id']}".encode()).hexdigest()
        groups[m["source"]].append((key, i))
    for values in groups.values():
        values.sort(reverse=True)
    selected, documents = [], set()
    while len(selected) < count:
        before = len(selected)
        for source in sorted(groups):
            while groups[source]:
                _, i = groups[source].pop()
                if metadata[i]["document_id"] not in documents:
                    selected.append(i)
                    documents.add(metadata[i]["document_id"])
                    break
            if len(selected) == count:
                break
        if len(selected) == before:
            raise ValueError(
                "Insufficient distinct documents for qualitative selection"
            )
    return selected


def rollout_summary(tokens, prompt_tokens):
    grams = [tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3))]
    prompt = {
        tuple(prompt_tokens[i : i + 4]) for i in range(max(0, len(prompt_tokens) - 3))
    }
    return {
        "token_count": len(tokens),
        "repeated_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else None,
        "prompt_4gram_copy_fraction": sum(g in prompt for g in grams) / len(grams)
        if grams
        else None,
    }


@torch.inference_mode()
def discrete_rollout(model, prompt, mode, seed, budget=64, eos=50256):
    device = model.embed_tokens.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    output, cache = [], None
    next_input = torch.tensor([prompt], device=device)
    capacity = model.backbone.config.n_positions
    reason = "token_budget"
    for _ in range(budget):
        if len(prompt) + len(output) > capacity:
            reason = "context_capacity"
            break
        result = model(input_ids=next_input, past_key_values=cache, use_cache=True)
        logits = result.last_hidden_state[0, -1].float() @ model.embed_tokens.weight.T
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite token logits")
        tok = (
            int(logits.argmax())
            if mode == "greedy"
            else int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
        )
        if tok == eos:
            reason = "eos"
            break
        output.append(tok)
        next_input = torch.tensor([[tok]], device=device)
        cache = result.past_key_values
    return {
        "token_ids": output,
        "stop_reason": reason,
        **rollout_summary(output, prompt),
    }


@torch.inference_mode()
def chunk_hidden(model, bank, contexts, alpha, scale):
    vectors = [
        bank.encode_ids(c, model.embed_tokens.weight, alpha, scale) for c in contexts
    ]
    width = max(map(len, vectors))
    if width > model.backbone.config.n_positions:
        raise ValueError("Chunk context exceeds capacity; no silent truncation")
    inputs = vectors[0].new_zeros(len(vectors), width, vectors[0].shape[-1])
    mask = torch.zeros(len(vectors), width, dtype=torch.long, device=inputs.device)
    for i, vector in enumerate(vectors):
        inputs[i, : len(vector)] = vector
        mask[i, : len(vector)] = 1
    hidden = model(
        inputs_embeds=inputs, attention_mask=mask, use_cache=False
    ).last_hidden_state
    return hidden[
        torch.arange(len(vectors), device=inputs.device), mask.sum(-1) - 1
    ].float()


@torch.inference_mode()
def select_catalogue(hidden, vectors, limit, mode, generator, block=65536):
    """Exact greedy or Gumbel-max draw from full categorical softmax; no shortlist."""
    best = torch.full((len(hidden),), -torch.inf, device=hidden.device)
    selected = torch.zeros(len(hidden), dtype=torch.long, device=hidden.device)
    for lo in range(0, limit, block):
        part = vectors[lo : min(limit, lo + block)]
        if not isinstance(part, torch.Tensor):
            part = torch.from_numpy(part.copy())
        scores = hidden @ part.to(hidden.device).T
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Nonfinite catalogue logits")
        if mode == "sample":
            # -log(Exp(1)) is Gumbel(0,1); maximization samples softmax(scores).
            noise = torch.empty_like(scores).exponential_(generator=generator)
            scores = scores - noise.clamp_min(torch.finfo(scores.dtype).tiny).log()
        values, ids = scores.max(-1)
        better = values > best
        selected = torch.where(better, ids + lo, selected)
        best = torch.maximum(best, values)
    return selected.tolist()


@torch.inference_mode()
def sentence_rollouts(
    model, alpha, bank, vectors, contexts, limit, mode, seed, budget=64, block=65536
):
    device = model.embed_tokens.weight.device
    generator = torch.Generator(device=device).manual_seed(seed)
    lexical = torch.arange(model.embed_tokens.num_embeddings - 1, device=device)
    scale = mean_embedding_norm(model.embed_tokens.weight, lexical)
    live = [list(c) for c in contexts]
    outputs = [
        {"chunk_ids": [], "token_ids": [], "stop_reason": "token_budget"}
        for _ in contexts
    ]
    for step in range(budget):
        active = [
            i
            for i, row in enumerate(outputs)
            if len(row["token_ids"]) < budget
            and len(live[i]) < model.backbone.config.n_positions
        ]
        if not active:
            break
        hidden = chunk_hidden(model, bank, [live[i] for i in active], alpha, scale)
        chosen = select_catalogue(hidden, vectors, limit, mode, generator, block)
        for i, c in zip(active, chosen):
            outputs[i]["chunk_ids"].append(c)
            outputs[i]["token_ids"].extend(bank.sequence(c))
            live[i].append(c)
        if step % 8 == 0:
            announce(
                "rollout", mode=mode, catalogue=limit, step=step, active=len(active)
            )
    for row, context in zip(outputs, contexts):
        if len(row["token_ids"]) < budget:
            row["stop_reason"] = "context_capacity_or_chunk_limit"
        prompt = [t for c in context for t in bank.sequence(c)]
        row.update(rollout_summary(row["token_ids"], prompt))
        row["repeated_chunk_fraction"] = 1 - len(set(row["chunk_ids"])) / max(
            1, len(row["chunk_ids"])
        )
        row["prompt_chunk_copy_fraction"] = sum(
            c in context for c in row["chunk_ids"]
        ) / max(1, len(row["chunk_ids"]))
    return outputs
