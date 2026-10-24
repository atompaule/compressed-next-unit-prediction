"""Fixed-bank validation and shared-catalogue candidate scoring."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from data import PackedBank, Windows, load_manifest
from model import mean_embedding_norm, queries
from objective import encode_chunks

CATALOGUE_SCORE_DEFINITION = "token_prefix_product_no_end_v1"
MONITOR_SELECTION = "source_balanced_window_hash_v2_full_default"


def monitor_selection(metadata: Path, limit: int):
    """Fixed source-stratified sample, independent of training seed and row order."""
    if limit < 0:
        raise ValueError(
            "monitor_windows must be nonnegative; zero selects all windows"
        )
    groups = defaultdict(list)
    identities = set()
    for i, line in enumerate(metadata.open()):
        item = json.loads(line)
        identity = item["window_id"]
        if identity in identities:
            raise ValueError("Duplicate validation window identity")
        identities.add(identity)
        rank = hashlib.sha256(("monitor-v1:" + identity).encode()).hexdigest()
        groups[item["source"]].append((rank, identity, i))
    for entries in groups.values():
        entries.sort()
    total = len(identities)
    if limit == 0:
        limit = total
    selected = []
    for position in range(max((len(v) for v in groups.values()), default=0)):
        for source in sorted(groups):
            if position < len(groups[source]) and len(selected) < limit:
                _, identity, i = groups[source][position]
                selected.append((i, identity, source))
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError("Empty validation sample")
    ids = [identity for _, identity, _ in selected]
    return [i for i, _, _ in selected], {
        "policy": MONITOR_SELECTION,
        "windows": len(selected),
        "scope": "full" if len(selected) == total else "subset",
        "sources": dict(Counter(source for _, _, source in selected)),
        "ordered_window_ids_sha256": hashlib.sha256(
            "\n".join(ids).encode()
        ).hexdigest(),
    }


@torch.no_grad()
def sentence_validation(
    model,
    alpha,
    lexical,
    bank,
    train_count,
    rows,
    block_size=4096,
    batch_windows=4,
    sentence_prompt_mode="chunk",
    query_block_size=1024,
):
    """Exact validation; encode each catalogue block once per frozen checkpoint.

    Query states are small enough to retain (roughly 55 MiB for 28k x 512).
    Scores are tiled over queries and candidates, so neither the entire score
    matrix nor the entire catalogue embedding matrix needs to stay in memory.
    Nothing is cached across parameter updates.
    """
    if not rows or min(block_size, batch_windows, query_block_size) <= 0:
        raise ValueError("Nonempty rows and positive validation block sizes required")
    model.eval()
    weight = model.embed_tokens.weight
    scale = mean_embedding_norm(weight, lexical)
    totals = {"all": [0.0, 0], "unseen": [0.0, 0], "seen": [0.0, 0]}
    top1 = {"all": 0, "unseen": 0, "seen": 0}
    copies = 0
    hidden, labels = [], []
    for first in range(0, len(rows), batch_windows):
        subset = rows[first : first + batch_windows]
        h, targets, _ = queries(
            model,
            alpha,
            bank,
            subset,
            scale,
            "sentence",
            False,
            sentence_prompt_mode=sentence_prompt_mode,
        )
        hidden.append(h.float())
        labels.append(targets)
    h, targets = torch.cat(hidden), torch.cat(labels)
    del hidden, labels
    logz = torch.full((len(h),), -torch.inf, device=h.device)
    best = torch.full((len(h),), -torch.inf, device=h.device)
    best_id = torch.zeros(len(h), dtype=torch.long, device=h.device)
    for lo, hi, tokens, offsets in bank.blocks(block_size, h.device):
        vectors = encode_chunks(weight, alpha, tokens, offsets, scale)
        for first in range(0, len(h), query_block_size):
            last = min(first + query_block_size, len(h))
            score = h[first:last] @ vectors.T
            logz[first:last] = torch.logaddexp(logz[first:last], score.logsumexp(-1))
            value, ix = score.max(-1)
            better = value > best[first:last]
            best[first:last] = torch.maximum(best[first:last], value)
            best_id[first:last] = torch.where(better, ix + lo, best_id[first:last])
    losses = []
    for first in range(0, len(h), query_block_size):
        last = min(first + query_block_size, len(h))
        positive = (
            h[first:last]
            * bank.encode_ids(targets[first:last].cpu().tolist(), weight, alpha, scale)
        ).sum(-1)
        losses.extend((logz[first:last] - positive).cpu().tolist())
    selected = best_id.cpu().tolist()
    gold = targets.cpu().tolist()
    contexts = [set(c) for c, t in rows for _ in t]
    for loss, pred, target, context in zip(losses, selected, gold, contexts):
        group = "unseen" if target >= train_count else "seen"
        for key in ("all", group):
            totals[key][0] += loss
            totals[key][1] += 1
            top1[key] += int(pred == target)
        copies += int(pred in context)
    return {
        "nll": {k: v[0] / v[1] if v[1] else None for k, v in totals.items()},
        "queries": {k: v[1] for k, v in totals.items()},
        "top1": {k: top1[k] / totals[k][1] if totals[k][1] else None for k in totals},
        "context_copy_rate": copies / totals["all"][1],
        "catalogue_size": len(bank),
    }


@torch.no_grad()
def token_validation(model, bank, rows, batch_windows=4, train_count=None):
    model.eval()
    weight = model.embed_tokens.weight
    totals = {key: [0.0, 0, 0] for key in ("all", "unseen", "seen")}
    for first in range(0, len(rows), batch_windows):
        subset = rows[first : first + batch_windows]
        h, targets, _ = queries(model, None, bank, subset, None, "discrete", False)
        losses = []
        for lo in range(0, len(h), 128):
            hi = min(lo + 128, len(h))
            losses.extend(
                F.cross_entropy(
                    h[lo:hi].float() @ weight.T, targets[lo:hi], reduction="none"
                )
                .cpu()
                .double()
                .tolist()
            )
        cursor = 0
        for _, chunks in subset:
            for chunk in chunks:
                length = len(bank.sequence(chunk))
                loss = math.fsum(losses[cursor : cursor + length])
                cursor += length
                keys = ["all"]
                if train_count is not None:
                    keys.append("unseen" if chunk >= train_count else "seen")
                for key in keys:
                    totals[key][0] += loss
                    totals[key][1] += length
                    totals[key][2] += 1
        assert cursor == len(losses)
    return {
        "nll_per_token": totals["all"][0] / totals["all"][1],
        "tokens": totals["all"][1],
        "nll_per_token_by_group": {
            k: v[0] / v[1] if v[1] else None for k, v in totals.items()
        },
        "nll_sum_per_chunk": {
            k: v[0] / v[2] if v[2] else None for k, v in totals.items()
        },
        "target_tokens": {k: v[1] for k, v in totals.items()},
        "target_chunks": {k: v[2] for k, v in totals.items()},
    }


def validation(
    model,
    alpha,
    lexical,
    data: Path,
    kind: str,
    monitor_windows: int,
    block_size: int,
    batch_windows: int,
    sentence_prompt_mode="chunk",
):
    load_manifest(data)
    train_count = len(PackedBank(data, "train_bank"))
    bank = PackedBank(data, "validation_bank")
    windows = Windows(data, "validation")
    indices, selection = monitor_selection(windows.meta, monitor_windows)
    rows = windows.batch(indices)
    if kind == "sentence":
        result = sentence_validation(
            model,
            alpha,
            lexical,
            bank,
            train_count,
            rows,
            block_size,
            batch_windows,
            sentence_prompt_mode,
        )
    else:
        result = token_validation(model, bank, rows, batch_windows, train_count)
    result["monitor_selection"] = selection
    return result


def clone_cache(cache, batch):
    result = copy.copy(cache)
    result.layers = []
    for layer in cache.layers:
        new = copy.copy(layer)
        for key, value in vars(layer).items():
            if isinstance(value, torch.Tensor):
                if value.ndim and value.shape[0] == 1:
                    value = value.expand(batch, *value.shape[1:]).clone()
                else:
                    value = value.clone()
                setattr(new, key, value)
        result.layers.append(new)
    return result


@torch.no_grad()
def token_candidate_scores(model, prompt: list[int], bank: PackedBank, batch: int = 32):
    """Yield summed log-probabilities of candidate content tokens, without EOS.

    The prefix KV state is shared across candidate batches. This is exact but
    expensive for multi-million-entry banks; callers may distribute queries.
    A candidate's score is its token-prefix probability, not a probability of
    stopping there. Normalizing these scores over the bank defines a
    categorical readout; it is not conditioning disjoint stopped-string events
    on catalogue membership when candidates can be prefixes of other entries.
    """
    device = model.embed_tokens.weight.device
    weight = model.embed_tokens.weight
    prefix = torch.tensor([prompt], device=device)
    out = model(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix, dtype=torch.bool),
        use_cache=True,
    )
    first = (out.last_hidden_state[0, -1].float() @ weight.T).log_softmax(-1)
    cache = out.past_key_values
    for lo in range(0, len(bank), batch):
        hi = min(lo + batch, len(bank))
        candidates = [bank.sequence(i) for i in range(lo, hi)]
        starts = torch.tensor([s[0] for s in candidates], device=device)
        values = first[starts].clone()
        length = max(len(s) - 1 for s in candidates)
        if length:
            x = torch.zeros((len(candidates), length), device=device, dtype=torch.long)
            mask = torch.zeros_like(x, dtype=torch.bool)
            groups, positions, labels = [], [], []
            for i, s in enumerate(candidates):
                count = len(s) - 1
                if count:
                    x[i, :count] = torch.tensor(s[:-1], device=device)
                    mask[i, :count] = True
                    groups.extend([i] * count)
                    positions.extend(range(count))
                    labels.extend(s[1:])
            full_mask = torch.cat(
                (
                    torch.ones(
                        (len(candidates), len(prompt)), device=device, dtype=torch.bool
                    ),
                    mask,
                ),
                dim=1,
            )
            result = model(
                input_ids=x,
                attention_mask=full_mask,
                past_key_values=clone_cache(cache, len(candidates)),
                use_cache=True,
            )
            hidden = result.last_hidden_state[
                torch.tensor(groups, device=device),
                torch.tensor(positions, device=device),
            ]
            labels_tensor = torch.tensor(labels, device=device)
            lp = -F.cross_entropy(
                hidden.float() @ weight.T, labels_tensor, reduction="none"
            )
            values.index_add_(0, torch.tensor(groups, device=device), lp)
        yield lo, values
