"""Causal supervision, cached generation, and independent delayed register queries."""

import copy

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .dataset import EncodedSplit, Vocabulary, load_manifest, load_data_config
from .metrics import cache_bytes, CounterMetrics
from .model import CONDITIONS, COT2, Model
from .cot2 import distribution_statistics, evaluate_distribution


def teacher_forced_inputs(model, batch):
    parts = [model.embeddings(batch["prompt_ids"])]
    cursor = batch["prompt_ids"].shape[1]
    positions = []
    for step in range(batch["state_ids"].shape[1]):
        # The prefix's <TRACE>, then the preceding payload, predicts the next state.
        positions.append(
            list(range(cursor - 1, cursor - 1 + model.num_registers))
            if model.condition == "discrete_conventional"
            else [cursor - 1]
        )
        feedback = model.feedback(batch["state_ids"][:, step])
        parts.append(feedback)
        cursor += feedback.shape[1]
    parts.append(model.embeddings(batch["query_ids"]))
    cursor += batch["query_ids"].shape[1]
    if cursor > model.model_config.max_positions:
        raise ValueError("Sequence exceeds max_positions")
    return (
        torch.cat(parts, 1),
        torch.tensor(positions, device=batch["prompt_ids"].device),
        cursor - 1,
    )


def state_logits(model, hidden, positions):
    if model.condition == COT2:
        return model.language_logits(hidden[:, positions[:, 0]])
    return (
        model.node_logits(hidden[:, positions])
        if model.condition == "discrete_conventional"
        else model.role_logits(hidden[:, positions[:, 0]])
    )


def supervised_loss(model, batch):
    inputs, positions, query_position = teacher_forced_inputs(model, batch)
    hidden = model.full_hidden(inputs)
    logits = state_logits(model, hidden, positions)
    state = (
        distribution_statistics(logits, batch["state_ids"])["ce"].mean()
        if model.condition == COT2
        else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch["state_ids"].reshape(-1)
        )
    )
    query = F.cross_entropy(
        model.node_logits(hidden[:, query_position]), batch["query_targets"]
    )
    losses = {"total": 0.5 * (state + query), "reasoning": state, "query": query}
    if model.condition == COT2:
        losses["state_kl"] = distribution_statistics(logits, batch["state_ids"])[
            "kl"
        ].mean()
    return losses


def query_tokens(model, steps, role):
    v = model.vocabulary
    return torch.stack(
        [
            torch.full_like(steps, v.id("<QUERY>")),
            torch.tensor([v.id(str(i)) for i in steps.tolist()], device=steps.device),
            torch.full_like(steps, v.id(v.roles[role])),
        ],
        -1,
    )


@torch.no_grad()
def rollout(model, batch, *, gold=False, branch_queries=True, historical=False):
    hidden, cache = model.cached_hidden(
        inputs_embeds=model.embeddings(batch["prompt_ids"])
    )
    trace = []
    distribution_logits = []
    n, length, _ = batch["state_ids"].shape
    for step in range(length):
        if model.condition == COT2:
            logits = model.language_logits(hidden[:, -1])
            distribution_logits.append(logits)
            feedback = (
                model.feedback(batch["state_ids"][:, step])
                if gold
                else model.distribution_feedback(logits.softmax(-1))
            )
            hidden, cache = model.cached_hidden(
                inputs_embeds=feedback, past_key_values=cache
            )
            continue
        if model.condition == "discrete_conventional":
            predictions = []
            for role in range(model.num_registers):
                prediction = model.node_logits(hidden[:, -1]).argmax(-1)
                predictions.append(prediction)
                token = batch["state_ids"][:, step, role] if gold else prediction
                hidden, cache = model.cached_hidden(
                    inputs_embeds=model.embeddings(token[:, None]),
                    past_key_values=cache,
                )
            prediction = torch.stack(predictions, 1)
        else:
            prediction = model.role_logits(hidden[:, -1]).argmax(-1)
            feedback = model.feedback(
                batch["state_ids"][:, step] if gold else prediction
            )
            hidden, cache = model.cached_hidden(
                inputs_embeds=feedback, past_key_values=cache
            )
        trace.append(prediction)
    positions, byte_count = int(cache.get_seq_length()), cache_bytes(cache) // n

    def ask(tokens):
        # Even a single branch gets a copy: the canonical completed trace never changes.
        qh, _ = model.cached_hidden(
            inputs_embeds=model.embeddings(tokens), past_key_values=copy.deepcopy(cache)
        )
        return model.node_logits(qh[:, -1]).argmax(-1)

    queries = (
        [
            ask(query_tokens(model, torch.full_like(batch["query_roles"], length), r))
            for r in range(model.num_registers)
        ]
        if branch_queries
        else [ask(batch["query_ids"])]
    )
    historical_queries = None
    if historical and length > 1:
        historical_queries = torch.stack(
            [
                ask(query_tokens(model, batch["historical_step"], r))
                for r in range(model.num_registers)
            ],
            1,
        )
    return dict(
        trace=torch.stack(trace, 1) if trace else None,
        distribution_logits=(
            torch.stack(distribution_logits, 1) if distribution_logits else None
        ),
        queries=torch.stack(queries, 1),
        historical_queries=historical_queries,
        cache_positions=positions,
        cache_bytes_per_sample=byte_count,
    )


def efficiency(model, prompt_length, length):
    width = model.num_registers if model.condition == "discrete_conventional" else 1
    return dict(
        num_registers=model.num_registers,
        prompt_positions=prompt_length,
        path_length=length,
        state_payload_positions=width * length,
        cache_positions_before_query=prompt_length + width * length,
        reasoning_decode_calls=width * length,
        single_query_forward_calls_including_prefill=2 + width * length,
        single_query_total_positions=prompt_length + width * length + 3,
    )


@torch.no_grad()
def evaluate(model, loader, device, *, gold=False, historical=True):
    if model.condition == COT2:
        return evaluate_distribution(
            model, loader, device, gold=gold, historical=historical
        )
    model.eval()
    counts = CounterMetrics()
    length = loader.dataset.config.path_length
    per_step = torch.zeros(length, dtype=torch.long)
    values_per_step = torch.zeros(length, dtype=torch.long)
    for raw in loader:
        batch = {k: v.to(device) for k, v in raw.items()}
        result = rollout(model, batch, gold=gold, historical=historical)
        target = batch["state_ids"]
        correct = result["trace"].eq(target)
        states = correct.all(-1)
        final = result["queries"].eq(target[:, -1])
        counts.add("query_in_final_state", result["queries"].unsqueeze(-1).eq(
            target[:, -1].unsqueeze(1)).any(-1).sum())
        counts.add("samples", len(target))
        counts.add("joint", final.all(-1).sum())
        counts.add("random", final.gather(1, batch["query_roles"][:, None]).sum())
        for i, role in enumerate(model.vocabulary.roles):
            counts.add(role, final[:, i].sum())
        counts.add("outputs", correct.numel())
        counts.add("outputs_correct", correct.sum())
        counts.add("states", states.numel())
        counts.add("states_correct", states.sum())
        counts.add("trace", states.all(-1).sum())
        per_step += states.sum(0).cpu()
        values_per_step += correct.sum((0, 2)).cpu()
        after = F.pad(batch["heldout"][:, :-1], (1, 0), value=False)
        for name, mask in [("heldout", batch["heldout"]), ("after_heldout", after)]:
            counts.add(name, mask.sum())
            counts.add(name + "_correct", states[mask].sum())
        if result["historical_queries"] is not None:
            previous = target[
                torch.arange(len(target), device=device), batch["historical_step"] - 1
            ]
            matches = result["historical_queries"].eq(previous)
            counts.add("historical_samples", len(target))
            counts.add("historical_joint", matches.all(-1).sum())
            counts.add(
                "historical_random",
                matches.gather(1, batch["query_roles"][:, None]).sum(),
            )
        expected = efficiency(model, loader.dataset.prompt_length, length)
        if result["cache_positions"] != expected["cache_positions_before_query"]:
            raise AssertionError("Actual cache length differs from position accounting")
    if not counts["samples"]:
        raise ValueError("Empty evaluation")
    return dict(
        samples=counts["samples"],
        feedback_mode="gold" if gold else "predicted",
        branched_full_state_exact_accuracy=counts.rate("joint", "samples"),
        random_query_accuracy=counts.rate("random", "samples"),
        per_role_query_accuracy={
            r: counts.rate(r, "samples") for r in model.vocabulary.roles
        },
        mean_register_query_accuracy=sum(counts[r] for r in model.vocabulary.roles)
        / (counts["samples"] * model.num_registers),
        query_value_in_final_state_accuracy=counts["query_in_final_state"]
        / (counts["samples"] * model.num_registers),
        trace_exact_accuracy=counts.rate("trace", "samples"),
        trace_state_exact_accuracy=counts.rate("states_correct", "states"),
        trace_value_accuracy=counts.rate("outputs_correct", "outputs"),
        first_update_exact_accuracy=per_step[0].item() / counts["samples"],
        state_exact_accuracy_by_step=(per_step / counts["samples"]).tolist(),
        state_value_accuracy_by_step=(values_per_step / (counts["samples"] * model.num_registers)).tolist(),
        heldout_state_count=counts["heldout"],
        heldout_state_exact_accuracy=counts.rate("heldout_correct", "heldout"),
        transition_after_heldout_count=counts["after_heldout"],
        transition_after_heldout_exact_accuracy=counts.rate(
            "after_heldout_correct", "after_heldout"
        ),
        historical_probe=dict(
            samples=counts["historical_samples"],
            all_register_exact_accuracy=counts.rate(
                "historical_joint", "historical_samples"
            ),
            random_query_accuracy=counts.rate(
                "historical_random", "historical_samples"
            ),
        ),
        cache_bytes_per_sample_before_query=result["cache_bytes_per_sample"],
        efficiency=expected,
    )
