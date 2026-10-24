"""Role-free distribution targets and diagnostics for the CoT2-style condition.

The target is the empirical distribution of all register values (including
multiplicity). Distribution metrics deliberately do not assign values to roles.
"""

import torch
import torch.nn.functional as F


def target_distribution(states, vocabulary_size, dtype=torch.float32):
    target = torch.zeros(
        (*states.shape[:-1], vocabulary_size), device=states.device, dtype=dtype
    )
    return target.scatter_add_(
        -1, states, torch.full_like(states, 1 / states.shape[-1], dtype=dtype)
    )


def distribution_statistics(logits, states):
    target = target_distribution(states, logits.shape[-1], logits.dtype)
    log_probs = F.log_softmax(logits, -1)
    # Gathering avoids multiplying zero target probabilities by extreme logits.
    ce = -log_probs.gather(-1, states).mean(-1)
    entropy = -(target * target.clamp_min(torch.finfo(target.dtype).tiny).log()).sum(-1)
    return {
        "ce": ce,
        "entropy": entropy,
        "kl": (ce - entropy).clamp_min(0),
        "tv": 0.5 * (log_probs.exp() - target).abs().sum(-1),
    }


@torch.no_grad()
def evaluate_distribution(model, loader, device, *, gold=False, historical=True):
    from .core import rollout, efficiency

    model.eval()
    n = joint = random = hn = hjoint = hrandom = 0
    query_in_final_state = 0
    roles = [0] * model.num_registers
    sums = {k: 0.0 for k in ("ce", "kl", "tv")}
    masksums = {
        key: {"count": 0, "kl": 0.0, "tv": 0.0} for key in ("heldout", "after_heldout")
    }
    length = loader.dataset.config.path_length
    step_kl = torch.zeros(length, dtype=torch.float64)
    step_tv = torch.zeros(length, dtype=torch.float64)
    for raw in loader:
        batch = {k: v.to(device) for k, v in raw.items()}
        result = rollout(model, batch, gold=gold, historical=historical)
        stats = distribution_statistics(
            result["distribution_logits"], batch["state_ids"]
        )
        for key in sums:
            sums[key] += stats[key].sum().item()
        step_kl += stats["kl"].sum(0).cpu().double()
        step_tv += stats["tv"].sum(0).cpu().double()
        after = F.pad(batch["heldout"][:, :-1], (1, 0), value=False)
        for key, mask in [("heldout", batch["heldout"]), ("after_heldout", after)]:
            masksums[key]["count"] += mask.sum().item()
            for metric in ("kl", "tv"):
                masksums[key][metric] += stats[metric][mask].sum().item()
        correct = result["queries"].eq(batch["state_ids"][:, -1])
        query_in_final_state += result["queries"].unsqueeze(-1).eq(
            batch["state_ids"][:, -1].unsqueeze(1)).any(-1).sum().item()
        n += len(correct)
        joint += correct.all(-1).sum().item()
        random += correct.gather(1, batch["query_roles"][:, None]).sum().item()
        for i in range(model.num_registers):
            roles[i] += correct[:, i].sum().item()
        if result["historical_queries"] is not None:
            target = batch["state_ids"][
                torch.arange(len(correct), device=device), batch["historical_step"] - 1
            ]
            matches = result["historical_queries"].eq(target)
            hn += len(matches)
            hjoint += matches.all(-1).sum().item()
            hrandom += matches.gather(1, batch["query_roles"][:, None]).sum().item()
        expected = efficiency(model, loader.dataset.prompt_length, length)
        if result["cache_positions"] != expected["cache_positions_before_query"]:
            raise AssertionError("Actual cache length differs from position accounting")
    if not n:
        raise ValueError("Empty evaluation")
    return dict(
        samples=n,
        feedback_mode="gold" if gold else "predicted",
        branched_full_state_exact_accuracy=joint / n,
        random_query_accuracy=random / n,
        per_role_query_accuracy={
            r: roles[i] / n for i, r in enumerate(model.vocabulary.roles)
        },
        mean_register_query_accuracy=sum(roles) / (n * model.num_registers),
        query_value_in_final_state_accuracy=query_in_final_state / (n * model.num_registers),
        trace_exact_accuracy=None,
        trace_state_exact_accuracy=None,
        trace_value_accuracy=None,
        first_update_exact_accuracy=None,
        state_exact_accuracy_by_step=None,
        state_value_accuracy_by_step=None,
        heldout_state_count=masksums["heldout"]["count"],
        heldout_state_exact_accuracy=None,
        transition_after_heldout_count=masksums["after_heldout"]["count"],
        transition_after_heldout_exact_accuracy=None,
        distribution_metrics={
            **{k: v / (n * length) for k, v in sums.items()},
            "kl_by_step": (step_kl / n).tolist(),
            "tv_by_step": (step_tv / n).tolist(),
            **{
                key: {
                    "count": val["count"],
                    **{
                        m: val[m] / val["count"] if val["count"] else None
                        for m in ("kl", "tv")
                    },
                }
                for key, val in masksums.items()
            },
        },
        trace_metric_note="No ordered trace predictions: distribution metrics assess unordered value multiplicities; queries still assess roles.",
        historical_probe={
            "samples": hn,
            "all_register_exact_accuracy": hjoint / hn if hn else None,
            "random_query_accuracy": hrandom / hn if hn else None,
        },
        cache_bytes_per_sample_before_query=result["cache_bytes_per_sample"],
        efficiency=expected,
    )
