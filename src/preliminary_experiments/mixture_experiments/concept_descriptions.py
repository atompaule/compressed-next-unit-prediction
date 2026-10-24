
# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.preliminary_experiments.methods.config import CFG
from src.preliminary_experiments.methods.soft_thinking import soft_thinking
from src.preliminary_experiments.methods.solver import geometric_solver


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    log_file: str = "results/preliminary_experiments/mixtures/llama_8b_concept_descriptions.txt"
    display_k: int = 20
    max_new_tokens: int = 80
    seed: int = 42


EXP_CFG = ExperimentConfig()

DESCRIPTION_PREFIX = (
    "Instruction: Describe the object represented by the following input in one "
    "concise paragraph.\n\n"
    "Object: "
)
DESCRIPTION_SUFFIX = "\n\nDescription: "

TARGET_LOGITS = [16.0, 15.8, 15.6]

EXPERIMENTS = [
    {
        "name": "Ambiguity: river bank",
        "baseline_word": " bank",
        "words": [" bank", " river", " land"],
        "logits": TARGET_LOGITS,
        "prefix": DESCRIPTION_PREFIX,
        "suffix": DESCRIPTION_SUFFIX,
    },
    {
        "name": "City: Las Vegas",
        "baseline_word": " city",
        "words": [" city", " casinos", " weddings"],
        "logits": TARGET_LOGITS,
        "prefix": DESCRIPTION_PREFIX,
        "suffix": DESCRIPTION_SUFFIX,
    },
    {
        "name": "City: Amsterdam",
        "baseline_word": " city",
        "words": [" city", " bicycles", " canal"],
        "logits": TARGET_LOGITS,
        "prefix": DESCRIPTION_PREFIX,
        "suffix": DESCRIPTION_SUFFIX,
    },
    {
        "name": "Mountain: huge",
        "baseline_word": " mountain",
        "words": [" mountain", " huge", " scary"],
        "logits": TARGET_LOGITS,
        "prefix": DESCRIPTION_PREFIX,
        "suffix": DESCRIPTION_SUFFIX,
    },
    {
        "name": "Mountain: tiny",
        "baseline_word": " mountain",
        "words": [" mountain", " tiny", " cute"],
        "logits": TARGET_LOGITS,
        "prefix": DESCRIPTION_PREFIX,
        "suffix": DESCRIPTION_SUFFIX,
    },
    {
        "name": "Singer: Amy Winehouse",
        "baseline_word": " singer",
        "words": [" singer", " rehab", " tattoos"],
        "logits": TARGET_LOGITS,
        "prefix": DESCRIPTION_PREFIX,
        "suffix": DESCRIPTION_SUFFIX,
    },
    {
        "name": "Singer: Bob Marley",
        "baseline_word": " singer",
        "words": [" singer", " Jamaica", " redemption"],
        "logits": TARGET_LOGITS,
        "prefix": DESCRIPTION_PREFIX,
        "suffix": DESCRIPTION_SUFFIX,
    },
]


def emit(text: str, output) -> None:
    print(text)
    print(text, file=output)


def single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} fragments into token IDs {ids}")
    return ids[0]


def display_token(tokenizer, token_id: int) -> str:
    token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    return token.replace("\n", "[newline]").replace("\t", "[tab]")


def validate_experiments(tokenizer) -> None:
    failures = []
    for experiment in EXPERIMENTS:
        for word in [experiment["baseline_word"], *experiment["words"]]:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) != 1:
                failures.append(f"{experiment['name']}: {word!r} -> {ids}")
    if failures:
        raise ValueError("Token fragmentation:\n" + "\n".join(failures))


def build_logits(target_ids, logit_values, vocab_size, device):
    logits = torch.full((vocab_size,), -100.0, device=device)
    logits[target_ids] = logit_values
    return logits


def build_vectors(full_logits, target_ids, vocab_embs):
    argmax = vocab_embs[target_ids[0]].unsqueeze(0)
    soft = soft_thinking(
        full_logits,
        vocab_embs,
        temperature=CFG.temperature,
        top_p=CFG.top_p,
    )
    centroid = vocab_embs[target_ids].mean(dim=0, keepdim=True)
    with torch.enable_grad():
        geometric = geometric_solver(full_logits, vocab_embs, use_cosine=True)
    return {
        "Argmax": argmax,
        "Soft Thinking": soft,
        "Centroid": centroid,
        "Geometric optimizer": geometric,
    }


def generate_description(model, tokenizer, prefix, suffix, vector, device) -> str:
    prefix_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
    suffix_ids = tokenizer(
        suffix, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    embedding_layer = model.get_input_embeddings()
    inputs = torch.cat(
        [
            embedding_layer(prefix_ids),
            vector.unsqueeze(0).to(dtype=model.dtype),
            embedding_layer(suffix_ids),
        ],
        dim=1,
    )
    attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
    with torch.no_grad():
        # The generation cache is created for this call only. No past_key_values
        # object is supplied to, retained from, or reused by another example.
        output = model.generate(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            max_new_tokens=EXP_CFG.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()


def emit_wrapped(text: str, output, width: int = 105) -> None:
    words = text.replace("\n", " [newline] ").split()
    lines = []
    current = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    for line in lines:
        emit(f"    {line}", output)


def emit_input_table(
    tokenizer, target_ids, logit_values, full_logits, vocab_embs, output
) -> None:
    raw_probs = F.softmax(full_logits, dim=-1)[target_ids]
    scaled_probs = F.softmax(full_logits / CFG.temperature, dim=-1)[target_ids]
    norms = vocab_embs[target_ids].norm(dim=1)

    emit("\nINPUT TOKENS", output)
    emit("Rank | Token        | Logit | Raw P  | Scaled P | Embedding norm", output)
    emit("-----|--------------|-------|--------|----------|---------------", output)
    for rank, (token_id, logit, raw_p, scaled_p, norm) in enumerate(
        zip(target_ids, logit_values, raw_probs, scaled_probs, norms), start=1
    ):
        token = display_token(tokenizer, token_id).strip()
        emit(
            f"{rank:>4} | {token:<12} | {logit.item():>5.1f} | "
            f"{raw_p.item() * 100:>6.2f}% | {scaled_p.item() * 100:>7.2f}% | "
            f"{norm.item():>13.3f}",
            output,
        )


def emit_cosine_rankings(
    tokenizer, vectors, target_ids, vocab_embs_normalized, output
) -> None:
    names = list(vectors)
    matrix = F.normalize(torch.cat(list(vectors.values()), dim=0), dim=1)
    similarities = matrix @ vocab_embs_normalized.T
    values, indices = torch.topk(similarities, EXP_CFG.display_k, dim=1)
    target_id_set = set(target_ids)
    column_width = 35

    emit("\nCOSINE-NEAREST VOCABULARY EMBEDDINGS (* = input token)", output)
    header = f"{'Rank':<4} | " + " | ".join(f"{name:<{column_width}}" for name in names)
    emit(header, output)
    emit("-" * len(header), output)
    for rank in range(EXP_CFG.display_k):
        cells = []
        for condition_index in range(len(names)):
            token_id = indices[condition_index, rank].item()
            marker = "*" if token_id in target_id_set else " "
            token = display_token(tokenizer, token_id)
            value = values[condition_index, rank].item()
            cells.append(f"{marker}{token!r} ({value:.3f})".ljust(column_width))
        emit(f"{rank + 1:<4} | " + " | ".join(cells), output)


def cancellation_metrics(component_embs, weights) -> dict[str, float]:
    """Decompose the squared norm of a non-negative weighted embedding mixture."""
    weights = weights.float()
    weights = weights / weights.sum()
    weighted_embs = weights.unsqueeze(1) * component_embs.float()
    mixture = weighted_embs.sum(dim=0)

    self_energy = weighted_embs.square().sum()
    positive_pair_energy = torch.zeros((), device=component_embs.device)
    negative_pair_loss = torch.zeros((), device=component_embs.device)
    for i in range(len(weights)):
        for j in range(i + 1, len(weights)):
            pair_term = 2.0 * torch.dot(weighted_embs[i], weighted_embs[j])
            positive_pair_energy += pair_term.clamp_min(0.0)
            negative_pair_loss += (-pair_term).clamp_min(0.0)

    positive_budget = self_energy + positive_pair_energy
    mixture_norm = mixture.norm()
    aligned_norm = (weights * component_embs.float().norm(dim=1)).sum()
    orthogonal_norm = self_energy.sqrt()

    return {
        "mixture_norm": mixture_norm.item(),
        "aligned_norm": aligned_norm.item(),
        "aligned_retention": (mixture_norm / aligned_norm).item(),
        "orthogonal_norm": orthogonal_norm.item(),
        "orthogonal_ratio": (mixture_norm / orthogonal_norm).item(),
        "negative_pair_loss": negative_pair_loss.item(),
        "cancellation_fraction": (negative_pair_loss / positive_budget).item(),
    }


def emit_cancellation_analysis(
    tokenizer, target_ids, full_logits, vocab_embs, output
) -> None:
    component_embs = vocab_embs[target_ids]
    component_unit = F.normalize(component_embs, dim=1)
    pairwise_cosines = component_unit @ component_unit.T
    token_names = [
        display_token(tokenizer, token_id).strip() for token_id in target_ids
    ]

    emit("\nCANCELLATION ANALYSIS", output)
    emit("Pairwise support geometry", output)
    emit("Token pair                    | Cosine similarity | Relation", output)
    emit("------------------------------|-------------------|----------", output)
    for i in range(len(target_ids)):
        for j in range(i + 1, len(target_ids)):
            cosine = pairwise_cosines[i, j].item()
            relation = "opposing" if cosine < 0.0 else "non-opposing"
            pair = f"{token_names[i]} / {token_names[j]}"
            emit(f"{pair:<30}| {cosine:>17.4f} | {relation}", output)

    soft_weights = F.softmax(full_logits / CFG.temperature, dim=-1)[target_ids]
    soft_weights = soft_weights / soft_weights.sum()
    centroid_weights = torch.full_like(soft_weights, 1.0 / len(target_ids))
    conditions = {
        "Soft Thinking": soft_weights,
        "Centroid": centroid_weights,
    }

    emit("\nMixture-level cancellation", output)
    emit(
        "Condition     | Weights                 | ||z|| | Aligned max | "
        "Retention | Orthogonal ref | z / ref | Opposition loss | Cancellation",
        output,
    )
    emit(
        "--------------|-------------------------|-------|-------------|-----------|"
        "----------------|---------|-----------------|-------------",
        output,
    )
    for name, weights in conditions.items():
        metrics = cancellation_metrics(component_embs, weights)
        weight_text = ", ".join(f"{weight.item():.4f}" for weight in weights)
        emit(
            f"{name:<14}| {weight_text:<24}| "
            f"{metrics['mixture_norm']:>5.3f} | "
            f"{metrics['aligned_norm']:>11.3f} | "
            f"{metrics['aligned_retention'] * 100:>8.2f}% | "
            f"{metrics['orthogonal_norm']:>14.3f} | "
            f"{metrics['orthogonal_ratio']:>7.3f} | "
            f"{metrics['negative_pair_loss']:>15.6f} | "
            f"{metrics['cancellation_fraction'] * 100:>10.2f}%",
            output,
        )

    emit(
        "Interpretation: aligned retention includes all directional separation. "
        "The opposition loss counts only negative pair terms in ||z||^2; its "
        "cancellation percentage is the share of otherwise positive squared-norm "
        "contributions removed by those terms. Orthogonal pairs reduce retention "
        "relative to the aligned maximum but do not count as destructive opposition.",
        output,
    )
    emit(
        "Argmax has one component, while the geometric optimizer is constructed "
        "directly rather than from explicit mixture weights; neither has a "
        "component-wise cancellation decomposition here.",
        output,
    )


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("This experiment requires an available MPS device.")
    device = torch.device("mps")
    torch.manual_seed(EXP_CFG.seed)

    print(f"Loading {EXP_CFG.model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        EXP_CFG.model_id,
        local_files_only=False,
        clean_up_tokenization_spaces=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        EXP_CFG.model_id,
        dtype=torch.float16,
        local_files_only=False,
    ).to(device)
    model.eval()

    validate_experiments(tokenizer)
    vocab_embs = model.get_input_embeddings().weight.detach().float()
    vocab_embs_normalized = F.normalize(vocab_embs, dim=1)

    _Path(EXP_CFG.log_file).parent.mkdir(parents=True, exist_ok=True)
    with open(EXP_CFG.log_file, "x", encoding="utf-8") as output:
        emit("QUALITATIVE CONTINUOUS-EMBEDDING CONCEPT DESCRIPTIONS", output)
        emit(
            f"Model: {EXP_CFG.model_id} | Device: {device} | Seed: {EXP_CFG.seed}",
            output,
        )
        emit(
            f"Temperature: {CFG.temperature} | Top-p: {CFG.top_p} | "
            f"Generation: greedy | Max new tokens: {EXP_CFG.max_new_tokens}",
            output,
        )

        for experiment in EXPERIMENTS:
            emit("\n" + "=" * 120, output)
            emit(f"CASE: {experiment['name']}", output)
            emit("=" * 120, output)
            emit(
                f"Prompt: {experiment['prefix']!r} + INPUT + {experiment['suffix']!r}",
                output,
            )

            target_ids = [single_token_id(tokenizer, w) for w in experiment["words"]]
            logit_values = torch.tensor(experiment["logits"], device=device)
            full_logits = build_logits(
                target_ids, logit_values, vocab_embs.size(0), device
            )

            emit_input_table(
                tokenizer,
                target_ids,
                logit_values,
                full_logits,
                vocab_embs,
                output,
            )

            emit_cancellation_analysis(
                tokenizer,
                target_ids,
                full_logits,
                vocab_embs,
                output,
            )

            torch.manual_seed(EXP_CFG.seed)
            vectors = build_vectors(full_logits, target_ids, vocab_embs)
            emit_cosine_rankings(
                tokenizer, vectors, target_ids, vocab_embs_normalized, output
            )

            emit("\nDISCRETE CONSTITUENT DESCRIPTIONS", output)
            for word, token_id in zip(experiment["words"], target_ids):
                emit(f"\n  {word.strip()}:", output)
                description = generate_description(
                    model,
                    tokenizer,
                    experiment["prefix"],
                    experiment["suffix"],
                    vocab_embs[token_id].unsqueeze(0),
                    device,
                )
                emit_wrapped(description, output)

            emit("\nCONDITION DESCRIPTIONS", output)
            for name, vector in vectors.items():
                emit(f"\n  {name}:", output)
                description = generate_description(
                    model,
                    tokenizer,
                    experiment["prefix"],
                    experiment["suffix"],
                    vector,
                    device,
                )
                emit_wrapped(description, output)

    print(f"Saved to {EXP_CFG.log_file}")


if __name__ == "__main__":
    main()
