"""
eval_arc_latent.py

Experiment 4.II — Priority 2 runs: ARC-Challenge
Compares Soft Thinking, Noisy Greedy Discrete (Isotropic Gaussian),
Greedy Discrete, and Default Baseline (temperature sampling).

One combined log + JSON for all 4 approaches.

Gemma 2 specifics:
  - Gemma2TextScaledWordEmbedding multiplies raw weights by sqrt(hidden_size).
    embed_layer(input_ids) returns scaled embeddings, but .weight is unscaled.
    We must scale vocab_embs by embed_scale so the decode-step embeddings
    match the magnitude the model saw during prefill.
  - Gemma 2 uses HybridCache by default; we force DynamicCache to avoid
    issues with our manual generation loop.
  - The attention mask must grow by 1 each decode step (sliding window layers).
  - apply_chat_template(tokenize=False) prepends <bos>, so we pass
    add_special_tokens=False when re-tokenizing to avoid a double <bos>.
  - Gemma 2 ends turns with <end_of_turn>, not <eos>.
"""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
import re
import time

import torch
import torch.nn.functional as F
from src.common.datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.preliminary_experiments.methods.config import CFG
from src.preliminary_experiments.soft_thinking.mixture_protocol import (
    benchmark_soft_thinking as soft_thinking, thesis_approach_configs,
)
from src.preliminary_experiments.methods.random_perturbation import noisy_target_sim
from src.preliminary_experiments.methods.utils import emit

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MAX_SAMPLES = 10000  # ARC-Challenge test split has 1172 items; cap is a safety net
MAX_NEW_TOKENS = 1024  # model reasons before giving final letter

APPROACH_CONFIGS = thesis_approach_configs()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# ARC-specific utils
# ---------------------------------------------------------------------------

_INT_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


def normalize_answer_key(key: str) -> str:
    key = key.strip().upper()
    return _INT_TO_LETTER.get(key, key)


def build_prompt(question: str, choices: dict) -> str:
    lines = [
        "Answer the following multiple choice question. "
        "First reason through it under 'reasoning: ', then give your final answer "
        "under 'answer: ' using exactly the format: answer: X "
        "(where X is a single letter A, B, C, or D).\n",
        f"Question: {question}",
    ]
    for label, text in zip(choices["label"], choices["text"]):
        display_label = _INT_TO_LETTER.get(label, label)
        lines.append(f"{display_label}) {text}")
    lines.append("\nreasoning:")
    return "\n".join(lines)


def extract_letter(text: str) -> str | None:
    matches = re.findall(r"answer\s*:\s*([A-Ea-e])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
) -> int:
    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    probs = F.softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        mask = (cumulative - sorted_probs) >= top_p
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    vocab_embs: torch.Tensor,  # already scaled by embed_scale
    vocab_embs_norm: torch.Tensor,
    config: dict,
    eot_id: int,
    pc_cache: dict | None = None,
) -> list[str]:
    """Batched autoregressive generation with custom embedding logic."""
    approach = config["name"]
    sampling_temp = config["sampling_temp"]
    top_p = config.get("top_p", 1.0)

    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    B = len(prompts)

    # apply_chat_template(tokenize=False) prepends <bos>, so pass
    # add_special_tokens=False to avoid a double <bos> on re-tokenization
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for p in prompts
    ]

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,  # <bos> already in the formatted string
    ).to(device)

    input_ids = enc["input_ids"]  # [B, L]
    attention_mask = enc["attention_mask"]  # [B, L]
    inputs_embeds = embed_layer(input_ids)  # [B, L, d] — embed_layer applies scale

    # Force DynamicCache: Gemma 2's default HybridCache (sliding window + global)
    # has pre-allocated fixed sizes that break our manual generation loop
    past_key_values = DynamicCache()

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :].float()  # [B, V]

    generated = [""] * B
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    # Sliding window layers need the mask to grow each step
    full_attention_mask = attention_mask

    for _ in range(MAX_NEW_TOKENS):
        if finished.all():
            break

        next_vecs = []
        fed_ids = []

        for b in range(B):
            if finished[b]:
                next_vecs.append(vocab_embs[tokenizer.eos_token_id])
                fed_ids.append(tokenizer.eos_token_id)
                continue

            b_logits = logits[b]  # [V]
            greedy_id = sample_token(b_logits, sampling_temp, top_p)

            if approach in ("discrete_top1", "default_baseline"):
                next_vec = vocab_embs[greedy_id]  # already scaled
                fed_id = greedy_id

            elif approach == "soft_thinking":
                next_vec = soft_thinking(b_logits, vocab_embs).squeeze(0)
                next_vec_unit = F.normalize(next_vec.unsqueeze(0), p=2, dim=1)
                fed_id = int(
                    (next_vec_unit @ vocab_embs_norm.T).squeeze(0).argmax().item()
                )

            elif approach == "noisy_discrete":
                v_soft = soft_thinking(b_logits, vocab_embs)
                v_soft_unit = F.normalize(v_soft, p=2, dim=1)
                all_cos_sims = (v_soft_unit @ vocab_embs_norm.T).squeeze(0)
                nearest_id = int(all_cos_sims.argmax().item())
                soft_max_sim = all_cos_sims[nearest_id].clamp(-1.0, 1.0).item()
                next_vec = noisy_target_sim(
                    vocab_embs[nearest_id], target_sim=soft_max_sim
                )
                next_vec_unit = F.normalize(next_vec.unsqueeze(0), p=2, dim=1)
                fed_id = int(
                    (next_vec_unit @ vocab_embs_norm.T).squeeze(0).argmax().item()
                )




            else:
                raise ValueError(f"Unknown approach: {approach!r}")

            token = tokenizer.decode([fed_id])
            generated[b] += token

            # Gemma 2 ends turns with <end_of_turn>, not <eos>
            if fed_id in (tokenizer.eos_token_id, eot_id) or re.search(
                r"answer\s*:\s*[A-Ea-e]\b", generated[b], re.IGNORECASE
            ):
                finished[b] = True

            next_vecs.append(next_vec)
            fed_ids.append(fed_id)

        next_embeds = torch.stack(next_vecs, dim=0).to(device=device, dtype=model.dtype)
        next_embeds = next_embeds.unsqueeze(1)  # [B, 1, d]

        # Grow the attention mask by 1 for the new token position
        full_attention_mask = torch.cat(
            [
                full_attention_mask,
                torch.ones(B, 1, device=device, dtype=full_attention_mask.dtype),
            ],
            dim=1,
        )

        with torch.no_grad():
            outputs = model(
                inputs_embeds=next_embeds,
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()  # [B, V]

    return generated


# ---------------------------------------------------------------------------
# Evaluate one approach config over the dataset
# ---------------------------------------------------------------------------


def evaluate_config(
    model, tokenizer, dataset, vocab_embs, vocab_embs_norm, config, eot_id, f
):
    emit(f"\n{'#'*70}", f)
    emit(f"CONFIG: {config}", f)
    emit(f"{'#'*70}", f)

    correct, total = 0, 0
    results = []
    samples = list(dataset)[:MAX_SAMPLES]
    n_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    pc_cache: dict = {}

    for batch_idx, batch_start in enumerate(
        tqdm(range(0, len(samples), BATCH_SIZE), desc=config["name"])
    ):
        t0 = time.time()
        batch = samples[batch_start : batch_start + BATCH_SIZE]

        prompts = [build_prompt(s["question"], s["choices"]) for s in batch]
        gt_keys = [normalize_answer_key(s["answerKey"]) for s in batch]

        pred_texts = generate_batch(
            model,
            tokenizer,
            prompts,
            vocab_embs,
            vocab_embs_norm,
            config,
            eot_id,
            pc_cache,
        )

        elapsed = time.time() - t0
        emit(
            f"[{config['name']}] batch {batch_idx+1}/{n_batches} "
            f"| {elapsed:.1f}s | {elapsed/len(batch):.2f}s/sample",
            f,
        )

        for i, (sample, gt_key, pred_text) in enumerate(
            zip(batch, gt_keys, pred_texts)
        ):
            global_i = batch_start + i
            pred_key = extract_letter(pred_text)
            is_correct = pred_key is not None and pred_key == gt_key
            correct += int(is_correct)
            total += 1

            emit(f"\n{'='*60}", f)
            emit(f"[{global_i+1}] Q: {sample['question']}", f)
            emit(f"GT: {gt_key}", f)
            emit(pred_text, f)
            emit(f"Pred: {pred_key} | {'✅' if is_correct else '❌'}", f)

            results.append(
                {
                    "index": global_i + 1,
                    "gt": gt_key,
                    "pred": pred_key,
                    "correct": is_correct,
                }
            )

    acc = correct / total
    emit(f"\napproach={config['name']} → {acc:.4f} ({correct}/{total})", f)
    return {
        "config": config,
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2-9b-it",
        help="HuggingFace model ID or local path (default: model_id)",
    )
    parser.add_argument("--output-dir", type=_Path, help="Fresh directory for this evaluation")
    parser.add_argument("--print-config", action="store_true", help="Print the thesis conditions without loading a model")
    args = parser.parse_args()
    if args.print_config:
        print(json.dumps({"model": args.model_id, "conditions": APPROACH_CONFIGS,
                          "soft_temperature": 0.6, "soft_top_p": 0.95}, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required; existing evaluation directories are protected")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_id = args.model_id

    model_slug = model_id.split("/")[-1].lower()
    log_file = str(args.output_dir / f"{model_slug}_eval_arc_thesis.txt")
    json_file = log_file.replace(".txt", ".json")

    with open(log_file, "w", encoding="utf-8") as f:
        emit(f"Loading model: {model_id}", f)
        emit(f"Experiment 4.II — Priority 2 | {model_id} × ARC-Challenge", f)
        emit(f"MAX_SAMPLES={MAX_SAMPLES} | MAX_NEW_TOKENS={MAX_NEW_TOKENS}", f)
        emit(f"BATCH_SIZE={BATCH_SIZE}", f)
        emit(f"Approach configs:", f)
        for cfg in APPROACH_CONFIGS:
            emit(f"  {cfg}", f)

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        model.eval()

        # Gemma 2's embedding layer scales weights by sqrt(hidden_size).
        # Calling embed_layer(input_ids) applies this scale automatically,
        # but indexing .weight directly gives unscaled values.
        # We scale vocab_embs here so decode-step embeddings match prefill magnitude.
        embed_layer = model.get_input_embeddings()
        embed_scale = getattr(embed_layer, "embed_scale", 1.0)
        emit(f"embed_scale: {embed_scale}", f)

        vocab_embs = embed_layer.weight.detach().float() * embed_scale
        vocab_embs_norm = F.normalize(vocab_embs, dim=1)

        # Gemma 2 ends turns with <end_of_turn>, not <eos>
        eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        emit(f"<end_of_turn> token id: {eot_id}", f)

        emit("Loading ARC-Challenge...", f)
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")

        all_results: list[dict] = []
        for cfg in APPROACH_CONFIGS:
            result = evaluate_config(
                model,
                tokenizer,
                dataset,
                vocab_embs,
                vocab_embs_norm,
                cfg,
                eot_id,
                f,
            )
            all_results.append(result)

        # --- Summary table ---
        emit(f"\n{'='*70}", f)
        emit("SWEEP SUMMARY — Experiment 4.II Priority 2", f)
        emit(f"Model:     {model_id}", f)
        emit(f"Benchmark: ai2_arc / ARC-Challenge / test", f)
        emit(
            f"\n{'approach':<25} {'samp_temp':<12} {'top_p':<8} {'accuracy':<12} correct/total",
            f,
        )
        emit("-" * 70, f)
        for r in all_results:
            c = r["config"]
            emit(
                f"{c['name']:<25} "
                f"{str(c['sampling_temp']):<12} "
                f"{str(c['top_p']):<8} "
                f"{r['accuracy']:.4f}       "
                f"{r['correct']}/{r['total']}",
                f,
            )
        emit(f"{'='*70}", f)

    json_payload = {
        "experiment": "4.II",
        "priority": 2,
        "model": model_id,
        "benchmark": "ai2_arc/ARC-Challenge",
        "max_samples": MAX_SAMPLES,
        "runs": all_results,
    }
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(json_payload, jf, indent=2)


if __name__ == "__main__":
    main()
