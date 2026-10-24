"""
eval_winogrande_latent.py

Experiment 4.II — Priority 3 runs: LLaMA 3.1 8B IT × WinoGrande
Compares Soft Thinking, Noisy Greedy Discrete (Isotropic Gaussian),
Greedy Discrete, and Default Baseline (temperature sampling).

One combined log + JSON for all 4 approaches.
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

from src.preliminary_experiments.methods.config import CFG
from src.preliminary_experiments.soft_thinking.mixture_protocol import (
    benchmark_soft_thinking as soft_thinking, thesis_approach_configs,
)
from src.preliminary_experiments.methods.random_perturbation import noisy_target_sim
from src.preliminary_experiments.methods.utils import emit

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MAX_SAMPLES = 10000  # WinoGrande-XL validation split has 1267 items
MAX_NEW_TOKENS = 1024  # model reasons before giving final answer

APPROACH_CONFIGS = thesis_approach_configs()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# WinoGrande-specific utils
# ---------------------------------------------------------------------------


def build_prompt(sentence: str, option1: str, option2: str) -> str:
    """Format a fill-in-the-blank prompt for WinoGrande."""
    return (
        "Choose the correct option to fill in the blank (_) in the following sentence. "
        "First reason through it under 'reasoning: ', then give your final answer "
        "under 'answer: ' using exactly the format: answer: 1 or answer: 2.\n\n"
        f"Sentence: {sentence}\n"
        f"Option 1: {option1}\n"
        f"Option 2: {option2}\n\n"
        "reasoning:"
    )


def extract_choice(text: str) -> str | None:
    # Require colon to avoid matching reasoning prose
    matches = re.findall(r"answer\s*:\s*([12])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1]  # take last in case model self-corrects

    return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_p: float = 1.0,
) -> int:
    """Greedy argmax if temperature == 0, else temperature + nucleus sampling."""
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
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    config: dict,
    pc_cache: dict | None = None,
) -> list[str]:
    """Batched autoregressive generation with custom embedding logic."""
    approach = config["name"]
    sampling_temp = config["sampling_temp"]
    top_p = config.get("top_p", 1.0)

    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()
    B = len(prompts)

    # Build chat-formatted strings for each prompt
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,  # get string, not token ids
            enable_thinking=False,
        )
        for p in prompts
    ]

    # Tokenize with left-padding so the last token aligns across the batch
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(formatted, return_tensors="pt", padding=True).to(device)

    input_ids = enc["input_ids"]  # [B, L]
    attention_mask = enc["attention_mask"]  # [B, L]

    inputs_embeds = embed_layer(input_ids)  # [B, L, d]

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :].float()  # [B, V]

    generated = [""] * B
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(MAX_NEW_TOKENS):
        if finished.all():
            break

        next_vecs = []
        fed_ids = []

        for b in range(B):
            if finished[b]:
                # Feed a dummy pad embedding — output ignored
                next_vecs.append(vocab_embs[tokenizer.eos_token_id])
                fed_ids.append(tokenizer.eos_token_id)
                continue

            b_logits = logits[b]  # [V]
            greedy_id = sample_token(b_logits, sampling_temp, top_p)

            if approach in ("discrete_top1", "default_baseline"):
                next_vec = vocab_embs[greedy_id]
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

            if fed_id == tokenizer.eos_token_id or re.search(
                r"answer\s*[:\-]?\s*[12]\b", generated[b], re.IGNORECASE
            ):
                finished[b] = True

            next_vecs.append(next_vec)
            fed_ids.append(fed_id)

        # Stack and feed back: [B, 1, d]
        next_embeds = torch.stack(next_vecs, dim=0).to(device=device, dtype=model.dtype)
        next_embeds = next_embeds.unsqueeze(1)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=next_embeds,
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
    model,
    tokenizer,
    dataset,
    vocab_embs: torch.Tensor,
    vocab_embs_norm: torch.Tensor,
    config: dict,
    f,
) -> dict:
    """Run pass@1 evaluation for one config. Returns a results dict."""
    emit(f"\n{'#'*70}", f)
    emit(f"CONFIG: {config}", f)
    emit(f"{'#'*70}", f)

    correct, total = 0, 0
    results: list[dict] = []
    samples = list(dataset)[:MAX_SAMPLES]
    n_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE
    pc_cache: dict = {}

    for batch_idx, batch_start in enumerate(
        tqdm(range(0, len(samples), BATCH_SIZE), desc=config["name"])
    ):
        t0 = time.time()
        batch = samples[batch_start : batch_start + BATCH_SIZE]

        prompts = [
            build_prompt(s["sentence"], s["option1"], s["option2"]) for s in batch
        ]
        gt_keys = [s["answer"] for s in batch]

        pred_texts = generate_batch(
            model, tokenizer, prompts, vocab_embs, vocab_embs_norm, config, pc_cache
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
            pred_answer = extract_choice(pred_text)
            is_correct = pred_answer is not None and pred_answer == gt_key
            correct += int(is_correct)
            total += 1

            emit(f"\n{'='*60}", f)
            emit(f"[{global_i+1}] Sentence: {sample['sentence']}", f)
            emit(f"Option 1: {sample['option1']} | Option 2: {sample['option2']}", f)
            emit(f"GT: {gt_key}", f)
            emit(pred_text, f)
            emit(f"Pred: {pred_answer} | {'✅' if is_correct else '❌'}", f)

            results.append(
                {
                    "index": global_i + 1,
                    "gt": gt_key,
                    "pred": pred_answer,
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
        default=CFG.model_id,
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
    log_file = str(args.output_dir / f"{model_slug}_eval_winogrande_thesis.txt")
    json_file = log_file.replace(".txt", ".json")

    with open(log_file, "w", encoding="utf-8") as f:
        emit(f"Loading model: {model_id}", f)
        emit(f"Experiment 4.II — Priority 3 | {model_id} × WinoGrande-XL", f)
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

        vocab_embs = model.get_input_embeddings().weight.detach().float()
        vocab_embs_norm = F.normalize(vocab_embs, dim=1)

        emit("Loading WinoGrande-XL (validation)...", f)
        dataset = load_dataset(
            "winogrande", "winogrande_xl", split="validation", trust_remote_code=True
        )

        # --- Sweep over all approach configs ---
        all_results: list[dict] = []
        for cfg in APPROACH_CONFIGS:
            result = evaluate_config(
                model,
                tokenizer,
                dataset,
                vocab_embs,
                vocab_embs_norm,
                cfg,
                f,
            )
            all_results.append(result)

        # --- Summary table ---
        emit(f"\n{'='*70}", f)
        emit("SWEEP SUMMARY — Experiment 4.II Priority 3", f)
        emit(f"Model:     {model_id}", f)
        emit(f"Benchmark: winogrande / winogrande_xl / validation", f)
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

    # --- Write combined JSON ---
    json_payload = {
        "experiment": "4.II",
        "priority": 3,
        "model": model_id,
        "benchmark": "winogrande/winogrande_xl/validation",
        "max_samples": MAX_SAMPLES,
        "runs": all_results,
    }
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(json_payload, jf, indent=2)


if __name__ == "__main__":
    main()
