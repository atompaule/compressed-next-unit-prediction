"""
GSM8K evaluation — accuracy across N test samples, optionally with a LoRA
checkpoint applied to the base model.

Methods
-------
  auto            : canonical thesis evaluation setting, inferred from the
                    training approach encoded in CKPT/CKPT_DIR:
                      base or discrete       -> discrete
                      gaussian_noise         -> soft_full
                      token_sampling         -> top_n_soft
                      dirichlet_sampling     -> top_n_soft
  discrete        : greedy discrete CoT. Feed back the embedding of the
                    highest-logit token at every reasoning step.
  soft_sampling   : matches rl_continuous_cot's token_sampling. Min-p nucleus,
                    sampled support and weighted sum. Retained as a diagnostic
                    override, but not used by the canonical thesis evaluation.
  sampled_token_subset
                  : explicit name for the same sampled-support policy.
  sampled_gaussian: full-vocabulary soft embedding plus isotropic Gaussian noise,
                    matching gaussian_noise training.
  sampled_dirichlet
                  : fixed top-n support with weights sampled from
                    Dir(gamma * p_norm), matching dirichlet_sampling training.
  soft_full       : full-vocab soft thinking. Weighted sum over the entire
                    softmax distribution (no nucleus filter), feed back.
  top_n_soft      : deterministic min-p support (topk-n), with renormalised
                    probabilities as weights. This is the deterministic
                    counterpart of token_sampling and the expected mixture
                    E[w] of dirichlet_sampling. No sampling.

Reports
-------
  - accuracy (fraction of samples where the boxed answer matches GT)
  - mean think length, mean answer length
  - per-sample line (correct? + think_len + answer_text)
  - a one-line SUMMARY for grep across runs

Usage
-----
  # Canonical setting is selected automatically from the checkpoint directory:
  CKPT_DIR=checkpoints/rl_cot_Llama-3.1-8B-Instruct_token_sampling_1e-05_1.0 \
  uv run python -m src.preliminary_experiments.reinforcement_learning.evaluate

  # Matched continuous base-model evaluations (CKPT=base explicitly ignores
  # any CKPT_DIR inherited from the shell):
  CKPT=base MODEL_ID=meta-llama/Llama-3.1-8B-Instruct METHOD=soft_full \
  uv run python -m src.preliminary_experiments.reinforcement_learning.evaluate
  CKPT=base MODEL_ID=meta-llama/Llama-3.1-8B-Instruct METHOD=top_n_soft \
  uv run python -m src.preliminary_experiments.reinforcement_learning.evaluate
  CKPT=base MODEL_ID=meta-llama/Llama-3.2-3B-Instruct METHOD=soft_full \
  uv run python -m src.preliminary_experiments.reinforcement_learning.evaluate
  CKPT=base MODEL_ID=meta-llama/Llama-3.2-3B-Instruct METHOD=top_n_soft \
  uv run python -m src.preliminary_experiments.reinforcement_learning.evaluate

  # Explicit diagnostic override:
  CKPT_DIR=checkpoints/rl_cot_Llama-3.1-8B-Instruct_gaussian_noise_1e-05_1.0 \
  METHOD=soft_full \
  uv run python -m src.preliminary_experiments.reinforcement_learning.evaluate
"""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import gc
import json
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from src.common.datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.preliminary_experiments.methods.soft_thinking import soft_thinking
from src.preliminary_experiments.methods.utils import emit
from src.preliminary_experiments.reinforcement_learning.gsm8k import get_reward, make_prompt

VALID_METHODS = (
    "auto",
    "discrete",
    "soft_sampling",
    "soft_full",
    "top_n_soft",
    "sampled_gaussian",
    "sampled_dirichlet",
    "sampled_token_subset",
)

STOCHASTIC_METHODS = {
    "soft_sampling",
    "sampled_gaussian",
    "sampled_dirichlet",
    "sampled_token_subset",
}

CANONICAL_INFERENCE_METHODS = {
    "base": "discrete",
    "discrete": "discrete",
    "gaussian_noise": "soft_full",
    "token_sampling": "top_n_soft",
    "dirichlet_sampling": "top_n_soft",
}


from src.preliminary_experiments.reinforcement_learning.checkpoints import (
    resolve_checkpoint, infer_training_approach, training_preset,
)


@dataclass
class Config:
    model_id: str = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    method: str = os.environ.get("METHOD", "auto")
    n_samples: int = int(os.environ.get("N", "2000"))
    temperature: float = float(os.environ.get("TEMP", "1.0"))
    max_think_steps: int = int(os.environ.get("MAX_THINK", "256"))
    max_answer_steps: int = int(os.environ.get("MAX_ANSWER", "128"))
    batch_size: int = int(os.environ.get("BATCH_SIZE", "8"))
    min_p: float = 0.03
    gaussian_scale: float = float(os.environ.get("GAUSSIAN_SCALE", "0.33"))
    dirichlet_gamma: float = float(os.environ.get("DIRICHLET_GAMMA", "4.0"))
    seed: int = int(os.environ.get("SEED", "42"))
    log_dir: str = os.environ.get("EVAL_LOG_DIR", "results")


CFG = Config()
assert (
    CFG.method in VALID_METHODS
), f"METHOD must be one of {VALID_METHODS}, got {CFG.method!r}"
if CFG.gaussian_scale < 0:
    raise ValueError("GAUSSIAN_SCALE must be non-negative")
if CFG.dirichlet_gamma <= 0:
    raise ValueError("DIRICHLET_GAMMA must be positive")

REQUESTED_METHOD = CFG.method
ckpt_label, ckpt_path = resolve_checkpoint(os.environ)
preset = training_preset(ckpt_path)
if preset is not None:
    if "MODEL_ID" in os.environ and CFG.model_id != preset["model_id"]:
        raise ValueError("MODEL_ID differs from the recorded adapter base model")
    CFG.model_id = preset["model_id"]
    CFG.min_p = preset["min_p"]
TRAINING_APPROACH = infer_training_approach(ckpt_path)
CANONICAL_METHOD = CANONICAL_INFERENCE_METHODS[TRAINING_APPROACH]
if CFG.method == "auto":
    CFG.method = CANONICAL_METHOD
elif TRAINING_APPROACH == "base":
    print(f"Evaluating base model with explicit METHOD={CFG.method!r}.")
elif CFG.method != CANONICAL_METHOD:
    print(
        f"Explicit METHOD={CFG.method!r} overrides the canonical "
        f"{CANONICAL_METHOD!r} setting for {TRAINING_APPROACH!r}."
    )


# ── Models ────────────────────────────────────────────────────────────────────

print(f"Loading base model: {CFG.model_id}...")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

base_model = AutoModelForCausalLM.from_pretrained(
    CFG.model_id,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map={"": device},
)
base_model.eval()
vocab_embs = base_model.get_input_embeddings().weight.detach()
embedding_rms = vocab_embs.pow(2).mean().sqrt()
gaussian_sigma = CFG.gaussian_scale * embedding_rms

if ckpt_path is not None:
    print(f"Loading LoRA adapter: {ckpt_path}")
    model = PeftModel.from_pretrained(base_model, ckpt_path, adapter_name=ckpt_label)
    model.eval()
else:
    print("No checkpoint — evaluating BASE model")
    model = base_model

print(f"Device: {device}")


# ── Trigger tokens (mirror the training script) ──────────────────────────────

_TRIGGER_STRINGS = ["answer", "Answer", " answer", " Answer", "\nanswer", "\nAnswer"]
_trigger_id_set: set[int] = set()
for _s in _TRIGGER_STRINGS:
    _ids = tokenizer.encode(_s, add_special_tokens=False)
    if _ids:
        _trigger_id_set.add(_ids[-1])
ANSWER_TRIGGER_IDS = torch.tensor(
    sorted(_trigger_id_set), dtype=torch.long, device=device
)

BOXED_STRING = "$\\boxed{"
ANSWER_STRING = "\nanswer: "

prefix_ids_threshold = tokenizer(
    BOXED_STRING, return_tensors="pt", add_special_tokens=False
).input_ids.to(device)
prefix_ids_maxsteps = tokenizer(
    ANSWER_STRING + BOXED_STRING, return_tensors="pt", add_special_tokens=False
).input_ids.to(device)


# ── Concept-embedding builders, one per soft method ──────────────────────────


def _safe_softmax(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """fp32 softmax with sanitisation (MPS bf16 can emit tiny negatives / NaNs)."""
    p = F.softmax(logits.float() / temp, dim=-1)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    return p / p.sum().clamp(min=1e-12)


def _soft_sampling_concept(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """
    Mirrors `build_concept_embedding(approach='token_sampling')` from
    rl_continuous_cot.py exactly, including the temperature handling:

      1. temperature-scaled probs → min-p nucleus → n = |{p_i >= min_p}|
      2. multinomial(probs_temp, n, replacement=False) selects the subset
      3. weighting = softmax(sampled_logits / temp) restricted to that subset

    Training uses temp in BOTH the nucleus/sampling step (1) and the weighting
    step (3) (see rl_continuous_cot.build_concept_embedding:
    `F.softmax(sampled_logits / temp)`). The model was trained at temp=1.0, but
    this path stays faithful for temp != 1.0 too.

    Implementation note: rather than materialising a [V, d] intermediate, we
    compute the restricted softmax directly over the n sampled rows.
    Mathematically identical, ~V/n times less memory.
    """
    # (1) Same nucleus / sampling distribution as training
    probs = _safe_softmax(logits, temp)
    n = max(int((probs >= CFG.min_p).sum().item()), 1)
    sample_ids = torch.multinomial(probs, n, replacement=False)

    # (2) Weighting: temperature-scaled softmax over the sampled logits (matches
    #     training's `F.softmax(sampled_logits / temp)`)
    sampled_logits = logits[sample_ids].float()
    sampled_weights = F.softmax(sampled_logits / temp, dim=-1)  # [n]

    # (3) Weighted sum of sampled token embeddings — only [n, d] intermediate
    sampled_embs = vocab_embs[sample_ids]  # [n, d]
    return (
        (sampled_weights.to(sampled_embs.dtype).unsqueeze(1) * sampled_embs)
        .sum(dim=0)
        .float()
    )


def _soft_full_concept(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """Full-vocab soft thinking: weighted sum over the entire softmax (no nucleus)."""
    probs = _safe_softmax(logits, temp)
    # Matrix-vector form: [V] @ [V, d] → [d]. No [V, d] intermediate.
    return (probs.to(vocab_embs.dtype) @ vocab_embs).float()


def _sampled_gaussian_concept(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """Full-soft embedding plus the isotropic noise used during training."""
    soft_vec = _soft_full_concept(logits, temp)
    noise = torch.randn_like(soft_vec) * gaussian_sigma
    return soft_vec + noise


def _sampled_dirichlet_concept(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """Sample mixture weights exactly as in dirichlet_sampling training."""
    probs = _safe_softmax(logits, temp)
    n = max(int((probs >= CFG.min_p).sum().item()), 1)
    sample_ids = torch.topk(probs, n).indices

    p_sub = probs[sample_ids]
    p_sub = p_sub / p_sub.sum().clamp(min=1e-12)
    alpha = (CFG.dirichlet_gamma * p_sub).clamp(min=1e-6)

    gamma_draws = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
    weights = (gamma_draws / gamma_draws.sum().clamp(min=1e-12)).clamp(min=1e-8)
    weights = weights / weights.sum()

    sampled_embs = vocab_embs[sample_ids]
    return (weights.to(sampled_embs.dtype) @ sampled_embs).float()


def _top_n_soft_concept(logits: torch.Tensor, temp: float) -> torch.Tensor:
    """
    Deterministic inference counterpart of `dirichlet_sampling` training.

    Under training, w ~ Dir(gamma * p_norm), with p_norm the renormalised nucleus
    probs. Its mean is E[w] = p_norm (gamma drops out — it only scales the width
    of the distribution around the mean). At inference we want the mean, which
    collapses to: take the topk-n nucleus, renormalise, weighted sum.

      n = |{p_i >= min_p}|
      sample_ids = topk(p, n).indices       # same support rule as training
      p_sub = p[sample_ids] / p[sample_ids].sum()
      concept_vec = p_sub @ E[sample_ids]   # [d]

    No Gamma draws, no `dirichlet_gamma`. Pair this with checkpoints trained via
    `dirichlet_sampling`.
    """
    probs = _safe_softmax(logits, temp)
    n = max(int((probs >= CFG.min_p).sum().item()), 1)
    sample_ids = torch.topk(probs, n).indices  # [n] nucleus token ids

    p_sub = probs[sample_ids]
    p_sub = p_sub / p_sub.sum().clamp(min=1e-12)

    sampled_embs = vocab_embs[sample_ids]  # [n, d]
    return (p_sub.to(sampled_embs.dtype) @ sampled_embs).float()


def _discrete_concept(token_id: int) -> torch.Tensor:
    """Return the embedding of a greedily selected discrete token."""
    return vocab_embs[token_id].float()


# ── Rollouts ─────────────────────────────────────────────────────────────────


def _iter_layer_kv(past_kv):
    for layer in past_kv.layers:
        yield layer.keys, layer.values


def _slice_past_kv(past_kv, b: int) -> DynamicCache:
    sliced = [
        (k[b : b + 1].contiguous(), v[b : b + 1].contiguous())
        for k, v in _iter_layer_kv(past_kv)
    ]
    new = DynamicCache()
    for layer_idx, (k, v) in enumerate(sliced):
        new.update(k, v, layer_idx)
    return new


@torch.no_grad()
def rollout_batch(prompts: list[str]) -> list[dict]:
    B = len(prompts)

    # Left-pad so per-row real-token positions align across the batch.
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        tok = tokenizer(
            prompts, padding=True, return_tensors="pt", add_special_tokens=False
        )
    finally:
        tokenizer.padding_side = prev_side

    prompt_ids = tok.input_ids.to(device)  # [B, max_P]
    prompt_attn = tok.attention_mask.to(device)  # [B, max_P]

    # Derive position_ids from the left-padded mask so rotary positions on real
    # tokens are independent of how much padding precedes them.
    position_ids = prompt_attn.long().cumsum(-1) - 1
    position_ids.masked_fill_(prompt_attn == 0, 1)

    embed = model.get_input_embeddings()
    prompt_embeds = embed(prompt_ids).to(model.dtype)

    out = model(
        inputs_embeds=prompt_embeds,
        attention_mask=prompt_attn,
        position_ids=position_ids,
        use_cache=True,
    )
    past_kv = out.past_key_values
    logits = out.logits[:, -1, :].float()  # [B, V]
    attn_mask = prompt_attn  # grows by 1 per think step

    d = vocab_embs.size(1)

    argmax_ids_lists: list[list[int]] = [[] for _ in range(B)]
    trigger_probs_lists: list[list[float]] = [[] for _ in range(B)]
    stop_reasons: list[str] = ["max_steps"] * B
    think_steps_list: list[int] = [0] * B
    active: list[bool] = [True] * B
    saved_past_kv: list[DynamicCache | None] = [None] * B
    saved_attn: list[torch.Tensor | None] = [None] * B

    for _ in range(CFG.max_think_steps):
        probs_t = F.softmax(logits, dim=-1)  # [B, V]
        trig_p_t = probs_t[:, ANSWER_TRIGGER_IDS].sum(dim=-1)  # [B]
        argmax_ids_t = logits.argmax(dim=-1)  # [B]

        # Single sync to CPU for all per-row decisions this step.
        synced = torch.stack([trig_p_t, argmax_ids_t.to(trig_p_t.dtype)]).cpu().tolist()
        trig_p_vals = synced[0]
        argmax_id_vals = [int(x) for x in synced[1]]

        concept_batch = torch.zeros(B, 1, d, device=device, dtype=model.dtype)

        for b in range(B):
            if not active[b]:
                continue

            trigger_probs_lists[b].append(trig_p_vals[b])

            if CFG.method == "discrete":
                selected_id = argmax_id_vals[b]
                concept_vec = _discrete_concept(selected_id)
                if selected_id in _trigger_id_set:
                    stop_reasons[b] = "threshold"
                    saved_past_kv[b] = _slice_past_kv(past_kv, b)
                    saved_attn[b] = attn_mask[b].clone()
                    active[b] = False
                    continue
                if selected_id == tokenizer.eos_token_id:
                    stop_reasons[b] = "eos"
                    saved_past_kv[b] = _slice_past_kv(past_kv, b)
                    saved_attn[b] = attn_mask[b].clone()
                    active[b] = False
                    continue
                argmax_ids_lists[b].append(selected_id)
            else:
                window_sum = sum(trigger_probs_lists[b][-3:])
                if (argmax_id_vals[b] in _trigger_id_set) or (window_sum > 0.8):
                    stop_reasons[b] = "threshold"
                    saved_past_kv[b] = _slice_past_kv(past_kv, b)
                    saved_attn[b] = attn_mask[b].clone()
                    active[b] = False
                    continue
                argmax_ids_lists[b].append(argmax_id_vals[b])
                if CFG.method in ("soft_sampling", "sampled_token_subset"):
                    concept_vec = _soft_sampling_concept(logits[b], CFG.temperature)
                elif CFG.method == "sampled_gaussian":
                    concept_vec = _sampled_gaussian_concept(logits[b], CFG.temperature)
                elif CFG.method == "sampled_dirichlet":
                    concept_vec = _sampled_dirichlet_concept(logits[b], CFG.temperature)
                elif CFG.method == "top_n_soft":
                    concept_vec = _top_n_soft_concept(logits[b], CFG.temperature)
                elif CFG.method == "soft_full":
                    concept_vec = _soft_full_concept(logits[b], CFG.temperature)
                else:
                    raise RuntimeError(f"Unsupported continuous method: {CFG.method}")

            concept_batch[b, 0] = concept_vec.to(model.dtype)
            think_steps_list[b] += 1

        if not any(active):
            break

        # Extend mask by one position; inactive rows still get a (wasted) 1 —
        # same trade-off as training (unavoidable for batched processing).
        attn_mask = torch.cat(
            [attn_mask, torch.ones(B, 1, device=device, dtype=attn_mask.dtype)],
            dim=1,
        )
        step_position_ids = attn_mask.long().sum(-1, keepdim=True) - 1  # [B, 1]

        out = model(
            inputs_embeds=concept_batch,
            past_key_values=past_kv,
            attention_mask=attn_mask,
            position_ids=step_position_ids,
            use_cache=True,
        )
        past_kv = out.past_key_values
        logits = out.logits[:, -1, :].float()

    # Save state for rows still active (hit max_think_steps).
    for b in range(B):
        if active[b]:
            stop_reasons[b] = "max_steps"
            saved_past_kv[b] = _slice_past_kv(past_kv, b)
            saved_attn[b] = attn_mask[b].clone()
            active[b] = False

    del past_kv, logits, out, attn_mask

    # Answer phase — per row, since saved cache lengths differ.
    results: list[dict] = []
    for b in range(B):
        prefix_ids = (
            prefix_ids_threshold
            if stop_reasons[b] in ("threshold", "eos")
            else prefix_ids_maxsteps
        )
        prefix_len = prefix_ids.shape[1]
        prefix_attn = torch.cat(
            [
                saved_attn[b].unsqueeze(0),
                torch.ones(1, prefix_len, device=device, dtype=saved_attn[b].dtype),
            ],
            dim=1,
        )
        prefix_start = int(saved_attn[b].long().sum().item())
        prefix_position_ids = torch.arange(
            prefix_start, prefix_start + prefix_len, device=device, dtype=torch.long
        ).unsqueeze(0)

        out = model(
            input_ids=prefix_ids,
            past_key_values=saved_past_kv[b],
            attention_mask=prefix_attn,
            position_ids=prefix_position_ids,
            use_cache=True,
        )
        past_kv_ans = out.past_key_values
        logits_ans = out.logits[0, -1, :].float()
        attn_ans = prefix_attn

        gen_ids: list[int] = []
        current = ""
        for _ in range(CFG.max_answer_steps):
            nxt = int(logits_ans.argmax(dim=-1).item())
            current += tokenizer.decode(nxt, skip_special_tokens=True)
            gen_ids.append(nxt)
            if nxt == tokenizer.eos_token_id or "}" in current or "\n\n" in current:
                break
            attn_ans = torch.cat(
                [attn_ans, torch.ones(1, 1, device=device, dtype=attn_ans.dtype)],
                dim=1,
            )
            next_position_ids = attn_ans.long().sum(-1, keepdim=True) - 1
            emb = embed(torch.tensor([[nxt]], device=device))
            out = model(
                inputs_embeds=emb,
                past_key_values=past_kv_ans,
                attention_mask=attn_ans,
                position_ids=next_position_ids,
                use_cache=True,
            )
            past_kv_ans = out.past_key_values
            logits_ans = out.logits[0, -1, :].float()

        results.append(
            {
                "think_steps": think_steps_list[b],
                "stop_reason": stop_reasons[b],
                "argmax_ids": argmax_ids_lists[b],
                "answer_text": BOXED_STRING
                + tokenizer.decode(gen_ids, skip_special_tokens=True),
                "answer_steps": len(gen_ids),
            }
        )

    return results


def _free_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────


print("Loading GSM8K test split...")
test_ds = load_dataset("gsm8k", "main", split="test")
N = min(CFG.n_samples, len(test_ds))

# Reset after model/adapter and dataset loading so every stochastic evaluation
# starts from the requested RNG state regardless of setup-time RNG consumption.
torch.manual_seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)

os.makedirs(CFG.log_dir, exist_ok=True)
ckpt_tag = ckpt_label or "base"

# Reflect the full model config in the filename, named exactly like the
# checkpoint dirs (e.g. "rl_cot_Llama-3.1-8B-Instruct_token_sampling_1e-05_1.0").
# A checkpoint path is ".../<exp_name>/step_XXXXX", so the experiment dir is the
# parent of the resolved checkpoint. The baseline (no adapter) falls back to the
# base model slug. Temperature is included so runs at different temps don't clash.
if ckpt_path is not None:
    exp_name = os.path.basename(os.path.dirname(os.path.normpath(ckpt_path)))
else:
    exp_name = CFG.model_id.split("/")[-1]

seed_tag = f"_seed{CFG.seed}" if CFG.method in STOCHASTIC_METHODS else ""
run_tag = f"{exp_name}_{ckpt_tag}_{CFG.method}_temp{CFG.temperature}_N{N}{seed_tag}"
log_file = os.path.join(CFG.log_dir, f"gsm8k_eval_{run_tag}.txt")
print(f"Writing log → {log_file}")

n_correct = 0
think_lens: list[int] = []
ans_lens: list[int] = []
corrects: list[int] = []
stop_reason_counts: dict[str, int] = {}
t_start = time.perf_counter()

with open(log_file, "w", encoding="utf-8") as f:
    emit(
        "GSM8K EVALUATION — base / checkpoint × "
        "{discrete, soft_full, top_n_soft, sampled Gaussian/Dirichlet/token subset}",
        f,
    )
    emit(f"Base model     : {CFG.model_id}", f)
    emit(f"Checkpoint     : {ckpt_path or '(none — baseline)'}", f)
    emit(f"Training mode  : {TRAINING_APPROACH}", f)
    emit(f"Requested mode : {REQUESTED_METHOD}", f)
    emit(f"Inference mode : {CFG.method}", f)
    emit(f"Temperature    : {CFG.temperature}", f)
    emit(f"Random seed    : {CFG.seed}", f)
    emit(f"Gaussian scale : {CFG.gaussian_scale}", f)
    emit(f"Gaussian sigma : {gaussian_sigma.item():.8f}", f)
    emit(f"Dirichlet gamma: {CFG.dirichlet_gamma}", f)
    emit(f"Max think steps: {CFG.max_think_steps}", f)
    emit(f"Batch size     : {CFG.batch_size}", f)
    emit(f"N samples      : {N}", f)
    emit("-" * 80, f)

    for start in range(0, N, CFG.batch_size):
        end = min(start + CFG.batch_size, N)
        batch_samples = [test_ds[j] for j in range(start, end)]
        prompts = [make_prompt(s["question"], tokenizer) for s in batch_samples]

        batch_results = rollout_batch(prompts)

        for offset, (sample, r) in enumerate(zip(batch_samples, batch_results)):
            global_i = start + offset
            reward = get_reward(r["answer_text"], sample["answer"])
            n_correct += int(reward)
            think_lens.append(r["think_steps"])
            ans_lens.append(r["answer_steps"])
            corrects.append(int(reward))
            stop_reason_counts[r["stop_reason"]] = (
                stop_reason_counts.get(r["stop_reason"], 0) + 1
            )

            think_text = tokenizer.decode(r["argmax_ids"], skip_special_tokens=True)
            emit(f"\n---- sample {global_i} ----", f)
            emit(f"  correct      : {int(reward)}", f)
            emit(f"  think_steps  : {r['think_steps']}", f)
            emit(f"  answer_steps : {r['answer_steps']}", f)
            emit(f"  stop_reason  : {r['stop_reason']}", f)
            emit(f"  question     : {sample['question']!r}", f)
            emit(f"  gt_answer    : {sample['answer']!r}", f)
            emit(f"  think        : {think_text!r}", f)
            emit(f"  answer       : {r['answer_text']!r}", f)

        elapsed = time.perf_counter() - t_start
        acc = n_correct / end
        secs_per_sample = elapsed / end
        eta_secs = secs_per_sample * (N - end)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_secs))
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print(
            f"  [{end:>3}/{N}]  acc={acc:.3f}  "
            f"avg_think={sum(think_lens)/len(think_lens):.1f}  "
            f"elapsed={elapsed_str}  "
            f"eta={eta_str}  "
            f"({secs_per_sample:.1f}s/sample)"
        )

        _free_cache()

    accuracy = n_correct / N
    mean_think = sum(think_lens) / len(think_lens)
    mean_ans = sum(ans_lens) / len(ans_lens)
    elapsed = time.perf_counter() - t_start

    emit("\n" + "=" * 80, f)
    emit(f"  Accuracy        : {accuracy:.4f}  ({n_correct}/{N})", f)
    emit(f"  Mean think len  : {mean_think:.2f}", f)
    emit(f"  Mean answer len : {mean_ans:.2f}", f)
    emit(f"  Total time      : {elapsed:.1f}s", f)

    # Grep-friendly summary line for cross-checkpoint sweeps.
    emit(
        f"\nSUMMARY method={CFG.method} checkpoint={ckpt_tag} "
        f"n={N} acc={accuracy:.4f} mean_think={mean_think:.2f} "
        f"mean_ans={mean_ans:.2f} temp={CFG.temperature}",
        f,
    )


# ── Machine-readable summary — written next to the .txt log so eval runs can be
#    pooled / plotted across checkpoints, methods and temperatures. ───────────
def _std(xs: list[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# Wald 95% CI on accuracy (handy because small-N eval is noisy).
acc_se = (accuracy * (1.0 - accuracy) / N) ** 0.5 if N else 0.0

try:
    step_num = int(ckpt_label.split("_")[1]) if ckpt_label else None
except (IndexError, ValueError):
    step_num = None

summary_json = {
    "experiment": exp_name,
    "method": CFG.method,
    "requested_method": REQUESTED_METHOD,
    "training_approach": TRAINING_APPROACH,
    "canonical_inference_method": CANONICAL_METHOD,
    "model_id": CFG.model_id,
    "checkpoint": ckpt_tag,
    "checkpoint_path": ckpt_path,
    "step": step_num,
    "temperature": CFG.temperature,
    "min_p": CFG.min_p,
    "seed": CFG.seed,
    "gaussian_scale": CFG.gaussian_scale,
    "gaussian_sigma": gaussian_sigma.item(),
    "dirichlet_gamma": CFG.dirichlet_gamma,
    "max_think_steps": CFG.max_think_steps,
    "max_answer_steps": CFG.max_answer_steps,
    "n_samples": N,
    "n_correct": n_correct,
    "accuracy": accuracy,
    "accuracy_se": acc_se,
    "accuracy_ci95": [accuracy - 1.96 * acc_se, accuracy + 1.96 * acc_se],
    "mean_think_len": mean_think,
    "std_think_len": _std([float(x) for x in think_lens], mean_think),
    "mean_answer_len": mean_ans,
    "std_answer_len": _std([float(x) for x in ans_lens], mean_ans),
    "stop_reason_counts": stop_reason_counts,
    "elapsed_seconds": elapsed,
    # Per-sample arrays so downstream tooling can recompute / bootstrap.
    "per_sample": {
        "correct": corrects,
        "think_len": think_lens,
        "answer_len": ans_lens,
    },
}

json_file = os.path.splitext(log_file)[0] + ".json"
with open(json_file, "w", encoding="utf-8") as jf:
    json.dump(summary_json, jf, indent=2)

print(f"\nDone → {log_file}")
print(f"Saved JSON → {json_file}")
