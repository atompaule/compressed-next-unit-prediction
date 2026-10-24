
# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import json
import os
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import bfloat16
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from src.preliminary_experiments.reinforcement_learning.config import CFG, MODELS, APPROACHES
from src.preliminary_experiments.reinforcement_learning.gsm8k import (
    load_training_dataset,
    get_reward,
    make_prompt,
)
from src.preliminary_experiments.reinforcement_learning.utils import (
    _latest_checkpoint,
    format_duration,
    log,
    log_rollouts,
)

def initialize(output_dir, resume=False):
    """Initialize an explicitly selected run; importing this module does not train."""
    global OUTPUT_DIR, RESUME_FROM, dataset
    global ANSWER_STRING, ANSWER_TRIGGER_IDS, BOXED_STRING, DATA_SEED, LOG_EVERY, LOG_FILE, METRICS_FILE, SAVE_EVERY
    global _TRIGGER_STRINGS, _trigger_id_set, base_model, device, lora_cfg, metrics_log, model, model_embeddings
    global optimizer, prefix_ids_maxsteps, prefix_ids_threshold, rms_norm, seq_log, sigma, start_step, state
    global state_path, step_label, tokenizer, vocab_embeds_device

    from dataclasses import asdict
    output = _Path(output_dir).resolve()
    specification = {"dataset": "gsm8k/main/train", "data_seed": 1234,
                     "preset": asdict(CFG), "step_indexing": "completed batches, one-based"}
    if resume:
        saved = json.loads((output / "run_config.json").read_text())
        if saved != json.loads(json.dumps(specification)):
            raise ValueError("Cannot resume with a different dataset or configuration")
        RESUME_FROM = _latest_checkpoint(str(output))
        if RESUME_FROM is None:
            raise ValueError("No checkpoint to resume")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "run_config.json").write_text(json.dumps(specification, indent=2) + "\n")
        RESUME_FROM = None
    OUTPUT_DIR = str(output)
    dataset = load_training_dataset()
    step_label = os.path.basename(RESUME_FROM) if RESUME_FROM else "step_00000"

    LOG_FILE = f"{OUTPUT_DIR}/{step_label}/log_{step_label}.txt"
    METRICS_FILE = f"{OUTPUT_DIR}/{step_label}/metrics_{step_label}.jsonl"

    LOG_EVERY = 1
    SAVE_EVERY = 100  # Save only after complete four-batch accumulation groups.

    DATA_SEED = 1234

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    seq_log = open(LOG_FILE, "w", encoding="utf-8")
    metrics_log = open(METRICS_FILE, "a", encoding="utf-8")

    log(seq_log, f"Loading {CFG.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        CFG.model_id,
        device_map="auto",
        torch_dtype=bfloat16,
    )

    # how lora works
    # W_eff = W_frozen + (alpha / r) * B @ A, whereas A.shape = (r, d_in) and B.shape = (d_out, r)
    # out = W_eff @ in = (W_frozen + (alpha / r) * B @ A) * in = W_frozen @ in + (alpha / r) * B @ (A @ in)

    if RESUME_FROM:
        model = PeftModel.from_pretrained(base_model, RESUME_FROM, is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=CFG.rl_config.lora_r,  # number of cols in lora matrices
            lora_alpha=CFG.rl_config.lora_alpha,  # scalar that determines relative influence of the lora matrices on frozen base
            target_modules=CFG.rl_config.lora_targets,  # fine-tune all projections of the model, except for embedding matrices
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(base_model, lora_cfg)

    device = model.device
    log(seq_log, f"Using {device}")

    model_embeddings = model.get_input_embeddings()
    vocab_embeds_device = model_embeddings.weight.detach()

    # these are only needed for the gaussian_noise approach, but compute them here for efficiency
    rms_norm = vocab_embeds_device.pow(2).mean().sqrt()
    sigma = 0.33 * rms_norm

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CFG.rl_config.learning_rate,
        weight_decay=0.01,
    )

    start_step = 0
    if RESUME_FROM:
        state_path = os.path.join(RESUME_FROM, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng_state"])
            start_step = state["step"]
            log(seq_log, f"Resumed after batch {start_step} ({RESUME_FROM})")
        else:
            raise FileNotFoundError(f"Cannot resume without {state_path}")


    _TRIGGER_STRINGS = [
        "answer",
        "Answer",
        " answer",
        " Answer",
        "\nanswer",
        "\nAnswer",
    ]
    _trigger_id_set = set()

    for _s in _TRIGGER_STRINGS:
        # We take the *last* token of the encoded string to bypass any strange prefix merging
        _ids = tokenizer.encode(_s, add_special_tokens=False)
        if _ids:
            _trigger_id_set.add(_ids[-1])

    ANSWER_TRIGGER_IDS = torch.tensor(
        sorted(_trigger_id_set), dtype=torch.long, device=device
    )

    ANSWER_STRING = "answer: "
    BOXED_STRING = "$\\boxed{"

    prefix_ids_threshold = tokenizer(
        ANSWER_STRING + BOXED_STRING, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prefix_ids_maxsteps = tokenizer(
        "\n" + ANSWER_STRING + BOXED_STRING, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)


def build_concept_embedding(
    logits: torch.Tensor,
    temp: float,
    approach: str,
    probs: torch.Tensor | None = None,
):
    concept_vec, sample_ids, sample_probs = None, None, None

    if probs is None:
        probs = F.softmax(logits / temp, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        probs = probs / probs.sum().clamp(min=1e-12)

    if approach == "token_sampling":
        n = max(int((probs >= CFG.min_p).sum().item()), 1)

        # note that i.i.d (replacement=True) would be needed to make log pi a trivial sum, but i violate this intentionally and account for it later in policy_log_prob
        sample_ids = torch.multinomial(probs, n, replacement=False)
        sample_probs = probs[sample_ids]

        # softmax over only the n sampled tokens
        sampled_logits = logits[sample_ids].float()
        sampled_weights = F.softmax(sampled_logits / temp, dim=-1)  # [n]
        sampled_embs = vocab_embeds_device[sample_ids]  # [n, d]
        concept_vec = (sampled_weights.to(sampled_embs.dtype) @ sampled_embs).float()
    elif approach == "gaussian_noise":
        # butt et al.'s approach: soft thinking over full vocab
        soft_vec = (probs.to(vocab_embeds_device.dtype) @ vocab_embeds_device).float()
        eps = torch.randn_like(soft_vec) * sigma
        concept_vec = soft_vec + eps
    elif approach == "discrete":
        # standard discrete CoT: sample one token, feed its embedding back
        sample_ids = torch.multinomial(probs, 1, replacement=False)  # [1]
        sample_probs = probs[sample_ids]  # [1]
        concept_vec = vocab_embeds_device[sample_ids[0]]  # [d]
    elif approach == "dirichlet_sampling":
        # Dirichlet-sampled weights -> soft thinking using those as factors
        # using only the min_p embeddings (as in token_sampling)
        # sampling w ~ Dir(gamma * p_renorm)

        n = max(int((probs >= CFG.min_p).sum().item()), 1)
        sample_ids = torch.topk(probs, n).indices  # [n] nucleus token ids

        p_sub = probs[sample_ids]
        p_sub = p_sub / p_sub.sum().clamp(min=1e-12)
        # gamma scales the concentration: directly using p would push samples to
        # the corners (near one-hot); gamma > 1 keeps them soft around the mean.
        alpha = (CFG.rl_config.dirichlet_gamma * p_sub).clamp(min=1e-6)

        # sample via Gamma: w_i = g_i / sum_j g_j
        g = torch.distributions.Gamma(alpha, torch.ones_like(alpha)).sample()
        w = (g / g.sum().clamp(min=1e-12)).clamp(min=1e-8)
        w = w / w.sum()  # valid simplex point; stored for the density at update time
        sample_probs = w

        sampled_embs = vocab_embeds_device[sample_ids]  # [n, d]
        concept_vec = (w.to(sampled_embs.dtype) @ sampled_embs).float()

    return concept_vec, sample_ids, sample_probs


@torch.no_grad()
def rollout_single_answer_sequence(
    past_kv_ans: object, logits: torch.Tensor, attn_mask: torch.Tensor
):
    gen_ids = []
    current_text = ""

    max_tokens = CFG.rl_config.max_tokens
    for _ in range(max_tokens):
        # TODO: should we specify a temperature != 1.0 exlusively for answer generation?
        probs = F.softmax(logits, dim=-1)
        next_id_t = torch.multinomial(probs, 1)
        next_id = next_id_t.item()

        current_text += tokenizer.decode(next_id, skip_special_tokens=True)
        gen_ids.append(next_id)

        if (
            next_id == tokenizer.eos_token_id
            or "}" in current_text
            or "\n\n" in current_text
        ):
            break

        # attention mask expanded for next token
        attn_mask = torch.cat(
            [attn_mask, torch.ones(1, 1, device=device, dtype=attn_mask.dtype)],
            dim=1,
        )

        next_embed = model_embeddings(next_id_t.unsqueeze(0))
        next_position_ids = attn_mask.long().sum(-1, keepdim=True) - 1  # [1, 1]
        out = model(
            inputs_embeds=next_embed,
            past_key_values=past_kv_ans,
            attention_mask=attn_mask,
            position_ids=next_position_ids,
            use_cache=True,
        )
        past_kv_ans = out.past_key_values
        logits = out.logits[0, -1, :].float()

    return gen_ids


def _iter_layer_kv(past_kv):
    # generator function to generate kv cache layer-wise
    layers = past_kv.layers
    for layer in layers:
        yield layer.keys, layer.values
    return


def _slice_past_kv(past_kv, g: int):
    # returns new DynamicCache containing kv of only rollout g from the batched kv
    sliced_per_layer = [
        (k[g : g + 1].contiguous(), v[g : g + 1].contiguous())
        for k, v in _iter_layer_kv(past_kv)
    ]  # [L, 2, 1, n_heads, seq, head_dim]

    new = DynamicCache()
    for layer_idx, (k, v) in enumerate(sliced_per_layer):
        new.update(k, v, layer_idx)
    return new


@torch.no_grad()
def rollout_batch(
    prompts: list[str], temp: float, approach: str, G: int
) -> list[list[dict]]:
    n_prompts = len(prompts)
    n_batch = n_prompts * G

    ### --- building the prompt token ids and attention masks ---

    # temporary left padding to allow equal indexing during batched rollout
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    prompts_tokenized_left_padded = tokenizer(
        prompts, padding=True, return_tensors="pt", add_special_tokens=False
    )
    tokenizer.padding_side = prev_side

    # get token ids and attention masks for all prompts of the batch
    prompt_ids = prompts_tokenized_left_padded.input_ids.to(
        device
    )  # [n_prompts, max_prompt_len]
    prompt_attn = prompts_tokenized_left_padded.attention_mask.to(
        device
    )  # [n_prompts, max_prompt_len]
    # representing each prompt's true length
    prompt_lens = prompt_attn.sum(dim=-1).cpu().tolist()  # [n_prompts]
    max_prompt_len = prompt_ids.size(1)

    ### --- preparing the batched prompt token ids and attention masks ---

    # repeat each prompt G times to get n_batch = n_prompts*G rows, sets of G consecutive prompts will be identical
    prompt_ids_batch = prompt_ids.repeat_interleave(G, dim=0)
    prompt_attn_batch = prompt_attn.repeat_interleave(G, dim=0)

    ### --- building position_ids such that they ignore left-padded masks (critical for correct rotary position transformation later) ---

    # derive position_ids from the left-padded attention mask so rotary position transformations
    # are independent of the left padding
    # necessary when using model() directly instead of model.generate()
    # start with 0 at the first real token
    position_ids_batch = prompt_attn_batch.long().cumsum(-1) - 1
    # note: make it 1s for padding (doesn't matter, as attention will be 0 there)
    position_ids_batch.masked_fill_(prompt_attn_batch == 0, 1)

    ### --- initial forward: get kv caches, logits, attention masks for all prompts of the batch, using model() ---

    prompt_embeds_batch = model_embeddings(prompt_ids_batch).to(model.dtype)

    # using model() instead of model.generate(), as we want just one parallel forward
    out_batch = model(
        inputs_embeds=prompt_embeds_batch,
        attention_mask=prompt_attn_batch,
        position_ids=position_ids_batch,  # pass these, because left-padding
        use_cache=True,
    )
    past_kv_batch = out_batch.past_key_values
    logits_batch_last = out_batch.logits[:, -1, :].float()  # [n_batch, n_vocab]
    attn_mask_batch = (
        prompt_attn_batch  # [n_batch, max_prompt_len]; will grow by 1 every think step
    )

    n_embed = vocab_embeds_device.size(1)

    ### --- batched rollout ---

    concept_vecs_lists: list[list] = [[] for _ in range(n_batch)]
    sample_ids_lists: list[list] = [[] for _ in range(n_batch)]
    sample_probs_lists: list[list] = [[] for _ in range(n_batch)]
    argmax_ids_lists: list[list] = [[] for _ in range(n_batch)]
    trigger_probs_last_sum_lists: list[list] = [[] for _ in range(n_batch)]
    trigger_probs_last_sum_window_lists: list[list] = [[] for _ in range(n_batch)]
    stop_reasons: list[str] = ["max_steps"] * n_batch
    active = [True] * n_batch
    saved_past_kv: list[DynamicCache] = [
        None
    ] * n_batch  # to store final kv per rollout
    saved_attn: list = [None] * n_batch  # attention mask matching saved_past_kv

    ### --- batched rollout: thinking ---

    max_tokens = CFG.rl_config.max_tokens
    for _ in range(max_tokens):
        probs_batch_last = F.softmax(logits_batch_last, dim=-1)  # [n_batch, n_vocab]
        argmax_ids_batch_last = logits_batch_last.argmax(dim=-1)  # [n_batch]
        trigger_probs_batch_last_sum = probs_batch_last[:, ANSWER_TRIGGER_IDS].sum(
            dim=-1
        )  # [n_batch]

        # init concept embeddings; simply zeros
        concept_embeds_batch_last = torch.zeros(
            n_batch, 1, n_embed, device=device, dtype=model.dtype
        )

        ### --- sequentially creating one concept embedding for each sequence ---
        for i_seq in range(n_batch):
            if not active[i_seq]:
                continue

            trigger_probs_last_sum_i = trigger_probs_batch_last_sum[i_seq].item()
            trigger_probs_last_sum_lists[i_seq].append(trigger_probs_last_sum_i)

            argmax_ids_last_i = argmax_ids_batch_last[i_seq].item()

            trigger_probs_last_sum_i_window = sum(
                trigger_probs_last_sum_lists[i_seq][-3:]
            )
            trigger_probs_last_sum_window_lists[i_seq].append(
                trigger_probs_last_sum_i_window
            )

            if (argmax_ids_last_i in _trigger_id_set) or (
                trigger_probs_last_sum_i_window > 0.8
            ):
                stop_reasons[i_seq] = "threshold"
                saved_past_kv[i_seq] = _slice_past_kv(past_kv_batch, i_seq)
                saved_attn[i_seq] = attn_mask_batch[i_seq].clone()
                active[i_seq] = False
                continue

            reusable_probs = probs_batch_last[i_seq] if temp == 1.0 else None
            concept_vec, s_ids, s_probs = build_concept_embedding(
                logits_batch_last[i_seq], temp, approach, probs=reusable_probs
            )
            # for discrete, log the actually-sampled token, else log the argmax
            if approach == "discrete":
                argmax_ids_lists[i_seq].append(int(s_ids[0].item()))
            else:
                argmax_ids_lists[i_seq].append(argmax_ids_last_i)

            concept_vecs_lists[i_seq].append(concept_vec)
            sample_ids_lists[i_seq].append(s_ids)
            sample_probs_lists[i_seq].append(s_probs)
            concept_embeds_batch_last[i_seq, 0] = concept_vec.to(model.dtype)

        if not any(active):
            break

        # extend attention mask for all sequences with 1s
        # (concept token is real for every row, even though it's just zeros for inactive rollouts)
        # however: wasted work to continue forward prop on inactive rollouts
        # TODO: unavoidable in batched processing?
        attn_mask_batch = torch.cat(
            [
                attn_mask_batch,
                torch.ones(n_batch, 1, device=device, dtype=attn_mask_batch.dtype),
            ],
            dim=1,
        )

        # position of the new concept token per row = number of real tokens so far - 1
        step_position_ids = attn_mask_batch.long().sum(-1, keepdim=True) - 1  # [B, 1]

        out_batch = model(
            inputs_embeds=concept_embeds_batch_last,
            past_key_values=past_kv_batch,
            attention_mask=attn_mask_batch,
            position_ids=step_position_ids,
            use_cache=True,
        )
        past_kv_batch = out_batch.past_key_values
        logits_batch_last = out_batch.logits[:, -1, :].float()

    for i_seq in range(n_batch):
        if active[i_seq]:
            stop_reasons[i_seq] = "max_steps"
            saved_past_kv[i_seq] = _slice_past_kv(past_kv_batch, i_seq)
            saved_attn[i_seq] = attn_mask_batch[i_seq].clone()
            active[i_seq] = False

    del past_kv_batch, logits_batch_last, out_batch, attn_mask_batch

    # --- batched rollout: answer generation ---
    # keeping padding in the cache and pass attention mask
    # so rotary positions stay aligned with rotations already encoded in cache
    all_rollouts: list[list[dict]] = [[] for _ in range(n_prompts)]
    for i_seq in range(n_batch):
        n = i_seq // G
        pad_len_i = max_prompt_len - prompt_lens[n]

        prefix_ids = (
            prefix_ids_threshold
            if stop_reasons[i_seq] == "threshold"
            else prefix_ids_maxsteps
        )
        prefix_len = prefix_ids.shape[1]
        prefix_attn = torch.cat(
            [
                saved_attn[i_seq].unsqueeze(0),
                torch.ones(1, prefix_len, device=device, dtype=saved_attn[i_seq].dtype),
            ],
            dim=1,
        )
        # compute kv and logits for prompt + think + prefix
        # different saved_past_kv / saved_attn will have different lengths, depending on when a rollout reached threshold
        # prefix positions continue from the count of real tokens already in the cache
        prefix_start = int(saved_attn[i_seq].long().sum().item())
        prefix_position_ids = torch.arange(
            prefix_start, prefix_start + prefix_len, device=device, dtype=torch.long
        ).unsqueeze(0)

        out_batch = model(
            input_ids=prefix_ids,
            past_key_values=saved_past_kv[i_seq],
            attention_mask=prefix_attn,
            position_ids=prefix_position_ids,
            use_cache=True,
        )
        past_kv_ans = out_batch.past_key_values
        first_answer_logits = out_batch.logits[0, -1, :].float()

        answer_ids = rollout_single_answer_sequence(
            past_kv_ans, first_answer_logits, prefix_attn
        )
        generated_text = BOXED_STRING + tokenizer.decode(
            answer_ids, skip_special_tokens=True
        )

        # return unpadded prompt_ids for downstream code (policy_log_prob, log_rollouts)
        unpadded_prompt_ids = prompt_ids[n : n + 1, pad_len_i:]

        all_rollouts[n].append(
            {
                "prompt_ids": unpadded_prompt_ids,
                "prefix_ids": prefix_ids,
                "concept_vecs": concept_vecs_lists[i_seq],
                "sample_ids_list": sample_ids_lists[i_seq],
                "sample_probs_list": sample_probs_lists[i_seq],
                "argmax_ids_list": argmax_ids_lists[i_seq],
                "trigger_probs": trigger_probs_last_sum_lists[i_seq],
                "window_sums": trigger_probs_last_sum_window_lists[i_seq],
                "stop_reason": stop_reasons[i_seq],
                "answer_ids": answer_ids,
                "generated_text": generated_text,
            }
        )

    return all_rollouts


def policy_log_prob(rollout, approach, temp):
    prompt_embeds = model_embeddings(rollout["prompt_ids"]).detach()
    prompt_len = prompt_embeds.size(1)

    concept_vecs = torch.stack(rollout["concept_vecs"]).to(device, dtype=model.dtype)
    concept_vecs = concept_vecs.unsqueeze(0)

    full_embeds = torch.cat([prompt_embeds, concept_vecs], dim=1)

    out = model(inputs_embeds=full_embeds, use_cache=False)

    logits = out.logits[0, prompt_len - 1 : -1, :].float()
    probs = F.softmax(logits / temp, dim=-1)

    total_log_prob = 0.0

    if approach == "token_sampling":
        log_probs = F.log_softmax(logits / temp, dim=-1)

        for t in range(len(rollout["concept_vecs"])):
            sample_ids = rollout["sample_ids_list"][t]
            sampled_probs = probs[t, sample_ids]

            cumulative_probs = sampled_probs.cumsum(dim=0)
            remaining = 1.0 - torch.cat(
                [torch.zeros(1, device=device), cumulative_probs[:-1]]
            )
            remaining = remaining.clamp(min=1e-8)

            log_conditionals = log_probs[t, sample_ids] - remaining.log()
            total_log_prob = total_log_prob + log_conditionals.sum()
    elif approach == "gaussian_noise":
        # mean of the Gaussian = full-softmax soft-thinking vector per position.
        # equivalent to soft_thinking(top_p=1.0) but over all T positions at once
        # ([T, V] @ [V, d] -> [T, d]) without the huge [T, V, d] gather.
        soft_probs = F.softmax(logits / temp, dim=-1)  # [T, V]
        soft_vecs = (
            soft_probs.to(vocab_embeds_device.dtype) @ vocab_embeds_device
        ).float()  # [T, d]
        total_log_prob += -(
            (concept_vecs.squeeze(0).float() - soft_vecs).pow(2).sum()
        ) / (2 * sigma**2)
    elif approach == "discrete":
        # standard policy gradient: log pi(a_t | s_t) summed over sequence
        log_probs = F.log_softmax(logits / temp, dim=-1)
        for t in range(len(rollout["concept_vecs"])):
            sample_id = rollout["sample_ids_list"][t]  # [1]
            total_log_prob = total_log_prob + log_probs[t, sample_id].sum()
    elif approach == "dirichlet_sampling":
        # log pi(w_t | s_t) = Dirichlet density of the sampled weights under the
        # recomputed (grad-tracking) concentration alpha = gamma * p_norm.
        for t in range(len(rollout["concept_vecs"])):
            sample_ids = rollout["sample_ids_list"][t]  # [n] nucleus token ids
            w = rollout["sample_probs_list"][t].to(device)  # [n] sampled weights

            p_sub = probs[t, sample_ids]
            p_sub = p_sub / p_sub.sum().clamp(min=1e-12)
            alpha = (CFG.rl_config.dirichlet_gamma * p_sub).clamp(min=1e-6)

            w = w.clamp(min=1e-8)
            w = w / w.sum()  # defensively re-project onto the simplex

            dist = torch.distributions.Dirichlet(alpha, validate_args=False)
            total_log_prob = total_log_prob + dist.log_prob(w)

    return total_log_prob


def batch_for_step(step: int) -> dict:
    batch_size = CFG.rl_config.batch_size
    n = len(dataset)
    steps_per_epoch = max(n // batch_size, 1)

    epoch = step // steps_per_epoch
    within_epoch = step % steps_per_epoch

    g = torch.Generator()
    g.manual_seed(DATA_SEED + epoch)
    perm = torch.randperm(n, generator=g)

    start = within_epoch * batch_size
    idx = perm[start : start + batch_size].tolist()
    return dataset[idx]


def train(start_step: int = 0):
    """Continue after start_step completed batches; checkpoint steps are one-based."""
    approach = CFG.rl_config.approach
    temp = CFG.rl_config.temperature
    max_batches = CFG.rl_config.max_batches
    accumulation = CFG.rl_config.gradient_accumulation_steps
    if accumulation <= 0 or max_batches <= 0 or max_batches % accumulation:
        raise ValueError("The batch budget must contain complete accumulation groups")
    if not 0 <= start_step <= max_batches or start_step % accumulation:
        raise ValueError("Resume must start after a completed optimizer update")
    if SAVE_EVERY <= 0 or SAVE_EVERY % accumulation:
        raise ValueError("Checkpoints must follow complete accumulation groups")

    optimizer.zero_grad()

    running_reward = 0.0  # length-discounted reward (shaped) — used for advantage
    running_correct = 0.0  # raw binary correctness, pre-shaping
    running_length = 0.0  # mean think-length per rollout
    running_loss = 0.0  # sum of per-rollout PG loss (-A * log_prob), unnormalized
    n_log_rollouts = 0  # denominator for the per-rollout reward/correct/length avgs
    n_loss_terms = 0  # denominator for the per-rollout loss avg (backprop'd rollouts)

    loss_history: list[float] = []
    reward_history: list[float] = []

    t_start = time.perf_counter()
    t_window = t_start

    for step in range(start_step + 1, max_batches + 1):
        batch = batch_for_step(step - 1)

        n_skipped_questions = 0

        # batch_size × group_size rollouts
        prompts = [make_prompt(question, tokenizer) for question in batch["question"]]
        all_group_rollouts = rollout_batch(
            prompts, temp, approach, CFG.rl_config.group_size
        )

        for q_i, (question, answer) in enumerate(
            zip(batch["question"], batch["answer"])
        ):
            group_rollouts = all_group_rollouts[q_i]

            correctness = [
                get_reward(rollout["generated_text"], answer)
                for rollout in group_rollouts
            ]
            lengths = [len(rollout["concept_vecs"]) for rollout in group_rollouts]

            # multiplicative length penalty: r = correctness × (1 − alpha · T / max_tokens)
            # this will effectively scale the gradients as part of advantages, just like the rewards
            alpha = CFG.rl_config.length_penalty_alpha
            T_max = CFG.rl_config.max_tokens
            group_rewards = torch.tensor(
                [c * (1.0 - alpha * (L / T_max)) for c, L in zip(correctness, lengths)],
                dtype=torch.float32,
            )
            running_reward += group_rewards.mean().item()
            running_correct += sum(correctness) / len(correctness)
            running_length += sum(lengths) / len(lengths)
            n_log_rollouts += 1

            if step % 50 == 0:
                log_rollouts(
                    seq_log,
                    tokenizer,
                    group_rollouts,
                    group_rewards,
                    question,
                    answer,
                    step,
                )

            if group_rewards.unique().numel() == 1:
                # no step given invariant rewards
                n_skipped_questions += 1
                continue

            # losses
            baseline_rewards = (group_rewards.sum() - group_rewards) / (
                len(group_rewards) - 1
            )
            advantages = group_rewards - baseline_rewards
            normalizer = (
                CFG.rl_config.group_size
                * CFG.rl_config.gradient_accumulation_steps
                * CFG.rl_config.batch_size
            )
            for i, rollout in enumerate(group_rollouts):
                if len(rollout["concept_vecs"]) == 0:
                    # there are no concept_vecs to backprop through
                    # the policy immediately reached threshold
                    continue

                log_prob = policy_log_prob(rollout, approach, temp)

                A = advantages[i].to(device)
                pg_loss = -A * log_prob
                # normalizer only scales the gradient for accumulation; log the
                # unnormalized per-rollout PG loss so the metric is independent of
                # group_size / grad_accum / batch_size.
                (pg_loss / normalizer).backward()

                running_loss += pg_loss.item()
                n_loss_terms += 1

        if step % accumulation == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                CFG.rl_config.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        if step % LOG_EVERY == 0:
            now = time.perf_counter()
            window_secs = now - t_window
            elapsed = now - t_start
            secs_per_step = window_secs / LOG_EVERY
            steps_left = max_batches - step
            eta_secs = secs_per_step * steps_left

            rollout_denom = max(n_log_rollouts, 1)
            avg_r = running_reward / rollout_denom  # shaped reward
            avg_correct = running_correct / rollout_denom  # raw correctness
            avg_len = running_length / rollout_denom  # mean think-length
            avg_l = running_loss / max(n_loss_terms, 1)  # mean per-rollout PG loss

            # Trim history to a fixed window so the sparkline stays readable
            # over long runs (last 80 log windows ≈ ~80 × LOG_EVERY steps).
            loss_history.append(avg_l)
            reward_history.append(avg_r)
            loss_history[:] = loss_history[-80:]
            reward_history[:] = reward_history[-80:]

            log(
                seq_log,
                f"\n>>> step {step:5d} | loss {avg_l:.4f}"
                f" | reward {avg_r:.3f} | correct {avg_correct:.3f}"
                f" | mean_len {avg_len:.1f}"
                f" | temp {temp:.3f}"
                f" | skipped {n_skipped_questions}/{CFG.rl_config.batch_size}"
                f" | {secs_per_step:.1f}s/step"
                f" | elapsed {format_duration(elapsed)}"
                f" | ETA {format_duration(eta_secs)}",
            )

            metrics_log.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": avg_l,
                        "reward": avg_r,
                        "correct": avg_correct,
                        "mean_len": avg_len,
                        "temp": temp,
                        "skipped": n_skipped_questions,
                        "elapsed_s": elapsed,
                    }
                )
                + "\n"
            )
            metrics_log.flush()

            running_reward = 0.0
            running_correct = 0.0
            running_length = 0.0
            running_loss = 0.0
            n_log_rollouts = 0
            n_loss_terms = 0
            t_window = now

        if step % SAVE_EVERY == 0 or step == max_batches:
            ckpt = os.path.join(OUTPUT_DIR, f"step_{step:05d}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            tmp_path = os.path.join(ckpt, "training_state.pt.tmp")
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "rng_state": torch.get_rng_state(),
                },
                tmp_path,
            )
            os.rename(tmp_path, os.path.join(ckpt, "training_state.pt"))
            log(seq_log, f"  -> saved {ckpt}")


def main():
    import argparse
    from dataclasses import asdict
    parser = argparse.ArgumentParser(description="GSM8K RL preset for thesis Appendix A")
    parser.add_argument("--model-id", choices=MODELS, default=CFG.model_id)
    parser.add_argument("--approach", choices=APPROACHES, default=CFG.rl_config.approach)
    parser.add_argument("--output-dir", type=_Path, help="Fresh run directory (required for training)")
    parser.add_argument("--resume", action="store_true", help="Resume a run with the same recorded configuration")
    parser.add_argument("--print-config", action="store_true", help="Print the preset without loading data or models")
    args = parser.parse_args()
    CFG.model_id = args.model_id
    CFG.rl_config.approach = args.approach
    if args.print_config:
        print(json.dumps(asdict(CFG), indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    initialize(args.output_dir, args.resume)
    try:
        train(start_step=start_step)
    finally:
        seq_log.close()
        metrics_log.close()


if __name__ == "__main__":
    main()
