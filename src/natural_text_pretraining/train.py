"""Resumable random GPT-2 training with exact native held-out validation.

SentenceDot uses fresh inclusion-corrected sampled catalogue partitions; discrete
uses full token-vocabulary CE. No test arrays or test metadata are opened here.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

HERE = Path(__file__).resolve().parent
SHARED = HERE
from data import PackedBank, Windows, atomic_json, digest, load_manifest
from evaluation import validation
from model import (
    batched_queries,
    mean_embedding_norm,
    token_objective,
)

from sampled_objective import sampled_catalogue_backward
from protocol import NEGATIVE_CANDIDATES
from compute_accounting import (
    PROTOCOL as COMPUTE_PROTOCOL,
    ComputeCounter,
    accumulate,
    counted_update,
)

STOP = False
ARCHITECTURE = {
    "vocab_size": 50257,
    "n_embd": 512,
    "n_head": 4,
    "n_inner": 2048,
    "n_layer": 8,
    "n_positions": 1024,
}


class TextModel(nn.Module):
    """Random GPT-2 backbone; output projections explicitly use the same wte."""

    def __init__(self, architecture=None, dropout=0.1):
        super().__init__()
        config = GPT2Config(
            **(architecture or ARCHITECTURE),
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        config._attn_implementation = "sdpa"
        self.backbone = GPT2Model(config)

    @property
    def embed_tokens(self):
        return self.backbone.wte

    def forward(self, **kwargs):
        return self.backbone(**kwargs)


def request_stop(*_):
    global STOP
    STOP = True


def human(seconds):
    if seconds is None:
        return None
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m {seconds % 60}s"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def accelerator_memory(device):
    if device == "mps":
        return {
            "accelerator_memory_bytes": int(torch.mps.current_allocated_memory()),
            "accelerator_driver_memory_bytes": int(torch.mps.driver_allocated_memory()),
        }
    if device == "cuda":
        return {
            "accelerator_memory_bytes": int(torch.cuda.memory_allocated()),
            "accelerator_driver_memory_bytes": int(torch.cuda.memory_reserved()),
        }
    return {"accelerator_memory_bytes": None, "accelerator_driver_memory_bytes": None}


def initialize(seed, device, architecture=None, dropout=0.1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TextModel(architecture, dropout)
    # Initialize both methods identically, including this RNG draw. Only the
    # sentence method optimizes alpha; it is irrelevant to discrete inference.
    alpha = nn.Parameter(
        torch.empty(model.embed_tokens.embedding_dim // 2).uniform_(-math.pi, math.pi)
    )
    model.to(device)
    alpha = nn.Parameter(alpha.detach().to(device))
    lexical = torch.arange(model.embed_tokens.num_embeddings - 1, device=device)
    return model, alpha, lexical


def make_optimizer(model, alpha, kind, peak):
    parameters = list(model.parameters()) + (
        [alpha] if kind == "sentence" and alpha.requires_grad else []
    )
    return torch.optim.AdamW(
        parameters,
        lr=peak,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
    )


def schedule(step, total, peak):
    warmup = max(1, math.ceil(total * 0.03))
    if step <= warmup:
        return peak * step / warmup
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))


@counted_update
def update(
    model,
    alpha,
    lexical,
    optimizer,
    bank,
    rows,
    kind,
    sample_rng,
    negatives=NEGATIVE_CANDIDATES,
    block_size=4096,
    microbatch_windows=8,
    bf16=True,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if kind == "sentence":
        source_scale = mean_embedding_norm(model.embed_tokens.weight, lexical)
        scale = source_scale.detach().requires_grad_()
        h, targets, coefficients = batched_queries(
            model,
            alpha,
            bank,
            rows,
            scale,
            kind,
            bf16,
            microbatch_windows=microbatch_windows,
        )
        value = sampled_catalogue_backward(
            h,
            targets,
            coefficients,
            bank,
            model.embed_tokens.weight,
            alpha,
            scale,
            sample_rng,
            negatives,
            block_size,
        )
        if scale.grad is None:
            raise RuntimeError("No gradient through chunk norm scale")
        source_scale.backward(scale.grad)
    else:
        h, targets, coefficients = batched_queries(
            model,
            alpha,
            bank,
            rows,
            None,
            kind,
            bf16,
            microbatch_windows=microbatch_windows,
        )
        value = token_objective(h, targets, coefficients, model.embed_tokens.weight)
    parameters = list(model.parameters()) + (
        [alpha] if kind == "sentence" and alpha.requires_grad else []
    )
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
    if not math.isfinite(float(norm)) or not math.isfinite(value["loss"]):
        raise FloatingPointError(
            "Nonfinite loss or gradient; optimizer update rejected"
        )
    optimizer.step()
    value.update(
        grad_norm_before_clip=float(norm),
        grad_clip_threshold=1.0,
        grad_norm_above_clip_threshold=bool(float(norm) > 1.0),
    )
    return value


def rng_state(sample_rng, device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "sampler": copy.deepcopy(sample_rng.bit_generator.state),
        "accelerator": (
            torch.mps.get_rng_state()
            if device == "mps"
            else torch.cuda.get_rng_state_all()
            if device == "cuda"
            else None
        ),
    }


def restore_rng(state, sample_rng, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    sample_rng.bit_generator.state = state["sampler"]
    if device == "mps":
        torch.mps.set_rng_state(state["accelerator"])
    elif device == "cuda":
        torch.cuda.set_rng_state_all(state["accelerator"])


def payload(model, alpha, kind, optimizer, sample_rng, state, config, recovery=True):
    value = {
        "model": model.state_dict(),
        "alpha": alpha.detach() if kind == "sentence" else None,
        "config": config,
        "state": copy.deepcopy(state),
    }
    if recovery:
        value.update(
            optimizer=optimizer.state_dict(),
            rng=rng_state(sample_rng, model.embed_tokens.weight.device.type),
        )
    return value


def restore(path, model, alpha, kind, optimizer, sample_rng, config):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != config:
        raise ValueError(
            "Resume configuration, source, runtime or data fingerprints differ"
        )
    model.load_state_dict(saved["model"])
    if kind == "sentence":
        alpha.data.copy_(saved["alpha"].to(alpha.device))
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"], sample_rng, model.embed_tokens.weight.device.type)
    return saved["state"]


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes():
    return {p.name: digest(p) for p in sorted(HERE.glob("*.py"))}


def run(args):
    global STOP
    STOP = False
    from protocol import CONDITIONS

    kind, rotation, negatives = CONDITIONS[args.condition]
    experiment = json.loads(args.config.read_text())
    architecture = experiment["architecture"]
    training = experiment["training"]
    manifest = load_manifest(args.data)
    verification = json.loads((args.data / "verification.json").read_text())
    if verification["status"] != "passed" or verification[
        "data_manifest_sha256"
    ] != digest(args.data / "manifest.json"):
        raise ValueError("A matching successful data verification is required")
    if manifest["status"] != "complete" or manifest["diagnostic"] != args.diagnostic:
        raise ValueError("Data readiness or diagnostic/production mismatch")
    if architecture["vocab_size"] != manifest["vocab_size"]:
        raise ValueError("Model and tokenizer vocabularies differ")
    if manifest["maximum_model_input_tokens"] > architecture["n_positions"]:
        raise ValueError("Data exceed model context capacity")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "worker.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(args.threads)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    train, bank = Windows(args.data, "train"), PackedBank(args.data, "train_bank")
    batch = training["batch_windows"]
    total = math.ceil(len(train) / batch)
    config = dict(
        kind=kind,
        rotation_mode=rotation,
        negatives=negatives,
        seed=args.seed,
        condition=args.condition,
        architecture=architecture,
        training=training,
        device=args.device,
        diagnostic=args.diagnostic,
        planned_updates=total,
        chunk_boundary="metadata_only",
        source_sha256=source_hashes(),
        data_sha256={"manifest.json": digest(args.data / "manifest.json")},
        runtime={
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "device_name": torch.cuda.get_device_name()
            if args.device == "cuda"
            else platform.machine(),
        },
        compute_accounting=COMPUTE_PROTOCOL,
        checkpoint_selection="end_of_one_epoch",
    )
    cp = args.out / "config.json"
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError("Resume configuration/source/data/runtime changed")
    atomic_json(cp, config)
    model, alpha, lexical = initialize(
        args.seed, args.device, architecture, training["dropout"]
    )
    alpha.requires_grad_(kind == "sentence" and rotation == "learned")
    optimizer = make_optimizer(model, alpha, kind, training["peak_lr"])
    rng = np.random.default_rng(args.seed + 771203)
    state = dict(
        step=0,
        planned_updates=total,
        training_seconds=0.0,
        validation_seconds=0.0,
        elapsed_seconds=0.0,
        validation_pending=True,
        validation_records=[],
    )
    last = args.out / "last.pt"
    if last.exists():
        state = restore(last, model, alpha, kind, optimizer, rng, config)
    else:
        h = hashlib.sha256()
        for name, t in model.state_dict().items():
            h.update(name.encode())
            h.update(t.detach().cpu().numpy().tobytes())
        atomic_json(
            args.out / "initialization.json",
            dict(
                model_sha256=h.hexdigest(),
                alpha_sha256=hashlib.sha256(
                    alpha.detach().cpu().numpy().tobytes()
                ).hexdigest(),
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_rotation_parameters=alpha.numel()
                if alpha.requires_grad
                else 0,
            ),
        )
    history = args.out / "history.jsonl"
    if history.exists():
        history.write_text(
            "".join(
                l
                for l in history.read_text().splitlines(True)
                if json.loads(l)["step"] <= state["step"]
            )
        )
    start = time.monotonic()
    previous = state["elapsed_seconds"]
    first = state["step"]
    order = np.random.default_rng(args.seed * 1000).permutation(len(train))
    checks = {0, total, *[math.ceil(total * i / 4) for i in range(1, 4)]}

    def status(label):
        atomic_json(
            args.out / "status.json",
            dict(status=label, pid=os.getpid(), updated_unix=time.time(), **state),
        )

    def save():
        state["elapsed_seconds"] = previous + time.monotonic() - start
        atomic_save(last, payload(model, alpha, kind, optimizer, rng, state, config))

    def assess():
        save()
        status("validating")
        before = time.monotonic()
        with ComputeCounter() as counter:
            result = validation(
                model,
                alpha,
                lexical,
                args.data,
                kind,
                training["monitor_samples"],
                training["catalogue_block"],
                training["microbatch_windows"],
                "chunk",
            )
        sync(args.device)
        seconds = time.monotonic() - before
        state["validation_seconds"] += seconds
        accumulate(state, counter.summary(), "validation")
        record = dict(
            step=state["step"],
            validation=result,
            seconds=seconds,
            compute=counter.summary(),
        )
        state["validation_records"].append(record)
        state["validation_pending"] = False
        atomic_json(args.out / f"validation_{state['step']}.json", record)
        save()
        print(
            json.dumps(dict(event="validation", step=state["step"], seconds=seconds)),
            flush=True,
        )

    try:
        if state["validation_pending"] and not STOP:
            assess()
        with history.open("a", buffering=1) as log:
            while state["step"] < total and not STOP:
                if args.max_updates and state["step"] - first >= args.max_updates:
                    break
                step = state["step"]
                indices = order[step * batch : (step + 1) * batch]
                lr = schedule(step + 1, total, training["peak_lr"])
                for g in optimizer.param_groups:
                    g["lr"] = lr
                sync(args.device)
                before = time.monotonic()
                value = update(
                    model,
                    alpha,
                    lexical,
                    optimizer,
                    bank,
                    train.batch(indices),
                    kind,
                    rng,
                    negatives,
                    training["catalogue_block"],
                    training["microbatch_windows"],
                    training["bf16"],
                )
                sync(args.device)
                seconds = time.monotonic() - before
                state["step"] += 1
                state["training_seconds"] += seconds
                accumulate(state, value["compute"], "training")
                state["validation_pending"] = state["step"] in checks
                row = dict(
                    step=state["step"],
                    epoch=min(state["step"] * batch, len(train)) / len(train),
                    windows_seen=min(state["step"] * batch, len(train)),
                    lr=lr,
                    update_seconds=seconds,
                    batch_window_indices_sha256=hashlib.sha256(
                        indices.tobytes()
                    ).hexdigest(),
                    **value,
                )
                log.write(json.dumps(row) + "\n")
                status("training")
                if state["step"] % 10 == 0 or state["step"] == first + 1:
                    print(json.dumps(row), flush=True)
                if state["validation_pending"] and not STOP:
                    assess()
                elif state["step"] % training["save_every"] == 0:
                    save()
        save()
        complete = state["step"] == total and not state["validation_pending"]
        if complete:
            atomic_save(
                args.out / "final.pt",
                payload(model, alpha, kind, optimizer, rng, state, config, False),
            )
        status("complete" if complete else "paused")
        return 0 if complete else 75
    except BaseException as error:
        atomic_json(
            args.out / "status.json",
            dict(
                status="failed",
                error=repr(error),
                step=state["step"],
                recovery=str(last),
            ),
        )
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    from protocol import CONDITIONS, SEEDS

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--config", type=Path, default=HERE / "experiment.json")
    p.add_argument("--condition", choices=CONDITIONS, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS, default=101)
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-updates", type=int, default=0)
    p.add_argument("--diagnostic", action="store_true")
    raise SystemExit(run(p.parse_args()))


if __name__ == "__main__":
    main()
