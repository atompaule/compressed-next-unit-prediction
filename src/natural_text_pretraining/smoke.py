"""Bounded readiness test: four ordinary updates, resume, and a long-window batch.

Uses the production runner and full catalogue. Only the validation monitor is
reduced to three examples. Outputs are diagnostic and never production resumes.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from data import PackedBank, Windows, atomic_json, digest
from protocol import CONDITIONS
import train

HERE = Path(__file__).resolve().parent


def longest_windows(windows, bank, count, kind):
    """Select by exact model input length, with index as deterministic tie break."""
    scores = np.zeros(len(windows), dtype=np.int64)
    lengths = np.diff(bank.offsets)
    for values, offsets in ((windows.context, windows.context_offsets),
                            (windows.target, windows.target_offsets)):
        if kind == "sentence":
            scores += np.diff(offsets)
        else:
            for lo in range(0, len(windows), 4096):
                hi = min(lo + 4096, len(windows))
                start, end = int(offsets[lo]), int(offsets[hi])
                sums = np.r_[0, np.cumsum(lengths[values[start:end]], dtype=np.int64)]
                scores[lo:hi] += np.diff(sums[offsets[lo:hi + 1] - start])
    # The runner feeds context plus all but the final target token/chunk (no BOS).
    scores -= 1
    order = np.lexsort((np.arange(len(scores)), -scores))[:count]
    return order, scores[order]


def worker(args):
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required")
        torch.cuda.reset_peak_memory_stats()
    cfg = json.loads(args.config.read_text())
    training = cfg["training"]
    condition_dir = args.out / args.worker
    common = dict(data=args.data, config=args.config, out=condition_dir,
                  condition=args.worker, seed=101, device=args.device,
                  threads=args.threads, diagnostic=args.diagnostic)
    start = time.monotonic()
    for updates, expected in ((3, 3), (1, 4)):
        result = train.run(argparse.Namespace(**common, max_updates=updates))
        if result != 75:
            raise RuntimeError(f"Expected bounded pause (75), received {result}")
        saved = torch.load(condition_dir / "last.pt", map_location="cpu", weights_only=False)
        if saved["state"]["step"] != expected:
            raise AssertionError("Resume did not continue at the expected step")
        if any(int(s["step"]) != expected for s in saved["optimizer"]["state"].values()):
            raise AssertionError("Optimizer update count was not restored")
        del saved
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    history = [json.loads(s) for s in (condition_dir / "history.jsonl").read_text().splitlines()]
    if [h["step"] for h in history] != [1, 2, 3, 4]:
        raise AssertionError("Missing or duplicated updates after resume")
    saved = torch.load(condition_dir / "last.pt", map_location="cpu", weights_only=False)
    windows = Windows(args.data, "train")
    total = math.ceil(len(windows) / training["batch_windows"])
    order = np.random.default_rng(101 * 1000).permutation(len(windows))
    for h in history:
        step = h["step"]
        expected_lr = train.schedule(step, total, training["peak_lr"])
        if not math.isclose(h["lr"], expected_lr, rel_tol=1e-12):
            raise AssertionError("Learning rate differs from the one-epoch schedule")
        indices = order[(step - 1) * training["batch_windows"]:step * training["batch_windows"]]
        if h["batch_window_indices_sha256"] != hashlib.sha256(indices.tobytes()).hexdigest():
            raise AssertionError("Resume changed the training data order")
    initialization = json.loads((condition_dir / "initialization.json").read_text())
    kind, rotation, negatives = CONDITIONS[args.worker]
    if rotation == "fixed_random":
        if hashlib.sha256(saved["alpha"].numpy().tobytes()).hexdigest() != initialization["alpha_sha256"]:
            raise AssertionError("Frozen rotations changed during ordinary training")

    # Separate, explicitly adversarial update at peak LR: repeat the longest
    # microbatch until the full effective batch is reached. Never save this model.
    model, alpha, lexical = train.initialize(101, args.device, cfg["architecture"], training["dropout"])
    model.load_state_dict(saved["model"])
    if kind == "sentence":
        alpha.data.copy_(saved["alpha"].to(alpha.device))
    alpha.requires_grad_(kind == "sentence" and rotation == "learned")
    del saved
    optimizer = train.make_optimizer(model, alpha, kind, training["peak_lr"])
    before_alpha = alpha.detach().clone()
    before_embedding = model.embed_tokens.weight.detach().clone()
    with PackedBank(args.data, "train_bank") as bank:
        indices, lengths = longest_windows(windows, bank, training["microbatch_windows"], kind)
        repeated = np.resize(indices, training["batch_windows"])
        train.sync(args.device)
        stress_start = time.monotonic()
        stress = train.update(model, alpha, lexical, optimizer, bank, windows.batch(repeated),
                              kind, np.random.default_rng(771304), negatives,
                              training["catalogue_block"], training["microbatch_windows"], training["bf16"])
        train.sync(args.device)
        stress_seconds = time.monotonic() - stress_start
    if torch.equal(before_embedding, model.embed_tokens.weight):
        raise AssertionError("Embedding weights did not update")
    alpha_changed = not torch.equal(before_alpha, alpha)
    if kind == "sentence" and alpha_changed != (rotation == "learned"):
        raise AssertionError("Unexpected rotation update behavior")
    result = dict(status="passed", condition=args.worker, seed=101,
                  ordinary_updates=4, resumed_after_step=3,
                  lr_and_data_order_verified=True, history=history,
                  ordinary_update_median_seconds=statistics.median(h["update_seconds"] for h in history[1:]),
                  stress=dict(seconds=stress_seconds, window_indices=indices.tolist(),
                              model_input_lengths=lengths.tolist(), rotation_changed=alpha_changed, **stress),
                  initialization=initialization, elapsed_seconds=time.monotonic() - start)
    if args.device == "cuda":
        result.update(gpu=torch.cuda.get_device_name(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
    atomic_json(args.out / f"{args.worker}_report.json", result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=HERE / "experiment.json")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--diagnostic", action="store_true", help="Allow a tiny synthetic integration fixture")
    p.add_argument("--worker", choices=CONDITIONS, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    if args.device == "cuda" and (not os.environ.get("SLURM_JOB_ID") or
                                  not os.environ.get("SLURMD_NODENAME")):
        p.error("Run this GPU test inside a Slurm compute job")
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    if not args.diagnostic:
        production = json.loads((HERE / "experiment.json").read_text())
        if cfg != production:
            raise ValueError("Production smoke must use the committed experiment configuration")
    cfg["training"]["monitor_samples"] = 3
    path = args.out / "smoke_config.json"
    atomic_json(path, cfg)
    report = dict(status="running", diagnostic_only=True,
                  data_manifest_sha256=digest(args.data / "manifest.json"),
                  source_sha256=train.source_hashes(), config=cfg, conditions={})
    atomic_json(args.out / "report.json", report)
    for condition in CONDITIONS:
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--data", str(args.data), "--out", str(args.out), "--config", str(path),
                   "--device", args.device, "--threads", str(args.threads), "--worker", condition]
        if args.diagnostic:
            command.append("--diagnostic")
        print(f"Starting {condition}", flush=True)
        with (args.out / f"{condition}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report["conditions"][condition] = (json.loads((args.out / f"{condition}_report.json").read_text())
                                           if result.returncode == 0 else
                                           dict(status="failed", exit_code=result.returncode,
                                                log=str(args.out / f"{condition}.log")))
        atomic_json(args.out / "report.json", report)
    report["status"] = "passed" if all(r["status"] == "passed" for r in report["conditions"].values()) else "failed"
    atomic_json(args.out / "report.json", report)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
