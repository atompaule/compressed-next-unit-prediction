#!/usr/bin/env python3
"""Serial synchronized inference benchmark for completed arithmetic-register runs.

Includes the full arithmetic program prompt, sequential execution, and one delayed
query. Excludes loading and tensor transfer. Save hardware, individual timings,
and checkpoint hashes; never infer a latency reduction just from payload size.
"""

from __future__ import annotations

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
if __package__ in (None, ""):
    __package__ = "src.arithmetic_reasoning"

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from .dataset import load_data_config
from src.arithmetic_reasoning.training import select_device
from src.arithmetic_reasoning.core import (
    CONDITIONS,
    EncodedSplit,
    Model,
    ModelConfig,
    Vocabulary,
    load_manifest,
    rollout,
    query_tokens,
)
from src.arithmetic_reasoning.train import (
    runtime_info,
    write_json,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def benchmark(
    model: Model,
    dataset: EncodedSplit,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict:
    if not 1 <= batch_size <= len(dataset) or warmup < 0 or repeats < 1:
        raise ValueError("Invalid batch size, warmup, or repeats")
    model.eval()
    batch = {
        key: value[:batch_size].to(device) for key, value in dataset.tensors.items()
    }
    batch["query_ids"] = query_tokens(
        model, torch.full_like(batch["query_roles"], dataset.config.path_length), 0
    )
    for _ in range(warmup):
        rollout(model, batch, branch_queries=False)
    durations = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = rollout(model, batch, branch_queries=False)
        synchronize(device)
        durations.append(time.perf_counter() - start)
    median = statistics.median(durations)
    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "batch_seconds_samples": durations,
        "median_batch_seconds": median,
        "amortized_ms_per_episode": median * 1000 / batch_size,
        "episodes_per_second": batch_size / median,
        "cache_positions_before_query": result["cache_positions"],
        "cache_bytes_per_sample_before_query": result["cache_bytes_per_sample"],
        "scope": "full prompt prefill + all operations/feedback + one delayed query; no data loading or transfer",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    output = args.run_dir / "latency_comparison.json"
    if output.exists():
        raise FileExistsError(
            f"{output} already exists; preserve or move it before rerunning"
        )
    run_manifest = json.loads((args.run_dir / "manifest.json").read_text())
    data_dir = Path(run_manifest["dataset_directory"])
    manifest = load_manifest(data_dir)
    if manifest != run_manifest["data_manifest"]:
        raise ValueError("Dataset manifest changed since training")
    vocabulary = Vocabulary(load_data_config(manifest))
    dataset = EncodedSplit(data_dir, "test", manifest, vocabulary)
    device = select_device(args.device)
    if args.conditions is None:
        args.conditions = [
            c for c in CONDITIONS if (args.run_dir / c / "best.pt").is_file()
        ]
    if not args.conditions:
        raise ValueError("No checkpoints found for benchmarking")
    conditions = {}
    for condition in args.conditions:
        path = args.run_dir / condition / "best.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if (
            checkpoint["condition"] != condition
            or checkpoint["data_manifest"] != manifest
        ):
            raise ValueError("Checkpoint condition or dataset mismatch")
        variant = checkpoint.get("model_variant")
        if variant is None:
            model = Model(
                condition, vocabulary, ModelConfig(**checkpoint["model_config"])
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            from .variants import NONORTHOGONAL_VARIANT, load_nonorthogonal_checkpoint

            if variant["name"] != NONORTHOGONAL_VARIANT:
                raise ValueError(f"Unsupported checkpoint variant: {variant}")
            model = load_nonorthogonal_checkpoint(path, device)
        results = {
            str(size): benchmark(
                model, dataset, device, size, args.warmup, args.repeats
            )
            for size in args.batch_sizes
        }
        conditions[condition] = {
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "batches": results,
        }
        print(
            json.dumps(
                {condition: {b: x["median_batch_seconds"] for b, x in results.items()}}
            ),
            flush=True,
        )
        del model
    write_json(
        output,
        {
            "runtime": runtime_info(device),
            "conditions": conditions,
            "condition_order": args.conditions,
            "caveat": "Run without concurrent accelerator workloads; timings are hardware-specific",
        },
    )


if __name__ == "__main__":
    main()
