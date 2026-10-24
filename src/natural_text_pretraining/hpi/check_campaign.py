"""Check the tested code/data/config before production, on a compute node only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(source, data, smoke):
    report = json.loads(smoke.read_text())
    config = json.loads((source / "experiment.json").read_text())
    if report["status"] != "passed" or not report["diagnostic_only"]:
        raise ValueError("A passed production-shape smoke report is required")
    if report["data_manifest_sha256"] != sha(data / "manifest.json"):
        raise ValueError("The smoke used different data")
    expected = copy.deepcopy(config)
    expected["training"]["monitor_samples"] = 3
    if report["config"] != expected:
        raise ValueError("Configuration differs from the tested production setup")
    if config["seeds"] != [101, 102, 103] or config["conditions"] != [
        "discrete", "learned_6000", "learned_12000", "learned_18000", "fixed_18000"
    ]:
        raise ValueError("Array mapping differs from the experiment configuration")
    if set(report["source_sha256"]) != {p.name for p in source.glob("*.py")}:
        raise ValueError("Python source file set differs from the smoke")
    for name, digest in report["source_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Untested source change: {name}")
    for condition in config["conditions"]:
        result = report["conditions"][condition]
        if result["status"] != "passed" or not result["lr_and_data_order_verified"]:
            raise ValueError(f"Missing successful smoke condition: {condition}")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "smoke"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--phase", choices=("training", "evaluation"), required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        p.error("Run inside an existing Slurm compute job")
    check(a.source, a.data, a.smoke)
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    gpu = torch.cuda.get_device_name()
    # The interactive smoke used PCIe; the production campaign uses SXM4 only,
    # preserving the evaluator's exact hardware match across all fifteen runs.
    if gpu != "NVIDIA A100-SXM4-40GB":
        raise RuntimeError(f"Production campaign requires A100-SXM4-40GB, got {gpu}")
    free = shutil.disk_usage(a.data).free / 2**30
    minimum = 20 if a.phase == "evaluation" else 18
    if free < minimum:
        raise RuntimeError(f"Only {free:.1f} GiB free; require {minimum} GiB")
    print(json.dumps(dict(status="ready", phase=a.phase, gpu=gpu,
                          node=os.environ["SLURMD_NODENAME"], free_gib=free)), flush=True)


if __name__ == "__main__":
    main()
