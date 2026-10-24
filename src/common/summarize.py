"""Summarize newly trained arithmetic or controlled-reasoning runs.

All numeric metrics retain their native units (accuracy is a fraction). Reports
include their inputs, settings, seed counts and missing/incomplete run paths.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .experiment_paths import checkpoint_dir, result_dir


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numbers(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for i, item in enumerate(value, 1):
            yield from numbers(item, f"{prefix}[{i}]")
    elif type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite metric: {prefix}")
        yield prefix, value


def load_run(path, experiment):
    metrics = read(path)
    if experiment == "arithmetic_reasoning":
        config_path = path.parent.parent / "manifest.json"
        config = read(config_path)
        condition = metrics["condition"]
        variant = config.get("model_variant")
        if variant:
            condition += ":" + variant["name"]
        settings = {key: config[key] for key in ("model_config", "training_config", "data_manifest", "source_sha256")}
        if variant:
            settings["model_variant"] = variant
        seed = metrics["seed"]
        if seed != config["model_seed"]:
            raise ValueError(f"Seed mismatch: {path}")
        values = {key: value for key, value in metrics.items() if key not in ("seed", "runtime")}
    else:
        config_path = path.parent / "config.json"
        config = read(config_path)
        condition = "/".join(str(config[key]) for key in ("kind", "prompt_mode", "rotation_mode"))
        # Seed-specific initialization text, paths and parameter counts do not
        # define a comparison. Keep all actual training/evaluation settings.
        keys = ("conclusions", "layers", "width", "heads", "batch_size", "schedule",
                "selection", "sampling", "novelty", "manifest_sha256", "source_hashes")
        settings = {key: config[key] for key in keys if key in config}
        settings["epochs"] = metrics["epochs"]
        seed = config["seed"]
        if not metrics["test_evaluated"]:
            raise ValueError(f"Final test evaluation is missing: {path}")
        values = {key: value for key, value in metrics.items() if key not in ("seed", "initial_validation")}
    history_path = path.with_name("history.jsonl")
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()]
    epochs = [row["epoch"] for row in history]
    if not epochs or epochs != sorted(set(epochs)):
        raise ValueError(f"Empty, duplicate or unordered history: {history_path}")
    return dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                config_source=str(config_path.resolve()), config_sha256=digest(config),
                history_source=str(history_path.resolve()), history_sha256=hashlib.sha256(history_path.read_bytes()).hexdigest(),
                group=digest(settings)[:16], settings=settings, condition=condition,
                seed=seed, metrics=dict(numbers(values)), history=history)


def collect(root, experiment):
    paths = sorted(root.rglob("final_metrics.json"))
    if not paths:
        raise ValueError(f"No final_metrics.json under {root}")
    runs = [load_run(path, experiment) for path in paths]
    seen = set()
    groups = defaultdict(list)
    for run in runs:
        key = run["group"], run["condition"], run["seed"]
        if key in seen:
            raise ValueError(f"Duplicate seed/configuration {key}; select one campaign root")
        seen.add(key)
        for metric, value in run["metrics"].items():
            groups[run["group"], run["condition"], metric].append((run["seed"], value))
    summary = []
    for (group, condition, metric), observations in sorted(groups.items()):
        values = [value for _, value in observations]
        summary.append(dict(group=group, condition=condition, metric=metric,
                            n=len(values), seeds=sorted(seed for seed, _ in observations),
                            mean=statistics.mean(values),
                            sample_sd=statistics.stdev(values) if len(values) > 1 else None))
    candidates = {path.parent for name in ("history.jsonl", "best.pt", "config.json") for path in root.rglob(name)}
    incomplete = sorted(str(path.resolve()) for path in candidates if not (path / "final_metrics.json").exists())
    return runs, summary, incomplete


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(runs, experiment, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    groups = defaultdict(list)
    for run in runs:
        groups[run["group"], run["condition"]].append(run)
    for (group, condition), members in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        for axis, split in zip(axes, ("Training", "Validation")):
            for run in members:
                history = run["history"]
                if experiment == "arithmetic_reasoning":
                    key = "train_losses" if split == "Training" else "validation_losses"
                    values = [row[key]["reasoning"] for row in history]
                    unit = "State cross entropy" if "cot2" not in condition else "Distribution cross entropy"
                else:
                    values = [row["train_nll"] if split == "Training" else row["validation"]["nll"] for row in history]
                    unit = "NLL per " + history[-1]["validation"]["unit"]
                axis.plot([row["epoch"] for row in history], values,
                          marker="o" if len(history) == 1 else None, label=f"Seed {run['seed']}")
            axis.set(xlabel="Epoch", ylabel=unit, title=split)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
        fig.suptitle(condition)
        stem = f"{group}_{digest(condition)[:8]}"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("arithmetic_reasoning", "controlled_reasoning"))
    parser.add_argument("--input-root", type=Path, help="Campaign root or individual run directory")
    parser.add_argument("--output-dir", type=Path, help="Must be a new directory")
    args = parser.parse_args()
    root = args.input_root or checkpoint_dir(args.experiment)
    output = args.output_dir or result_dir(args.experiment, "summary")
    runs, summary, incomplete = collect(root, args.experiment)
    output.mkdir(parents=True, exist_ok=False)
    curves = output / "learning_curves"
    curves.mkdir()
    plot_curves(runs, args.experiment, curves)
    report = dict(experiment=args.experiment, input_root=str(root.resolve()),
                  units="Native metric units; accuracy fractions are not multiplied by 100. sample_sd is null for one seed.",
                  incomplete_runs=incomplete, settings={r["group"]: r["settings"] for r in runs},
                  runs=[{k: v for k, v in r.items() if k not in ("settings", "history")} for r in runs], summary=summary)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(output / "summary.csv", summary)
    write_csv(output / "per_seed.csv", [dict(group=r["group"], condition=r["condition"], seed=r["seed"],
              metric=metric, value=value, source=r["source"]) for r in runs for metric, value in r["metrics"].items()])
    print(f"Summarized {len(runs)} completed runs; {len(incomplete)} incomplete runs listed. Results: {output}")


if __name__ == "__main__":
    main()
