#!/usr/bin/env python3
"""Train supervised feedback conditions on a verified dataset bundle.

Uses atomic task tokens, not an external language tokenizer or pretrained model.
Training is full causal teacher forcing with equal state/query loss weighting.
Gold constituents enter continuous/unbound/discrete feedback; cache-only gets
only a constant token, with identical state targets used solely in its losses.
Evaluation always reports actual free-running KV-cached decoding separately.
"""

from __future__ import annotations

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
if __package__ in (None, ""):
    __package__ = "src.arithmetic_reasoning"

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import platform
from pathlib import Path
from src.common.experiment_paths import data_dir, checkpoint_dir
import random
import time
import os
import tempfile

import torch
import transformers
from torch.utils.data import DataLoader

from .dataset import SPLITS, load_data_config
from src.arithmetic_reasoning.model import assert_orthogonal_role_maps
from src.arithmetic_reasoning.training import make_scheduler, select_device
from src.arithmetic_reasoning.core import (
    CONDITIONS,
    EncodedSplit,
    Model,
    ModelConfig,
    Vocabulary,
    evaluate,
    load_manifest,
    supervised_loss,
    teacher_forced_inputs,
    state_logits,
    COT2,
    distribution_statistics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as f:
        temporary = Path(f.name)
        json.dump(data, f, indent=2)
        f.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def runtime_info(device: torch.device) -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "accelerator": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor()
        ),
        "dtype": "float32",
        "torch_cpu_threads": torch.get_num_threads(),
    }


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "src/arithmetic_reasoning").glob("*.py"))
    paths.extend((root / "src/common").glob("*.py"))
    paths.append(root / "src/arithmetic_reasoning/generate_data.py")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


@torch.no_grad()
def validation_losses(model, loader, device):
    model.eval()
    sums = dict(total=0.0, reasoning=0.0, query=0.0)
    count = correct = outputs = first_correct = 0
    if model.condition == COT2:
        sums.update(
            state_kl=0.0, state_target_entropy=0.0, state_tv=0.0, first_update_tv=0.0
        )
    for raw in loader:
        batch = {k: v.to(device) for k, v in raw.items()}
        inputs, positions, query_position = teacher_forced_inputs(model, batch)
        hidden = model.full_hidden(inputs)
        logits = state_logits(model, hidden, positions)
        if model.condition == COT2:
            stats = distribution_statistics(logits, batch["state_ids"])
            reasoning = stats["ce"].mean()
        else:
            reasoning = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), batch["state_ids"].reshape(-1)
            )
        query = torch.nn.functional.cross_entropy(
            model.node_logits(hidden[:, query_position]), batch["query_targets"]
        )
        n = len(batch["query_targets"])
        for key, value in dict(
            total=0.5 * (reasoning + query), reasoning=reasoning, query=query
        ).items():
            sums[key] += value.item() * n
        if model.condition == COT2:
            for key, value in dict(
                state_kl=stats["kl"].mean(),
                state_target_entropy=stats["entropy"].mean(),
                state_tv=stats["tv"].mean(),
                first_update_tv=stats["tv"][:, 0].mean(),
            ).items():
                sums[key] += value.item() * n
            count += n
            continue
        matches = logits.argmax(-1).eq(batch["state_ids"])
        correct += matches.sum().item()
        outputs += matches.numel()
        first_correct += matches[:, 0].all(-1).sum().item()
        count += n
    return {
        **{k: v / count for k, v in sums.items()},
        "state_value_accuracy": correct / outputs if outputs else None,
        "first_update_exact_accuracy": (
            first_correct / count if model.condition != COT2 else None
        ),
    }


def train_condition(
    *,
    condition: str,
    seed: int,
    datasets: dict,
    vocabulary: Vocabulary,
    config: ModelConfig,
    initial_state: dict,
    output: Path,
    device: torch.device,
    args: argparse.Namespace,
    data_manifest: dict,
    model_class: type[Model] = Model,
    model_metadata: dict | None = None,
    validate_model=assert_orthogonal_role_maps,
) -> dict:
    directory = output / condition
    # Never silently replace an earlier or in-progress experiment.
    directory.mkdir(exist_ok=False)
    set_seed(seed)
    model = model_class(condition, vocabulary, config).to(device)
    state = {
        key: value for key, value in initial_state.items() if key in model.state_dict()
    }
    model.load_state_dict(state, strict=True)
    validate_model(model)
    set_seed(seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(seed) if split == "train" else None,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }
    probe_dataset = copy.copy(datasets["validation"])
    probe_dataset.tensors = {
        k: v[: args.rollout_eval_samples] for k, v in probe_dataset.tensors.items()
    }
    probe_loader = DataLoader(probe_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = make_scheduler(
        optimizer,
        total_steps=args.epochs * len(loaders["train"]),
        warmup_ratio=0.05,
        min_lr_ratio=args.min_learning_rate / args.learning_rate,
    )
    best_loss, stopping_loss, stale, best_epoch, step = math.inf, math.inf, 0, 0, 0
    started = time.perf_counter()
    with (directory / "history.jsonl").open("x") as history:
        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            sums, count = dict(total=0.0, reasoning=0.0, query=0.0), 0
            if condition == COT2:
                sums["state_kl"] = 0.0
            for raw in loaders["train"]:
                batch = {key: value.to(device) for key, value in raw.items()}
                optimizer.zero_grad(set_to_none=True)
                losses = supervised_loss(model, batch)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError(
                        f"Non-finite loss: {condition}, step {step}"
                    )
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                model.project_role_maps_()
                scheduler.step()
                n = batch["query_ids"].shape[0]
                for key in sums:
                    sums[key] += float(losses[key].detach()) * n
                count += n
                step += 1
                if step == 1 or step % 50 == 0:
                    print(
                        f"seed={seed} {condition} epoch={epoch} step={step} "
                        f"loss={float(losses['total'].detach()):.5f}",
                        flush=True,
                    )
            validation = validation_losses(model, loaders["validation"], device)
            if not all(math.isfinite(x) for x in validation.values() if x is not None):
                raise FloatingPointError("Non-finite validation loss")
            improved = validation["reasoning"] < best_loss
            if improved:
                best_loss, best_epoch = validation["reasoning"], epoch
                checkpoint_data = {
                    "model_state_dict": model.state_dict(),
                    "condition": condition,
                    "model_config": asdict(config),
                    "data_config": data_manifest["data_config"],
                    "data_manifest": data_manifest,
                    "vocabulary": list(vocabulary.tokens),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_losses": validation,
                }
                if model_metadata is not None:
                    checkpoint_data["model_variant"] = model_metadata
                torch.save(checkpoint_data, directory / "best.pt")
            if validation["reasoning"] < stopping_loss - args.min_delta:
                stopping_loss, stale = validation["reasoning"], 0
            else:
                stale += 1
            entry = {
                "epoch": epoch,
                "global_step": step,
                "train_losses": {key: value / count for key, value in sums.items()},
                "validation_losses": validation,
                "checkpoint_improved": improved,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_seconds": time.perf_counter() - epoch_started,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if args.rollout_eval_every and (
                epoch % args.rollout_eval_every == 0 or epoch == args.epochs
            ):
                entry["validation_free_running_probe"] = evaluate(
                    model, probe_loader, device, historical=False
                )
                entry["epoch_seconds"] = time.perf_counter() - epoch_started
                entry["elapsed_seconds"] = time.perf_counter() - started
                print(
                    f"probe seed={seed} {condition} epoch={epoch} "
                    f"all_register={entry['validation_free_running_probe']['branched_full_state_exact_accuracy']:.4%} "
                    f"per_register={entry['validation_free_running_probe']['mean_register_query_accuracy']:.4%}",
                    flush=True,
                )
            history.write(json.dumps(entry) + "\n")
            history.flush()
            diagnostic = (
                f"state_KL={validation['state_kl']:.5f} first_update_TV={validation['first_update_tv']:.5f}"
                if condition == COT2
                else f"first_update={validation['first_update_exact_accuracy']:.2%}"
            )
            print(
                f"seed={seed} {condition} epoch={epoch} "
                f"state_train={entry['train_losses']['reasoning']:.5f} state_val={validation['reasoning']:.5f} "
                f"{diagnostic}",
                flush=True,
            )
            if args.patience and stale >= args.patience:
                break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(
        directory / "best.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    validate_model(model)
    metrics = {}
    for split in ("validation", "test"):
        metrics[split] = {
            "free_running": evaluate(model, loaders[split], device),
            "gold_feedback": evaluate(model, loaders[split], device, gold=True),
        }
    result = {
        "condition": condition,
        "seed": seed,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "checkpoint_selection": (
            "minimum validation teacher-forced state cross entropy (equivalent to KL for fixed distribution targets)"
            if condition == COT2
            else "minimum validation teacher-forced state cross entropy"
        ),
        "best_validation_losses": checkpoint["validation_losses"],
        "training_and_validation_seconds": training_seconds,
        "metrics": metrics,
        "runtime": runtime_info(device),
    }
    if model_metadata is not None:
        result["model_variant"] = model_metadata
    write_json(directory / "final_metrics.json", result)
    print(
        f"{condition}: test all-register accuracy="
        f"{metrics['test']['free_running']['branched_full_state_exact_accuracy']:.4%}",
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("baseline", "nonorthogonal"), default="baseline"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=data_dir("arithmetic_reasoning", "register")
    )
    parser.add_argument(
        "--output-root", type=Path, default=checkpoint_dir("arithmetic_reasoning", "baseline")
    )
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101])
    for name in ("d_model", "layers", "heads", "max_positions"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(ModelConfig(), name),
        )
    parser.add_argument("--dropout", type=float, default=0.0)
    for name, default in (
        ("batch_size", 64),
        ("epochs", 40),
        ("patience", 0),
        ("num_workers", 0),
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in (
        ("learning_rate", 0.001),
        ("min_learning_rate", 0.0001),
        ("weight_decay", 0.01),
        ("gradient_clip", 1.0),
        ("min_delta", 0.0001),
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument(
        "--num-registers",
        type=int,
        default=None,
        help="Optional assertion of the register count stored in the dataset manifest",
    )
    parser.add_argument(
        "--rollout-eval-every",
        type=int,
        default=0,
        help="Validation rollout interval; 0 disables diagnostic rollouts",
    )
    parser.add_argument("--rollout-eval-samples", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_training(
    args: argparse.Namespace | None = None,
    *,
    model_class: type[Model] = Model,
    model_metadata: dict | None = None,
    validate_model=assert_orthogonal_role_maps,
) -> None:
    """Run the common protocol with the selected model and validation hook."""
    args = parse_args() if args is None else args
    if args.rollout_eval_every < 0 or args.rollout_eval_samples < 1:
        raise ValueError("Invalid rollout diagnostic settings")
    if (
        min(args.batch_size, args.epochs) < 1
        or min(args.num_workers, args.patience) < 0
    ):
        raise ValueError("Invalid training lengths/worker count")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate model seeds")
    if not 0 <= args.min_learning_rate <= args.learning_rate or args.learning_rate <= 0:
        raise ValueError("Invalid learning rates")
    if args.gradient_clip <= 0 or min(args.weight_decay, args.min_delta) < 0:
        raise ValueError("Invalid gradient clipping/weight decay/min_delta")
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(args.cpu_threads)
    data_dir = args.data_dir.resolve()
    manifest = load_manifest(data_dir)
    data_config = load_data_config(manifest)
    vocabulary = Vocabulary(data_config)
    num_registers = len(vocabulary.roles)
    if args.num_registers is not None and args.num_registers != num_registers:
        raise ValueError("--num-registers differs from the dataset manifest")
    config = ModelConfig(
        **{key: getattr(args, key) for key in ModelConfig.__dataclass_fields__}
    )
    config.validate()
    prompt_lengths = {manifest["stats"][s]["prompt_positions"] for s in SPLITS}
    if (
        len(prompt_lengths) != 1
        or max(prompt_lengths) + num_registers * data_config.path_length + 3
        > config.max_positions
    ):
        raise ValueError(
            "Unequal prompt lengths or max_positions too small for conventional serialization"
        )
    device = select_device(args.device)
    conditions = list(CONDITIONS) if args.condition == "all" else [args.condition]
    for seed in args.seeds:
        for condition in conditions:
            if (args.output_root / f"seed_{seed}" / condition).exists():
                raise FileExistsError(
                    f"Existing run seed={seed}, condition={condition}; use a fresh output-root"
                )
    print("Loading and validating all dataset rows and split hashes...", flush=True)
    datasets = {s: EncodedSplit(data_dir, s, manifest, vocabulary) for s in SPLITS}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if datasets[left].fingerprints & datasets[right].fingerprints:
            raise ValueError("Episode overlap across splits")
    if not manifest["stats"]["train"]["all_role_pairs_observed"]:
        print(
            "NOTE: this dataset does not cover all constituent pairs in training; "
            "treat as a pilot, not a fully covered compositional experiment.",
            flush=True,
        )
    if not manifest["stats"]["train"]["all_primitive_cases_observed"]:
        print(
            "NOTE: incomplete arithmetic case coverage; treat as a pilot.", flush=True
        )
    hashes = source_hashes()
    for seed in args.seeds:
        output = args.output_root.resolve() / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        experiment = {
            "task": manifest["task"],
            "model_seed": seed,
            "model_config": asdict(config),
            "dataset_directory": str(data_dir),
            "data_manifest": manifest,
            "source_sha256": hashes,
            "training_config": {
                k: getattr(args, k)
                for k in (
                    "batch_size",
                    "epochs",
                    "patience",
                    "learning_rate",
                    "min_learning_rate",
                    "weight_decay",
                    "gradient_clip",
                    "min_delta",
                    "rollout_eval_every",
                    "rollout_eval_samples",
                )
            },
            "protocol": {
                "training": "full program prefix; causal teacher-forced trace; .5 state CE + .5 delayed-query CE; half queries final and half historical for L>1",
                "checkpoint_selection": "minimum validation state CE; patience 0 disables early stopping",
                "evaluation": "predicted feedback; full KV cache; K independent final-query branches primary; K historical branches diagnostic",
                "unbound": "feedback-only ablation; distinct role heads and norm retained",
                "cache_only": "same role heads and losses; state targets never enter feedback",
                "cot2": {
                    "condition": COT2,
                    "targets": "value multiplicities / K; one full-vocabulary head; no role maps",
                    "training_feedback": "gold distribution embedding; teacher forcing",
                    "inference_feedback": "full predicted softmax over entire vocabulary; no argmax, top-k or sampling",
                    "normalization": "same target node norm as other continuous controls (CoT2-style adaptation)",
                    "loss": ".5 distribution CE + .5 query CE; also log CE minus target entropy as KL",
                    "evaluation": "same independent delayed role queries; ordered trace accuracies unavailable; KL and TV diagnostics",
                },
                "limitations": "no learned discrete code; modular shifts and swaps; full future program visible; no pretraining claim",
            },
        }
        if model_metadata is not None:
            experiment["model_variant"] = model_metadata
        manifest_path = output / "manifest.json"
        if (
            manifest_path.exists()
            and json.loads(manifest_path.read_text()) != experiment
        ):
            raise ValueError("Existing seed manifest differs; choose a new output-root")
        write_json(manifest_path, experiment)
        write_json(output / "runtime.json", runtime_info(device))
        set_seed(seed)
        template = model_class("continuous", vocabulary, config)
        initial_state = {
            k: v.detach().clone() for k, v in template.state_dict().items()
        }
        del template
        for condition in conditions:
            train_condition(
                condition=condition,
                seed=seed,
                datasets=datasets,
                vocabulary=vocabulary,
                config=config,
                initial_state=initial_state,
                output=output,
                device=device,
                args=args,
                data_manifest=manifest,
                model_class=model_class,
                model_metadata=model_metadata,
                validate_model=validate_model,
            )
        comparison = {}
        for condition in CONDITIONS:
            path = output / condition / "final_metrics.json"
            if path.exists():
                result = json.loads(path.read_text())
                comparison[condition] = result["metrics"]["test"]["free_running"]
        write_json(output / "comparison.json", comparison)
    print(f"Wrote runs to {args.output_root.resolve()}")


def main(argv=None):
    """Train the baseline conditions or the ablation without QR projection."""
    import sys
    from .variants import (
        NonOrthogonalModel,
        assert_finite_role_maps,
        nonorthogonal_metadata,
        load_nonorthogonal_checkpoint,
        role_map_diagnostics,
    )

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    def supplied(flag):
        return any(x == flag or x.startswith(flag + "=") for x in argv)

    if args.variant == "baseline":
        return run_training(args)
    if not supplied("--condition"):
        args.condition = "continuous"
    if args.condition != "continuous":
        parser.error("the nonorthogonal variant requires --condition continuous")
    if not supplied("--output-root"):
        args.output_root = checkpoint_dir("arithmetic_reasoning", args.variant)
    run_training(
        args,
        model_class=NonOrthogonalModel,
        model_metadata=nonorthogonal_metadata(),
        validate_model=assert_finite_role_maps,
    )
    for seed in args.seeds:
        directory = args.output_root / f"seed_{seed}" / "continuous"
        model = load_nonorthogonal_checkpoint(directory / "best.pt")
        write_json(directory / "role_map_diagnostics.json", role_map_diagnostics(model))


if __name__ == "__main__":
    main()
