"""Frozen, exact-length chunk continuation choices for paired GPT-2 evaluation.

Build on validation before freezing the protocol; build test menus only afterwards.
No training, full-catalogue search, or model-dependent distractor selection.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SHARED = HERE
from data import PackedBank, Windows, atomic_json, digest, load_manifest
from evaluation import token_candidate_scores
from model import mean_embedding_norm


POLICY = "first_chunk_exact_token_length_source_membership_v1"
DEFAULT_CHOICES = 8
DEFAULT_SEED = 314159


def corpus(folder, split, diagnostic=False):
    if split not in ("validation", "test"):
        raise ValueError("Choices require a held-out split")
    manifest = load_manifest(folder)
    if not diagnostic and not (
        manifest.get("status") == "complete"
        and manifest.get("vocab_size") == 50257
        and manifest.get("whole_window_tokenization_verified") is True
        and manifest.get("token_boundary_policy") == "canonical_gpt2_whole_window_v1"
    ):
        raise ValueError("A completed canonical GPT-2 corpus audit is required")
    names = ["train_bank_tokens.npy", "train_bank_offsets.npy"]
    names += [f"{split}_bank_{part}.npy" for part in ("tokens", "offsets")]
    names += [
        f"{split}_{part}{suffix}.npy"
        for part in ("context", "target")
        for suffix in ("", "_offsets")
    ]
    names += [f"{split}_meta.jsonl"]
    fingerprints = {name: digest(folder / name) for name in names}
    if not diagnostic:
        if any(
            manifest["files"].get(name) != value for name, value in fingerprints.items()
        ):
            raise ValueError("Corpus content differs from its manifest")
        if digest(folder / manifest["audit_file"]) != manifest["audit_sha256"]:
            raise ValueError("Corpus audit fingerprint mismatch")
    bank, train = PackedBank(folder, f"{split}_bank"), PackedBank(folder, "train_bank")
    if not (
        np.array_equal(bank.offsets[: len(train) + 1], train.offsets)
        and np.array_equal(bank.tokens[: len(train.tokens)], train.tokens)
    ):
        raise ValueError("Held-out bank must start with the unchanged training bank")
    windows = Windows(folder, split)
    metadata = [json.loads(line) for line in windows.meta.read_text().splitlines()]
    if len(metadata) != len(windows) or len({m["window_id"] for m in metadata}) != len(
        windows
    ):
        raise ValueError("Window metadata missing or duplicated")
    return bank, windows, metadata, len(train), fingerprints


def build_suite(
    folder,
    split="validation",
    choices=DEFAULT_CHOICES,
    seed=DEFAULT_SEED,
    diagnostic=False,
):
    if choices < 2 or seed < 0:
        raise ValueError("At least two choices and a nonnegative seed are required")
    bank, windows, meta, train_count, fingerprints = corpus(folder, split, diagnostic)
    lengths = np.diff(bank.offsets)
    pools, documents = defaultdict(set), defaultdict(set)
    for index, item in enumerate(meta):
        context, targets = windows.row(index)
        if (
            not context
            or not targets
            or any(c < 0 or c >= len(bank) for c in context + targets)
        ):
            raise ValueError("Invalid context or target references")
        for c in targets:
            pools[item["source"], int(lengths[c]), c < train_count].add(c)
        for c in context + targets:
            documents[c].add(item["document_id"])
    # Distinct token sequences, not frequency-weighted target occurrences.
    sequences = {c: tuple(bank.sequence(c)) for pool in pools.values() for c in pool}
    if len(set(sequences.values())) != len(sequences):
        raise ValueError("Duplicate target text classes in the chunk catalogue")
    pools = {key: sorted(ids) for key, ids in pools.items()}
    tasks, skipped = [], []
    coverage = defaultdict(Counter)
    for index, item in enumerate(meta):
        context, targets = windows.row(index)
        gold = targets[0]
        length, seen = int(lengths[gold]), gold < train_count
        if length <= 0:
            raise ValueError("Empty candidate")
        key = item["source"], length, seen
        counts = coverage[key]
        counts["total"] += 1
        # Symmetric copy exclusion: a gold already in the prompt is skipped too.
        reason = "gold_in_context" if gold in context else None
        eligible = [
            c
            for c in pools[key]
            if c != gold
            and c not in context
            and item["document_id"] not in documents[c]
        ]
        if reason is None and len(eligible) < choices - 1:
            reason = "insufficient_exact_length_alternatives"
        if reason:
            counts[reason] += 1
            skipped.append(
                {
                    "window_index": index,
                    "window_id": item["window_id"],
                    "reason": reason,
                    "eligible_alternatives": len(eligible),
                }
            )
            continue
        identity = f"{POLICY}:{split}:{seed}:{item['window_id']}"
        rng = np.random.default_rng(
            int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
        )
        candidates = [gold] + [
            int(c) for c in rng.choice(eligible, choices - 1, replace=False)
        ]
        rng.shuffle(candidates)
        counts["included"] += 1
        tasks.append(
            {
                "window_index": index,
                "window_id": item["window_id"],
                "source": item["source"],
                "document_id": item["document_id"],
                "target_position": 0,
                "token_count": length,
                "seen_in_training_catalogue": seen,
                "candidate_ids": candidates,
                "gold_index": candidates.index(gold),
            }
        )
    return {
        "policy": POLICY,
        "split": split,
        "choices": choices,
        "seed": seed,
        "diagnostic": diagnostic,
        "data_manifest_sha256": digest(folder / "manifest.json"),
        "data_files_sha256": fingerprints,
        "coverage": {
            "total_windows": len(windows),
            "included": len(tasks),
            "skipped": len(skipped),
            "by_source_length_membership": [
                {
                    "source": key[0],
                    "token_count": key[1],
                    "seen_in_training_catalogue": key[2],
                    **dict(counts),
                }
                for key, counts in sorted(coverage.items())
            ],
        },
        "tasks": tasks,
        "skipped": skipped,
    }


class CandidateBank:
    """Small view for the existing cached autoregressive token scorer."""

    def __init__(self, bank, ids):
        self.bank, self.ids = bank, ids

    def __len__(self):
        return len(self.ids)

    def sequence(self, i):
        return self.bank.sequence(self.ids[i])


@torch.no_grad()
def choice_scores(model, alpha, bank, context, candidates, kind, scale=None):
    model.eval()
    lengths = {len(bank.sequence(c)) for c in candidates}
    if len(lengths) != 1 or 0 in lengths or len(set(candidates)) != len(candidates):
        raise ValueError(
            "Candidates must be distinct and have exactly equal positive token lengths"
        )
    prompt = [t for c in context for t in bank.sequence(c)]
    if (
        not prompt
        or len(prompt) + next(iter(lengths)) - 1 > model.backbone.config.n_positions
    ):
        raise ValueError(
            "Empty context or window exceeds model capacity; no truncation"
        )
    if kind == "discrete":
        selected = CandidateBank(bank, candidates)
        return torch.cat(
            [
                scores
                for _, scores in token_candidate_scores(
                    model, prompt, selected, batch=len(candidates)
                )
            ]
        ).float()
    if kind != "sentence":
        raise ValueError("Unknown model kind")
    weight = model.embed_tokens.weight
    if scale is None:
        lexical = torch.arange(len(weight) - 1, device=weight.device)
        scale = mean_embedding_norm(weight, lexical)
    # The prediction state only sees the prompt; no gold or candidate is fed back.
    inputs = bank.encode_ids(context, weight, alpha, scale)[None]
    h = model(inputs_embeds=inputs, use_cache=False).last_hidden_state[0, -1].float()
    return bank.encode_ids(candidates, weight, alpha, scale) @ h


def score_metrics(scores, gold):
    scores = scores.detach().cpu().double()
    if scores.ndim != 1 or not torch.isfinite(scores).all():
        raise ValueError("Invalid candidate scores")
    logp = scores.log_softmax(0)
    prediction = int(scores.argmax())
    return {
        "scores": scores.tolist(),
        "probabilities": logp.exp().tolist(),
        "predicted_index": prediction,
        "correct": prediction == gold,
        "nll": float(-logp[gold]),
        "top_tie_count": int((scores == scores.max()).sum()),
    }


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups["all"].append(row)
        groups["seen" if row["seen_in_training_catalogue"] else "unseen"].append(row)
        groups["source:" + row["source"]].append(row)
    return {
        key: {
            "count": len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "candidate_nll": math.fsum(r["nll"] for r in rows) / len(rows),
            "top_tie_rate": sum(r["top_tie_count"] > 1 for r in rows) / len(rows),
        }
        for key, rows in sorted(groups.items())
    }


def check_checkpoint(config, kind, suite):
    if config["kind"] != kind or config.get("chunk_boundary") != "metadata_only":
        raise ValueError("Wrong checkpoint kind or boundary convention")
    if config.get("diagnostic", False) != suite["diagnostic"]:
        raise ValueError("Diagnostic/production mismatch")
    if config["data_sha256"].get("manifest.json") != suite["data_manifest_sha256"]:
        raise ValueError("Checkpoint and choices belong to different corpora")
    for name, value in suite["data_files_sha256"].items():
        # Test arrays are deliberately absent from training fingerprints.
        if name in config["data_sha256"] and config["data_sha256"][name] != value:
            raise ValueError(f"Checkpoint data mismatch: {name}")
    if not suite["diagnostic"] and config["architecture"]["vocab_size"] != 50257:
        raise ValueError("Original GPT-2 vocabulary required")
    for name in ("model.py", "objective.py"):
        key = name
        if config.get("source_sha256", {}).get(key) != digest(SHARED / name):
            raise ValueError(f"Checkpoint encoding differs from evaluator: {name}")


def evaluate_pair(folder, suite_path, discrete_path, sentence_path, out, device="cpu"):
    from train import TextModel

    if out.exists():
        raise FileExistsError("Use a new evaluation output directory")
    suite = json.loads(suite_path.read_text())
    if suite != build_suite(
        folder, suite["split"], suite["choices"], suite["seed"], suite["diagnostic"]
    ):
        raise ValueError(
            "Choices or corpus changed; regenerate with the frozen protocol"
        )
    if not suite["tasks"]:
        raise ValueError("No eligible tasks; inspect coverage without relaxing lengths")
    bank = PackedBank(folder, f"{suite['split']}_bank")
    windows = Windows(folder, suite["split"])
    configs, provenance, metrics, results = {}, {}, {}, {}
    # Validate both checkpoints before evaluating either arm.
    for kind, path in (("discrete", discrete_path), ("sentence", sentence_path)):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        config = saved["config"]
        check_checkpoint(config, kind, suite)
        configs[kind] = config
        provenance[kind] = {
            "path": str(path.resolve()),
            "sha256": digest(path),
            "step": saved["state"]["step"],
            "seed": config["seed"],
        }
        del saved
    for field in ("seed", "architecture", "data_sha256"):
        if configs["discrete"][field] != configs["sentence"][field]:
            raise ValueError(f"Paired checkpoints differ in {field}")
    out.mkdir(parents=True)
    with torch.no_grad():
        for kind, path in (("discrete", discrete_path), ("sentence", sentence_path)):
            if digest(path) != provenance[kind]["sha256"]:
                raise ValueError("Checkpoint changed during evaluation")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model = TextModel(configs[kind]["architecture"], dropout=0.0).to(device)
            model.load_state_dict(saved["model"])
            model.eval()
            alpha = saved["alpha"].to(device) if kind == "sentence" else None
            del saved
            lexical = torch.arange(model.embed_tokens.num_embeddings - 1, device=device)
            scale = (
                mean_embedding_norm(model.embed_tokens.weight, lexical)
                if kind == "sentence"
                else None
            )
            records = []
            with (out / f"{kind}.jsonl").open("w") as stream:
                for index, task in enumerate(suite["tasks"]):
                    context, _ = windows.row(task["window_index"])
                    scores = choice_scores(
                        model, alpha, bank, context, task["candidate_ids"], kind, scale
                    )
                    row = task | score_metrics(scores, task["gold_index"])
                    stream.write(json.dumps(row) + "\n")
                    records.append(row)
                    if (index + 1) % 128 == 0 or index + 1 == len(suite["tasks"]):
                        stream.flush()
                        print(
                            json.dumps(
                                {
                                    "stage": "choice_evaluation",
                                    "split": suite["split"],
                                    "kind": kind,
                                    "completed": index + 1,
                                    "total": len(suite["tasks"]),
                                }
                            ),
                            flush=True,
                        )
            results[kind] = records
            metrics[kind] = summarize(records)
            del model, alpha, scale, lexical, scores
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
    wins = Counter()
    for d, s in zip(results["discrete"], results["sentence"]):
        wins[f"discrete_{int(d['correct'])}_sentence_{int(s['correct'])}"] += 1
    report = {
        "status": "complete",
        "policy": POLICY,
        "split": suite["split"],
        "diagnostic": suite["diagnostic"],
        "device": device,
        "suite_sha256": digest(suite_path),
        "data_manifest_sha256": suite["data_manifest_sha256"],
        "coverage": suite["coverage"],
        "choices": suite["choices"],
        "primary_metric": "accuracy",
        "secondary_metric": "candidate_nll",
        "chance_accuracy": 1 / suite["choices"],
        "uniform_candidate_nll": math.log(suite["choices"]),
        "temperature": 1.0,
        "length_normalization": "none; exact equal token counts within tasks",
        "tie_policy": "first maximum in frozen pseudorandom option order",
        "checkpoints": provenance,
        "metrics": metrics,
        "paired_correctness_counts": dict(wins),
        "sentence_minus_discrete_accuracy": metrics["sentence"]["all"]["accuracy"]
        - metrics["discrete"]["all"]["accuracy"],
        "source_sha256": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                HERE / "train.py",
                SHARED / "model.py",
                SHARED / "objective.py",
                SHARED / "evaluation.py",
                SHARED / "data.py",
            )
        },
        "limitations": [
            "Observed continuation retrieval among supplied chunks; not unrestricted text generation.",
            "Other continuations may also be plausible; gold is the observed corpus continuation.",
            "Native token/chunk losses are not compared; candidate NLL measures this choice distribution and calibration.",
            "Coverage excludes copy targets and tasks without enough exact-length alternatives.",
            "Catalogue membership is not identical to positive training exposure.",
            "Overlapping windows/documents are not independent observations.",
        ],
    }
    atomic_json(out / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--data", type=Path, required=True)
    build.add_argument(
        "--out", type=Path, required=True, help="New JSON task-list file"
    )
    build.add_argument("--split", choices=("validation", "test"), default="validation")
    build.add_argument("--choices", type=int, default=DEFAULT_CHOICES)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--diagnostic", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--suite", type=Path, required=True)
    evaluate.add_argument("--discrete-checkpoint", type=Path, required=True)
    evaluate.add_argument("--sentence-checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--out", type=Path, required=True, help="New result directory"
    )
    evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    if args.command == "build":
        if args.out.exists():
            raise FileExistsError("Task lists are immutable; use a new path")
        suite = build_suite(
            args.data, args.split, args.choices, args.seed, args.diagnostic
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out, suite)
        print(json.dumps({"out": str(args.out), "coverage": suite["coverage"]}))
    else:
        torch.set_num_threads(2)
        report = evaluate_pair(
            args.data,
            args.suite,
            args.discrete_checkpoint,
            args.sentence_checkpoint,
            args.out,
            args.device,
        )
        print(json.dumps({"out": str(args.out), "metrics": report["metrics"]}))


if __name__ == "__main__":
    main()
