"""Evaluate final fixed-budget checkpoints, export shared metrics and generations."""

from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import html
import json
from pathlib import Path
import time
import numpy as np
import torch
from data import PackedBank, Windows, atomic_json, digest, load_manifest
from train import TextModel, sync
from model import mean_embedding_norm
from objective import encode_chunks
from choice_evaluation import (
    build_suite,
    check_checkpoint,
    choice_scores,
    score_metrics,
    summarize,
)
from generation import select_windows, discrete_rollout, sentence_rollouts

HERE = Path(__file__).resolve().parent


def checkpoint(path, suite, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    c = saved["config"]
    s = saved["state"]
    check_checkpoint(c, c["kind"], suite)
    if s["step"] != c["planned_updates"] or s["validation_pending"]:
        raise ValueError(f"Incomplete run: {path}")
    model = TextModel(c["architecture"], dropout=0).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    alpha = saved["alpha"].to(device) if c["kind"] == "sentence" else None
    return model, alpha, c, s


def score_run(model, alpha, kind, data, suite):
    bank = PackedBank(data, suite["split"] + "_bank")
    windows = Windows(data, suite["split"])
    records = []
    weight = model.embed_tokens.weight
    scale = (
        mean_embedding_norm(weight, torch.arange(len(weight) - 1, device=weight.device))
        if kind == "sentence"
        else None
    )
    for i, t in enumerate(suite["tasks"]):
        context, _ = windows.row(t["window_index"])
        records.append(
            t
            | score_metrics(
                choice_scores(
                    model, alpha, bank, context, t["candidate_ids"], kind, scale
                ),
                t["gold_index"],
            )
        )
        if (i + 1) % 256 == 0:
            print(
                json.dumps(
                    dict(
                        stage="choices",
                        split=suite["split"],
                        completed=i + 1,
                        total=len(suite["tasks"]),
                    )
                ),
                flush=True,
            )
    return records


@torch.inference_mode()
def qualitative(model, alpha, kind, data, indices, metadata, tok, out, budget, seed):
    bank = PackedBank(data, "validation_bank")
    windows = Windows(data, "validation")
    train_count = len(PackedBank(data, "train_bank"))
    contexts = [windows.row(i)[0] for i in indices]
    results = []
    vectors = None
    temporary = out / "catalogue_vectors.npy"
    if kind == "sentence":
        # Frozen checkpoint only: compose once, stream dot products from a disk-backed
        # array. Never reuse representations across training updates/checkpoints.
        vectors = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(train_count, model.embed_tokens.embedding_dim),
        )
        weight = model.embed_tokens.weight
        scale = mean_embedding_norm(
            weight, torch.arange(len(weight) - 1, device=weight.device)
        )
        trainbank = PackedBank(data, "train_bank")
        for lo, hi, tokens, offsets in trainbank.blocks(8192, weight.device):
            vectors[lo:hi] = (
                encode_chunks(weight, alpha, tokens, offsets, scale)
                .float()
                .cpu()
                .numpy()
            )
        vectors.flush()
    for mode in ("greedy", "sample"):
        if kind == "sentence":
            generations = sentence_rollouts(
                model, alpha, bank, vectors, contexts, train_count, mode, seed, budget
            )
        else:
            generations = [
                discrete_rollout(
                    model,
                    [t for c in ctx for t in bank.sequence(c)],
                    mode,
                    seed + j,
                    budget,
                )
                for j, ctx in enumerate(contexts)
            ]
        for i, generation in zip(indices, generations):
            context, target = windows.row(i)
            results.append(
                dict(
                    window_id=metadata[i]["window_id"],
                    source=metadata[i]["source"],
                    mode=mode,
                    context_chunks=[
                        tok.decode(bank.sequence(c), clean_up_tokenization_spaces=False)
                        for c in context
                    ],
                    reference_chunks=[
                        tok.decode(bank.sequence(c), clean_up_tokenization_spaces=False)
                        for c in target
                    ],
                    generated_text=tok.decode(
                        generation["token_ids"], clean_up_tokenization_spaces=False
                    ),
                    generated_chunks=[
                        tok.decode(bank.sequence(c), clean_up_tokenization_spaces=False)
                        for c in generation.get("chunk_ids", [])
                    ],
                    **generation,
                )
            )
    del vectors
    temporary.unlink(missing_ok=True)
    return results


def plot_learning(campaign, conditions, seeds, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.tab10.colors
    for k, condition in enumerate(conditions):
        for j, seed in enumerate(seeds):
            state = json.loads(
                (campaign / condition / str(seed) / "status.json").read_text()
            )
            records = state["validation_records"]
            kind = condition == "discrete"
            x = [r["step"] / state["planned_updates"] for r in records]
            y = [
                r["validation"]["nll_per_token"]
                if kind
                else r["validation"]["nll"]["all"]
                for r in records
            ]
            axes[0 if kind else 1].plot(
                x, y, color=colors[k], alpha=0.7, label=condition if j == 0 else None
            )
    for ax, title in zip(
        axes, ("Discrete: NLL per token", "SentenceDot: NLL per chunk")
    ):
        ax.set(
            xlabel="Fraction of one epoch", ylabel="Native validation NLL", title=title
        )
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Native losses have different units; compare within each panel")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=180)
    fig.savefig(out / "learning_curves.pdf")
    plt.close(fig)


def aggregate(outputs, conditions, seeds, draws):
    summary = {}
    paired = {}
    for condition in conditions:
        rows = [outputs[condition, str(s)] for s in seeds]
        metrics = {}
        for split in ("validation", "test"):
            groups = set.intersection(*(set(r["metrics"][split]) for r in rows))
            metrics[split] = {
                g: {
                    field: {
                        "mean": float(
                            np.mean(v := [r["metrics"][split][g][field] for r in rows])
                        ),
                        "seed_std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    }
                    for field in ("accuracy", "candidate_nll", "count")
                }
                for g in groups
            }
        summary[condition] = dict(
            metrics=metrics,
            training_seconds_mean=float(
                np.mean([r["state"]["training_seconds"] for r in rows])
            ),
            training_flops_mean=float(
                np.mean(
                    [
                        r["state"]["compute"]["training"]["accounted_flops_estimate"]
                        for r in rows
                    ]
                )
            ),
            validation_seconds_mean=float(
                np.mean([r["state"]["validation_seconds"] for r in rows])
            ),
            seeds=seeds,
        )
    if "discrete" in conditions:
        for condition in conditions:
            if condition == "discrete":
                continue
            by_doc = defaultdict(lambda: [0.0, 0.0])
            for seed in seeds:
                baseline = outputs["discrete", str(seed)]["records"]["test"]
                other = outputs[condition, str(seed)]["records"]["test"]
                if [r["window_id"] for r in baseline] != [
                    r["window_id"] for r in other
                ]:
                    raise ValueError("Unpaired task menus")
                for a, b in zip(baseline, other):
                    by_doc[a["document_id"]][0] += int(b["correct"]) - int(a["correct"])
                    by_doc[a["document_id"]][1] += 1
            values = np.array(list(by_doc.values()))
            rng = np.random.default_rng(314159)
            bootstrap = []
            for _ in range(draws):
                sampled = values[rng.integers(len(values), size=len(values))].sum(0)
                bootstrap.append(sampled[0] / sampled[1])
            paired[condition] = dict(
                accuracy_difference=float(values[:, 0].sum() / values[:, 1].sum()),
                document_bootstrap_95ci=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                documents=len(values),
                note="Document resampling, with training seeds averaged; not a confidence interval over all possible training seeds",
            )
    return dict(conditions=summary, paired_vs_discrete=paired)


def run(a):
    from transformers import AutoTokenizer

    config = json.loads(a.config.read_text())
    conditions = a.conditions or config["conditions"]
    seeds = a.seeds or config["seeds"]
    m = load_manifest(a.data)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    suites = {
        s: build_suite(
            a.data,
            s,
            config["evaluation"]["choices"],
            config["evaluation"]["seed"],
            m["diagnostic"],
        )
        for s in ("validation", "test")
    }
    for split, suite in suites.items():
        if not suite["tasks"]:
            raise ValueError(
                f"No eligible {split} tasks; inspect the data; do not change the frozen metric"
            )
        atomic_json(a.out / f"{split}_choices.json", suite)
    metadata = [json.loads(l) for l in (a.data / "validation_meta.jsonl").open()]
    indices = select_windows(
        metadata,
        config["evaluation"]["qualitative_examples"],
        config["evaluation"]["seed"],
    )
    tok = AutoTokenizer.from_pretrained(a.data / "tokenizer", local_files_only=True)
    outputs = {}
    expected_arch = None
    expected_training = None
    expected_orders = {}
    expected_runtime = expected_source = None
    for seed in seeds:
        for condition in conditions:
            path = a.campaign / condition / str(seed) / "final.pt"
            destination = a.out / condition / str(seed)
            destination.mkdir(parents=True, exist_ok=True)
            model, alpha, c, state = checkpoint(path, suites["test"], a.device)
            if c["condition"] != condition or c["seed"] != seed:
                raise ValueError("Checkpoint identity mismatch")
            if expected_arch is None:
                expected_arch = c["architecture"]
                expected_training = c["training"]
                expected_runtime = c["runtime"]
                expected_source = c["source_sha256"]
            if expected_arch != c["architecture"] or expected_training != c["training"]:
                raise ValueError("Unmatched model architecture or training settings")
            if (
                expected_runtime != c["runtime"]
                or expected_source != c["source_sha256"]
            ):
                raise ValueError(
                    "Training hardware/software or source differed across conditions"
                )
            history = [json.loads(l) for l in (path.parent / "history.jsonl").open()]
            order = [r["batch_window_indices_sha256"] for r in history]
            if seed in expected_orders and order != expected_orders[seed]:
                raise ValueError("Different sample ordering/exposure")
            expected_orders[seed] = order
            provenance = dict(
                checkpoint_sha256=digest(path),
                data_manifest_sha256=digest(a.data / "manifest.json"),
                evaluator_sha256={p.name: digest(p) for p in HERE.glob("*.py")},
                evaluation_config=config["evaluation"],
                device=a.device,
            )
            cache = destination / "result.json"
            if cache.exists():
                result = json.loads(cache.read_text())
                if result["provenance"] != provenance:
                    raise ValueError(
                        "Evaluation cache inputs changed; use a new output directory"
                    )
            else:
                before = time.monotonic()
                with torch.no_grad():
                    records = {
                        s: score_run(model, alpha, c["kind"], a.data, suite)
                        for s, suite in suites.items()
                    }
                sync(a.device)
                result = dict(
                    provenance=provenance,
                    config=c,
                    state=state,
                    records=records,
                    metrics={s: summarize(r) for s, r in records.items()},
                    choice_evaluation_seconds=time.monotonic() - before,
                )
                # A fixed seed provides the qualitative comparison; all three seeds
                # are scored quantitatively. Generation cannot select checkpoints.
                if seed == config["seeds"][0]:
                    result["qualitative"] = qualitative(
                        model,
                        alpha,
                        c["kind"],
                        a.data,
                        indices,
                        metadata,
                        tok,
                        destination,
                        config["evaluation"]["generation_tokens"],
                        config["evaluation"]["seed"],
                    )
                atomic_json(cache, result)
            outputs[condition, str(seed)] = result
            del model, alpha
            gc.collect()
            if a.device == "cuda":
                torch.cuda.empty_cache()
    report = dict(
        status="complete",
        diagnostic=m["diagnostic"],
        data_manifest_sha256=digest(a.data / "manifest.json"),
        coverage={s: t["coverage"] for s, t in suites.items()},
        **aggregate(
            outputs, conditions, seeds, config["evaluation"]["bootstrap_draws"]
        ),
        limitations=[
            "Observed continuation selection among eight supplied candidates, not unrestricted generation quality.",
            "Seen means membership in the train catalogue, not necessarily positive training exposure.",
            "Native token and chunk losses have different units. FLOPs are analytical operation estimates, with unsupported operators reported.",
            "Main SentenceDot generation uses only the training catalogue, so it cannot emit an entirely novel chunk.",
            "Balanced document counts do not imply balanced sample or token counts.",
        ],
    )
    atomic_json(a.out / "summary.json", report)
    plot_learning(a.campaign, conditions, seeds, a.out)
    table = [
        "condition\taccuracy_mean\taccuracy_seed_std\ttraining_seconds_mean\ttraining_flops_mean"
    ]
    for condition, r in report["conditions"].items():
        acc = r["metrics"]["test"]["all"]["accuracy"]
        table.append(
            f"{condition}\t{acc['mean']}\t{acc['seed_std']}\t{r['training_seconds_mean']}\t{r['training_flops_mean']}"
        )
    (a.out / "summary.tsv").write_text("\n".join(table) + "\n")
    content = [
        '<!doctype html><meta charset="utf-8"><title>Free generation</title><style>body{max-width:1200px;margin:40px auto;font:17px/1.5 system-ui}section{border-top:2px solid #aaa;padding:24px 0}p{white-space:pre-wrap}h3{color:#345}</style><h1>Free generation</h1><p>Identical validation prompts, seed 101, greedy and temperature 1. SentenceDot selects training chunks; the final chunk may exceed the displayed token budget.</p>'
    ]
    by_example = defaultdict(list)
    for (condition, seed), r in outputs.items():
        for row in r.get("qualitative", []):
            by_example[row["window_id"], row["mode"]].append((condition, row))
    for (key, mode), rows in by_example.items():
        row = rows[0][1]
        content.append(
            f"<section><h2>{html.escape(row['source'])} · {mode}</h2><small>{key}</small><h3>Context</h3><p>{html.escape(' | '.join(row['context_chunks']))}</p><h3>Reference</h3><p>{html.escape(' | '.join(row['reference_chunks']))}</p>"
        )
        for condition, row in rows:
            content.append(
                f"<h3>{condition}</h3><p>{html.escape(' | '.join(row['generated_chunks']) if row['generated_chunks'] else row['generated_text'])}</p>"
            )
        content.append("</section>")
    (a.out / "qualitative.html").write_text("\n".join(content))
    print(json.dumps(dict(status="complete", out=str(a.out))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "campaign", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--config", type=Path, default=HERE / "experiment.json")
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--threads", type=int, default=4)
    run(p.parse_args())
