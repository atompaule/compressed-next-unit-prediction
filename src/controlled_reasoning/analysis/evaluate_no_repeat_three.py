"""Extend the frozen no-repeat protocol to the three-inference task.

Evaluate greedy and both original temperature-one samples on every test world.
No training, checkpoint selection, or changes to six-inference diagnostics.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import time

import torch

from evaluate_no_repeat import (
    ROOT, BASE, checked_source_hashes, model, runtime, score, aggregate, read, rows, sha, write, serial,
)
from evaluate_no_repeat_sampled import generate, reference_no_repeat, sampling_seed

from src.common.experiment_paths import data_dir, result_dir

OUT = result_dir("controlled_reasoning", "compressed_premises", "no_repeat_three")
DATA = data_dir("controlled_reasoning", "full", "3_conclusions")


@torch.inference_mode()
def reference_greedy(m, row, cat, inv):
    """Mask the frozen original generator, independently of serial()."""
    active = m.candidate_ids().tolist()
    local = {tuple(cat[g]["token_ids"]): i for i, g in enumerate(active)}
    used = set()
    plan, accept = m.plan_log_probs, m.accept

    def masked(h):
        logits = plan(h).clone()
        if used:
            logits[:, sorted(used)] = -torch.inf
        return logits

    def record(tokens, cache):
        used.add(local[tuple(tokens)])
        return accept(tokens, cache)

    m.plan_log_probs, m.accept = masked, record
    try:
        return runtime.generate(m, "sentence", row, inv, cat, 0.0, 0)
    finally:
        m.plan_log_probs, m.accept = plan, accept


def evaluate_job(job, protocol_hash, threads):
    torch.set_num_threads(threads)
    started = time.monotonic()
    run = BASE / "full" / job["run"]
    assert sha(run / "best.pt") == job["checkpoint_sha256"]
    cat, masks, vocabulary = (read(DATA / n) for n in
                              ("catalogue.json", "candidate_ids_by_phase.json", "vocabulary.json"))
    inv = {i: token for token, i in vocabulary.items()}
    worlds = {r["world_id"]: r for r in rows(DATA / "worlds.jsonl")}
    test, validation = rows(DATA / "test.jsonl"), rows(DATA / "validation.jsonl")
    assert len(test) == len({r["world_id"] for r in test}) == 1024
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    m = model.SentenceDot(cat, masks, len(vocabulary), 512, job["layers"],
                          prompt_mode=job["condition"])
    m.load_state_dict(saved["state"])
    m.eval().requires_grad_(False)
    final = read(run / "final_metrics.json")
    assert saved["epoch"] == final["selected_epoch"]
    features = m.candidate_features
    with torch.inference_mode():
        m.set_phase("validation")
        table = features()
        m.candidate_features = lambda: table
        selected = {r["world_id"] for r in runtime.subset(validation, 128)}
        probe = runtime.subset([r for r in validation if r["world_id"] not in selected], 32)
        for row in probe:
            for mask in (False, True):
                actual = serial(m, row, cat, inv, table, mask)
                expected = (reference_greedy(m, row, cat, inv) if mask else
                            runtime.generate(m, "sentence", row, inv, cat, 0.0, 0))
                assert all(actual[k] == expected[k] for k in actual)
            for sample in (1, 2):
                seed = sampling_seed(row["world_id"], sample)
                assert generate(m, row, cat, inv, table, seed, False) == runtime.generate(
                    m, "sentence", row, inv, cat, 1.0, seed)
                assert generate(m, row, cat, inv, table, seed, True) == reference_no_repeat(
                    m, row, cat, inv, seed)
        m.set_phase("test")
        table = features()
        stored = read(run / "test_generations.json")
        originals = {(r["world_id"], r["generation"]["seed"]): r["generation"]
                     for r in stored["items"]}
        assert len(originals) == 3072
        normal_scores = {mode: [] for mode in ("greedy", "sampled")}
        items = []
        for row in test:
            wid = row["world_id"]
            for sample in (0, 1, 2):
                seed = sampling_seed(wid, sample)
                mode = "greedy" if sample == 0 else "sampled"
                if sample == 0:
                    normal = serial(m, row, cat, inv, table, False)
                    output = serial(m, row, cat, inv, table, True)
                    for g in (normal, output):
                        g.update(temperature=0.0, seed=seed, tokens=len(g["ids"]))
                else:
                    normal = generate(m, row, cat, inv, table, seed, False)
                    output = generate(m, row, cat, inv, table, seed, True)
                assert normal == originals[wid, seed], (job["run"], wid, seed)
                normal_scores[mode].append(score(normal["sentences"], worlds[wid], normal["done"]))
                assert len(output["sentences"]) == len(set(output["sentences"]))
                assert len(output["ids"]) <= 96 and len(output["sentences"]) <= 12
                items.append(dict(world_id=wid, sample=sample, decoding=mode, generation=output,
                                  score=score(output["sentences"], worlds[wid], output["done"])))
    baseline = {mode: aggregate(values) for mode, values in normal_scores.items()}
    metrics = {mode: aggregate([r["score"] for r in items if r["decoding"] == mode])
               for mode in normal_scores}
    assert baseline == final["generation"]["test"]
    for mode, count in (("greedy", 1024), ("sampled", 2048)):
        assert baseline[mode] == stored[mode]
        assert metrics[mode]["duplicate_fraction"] == 0 and metrics[mode]["samples"] == count
    assert sha(run / "best.pt") == job["checkpoint_sha256"]
    result = dict(**job, selected_epoch=saved["epoch"], device="cpu",
                  protocol_sha256=protocol_hash, seconds=time.monotonic() - started,
                  reference_outputs_matched=6 * len(probe), stored_test_outputs_matched=3072,
                  normal=baseline, no_repeat=metrics, items=items)
    write(OUT / (job["run"] + ".json"), result)
    return {k: v for k, v in result.items() if k != "items"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    assert args.workers > 0 and args.threads > 0
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = checked_source_hashes()
    jobs = []
    for layers in (2, 4):
        for seed in range(101, 106):
            for condition in ("token", "chunk"):
                name = f"sentence_3_seed{seed}" + ("_4l" if layers == 4 else "")
                name += "_chunked_prompt" if condition == "chunk" else ""
                jobs.append(dict(run=name, layers=layers, seed=seed, condition=condition,
                                 checkpoint_sha256=sha(BASE / "full" / name / "best.pt")))
    protocol = dict(
        conclusions=3, seeds=list(range(101, 106)), layers=[2, 4], conditions=["token", "chunk"],
        split="test", worlds=1024, decoders=["greedy", "sampled"], samples_per_world=2,
        sampling_temperature=1.0,
        sampling_seed="7000 + int(world_id[:8], 16) * 3 + sample; sample in {1, 2}.",
        decoding="Same frozen serial decoders as six-inference diagnostics; mask emitted catalogue IDs; DONE stays eligible.",
        stopping="Original 96-token/12-sentence limits; no forced conclusion count.",
        selection="Original validation-selected best.pt after 120 training epochs; no training or reselection.",
        scoring="Original exact-completion, precision, recall and reserved-recall criteria; five-seed mean and sample SD.",
        verification="Reproduce all 61,440 standard completions; independently mask the original generator on 32 validation worlds per model.",
        source_hashes={name: sha(Path(__file__).with_name(name)) for name in
                       (Path(__file__).name, "evaluate_no_repeat.py", "evaluate_no_repeat_sampled.py")},
        frozen_source_hashes=frozen, source_kind="maintained",
        data_hashes={name: sha(DATA / name) for name in
                     ("test.jsonl", "validation.jsonl", "worlds.jsonl", "catalogue.json",
                      "candidate_ids_by_phase.json", "vocabulary.json")},
        baseline_hashes={job["run"]: {name: sha(BASE / "full" / job["run"] / name)
                                     for name in ("final_metrics.json", "test_generations.json")}
                         for job in jobs}, jobs=jobs,
    )
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == protocol, "Frozen three-inference protocol changed"
    else:
        write(path, protocol)
    digest, results, pending = sha(path), [], []
    for job in jobs:
        target = OUT / (job["run"] + ".json")
        if target.exists():
            result = read(target)
            assert result["protocol_sha256"] == digest
            results.append({k: v for k, v in result.items() if k != "items"})
        else:
            pending.append(job)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_job, job, digest, args.threads) for job in pending]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda r: (r["layers"], r["seed"], r["condition"]))
            write(OUT / "progress.json", dict(completed=len(results), total=len(jobs), runs=results))
            print(result["run"], "seconds", round(result["seconds"], 1),
                  {mode: [round(100 * result[k][mode]["exact_rate"], 2)
                          for k in ("normal", "no_repeat")] for mode in ("greedy", "sampled")},
                  "completed", len(results), "/", len(jobs), flush=True)
    assert len(results) == len(jobs) == 20
    write(OUT / "summary.json", dict(status="complete", protocol_sha256=digest, runs=results))
    print("COMPLETE", len(results), "checkpoints; 61,440 no-repeat completions", flush=True)


if __name__ == "__main__":
    main()
