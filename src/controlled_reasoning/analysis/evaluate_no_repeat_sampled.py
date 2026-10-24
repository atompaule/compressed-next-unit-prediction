"""Paired temperature-one no-repeat evaluation of the frozen six-inference models.

Uses the original two world-based sampling seeds, CPU sampling, selected
checkpoints, candidate sets, and stopping rules. The existing greedy diagnostic
and historical evaluation files are read-only. A complete run must reproduce
every stored standard sampled output before its no-repeat result is accepted.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import gc
import time

import torch

from evaluate_no_repeat import (
    ROOT, BASE, checked_source_hashes, SOURCE, model, runtime, score, aggregate, read, rows, sha, write,
)

from src.common.experiment_paths import data_dir, result_dir

OUT = result_dir("controlled_reasoning", "compressed_premises", "no_repeat_sampled")
DATA = data_dir("controlled_reasoning", "full", "6_conclusions")


def sampling_seed(world_id, sample):
    return 7000 + int(world_id[:8], 16) * 3 + sample


@torch.inference_mode()
def generate(m, row, cat, inv, table, seed, no_repeat):
    rng = torch.Generator().manual_seed(seed)
    h, cache = m.start(row["prompt_ids"])
    active = m.candidate_ids().tolist()
    used, tokens, sentences = [], [], []
    done = False
    while len(sentences) < 12 and len(tokens) < 96:
        logits = (h @ table.T).log_softmax(-1)[0]
        if no_repeat:
            logits[used] = -torch.inf
        # Match runtime.generate exactly; masking only changes eligible outputs.
        local = int(torch.multinomial(logits.detach().cpu().softmax(-1), 1, generator=rng))
        unit = cat[active[local]]["token_ids"]
        if unit == [3]:
            tokens.append(3)
            done = True
            break
        if len(tokens) + len(unit) > 96:
            break
        used.append(local)
        tokens.extend(unit)
        sentences.append(" ".join(inv[t] for t in unit if t != 2))
        h, cache = m.accept(unit, cache)
    return dict(ids=tokens, sentences=sentences, done=done, unfinished=[],
                temperature=1.0, seed=seed, tokens=len(tokens))


@torch.inference_mode()
def reference_no_repeat(m, row, cat, inv, seed):
    """Independently apply the mask through the frozen original generator."""
    active = m.candidate_ids().tolist()
    local_by_tokens = {tuple(cat[c]["token_ids"]): i for i, c in enumerate(active)}
    used = set()
    original_plan, original_accept = m.plan_log_probs, m.accept

    def masked_plan(h):
        logits = original_plan(h).clone()
        if used:
            logits[:, sorted(used)] = -torch.inf
        return logits

    def record_accept(tokens, cache):
        used.add(local_by_tokens[tuple(tokens)])
        return original_accept(tokens, cache)

    m.plan_log_probs, m.accept = masked_plan, record_accept
    try:
        return runtime.generate(m, "sentence", row, inv, cat, 1.0, seed)
    finally:
        m.plan_log_probs, m.accept = original_plan, original_accept


def evaluate_job(job, protocol_hash, threads):
    torch.set_num_threads(threads)
    started = time.monotonic()
    run = BASE / "extended_full" / job["run"]
    assert sha(run / "best.pt") == job["checkpoint_sha256"]
    cat, masks, vocabulary = (read(DATA / name) for name in
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
    original_features = m.candidate_features
    with torch.inference_mode():
        m.set_phase("validation")
        table = original_features()
        m.candidate_features = lambda: table
        selected = {r["world_id"] for r in runtime.subset(validation, 128)}
        probe = runtime.subset([r for r in validation if r["world_id"] not in selected], 32)
        for row in probe:
            for sample in (1, 2):
                seed = sampling_seed(row["world_id"], sample)
                assert generate(m, row, cat, inv, table, seed, False) == runtime.generate(
                    m, "sentence", row, inv, cat, 1.0, seed)
                assert generate(m, row, cat, inv, table, seed, True) == reference_no_repeat(
                    m, row, cat, inv, seed)
        m.set_phase("test")
        table = original_features()
        stored = read(run / "test_generations.json")
        originals = {(r["world_id"], r["generation"]["seed"]): r["generation"]
                     for r in stored["items"] if r["generation"]["temperature"] == 1.0}
        assert len(originals) == 2048
        normal_scores, items = [], []
        for row in test:
            wid = row["world_id"]
            for sample in (1, 2):
                seed = sampling_seed(wid, sample)
                normal = generate(m, row, cat, inv, table, seed, False)
                assert normal == originals[wid, seed], (job["run"], wid, seed, "Stored sampled output differs")
                normal_scores.append(score(normal["sentences"], worlds[wid], normal["done"]))
                output = generate(m, row, cat, inv, table, seed, True)
                assert len(output["sentences"]) == len(set(output["sentences"]))
                assert len(output["ids"]) <= 96 and len(output["sentences"]) <= 12
                items.append(dict(world_id=wid, sample=sample, generation=output,
                                  score=score(output["sentences"], worlds[wid], output["done"])))
    baseline, metrics = aggregate(normal_scores), aggregate([r["score"] for r in items])
    assert baseline == final["generation"]["test"]["sampled"] == stored["sampled"]
    assert metrics["duplicate_fraction"] == 0 and metrics["samples"] == 2048
    assert sha(run / "best.pt") == job["checkpoint_sha256"]
    result = dict(**job, selected_epoch=saved["epoch"], device="cpu",
                  protocol_sha256=protocol_hash, seconds=time.monotonic() - started,
                  reference_outputs_matched=4 * len(probe), stored_test_outputs_matched=2048,
                  normal=baseline, no_repeat=metrics, items=items)
    write(OUT / (job["run"] + ".json"), result)
    del m, saved, table
    gc.collect()
    return {k: v for k, v in result.items() if k != "items"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=20)
    args = parser.parse_args()
    assert args.workers > 0 and args.threads > 0 and 0 < args.max_runs <= 20
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_hashes = checked_source_hashes()
    prior = read(OUT.parent / "no_repeat/protocol.json")
    jobs = prior["jobs"]
    for job in jobs:
        assert sha(BASE / "extended_full" / job["run"] / "best.pt") == job["checkpoint_sha256"]
    protocol = dict(
        conclusions=6, seeds=list(range(101, 106)), layers=[2, 4], conditions=["token", "chunk"],
        split="test", worlds=1024, samples_per_world=2, temperature=1.0,
        sampling_seed="7000 + int(world_id[:8], 16) * 3 + sample; sample in {1, 2}.",
        decoding="CPU torch.multinomial with per-completion generator; mask emitted catalogue IDs and renormalize; DONE stays eligible.",
        stopping="Original 96-token/12-sentence limits; no forced conclusion count.",
        selection="Same original validation-selected best.pt as standard and greedy no-repeat; no training or reselection.",
        scoring="Original exact-completion, precision, recall and reserved-recall criteria; five-seed mean and sample SD.",
        verification="Serial CPU outputs must reproduce all stored standard sampled outputs; both decoders checked against frozen generator on 32 validation worlds per model.",
        source_sha256=sha(Path(__file__)), helper_sha256=sha(Path(__file__).with_name("evaluate_no_repeat.py")),
        frozen_source_hashes=frozen_hashes, source_kind="maintained",
        data_hashes={name: sha(DATA / name) for name in prior["data_hashes"]},
        baseline_hashes={job["run"]: {name: sha(BASE / "extended_full" / job["run"] / name)
                                     for name in ("final_metrics.json", "test_generations.json")}
                         for job in jobs}, jobs=jobs,
    )
    assert protocol["data_hashes"] == prior["data_hashes"]
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == protocol, "Frozen sampled diagnostic protocol changed"
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
    pending = pending[:max(0, args.max_runs - len(results))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_job, job, digest, args.threads) for job in pending]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            results.sort(key=lambda r: (r["layers"], r["seed"], r["condition"]))
            write(OUT / "progress.json", dict(completed=len(results), total=len(jobs), runs=results))
            print(result["run"], "seconds", round(result["seconds"], 1), "exact",
                  round(100 * result["normal"]["exact_rate"], 2), "->",
                  round(100 * result["no_repeat"]["exact_rate"], 2),
                  "completed", len(results), "/", len(jobs), flush=True)
    status = "complete" if len(results) == len(jobs) else "partial"
    write(OUT / "summary.json", dict(status=status, protocol_sha256=digest, runs=results))
    print(status.upper(), len(results), "runs;", 2048 * len(results), "sampled completions", flush=True)


if __name__ == "__main__":
    main()
