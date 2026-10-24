"""Frozen-checkpoint greedy no-repeat diagnostic; never trains or reselects models.

The protocol is written before test inference. Serial decoding avoids batch-size
rounding differences and checks every standard output against the stored result.
"""
from pathlib import Path
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from src.common.experiment_paths import data_dir, checkpoint_dir, result_dir
BASE = checkpoint_dir("controlled_reasoning")
SOURCE = ROOT / "src/controlled_reasoning"
OUT = result_dir("controlled_reasoning", "compressed_premises", "no_repeat")
sys.path.insert(0, str(SOURCE))
import model
import runtime
from metrics import score, aggregate

read = lambda p: json.loads(p.read_text())
rows = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def checked_source_hashes():
    """Record the model implementation shipped in this checkout."""
    files = sorted(SOURCE.rglob("*.py")) + sorted((ROOT / "src/common").glob("*.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@torch.inference_mode()
def serial(m, row, cat, inv, table, no_repeat=True):
    h, cache = m.start(row["prompt_ids"])
    active = m.candidate_ids().tolist()
    used, tokens, sentences = [], [], []
    done = False
    while len(sentences) < 12 and len(tokens) < 96:
        logits = (h @ table.T).log_softmax(-1)[0]
        if no_repeat:
            logits[used] = -torch.inf
        local = int(logits.argmax())
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
    return dict(ids=tokens, sentences=sentences, done=done, unfinished=[])


@torch.inference_mode()
def batched(m, rr, cat, inv, table, no_repeat, batch_size):
    device = m.emb.weight.device
    groups = defaultdict(list)
    for row in rr:
        length = (len(row["prompt_ids"]) + 1 if m.prompt_mode == "token"
                  else row["prompt_ids"].count(2) + 2)
        groups[length].append(row)
    active = m.candidate_ids()
    global_ids = active.tolist()
    result = {}
    for group in groups.values():
        for off in range(0, len(group), batch_size):
            batch = group[off:off + batch_size]
            if m.prompt_mode == "token":
                p = torch.tensor([[1, *r["prompt_ids"]] for r in batch], device=device)
                x = m.emb(p)
            else:
                p = model.chunked_prompt_tensors([r["prompt_ids"] for r in batch], device)
                x, mask = m.prompt_embeddings(p)
                assert bool(mask.all())  # groups have identical prefix lengths
            h, cache = m.slow(x, cache=True)
            h = h[:, -1]
            outputs = [dict(ids=[], sentences=[], done=False, unfinished=[]) for _ in batch]
            stopped = [False] * len(batch)
            used = torch.zeros(len(batch), len(global_ids), dtype=torch.bool, device=device)
            for _ in range(12):
                logits = h @ table.T
                if no_repeat:
                    logits = logits.masked_fill(used, -torch.inf)
                local = logits.argmax(-1)
                selected = local.tolist()
                for i, z in enumerate(selected):
                    if stopped[i]:
                        continue
                    g = outputs[i]
                    unit = cat[global_ids[z]]["token_ids"]
                    if unit == [3]:
                        g["ids"].append(3)
                        g["done"] = True
                        stopped[i] = True
                    elif len(g["ids"]) + len(unit) > 96:
                        stopped[i] = True
                    else:
                        g["ids"].extend(unit)
                        g["sentences"].append(" ".join(inv[t] for t in unit if t != 2))
                        used[i, z] = True
                        if len(g["sentences"]) == 12 or len(g["ids"]) >= 96:
                            stopped[i] = True
                if all(stopped):
                    break
                feedback = m.feedback(m.canonical_tokens[active[local]])
                hh, cache = m.slow(feedback[:, None], past=cache, cache=True)
                h = hh[:, -1]
            result.update((r["world_id"], g) for r, g in zip(batch, outputs))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    d = data_dir("controlled_reasoning", "full", "6_conclusions")
    frozen_hashes = checked_source_hashes()
    jobs = []
    for layers in [2, 4]:
        for seed in range(101, 106):
            for condition in ["token", "chunk"]:
                name = f"sentence_6_seed{seed}" + ("_4l" if layers == 4 else "")
                name += "_chunked_prompt" if condition == "chunk" else ""
                p = BASE / "extended_full" / name
                jobs.append(dict(run=name, layers=layers, seed=seed, condition=condition,
                                 checkpoint_sha256=sha(p / "best.pt")))
    protocol = dict(
        conclusions=6, seeds=list(range(101, 106)), layers=[2, 4],
        conditions=["token", "chunk"], split="test", worlds=1024,
        decoding="Greedy; mask only previously emitted catalogue IDs. DONE remains eligible.",
        stopping="Original 96-token/12-sentence limits; no forced conclusion count.",
        selection="Original validation-selected best.pt; no training or checkpoint reselection.",
        primary_metric="Exact completion with original criteria; mean and sample SD over five seeds.",
        implementation="Serial cached decoding; log_softmax scores as in original generator. Full standard-output reproduction checked.",
        additional_metrics=["precision", "recall", "reserved_recall", "duplicate_fraction", "done_rate"],
        diagnostic_origin="Follow-up to the two-seed 128-world validation diagnostic; descriptive, no significance testing.",
        source_sha256=sha(Path(__file__)), frozen_source_hashes=frozen_hashes, source_kind="maintained",
        data_hashes={n: sha(d / n) for n in ["test.jsonl", "validation.jsonl", "worlds.jsonl", "catalogue.json", "candidate_ids_by_phase.json", "vocabulary.json"]},
        jobs=jobs,
    )
    pp = OUT / "protocol.json"
    if pp.exists():
        assert read(pp) == protocol, "Frozen diagnostic protocol changed"
    else:
        write(pp, protocol)
    v = read(d / "vocabulary.json")
    inv = {i: t for t, i in v.items()}
    cat = read(d / "catalogue.json")
    masks = read(d / "candidate_ids_by_phase.json")
    worlds = {r["world_id"]: r for r in rows(d / "worlds.jsonl")}
    test = rows(d / "test.jsonl")
    assert len(test) == len({r["world_id"] for r in test}) == 1024
    val = rows(d / "validation.jsonl")
    selected = {r["world_id"] for r in runtime.subset(val, 128)}
    probe = runtime.subset([r for r in val if r["world_id"] not in selected], 32)
    summary = []
    for job in jobs:
        target = OUT / (job["run"] + ".json")
        if target.exists():
            result = read(target)
            assert result["protocol_sha256"] == sha(pp)
            summary.append({k: v for k, v in result.items() if k != "items"})
            continue
        started = time.monotonic()
        p = BASE / "extended_full" / job["run"]
        saved = torch.load(p / "best.pt", map_location="cpu", weights_only=True)
        m = model.SentenceDot(cat, masks, len(v), 512, job["layers"], prompt_mode=job["condition"]).to(a.device)
        m.load_state_dict(saved["state"])
        m.eval().requires_grad_(False)
        assert saved["epoch"] == read(p / "final_metrics.json")["selected_epoch"]
        with torch.inference_mode():
            m.set_phase("validation")
            table = m.candidate_features()
            original_features = m.candidate_features
            m.candidate_features = lambda: table
            for r in probe:
                actual = serial(m, r, cat, inv, table, False)
                expected = runtime.generate(m, "sentence", r, inv, cat, 0., 0)
                assert all(actual[k] == expected[k] for k in actual)
            m.candidate_features = original_features
            m.set_phase("test")
            table = m.candidate_features()
            normal = {r["world_id"]: serial(m, r, cat, inv, table, False) for r in test}
            originals = {x["world_id"]: x["generation"] for x in read(p / "test_generations.json")["items"] if x["generation"]["temperature"] == 0}
            for wid, g in normal.items():
                assert all(g[k] == originals[wid][k] for k in g), (job, wid, "Stored greedy output differs")
            generated = {r["world_id"]: serial(m, r, cat, inv, table, True) for r in test}
        items = [dict(world_id=r["world_id"], generation=generated[r["world_id"]],
                      score=score(generated[r["world_id"]]["sentences"], worlds[r["world_id"]], generated[r["world_id"]]["done"])) for r in test]
        metrics = aggregate([x["score"] for x in items])
        assert metrics["duplicate_fraction"] == 0
        baseline = aggregate([score(g["sentences"], worlds[wid], g["done"]) for wid, g in normal.items()])
        assert baseline == read(p / "final_metrics.json")["generation"]["test"]["greedy"]
        result = dict(**job, selected_epoch=saved["epoch"], device=a.device,
                      protocol_sha256=sha(pp), seconds=time.monotonic() - started,
                      serial_validation_outputs_matched=len(probe), stored_test_outputs_matched=len(normal),
                      normal=baseline, no_repeat=metrics, items=items)
        write(target, result)
        summary.append({k: v for k, v in result.items() if k != "items"})
        write(OUT / "progress.json", dict(completed=len(summary), total=len(jobs), runs=summary))
        print(job["run"], "seconds", round(result["seconds"], 1), "exact", round(100 * baseline["exact_rate"], 2), "->", round(100 * metrics["exact_rate"], 2), flush=True)
        del m, saved, table
        gc.collect()
    write(OUT / "summary.json", dict(status="complete", protocol_sha256=sha(pp), runs=summary))
    print("COMPLETE", len(summary), "runs;", 1024 * len(summary), "test worlds", flush=True)


if __name__ == "__main__":
    main()
