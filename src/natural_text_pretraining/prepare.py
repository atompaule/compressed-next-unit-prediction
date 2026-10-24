"""Dolma -> SaT sentences -> grammatical chunks -> canonical GPT-2 samples.

No additional linguistic filters. Full preparation requires a Slurm compute job.
The local --pilot mode is bounded to at most 200 documents per category.
"""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import socket
import time

import numpy as np
from data import atomic_json, digest
from phrase_chunker import split_sentence, VERSION

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")


def check_storage(path):
    """Stop resumably before exhausting the filesystem; never delete inputs."""
    minimum = float(os.environ.get("DOLMA_MIN_FREE_GIB", "0"))
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum:
        raise RuntimeError(
            f"Only {free:.1f} GiB free at {path}; reserve is {minimum:g} GiB. "
            "Preparation can be resumed after freeing space."
        )


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def connect(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def freeze_inputs(config, out):
    """Resolve immutable revisions once; source paths are SHA-ranked reproducibly."""
    from huggingface_hub import HfApi

    path = out / "inputs.json"
    if path.exists():
        lock = json.loads(path.read_text())
        if lock["requested_config"] != config:
            raise ValueError("Preparation configuration changed")
        return lock
    api = HfApi()
    ds = config["dataset"]
    info = api.dataset_info(ds["repository"], revision=ds["revision"])
    files = {s: [] for s in ds["sources"]}
    for f in info.siblings:
        for source, prefix in ds["sources"].items():
            if f.rfilename.startswith("data/" + prefix) and f.rfilename.endswith(
                ".jsonl.zst"
            ):
                files[source].append(f.rfilename)
    for source, paths in files.items():
        if not paths:
            raise ValueError(f"No published source files: {source}")
        paths.sort(key=lambda p: hash_text(f"{config['data']['seed']}:{p}"))
    revisions = {}
    for key in ("tokenizer", "segmenter", "segmenter_tokenizer"):
        revisions[key] = api.model_info(
            config["assets"][key], revision=config["assets"][key + "_revision"]
        ).sha
    lock = dict(
        requested_config=config,
        dataset_revision=info.sha,
        files=files,
        asset_revisions=revisions,
    )
    atomic_json(path, lock)
    return lock


def assets(config, lock, out, local_assets=None):
    from huggingface_hub import snapshot_download
    import logging

    logging.getLogger("transformers").setLevel(logging.ERROR)
    import skops.io  # noqa: F401 -- must precede transformers lazy-module discovery.
    from transformers import AutoTokenizer
    from wtpsplit import SaT
    import spacy

    paths = {}
    for key in ("tokenizer", "segmenter", "segmenter_tokenizer"):
        if local_assets:
            paths[key] = str((local_assets / key).resolve())
        else:
            paths[key] = snapshot_download(
                config["assets"][key],
                revision=lock["asset_revisions"][key],
                cache_dir=str(out / "hf_cache"),
                ignore_patterns=[
                    "*.onnx",
                    "onnx/*",
                    "*.h5",
                    "*.msgpack",
                    "pytorch_model.bin",
                ]
                if key == "segmenter"
                else ["*.bin", "*.safetensors", "*.h5", "*.msgpack", "*.onnx"],
            )
    tok = AutoTokenizer.from_pretrained(paths["tokenizer"], local_files_only=True)
    if not tok.is_fast or len(tok) != 50257:
        raise ValueError("Original fast GPT-2 tokenizer required")
    sat = SaT(paths["segmenter"], tokenizer_name_or_path=paths["segmenter_tokenizer"])
    nlp = spacy.load(
        str(local_assets / "parser") if local_assets else config["assets"]["parser"],
        disable=["ner"],
    )
    if nlp.meta["version"] != config["assets"]["parser_version"]:
        raise ValueError("Unexpected parser version")
    # Bundle the small tokenizer so the final data directory is portable.
    tok.save_pretrained(out / "tokenizer")
    return tok, sat, nlp


def sample_document(text, doc_hash, tok, sat, nlp, cfg):
    """Use the first valid samples; retain every source character inside a sample."""
    sentences = list(
        sat.split(text, strip_whitespace=False, split_on_input_newlines=False)
    )
    if "".join(sentences) != text:
        raise ValueError("SaT did not preserve the document text")
    # SaT may attach separator whitespace to the preceding sentence, whereas
    # GPT-2 attaches it to the following word. Only whitespace ownership moves;
    # no sentence is merged, removed, or linguistically resegmented.
    starts = [0]
    for sentence in sentences:
        starts.append(starts[-1] + len(sentence))
    for i in range(1, len(starts) - 1):
        while starts[i] > starts[i - 1] and text[starts[i] - 1].isspace():
            starts[i] -= 1
    sentences = [text[a:b] for a, b in zip(starts, starts[1:])]
    parsed = {}
    samples = []
    counts = Counter()
    for target in range(1, len(sentences) - cfg["target_sentences"] + 1):
        seed = int(hash_text(f"{cfg['seed']}:{doc_hash}:{target}")[:16], 16)
        prefix = random.Random(seed).randint(*cfg["prefix_sentences"])
        if target < prefix:
            continue
        lo = target - prefix
        hi = target + cfg["target_sentences"]
        begin = starts[lo]
        content = text[begin : starts[hi]]
        encoded = tok(content, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        # The final target token is predicted, not fed to the Transformer.
        if len(ids) - 1 > cfg["max_model_input_tokens"]:
            counts["over_capacity_samples"] += 1
            continue
        cuts = {0, len(content), starts[target] - begin}
        for j in range(lo, hi):
            if j not in parsed:
                sentence = sentences[j]
                nlp.max_length = max(nlp.max_length, len(sentence) + 1)
                parts = split_sentence(sentence, nlp(sentence), tok)
                parsed[j] = [p["end"] for p in parts]
            cuts.update(starts[j] - begin + k for k in parsed[j])
        token_cuts = {0: 0, len(content): len(ids)}
        # A byte-level BPE token may cross a linguistic boundary. Merge that
        # boundary instead of changing text or tokenization.
        for cut in sorted(cuts):
            if not any(a < cut < b for a, b in offsets):
                index = sum(b <= cut for a, b in offsets)
                token_cuts[cut] = index
        boundary = starts[target] - begin
        if boundary not in token_cuts or token_cuts[boundary] in (0, len(ids)):
            counts["unaligned_prompt_target_samples"] += 1
            continue
        positions = sorted(set(token_cuts.values()))
        chunks = [ids[a:b] for a, b in zip(positions, positions[1:]) if b > a]
        context = positions.index(token_cuts[boundary])
        if tok.decode(ids, clean_up_tokenization_spaces=False) != content:
            raise ValueError("GPT-2 round trip changed sample text")
        samples.append(
            dict(
                chunks=chunks,
                context=context,
                window_id=f"{doc_hash}:{target}",
                source_char_start=begin,
                source_char_end=starts[hi],
                prefix_sentences=prefix,
                target_sentences=cfg["target_sentences"],
                target_hash=hash_text(text[starts[target] : starts[hi]]),
                text_sha256=hash_text(content),
            )
        )
        if len(samples) >= cfg["max_windows_per_document"]:
            break
    counts["detected_sentences"] = len(sentences)
    return samples, counts


def records(config, lock, source, out):
    from huggingface_hub import hf_hub_download
    import zstandard

    pending = list(lock["files"][source])
    first = 0
    stride = config["data"]["records_per_shard"]
    if stride < 1:
        raise ValueError("records_per_shard must be positive")
    # Visit successive blocks across source shards. A small source can be
    # revisited without imposing an arbitrary maximum number of source records.
    while pending:
        remaining = []
        for name in pending:
            check_storage(out)
            path = hf_hub_download(
                config["dataset"]["repository"],
                name,
                repo_type="dataset",
                revision=lock["dataset_revision"],
                cache_dir=str(out / "hf_cache"),
            )
            with (
                open(path, "rb") as raw,
                zstandard.ZstdDecompressor().stream_reader(raw) as reader,
            ):
                for line_no, line in enumerate(
                    io.TextIOWrapper(reader, encoding="utf-8")
                ):
                    if line_no < first:
                        continue
                    if line_no >= first + stride:
                        remaining.append(name)
                        break
                    yield name, line_no, json.loads(line)
        pending = remaining
        first += stride


def gather(config, lock, out, tok, sat, nlp, desired, local_input=None):
    db = connect(out / "documents.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, source TEXT, original_id TEXT, file TEXT, line INTEGER, samples TEXT)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS progress (source TEXT PRIMARY KEY, value TEXT)"
    )
    db.execute("CREATE INDEX IF NOT EXISTS by_source ON documents(source)")
    for source in sorted(config["dataset"]["sources"]):
        accepted = db.execute(
            "SELECT count(*) FROM documents WHERE source=?", (source,)
        ).fetchone()[0]
        if accepted >= desired:
            continue
        row = db.execute(
            "SELECT value FROM progress WHERE source=?", (source,)
        ).fetchone()
        progress = (
            json.loads(row[0])
            if row
            else dict(records=0, duplicates=0, no_samples=0, counts={})
        )
        iterator = (
            records(config, lock, source, out)
            if local_input is None
            else (
                (str(local_input / source), i, json.loads(l))
                for i, l in enumerate((local_input / f"{source}.jsonl").open())
            )
        )
        for ordinal, (filename, line_no, row) in enumerate(iterator):
            if ordinal < progress["records"]:
                continue
            if ordinal % 128 == 0:
                check_storage(out)
            key = hash_text(row["text"])
            progress["records"] = ordinal + 1
            if db.execute("SELECT 1 FROM documents WHERE hash=?", (key,)).fetchone():
                progress["duplicates"] += 1
            else:
                samples, counts = sample_document(
                    row["text"], key, tok, sat, nlp, config["data"]
                )
                progress["counts"] = dict(Counter(progress["counts"]) + counts)
                if samples:
                    db.execute(
                        "INSERT INTO documents VALUES (?,?,?,?,?,?)",
                        (
                            key,
                            source,
                            str(row.get("id", "")),
                            filename,
                            line_no,
                            json.dumps(samples),
                        ),
                    )
                    accepted += 1
                else:
                    progress["no_samples"] += 1
            db.execute(
                "INSERT OR REPLACE INTO progress VALUES (?,?)",
                (source, json.dumps(progress)),
            )
            db.commit()
            if accepted % 25 == 0 or accepted >= desired:
                atomic_json(
                    out / "progress.json",
                    dict(
                        stage="prepare_documents",
                        source=source,
                        documents=accepted,
                        requested=desired,
                        updated_unix=time.time(),
                        **progress,
                    ),
                )
                print(
                    json.dumps(
                        dict(source=source, documents=accepted, requested=desired)
                    ),
                    flush=True,
                )
            if accepted >= desired:
                break
        if accepted < desired:
            raise ValueError(
                f"Insufficient usable documents in {source}: {accepted}/{desired}"
            )
    return db


class PackedWriter:
    def __init__(self, folder, name, dtype):
        self.folder, self.name, self.dtype = folder, name, np.dtype(dtype)
        self.path = folder / f"{name}.bin"
        self.stream = self.path.open("wb")
        self.count = 0

    def add(self, values):
        a = np.asarray(values, dtype=self.dtype)
        self.stream.write(a.tobytes())
        self.count += len(a)

    def close(self):
        self.stream.close()
        a = np.lib.format.open_memmap(
            self.folder / f"{self.name}.npy",
            mode="w+",
            dtype=self.dtype,
            shape=(self.count,),
        )
        if self.count:
            b = np.memmap(self.path, mode="r", dtype=self.dtype)
            for i in range(0, self.count, 1000000):
                a[i : i + 1000000] = b[i : i + 1000000]
            del b
        a.flush()
        del a
        self.path.unlink()


def export(config, lock, out, db, diagnostic):
    """Partition documents before samples; exact target overlaps favor train/val/test."""
    folder = out / "data"
    if (folder / "manifest.json").exists():
        return json.loads((folder / "manifest.json").read_text())
    if folder.exists():
        shutil.rmtree(folder)  # Incomplete derived output, inputs untouched.
    folder.mkdir()
    db.execute("DROP TABLE IF EXISTS windows")
    db.execute("DROP TABLE IF EXISTS used_targets")
    db.execute("CREATE TABLE windows (doc TEXT, source TEXT, split TEXT, value TEXT)")
    db.execute("CREATE INDEX windows_lookup ON windows(split,source,doc)")
    db.execute("CREATE INDEX windows_by_document ON windows(doc)")
    db.execute("CREATE TABLE used_targets (hash TEXT PRIMARY KEY, split TEXT)")
    sources = sorted(config["dataset"]["sources"])
    assignments = {}
    selected_counts = {}
    overlap = Counter()
    for source in sources:
        keys = [
            r[0]
            for r in db.execute("SELECT hash FROM documents WHERE source=?", (source,))
        ]
        keys.sort(key=lambda k: hash_text(f"partition:{config['data']['seed']}:{k}"))
        n = len(keys)
        percent = config["data"]["split_percentages"]
        if len(percent) != 3 or sum(percent) != 100 or min(percent) <= 0:
            raise ValueError("Invalid split percentages")
        val, test = [max(1, round(n * v / 100)) for v in percent[1:]]
        counts = [n - val - test, val, test]
        position = 0
        if min(counts) <= 0:
            raise ValueError("Insufficient documents for partitions")
        for split, count in zip(SPLITS, counts):
            assignments[source, split] = keys[position : position + count]
            position += count
    for split in SPLITS:
        for source in sources:
            for key in assignments[source, split]:
                check_storage(out)
                samples = json.loads(
                    db.execute(
                        "SELECT samples FROM documents WHERE hash=?", (key,)
                    ).fetchone()[0]
                )
                for sample in samples:
                    existing = db.execute(
                        "SELECT split FROM used_targets WHERE hash=?",
                        (sample["target_hash"],),
                    ).fetchone()
                    if existing and existing[0] != split:
                        overlap[f"{split}_duplicate_target_passages"] += 1
                        continue
                    db.execute(
                        "INSERT OR IGNORE INTO used_targets VALUES (?,?)",
                        (sample["target_hash"], split),
                    )
                    db.execute(
                        "INSERT INTO windows VALUES (?,?,?,?)",
                        (key, source, split, json.dumps(sample)),
                    )
            db.commit()
        # Exact same contributing document count per category, without repetition.
        groups = {
            s: [
                r[0]
                for r in db.execute(
                    "SELECT DISTINCT doc FROM windows WHERE split=? AND source=? ORDER BY doc",
                    (split, s),
                )
            ]
            for s in sources
        }
        minimum = min(map(len, groups.values()))
        if not minimum:
            raise ValueError(f"No contributing documents for a category in {split}")
        selected_counts[split] = {s: minimum for s in sources}
        for source, keys in groups.items():
            for key in keys[minimum:]:
                db.execute(
                    "DELETE FROM windows WHERE split=? AND source=? AND doc=?",
                    (split, source, key),
                )
        # Rebuild this split's target index after equalizing document counts.
        db.execute("DELETE FROM used_targets WHERE split=?", (split,))
        for (value,) in db.execute("SELECT value FROM windows WHERE split=?", (split,)):
            db.execute(
                "INSERT OR IGNORE INTO used_targets VALUES (?,?)",
                (json.loads(value)["target_hash"], split),
            )
        db.commit()
    # Disk-backed catalogue index avoids a multi-million-entry Python dictionary.
    bankdb = connect(out / "catalogue.sqlite")
    bankdb.execute("DROP TABLE IF EXISTS chunks")
    bankdb.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, tokens BLOB UNIQUE)")
    stats = {}
    train_count = 0
    for split in SPLITS:
        if split == "test":
            bankdb.execute("DELETE FROM chunks WHERE id>=?", (train_count,))
            bankdb.commit()
        next_id = bankdb.execute("SELECT count(*) FROM chunks").fetchone()[0]
        writers = {
            p: PackedWriter(folder, f"{split}_{p}", np.int32)
            for p in ("context", "target")
        }
        offsets = {
            p: PackedWriter(folder, f"{split}_{p}_offsets", np.int64) for p in writers
        }
        for w in offsets.values():
            w.add([0])
        counts = Counter()
        max_tokens = 0
        lengths = Counter()
        meta = (folder / f"{split}_meta.jsonl").open("w")
        for key, source, value in db.execute(
            "SELECT doc,source,value FROM windows WHERE split=? ORDER BY source,doc,rowid",
            (split,),
        ):
            sample = json.loads(value)
            chunkids = []
            for tokens in sample["chunks"]:
                blob = np.asarray(tokens, dtype="<i4").tobytes()
                found = bankdb.execute(
                    "SELECT id FROM chunks WHERE tokens=?", (blob,)
                ).fetchone()
                if found is None:
                    bankdb.execute("INSERT INTO chunks VALUES (?,?)", (next_id, blob))
                    found = (next_id,)
                    next_id += 1
                chunkids.append(found[0])
            n = sample["context"]
            ctx, tgt = chunkids[:n], chunkids[n:]
            for p, values in (("context", ctx), ("target", tgt)):
                writers[p].add(values)
                offsets[p].add([writers[p].count])
            nt = sum(map(len, sample["chunks"][n:]))
            nprompt = sum(map(len, sample["chunks"][:n]))
            max_tokens = max(max_tokens, nprompt + nt - 1)
            counts["samples"] += 1
            counts[source] += 1
            counts["prompt_tokens"] += nprompt
            counts["target_tokens"] += nt
            counts["target_chunks"] += len(tgt)
            counts["unseen_target_chunks"] += (
                sum(i >= train_count for i in tgt) if split != "train" else 0
            )
            lengths.update(map(len, sample["chunks"][n:]))
            metadata = {
                k: v for k, v in sample.items() if k not in ("chunks", "context")
            }
            meta.write(
                json.dumps(dict(source=source, document_id=key, **metadata)) + "\n"
            )
            if counts["samples"] % 10000 == 0:
                bankdb.commit()
                print(
                    json.dumps(
                        dict(stage="catalogue", split=split, samples=counts["samples"])
                    ),
                    flush=True,
                )
        meta.close()
        bankdb.commit()
        for w in [*writers.values(), *offsets.values()]:
            w.close()
        bt = PackedWriter(folder, f"{split}_bank_tokens", np.int32)
        bo = PackedWriter(folder, f"{split}_bank_offsets", np.int64)
        bo.add([0])
        for (blob,) in bankdb.execute("SELECT tokens FROM chunks ORDER BY id"):
            bt.add(np.frombuffer(blob, dtype="<i4"))
            bo.add([bt.count])
        bt.close()
        bo.close()
        stats[split] = dict(
            counts,
            documents=selected_counts[split],
            catalogue_chunks=next_id,
            maximum_model_input_tokens=max_tokens,
            target_chunk_length_histogram=dict(sorted(lengths.items())),
        )
        if split == "train":
            train_count = next_id
    bankdb.close()
    with (folder / "documents.jsonl").open("w") as stream:
        for key, source, identity, filename, line in db.execute(
            "SELECT hash,source,original_id,file,line FROM documents WHERE hash IN (SELECT DISTINCT doc FROM windows) ORDER BY source,hash"
        ):
            split = db.execute(
                "SELECT split FROM windows WHERE doc=? LIMIT 1", (key,)
            ).fetchone()[0]
            stream.write(
                json.dumps(
                    dict(
                        document_id=key,
                        source=source,
                        original_id=identity,
                        file=filename,
                        line=line,
                        split=split,
                    )
                )
                + "\n"
            )
    shutil.copyfile(out / "inputs.json", folder / "inputs.json")
    for name in ("environment.json", "preparation_signature.json", "calibration.json"):
        if (out / name).exists():
            shutil.copyfile(out / name, folder / name)
    shutil.copytree(out / "tokenizer", folder / "tokenizer")
    files = {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()}
    audit = dict(
        status="passed",
        checks=[
            "canonical_gpt2_text_roundtrip",
            "document_disjoint",
            "no_cross_split_exact_target_passages",
            "model_capacity",
            "equal_contributing_document_counts",
        ],
        splits=stats,
        exclusions=dict(overlap),
        documents={
            s: json.loads(v) for s, v in db.execute("SELECT source,value FROM progress")
        },
    )
    atomic_json(folder / "audit.json", audit)
    manifest = dict(
        status="complete",
        schema_version=2,
        chunk_boundary="metadata_only",
        chunk_end_id=None,
        diagnostic=diagnostic,
        vocab_size=50257,
        chunking=dict(policy="phrases", version=VERSION),
        token_boundary_policy="canonical_gpt2_whole_window_v1",
        whole_window_tokenization_verified=True,
        tokenizer=dict(
            name="gpt2",
            path="tokenizer",
            files={
                p.name: digest(p)
                for p in (folder / "tokenizer").iterdir()
                if p.is_file()
            },
        ),
        maximum_model_input_tokens=max(
            s["maximum_model_input_tokens"] for s in stats.values()
        ),
        splits=stats,
        files=files,
        audit_file="audit.json",
        audit_sha256=digest(folder / "audit.json"),
        inputs_sha256=digest(out / "inputs.json"),
        dataset_revision=lock["dataset_revision"],
        config=config,
        code_sha256={p.name: digest(p) for p in HERE.glob("*.py")},
    )
    atomic_json(folder / "manifest.json", manifest)
    return manifest


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=HERE / "experiment.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--pilot",
        type=int,
        default=0,
        help="Local bounded pilot: 3..200 documents/category",
    )
    p.add_argument("--local-input", type=Path)
    p.add_argument("--local-assets", type=Path)
    p.add_argument("--calibrate-only", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    a = p.parse_args()
    if a.pilot and not 3 <= a.pilot <= 200:
        raise ValueError("Pilot must have 3..200 documents/category")
    if not a.pilot and (
        not os.environ.get("SLURM_JOB_ID")
        or socket.gethostname().split(".")[0].startswith("lx")
    ):
        raise RuntimeError(
            "Full preparation requires a manually submitted Slurm compute job"
        )
    if a.local_input and not a.pilot:
        raise ValueError("Local fixtures are diagnostic only")
    a.out.mkdir(parents=True, exist_ok=True)
    check_storage(a.out)
    import fcntl

    with (a.out / "prepare.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = json.loads(a.config.read_text())
        torch.set_num_threads(4)
        if a.local_input:
            lock = dict(
                requested_config=cfg,
                dataset_revision="diagnostic_fixture",
                files={},
                asset_revisions={},
            )
            atomic_json(a.out / "inputs.json", lock)
        else:
            lock = freeze_inputs(cfg, a.out)
        from importlib.metadata import version

        atomic_json(
            a.out / "environment.json",
            {
                name: version(name)
                for name in (
                    "torch",
                    "transformers",
                    "wtpsplit",
                    "spacy",
                    "numpy",
                    "huggingface_hub",
                )
            },
        )
        signature = {
            "diagnostic": bool(a.pilot),
            "environment": json.loads((a.out / "environment.json").read_text()),
            "asset_revisions": lock["asset_revisions"],
            "config": cfg,
            "code": {
                name: digest(HERE / name)
                for name in ("prepare.py", "phrase_chunker.py")
            },
            "local_input": {
                p.name: digest(p) for p in sorted(a.local_input.glob("*.jsonl"))
            }
            if a.local_input
            else None,
        }
        signature_path = a.out / "preparation_signature.json"
        if (
            signature_path.exists()
            and json.loads(signature_path.read_text()) != signature
        ):
            raise ValueError(
                "Preparation inputs or code changed; use a new output directory"
            )
        atomic_json(signature_path, signature)
        tok, sat, nlp = assets(cfg, lock, a.out, a.local_assets)
        if a.device != "cpu":
            sat.to(a.device)
        desired = a.pilot or cfg["data"]["calibration_documents_per_category"]
        db = gather(cfg, lock, a.out, tok, sat, nlp, desired, a.local_input)
        calibration = a.out / "calibration.json"
        if not calibration.exists():
            means = {
                s: sum(
                    len(json.loads(v))
                    for (v,) in db.execute(
                        "SELECT samples FROM documents WHERE source=? ORDER BY rowid LIMIT ?",
                        (s, desired),
                    )
                )
                / desired
                for s in cfg["dataset"]["sources"]
            }
            recommended = (
                math.ceil(
                    cfg["data"]["target_training_samples_approx"]
                    / (sum(means.values()) * cfg["data"]["split_percentages"][0] / 100)
                    / 100
                )
                * 100
            )
            token_means = {}
            for source in cfg["dataset"]["sources"]:
                rows = [
                    json.loads(v)
                    for (v,) in db.execute(
                        "SELECT samples FROM documents WHERE source=? ORDER BY rowid LIMIT ?",
                        (source, desired),
                    )
                ]
                token_means[source] = (
                    sum(
                        sum(len(c) for c in sample["chunks"][sample["context"] :])
                        for samples in rows
                        for sample in samples
                    )
                    / desired
                )
            atomic_json(
                calibration,
                dict(
                    documents_per_category=desired,
                    mean_samples_per_document=means,
                    mean_target_tokens_per_document=token_means,
                    recommended_documents_per_category=recommended,
                    target_samples=cfg["data"]["target_training_samples_approx"],
                    estimated_training_target_tokens=recommended
                    * sum(token_means.values())
                    * cfg["data"]["split_percentages"][0]
                    / 100,
                    note="Estimate from initial source documents; actual totals may differ",
                ),
            )
        recommended = json.loads(calibration.read_text())[
            "recommended_documents_per_category"
        ]
        if a.calibrate_only:
            return
        if not a.pilot:
            db.close()
            db = gather(
                cfg,
                lock,
                a.out,
                tok,
                sat,
                nlp,
                cfg["data"].get("documents_per_category") or recommended,
            )
        manifest = export(cfg, lock, a.out, db, bool(a.pilot))
        db.close()
        from audit import verify

        verify(a.out / "data")
        atomic_json(
            a.out / "progress.json", dict(stage="complete", splits=manifest["splits"])
        )
        print(json.dumps(dict(stage="complete", data=str(a.out / "data"))), flush=True)


if __name__ == "__main__":
    main()
