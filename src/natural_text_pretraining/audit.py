"""Technical data checks and a deterministic qualitative sample (no text filters)."""

from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from data import PackedBank, Windows, atomic_json, digest, load_manifest


def verify(folder, review_count=50):
    from transformers import AutoTokenizer

    m = load_manifest(folder)
    for name, expected in m["files"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Changed data file: {name}")
    for name, expected in m["tokenizer"]["files"].items():
        if digest(folder / "tokenizer" / name) != expected:
            raise ValueError(f"Changed tokenizer: {name}")
    if digest(folder / m["audit_file"]) != m["audit_sha256"]:
        raise ValueError("Changed construction audit")
    tok = AutoTokenizer.from_pretrained(folder / "tokenizer", local_files_only=True)
    train = PackedBank(folder, "train_bank")
    prior_docs = set()
    prior_targets = set()
    report = {}
    reviews = []
    for split in ("train", "validation", "test"):
        bank = PackedBank(folder, split + "_bank")
        windows = Windows(folder, split)
        if (np.diff(bank.offsets) <= 0).any():
            raise ValueError("Empty chunk")
        if (bank.tokens < 0).any() or (bank.tokens >= m["vocab_size"]).any():
            raise ValueError("Invalid token")
        if not np.array_equal(
            bank.offsets[: len(train) + 1], train.offsets
        ) or not np.array_equal(bank.tokens[: len(train.tokens)], train.tokens):
            raise ValueError("Bank prefix mismatch")
        for values, offsets in (
            (windows.context, windows.context_offsets),
            (windows.target, windows.target_offsets),
        ):
            if offsets[0] != 0 or (np.diff(offsets) <= 0).any():
                raise ValueError("Empty sample part")
            if (values < 0).any() or (values >= len(bank)).any():
                raise ValueError("Invalid catalogue reference")
        lengths = np.diff(bank.offsets)
        max_input = 0
        prompt_tokens = target_tokens = 0
        for lo in range(0, len(windows), 8192):
            hi = min(lo + 8192, len(windows))
            totals = []
            for values, offsets in (
                (windows.context, windows.context_offsets),
                (windows.target, windows.target_offsets),
            ):
                a, b = int(offsets[lo]), int(offsets[hi])
                sums = np.r_[0, np.cumsum(lengths[values[a:b]])]
                totals.append(np.diff(sums[offsets[lo : hi + 1] - a]))
            max_input = max(max_input, int((totals[0] + totals[1] - 1).max()))
            prompt_tokens += int(totals[0].sum())
            target_tokens += int(totals[1].sum())
        if max_input > m["config"]["data"]["max_model_input_tokens"]:
            raise ValueError("Capacity exceeded")
        docs = set()
        targets = set()
        source_docs = {}
        counts = Counter()
        selected = []
        window_ids = set()
        for i, line in enumerate(windows.meta.open()):
            row = json.loads(line)
            doc = row["document_id"]
            target = row["target_hash"]
            if row["window_id"] in window_ids:
                raise ValueError("Duplicate sample identity")
            window_ids.add(row["window_id"])
            if doc in prior_docs or target in prior_targets:
                raise ValueError("Cross-partition overlap")
            docs.add(doc)
            targets.add(target)
            counts[row["source"]] += 1
            source_docs.setdefault(row["source"], set()).add(doc)
            rank = hashlib.sha256(("review:" + row["window_id"]).encode()).hexdigest()
            selected.append((rank, i, row))
            if len(selected) > review_count * 2:
                selected = sorted(selected)[:review_count]
        if len(window_ids) != len(windows):
            raise ValueError("Metadata count mismatch")
        if len(set(map(len, source_docs.values()))) != 1:
            raise ValueError("Category document imbalance")
        for _, i, row in sorted(selected)[:review_count]:
            context, target = windows.row(i)
            chunks = [bank.sequence(c) for c in context + target]
            ids = sum(chunks, [])
            text = tok.decode(ids, clean_up_tokenization_spaces=False)
            if (
                hashlib.sha256(text.encode()).hexdigest() != row["text_sha256"]
                or tok.encode(text, add_special_tokens=False) != ids
            ):
                raise ValueError("Canonical text/token mismatch")
            target_text = tok.decode(
                sum(chunks[len(context) :], []), clean_up_tokenization_spaces=False
            )
            if hashlib.sha256(target_text.encode()).hexdigest() != row["target_hash"]:
                raise ValueError("Target boundary changed text")
            reviews.append(
                dict(
                    split=split,
                    source=row["source"],
                    window_id=row["window_id"],
                    context=[tok.decode(c) for c in chunks[: len(context)]],
                    target=[tok.decode(c) for c in chunks[len(context) :]],
                )
            )
        report[split] = dict(
            samples=len(windows),
            documents=len(docs),
            documents_by_source={s: len(d) for s, d in source_docs.items()},
            samples_by_source=dict(counts),
            prompt_tokens=prompt_tokens,
            target_tokens=target_tokens,
            target_chunks=len(windows.target),
            catalogue_chunks=len(bank),
            unseen_target_chunks=int((windows.target >= len(train)).sum())
            if split != "train"
            else 0,
            maximum_model_input_tokens=max_input,
        )
        for field in (
            "samples",
            "prompt_tokens",
            "target_tokens",
            "target_chunks",
            "catalogue_chunks",
            "maximum_model_input_tokens",
        ):
            if report[split][field] != m["splits"][split][field]:
                raise ValueError(f"Count mismatch: {split}/{field}")
        prior_docs.update(docs)
        prior_targets.update(targets)
    result = dict(
        status="passed",
        data_manifest_sha256=digest(folder / "manifest.json"),
        splits=report,
        canonical_samples_checked=len(reviews),
        diagnostic=m["diagnostic"],
    )
    atomic_json(folder / "verification.json", result)
    atomic_json(folder / "review.json", reviews)
    body = [
        '<!doctype html><meta charset="utf-8"><title>Corpus sample</title><style>body{max-width:1100px;margin:40px auto;font:18px/1.6 system-ui}article{border-top:1px solid #ccc;padding:24px 0}.sep{color:#b33545;font-weight:bold}</style><h1>Deterministic corpus sample</h1><p>Bars mark chunk boundaries. Text is unchanged.</p>'
    ]
    for r in reviews:
        render = lambda parts: '<span class="sep"> | </span>'.join(
            html.escape(x) for x in parts
        )
        body.append(
            f"<article><h2>{html.escape(r['split'] + ' · ' + r['source'])}</h2><small>{r['window_id']}</small><h3>Context</h3><p>{render(r['context'])}</p><h3>Reference</h3><p>{render(r['target'])}</p></article>"
        )
    (folder / "review.html").write_text("\n".join(body))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    print(json.dumps(verify(p.parse_args().data), indent=2))
