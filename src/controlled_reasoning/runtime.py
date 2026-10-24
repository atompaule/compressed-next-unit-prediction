"""Shared objective, evaluation and generation for controlled reasoning."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from model import batch
from models import loss as token_loss
from metrics import score, aggregate


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + "\n")


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()]


def fingerprint(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def targets(rr, index, device):
    return torch.tensor(
        [[index[tuple(s)] for s in r["targets"]] for r in rr], device=device
    )


def objective(m, kind, rr, index):
    device = next(m.parameters()).device
    b = batch(rr, device, prompt_mode=getattr(m, "prompt_mode", "token"))
    if kind == "discrete":
        return token_loss(m(b), b["y"])
    h = m.contexts(b).flatten(0, 1)
    z = m.local_target_ids(targets(rr, index, device).flatten())
    lp = m.plan_log_probs(h)
    losses = -lp[torch.arange(len(z), device=device), z]
    return losses.mean(), losses.sum(), len(z)


@torch.no_grad()
def evaluate(m, kind, rr, index, phase):
    m.eval()
    if kind == "sentence":
        m.set_phase(phase)
    total = count = 0
    for off in range(0, len(rr), 32):
        _, n, d = objective(m, kind, rr[off : off + 32], index)
        total += float(n)
        count += int(d)
    return dict(
        nll=total / count,
        units=count,
        unit="token" if kind == "discrete" else "sentence",
    )


@torch.no_grad()
def generate(m, kind, r, inv, cat, temp, seed):
    rng = torch.Generator().manual_seed(seed)
    out = []
    sentences = []
    current = []
    done = False
    choose = lambda logits: (
        int(logits.argmax())
        if temp == 0
        else int(
            torch.multinomial(
                (logits.detach().cpu() / temp).softmax(-1), 1, generator=rng
            )
        )
    )
    if kind == "sentence":
        h, cache = m.start(r["prompt_ids"])
        active = m.candidate_ids()
        while len(sentences) < 12 and len(out) < 96:
            z = int(active[choose(m.plan_log_probs(h)[0])])
            tokens = cat[z]["token_ids"]
            if tokens == [3]:
                out.append(3)
                done = True
                break
            if len(out) + len(tokens) > 96:
                break
            out.extend(tokens)
            sentences.append(" ".join(inv[t] for t in tokens if t != 2))
            h, cache = m.accept(tokens, cache)
    else:
        state = m.start(r["prompt_ids"])
        while len(out) < 96 and len(sentences) < 12:
            t = choose(m.predict(state))
            out.append(t)
            if t == 3:
                done = True
                break
            if t == 2:
                sentences.append(" ".join(inv[x] for x in current))
                m.accept(state, t, sentence=current + [2])
                current = []
            else:
                current.append(t)
                m.accept(state, t)
    return dict(
        sentences=sentences,
        ids=out,
        done=done,
        unfinished=[inv[x] for x in current],
        temperature=temp,
        seed=seed,
        tokens=len(out),
    )


@torch.no_grad()
def generations(m, kind, rr, worlds, inv, cat, out, phase="validation"):
    m.eval()
    items = []
    original = None
    if kind == "sentence":
        m.set_phase(phase)
        original = m.candidate_features
        table = original()
        m.candidate_features = lambda: table
    try:
        for i, r in enumerate(rr):
            for j, temp in enumerate([0.0, 1.0, 1.0]):
                seed = 7000 + int(r["world_id"][:8], 16) * 3 + j
                g = generate(m, kind, r, inv, cat, temp, seed)
                w = worlds[r["world_id"]]
                items.append(
                    dict(
                        world_id=r["world_id"],
                        generation=g,
                        score=score(
                            g["sentences"], w, g["done"], bool(g["unfinished"])
                        ),
                    )
                )
            if (i + 1) % 128 == 0:
                print(phase.upper() + "_GENERATION", i + 1, len(rr), flush=True)
    finally:
        if original is not None:
            m.candidate_features = original
    result = {
        key: aggregate(
            [
                x["score"]
                for x in items
                if (x["generation"]["temperature"] == 0) == (key == "greedy")
            ]
        )
        for key in ["greedy", "sampled"]
    }
    write(out, dict(**result, items=items))
    return result


def subset(rr, n):
    # Same proportional subset for both architectures, selected without model results.
    from create_data import counts

    return [
        r
        for family, count in counts(n).items()
        for r in sorted(
            [x for x in rr if x["family"] == family], key=lambda x: x["world_id"]
        )[:count]
    ]
