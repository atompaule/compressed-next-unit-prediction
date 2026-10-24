"""Fresh matched seed101 training for120rounds; evaluate selected checkpoints."""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import runtime as p
import torch
from src.common.experiment_paths import data_dir, artifact_dir
from model import SentenceDot, SentenceDotFixedPE, SentenceDotNoPE, discrete
from conditions import (add_prompt_argument, add_rotation_argument,
                        check_resume_prompt, check_resume_rotation, rotation_mode, run_name)

R = artifact_dir(__file__)
# Use the shared phase-aware generation path for validation and final evaluation.
generations = p.generations


def scheduler_for(opt, total, rate):
    for g in opt.param_groups:
        g["lr"] = rate
        g["initial_lr"] = rate

    def factor(step):
        if step < 16:
            return (step + 1) / 16
        t = min(1.0, (step - 16) / (total - 16))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def quality(g, nll):
    def f1(x):
        a, b = x["precision"], x["recall"]
        return 2 * a * b / (a + b) if a + b else 0.0

    return (
        (g["greedy"]["exact_rate"] + g["sampled"]["exact_rate"]) / 2,
        (f1(g["greedy"]) + f1(g["sampled"])) / 2,
        -nll,
    )


def main(a):
    torch.set_num_threads(1)
    assert torch.backends.mps.is_available()
    torch.manual_seed(a.seed)
    key = run_name(
        a.kind, a.conclusions, a.seed, a.layers, a.no_token_pe, a.prompt_mode,
        a.fixed_rotation_rates,
    )
    rates = rotation_mode(a.kind, a.no_token_pe, a.fixed_rotation_rates)
    d = data_dir("controlled_reasoning", "full") / f"{a.conclusions}_conclusions"
    out = R / "full" / key
    out.mkdir(parents=True, exist_ok=False)
    read = lambda f: json.loads((d / f).read_text())
    manifest = read("manifest.json")
    v = read("vocabulary.json")
    inv = {i: t for t, i in v.items()}
    cat = read("catalogue.json")
    masks = read("candidate_ids_by_phase.json")
    index = {tuple(c["token_ids"]): c["id"] for c in cat}
    val = p.rows(d / "validation.jsonl")
    worlds = {w["world_id"]: w for w in p.rows(d / "worlds.jsonl")}
    selection = p.subset(val, 128)
    history = []
    rng = torch.Generator().manual_seed(a.seed)
    sentence_model = (SentenceDotNoPE if a.no_token_pe else
                      SentenceDotFixedPE if a.fixed_rotation_rates else SentenceDot)
    m = (
        sentence_model(cat, masks, len(v), 512, a.layers, prompt_mode=a.prompt_mode)
        if a.kind == "sentence"
        else discrete(len(v), 512, a.layers)
    ).to("mps")
    opt = torch.optim.AdamW(
        m.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0.01
    )
    sched = scheduler_for(opt, 320, 0.001)
    initial = p.evaluate(m, a.kind, val, index, "validation")
    p.write(out / "initial_metrics.json", initial)
    meta = dict(
        sentence_composition=(
            "unrotated_sum"
            if a.no_token_pe
            else "rotary_sum" if a.kind == "sentence" else None
        ),
        prompt_mode=a.prompt_mode,
        rotation_mode=rates,
        embedding_norm_gradient="differentiable" if a.kind == "sentence" else None,
        backbone_sequence_positions=True,
        kind=a.kind,
        seed=a.seed,
        conclusions=a.conclusions,
        device="mps",
        layers=a.layers,
        width=512,
        heads=4,
        batch_size=64,
        epochs=120,
        parameters=sum(x.numel() for x in m.parameters()),
        initialization=f"Fresh seed{a.seed} model, optimizer and scheduler",
        data=str(d),
        manifest_sha256=p.fingerprint(d / "manifest.json"),
        schedule="Rounds1-20 .001->.0001;21-60 .0003->.00003;61-120 .0001->.00001;16-update warmup per phase. Fresh initialization; random training worlds with replacement from round1.",
        selection="Same fixed128validationworlds, rounds20,40,60,80,100,120; mean greedy/sample exact then mean F1 then lower NLL. All512validationworlds for NLL each round. Final tests only after selection.",
        sampling="Greedy plus two samples at temperature1 per world; world-based seeds and caps96tokens/12sentences",
        novelty="Reserved validation/test sentences excluded from training inputs, targets and candidate scores. Test-reserved candidates also excluded from validation.",
        source_hashes={
            f: p.fingerprint(Path(__file__).resolve().parent / f)
            for f in [
                "full.py",
                "runtime.py",
                "model.py",
                "conditions.py",
                "metrics.py",
                "source/rotation.py",
                "source/models.py",
            ]
        },
    )
    p.write(out / "config.json", meta)
    start = time.perf_counter()
    base = 0.0
    best = None
    selected = None

    def checkpoint(epoch):
        nonlocal best, selected
        p.write(out / "status.json", dict(status="validation_selection", epoch=epoch))
        g = generations(
            m,
            a.kind,
            selection,
            worlds,
            inv,
            cat,
            out / f"validation_epoch{epoch}.json",
        )
        nll = p.evaluate(m, a.kind, val, index, "validation")["nll"]
        q = quality(g, nll)
        if best is None or q > best:
            best = q
            selected = epoch
            torch.save(
                dict(state=m.state_dict(), epoch=epoch, prompt_mode=a.prompt_mode,
                     rotation_mode=rates),
                out / "best.pt",
            )
        print("SELECTION", epoch, q, "best", selected, flush=True)
        return dict(generation=g, selection_score=q)

    (out / "history.jsonl").write_text("".join(json.dumps(x) + "\n" for x in history))
    for epoch in range(1, 121):
        if epoch == 21:
            sched = scheduler_for(opt, 640, 0.0003)
        if epoch == 61:
            sched = scheduler_for(opt, 960, 0.0001)
        p.write(
            out / "status.json",
            dict(status="training", epoch=epoch, selected_epoch=selected),
        )
        path = d / f"epoch{epoch:02d}.jsonl"
        assert p.fingerprint(path) == manifest["epochs"][epoch - 1]["sha256"]
        tr = p.rows(path)
        order = torch.randperm(len(tr), generator=rng).tolist()
        m.train()
        if a.kind == "sentence":
            m.set_phase("train")
        total = count = 0
        for off in range(0, len(tr), 64):
            rr = [tr[i] for i in order[off : off + 64]]
            opt.zero_grad(set_to_none=True)
            loss, n, z = p.objective(m, a.kind, rr, index)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            sched.step()
            total += float(n.detach())
            count += int(z)
        vm = p.evaluate(m, a.kind, val, index, "validation")
        entry = dict(
            epoch=epoch,
            train_nll=total / count,
            validation=vm,
            lr=opt.param_groups[0]["lr"],
            seconds=base + time.perf_counter() - start,
            data_sha256=p.fingerprint(path),
            batch_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        )
        if epoch % 20 == 0:
            entry.update(checkpoint(epoch))
            torch.save(
                dict(
                    state=m.state_dict(),
                    epoch=epoch,
                    prompt_mode=a.prompt_mode,
                    rotation_mode=rates,
                    optimizer=opt.state_dict(),
                    scheduler=sched.state_dict(),
                    data_rng=rng.get_state(),
                ),
                out / "last.pt",
            )
        history.append(entry)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
    elapsed = time.perf_counter() - start
    state = torch.load(out / "best.pt", map_location="mps", weights_only=True)
    check_resume_prompt(state, a.prompt_mode)
    check_resume_rotation(state, rates)
    m.load_state_dict(state["state"])
    m.eval()
    m.requires_grad_(False)
    result = dict(
        sentence_composition=meta["sentence_composition"],
        prompt_mode=meta["prompt_mode"],
        rotation_mode=meta["rotation_mode"],
        embedding_norm_gradient=meta["embedding_norm_gradient"],
        parameters=meta["parameters"],
        epochs=120,
        selected_epoch=state["epoch"],
        device="mps",
        fresh_initialization=True,
        initial_validation=initial,
        continuation_seconds=elapsed,
        last_training_nll=history[-1]["train_nll"],
        last_validation=history[-1]["validation"],
        generation={},
        likelihood={},
        test_evaluated=True,
    )
    for split in ["validation", "test"]:
        rr = val if split == "validation" else p.rows(d / "test.jsonl")
        p.write(
            out / "status.json",
            dict(status=f"final_{split}", epoch=120, selected_epoch=selected),
        )
        result["likelihood"][split] = p.evaluate(m, a.kind, rr, index, split)
        result["generation"][split] = generations(
            m,
            a.kind,
            rr,
            worlds,
            inv,
            cat,
            out / f"{split}_generations.json",
            phase=split,
        )
    p.write(out / "final_metrics.json", result)
    p.write(
        out / "status.json", dict(status="complete", epoch=120, selected_epoch=selected)
    )
    print("COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--kind", choices=["sentence", "discrete"], required=True)
    a.add_argument("--conclusions", type=int, choices=[3, 6], required=True)
    a.add_argument("--seed", type=int, default=101)
    a.add_argument("--layers", type=int, choices=[2, 4], default=2)
    a.add_argument(
        "--no-token-pe",
        action="store_true",
        help="Disable within-sentence rotations; retain backbone sequence positions",
    )
    add_prompt_argument(a)
    add_rotation_argument(a)
    main(a.parse_args())
