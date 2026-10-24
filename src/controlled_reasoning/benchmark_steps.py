"""Sequential MPS benchmark: matched batch64 optimizer updates and greedy decoding steps.
Five checkpoints per configuration. Restore weights after timed training; no saved
checkpoint mutation. Inference excludes prefill/candidate-cache construction,
includes readout, selection and feedback, with actual decision counts including DONE.
"""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

from src.common.experiment_paths import data_dir, artifact_dir
from pathlib import Path
import json, time, statistics, subprocess
import torch
import runtime as p
from model import SentenceDot, SentenceDotNoPE, discrete

R = artifact_dir(__file__)


def sync():
    torch.mps.synchronize()


def decode(m, kind, r, cat):
    if kind != "discrete":
        h, cache = m.start(r["prompt_ids"])
        active = m.candidate_ids()
    else:
        state = m.start(r["prompt_ids"])
    sync()
    t = time.perf_counter()
    steps = tokens = sentences = 0
    current = []
    while tokens < 96 and sentences < 12:
        if kind != "discrete":
            c = int(active[int(m.plan_log_probs(h)[0].argmax())])
            steps += 1
            ts = cat[c]["token_ids"]
            if ts == [3]:
                break
            if tokens + len(ts) > 96:
                break
            tokens += len(ts)
            sentences += 1
            h, cache = m.accept(ts, cache)
        else:
            token = int(m.predict(state).argmax())
            steps += 1
            tokens += 1
            if token == 3:
                break
            if token == 2:
                sentences += 1
                m.accept(state, token, sentence=current + [2])
                current = []
            else:
                current.append(token)
                m.accept(state, token)
    sync()
    return time.perf_counter() - t, steps


def main():
    torch.set_num_threads(1)
    assert torch.backends.mps.is_available()
    results = []
    for n in (3, 6):
        d = data_dir("controlled_reasoning", "full") / f"{n}_conclusions"
        v = json.loads((d / "vocabulary.json").read_text())
        cat = json.loads((d / "catalogue.json").read_text())
        masks = json.loads((d / "candidate_ids_by_phase.json").read_text())
        index = {tuple(c["token_ids"]): c["id"] for c in cat}
        train = p.rows(d / "epoch01.jsonl")[:64]
        test = p.subset(p.rows(d / "test.jsonl"), 32)
        for layers in (2, 4):
            for seed in range(101, 106):
                kinds = ["discrete", "sentence", "nope"]
                kinds = kinds[(seed - 101) % 3 :] + kinds[: (seed - 101) % 3]
                for kind in kinds:
                    name = (
                        f"{'sentence' if kind=='nope' else kind}_{n}_seed{seed}"
                        + ("_4l" if layers == 4 else "")
                        + ("_nope" if kind == "nope" else "")
                    )
                    source = (
                        R / ("extended_full" if n == 6 else "full") / name / "best.pt"
                    )
                    state = torch.load(source, map_location="cpu", weights_only=True)
                    cls = SentenceDotNoPE if kind == "nope" else SentenceDot
                    m = (
                        discrete(len(v), 512, layers)
                        if kind == "discrete"
                        else cls(cat, masks, len(v), 512, layers)
                    ).to("mps")
                    m.load_state_dict(state["state"])
                    m.train()
                    if kind != "discrete":
                        m.set_phase("train")
                    opt = torch.optim.AdamW(
                        m.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.01
                    )

                    def update():
                        opt.zero_grad(set_to_none=True)
                        loss, _, _ = p.objective(
                            m,
                            "discrete" if kind == "discrete" else "sentence",
                            train,
                            index,
                        )
                        assert torch.isfinite(loss)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()

                    for _ in range(3):
                        update()
                    train_times = []
                    for rep in range(3):
                        sync()
                        start = time.perf_counter()
                        for _ in range(3):
                            update()
                        sync()
                        train_times.append(1000 * (time.perf_counter() - start) / 3)
                    del opt
                    m.load_state_dict(state["state"])
                    m.eval()
                    m.requires_grad_(False)
                    with torch.inference_mode():
                        if kind != "discrete":
                            m.set_phase("test")
                            original = m.candidate_features
                            table = original()
                            m.candidate_features = lambda: table
                        for r in test[:2]:
                            decode(m, kind, r, cat)
                        infer = []
                        counts = []
                        for rep in range(3):
                            records = [decode(m, kind, r, cat) for r in test]
                            count = sum(x[1] for x in records)
                            counts.append(count)
                            infer.append(1000 * sum(x[0] for x in records) / count)
                    row = dict(
                        conclusions=n,
                        layers=layers,
                        kind=kind,
                        seed=seed,
                        checkpoint_epoch=state["epoch"],
                        training_ms_per_update=statistics.median(train_times),
                        inference_ms_per_step=statistics.median(infer),
                        training_repetitions_ms=train_times,
                        inference_repetitions_ms=infer,
                        inference_steps=counts,
                    )
                    results.append(row)
                    p.write(
                        R / "step_efficiency.json",
                        dict(
                            hardware=subprocess.check_output(
                                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                            ).strip(),
                            device="mps",
                            torch_version=torch.__version__,
                            scope=__doc__,
                            results=results,
                            complete=len(results) == 60,
                        ),
                    )
                    print(json.dumps(row), flush=True)
                    del m, state
                    if kind != "discrete":
                        del table, original
                    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
