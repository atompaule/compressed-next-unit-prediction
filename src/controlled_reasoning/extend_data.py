"""Generate only epochs 121-240; reuse the exact existing evaluation split."""

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
import json, hashlib
import create_data as c

R = artifact_dir(__file__)
base = data_dir("controlled_reasoning", "full", "6_conclusions")
out = data_dir("controlled_reasoning", "extended", "6_conclusions")


def fingerprint(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == "__main__":
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((base / "manifest.json").read_text())
    v = json.loads((base / "vocabulary.json").read_text())
    excluded = {
        json.loads(s)["world_id"]
        for split in ("validation", "test")
        for s in (base / f"{split}.jsonl").read_text().splitlines()
    }
    for name in (
        "vocabulary.json",
        "catalogue.json",
        "candidate_ids_by_phase.json",
        "validation.jsonl",
        "test.jsonl",
        "worlds.jsonl",
    ):
        (out / name).symlink_to((base / name).resolve())
    # Verify the regeneration procedure against an existing epoch before extending.
    rr, _, _ = c.make_split(
        6,
        "train",
        1024,
        manifest["data_seed"] + 600000 + 1000 + 120,
        set(),
        v,
        allow_repeats=True,
        excluded_worlds=excluded,
    )
    original = [
        json.loads(x) for x in (base / "epoch120.jsonl").read_text().splitlines()
    ]
    assert rr == original, "Generator does not reproduce original epoch 120"
    epochs = []
    for epoch in range(121, 241):
        rows, worlds, _ = c.make_split(
            6,
            "train",
            1024,
            manifest["data_seed"] + 600000 + 1000 + epoch,
            set(),
            v,
            allow_repeats=True,
            excluded_worlds=excluded,
        )
        assert all(w["world_id"] not in excluded for w in worlds)
        assert all(
            c.partition(f) == "train"
            for w in worlds
            for f in w["premises"] + w["conclusion_facts"]
        )
        path = out / f"epoch{epoch:02d}.jsonl"
        c.jsonl(path, rows)
        epochs.append(dict(epoch=epoch, worlds=len(rows), sha256=fingerprint(path)))
        print("Generated", epoch, flush=True)
    c.write(
        out / "manifest.json",
        dict(
            source_manifest_sha256=fingerprint(base / "manifest.json"),
            data_seed=manifest["data_seed"],
            epochs=epochs,
            epoch120_reproduced=True,
            held_out_worlds_excluded=len(excluded),
            held_out_facts_excluded=True,
        ),
    )
