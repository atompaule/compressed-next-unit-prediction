"""Create a portable verified backup of weights, reports, source and optional data."""

import argparse
import json
from pathlib import Path
import shutil
from data import digest, atomic_json


def export(campaign, data, out, include_data=False, config_path=None):
    if out.exists():
        raise FileExistsError("Use a new backup destination")
    config_path = config_path or Path(__file__).parent / "experiment.json"
    config = json.loads(config_path.read_text())
    required = [
        campaign / c / str(s) / "final.pt"
        for c in config["conditions"]
        for s in config["seeds"]
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not (campaign / "evaluation" / "summary.json").exists():
        raise ValueError("Final evaluation missing")
    for path in required:
        run = json.loads((path.parent / "config.json").read_text())
        for name, value in run["source_sha256"].items():
            if digest(Path(__file__).parent / name) != value:
                raise ValueError("Source differs from trained checkpoint")
    out.mkdir(parents=True)
    shutil.copytree(
        Path(__file__).parent,
        out / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(config_path, out / "source" / "experiment.json")
    for path in required:
        dest = out / path.relative_to(campaign).parent
        dest.mkdir(parents=True)
        for p in path.parent.iterdir():
            if p.name == "final.pt" or p.suffix in (".json", ".jsonl"):
                shutil.copy2(p, dest / p.name)
    shutil.copytree(
        campaign / "evaluation",
        out / "evaluation",
        ignore=shutil.ignore_patterns("catalogue_vectors.npy"),
    )
    if include_data:
        shutil.copytree(data, out / "data")
    else:
        (out / "data").mkdir()
        for name in (
            "manifest.json",
            "audit.json",
            "verification.json",
            "inputs.json",
            "documents.jsonl",
            "review.json",
            "review.html",
        ):
            shutil.copy2(data / name, out / "data" / name)
        shutil.copytree(data / "tokenizer", out / "data" / "tokenizer")
    hashes = {
        str(p.relative_to(out)): digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }
    atomic_json(
        out / "backup.json",
        dict(status="complete", includes_data_arrays=include_data, files=hashes),
    )
    for name, h in hashes.items():
        if digest(out / name) != h:
            raise ValueError(f"Backup readback mismatch: {name}")
    return len(hashes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("campaign", "data", "out"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--include-data", action="store_true")
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            dict(files=export(a.campaign, a.data, a.out, a.include_data, a.config))
        )
    )
