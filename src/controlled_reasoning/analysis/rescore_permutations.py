"""Rescore saved generations as token multisets without inference or reselection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import hashlib
import json
from src.controlled_reasoning.metrics import score, aggregate


def signature(statement):
    return tuple(sorted(statement.split()))


def rescore(run):
    config = json.loads((run / "config.json").read_text())
    data = Path(config["data"])
    manifest = data / "manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != config["manifest_sha256"]:
        raise ValueError(f"Dataset manifest changed: {manifest}")
    worlds = {w["world_id"]: w for w in map(json.loads, (data / "worlds.jsonl").read_text().splitlines())}
    source = run / "test_generations.json"
    generations = json.loads(source.read_text())
    scores = {"greedy": [], "sampled": []}
    for item in generations["items"]:
        world, generation = worlds[item["world_id"]], item["generation"]
        if len(set(map(signature, world["conclusions"]))) != len(set(world["conclusions"])):
            raise ValueError("Required targets collide under token-multiset scoring")
        if score(generation["sentences"], world, generation["done"], bool(generation["unfinished"])) != item["score"]:
            raise ValueError(f"Saved ordinary score does not reproduce: {source}")
        adjusted = dict(world)
        for key in ("conclusions", "unseen_conclusions", "reserved_conclusions"):
            adjusted[key] = list(map(signature, world[key]))
        mode = "greedy" if generation["temperature"] == 0 else "sampled"
        scores[mode].append(score(list(map(signature, generation["sentences"])), adjusted,
                                  generation["done"], bool(generation["unfinished"])))
    if not all(scores.values()):
        raise ValueError(f"Missing greedy or sampled generations: {source}")
    return dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                config=config, metrics={mode: aggregate(items) for mode, items in scores.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = [rescore(run) for run in args.run_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = dict(definition="Exact case-sensitive token multisets; ordinary score/aggregate. Reserved recall inherits original target membership, so matching a seen permutation does not establish novel-statement generation.", runs=runs)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Rescored {len(runs)} runs into {args.output_dir}")


if __name__ == "__main__":
    main()
