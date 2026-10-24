"""Independently rescore the completed sampled diagnostic and check provenance."""
from pathlib import Path
import hashlib
import argparse
import json

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT))
from src.common.experiment_paths import data_dir, checkpoint_dir, result_dir
BASE = checkpoint_dir("controlled_reasoning")
OUT = result_dir("controlled_reasoning", "compressed_premises", "no_repeat_sampled")
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main(verification_output=None):
    protocol = read(OUT / "protocol.json")
    summary = read(OUT / "summary.json")
    digest = sha(OUT / "protocol.json")
    assert summary["status"] == "complete" and len(summary["runs"]) == 20
    assert summary["protocol_sha256"] == digest
    sources = {"evaluate_no_repeat_sampled.py": protocol["source_sha256"],
               "evaluate_no_repeat.py": protocol["helper_sha256"]}
    for name, expected in sources.items():
        candidates = (Path(__file__).with_name(name), BASE / "evaluation_sources/no_repeat" / name)
        assert any(p.is_file() and sha(p) == expected for p in candidates), name
    data = data_dir("controlled_reasoning", "full", "6_conclusions")
    for name, expected in protocol["data_hashes"].items():
        assert sha(data / name) == expected
    campaign = ROOT if protocol.get("source_kind") == "maintained" else BASE / "campaigns/chunked_prompt_20261021/code"
    for name, expected in protocol["frozen_source_hashes"].items():
        assert sha(campaign / name) == expected
    worlds = {r["world_id"]: r for r in map(json.loads, (data / "worlds.jsonl").read_text().splitlines())}
    test_worlds = {json.loads(line)["world_id"] for line in (data / "test.jsonl").read_text().splitlines()}
    inv = {i: t for t, i in read(data / "vocabulary.json").items()}
    checked, references, matched, hashes = 0, 0, 0, {}
    for report in summary["runs"]:
        name = report["run"]
        path = OUT / (name + ".json")
        result = read(path)
        assert {k: v for k, v in result.items() if k != "items"} == report
        assert result["protocol_sha256"] == digest
        run = BASE / "extended_full" / name
        assert sha(run / "best.pt") == result["checkpoint_sha256"]
        for filename, expected in protocol["baseline_hashes"][name].items():
            assert sha(run / filename) == expected
        original = read(run / "final_metrics.json")
        assert result["normal"] == original["generation"]["test"]["sampled"]
        assert result["selected_epoch"] == original["selected_epoch"]
        assert result["stored_test_outputs_matched"] == 2048
        assert result["reference_outputs_matched"] == 128
        samples = {(r["world_id"], r["sample"]) for r in result["items"]}
        assert samples == {(wid, sample) for wid in test_worlds for sample in (1, 2)}
        assert len(result["items"]) == 2048
        counts, exact, done = {}, 0, 0
        for item in result["items"]:
            output, wid = item["generation"], item["world_id"]
            assert output["seed"] == 7000 + int(wid[:8], 16) * 3 + item["sample"]
            assert output["temperature"] == 1.0 and not output["unfinished"]
            assert output["tokens"] == len(output["ids"]) <= 96
            assert bool(output["done"]) == bool(output["ids"] and output["ids"][-1] == 3)
            tokens = output["ids"][:-1] if output["done"] else output["ids"]
            reconstructed, current = [], []
            for token in tokens:
                if token == 2:
                    reconstructed.append(" ".join(inv[t] for t in current))
                    current = []
                else:
                    assert token not in (0, 1, 3, 4)
                    current.append(token)
            assert not current and reconstructed == output["sentences"]
            sequence, world = output["sentences"], worlds[wid]
            unique, gold = set(sequence), set(world["conclusions"])
            unseen, reserved = set(world["unseen_conclusions"]), set(world["reserved_conclusions"])
            assert len(sequence) == len(unique) <= 12
            values = dict(generated=len(sequence), valid_occurrences=sum(s in gold for s in sequence),
                          correct_unique=len(unique & gold), required=len(gold),
                          duplicate_count=len(sequence) - len(unique), unseen_correct=len(unique & unseen),
                          unseen_required=len(unseen), reserved_correct=len(unique & reserved),
                          reserved_required=len(reserved))
            is_exact = bool(output["done"] and len(sequence) == len(gold) and unique == gold)
            assert item["score"]["exact"] == is_exact
            for key, value in values.items():
                assert item["score"][key] == value
                counts[key] = counts.get(key, 0) + value
            exact += is_exact
            done += output["done"]
            checked += 1
        ratio = lambda a, b: counts[a] / counts[b] if counts[b] else None
        metrics = dict(samples=2048, exact_rate=exact / 2048,
                       precision=ratio("valid_occurrences", "generated"), recall=ratio("correct_unique", "required"),
                       unseen_recall=ratio("unseen_correct", "unseen_required"),
                       reserved_recall=ratio("reserved_correct", "reserved_required"),
                       duplicate_fraction=ratio("duplicate_count", "generated"), done_rate=done / 2048, counts=counts)
        assert metrics == result["no_repeat"]
        references += result["reference_outputs_matched"]
        matched += result["stored_test_outputs_matched"]
        hashes[name] = sha(path)
    verification = dict(passed=True, runs=20, protocol_sha256=digest,
                        no_repeat_outputs_independently_scored=checked,
                        standard_outputs_reproduced=matched,
                        reference_validation_checks=references,
                        unchanged_checkpoints=20, result_hashes=hashes,
                        verification_source_sha256=sha(Path(__file__)))
    (verification_output or OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print("Verified", checked, "no-repeat outputs,", matched, "reproduced standard outputs,",
          references, "reference checks, and all 20 unchanged checkpoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-output", type=Path, help="Write verification to a separate file")
    main(parser.parse_args().verification_output)
