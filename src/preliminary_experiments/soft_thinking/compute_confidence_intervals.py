"""Compute confidence intervals directly from new benchmark runner JSON outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

METHODS = {"Sampled": "default_baseline", "Soft": "soft_thinking",
           "Greedy": "discrete_top1", "Noisy": "noisy_discrete"}


def wilson(k, n):
    if type(n) is not int or type(k) is not int or n < 1 or not 0 <= k <= n:
        raise ValueError("Need integer counts with 0 <= correct <= n and n > 0")
    p, z = k / n, 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [100 * (centre - half), 100 * (centre + half)]


def paired_interval(soft_only, noisy_only, n):
    if (type(n) is not int or n < 2 or
            any(type(k) is not int or k < 0 for k in (soft_only, noisy_only)) or
            soft_only + noisy_only > n):
        raise ValueError("Invalid paired counts")
    delta = (soft_only - noisy_only) / n
    # D_i is -1, 0, or 1; these two counts determine its sample variance.
    se = math.sqrt(max(0.0, soft_only + noisy_only - n * delta * delta) / (n * (n - 1)))
    return {"difference_pp": 100 * delta,
            "ci95_pp": [100 * (delta - 1.96 * se), 100 * (delta + 1.96 * se)],
            "soft_only_correct": soft_only, "noisy_only_correct": noisy_only}


def validate_raw(path, row):
    if row.get("raw_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != row["raw_sha256"]:
        raise ValueError(f"Raw source changed: {path}")
    data = json.loads(path.read_text())
    runs = {}
    for run in data["runs"]:
        name = run["config"]["name"]
        if name in runs:
            raise ValueError(f"Duplicate condition: {name}")
        records = run["results"]
        indexed = {r["index"]: r for r in records}
        if len(indexed) != len(records) or len(records) != row["n"] or run["total"] != row["n"]:
            raise ValueError("Question count or unique indices do not match")
        if any(type(r["correct"]) is not bool for r in records):
            raise ValueError("Correctness must be boolean")
        count = sum(r["correct"] for r in records)
        if count != run["correct"] or not math.isclose(run["accuracy"], count / row["n"], abs_tol=1e-12):
            raise ValueError("Aggregate counts disagree with question-level results")
        runs[name] = indexed
    if set(runs) != set(METHODS.values()):
        raise ValueError("Expected exactly the four thesis conditions")
    reference = runs["soft_thinking"]
    for label, name in METHODS.items():
        records = runs[name]
        if records.keys() != reference.keys() or any(records[i]["gt"] != reference[i]["gt"] for i in reference):
            raise ValueError("Conditions do not contain the same questions/ground truths")
        if sum(r["correct"] for r in records.values()) != row["counts"][label]:
            raise ValueError(f"Recorded count differs: {label}")
    b = runs["noisy_discrete"]
    counts = dict(soft_only_correct=sum(reference[i]["correct"] and not b[i]["correct"] for i in reference),
                  noisy_only_correct=sum(b[i]["correct"] and not reference[i]["correct"] for i in reference))
    if row.get("paired_counts") and counts != row["paired_counts"]:
        raise ValueError("Recorded paired counts differ from raw outcomes")
    return counts


def compute(paths):
    output = []
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text())
        runs = {run["config"]["name"]: run for run in data["runs"]}
        if set(runs) != set(METHODS.values()):
            raise ValueError(f"Expected exactly four methods in {path}")
        n = len(runs["soft_thinking"]["results"])
        if n < 2:
            raise ValueError("Paired intervals require at least two questions")
        counts = {label: sum(r["correct"] for r in runs[name]["results"])
                  for label, name in METHODS.items()}
        paired = validate_raw(path, {"n": n, "counts": counts})
        output.append(dict(
            benchmark=data["benchmark"], model=data["model"], n=n,
            count_source="question-level outcomes", raw_verified=True,
            source_file=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            methods={label: dict(correct=k, accuracy_percent=100*k/n, ci95_percent=wilson(k, n))
                     for label, k in counts.items()},
            paired_soft_minus_noisy=paired_interval(paired["soft_only_correct"], paired["noisy_only_correct"], n),
            paired_count_source="raw outcomes validated in this run"))
    if not output:
        raise ValueError("No benchmark result files supplied")
    return output


def report(rows):
    lines = ["# Soft Thinking versus matched random perturbations", "",
             "Wilson 95% intervals use z=1.96. Paired differences use the question-level normal approximation. These intervals do not measure variation across training or decoding seeds.", "",
             "Each input file is evaluated separately; repeated runs are not pooled.", "",
             "| Benchmark | Model | N | Sampled | Soft | Greedy | Noisy | Input |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        values = []
        for label in METHODS:
            v = r["methods"][label]; lo, hi = v["ci95_percent"]
            values.append(f"{v['accuracy_percent']:.2f} [{lo:.2f}, {hi:.2f}]")
        lines.append("| " + " | ".join([r["benchmark"], r["model"], str(r["n"]), *values, Path(r["source_file"]).name]) + " |")
    lines += ["", "## Paired Soft minus Noisy differences", "",
              "| Benchmark | Model | Difference (pp) | 95% CI (pp) | Verification |",
              "|---|---|---:|---:|---|"]
    for r in rows:
        if "paired_soft_minus_noisy" not in r:
            continue
        p = r["paired_soft_minus_noisy"]; lo, hi = p["ci95_pp"]
        lines.append(f"| {r['benchmark']} | {r['model']} | {p['difference_pp']:.2f} | [{lo:.2f}, {hi:.2f}] | {r['paired_count_source']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="JSON files written by the benchmark runners")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory; omission prints JSON only")
    args = parser.parse_args()
    rows = compute(args.inputs)
    if args.output_dir is None:
        print(json.dumps(rows, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(report(rows))
    print(f"Computed {len(rows)} rows in {args.output_dir}")


if __name__ == "__main__":
    main()
