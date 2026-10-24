"""Plot the metrics available for a newly trained or resumed RL run."""
import argparse
import csv
import json
from pathlib import Path


def load_metrics(root):
    config = json.loads((root / "run_config.json").read_text())
    if config["step_indexing"] != "completed batches, one-based":
        raise ValueError("Expected the current one-based batch protocol")
    maximum = config["preset"]["rl_config"]["max_batches"]
    records = {}
    for path in sorted(root.glob("step_*/metrics_step_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = row["step"]
            if type(step) is not int or not 1 <= step <= maximum:
                raise ValueError(f"Invalid completed batch in {path}: {step}")
            if step in records:
                # Replay after a crash can repeat metrics with a different clock.
                keys = ("loss", "reward", "correct", "mean_len", "temp", "skipped")
                if any(records[step][key] != row[key] for key in keys):
                    raise ValueError(f"Conflicting replay metrics for batch {step}")
            records[step] = row
    if not records:
        raise ValueError(f"No training metrics found in {root}")
    return config, [records[step] for step in sorted(records)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config, rows = load_metrics(args.run_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    for axis, key, label in zip(axes, ("correct", "reward", "mean_len"),
                                ("Correct fraction", "Shaped reward", "Mean reasoning length")):
        axis.plot([r["step"] for r in rows], [r[key] for r in rows], linewidth=.8)
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Completed training batches")
    fig.suptitle(config["preset"]["rl_config"]["approach"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"training.{extension}", dpi=160)
    plt.close(fig)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected = config["preset"]["rl_config"]["max_batches"]
    summary = dict(run_dir=str(args.run_dir.resolve()), configuration=config,
                   logged_batches=len(rows), last_logged_batch=rows[-1]["step"],
                   expected_batches=expected,
                   missing_batches=sorted(set(range(1, expected + 1)) - {r["step"] for r in rows}),
                   final_logged_metrics=rows[-1])
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Plotted {len(rows)} logged batches in {args.output_dir}")


if __name__ == "__main__":
    main()
