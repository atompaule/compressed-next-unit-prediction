#!/usr/bin/env python3
"""Reversible arithmetic register traces with deliberately held-out complete states at configurable register width.

Only the standard library is required. The complete instruction sequence is in
the prefix; final and historical queries are disclosed only after the trace.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import asdict, dataclass
from itertools import combinations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from src.common.experiment_paths import data_dir
import random

SPLITS = ("train", "validation", "test")
TASK = "arithmetic_register"
WIDTH_TASK = "arithmetic_register_width"
TASKS = (TASK, WIDTH_TASK)
ROLES = ("A", "B", "C")  # Original three-register format.
FORMAT_VERSION = 1


@dataclass(frozen=True)
class DataConfig:
    num_registers: int = 3
    num_values: int = 128
    offsets: tuple[int, ...] = (1, 2, 3, 8)
    path_length: int = 12
    data_seed: int = 20260905
    heldout_fraction: float = 0.05  # EACH evaluation split
    swap_probability: float = 0.5
    max_attempts_per_sample: int = 2000

    def __post_init__(self):
        object.__setattr__(self, "offsets", tuple(self.offsets))

    @property
    def roles(self):
        return tuple(chr(65 + i) for i in range(self.num_registers))

    def validate(self):
        if not 2 <= self.num_registers <= 26:
            raise ValueError("Require 2..26 register slots")
        if not 4 <= self.num_values <= 256 or not 1 <= self.path_length <= 256:
            raise ValueError("Require 4..256 values and 1..256 updates")
        if not self.offsets or len(set(self.offsets)) != len(self.offsets):
            raise ValueError("Offsets must be nonempty and distinct")
        if any(
            not isinstance(x, int) or not 0 < x < self.num_values for x in self.offsets
        ):
            raise ValueError("Offsets must be positive integers below num_values")
        if not 0 < self.heldout_fraction < 0.2 or not 0 <= self.swap_probability <= 1:
            raise ValueError("Invalid heldout_fraction or swap_probability")
        if self.max_attempts_per_sample < 1:
            raise ValueError("max_attempts_per_sample must be positive")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stream_seed(seed, *parts):
    return int(digest([seed, *parts])[:16], 16)


@lru_cache(maxsize=4)
def partition_labels(n, seed, fraction):
    """Compact exact partition: ~2 MB at 128 values, with no tuple-set expansion."""
    total = n**3
    count = max(1, int(total * fraction))
    reserved = random.Random(stream_seed(seed, "arithmetic-state-partition")).sample(
        range(total), 2 * count
    )
    labels = bytearray(total)
    for index in reserved[:count]:
        labels[index] = 1
    for index in reserved[count:]:
        labels[index] = 2
    return bytes(labels)


class StatePartition:
    """Legacy exact partition for K=3; reproducible constant-memory hash for K!=3.

    Hash thresholds define disjoint supports with nominal 5% per held-out split.
    Their exact population sizes are deliberately not claimed or enumerated.
    """

    def __init__(self, config):
        config.validate()
        self.n, self.width = config.num_values, config.num_registers
        if self.width == 3:
            self.labels = partition_labels(
                self.n, config.data_seed, config.heldout_fraction
            )
            self.fingerprint = hashlib.sha256(self.labels).hexdigest()
            self.sizes = {split: self.labels.count(i) for i, split in enumerate(SPLITS)}
            self.spec = dict(algorithm="legacy_exact_v1", exact_sizes=True)
        else:
            self.labels = None
            self.threshold = int(config.heldout_fraction * (1 << 64))
            self.spec = dict(
                algorithm="blake2b64_state_bytes_v1",
                width=self.width,
                num_values=self.n,
                seed=config.data_seed,
                threshold=self.threshold,
                exact_sizes=False,
                nominal_fractions=dict(
                    train=1 - 2 * config.heldout_fraction,
                    validation=config.heldout_fraction,
                    test=config.heldout_fraction,
                ),
            )
            self.fingerprint = digest(self.spec)
            self.prefix = bytes.fromhex(self.fingerprint)
            self.sizes = None

    def index(self, state):
        if len(state) != self.width or any(
            not isinstance(v, int) or not 0 <= v < self.n for v in state
        ):
            raise ValueError("Invalid register state")
        result = 0
        for value in state:
            result = result * self.n + value
        return result

    def label(self, state):
        index = self.index(state)
        if self.labels is not None:
            return SPLITS[self.labels[index]]
        value = int.from_bytes(
            hashlib.blake2b(self.prefix + bytes(state), digest_size=8).digest(), "big"
        )
        return (
            "validation"
            if value < self.threshold
            else "test" if value < 2 * self.threshold else "train"
        )


def compile_operation(operation, config):
    """K slot instructions, executed left to right, compiled against OLD state.

    Example ADD 8 / SUBTRACT 3 / SWITCH B compiles to sources [A,C,B]
    and offsets [8,0,-3]. Every output uses exactly one old value, with no loss.
    """
    if len(operation) != config.num_registers:
        raise ValueError("An update must contain one instruction per register")
    sources, offsets, switches = (
        list(range(config.num_registers)),
        [0] * config.num_registers,
        0,
    )
    for slot, instruction in enumerate(operation):
        if len(instruction) != 2:
            raise ValueError("Instructions have opcode and argument")
        kind, arg = instruction
        if kind in ("ADD", "SUBTRACT"):
            if arg not in config.offsets:
                raise ValueError("Arithmetic operand outside configured offset set")
            offsets[slot] += arg if kind == "ADD" else -arg
        elif kind == "SWITCH":
            if arg not in config.roles or config.roles.index(arg) == slot:
                raise ValueError("SWITCH must name another register")
            other = config.roles.index(arg)
            sources[slot], sources[other] = sources[other], sources[slot]
            offsets[slot], offsets[other] = offsets[other], offsets[slot]
            switches += 1
        else:
            raise ValueError("Unknown arithmetic opcode")
    if switches > 1:
        raise ValueError("At most one SWITCH per update")
    return sources, [x % config.num_values for x in offsets]


def apply_operation(state, operation, config):
    sources, offsets = compile_operation(operation, config)
    return tuple(
        (state[source] + offset) % config.num_values
        for source, offset in zip(sources, offsets)
    )


def sample_operation(rng, config):
    operation = [
        [rng.choice(("ADD", "SUBTRACT")), rng.choice(config.offsets)]
        for _ in config.roles
    ]
    if rng.random() < config.swap_probability:
        slot = rng.randrange(config.num_registers)
        operation[slot] = [
            "SWITCH",
            rng.choice([r for r in config.roles if r != config.roles[slot]]),
        ]
    return operation


def prompt_tokens(sample):
    return [
        "<BOS>",
        *map(str, sample["initial_state"]),
        "<OPS>",
        *[
            str(token)
            for op in sample["operations"]
            for instruction in op
            for token in instruction
        ],
        "<TRACE>",
    ]


def episode_fingerprint(sample):
    return digest({k: sample[k] for k in ("initial_state", "operations")})


def resolve_task(config, task=None):
    """Keep both v1 serializations; original v1 is only defined for three slots."""
    config.validate()
    if task is None:
        task = TASK if config.num_registers == 3 else WIDTH_TASK
    if task not in TASKS or (task == TASK and config.num_registers != 3):
        raise ValueError("Unsupported task or register count for this dataset format")
    return task


def config_from_manifest(manifest):
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("task") not in TASKS
    ):
        raise ValueError("Wrong arithmetic dataset format")
    config = DataConfig(**manifest["data_config"])
    resolve_task(config, manifest["task"])
    return config


def iter_dataset(*, num_samples, split, config, partition=None, task=None):
    task = resolve_task(config, task)
    if split not in SPLITS or num_samples < 1:
        raise ValueError("Invalid split/count")
    partition = partition or StatePartition(config)
    seen = set()
    for index in range(num_samples):
        rng = random.Random(stream_seed(config.data_seed, split, index))
        for attempt in range(config.max_attempts_per_sample):
            initial = tuple(rng.randrange(config.num_values) for _ in config.roles)
            if partition.label(initial) != "train":
                continue
            state, states, labels, operations, visited = initial, [], [], [], {initial}
            for _ in range(config.path_length):
                operation = sample_operation(rng, config)
                state = apply_operation(state, operation, config)
                label = partition.label(state)
                if state in visited or label not in ("train", split):
                    break
                visited.add(state)
                states.append(list(state))
                labels.append(label)
                operations.append(operation)
            if len(states) != config.path_length:
                continue
            required = labels[:-1] if config.path_length > 1 else labels
            if split != "train" and split not in required:
                continue
            historical = (
                1 + index % (config.path_length - 1) if config.path_length > 1 else 1
            )
            query_step = config.path_length if index % 2 == 0 else historical
            sample = dict(
                format_version=FORMAT_VERSION,
                task=task,
                id=f"arithmetic_{split}_{index:07d}",
                split=split,
                initial_state=list(initial),
                operations=operations,
                gold_states=states,
                state_partitions=labels,
                query_role=(
                    index % config.num_registers
                    if config.num_registers % 2
                    else (index // 2) % config.num_registers
                ),
                training_query_step=query_step,
                historical_probe_step=historical,
                training_query_answer=states[query_step - 1][
                    (
                        index % config.num_registers
                        if config.num_registers % 2
                        else (index // 2) % config.num_registers
                    )
                ],
                generation_attempts=attempt + 1,
                partition_fingerprint=partition.fingerprint,
            )
            sample["fingerprint"] = episode_fingerprint(sample)
            if sample["fingerprint"] in seen:
                continue
            sample["prompt_tokens"] = prompt_tokens(sample)
            seen.add(sample["fingerprint"])
            yield sample
            break
        else:
            raise RuntimeError(
                f"Rejection limit at {split} sample {index}; reduce path length/heldout fraction"
            )


def validate_sample(sample, config, partition, *, task=None):
    expected_task = resolve_task(config, sample.get("task") if task is None else task)
    if (
        sample["format_version"] != FORMAT_VERSION
        or sample["task"] != expected_task
        or sample["split"] not in SPLITS
    ):
        raise ValueError("Unsupported dataset format/task/split")
    if sample["partition_fingerprint"] != partition.fingerprint:
        raise ValueError("Partition fingerprint mismatch")
    if len(sample["operations"]) != config.path_length:
        raise ValueError("Wrong operation count")
    state = tuple(sample["initial_state"])
    if partition.label(state) != "train":
        raise ValueError("Initial states must be from training support")
    states, labels, visited = [], [], {state}
    for operation in sample["operations"]:
        state = apply_operation(state, operation, config)
        if state in visited:
            raise ValueError("Repeated complete state")
        visited.add(state)
        states.append(list(state))
        labels.append(partition.label(state))
    if states != sample["gold_states"] or labels != sample["state_partitions"]:
        raise ValueError("Incorrect execution trace")
    if any(label not in ("train", sample["split"]) for label in labels):
        raise ValueError("Reserved state leaked across splits")
    if sample["split"] != "train" and sample["split"] not in (
        labels[:-1] if len(labels) > 1 else labels
    ):
        raise ValueError("Missing required reserved state/subsequent computation")
    step, role, historical = (
        sample["training_query_step"],
        sample["query_role"],
        sample["historical_probe_step"],
    )
    if role not in range(config.num_registers) or not 1 <= step <= config.path_length:
        raise ValueError("Invalid training query")
    if not 1 <= historical <= max(1, config.path_length - 1):
        raise ValueError("Invalid historical probe index")
    if sample["training_query_answer"] != states[step - 1][role]:
        raise ValueError("Incorrect query answer")
    if sample["prompt_tokens"] != prompt_tokens(sample) or sample[
        "fingerprint"
    ] != episode_fingerprint(sample):
        raise ValueError("Incorrect prompt/fingerprint")


def render_episode(sample):
    role_names = tuple(chr(65 + i) for i in range(len(sample["initial_state"])))
    lines = [
        "# Arithmetic-register example",
        "",
        "Exact atomic prefix (line breaks are for readability):",
        "",
        "```text",
        "<BOS> " + " ".join(map(str, sample["initial_state"])),
        "<OPS>",
    ]
    lines += [
        " ".join(str(x) for instruction in op for x in instruction)
        for op in sample["operations"]
    ]
    lines += [
        "<TRACE>",
        "```",
        "",
        (
            "Reference completions; z(a,b,c) denotes one role-bound vector, not text."
            if sample["task"] == TASK
            else "Reference completions; z(v_1,...,v_K) denotes one role-bound vector, not text."
        ),
        "",
        "| Step | Discrete | Continuous | State partition |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {i} | {' '.join(map(str,state))} | z({','.join(map(str,state))}) | {label} |"
        for i, (state, label) in enumerate(
            zip(sample["gold_states"], sample["state_partitions"]), 1
        )
    ]
    length = len(sample["gold_states"])
    lines += [
        "",
        "Independent final probes from copies of the completed cache:",
        "",
        "```text",
    ]
    lines += [
        f"<QUERY> {length} {r} -> {v}"
        for r, v in zip(role_names, sample["gold_states"][-1])
    ]
    lines += ["```"]
    if length > 1:
        h, r = sample["historical_probe_step"], sample["query_role"]
        lines += [
            "",
            "Separate historical diagnostic:",
            "",
            f"`<QUERY> {h} {role_names[r]} -> {sample['gold_states'][h-1][r]}`",
        ]
    return "\n".join(lines) + "\n"


def write_dataset(output_dir, config, counts, *, task=None):
    task = resolve_task(config, task)
    if set(counts) != set(SPLITS) or any(n < 1 for n in counts.values()):
        raise ValueError("Positive counts required for all splits")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (output_dir / name).exists()
        for name in [*[s + ".jsonl" for s in SPLITS], "manifest.json", "example.md"]
    ):
        raise FileExistsError("Dataset already exists; choose a new directory")
    partition = StatePartition(config)
    stats, fingerprints, actual_pairs = {}, set(), {}
    role_pairs = tuple(combinations(range(config.num_registers), 2))
    for split in SPLITS:
        states, reserved, pairs, values, primitive_cases = (
            set(),
            set(),
            [set() for _ in role_pairs],
            [set() for _ in config.roles],
            set(),
        )
        hasher = hashlib.sha256()
        attempts = 0
        swaps = 0
        final_queries = 0
        with (output_dir / f"{split}.jsonl").open("w") as handle:
            for index, sample in enumerate(
                iter_dataset(
                    num_samples=counts[split],
                    split=split,
                    config=config,
                    partition=partition,
                    task=task,
                )
            ):
                validate_sample(sample, config, partition, task=task)
                if sample["fingerprint"] in fingerprints:
                    raise ValueError("Duplicate episode across splits")
                fingerprints.add(sample["fingerprint"])
                trace = [sample["initial_state"], *sample["gold_states"]]
                for state in trace:
                    states.add(partition.index(state))
                    for i, (a, b) in enumerate(role_pairs):
                        pairs[i].add((state[a], state[b]))
                    for i, value in enumerate(state):
                        values[i].add(value)
                    if partition.label(state) == split and split != "train":
                        reserved.add(tuple(state))
                for old, op in zip(trace, sample["operations"]):
                    current = list(old)
                    for slot, (kind, arg) in enumerate(op):
                        if kind == "SWITCH":
                            other = config.roles.index(arg)
                            current[slot], current[other] = (
                                current[other],
                                current[slot],
                            )
                            swaps += 1
                        else:
                            primitive_cases.add((kind, arg, current[slot]))
                            current[slot] = (
                                current[slot] + (arg if kind == "ADD" else -arg)
                            ) % config.num_values
                attempts += sample["generation_attempts"]
                final_queries += sample["training_query_step"] == config.path_length
                line = json.dumps(sample, separators=(",", ":")) + "\n"
                handle.write(line)
                hasher.update(line.encode())
                if split == "test" and index == 0:
                    (output_dir / "example.md").write_text(render_episode(sample))
                if (index + 1) % 10000 == 0:
                    print(f"{split}: {index+1:,}/{counts[split]:,}", flush=True)
        actual_pairs[split] = pairs
        stats[split] = dict(
            samples=counts[split],
            sha256=hasher.hexdigest(),
            prompt_positions=len(sample["prompt_tokens"]),
            unique_states=len(states),
            unique_reserved_states=len(reserved),
            generation_attempts=attempts,
            swaps=swaps,
            training_final_queries=final_queries,
            training_historical_queries=counts[split] - final_queries,
            role_value_counts=[len(x) for x in values],
            role_pair_counts=[len(x) for x in pairs],
            all_role_pairs_observed=all(len(x) == config.num_values**2 for x in pairs),
            primitive_cases_observed=len(primitive_cases),
            primitive_cases_possible=2 * len(config.offsets) * config.num_values,
            all_primitive_cases_observed=len(primitive_cases)
            == 2 * len(config.offsets) * config.num_values,
        )
        if split != "train":
            stats[split]["reserved_states_with_all_pairs_seen_in_training"] = sum(
                all(
                    (s[a], s[b]) in actual_pairs["train"][i]
                    for i, (a, b) in enumerate(role_pairs)
                )
                for s in reserved
            )
    # Preserve historical metadata and its field order so old bundles reproduce
    # byte for byte. Sampling, execution, validation and statistics are shared.
    serialized_config = asdict(config)
    if task == TASK:
        serialized_config.pop("num_registers")
        manifest = dict(
            format_version=FORMAT_VERSION,
            task=task,
            data_config=serialized_config,
            counts=counts,
            stats=stats,
            partition_fingerprint=partition.fingerprint,
            partition_sizes=partition.sizes,
            semantics="A/B/C slots, instructions left to right; each row compiles to old-state permutation plus modular offsets; at most one swap",
            input_protocol="entire program precedes generated trace; no probe in prefix; no intermediate arithmetic substeps in the completion",
            split_contract="reserved triples excluded from all training states; evaluation starts in training support and encounters own reserved states; L>1 requires further computation",
            query_protocol="one training query per trace, half final and half historical for L>1; independent final A/B/C branches primary; historical branches diagnostic; step indices are one-based",
        )
    else:
        manifest = dict(
            format_version=FORMAT_VERSION,
            task=task,
            data_config=serialized_config,
            counts=counts,
            stats=stats,
            partition_fingerprint=partition.fingerprint,
            partition_sizes=partition.sizes,
            partition_spec=partition.spec,
            role_pairs=[[config.roles[a], config.roles[b]] for a, b in role_pairs],
            semantics="ordered register slots, instructions left to right; each row compiles to old-state permutation plus modular offsets; at most one swap",
            input_protocol="entire program precedes generated trace; no probe in prefix; no intermediate arithmetic substeps in the completion",
            split_contract="reserved complete states excluded from all training states; evaluation starts in training support and encounters own reserved states; L>1 requires further computation",
            query_protocol="one training query per trace, half final and half historical for L>1; independent final branches for every register primary; historical branches diagnostic; step indices are one-based",
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir("arithmetic_reasoning", "register"),
    )
    for name, default in [("train", 65536), ("validation", 4096), ("test", 4096)]:
        parser.add_argument(f"--{name}-samples", type=int, default=default)
    for name in [
        "num_registers",
        "num_values",
        "path_length",
        "data_seed",
        "max_attempts_per_sample",
    ]:
        parser.add_argument(
            "--" + name.replace("_", "-"), type=int, default=getattr(DataConfig(), name)
        )
    parser.add_argument(
        "--offsets", nargs="+", type=int, default=list(DataConfig().offsets)
    )
    parser.add_argument("--heldout-fraction", type=float, default=0.05)
    parser.add_argument("--swap-probability", type=float, default=0.5)
    parser.add_argument(
        "--format",
        choices=("original", "width"),
        default=None,
        help="Optional v1 serialization; default is original for three registers and width otherwise",
    )
    args = parser.parse_args()
    config = DataConfig(
        **{k: getattr(args, k) for k in DataConfig.__dataclass_fields__}
    )
    task = {"original": TASK, "width": WIDTH_TASK}.get(args.format)
    print(
        json.dumps(
            write_dataset(
                args.output_dir,
                config,
                {s: getattr(args, s + "_samples") for s in SPLITS},
                task=task,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
