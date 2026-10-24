"""Load both arithmetic v1 formats without changing their split or row checks."""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.arithmetic_reasoning import generate_data as data

SPLITS = data.SPLITS


class Vocabulary:
    def __init__(self, config):
        config.validate()
        self.roles = config.roles
        self.nodes = tuple(map(str, range(config.num_values)))
        self.tokens = (
            "<PAD>",
            "<BOS>",
            "<LATENT>",
            "<OPS>",
            "<TRACE>",
            "<QUERY>",
            "ADD",
            "SUBTRACT",
            "SWITCH",
            *self.roles,
            *self.nodes,
            *map(str, range(config.num_values, config.path_length + 1)),
        )
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.node_ids = tuple(self.id(x) for x in self.nodes)

    def id(self, token):
        return self.token_to_id[token]

    def encode(self, tokens):
        return [self.id(t) for t in tokens]

    def __len__(self):
        return len(self.tokens)


def load_data_config(manifest):
    return data.config_from_manifest(manifest)


def load_manifest(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    load_data_config(manifest)
    return manifest


class EncodedSplit(Dataset):
    def __init__(self, directory, split, manifest, vocabulary):
        config = load_data_config(manifest)
        if vocabulary.tokens != Vocabulary(config).tokens:
            raise ValueError("Vocabulary differs from dataset configuration")
        width = len(vocabulary.roles)
        partition = data.StatePartition(config)
        if partition.fingerprint != manifest["partition_fingerprint"]:
            raise ValueError("Dataset partition differs from generator")
        n, length = manifest["counts"][split], config.path_length
        self.prompt_length = 3 + width + 2 * width * length
        if self.prompt_length != manifest["stats"][split]["prompt_positions"]:
            raise ValueError("Incorrect manifest prompt length")
        self.tensors = dict(
            prompt_ids=torch.empty((n, self.prompt_length), dtype=torch.long),
            state_ids=torch.empty((n, length, width), dtype=torch.long),
            query_ids=torch.empty((n, 3), dtype=torch.long),
            query_targets=torch.empty(n, dtype=torch.long),
            query_roles=torch.empty(n, dtype=torch.long),
            historical_step=torch.empty(n, dtype=torch.long),
            heldout=torch.empty((n, length), dtype=torch.bool),
        )
        file_hash = hashlib.sha256()
        count = 0
        self.fingerprints = set()
        with (Path(directory) / f"{split}.jsonl").open("rb") as f:
            for line in f:
                file_hash.update(line)
                s = json.loads(line)
                data.validate_sample(s, config, partition, task=manifest["task"])
                if (
                    count >= n
                    or s["split"] != split
                    or s["fingerprint"] in self.fingerprints
                ):
                    raise ValueError("Extra/duplicate/misplaced dataset row")
                self.fingerprints.add(s["fingerprint"])
                t = self.tensors
                t["prompt_ids"][count] = torch.tensor(
                    vocabulary.encode(s["prompt_tokens"])
                )
                t["state_ids"][count] = torch.tensor(
                    [
                        [vocabulary.node_ids[v] for v in state]
                        for state in s["gold_states"]
                    ]
                )
                t["query_ids"][count] = torch.tensor(
                    vocabulary.encode(
                        [
                            "<QUERY>",
                            str(s["training_query_step"]),
                            vocabulary.roles[s["query_role"]],
                        ]
                    )
                )
                t["query_targets"][count] = vocabulary.node_ids[
                    s["training_query_answer"]
                ]
                t["query_roles"][count] = s["query_role"]
                t["historical_step"][count] = s["historical_probe_step"]
                t["heldout"][count] = torch.tensor(
                    [x != "train" for x in s["state_partitions"]]
                )
                count += 1
        if count != n or file_hash.hexdigest() != manifest["stats"][split]["sha256"]:
            raise ValueError("Dataset row count or hash mismatch")
        self.config, self.split = config, split

    def __len__(self):
        return len(self.tensors["query_roles"])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.tensors.items()}
