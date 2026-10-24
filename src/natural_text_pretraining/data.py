"""Memory-mapped chunk catalogue and document-disjoint training windows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from objective import encode_chunks


def load_manifest(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("chunk_boundary") != "metadata_only"
        or manifest.get("chunk_end_id") is not None
    ):
        raise ValueError(
            "This runner requires marker-free schema v2 data. "
            "Build data with prepare.py into a new directory."
        )
    return manifest


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


class PackedBank:
    def __init__(self, folder: Path, name: str):
        self.tokens = np.load(folder / f"{name}_tokens.npy", mmap_mode="r")
        self.offsets = np.load(folder / f"{name}_offsets.npy", mmap_mode="r")
        assert self.offsets[0] == 0 and self.offsets[-1] == len(self.tokens)

    def close(self):
        """Release file mappings before deleting a catalogue (including on NFS)."""
        self.tokens._mmap.close()
        self.offsets._mmap.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        return len(self.offsets) - 1

    def blocks(self, size: int, device) -> tuple:
        for lo in range(0, len(self), size):
            hi = min(lo + size, len(self))
            start, stop = int(self.offsets[lo]), int(self.offsets[hi])
            ids = torch.from_numpy(
                np.array(self.tokens[start:stop], dtype=np.int64)
            ).to(device)
            offsets = torch.from_numpy(
                np.array(self.offsets[lo : hi + 1] - start, dtype=np.int64)
            ).to(device)
            yield lo, hi, ids, offsets

    def encode_ids(self, ids, weight, alpha, scale):
        if not ids:
            return weight.new_empty((0, weight.shape[1]))
        sequences = [
            self.tokens[int(self.offsets[i]) : int(self.offsets[i + 1])] for i in ids
        ]
        offsets = np.r_[0, np.cumsum([len(s) for s in sequences])]
        tokens = torch.from_numpy(np.concatenate(sequences).astype(np.int64)).to(
            weight.device
        )
        boundaries = torch.from_numpy(offsets.astype(np.int64)).to(weight.device)
        return encode_chunks(weight, alpha, tokens, boundaries, scale)

    def sequence(self, i):
        return self.tokens[int(self.offsets[i]) : int(self.offsets[i + 1])].tolist()


class Windows:
    def __init__(self, folder: Path, split: str):
        self.context = np.load(folder / f"{split}_context.npy", mmap_mode="r")
        self.context_offsets = np.load(
            folder / f"{split}_context_offsets.npy", mmap_mode="r"
        )
        self.target = np.load(folder / f"{split}_target.npy", mmap_mode="r")
        self.target_offsets = np.load(
            folder / f"{split}_target_offsets.npy", mmap_mode="r"
        )
        self.meta = folder / f"{split}_meta.jsonl"
        assert len(self.context_offsets) == len(self.target_offsets)
        assert self.context_offsets[-1] == len(self.context)
        assert self.target_offsets[-1] == len(self.target)

    def __len__(self):
        return len(self.context_offsets) - 1

    def row(self, i):
        c0, c1 = int(self.context_offsets[i]), int(self.context_offsets[i + 1])
        t0, t1 = int(self.target_offsets[i]), int(self.target_offsets[i + 1])
        return self.context[c0:c1].tolist(), self.target[t0:t1].tolist()

    def batch(self, indices):
        return [self.row(int(i)) for i in indices]


def save_bank(folder: Path, name: str, sequences: list[list[int]]) -> None:
    offsets = np.r_[0, np.cumsum([len(s) for s in sequences], dtype=np.int64)]
    np.save(
        folder / f"{name}_tokens.npy",
        np.asarray([t for s in sequences for t in s], dtype=np.int32),
    )
    np.save(folder / f"{name}_offsets.npy", offsets)


def save_windows(folder: Path, split: str, rows, metadata) -> None:
    for part in ("context", "target"):
        values = [r[0 if part == "context" else 1] for r in rows]
        offsets = np.r_[0, np.cumsum([len(v) for v in values], dtype=np.int64)]
        np.save(
            folder / f"{split}_{part}.npy",
            np.asarray([i for v in values for i in v], dtype=np.int32),
        )
        np.save(folder / f"{split}_{part}_offsets.npy", offsets)
    with (folder / f"{split}_meta.jsonl").open("w") as stream:
        for item in metadata:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
