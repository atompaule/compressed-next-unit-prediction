"""Revised EOS-only conditions, with explicit sentence holdout masks."""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
import torch
from torch.nn import functional as F
from models import Model, Config, batch as token_batch
from rotation import SentenceTokenRotation


def chunked_prompt_tensors(prompts, device):
    """Pack existing EOS-delimited premises, retaining BOS and QUERY as markers.

    No catalogue lookup is needed: input premises are composed from their token
    IDs, including premises that are not allowed output candidates in this phase.
    The source prompt excludes BOS, as in the original dataset and start() API.
    """
    sequences = []
    for prompt in prompts:
        if not prompt or prompt[-1] != 4:
            raise ValueError("A premise prompt must end with QUERY (4)")
        chunks, current = [[1]], []
        for token in prompt[:-1]:
            if token == 2:
                if not current:
                    raise ValueError("Empty premise before EOS")
                chunks.append([*current, 2])
                current = []
            elif token < 5:
                raise ValueError("Only lexical tokens and EOS may occur in premises")
            else:
                current.append(token)
        if current or len(chunks) == 1:
            raise ValueError("Each premise must be nonempty and end with EOS")
        sequences.append([*chunks, [4]])
    if not sequences:
        raise ValueError("At least one prompt is required")
    length = max(len(chunks) for chunks in sequences)
    width = max(len(chunk) for chunks in sequences for chunk in chunks)
    tokens = torch.zeros(len(sequences), length, width, dtype=torch.long)
    for row, chunks in enumerate(sequences):
        for col, chunk in enumerate(chunks):
            tokens[row, col, : len(chunk)] = torch.tensor(chunk)
    return tokens.to(device)


def batch(rows, device, prompt_mode="token"):
    result = token_batch(rows, device)
    if prompt_mode == "chunk":
        result["prompt_chunks"] = chunked_prompt_tensors(
            [row["prompt_ids"] for row in rows], device
        )
    elif prompt_mode != "token":
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    return result


class SentenceDot(SentenceTokenRotation):
    def __init__(
        self, catalogue, masks, vocab_size=179, width=512, layers=2,
        *, prompt_mode="token",
    ):
        if prompt_mode not in ("token", "chunk"):
            raise ValueError(f"Unknown prompt mode: {prompt_mode}")
        longest = max(len(c["token_ids"]) for c in catalogue)
        super().__init__(vocab_size, width, layers, 1, longest)
        self.prompt_mode = prompt_mode
        del self.sentence_rates
        del self.selection_vector
        del self.frequencies
        tokens = torch.zeros(len(catalogue), longest, dtype=torch.long)
        for i, c in enumerate(catalogue):
            tokens[i, : len(c["token_ids"])] = torch.tensor(c["token_ids"])
        self.register_buffer("canonical_tokens", tokens)
        self.phase_masks = {k: tuple(v) for k, v in masks.items()}
        self.phase = "train"

    def prompt_embeddings(self, chunks):
        # Reuse output/feedback composition, including differentiable mean norm.
        # Structural markers retain their original discrete input embeddings.
        markers = chunks[..., 0].eq(1) | chunks[..., 0].eq(4)
        embeddings = torch.where(
            markers[..., None], self.emb(chunks[..., 0]), self.feedback(chunks)
        )
        return embeddings, chunks.ne(0).any(-1)

    def contexts(self, b):
        if self.prompt_mode == "token":
            return super().contexts(b)
        if "prompt_chunks" not in b:
            raise ValueError("Chunk prompts require batch(..., prompt_mode='chunk')")
        prefix, prompt_mask = self.prompt_embeddings(b["prompt_chunks"])
        y = b["y"]
        B, S, _ = y.shape
        P = prefix.shape[1]
        x = torch.cat([prefix, self.feedback(y[:, :-1])], 1)
        mask = torch.cat([prompt_mask, y[:, :-1].ne(0).any(-1)], 1)
        pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        h, _ = self.slow(x, keymask=mask, positions=pos)
        ix = torch.cat(
            [
                (prompt_mask.sum(-1) - 1)[:, None],
                torch.arange(P, P + S - 1, device=y.device)[None].expand(B, -1),
            ],
            1,
        )
        return h.gather(1, ix[..., None].expand(-1, -1, h.shape[-1]))

    def start(self, prompt):
        if self.prompt_mode == "token":
            return super().start(prompt)
        chunks = chunked_prompt_tensors([prompt], self.emb.weight.device)
        embeddings, _ = self.prompt_embeddings(chunks)
        h, cache = self.slow(embeddings, cache=True)
        return h[:, -1], cache

    def set_phase(self, phase):
        if phase not in self.phase_masks:
            raise ValueError(phase)
        self.phase = phase

    def candidate_ids(self):
        return torch.tensor(self.phase_masks[self.phase], device=self.emb.weight.device)

    def candidate_features(self):
        # Construct embeddings only for this phase's allowed sentences.
        y = self.canonical_tokens[self.candidate_ids()]
        pos = torch.arange(1, y.shape[1] + 1, device=y.device)[:, None] * self.alpha
        table = self.rotation(self.emb.weight[None, :, :], pos[:, None, :])
        mask = y.ne(0) & (y.eq(2).cumsum(-1) - y.eq(2).long()).eq(0)
        total = table[0, y[:, 0]] * mask[:, 0, None]
        for t in range(1, y.shape[1]):
            total = total + table[t, y[:, t]] * mask[:, t, None]
        return (
            F.normalize(total, dim=-1, eps=1e-8)
            * self.emb.weight[5:].norm(dim=-1).mean()
        )

    def plan_log_probs(self, h):
        return (h @ self.candidate_features().T).log_softmax(-1)

    def local_target_ids(self, global_ids):
        mapping = torch.full(
            (len(self.canonical_tokens),),
            -1,
            dtype=torch.long,
            device=self.emb.weight.device,
        )
        active = self.candidate_ids()
        mapping[active] = torch.arange(len(active), device=active.device)
        result = mapping[global_ids]
        if (result < 0).any():
            raise ValueError("Target excluded from this phase")
        return result


class SentenceDotFixedPE(SentenceDot):
    """Keep the seed-matched, nonzero positional rates fixed throughout training.

    All initialization draws match SentenceDot. Only alpha becomes a buffer;
    token embeddings, backbone weights and differentiable normalization train
    normally. The same fixed rates serve readout, feedback and chunked premises.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fixed_rates = self.alpha.detach().clone()
        del self.alpha
        self.register_buffer("alpha", fixed_rates)


class SentenceDotNoPE(SentenceDot):
    """Unrotated sentence composition; backbone sequence positions stay enabled.

    Initialize identically to SentenceDot, then replace the rotary rates with a
    fixed zero buffer. Both candidate scoring and feedback consequently use
    normalized sums of the same token embeddings, including EOS.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        zero_rates = torch.zeros_like(self.alpha)
        del self.alpha
        self.register_buffer("alpha", zero_rates)


def discrete(vocab_size=179, width=512, layers=2):
    return Model(
        Config(
            kind="token_only",
            vocab=vocab_size,
            width=width,
            heads=4,
            token_layers=layers,
        )
    )
