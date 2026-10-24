"""Matched GPT-style models with explicit causal, cached inference paths."""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    kind: str = "continuous"
    vocab: int = 256
    width: int = 128
    heads: int = 4
    max_positions: int = 256
    slow_layers: int = 2
    token_layers: int = 2


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.width
        self.heads = c.heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x, keymask=None, past=None, cache=False):
        B, T, D = x.shape
        H = self.heads
        q, k, v = (
            self.qkv(self.ln1(x))
            .reshape(B, T, 3, H, D // H)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        offset = 0 if past is None else past[0].shape[-2]
        if past is not None:
            k = torch.cat([past[0], k], -2)
            v = torch.cat([past[1], v], -2)
        allowed = (
            torch.arange(k.shape[-2], device=x.device)[None, :]
            <= torch.arange(offset, offset + T, device=x.device)[:, None]
        )
        allowed = allowed[None, None]
        if keymask is not None:
            allowed = allowed & keymask[:, None, None, :].bool()
        a = (
            F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
            .transpose(1, 2)
            .reshape(B, T, D)
        )
        x = x + self.proj(a)
        x = x + self.down(F.gelu(self.up(self.ln2(x)), approximate="tanh"))
        return x, (k, v) if cache else None


class Stack(nn.Module):
    def __init__(self, c, layers):
        super().__init__()
        self.position = nn.Embedding(c.max_positions, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(layers)])
        self.norm = nn.LayerNorm(c.width)

    def forward(self, x, keymask=None, positions=None, past=None, cache=False):
        offset = 0 if past is None else past[0][0].shape[-2]
        if positions is None:
            positions = torch.arange(offset, offset + x.shape[1], device=x.device)[None]
        x = x + self.position(positions)
        out = []
        for i, b in enumerate(self.blocks):
            x, kv = b(x, keymask, None if past is None else past[i], cache)
            if cache:
                out.append(kv)
        return self.norm(x), out if cache else None


def rms(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


class Model(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.emb = nn.Embedding(c.vocab, c.width, padding_idx=0)
        self.continuous = c.kind.startswith("continuous")
        if self.continuous:
            self.slow = Stack(c, c.slow_layers)
            self.fast = Stack(c, c.token_layers)
            self.gate = nn.Parameter(torch.tensor(1.0))
        else:
            self.fast = Stack(c, 4 if c.kind == "discrete" else c.token_layers)
        self.register_buffer(
            "freq",
            torch.exp(-math.log(10000) * torch.arange(0, c.width, 2) / c.width),
            persistent=False,
        )
        self.apply(self.initialize)
        with torch.no_grad():
            self.emb.weight[0].zero_()

    @staticmethod
    def initialize(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def rotate(self, x, positions, inverse=False):
        angle = positions[..., None] * self.freq
        if inverse:
            angle = -angle
        co, si = angle.cos(), angle.sin()
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack([a * co - b * si, a * si + b * co], -1).flatten(-2)

    def scale(self):
        return self.emb.weight[5:].norm(dim=-1).mean()

    def feedback(self, ids):
        e = self.emb(ids)
        pos = torch.arange(1, ids.shape[-1] + 1, device=ids.device)
        e = self.rotate(e, pos)
        e = e * ids.ne(0)[..., None]
        return F.normalize(e.sum(-2), dim=-1, eps=1e-8) * self.scale()

    def injection(self, h, positions, mode="full"):
        z = (
            h
            if self.config.kind == "continuous_unrotated"
            else self.rotate(h, positions, inverse=True)
        )
        if mode == "no_h":
            z = torch.zeros_like(z)
        return (
            torch.tanh(self.gate) * rms(z) * self.scale() / math.sqrt(self.config.width)
        )

    def head(self, h):
        logits = F.linear(h, self.emb.weight)
        # PAD, BOS and QUERY are input/control-only. EOS and DONE remain learnable outputs.
        logits[..., 0] = -1e4
        logits[..., 1] = -1e4
        logits[..., 4] = -1e4
        return logits

    def contexts(self, b):
        p = b["prompt"]
        y = b["y"]
        B, S, L = y.shape
        P = p.shape[1]
        f = self.feedback(y[:, :-1])
        x = torch.cat([self.emb(p), f], 1)
        mask = torch.cat([p.ne(0), y[:, :-1].ne(0).any(-1)], 1)
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        h, _ = self.slow(x, keymask=mask, positions=positions)
        ix = torch.cat(
            [
                (p.ne(0).sum(-1) - 1)[:, None],
                torch.arange(P, P + S - 1, device=p.device)[None].expand(B, -1),
            ],
            1,
        )
        return h.gather(1, ix[..., None].expand(-1, -1, h.shape[-1]))

    def forward(self, b, mode="full"):
        if self.continuous:
            y = b["y"]
            B, S, L = y.shape
            h = self.contexts(b)
            if mode == "swap_h":
                h = h.roll(1, 0)
            prev = torch.cat([torch.ones_like(y[..., :1]), y[..., :-1]], -1)
            if mode == "no_prefix":
                prev = torch.ones_like(prev)
            z = self.injection(
                h[:, :, None].expand(-1, -1, L, -1),
                torch.arange(1, L + 1, device=y.device),
                mode,
            )
            x = self.emb(prev) + z
            out, _ = self.fast(x.reshape(B * S, L, -1))
            return self.head(out).reshape(B, S, L, -1)
        x = b["discrete"]
        mask = x.ne(0)
        h, _ = self.fast(self.emb(x), keymask=mask)
        ix = b["indices"]
        h = h[torch.arange(x.shape[0], device=x.device)[:, None, None], ix]
        return self.head(h)

    @torch.no_grad()
    def start(self, prompt, mode="full"):
        p = torch.tensor([[1, *prompt]], device=self.emb.weight.device)
        stack = self.slow if self.continuous else self.fast
        hh, cache = stack(self.emb(p), cache=True)
        return dict(
            h=hh[:, -1],
            slow_cache=cache if self.continuous else None,
            fast_cache=None if self.continuous else cache,
            prev=1,
            pos=0,
            mode=mode,
        )

    @torch.no_grad()
    def predict(self, state):
        if self.continuous:
            p = torch.tensor([state["pos"] + 1], device=self.emb.weight.device)
            prev = 1 if state["mode"] == "no_prefix" else state["prev"]
            token = torch.tensor([[prev]], device=p.device)
            x = self.emb(token) + self.injection(state["h"], p, state["mode"])[:, None]
            hh, cache = self.fast(x, past=state["fast_cache"], cache=True)
            state["fast_cache"] = cache
            return self.head(hh[:, 0])[0]
        return self.head(state["h"])[0]

    @torch.no_grad()
    def accept(self, state, token, sentence=None):
        if self.continuous:
            state["prev"] = token
            state["pos"] += 1
            if sentence is not None:
                ids = torch.tensor([[*sentence]], device=self.emb.weight.device)
                hh, cache = self.slow(
                    self.feedback(ids)[:, None], past=state["slow_cache"], cache=True
                )
                state.update(
                    h=hh[:, -1], slow_cache=cache, fast_cache=None, prev=1, pos=0
                )
        else:
            ids = torch.tensor([[token]], device=self.emb.weight.device)
            hh, cache = self.fast(self.emb(ids), past=state["fast_cache"], cache=True)
            state.update(h=hh[:, -1], fast_cache=cache)


def batch(rows, device):
    B = len(rows)
    P = max(len(r["prompt_ids"]) + 1 for r in rows)
    S = max(len(r["targets"]) for r in rows)
    L = max(len(s) for r in rows for s in r["targets"])
    T = max(1 + len(r["prompt_ids"]) + sum(map(len, r["targets"])) - 1 for r in rows)
    p = torch.zeros(B, P, dtype=torch.long)
    y = torch.zeros(B, S, L, dtype=torch.long)
    d = torch.zeros(B, T, dtype=torch.long)
    ix = torch.zeros(B, S, L, dtype=torch.long)
    for j, r in enumerate(rows):
        pre = [1, *r["prompt_ids"]]
        p[j, : len(pre)] = torch.tensor(pre)
        seq = pre + [t for s in r["targets"] for t in s]
        d[j, : len(seq) - 1] = torch.tensor(seq[:-1])
        off = len(pre) - 1
        for k, s in enumerate(r["targets"]):
            y[j, k, : len(s)] = torch.tensor(s)
            ix[j, k, : len(s)] = torch.arange(off, off + len(s))
            off += len(s)
    return {
        k: v.to(device) for k, v in dict(prompt=p, y=y, discrete=d, indices=ix).items()
    }


def loss(logits, y):
    ce = F.cross_entropy(
        logits.flatten(0, -2), y.flatten(), ignore_index=0, reduction="none"
    ).reshape(y.shape)
    valid = y.ne(0)
    perrow = ce.sum((1, 2)) / valid.sum((1, 2)).clamp_min(1)
    return perrow.mean(), ce.sum(), valid.sum()
