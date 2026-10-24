"""Earlier functional sentence+token rotations, using the logical-task backbone.
Readout/selector and uncentered feedback reproduce FullRotary/RotarySelector.
No token decoder, prefix correction, oracle option count, or sentence lookup.
"""

# Allow direct execution as well as python -m.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

# Source lives in the code repository; artifacts have a separate root.
import sys as _sys
from pathlib import Path as _Path

from src.common.experiment_paths import artifact_dir
from pathlib import Path
import sys, math
import torch
from torch import nn
from torch.nn import functional as F

ROOT = artifact_dir(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import Config, Stack, Model


class SentenceTokenRotation(nn.Module):
    def __init__(self, vocab=180, width=512, layers=4, options=16, max_tokens=8):
        super().__init__()
        self.options = options
        self.max_tokens = max_tokens
        self.config = Config(
            kind="sentence_token_rotation",
            vocab=vocab,
            width=width,
            slow_layers=layers,
            token_layers=0,
        )
        self.emb = nn.Embedding(vocab, width, padding_idx=0)
        self.slow = Stack(self.config, layers)
        self.apply(Model.initialize)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        gen = torch.Generator().manual_seed(torch.initial_seed())
        self.alpha = nn.Parameter(
            torch.empty(width // 2).uniform_(-math.pi, math.pi, generator=gen)
        )
        gen = torch.Generator().manual_seed(20260913)
        self.register_buffer("frequencies", torch.rand(8, generator=gen) * math.pi)
        self.sentence_rates = nn.Parameter(torch.randn(16, width // 2) / 4)
        gen = torch.Generator().manual_seed(torch.initial_seed() + 314159)
        self.selection_vector = nn.Parameter(torch.randn(width, generator=gen) * 0.02)

    def features(self, s):
        p = s.to(self.frequencies)[:, None] * self.frequencies
        return torch.cat([p.cos(), p.sin()], -1)

    @staticmethod
    def rotation(x, a):
        c, s = a.cos(), a.sin()
        e, o = x[..., ::2], x[..., 1::2]
        return torch.stack([c * e - s * o, s * e + c * o], -1).flatten(-2)

    def scores(self, h):
        a = (
            self.features(torch.arange(self.options, device=h.device))
            @ self.sentence_rates
        )
        return h @ self.rotation(self.selection_vector, a).T

    def logits(self, h, slots=None, length=None):
        length = length or self.max_tokens
        p = torch.arange(1, length + 1, device=h.device)[:, None] * self.alpha
        if slots is None:
            a = (
                self.features(torch.arange(self.options, device=h.device))
                @ self.sentence_rates
            )
            u = self.rotation(h[:, None, None, :], -(a[:, None, :] + p)[None])
        else:
            a = self.features(slots) @ self.sentence_rates
            u = self.rotation(h[:, None, :], -(a[:, None, :] + p))
        z = F.linear(u, self.emb.weight)
        z[..., 0] = -1e4
        z[..., 1] = -1e4
        z[..., 4] = -1e4
        return z

    def feedback(self, y):
        p = torch.arange(1, y.shape[-1] + 1, device=y.device)[:, None] * self.alpha
        mask = y.ne(0) & (y.eq(2).cumsum(-1) - y.eq(2).long()).eq(0)
        e = self.rotation(self.emb(y), p) * mask[..., None]
        return (
            F.normalize(e.sum(-2), dim=-1, eps=1e-8)
            * self.emb.weight[5:].norm(dim=-1).mean()
        )

    def contexts(self, b):
        p, y = b["prompt"], b["y"]
        B, S, L = y.shape
        P = p.shape[1]
        f = self.feedback(y[:, :-1])
        x = torch.cat([self.emb(p), f], 1)
        mask = torch.cat([p.ne(0), y[:, :-1].ne(0).any(-1)], 1)
        pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
        h, _ = self.slow(x, keymask=mask, positions=pos)
        ix = torch.cat(
            [
                (p.ne(0).sum(-1) - 1)[:, None],
                torch.arange(P, P + S - 1, device=p.device)[None].expand(B, -1),
            ],
            1,
        )
        return h.gather(1, ix[..., None].expand(-1, -1, h.shape[-1]))

    def start(self, prompt):
        p = torch.tensor([[1, *prompt]], device=self.emb.weight.device)
        h, cache = self.slow(self.emb(p), cache=True)
        return h[:, -1], cache

    def accept(self, y, cache):
        y = torch.tensor([y], device=self.emb.weight.device)
        h, cache = self.slow(self.feedback(y)[:, None], past=cache, cache=True)
        return h[:, -1], cache

    def components(self, h, y):
        lp = self.logits(h, length=y.shape[-1]).log_softmax(-1)
        return -(
            lp.gather(-1, y[:, None, :, None].expand(-1, self.options, -1, 1)).squeeze(
                -1
            )
            * y.ne(0)[:, None]
        ).sum(-1)
