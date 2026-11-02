"""The ribosome: a small frozen transformer that translates genome strings into
cell-net weight deltas, traits and a colouring embedding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .cellnet import CellNetLayout
from .genome import PAD

TRAITS = ("fidelity", "permeability", "aggression", "spore_threshold")
FIDELITY, PERMEABILITY, AGGRESSION, SPORE_THRESHOLD = range(4)
EMBED_DIM = 8


class Ribosome(nn.Module):
    def __init__(self, alphabet: int, genome_len: int, layout: CellNetLayout,
                 d_model: int = 64, layers: int = 2, heads: int = 4, mlp_ratio: int = 2,
                 delta_scale: float = 1.0, seed: int = 1234, frozen: bool = True):
        super().__init__()
        self.layout = layout
        # Seeded init without disturbing the global RNG: the "genetic code" is a
        # pure function of `seed`.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.tok = nn.Embedding(alphabet, d_model)
            self.pos = nn.Parameter(torch.randn(genome_len, d_model) * 0.5)
            enc = nn.TransformerEncoderLayer(d_model, heads, d_model * mlp_ratio,
                                             dropout=0.0, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
            self.delta_head = nn.Linear(d_model, layout.n_params)
            self.trait_head = nn.Linear(d_model, len(TRAITS))
            self.embed_head = nn.Linear(d_model, EMBED_DIM)
            for head, std in ((self.delta_head, 1.0), (self.trait_head, 0.7),
                              (self.embed_head, 1.0)):
                nn.init.normal_(head.weight, std=std / math.sqrt(d_model))
                nn.init.zeros_(head.bias)
        self.register_buffer("scale", layout.param_scale() * delta_scale)
        if frozen:
            self.requires_grad_(False)
        self.eval()

    def forward(self, genomes: torch.Tensor) -> dict[str, torch.Tensor]:
        """genomes: (n, genome_len) uint8 -> dict of delta (n, P), traits (n, 4) in
        [0, 1], embed (n, 8)."""
        g = genomes.long()
        pad = g == PAD
        all_pad = pad.all(-1, keepdim=True)
        pad = pad & ~all_pad              # an all-pad genome attends over its pads
        x = self.tok(g) + self.pos[: g.shape[1]]
        x = self.encoder(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).float()
        z = self.norm((x * keep).sum(1) / keep.sum(1))
        return {
            "delta": self.delta_head(z) * self.scale,
            "traits": torch.sigmoid(self.trait_head(z)),
            "embed": self.embed_head(z),
        }

    @torch.no_grad()
    def translate(self, genomes: torch.Tensor, batch: int = 4096) -> dict[str, torch.Tensor]:
        outs = [self(genomes[s:s + batch]) for s in range(0, genomes.shape[0], batch)]
        return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
