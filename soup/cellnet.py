"""The per-cell network: layout of its flat weight vector, the hand-set base
weights, and a batched forward pass where every cell uses its own genome's weights.

Inputs  (n_in = 4*(S+M) + 4): perception of S state + M morphogen channels
        through 4 filters (identity, sobel-x, sobel-y, laplacian), then
        energy, light, kin fraction, heat.
Outputs (n_out = S + M + 8): state delta, morphogen emission,
        4 action logits (grow, spore, attack, rest), 4 direction logits (N, E, S, W).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

GROW, SPORE, ATTACK, REST = range(4)
N_FILTERS = 4
N_EXTRA = 4


@dataclass(frozen=True)
class CellNetLayout:
    state_ch: int = 8
    morph_ch: int = 4
    hidden: int = 16

    @property
    def n_percept(self) -> int:
        return N_FILTERS * (self.state_ch + self.morph_ch)

    @property
    def n_in(self) -> int:
        return self.n_percept + N_EXTRA

    @property
    def n_out(self) -> int:
        return self.state_ch + self.morph_ch + 8

    # input indices of the extras
    @property
    def in_energy(self) -> int: return self.n_percept
    @property
    def in_light(self) -> int: return self.n_percept + 1
    @property
    def in_kin(self) -> int: return self.n_percept + 2
    @property
    def in_heat(self) -> int: return self.n_percept + 3

    # output slices
    @property
    def out_state(self) -> slice: return slice(0, self.state_ch)
    @property
    def out_morph(self) -> slice: return slice(self.state_ch, self.state_ch + self.morph_ch)
    @property
    def out_act(self) -> slice:
        o = self.state_ch + self.morph_ch
        return slice(o, o + 4)
    @property
    def out_dir(self) -> slice:
        o = self.state_ch + self.morph_ch + 4
        return slice(o, o + 4)

    @property
    def sizes(self) -> tuple[int, int, int, int]:
        return (self.n_in * self.hidden, self.hidden, self.hidden * self.n_out, self.n_out)

    @property
    def n_params(self) -> int:
        return sum(self.sizes)

    def unpack(self, flat: torch.Tensor):
        """(N, P) -> W1 (N, in, h), b1 (N, h), W2 (N, h, out), b2 (N, out)."""
        n = flat.shape[0]
        a, b, c, _ = self.sizes
        w1 = flat[:, :a].view(n, self.n_in, self.hidden)
        b1 = flat[:, a:a + b]
        w2 = flat[:, a + b:a + b + c].view(n, self.hidden, self.n_out)
        b2 = flat[:, a + b + c:]
        return w1, b1, w2, b2

    def pack(self, w1, b1, w2, b2) -> torch.Tensor:
        n = w1.shape[0]
        return torch.cat([w1.reshape(n, -1), b1, w2.reshape(n, -1), b2], dim=1)

    def param_scale(self) -> torch.Tensor:
        """Per-parameter scale for the ribosome's dW so each block has a sensible
        magnitude (fan-in scaled weights, smaller biases)."""
        a, b, c, d = self.sizes
        return torch.cat([
            torch.full((a,), 1.0 / math.sqrt(self.n_in)),
            torch.full((b,), 0.3),
            torch.full((c,), 1.0 / math.sqrt(self.hidden)),
            torch.full((d,), 0.3),
        ])


def base_weights(layout: CellNetLayout, seed: int = 0) -> torch.Tensor:
    """A hand-set "reasonable CA": grow when energy is high, otherwise mostly rest;
    spore occasionally when rich; rarely attack. Small random weights elsewhere
    give the hidden state some texture. Returns a flat (P,) vector."""
    gen = torch.Generator().manual_seed(seed)
    h, ni, no = layout.hidden, layout.n_in, layout.n_out
    w1 = torch.randn(ni, h, generator=gen) * (0.3 / math.sqrt(ni))
    b1 = torch.zeros(h)
    w2 = torch.randn(h, no, generator=gen) * (0.3 / math.sqrt(h))
    b2 = torch.zeros(no)

    # hidden 0: energy detector, tanh(4E - 2) > 0 when E > 0.5
    w1[:, 0] = 0.0
    w1[layout.in_energy, 0] = 4.0
    b1[0] = -2.0
    # hidden 1: kin crowding detector, > 0 when more than half the neighbours are kin
    w1[:, 1] = 0.0
    w1[layout.in_kin, 1] = 4.0
    b1[1] = -2.0

    act = layout.out_act.start
    w2[:, act:act + 4] = 0.0
    w2[0, act + GROW] = 2.5
    b2[act + GROW] = 0.5
    w2[0, act + SPORE] = 1.5
    w2[1, act + SPORE] = 1.0      # spore more when buried inside own body
    b2[act + SPORE] = -1.5
    b2[act + ATTACK] = -2.5
    b2[act + REST] = 0.0

    b2[layout.out_morph] = -1.0   # modest baseline emission
    return layout.pack(w1[None], b1[None], w2[None], b2[None])[0]


def forward(layout: CellNetLayout, x: torch.Tensor, weights: torch.Tensor,
            gid: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
    """x: (N, n_in) inputs; weights: (S, P) per-slot cache; gid: (N,) slot per cell.

    Gathers each cell's weights and runs a batched matmul, in chunks to bound
    memory. This is the simple eager path; the tiled shared-memory kernel (M6)
    replaces it at Tier M/L.
    """
    out = torch.empty(x.shape[0], layout.n_out, device=x.device, dtype=torch.float32)
    for s in range(0, x.shape[0], chunk):
        w = weights[gid[s:s + chunk]]
        w1, b1, w2, b2 = layout.unpack(w)
        xi = x[s:s + chunk].to(w.dtype).unsqueeze(1)
        hdn = torch.tanh(torch.bmm(xi, w1).squeeze(1) + b1)
        out[s:s + chunk] = (torch.bmm(hdn.unsqueeze(1), w2).squeeze(1) + b2).float()
    return out
