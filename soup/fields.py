"""Slowly drifting environment fields (light, heat) on the torus.

Each field is a sum of a few low-frequency Fourier modes with integer wave
numbers (so it tiles the torus seamlessly) and phases that drift over time.
"""
from __future__ import annotations

import math

import torch


class DriftingField:
    def __init__(self, h: int, w: int, modes: int, max_k: int, drift: float,
                 generator: torch.Generator, device: torch.device,
                 contrast: float | None = None):
        ks = []
        while len(ks) < modes:
            kx, ky = (int(v) for v in torch.randint(-max_k, max_k + 1, (2,), generator=generator))
            if (kx, ky) != (0, 0):
                ks.append((kx, ky))
        k = torch.tensor(ks, dtype=torch.float32)
        ys = torch.arange(h, dtype=torch.float32).view(1, h, 1)
        xs = torch.arange(w, dtype=torch.float32).view(1, 1, w)
        self.angle = (2 * math.pi * (k[:, 0].view(-1, 1, 1) * xs / w
                                     + k[:, 1].view(-1, 1, 1) * ys / h)).to(device)
        self.amp = (0.5 + torch.rand(modes, generator=generator)).to(device)
        self.phase0 = (2 * math.pi * torch.rand(modes, generator=generator)).to(device)
        self.vel = (torch.randn(modes, generator=generator) * drift * 2 * math.pi).to(device)
        self.contrast = contrast

    def at(self, t: float) -> torch.Tensor:
        """(h, w) field in [0, 1] at time t (ticks)."""
        ph = (self.phase0 + self.vel * t).view(-1, 1, 1)
        f = (self.amp.view(-1, 1, 1) * torch.cos(self.angle + ph)).sum(0) / self.amp.sum()
        f = 0.5 * (f + 1.0)
        if self.contrast:
            f = torch.sigmoid(self.contrast * (f - 0.5))
        return f
