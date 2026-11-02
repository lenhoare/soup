"""Per-epoch metrics and analysis helpers."""
from __future__ import annotations

import math
import zlib

import torch

from .genome import PAD
from .ribosome import TRAITS, Ribosome


def snapshot(world, tile: int = 16) -> dict:
    """Cheap summary of the world's current state."""
    counts = world.slot_counts()
    live_slots = (counts > 0).nonzero().flatten()
    cells = int(counts.sum())
    m: dict = {"tick": world.tick_count, "cells": cells, "organisms": live_slots.numel()}
    if cells == 0:
        return m
    p = counts[live_slots].float() / cells
    m["shannon"] = float(-(p * p.log()).sum())
    g = world.genomes[live_slots]
    m["distinct_genomes"] = int(torch.unique(g, dim=0).shape[0])
    L = world.glen[live_slots]
    m["mean_len"] = float(L.mean())
    m["median_len"] = float(L.median())
    raw = g.cpu().numpy().tobytes()
    m["zlib_ratio"] = len(zlib.compress(raw, 9)) / max(len(raw), 1)
    # cell-weighted trait means
    w = counts[live_slots].float()
    tr = (world.traits[live_slots] * w[:, None]).sum(0) / w.sum()
    m.update({f"mean_{name}": float(v) for name, v in zip(TRAITS, tr)})
    m["mean_energy"] = float(world.energy[world.gid >= 0].mean())
    m["max_depth"] = int(world.depth[live_slots].max())
    m["genomes_per_tile"] = genomes_per_tile(world.gid, tile)
    m.update(world.counters)
    return m


def genomes_per_tile(gid: torch.Tensor, tile: int = 16) -> float:
    """Average number of distinct genomes among occupied tiles (kernel locality health)."""
    H, W = gid.shape
    ty = torch.arange(H, device=gid.device).view(-1, 1) // tile
    tx = torch.arange(W, device=gid.device).view(1, -1) // tile
    tid = ty * ((W + tile - 1) // tile) + tx
    occ = gid >= 0
    if not occ.any():
        return 0.0
    key = tid[occ] * (int(gid.max()) + 1) + gid[occ]
    pairs = torch.unique(key)
    tiles = torch.unique(tid[occ]).numel()
    return pairs.numel() / tiles


@torch.no_grad()
def phenotype_smoothness(ribosome: Ribosome, w_base: torch.Tensor, alpha: float,
                         genomes: torch.Tensor, alphabet: int,
                         generator: torch.Generator) -> float:
    """Median relative change ||W' - W|| / ||W|| for random single-token
    substitution mutants (spec target ~5-15%). genomes: (n, L) on CPU."""
    mut = genomes.clone()
    for i in range(mut.shape[0]):
        pos = (mut[i] != PAD).nonzero().flatten()
        if pos.numel() == 0:
            continue
        j = int(pos[torch.randint(0, pos.numel(), (1,), generator=generator)])
        old = int(mut[i, j])
        new = old
        while new == old:
            new = int(torch.randint(1, alphabet, (1,), generator=generator))
        mut[i, j] = new
    dev = w_base.device
    w0 = w_base + alpha * ribosome.translate(genomes.to(dev))["delta"]
    w1 = w_base + alpha * ribosome.translate(mut.to(dev))["delta"]
    rel = (w1 - w0).norm(dim=1) / w0.norm(dim=1)
    return float(rel.median())


def format_line(m: dict) -> str:
    keys = ["tick", "cells", "organisms", "shannon", "mean_len", "mean_energy",
            "births", "spores", "attacks_won", "deaths", "genomes_per_tile"]
    parts = []
    for k in keys:
        if k in m:
            v = m[k]
            parts.append(f"{k}={v:.3g}" if isinstance(v, float) and not math.isnan(v) else f"{k}={v}")
    return "  ".join(parts)
