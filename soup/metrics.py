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
    m: dict = {"tick": world.tick_count, "cells": cells, "organisms": live_slots.numel(),
               "occupancy": cells / world.gid.numel()}
    if cells == 0:
        return m
    m["lineages"] = int(torch.unique(world.root[live_slots]).numel())
    m["mean_depth"] = float(world.depth[live_slots].float().mean())
    p = counts[live_slots].float() / cells
    m["shannon"] = float(-(p * p.log()).sum())
    g = world.genomes[live_slots]
    m["distinct_genomes"] = int(torch.unique(g, dim=0).shape[0])
    L = world.glen[live_slots]
    m["mean_len"] = float(L.mean())
    m["mean_expressed"] = float(world.expressed[live_slots].mean())
    m["median_len"] = float(L.median())
    raw = g.cpu().numpy().tobytes()
    m["zlib_ratio"] = len(zlib.compress(raw, 9)) / max(len(raw), 1)
    # cell-weighted trait means
    w = counts[live_slots].float()
    tr = (world.traits[live_slots] * w[:, None]).sum(0) / w.sum()
    m.update({f"mean_{name}": float(v) for name, v in zip(TRAITS, tr)})
    m["mean_energy"] = float(world.energy[world.gid >= 0].mean())
    m["hgt_frac"] = float(world.has_hgt[live_slots].float().mean())
    m["mean_org_size"] = cells / live_slots.numel()
    # do hot-region organisms carry higher fidelity? (cell-level correlation)
    alive = world.gid >= 0
    fid = world.traits[world.gid[alive], 0]
    heat = world.heat[alive]
    if cells > 2 and fid.std() > 0 and heat.std() > 0:
        m["fid_heat_corr"] = float(torch.corrcoef(torch.stack([fid, heat]))[0, 1])
    # robustness routes, hot vs cold quartile of occupied cells
    rep = repeat_frac(world.genomes)
    m["repeat_frac"] = float((rep[live_slots] * w).sum() / w.sum())
    if cells >= 8:
        lo, hi = torch.quantile(heat, torch.tensor([0.25, 0.75], device=heat.device))
        hot, cold = heat >= hi, heat <= lo
        if hot.any() and cold.any():
            g = world.gid[alive]
            m["fid_hot_minus_cold"] = float(fid[hot].mean() - fid[cold].mean())
            m["repeat_hot_minus_cold"] = float(rep[g[hot]].mean() - rep[g[cold]].mean())
            m["heat_hot_minus_cold"] = float(heat[hot].mean() - heat[cold].mean())
    m["max_depth"] = int(world.depth[live_slots].max())
    m["genomes_per_tile"] = genomes_per_tile(world.gid, tile)
    m.update(world.counters)
    return m


def repeat_frac(genomes: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Per genome: fraction of its pad-free k-mers that occur more than once in the same
    genome (a simple redundancy measure: duplicated segments raise it)."""
    g = genomes.long()
    n, L = g.shape
    win = g.unfold(1, k, 1)                                   # (n, L-k+1, k)
    valid = (win != PAD).all(-1)
    code = (win * (16 ** torch.arange(k, device=g.device))).sum(-1)
    code = torch.where(valid, code, -1 - torch.arange(L - k + 1, device=g.device))  # unique
    s, _ = code.sort(dim=1)
    eq = s[:, 1:] == s[:, :-1]
    dup = torch.zeros_like(s, dtype=torch.bool)
    dup[:, 1:] |= eq
    dup[:, :-1] |= eq
    dup &= s >= 0
    return dup.sum(1).float() / valid.sum(1).clamp(min=1).float()


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
