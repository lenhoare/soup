"""Mutational robustness ("survival of the flattest", docs/tests.md #1), phenotype level.

How much do k heat-style flips change a genome's translated phenotype? Compared across:
  evolved          deep lineages from a 4096-tick soup (cached in runs/competition)
  random           32-token random founders
  layout-matched   random tokens at the evolved genomes' exact non-pad positions
  length-matched   random genomes with the evolved genomes' lengths, compact layout
  prescreened      best isolated growers of 1024 random genomes (selected, not evolved)

If evolved genomes change less per flip than the matched controls, robustness
evolved. The fitness-level counterpart is soup_competition --mode damage with
--parents deep vs --parents prescreened.

    python -m experiments.robustness --flips 1,3,10,20
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from experiments.competition import flip, regions
from experiments.soup_competition import (evolve_deep, layout_matched, length_matched,
                                          prescreen)
from soup import genome
from soup.config import load_config
from soup.world import World


def cached(path: str, make):
    if os.path.exists(path):
        return torch.load(path)
    x = make()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(x, path)
    return x


@torch.no_grad()
def sensitivity(world: World, g: torch.Tensor, k: int, draws: int, gen: torch.Generator) -> dict:
    """Mean over genomes and draws of the relative weight change and trait change."""
    r = world.ribosome
    alpha = world.cfg.ribosome.alpha
    base = r.translate(g.to(world.device))
    w0 = world.w_base + alpha * base["delta"]
    where = regions(g, world.cfg.ribosome.stop_token)["any"]
    rel, dtr = [], []
    for _ in range(draws):
        m, _ = flip(g, k, where, world.cfg.alphabet, gen, cap=True)
        out = r.translate(m.to(world.device))
        w1 = world.w_base + alpha * out["delta"]
        rel.append(((w1 - w0).norm(dim=1) / w0.norm(dim=1)).cpu())
        dtr.append((out["traits"] - base["traits"]).abs().mean(1).cpu())
    rel, dtr = torch.stack(rel).mean(0), torch.stack(dtr).mean(0)   # per genome
    return {"rel_w": float(rel.mean()), "rel_w_se": float(rel.std() / len(rel) ** .5),
            "d_traits": float(dtr.mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--over", default='{"heat": {"base_flip": 0.03}, "fields": {"heat_drift": 0.0}}')
    ap.add_argument("--flips", default="1,3,10,20")
    ap.add_argument("--draws", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--cache", default="runs/competition")
    ap.add_argument("--json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    over = json.loads(args.over)
    cfg = load_config(args.config, over)
    stop = cfg.ribosome.stop_token
    gen = torch.Generator().manual_seed(3)

    evolved = cached(f"{args.cache}/deep_parents.pt",
                     lambda: evolve_deep(args.config, over, 4096, 128, args.n))
    good = cached(f"{args.cache}/prescreened_parents.pt",
                  lambda: prescreen(args.config, over, args.n))
    groups = {
        "evolved": evolved,
        "random": genome.random_genomes(len(evolved), cfg.genome_len, cfg.alphabet,
                                        cfg.init.genome_len, gen, exclude=stop),
        "layout-matched": layout_matched(evolved, cfg.alphabet, gen, stop),
        "length-matched": length_matched(evolved, cfg.alphabet, gen, stop),
        "prescreened": good,
    }
    world = World(load_config(args.config, {**over, "grid": [8, 8]}))
    ks = [int(x) for x in args.flips.split(",")]
    res = {}
    print("relative weight change ||W'-W||/||W|| per k flips (mean ± se over genomes)")
    print(f"{'group':16s} {'len':>5s}  " + "  ".join(f"{f'k={k}':>15s}" for k in ks))
    for name, g in groups.items():
        L = float((g != 0).sum(1).float().mean())
        row = {k: sensitivity(world, g, k, args.draws, gen) for k in ks}
        res[name] = {"length": L, **{str(k): v for k, v in row.items()}}
        print(f"{name:16s} {L:5.1f}  " + "  ".join(
            f"{row[k]['rel_w']:.3f} ± {row[k]['rel_w_se']:.3f}" for k in ks), flush=True)
    print("\nmean |trait change| per k flips")
    for name in groups:
        print(f"{name:16s}        " + "  ".join(f"{res[name][str(k)]['d_traits']:15.3f}" for k in ks))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
