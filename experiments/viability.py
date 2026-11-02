"""Viability scan (M2): what fraction of random genome strings produce a living,
growing organism?

Founders with random genomes are placed on a lattice (spacing >> growth over the
run, so colonies barely interact) and run for T ticks. A genome is *viable* if its
lineage (founder + spore descendants) has at least `min_cells` cells at the end.
Also reports phenotype smoothness for the same ribosome.

    python -m experiments.viability --alpha 0.25,0.5,1,2
"""
from __future__ import annotations

import argparse
import json

import torch

from soup import genome
from soup.config import load_config
from soup.metrics import phenotype_smoothness
from soup.world import World


def lattice(h: int, w: int, spacing: int) -> torch.Tensor:
    ys = torch.arange(spacing // 2, h, spacing)
    xs = torch.arange(spacing // 2, w, spacing)
    return (ys.view(-1, 1) * w + xs.view(1, -1)).flatten()


def scan(cfg, n_genomes: int, ticks: int, spacing: int, min_cells: int,
         gen: torch.Generator) -> dict:
    h, w = cfg.grid
    pos = lattice(h, w, spacing)
    sizes = []
    lens = []
    done = 0
    while done < n_genomes:
        world = World(cfg)
        k = min(pos.numel(), n_genomes - done)
        g = genome.random_genomes(k, cfg.genome_len, cfg.alphabet, cfg.init.genome_len, gen)
        world.seed_founders(genomes=g, positions=pos[:k])
        world.run(ticks)
        alive = world.gid >= 0
        roots = world.root[world.gid[alive]]
        sizes.append(torch.bincount(roots, minlength=world.next_root)[:k].cpu())
        lens.append(genome.effective_length(g))
        done += k
    sizes = torch.cat(sizes)
    viable = sizes >= min_cells
    return {
        "viable_frac": float(viable.float().mean()),
        "extinct_frac": float((sizes == 0).float().mean()),
        "median_size": float(sizes.float().median()),
        "p90_size": float(sizes.float().quantile(0.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--alpha", default="0.5")
    ap.add_argument("--genomes", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=128)
    ap.add_argument("--spacing", type=int, default=32)
    ap.add_argument("--min-cells", type=int, default=16)
    ap.add_argument("--json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    print("alpha  viable  extinct  median_size  p90_size  smoothness")
    for a in [float(x) for x in args.alpha.split(",")]:
        cfg = load_config(args.config, {"ribosome": {"alpha": a}, "grid": [512, 512]})
        gen = torch.Generator().manual_seed(0)
        r = scan(cfg, args.genomes, args.ticks, args.spacing, args.min_cells, gen)
        w = World(load_config(args.config, {"ribosome": {"alpha": a}, "grid": [8, 8]}))
        g = genome.random_genomes(512, cfg.genome_len, cfg.alphabet, cfg.init.genome_len, gen)
        r["smoothness"] = phenotype_smoothness(w.ribosome, w.w_base, a, g, cfg.alphabet, gen)
        r["alpha"] = a
        print(f"{a:<5}  {r['viable_frac']:.2f}    {r['extinct_frac']:.2f}     "
              f"{r['median_size']:<11.0f}  {r['p90_size']:<8.0f}  {r['smoothness']:.3f}",
              flush=True)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
