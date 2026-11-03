"""Competition in the evolution environment itself (Lenski-style): seed a full soup
50/50 with founders from set A and set B at shuffled positions, run with all
features on, and score the share of live cells descending from A's founders.

    A = deep evolved genomes,      B = random   -> is evolution adaptive?
    A = prescreened good growers,  B = random   -> positive control

    python -m experiments.soup_competition --over '{"heat": {"base_flip": 0.03}}'
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context

import torch


def evolve_deep(config: str, over: dict, ticks: int, grid: int, n: int) -> torch.Tensor:
    from soup.config import load_config
    from soup.world import World
    w = World(load_config(config, {**over, "grid": [grid, grid], "init": {"founders": 64}}))
    w.seed_founders()
    w.run(ticks)
    counts = w.slot_counts()
    key = torch.where(counts > 0, w.depth.float() + counts.float() / 1e6, torch.tensor(-1.0))
    top = key.argsort(descending=True)[:n]
    print(f"deep parents: depth mean {w.depth[top].float().mean():.1f}", flush=True)
    return w.genomes[top].cpu()


def prescreen(config: str, over: dict, n: int) -> torch.Tensor:
    from experiments.damage_cost import grow
    from soup import genome
    from soup.config import load_config
    cfg = load_config(config, over)
    pool = genome.random_genomes(1024, cfg.genome_len, cfg.alphabet, cfg.init.genome_len,
                                 torch.Generator().manual_seed(99),
                                 exclude=cfg.ribosome.stop_token)
    size = grow(dict(load_config(config, {**over, "grid": [512, 512]})), pool, 128, 32, 1)
    return pool[size.argsort(descending=True)[:n]]


def one(job) -> dict:
    label, a, b, seed, config, over, ticks, grid = job
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    from soup.config import load_config
    from soup.world import World
    w = World(load_config(config, {**over, "grid": [grid, grid], "seed": seed,
                                   "init": {"founders": len(a) + len(b)}}))
    g = torch.cat([a, b])
    perm = torch.randperm(len(g), generator=torch.Generator().manual_seed(seed))
    w.seed_founders(genomes=g[perm])        # random empty positions
    is_a = perm < len(a)                    # founder (root id) -> came from set A?
    curve = []
    for t in range(ticks):
        w.tick()
        if (t + 1) % 256 == 0:
            alive = w.gid >= 0
            roots = w.root[w.gid[alive]].cpu()
            curve.append(float(is_a[roots].float().mean()) if roots.numel() else float("nan"))
    print(f"{label} seed {seed}: share A {curve[-1]:.3f}", flush=True)
    return {"label": label, "seed": seed, "curve": curve}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--over", default="{}")
    ap.add_argument("--n", type=int, default=64, help="founders per side")
    ap.add_argument("--ticks", type=int, default=2048)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    over = json.loads(args.over)

    from soup import genome
    from soup.config import load_config
    cfg = load_config(args.config, over)
    rand = genome.random_genomes(args.n, cfg.genome_len, cfg.alphabet, cfg.init.genome_len,
                                 torch.Generator().manual_seed(7), exclude=cfg.ribosome.stop_token)
    deep = evolve_deep(args.config, over, 4096, 128, args.n)
    good = prescreen(args.config, over, args.n)
    rand2 = genome.random_genomes(args.n, cfg.genome_len, cfg.alphabet, cfg.init.genome_len,
                                  torch.Generator().manual_seed(8), exclude=cfg.ribosome.stop_token)
    sets = [("evolved vs random", deep, rand), ("prescreened vs random", good, rand),
            ("random vs random (null)", rand2, rand)]
    jobs = [(lab, a, b, int(s), args.config, over, args.ticks, args.grid)
            for lab, a, b in sets for s in args.seeds.split(",")]
    with get_context("spawn").Pool(args.procs) as pool:
        res = pool.map(one, jobs, chunksize=1)
    print(f"\nshare of live cells descending from set A after {args.ticks} ticks "
          f"(0.5 = no advantage)")
    for lab, _, _ in sets:
        v = torch.tensor([r["curve"][-1] for r in res if r["label"] == lab])
        print(f"  {lab:26s} {v.mean():.3f} ± {v.std() / len(v) ** .5:.3f}   "
              f"per seed {' '.join(f'{x:.2f}' for x in v.tolist())}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
