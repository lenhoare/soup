"""Is heat damage harmful? Fitness cost of k random token flips.

1. Evolve a population (all features on) and take the most abundant genomes.
2. Make k-flip mutants with the heat operator (non-pad token -> another symbol).
3. Grow every parent and mutant from a single founder in isolation (viability
   lattice) and compare colony sizes, paired. Random genomes are a reference.

    python -m experiments.damage_cost --evolve-ticks 4096 --top 128
"""
from __future__ import annotations

import argparse
import json

import torch

from experiments.viability import lattice
from soup import genome
from soup.config import load_config
from soup.world import World


def flip(g: torch.Tensor, k: int, alphabet: int, gen: torch.Generator) -> torch.Tensor:
    out = g.clone()
    for i in range(out.shape[0]):
        nz = (out[i] != genome.PAD).nonzero().flatten()
        if nz.numel() == 0:
            continue
        pos = nz[torch.randperm(nz.numel(), generator=gen)[:k]]
        off = torch.randint(1, alphabet, (pos.numel(),), generator=gen, dtype=torch.uint8)
        out[i, pos] = (out[i, pos] + off) % alphabet
    return out


def grow(cfg, genomes: torch.Tensor, ticks: int, spacing: int, reps: int) -> torch.Tensor:
    """Mean log colony size per genome over `reps` world seeds."""
    h, w = cfg["grid"]
    pos = lattice(h, w, spacing)
    out = torch.zeros(genomes.shape[0])
    for rep in range(reps):
        for s in range(0, genomes.shape[0], pos.numel()):
            chunk = genomes[s:s + pos.numel()]
            # shuffle placement each rep so position effects average out
            perm = torch.randperm(chunk.shape[0], generator=torch.Generator().manual_seed(rep))
            world = World(load_config(None, {**cfg, "seed": 100 + rep}))
            world.seed_founders(genomes=chunk[perm], positions=pos[: chunk.shape[0]])
            world.run(ticks)
            alive = world.gid >= 0
            size = torch.bincount(world.root[world.gid[alive]],
                                  minlength=world.next_root)[: chunk.shape[0]].float()
            res = torch.empty_like(size)
            res[perm] = size
            out[s:s + chunk.shape[0]] += (res + 1).log()
    return out / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--evolve-ticks", type=int, default=4096)
    ap.add_argument("--evolve-grid", type=int, default=128)
    ap.add_argument("--top", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=128)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--flips", default="1,3")
    ap.add_argument("--over", default="{}", help="JSON config overrides (evolve and grow)")
    args = ap.parse_args()
    over = json.loads(args.over)
    torch.set_grad_enabled(False)

    ecfg = load_config(args.config, {**over, "grid": [args.evolve_grid] * 2,
                                     "init": {"founders": 64}})
    w = World(ecfg)
    w.seed_founders()
    w.run(args.evolve_ticks)
    counts = w.slot_counts()
    top = counts.argsort(descending=True)[: args.top]
    top = top[counts[top] > 0]
    evolved = w.genomes[top].cpu()
    if ecfg.ribosome.stop_token is not None:
        ex = w.expressed[top].float()
        print(f"expressed length of evolved genomes: mean {ex.mean():.1f} "
              f"(non-pad {w.glen[top].float().mean():.1f})")
    print(f"evolved {args.evolve_ticks} ticks: took {evolved.shape[0]} most abundant genomes "
          f"(cells {int(counts[top].min())}-{int(counts[top].max())})")

    gcfg = dict(load_config(args.config, {**over, "grid": [512, 512]}))
    gen = torch.Generator().manual_seed(0)
    rand = genome.random_genomes(evolved.shape[0], ecfg.genome_len, ecfg.alphabet,
                                 ecfg.init.genome_len, gen, exclude=ecfg.ribosome.stop_token)
    print("set       k  mean log size  paired diff vs parent (mean ± se)  frac worse")
    for name, base in (("evolved", evolved), ("random", rand)):
        b = grow(gcfg, base, args.ticks, 32, args.reps)
        print(f"{name:8s}  0  {b.mean():.3f}")
        for k in (int(x) for x in args.flips.split(",")):
            m = grow(gcfg, flip(base, k, ecfg.alphabet, gen), args.ticks, 32, args.reps)
            d = m - b
            print(f"{name:8s}  {k}  {m.mean():.3f}          {d.mean():+.3f} ± {d.std() / len(d) ** .5:.3f}"
                  f"                     {(d < 0).float().mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
