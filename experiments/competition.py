"""Head-to-head competition assay: what does a heat flip really cost?

1. Evolve a population (config + --over) and take the most abundant genomes.
2. Make mutants (k heat-style flips; for STOP genomes optionally restricted to the
   expressed region, the first STOP itself, or the untranslated junk).
3. Put parent and mutant founders side by side in walled 31x31 arenas (both
   orientations, to cancel position/light luck) and run without heat or gene
   transfer. Score = mutant's share of the arena's live cells (0.5 = neutral).
4. A parent-vs-identical-copy null gives the noise level.

    python -m experiments.competition --over '{"heat": {"base_flip": 0.03}}' --label base
"""
from __future__ import annotations

import argparse
import json

import torch

from soup import genome
from soup.config import load_config
from soup.world import World

ARENA = 32


def arena_layout(h: int, w: int, size: int = ARENA):
    ys = torch.arange(h).view(-1, 1)
    xs = torch.arange(w).view(1, -1)
    barrier = (ys % size == 0) | (xs % size == 0)
    origins = [(ay * size, ax * size) for ay in range(h // size) for ax in range(w // size)]
    return barrier, origins


def regions(g: torch.Tensor, stop: int | None) -> dict[str, torch.Tensor]:
    """Boolean masks (n, L) of candidate flip positions per region."""
    nonpad = g != genome.PAD
    if stop is None:
        return {"any": nonpad}
    is_stop = g == stop
    before = torch.cumsum(is_stop.int(), 1) == 0
    first = is_stop & (torch.cumsum(is_stop.int(), 1) == 1)
    return {"any": nonpad, "expressed": nonpad & before, "stop": first,
            "junk": nonpad & ~before & ~first}


def flip(g: torch.Tensor, k: int, where: torch.Tensor, alphabet: int,
         gen: torch.Generator, avoid: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """k heat-style flips per genome at positions allowed by `where`. Returns the
    mutants and a mask of genomes that had enough positions."""
    out = g.clone()
    ok = where.sum(1) >= k
    for i in ok.nonzero().flatten().tolist():
        cand = where[i].nonzero().flatten()
        pos = cand[torch.randperm(cand.numel(), generator=gen)[:k]]
        for p in pos.tolist():
            old = int(out[i, p])
            choices = [t for t in range(alphabet) if t != old and t != avoid]
            out[i, p] = choices[int(torch.randint(0, len(choices), (1,), generator=gen))]
    return out, ok


def compete(cfg_over: dict, config: str, parents: torch.Tensor, mutants: torch.Tensor,
            ticks: int, grid: int, seed: int) -> torch.Tensor:
    """Mutant share per pair (averaged over both orientations); NaN if both died.
    Pairs beyond one world's arena count run in further worlds."""
    per_world = len(arena_layout(grid, grid)[1]) // 2
    return torch.cat([_compete(cfg_over, config, parents[s:s + per_world],
                               mutants[s:s + per_world], ticks, grid, seed + 1000 * s)
                      for s in range(0, parents.shape[0], per_world)])


def _compete(cfg_over: dict, config: str, parents: torch.Tensor, mutants: torch.Tensor,
             ticks: int, grid: int, seed: int) -> torch.Tensor:
    cfg = load_config(config, {**cfg_over, "grid": [grid, grid], "seed": seed,
                               "features": {"heat": False, "hgt": False, "cost": True}})
    barrier, origins = arena_layout(grid, grid)
    n = parents.shape[0]
    w = World(cfg)
    w.barrier = barrier.to(w.device)
    gs, pos = [], []
    for i in range(n):
        for j, (first, second) in enumerate(((parents[i], mutants[i]),
                                             (mutants[i], parents[i]))):
            oy, ox = origins[2 * i + j]
            gs += [first, second]
            pos += [(oy + 8) * grid + ox + 8, (oy + 24) * grid + ox + 24]
    w.seed_founders(genomes=torch.stack(gs), positions=torch.tensor(pos))
    w.run(ticks)
    alive = w.gid >= 0
    cells = torch.bincount(w.root[w.gid[alive]], minlength=w.next_root).float().cpu()
    cells = cells[: 4 * n].view(n, 2, 2)                 # pair, orientation, founder
    mut = torch.stack([cells[:, 0, 1], cells[:, 1, 0]], 1)
    par = torch.stack([cells[:, 0, 0], cells[:, 1, 1]], 1)
    tot = mut + par
    share = torch.where(tot > 0, mut / tot.clamp(min=1), torch.full_like(tot, float("nan")))
    return share.nanmean(1)


def summarize(name: str, s: torch.Tensor) -> str:
    s = s[~s.isnan()]
    se = s.std() / len(s) ** 0.5
    return (f"{name:22s} n={len(s):3d}  mutant share {s.mean():.3f} ± {se:.3f}  "
            f"(Δ {s.mean() - 0.5:+.3f})  mutant loses in {(s < 0.5).float().mean():.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--over", default="{}", help="JSON config overrides (evolve + compete)")
    ap.add_argument("--label", default="")
    ap.add_argument("--evolve-ticks", type=int, default=4096)
    ap.add_argument("--evolve-grid", type=int, default=128)
    ap.add_argument("--top", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=384)
    ap.add_argument("--grid", type=int, default=512)
    ap.add_argument("--parents", default="abundant", choices=["abundant", "deep", "prescreened"],
                    help="abundant: most cells after evolving; deep: most generations "
                         "of descent; prescreened: best isolated growers of 1024 random "
                         "genomes (a positive control with a known advantage)")
    ap.add_argument("--json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    over = json.loads(args.over)

    ecfg = load_config(args.config, {**over, "grid": [args.evolve_grid] * 2,
                                     "init": {"founders": 64}})
    stop = ecfg.ribosome.stop_token
    gen = torch.Generator().manual_seed(0)
    if args.parents == "prescreened":
        from experiments.damage_cost import grow
        pool = genome.random_genomes(1024, ecfg.genome_len, ecfg.alphabet,
                                     ecfg.init.genome_len, torch.Generator().manual_seed(99),
                                     exclude=stop)
        size = grow(dict(load_config(args.config, {**over, "grid": [512, 512]})), pool,
                    128, 32, 1)
        best = size.argsort(descending=True)[: args.top]
        parents = pool[best]
        print(f"[{args.label}] prescreened: top {len(best)} of 1024 random genomes by "
              f"isolated growth (log size {size[best].mean():.2f} vs pool {size.mean():.2f})",
              flush=True)
    else:
        w = World(ecfg)
        w.seed_founders()
        w.run(args.evolve_ticks)
        counts = w.slot_counts()
        key = counts.float() if args.parents == "abundant" else \
            torch.where(counts > 0, w.depth.float() + counts.float() / 1e6, torch.tensor(-1.0))
        top = key.argsort(descending=True)[: args.top]
        top = top[counts[top] > 0]
        parents = w.genomes[top].cpu()
        d = w.depth[top].float()
        print(f"[{args.label}] evolved {args.evolve_ticks} ticks; {parents.shape[0]} "
              f"{args.parents} parents; depth mean {d.mean():.1f} (founders "
              f"{(d == 0).float().mean():.0%}); length {w.glen[top].mean():.1f}, expressed "
              f"{w.expressed[top].mean():.1f}", flush=True)

    reg = regions(parents, stop)
    every = torch.ones(len(parents), dtype=torch.bool)
    rand = genome.random_genomes(len(parents), ecfg.genome_len, ecfg.alphabet,
                                 ecfg.init.genome_len, gen, exclude=stop)
    tests = [("null (identical copy)", parents, every),
             ("random genome (+ctrl)", rand, every)]
    for k in (1, 3):
        m, ok = flip(parents, k, reg["any"], ecfg.alphabet, gen)
        tests.append((f"{k} flip(s), anywhere", m, ok))
    if stop is not None:
        for r in ("expressed", "junk"):
            m, ok = flip(parents, 1, reg[r], ecfg.alphabet, gen)
            tests.append((f"1 flip, {r}", m, ok))
        m, ok = flip(parents, 1, reg["stop"], ecfg.alphabet, gen, avoid=genome.PAD)
        tests.append(("1 flip, the STOP itself", m, ok))

    out = {}
    for i, (name, mut, ok) in enumerate(tests):
        s = compete(over, args.config, parents[ok], mut[ok], args.ticks, args.grid, seed=11 + i)
        print(f"[{args.label}] " + summarize(name, s), flush=True)
        out[name] = s.tolist()
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"label": args.label, "over": over, "results": out}, f)


if __name__ == "__main__":
    main()
