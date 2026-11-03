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


def layout_matched(parents: torch.Tensor, alphabet: int, gen: torch.Generator,
                   exclude: int | None = None) -> torch.Tensor:
    """Random non-pad tokens at exactly the parents' non-pad positions: same length and
    pad layout, no token identity."""
    allowed = torch.tensor([t for t in range(1, alphabet) if t != exclude], dtype=torch.uint8)
    rnd = allowed[torch.randint(0, len(allowed), parents.shape, generator=gen)]
    return torch.where(parents != 0, rnd, torch.zeros_like(parents))


def length_matched(parents: torch.Tensor, alphabet: int, gen: torch.Generator,
                   exclude: int | None = None) -> torch.Tensor:
    """Random genomes with each parent's non-pad length, packed at the start (the
    founders' compact layout)."""
    from soup import genome
    out = genome.random_genomes(parents.shape[0], parents.shape[1], alphabet, parents.shape[1],
                                gen, exclude=exclude)
    L = (parents != 0).sum(1)
    out[torch.arange(parents.shape[1]).view(1, -1) >= L.view(-1, 1)] = 0
    return out


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
    ap.add_argument("--mode", default="adaptation", choices=["adaptation", "damage", "controls"],
                    help="adaptation: evolved/prescreened vs random; damage: parents vs "
                         "their own k-flip mutants; controls: evolved parents vs layout- "
                         "and length-matched random genomes")
    ap.add_argument("--parents", default="deep", choices=["deep", "prescreened"],
                    help="damage mode: which genomes are the parents")
    ap.add_argument("--compete-over", default="{}",
                    help="extra JSON overrides for the competition runs only (not evolution)")
    ap.add_argument("--only", help="comma-separated set labels to run")
    ap.add_argument("--flips", default="1,3", help="damage mode: flip counts to test")
    ap.add_argument("--json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    over = json.loads(args.over)

    from soup import genome
    from soup.config import load_config
    cfg = load_config(args.config, over)
    rand = genome.random_genomes(args.n, cfg.genome_len, cfg.alphabet, cfg.init.genome_len,
                                 torch.Generator().manual_seed(7), exclude=cfg.ribosome.stop_token)
    stop = cfg.ribosome.stop_token
    if args.mode == "damage" and args.parents == "prescreened":
        deep = prescreen(args.config, over, args.n)
        print(f"prescreened parents: mean length {(deep != 0).sum(1).float().mean():.1f}")
    else:
        deep = evolve_deep(args.config, over, 4096, 128, args.n)
    if args.mode == "controls":
        gen = torch.Generator().manual_seed(5)
        sets = [("layout-matched random vs parents", layout_matched(deep, cfg.alphabet, gen, stop), deep),
                ("length-matched random vs parents", length_matched(deep, cfg.alphabet, gen, stop), deep),
                ("32-token random vs parents", rand, deep),
                ("copies vs parents (null)", deep.clone(), deep)]
    elif args.mode == "damage":
        from experiments.competition import flip, regions
        gen = torch.Generator().manual_seed(0)
        anywhere = regions(deep, cfg.ribosome.stop_token)["any"]
        # A = mutants, so share A < 0.5 means damage costs fitness. Flips are capped at
        # each genome's non-pad length; a random genome is the "fully destroyed" reference.
        sets = []
        for k in (int(x) for x in args.flips.split(",")):
            m, _ = flip(deep, k, anywhere, cfg.alphabet, gen, cap=True)
            sets.append((f"{k}-flip mutants vs parents", m, deep))
        sets += [("random vs parents (max damage)", rand, deep),
                 ("copies vs parents (null)", deep.clone(), deep)]
    else:
        good = prescreen(args.config, over, args.n)
        rand2 = genome.random_genomes(args.n, cfg.genome_len, cfg.alphabet, cfg.init.genome_len,
                                      torch.Generator().manual_seed(8),
                                      exclude=cfg.ribosome.stop_token)
        sets = [("evolved vs random", deep, rand), ("prescreened vs random", good, rand),
                ("random vs random (null)", rand2, rand)]
    if args.only:
        keep = args.only.split(",")
        sets = [x for x in sets if any(x[0].startswith(k) for k in keep)]
    from experiments.heat_options import merge
    cover = merge(over, json.loads(args.compete_over))
    jobs = [(lab, a, b, int(s), args.config, cover, args.ticks, args.grid)
            for lab, a, b in sets for s in args.seeds.split(",")]
    with get_context("spawn").Pool(args.procs) as pool:
        res = pool.map(one, jobs, chunksize=1)
    print(f"\nshare of live cells descending from set A after {args.ticks} ticks "
          f"(0.5 = no advantage)")
    for lab, _, _ in sets:
        v = torch.tensor([r["curve"][-1] for r in res if r["label"] == lab])
        print(f"  {lab:34s} {v.mean():.3f} ± {v.std() / len(v) ** .5:.3f}   "
              f"per seed {' '.join(f'{x:.2f}' for x in v.tolist())}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
