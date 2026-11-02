"""Headless runner and benchmark.

    python -m soup.run --config config/tier_s.yaml --ticks 2000 --snapshot out.png
    python -m soup.run --config config/tier_s.yaml --bench
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .config import load_config
from .metrics import format_line, snapshot
from .world import World


def _sync(world: World) -> None:
    if world.device.type == "cuda":
        torch.cuda.synchronize()


def bench(world: World, ticks: int = 200, warmup: int = 20) -> float:
    world.run(warmup)
    _sync(world)
    t0 = time.perf_counter()
    world.run(ticks)
    _sync(world)
    return ticks / (time.perf_counter() - t0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--ticks", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=4, help="epochs between log lines")
    ap.add_argument("--log", help="write per-epoch metrics as JSON lines")
    ap.add_argument("--snapshot", help="write a PNG of the final state")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also write <snapshot>_<tick>.png every N ticks")
    ap.add_argument("--bench", action="store_true", help="time ticks/s and exit")
    ap.add_argument("--device", help="override device (cpu/cuda)")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)

    over = {}
    if args.device:
        over["device"] = args.device
    if args.seed is not None:
        over["seed"] = args.seed
    cfg = load_config(args.config, over)
    torch.set_grad_enabled(False)
    world = World(cfg)
    world.seed_founders()
    print(f"device={world.device} grid={cfg.grid} slots={cfg.genome_slots} "
          f"cellnet_params={world.layout.n_params} "
          f"ribosome_params={sum(p.numel() for p in world.ribosome.parameters())}")

    if args.bench:
        # grow a population first so the benchmark measures a busy world
        world.run(cfg.epoch_ticks * 8)
        tps = bench(world)
        print(format_line(snapshot(world)))
        print(f"ticks/s = {tps:.1f}")
        return

    log = open(args.log, "w") if args.log else None
    t0 = time.perf_counter()
    for t in range(1, args.ticks + 1):
        world.tick()
        if args.snapshot and args.snapshot_every and t % args.snapshot_every == 0:
            from render.snapshot import render, write_png
            write_png(args.snapshot.replace(".png", f"_{t:06d}.png"), render(world))
        if t % cfg.epoch_ticks == 0:
            m = snapshot(world)
            m["ticks_per_s"] = t / (time.perf_counter() - t0)
            if log:
                log.write(json.dumps(m) + "\n")
            if (t // cfg.epoch_ticks) % args.log_every == 0:
                print(format_line(m) + f"  tps={m['ticks_per_s']:.1f}", flush=True)
            if m["cells"] == 0:
                print("extinct")
                break
    if log:
        log.close()
    if args.snapshot:
        from render.snapshot import render, write_png
        write_png(args.snapshot, render(world), scale=2)
        print(f"wrote {args.snapshot}")


if __name__ == "__main__":
    main()
