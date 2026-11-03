"""M6 benchmark: eager vs Triton cell-net, and whole-tick speed per tier. Needs CUDA.

    python -m experiments.bench_kernel                 # micro + tiers S, M, L
    python -m experiments.bench_kernel --tiers s,m     # skip tier L
    python -m experiments.bench_kernel --profile m     # top CUDA ops for one tier

Prints: kernel time per cell-net pass at 65k / 262k / 1M cells (max error vs eager),
then ticks/s per tier for both backends on the *same* evolved world state, with the
spec targets (S >= 60, M >= 200, L >= 60 ticks/s), genomes per 16x16 tile and mean
cells per Triton block (the weight-reuse factor).
"""
from __future__ import annotations

import argparse
import time

import torch

from soup import cellnet
from soup.cellnet import CellNetLayout
from soup.config import load_config
from soup.metrics import genomes_per_tile
from soup.world import World

TARGETS = {"s": 60, "m": 200, "l": 60}


def timeit(fn, reps: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def micro(dev: torch.device, dtype: torch.dtype) -> None:
    from soup.kernels import cellnet_triton as K
    lay = CellNetLayout()
    print(f"\n== cell-net pass only ({dtype}) ==")
    print(f"{'cells':>9s} {'eager ms':>9s} {'triton ms':>10s} {'speedup':>8s} {'max err':>9s}")
    for n in (65_536, 262_144, 1_048_576):
        S = max(256, n // 16)                       # ~16-cell organisms, like Tier L
        w = (torch.randn(S, lay.n_params, device=dev) * 0.2).to(dtype)
        sizes = torch.randint(1, 48, (S,), device=dev)
        gid = torch.repeat_interleave(torch.arange(S, device=dev), sizes)[:n]
        gid = gid[torch.randperm(len(gid), device=dev)]      # cells arrive unsorted
        x = torch.randn(len(gid), lay.n_in, device=dev)
        te = timeit(lambda: cellnet.forward(lay, x, w, gid))
        tt = timeit(lambda: K.forward(lay, x, w, gid))
        err = (K.forward(lay, x, w, gid) - cellnet.forward(lay, x, w, gid)).abs().max()
        print(f"{len(gid):9d} {te * 1e3:9.2f} {tt * 1e3:10.2f} {te / tt:7.1f}x {float(err):9.1e}")


def tier(name: str, warmup: int, ticks: int, profile: bool) -> None:
    cfg = load_config(f"config/tier_{name}.yaml", {"device": "cuda", "kernel": "eager"})
    w = World(cfg)
    w.seed_founders()
    t0 = time.perf_counter()
    w.run(warmup)                                   # grow a busy population first
    torch.cuda.synchronize()
    cells = int((w.gid >= 0).sum())
    alive_g = w.gid[w.gid >= 0]
    from soup.kernels.cellnet_triton import segments
    _, _, seg_len = segments(torch.sort(alive_g).values, cfg.kernel_block)
    print(f"\n== tier {name.upper()} {cfg.grid} ({cfg.dtype}) — warmed {warmup} ticks in "
          f"{time.perf_counter() - t0:.0f}s: {cells:,} cells, "
          f"{genomes_per_tile(w.gid):.1f} genomes/tile, "
          f"{seg_len.float().mean():.1f} cells per triton block ==")
    if profile:
        from torch.profiler import ProfilerActivity, profile as prof
        w.kernel = "triton"
        with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            w.run(10)
            torch.cuda.synchronize()
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=15))
        return
    for backend in ("eager", "triton"):
        w.kernel = backend
        w.run(5)
        torch.cuda.synchronize()
        t = time.perf_counter()
        w.run(ticks)
        torch.cuda.synchronize()
        tps = ticks / (time.perf_counter() - t)
        ok = "meets" if tps >= TARGETS[name] else "below"
        print(f"  {backend:7s} {tps:8.1f} ticks/s   ({ok} spec target {TARGETS[name]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="s,m,l")
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--profile", metavar="TIER", help="profile one tier instead of timing")
    ap.add_argument("--no-micro", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required (the CPU only runs the eager path; kernel "
                         "correctness is covered by tests/test_kernel.py)")
    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    print(torch.cuda.get_device_name(), "| torch", torch.__version__)
    if not a.no_micro and not a.profile:
        micro(dev, torch.float32)
        micro(dev, torch.float16)
    for name in ([a.profile] if a.profile else a.tiers.split(",")):
        tier(name, a.warmup, a.ticks, bool(a.profile))


if __name__ == "__main__":
    main()
