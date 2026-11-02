"""Parameter sweep for calibration (M2).

    python -m experiments.sweep --ticks 1024 --grid 128 \
        --set energy.crowding=0.5,0.9 --set energy.photo=0.08,0.12

Every combination of the --set values is run; prints one row per run with
end-of-run metrics plus rates averaged over the second half of the run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import torch

from soup.config import load_config
from soup.metrics import snapshot
from soup.world import World


def nested(key: str, value) -> dict:
    out: dict = {}
    cur = out
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return out


def merge(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v
    return a


def run_one(config: str, over: dict, ticks: int) -> dict:
    cfg = load_config(config, over)
    w = World(cfg)
    w.seed_founders()
    half = ticks // 2
    w.run(half)
    c0 = dict(w.counters)
    cells_sum, n = 0, 0
    for _ in range(ticks - half):
        w.tick()
        if w.tick_count % cfg.epoch_ticks == 0:
            cells_sum += int((w.gid >= 0).sum())
            n += 1
    m = snapshot(w)
    span = ticks - half
    mean_cells = max(cells_sum / max(n, 1), 1)
    for k in ("births", "spores", "deaths", "attacks_won"):
        m[f"{k}_per_tick"] = (w.counters[k] - c0[k]) / span
    m["turnover"] = m["deaths_per_tick"] / mean_cells   # fraction of cells dying per tick
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--ticks", type=int, default=1024)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--founders", type=int, default=64)
    ap.add_argument("--set", action="append", default=[], help="key=v1,v2,...")
    ap.add_argument("--json", help="append results as JSON lines")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    keys, values = [], []
    for s in args.set:
        k, vs = s.split("=", 1)
        keys.append(k)
        values.append([json.loads(v) for v in vs.split(",")])

    cols = ["occupancy", "organisms", "mean_org_size", "lineages", "mean_energy", "mean_depth", "mean_len", "hgt_frac",
            "spores_per_tick", "turnover", "attacks_won_per_tick"]
    print("  ".join(keys + cols))
    for combo in itertools.product(*values) if values else [()]:
        over = {"grid": [args.grid, args.grid], "init": {"founders": args.founders}}
        for k, v in zip(keys, combo):
            merge(over, nested(k, v))
        t0 = time.perf_counter()
        m = run_one(args.config, over, args.ticks)
        m["seconds"] = time.perf_counter() - t0
        row = [str(v) for v in combo] + [
            f"{m.get(c, 0):.3g}" if isinstance(m.get(c, 0), float) else str(m.get(c, 0))
            for c in cols]
        print("  ".join(row), flush=True)
        if args.json:
            with open(args.json, "a") as f:
                f.write(json.dumps({"over": over, **m}) + "\n")


if __name__ == "__main__":
    main()
