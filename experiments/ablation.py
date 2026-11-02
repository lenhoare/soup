"""Ablation grid (spec 8.1): 2^3 feature combinations + the direct-weight control.

    python -m experiments.ablation --out runs/ablation --ticks 8192 --seeds 1,2 --procs 4

Each run writes <out>/<name>_s<seed>.jsonl (per-epoch metrics) and a PNG of its
final state; a summary table (means over the last quarter of each run, averaged
over seeds) is printed and written to <out>/summary.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import get_context

CONDITIONS: dict[str, dict] = {}
for heat, cost, hgt in itertools.product([False, True], repeat=3):
    on = [n for n, f in (("heat", heat), ("cost", cost), ("hgt", hgt)) if f]
    name = "+".join(on) if on else "baseline"
    if len(on) == 3:
        name = "full"
    CONDITIONS[name] = {"features": {"heat": heat, "cost": cost, "hgt": hgt}}
CONDITIONS["control"] = {"features": {"heat": True, "cost": True, "hgt": True}, "mode": "direct"}

COLS = ["organisms", "lineages", "shannon", "mean_org_size", "mean_len", "zlib_ratio",
        "hgt_frac", "mean_fidelity", "fid_heat_corr", "mean_permeability",
        "mean_aggression", "mean_energy", "occupancy", "mean_depth"]


def run(job: tuple) -> dict:
    name, seed, args = job
    import torch
    torch.set_num_threads(args["threads"])
    torch.set_grad_enabled(False)
    from render.snapshot import render, write_png
    from soup.config import load_config
    from soup.metrics import snapshot
    from soup.world import World

    over = json.loads(json.dumps(CONDITIONS[name]))
    over.update({"seed": seed, "grid": [args["grid"], args["grid"]],
                 "init": {"founders": args["founders"]}, "device": "cpu"})
    cfg = load_config(args["config"], over)
    w = World(cfg)
    w.seed_founders()
    rows = []
    t0 = time.perf_counter()
    path = os.path.join(args["out"], f"{name}_s{seed}")
    with open(path + ".jsonl", "w") as f:
        for t in range(args["ticks"]):
            w.tick()
            if w.tick_count % cfg.epoch_ticks == 0:
                m = snapshot(w)
                rows.append(m)
                f.write(json.dumps(m) + "\n")
                if m["cells"] == 0:
                    break
    write_png(path + ".png", render(w), scale=2)
    summ = tail_summary(rows)
    summ.update(name=name, seed=seed, seconds=time.perf_counter() - t0,
                extinct=rows[-1]["cells"] == 0)
    print(f"done {name} s{seed} in {summ['seconds']:.0f}s", flush=True)
    return summ


def tail_summary(rows: list[dict]) -> dict:
    tail = rows[-max(1, len(rows) // 4):]
    return {c: sum(r.get(c, 0.0) for r in tail) / len(tail) for c in COLS}


def summarize_dir(out: str) -> list[dict]:
    import glob
    res = []
    for p in sorted(glob.glob(os.path.join(out, "*.jsonl"))):
        name, seed = os.path.basename(p)[:-6].rsplit("_s", 1)
        with open(p) as f:
            rows = [json.loads(line) for line in f]
        res.append({**tail_summary(rows), "name": name, "seed": int(seed)})
    return res


def table(results: list[dict]) -> str:
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["name"], []).append(r)
    lines = ["| condition | " + " | ".join(COLS) + " |",
             "|---|" + "---|" * len(COLS)]
    for name in CONDITIONS:
        if name not in by:
            continue
        rs = by[name]
        vals = [sum(r[c] for r in rs) / len(rs) for c in COLS]
        lines.append(f"| {name} | " + " | ".join(f"{v:.3g}" for v in vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--ticks", type=int, default=8192)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--founders", type=int, default=64)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", help="comma-separated condition names")
    ap.add_argument("--summarize", action="store_true",
                    help="rebuild summary.md from the .jsonl files in --out without running")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.summarize:
        t = table(summarize_dir(args.out))
        with open(os.path.join(args.out, "summary.md"), "w") as f:
            f.write(f"{t}\n")
        print(t)
        return
    names = args.only.split(",") if args.only else list(CONDITIONS)
    a = {k: getattr(args, k) for k in ("config", "out", "ticks", "grid", "founders", "threads")}
    jobs = [(n, int(s), a) for s in args.seeds.split(",") for n in names]
    with get_context("spawn").Pool(args.procs) as pool:
        results = pool.map(run, jobs, chunksize=1)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    t = table(results)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"ticks={args.ticks} grid={args.grid} seeds={args.seeds}\n\n{t}\n")
    print(t)


if __name__ == "__main__":
    main()
