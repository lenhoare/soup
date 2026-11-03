"""Heat sweep: why does fidelity not adapt to heat? (M3 follow-up)

Crosses heat strength (base_flip) with a drifting vs static heat field, plus a
free-fidelity check (c_fid = 0) at the strongest heat. All other features on.

    python -m experiments.heat_sweep --out runs/heat --ticks 8192 --seeds 1,2 --procs 4
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context

from experiments.ablation import run

FLIPS = [0.006, 0.03, 0.1]
CONDITIONS: dict[str, dict] = {}
for bf in FLIPS:
    for drift, tag in ((0.001, "drift"), (0.0, "static")):
        CONDITIONS[f"flip{bf}_{tag}"] = {"heat": {"base_flip": bf},
                                         "fields": {"heat_drift": drift}}
CONDITIONS["flip0.1_static_freefid"] = {"heat": {"base_flip": 0.1},
                                        "fields": {"heat_drift": 0.0},
                                        "energy": {"c_fid": 0.0}}

SHOW = ["organisms", "shannon", "mean_len", "flips_per_org_epoch", "mean_fidelity",
        "fid_heat_corr", "fid_hot_minus_cold", "repeat_frac", "repeat_hot_minus_cold"]


def run_conditions(conditions: dict[str, dict], show: list[str], default_out: str) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--ticks", type=int, default=8192)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--founders", type=int, default=64)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", help="comma-separated condition names")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    names = args.only.split(",") if args.only else list(conditions)
    a = {k: getattr(args, k) for k in ("config", "out", "ticks", "grid", "founders", "threads")}
    jobs = [(n, int(s), a, conditions[n]) for s in args.seeds.split(",") for n in names]
    with get_context("spawn").Pool(args.procs) as pool:
        results = pool.map(run, jobs, chunksize=1)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    lines = ["| condition | seeds | " + " | ".join(show) + " |", "|---|---|" + "---|" * len(show)]
    for name in names:
        rs = sorted((r for r in results if r["name"] == name), key=lambda r: r["seed"])
        cells = []
        for c in show:
            vals = [r.get(c, float("nan")) for r in rs]
            cells.append(f"{sum(vals) / len(vals):.3g} ({'/'.join(f'{v:.2g}' for v in vals)})")
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    t = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(t + "\n")
    print(t)


def main() -> None:
    run_conditions(CONDITIONS, SHOW, "runs/heat")


if __name__ == "__main__":
    main()
