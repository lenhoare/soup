"""Small-multiple time series for an ablation run directory.

    python -m experiments.plot_ablation runs/ablation docs/m3_ablation.png

Highlights full / baseline / control; the other six conditions are grey context.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [("organisms", "Organisms (live slots)"), ("lineages", "Founder lineages alive"),
          ("mean_len", "Mean genome length (tokens)"), ("hgt_frac", "Genomes carrying a splice"),
          ("mean_fidelity", "Mean fidelity (cell-weighted)"),
          ("fid_heat_corr", "corr(fidelity, local heat)")]
HIGHLIGHT = {"full": "#2a78d6", "baseline": "#eb6834", "control": "#1baf7a"}
CONTEXT = "#c3c2b7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load(run_dir: str) -> dict[str, list[list[dict]]]:
    runs: dict[str, list[list[dict]]] = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        name = os.path.basename(p).rsplit("_s", 1)[0]
        with open(p) as f:
            runs[name].append([json.loads(line) for line in f])
    return runs


def seed_mean(rows_list: list[list[dict]], key: str) -> tuple[list[int], list[float]]:
    n = min(len(r) for r in rows_list)
    ticks = [rows_list[0][i]["tick"] for i in range(n)]
    vals = [sum(r[i].get(key, float("nan")) for r in rows_list) / len(rows_list)
            for i in range(n)]
    return ticks, vals


def main(run_dir: str, out: str) -> None:
    runs = load(run_dir)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        for name, rows in runs.items():
            if name in HIGHLIGHT:
                continue
            ax.plot(*seed_mean(rows, key), color=CONTEXT, lw=1.2, zorder=1)
        for name, color in HIGHLIGHT.items():
            if name in runs:
                ax.plot(*seed_mean(runs[name], key), color=color, lw=2, zorder=3, label=name)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[1, 0].set_xlabel("tick", color=MUTED, fontsize=9)
    handles = [plt.Line2D([], [], color=c, lw=2, label=n) for n, c in HIGHLIGHT.items()]
    handles.append(plt.Line2D([], [], color=CONTEXT, lw=1.2, label="other 6 ablations"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
