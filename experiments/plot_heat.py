"""Dose-response plot for the heat sweep: fidelity vs heat strength.

    python -m experiments.plot_heat runs/heat docs/m3_heat_sweep.png
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = {"drift": ("#2a78d6", "drifting heat field"), "static": ("#eb6834", "static heat field"),
          "freefid": ("#1baf7a", "static, fidelity free (c_fid = 0)")}
PANELS = [("mean_fidelity", "Mean fidelity (cell-weighted)"),
          ("fid_hot_minus_cold", "Fidelity: hot quartile − cold quartile"),
          ("organisms", "Organisms")]


def main(run_dir: str, out: str) -> None:
    with open(os.path.join(run_dir, "results.json")) as f:
        res = json.load(f)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for tag, (col, label) in SERIES.items():
            pts = []
            for r in res:
                n = r["name"]
                kind = "freefid" if n.endswith("freefid") else n.split("_")[1]
                if kind != tag:
                    continue
                pts.append((float(n.split("_")[0][4:]), r[key]))
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            jitter = {"drift": 0.93, "static": 1.0, "freefid": 1.07}[tag]
            ax.scatter([x * jitter for x in xs], ys, s=40, color=col, label=label, zorder=3,
                       edgecolors=SURFACE, linewidths=1.5)
            means = {}
            for x, y in pts:
                means.setdefault(x, []).append(y)
            mx = sorted(means)
            if len(mx) > 1:
                ax.plot([x * jitter for x in mx], [sum(means[x]) / len(means[x]) for x in mx],
                        color=col, lw=2, zorder=2)
        if key == "fid_hot_minus_cold":
            ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([0.006, 0.03, 0.1], ["0.006", "0.03", "0.1"])
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("base_flip (heat damage rate)", color=MUTED, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
