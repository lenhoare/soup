# Ribosome Soup

An artificial-life system where organisms grow from short, mutable code strings. Each organism's genome (64 tokens) is translated by a shared, frozen neural "ribosome" into the weights of a tiny cell network. Every cell runs that network on a 2D torus: it senses neighbours, emits chemicals, grows, spores, attacks or rests. Genomes mutate, get spliced between neighbours (horizontal gene transfer), are damaged by heat, and cost energy per token.

- Design: [`initial_spec.md`](initial_spec.md)
- Prior work: [`docs/m0_literature.md`](docs/m0_literature.md)
- Findings so far: [Results](#results)
- Experiments still to run: [`docs/tests.md`](docs/tests.md)

![Viewer](docs/m4_viewer.png)

## Install

```bash
# PyTorch first (CUDA build from pytorch.org for a GPU), then:
pip install -r requirements.txt
python -m pytest -q                 # 33 tests; GL tests skip without EGL
```

Triton powers the fast GPU kernel (M6). On Linux it comes with CUDA PyTorch. On Windows use `pip install triton-windows`. Without Triton, everything runs on the eager PyTorch path.

## On the RTX 3060 — first things to try

```bash
python -m pytest -q tests/test_kernel.py            # Triton kernel matches eager
python -m experiments.bench_kernel                  # eager vs Triton, ticks/s per tier
python -m experiments.bench_kernel --profile m      # where tier-M time goes
python -m render.viewer --config config/tier_s.yaml # the interactive viewer
```

`docs/tests.md` #6 says what to look for in the benchmark output.

## Running

```bash
python -m soup.run --config config/tier_s.yaml --ticks 20000 --log metrics.jsonl \
    --snapshot out.png --checkpoint-every 2000          # headless run
python -m soup.run --load checkpoints/soup_00010000.pt --ticks 5000   # resume (exact)
python -m soup.run --config config/tier_s.yaml --bench
```

- **Presets:**
  - `config/tier_s.yaml`: 256², dev loop
  - `tier_m.yaml`: 512²
  - `tier_l.yaml`: 1024², "1M cells"
  - `tier_s_heat_zones.yaml`: sharp heat zones, where fidelity adapts locally
- **Settings** are documented inline in `soup/config.py`. Override them with YAML or `--seed` / `--device`.
- **Backends:** `kernel: auto` uses Triton on CUDA, eager elsewhere.

## Viewer

```bash
python -m render.viewer --config config/tier_s.yaml                    # window
python -m render.viewer --video soup.mp4 --frames 1800 --size 1920x1080 --warmup 1000
python -m render.viewer --load checkpoints/soup_00010000.pt --video hq.mp4 \
    --size 1920x1080 --supersample 2 --frames 900                     # re-render a moment
```

| Input | Action |
|---|---|
| mouse wheel / drag | zoom at cursor / pan |
| click / Esc | inspect organism (nearest within 3 cells) / clear |
| space / `.` | pause / single tick |
| `[` `]` | halve / double ticks per frame |
| `1`–`6` | lineage, energy, morphogens, genome distance, heat/light, lineage depth |
| `H` `C` `G` | toggle heat / token cost / gene transfer live |
| `F` `S` `B` `O` | fit view / snapshot PNG / bloom / HUD |
| `K` | save a checkpoint to `checkpoints/` |

**What you're seeing:**
- **Hue** is the lineage colour, from the ribosome's genome embedding, so similar genomes get similar colours.
- **Brightness** is energy.
- **Interior texture** comes from the cells' hidden state.
- **Borders:** bright membranes separate lineages; faint lines separate sibling organisms (zoom in to see them).
- **Flashes:** gold-white is a gene transfer, cold blue is heat damage, white pops are spores.
- **Orange embers** in empty space show heat.
- **The inspector** shows the 64-token genome. Pads are grey, spliced-in tokens have gold outlines, and with a STOP token the untranslated tail fades out.

Headless and video modes use EGL on Linux, or the platform's default offscreen context elsewhere.

## Experiments

| script | what it does | write-up |
|---|---|---|
| `experiments/sweep.py` | parameter grid, end-of-run metrics | `m2_calibration.md` |
| `experiments/viability.py` | how many random strings make a living organism; α scan | `m2_calibration.md` |
| `experiments/ablation.py` (+ `plot_ablation.py`) | 2³ feature ablation + direct-weight control | `m3_features.md` |
| `experiments/heat_sweep.py` (+ `plot_heat.py`) | heat strength × drifting/static field | `m3_features.md` |
| `experiments/heat_options.py` | STOP token, `fid_power`, heat zones, direct heat effects | `heat_options.md` |
| `experiments/damage_cost.py` | isolated-growth cost of flips (too blunt; superseded) | `heat_options.md` |
| `experiments/competition.py` | walled-arena parent vs mutant | `competition.md` |
| `experiments/soup_competition.py` | 50/50 competition in the full soup: adaptation, damage dose-response, length controls | `competition.md` |
| `experiments/robustness.py` | phenotype-level flip sensitivity | `competition.md` |
| `experiments/bench_kernel.py` | M6 kernel + per-tier speed (CUDA) | `tests.md` #6 |

All experiments write to `runs/` (git-ignored). Most take 15–60 minutes on a 4-core CPU.

## Results

*Mostly 2–8 seeds at 128×128 on a cloud CPU. Read them as strong leads; `docs/tests.md` lists the replications to run.*

1. **Calibration** (`m2_calibration.md`).
   - At the spec's α = 0.5, the genome barely affected fitness (heritability r ≈ 0.05).
   - At **α = 6**: r ≈ 0.57, 59% of random strings are viable, and smoothness is ~12% weight change per token.
   - Crowding 0.9 stops the grid saturating.
2. **Gene transfer + token cost interact** (`m3_features.md`).
   - Together they give ~2.5× the organisms and Shannon ≈ 5, vs ≤ 3.6 for any other feature combination.
   - The string + ribosome system beats the direct-weight control on diversity and lineage depth.
3. **Heat on code** (`m3_features.md`, `heat_options.md`).
   - Fidelity rises globally with heat, but not locally, until heat comes in **sharp zones**. Then hot-zone organisms carry +0.13 fidelity, and both seeds agree.
   - `fid_power: 3` also works.
   - A STOP token plus heat creates ~60% untranslated "junk DNA", which blocks fidelity adaptation.
4. **What evolution actually found** (`competition.md`).
   - Evolved lineages take ~80% of a mixed soup from random genomes, mostly by **being short** (token cost). With a flat cost the advantage shrinks to 0.59 : 0.41.
   - There is real token content too: at the evolved length, random tokens lose ~0.3 : 0.7.
   - Because flips keep length, evolved genomes shrug off 10–20 heat flips, even though each flip changes their network as much as anyone's. Genomes whose advantage is in their tokens lose heavily.
   - So the soup's fitness depends weakly on most cell-net weights. Making behaviour matter more is the next big lever (`tests.md` #9).

## Status

| milestone | status |
|---|---|
| M0 literature | ✅ `docs/m0_literature.md` |
| M1 Tier S skeleton | ✅ |
| M2 evolution + calibration | ✅ `docs/m2_calibration.md` |
| M3 heat, cost, HGT + ablations | ✅ `docs/m3_features.md`, `heat_options.md`, `competition.md` |
| M4 viewer | ✅ |
| M5 polish | ✅ bloom/tonemap, flashes, six view modes, video, checkpoints, 2× supersampling |
| M6 fast kernel | 🟡 Triton per-genome kernel written and verified for correctness; **needs benchmarking on a GPU** |
| M7 showpiece run | — |

## Layout

```
soup/
  config.py       defaults (documented) + YAML presets in config/
  genome.py       token strings, effective length, spore-time mutation
  ribosome.py     frozen seeded transformer: genome -> dW, traits, embedding (+ STOP)
  cellnet.py      cell-net weight layout, hand-set base net, eager per-cell forward
  kernels/        cellnet_triton.py: per-genome segmented MLP kernel (M6)
  fields.py       drifting light / heat fields (optionally sharp heat zones)
  world.py        grid, slots, tick loop, epoch (heat damage, HGT), barriers
  metrics.py      population / diversity / compression / robustness metrics
  checkpoint.py   full-state save / exact resume
  run.py          headless runner + benchmark
render/
  viewer.py       glfw window, input, HUD/inspector; headless PNG / ffmpeg video
  gpu.py          moderngl renderer: textures, view modes, bloom, flashes
  shaders.py      GLSL (OKLCH colour, membranes, embers, tonemap)
  overlay.py      HUD + inspector (genome ribbon) drawn with Pillow
  snapshot.py     simple CPU PNG renderer (no GL needed)
experiments/      see the table above
docs/             write-ups and figures
tests/            unit tests (pytest)
```
