# Ribosome Soup

An artificial-life system where organisms grow from short mutable code strings, translated by a shared neural "ribosome". See [`initial_spec.md`](initial_spec.md) for the design and [`docs/m0_literature.md`](docs/m0_literature.md) for prior work.

## Quick start

```bash
pip install -r requirements.txt          # for CUDA, install torch from pytorch.org first
python -m pytest -q
python -m soup.run --config config/tier_s.yaml --ticks 2000 --snapshot out.png
python -m soup.run --config config/tier_s.yaml --bench
```

Useful flags: `--log metrics.jsonl` (per-epoch metrics), `--snapshot-every 100` (image sequence), `--device cpu|cuda`, `--seed N`.

## Viewer

```bash
python -m render.viewer --config config/tier_s.yaml           # interactive window
python -m render.viewer --video soup.mp4 --frames 1800 --size 1920x1080 --warmup 1000
python -m render.viewer --headless frames/ --frames 300        # PNG sequence
```

| Input | Action |
|---|---|
| mouse wheel / drag | zoom at cursor / pan |
| click / Esc | inspect organism (nearest within 3 cells) / clear |
| space / `.` | pause / single tick |
| `[` `]` | halve / double ticks per frame |
| `1`–`6` | lineage, energy, morphogens, genome distance, heat/light, lineage depth |
| `H` `C` `G` | toggle heat / token cost / gene transfer live |
| `F` / `S` / `B` / `O` | fit view / snapshot PNG / bloom / HUD |
| `K` | save a checkpoint to `checkpoints/` |

**Checkpoints:**
- `python -m soup.run --ticks 20000 --checkpoint-every 2000` saves periodic checkpoints. Only live slots' weights are stored, so they stay small: about 2.5 MB at 128×128.
- `python -m soup.run --load checkpoints/soup_00010000.pt --ticks 5000` resumes. On the same device this is exact: identical to never having stopped.
- `python -m render.viewer --load checkpoints/soup_00010000.pt --video hq.mp4 --size 1920x1080 --supersample 2 --frames 900` re-renders a saved moment at high quality.

**What you're seeing:**
- **Hue** is the lineage colour, taken from the ribosome's genome embedding, so similar genomes get similar colours.
- **Brightness** is energy.
- **Texture inside a body** comes from the cells' hidden state.
- **Borders:** bright membranes separate lineages, and faint lines separate sibling organisms (visible when zoomed in).
- **Flashes:** gold-white is a gene transfer, cold-blue is heat damage, and white pops are spores.
- **Orange embers** in empty space show heat.
- **The inspector** shows the genome as 64 tiles: pads are grey, and spliced-in tokens have gold outlines.

The headless modes need EGL; the window needs a display. Tier S/M copy world state to textures each frame. At Tier L that copy is the bottleneck, and CUDA-GL interop would remove it.

![Viewer](docs/m4_viewer.png)

## Layout

```
soup/
  config.py    defaults + YAML presets (config/tier_{s,m,l}.yaml)
  genome.py    token strings, effective length, spore-time mutation
  ribosome.py  frozen seeded transformer: genome -> dW, traits, embedding
  cellnet.py   cell-net weight layout, hand-set base net, batched per-cell forward
  fields.py    drifting light / heat fields on the torus
  world.py     grid, slots, tick loop (perceive, act, energy, death), epoch
  metrics.py   population/diversity/compression metrics, phenotype smoothness
  run.py       headless runner + benchmark
render/
  viewer.py    glfw window, input, HUD/inspector; headless PNG / ffmpeg video
  gpu.py       moderngl renderer: textures, view modes, bloom, flashes
  shaders.py   GLSL (OKLCH colour, membranes, embers, tonemap)
  overlay.py   HUD + inspector (genome ribbon) drawn with Pillow
  snapshot.py  simple CPU PNG renderer (no GL needed)
```

## Status

- **M0 literature check:** done.
- **M1 Tier S skeleton:** done.
- **M2 evolution + calibration:** done. See [`docs/m2_calibration.md`](docs/m2_calibration.md). The key changes:
  - `crowding 0.9` stops the grid saturating.
  - Spore thresholds were lowered so reproduction happens.
  - **α raised from 0.5 to 6.** At 0.5 the genome had almost no effect on fitness (heritability r ≈ 0.05). At 6, r ≈ 0.48, 54% of random strings are viable, and phenotype smoothness is 12%.
  - In a 4096-tick run, lineages drop from 256 to 51 and mean genome length shrinks from 32 to 28.
- **M3 three features + ablation:** done. See [`docs/m3_features.md`](docs/m3_features.md).
  - Heat damage, gene transfer, and a matched direct-weight control (`mode: direct`) are built.
  - With 2 seeds: **gene transfer + token cost together** give about 2.5× the organisms and Shannon ≈ 5, vs. ≤ 3.6 for any other combination.
  - The string version beats the direct-weight control on diversity and lineage depth.
  - Heat on code shows no adaptive signature yet.
- **M4 viewer:** done (see below).
- **Heat follow-up:** see the end of [`docs/m3_features.md`](docs/m3_features.md).
  - Fidelity is selected down under weak heat and up under strong heat, but never locally (hot vs. cold regions).
  - Heat damage turns out to be nearly neutral for evolved genomes (about −7% colony size per flip), so selection on fidelity is weak.
- **Heat options:** see [`docs/heat_options.md`](docs/heat_options.md). Four switches, all off by default: STOP token, `fid_power`, heat zones, direct heat effects.
  - **Heat zones** give clear local adaptation: hot-zone organisms carry 0.13 more fidelity, and both seeds agree.
  - **`fid_power: 3`** gives about 4× the baseline effect.
  - **STOP + heat** creates junk DNA (61% of each genome untranslated), which blocks fidelity adaptation.
- **Competition assays:** see [`docs/competition.md`](docs/competition.md).
  - **Evolution is adaptive:** evolved lineages take about 80% of a mixed soup from random genomes (4 of 4 seeds).
  - **The adaptation is robust:** 3 heat flips cost evolved genomes nothing measurable, even in the soup (8 seeds). That explains why fidelity is only weakly selected.
  - Empty-arena and isolated-growth assays measure colonisation, not soup fitness.
- **Flip dose-response:** there's a threshold. 10–20 flips (of ~28 tokens) cost nothing, 30 flips cost a third of the territory, and a random genome loses badly. Part of the evolved advantage seems to be just *shorter genomes* (27.5 vs 32 tokens).
- **M5:** checkpoints (`--checkpoint-every`, `--load`, `K` in the viewer, exact resume) and 2× supersampled video.
- **Important tests to run later:** [`docs/tests.md`](docs/tests.md), including mutational robustness and a length-matched control.
- **Next:** M6 fused GPU kernel, then the tests in `docs/tests.md` on the 3060.

Calibration tools:

```bash
python -m experiments.sweep --set energy.crowding=0.5,0.9 --set energy.photo=0.08,0.12
python -m experiments.viability --alpha 2,6,8
python -m experiments.ablation --out runs/ablation --ticks 8192 --seeds 1,2 --procs 4
python -m experiments.plot_ablation runs/ablation docs/m3_ablation.png
```

![Ablation time series](docs/m3_ablation.png)

![Tier S after 4096 ticks](docs/m2_snapshot.png)
