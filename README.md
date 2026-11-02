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
  snapshot.py  PNG renderer (OKLCH lineage colour, membranes, heat speckle)
```

## Status

- **M0 literature check:** done.
- **M1 Tier S skeleton:** done.
- **M2 evolution + calibration:** done. See [`docs/m2_calibration.md`](docs/m2_calibration.md). The key changes:
  - `crowding 0.9` stops the grid saturating.
  - Spore thresholds were lowered so reproduction happens.
  - **α raised from 0.5 to 6.** At 0.5 the genome had almost no effect on fitness (heritability r ≈ 0.05). At 6, r ≈ 0.48, 54% of random strings are viable, and phenotype smoothness is 12%.
  - In a 4096-tick run, lineages drop from 256 to 51 and mean genome length shrinks from 32 to 28.
- **Next (M3):** heat damage and HGT (marked `TODO(M3)` in `World.epoch`), then the ablation grid. Attack churn also needs tuning.

Calibration tools:

```bash
python -m experiments.sweep --set energy.crowding=0.5,0.9 --set energy.photo=0.08,0.12
python -m experiments.viability --alpha 2,6,8
```

![Tier S after 4096 ticks](docs/m2_snapshot.png)
