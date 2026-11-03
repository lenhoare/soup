# Important tests to run later

Experiments that would firm up or explain results so far. Most were too slow for the cloud CPU and should be cheap on the RTX 3060. Each entry says why it matters, how to run it, and how to read the result. Unit tests are separate: `python -m pytest -q`.

Shared setting unless stated: static mid heat, `--over '{"heat": {"base_flip": 0.03}, "fields": {"heat_drift": 0.0}}'`.

---

## 1. Mutational robustness — "survival of the flattest"  ⭐ priority

**Why.** Three heat flips cost evolved genomes nothing measurable, even in the soup (`competition.md`). One explanation is that evolution has made them *robust*: they sit on flat regions of the genotype→phenotype landscape, where flips barely change the phenotype. Avida showed this at high mutation rates (Wilke et al., *Nature* 2001). The alternative is that *all* genomes, random ones included, are this insensitive, simply because the map is smooth.

**How.** Compare flip sensitivity of evolved and random genomes, at two levels:
- **Phenotype level (cheap, minutes).** For each genome, apply k = 1, 3, 10 flips and measure the relative change in cell-net weights, ‖W′ − W‖ / ‖W‖ (as in `metrics.phenotype_smoothness`), and in traits. Compare: (a) evolved genomes, (b) random genomes, (c) random genomes prescreened to *match the evolved genomes' fitness*, so robustness isn't confused with fitness.
- **Fitness level.** Run `experiments.soup_competition --mode damage` with the same flip counts on (a) evolved parents and (c) matched prescreened parents. More robust genomes lose less share per flip.
- Repeat across heat levels (`base_flip` 0.006 / 0.03 / 0.1): robustness should increase with heat if it is adaptive.
- *To build:* a small `experiments/robustness.py` that reuses `competition.flip` and `metrics.phenotype_smoothness` (the evolve and prescreen helpers already exist in `soup_competition.py`).

**Reading it.**
- Evolved genomes change *less* per flip than random genomes of equal fitness, and more so under stronger heat → robustness evolved (survival of the flattest). A strong result.
- No difference → insensitivity is a property of the ribosome map, not of evolution. Then "make the code more fragile" is the lever for making heat matter.

## 1b. Length-matched random control  ⭐ priority

**Why.** Evolved genomes are shorter (27.5 vs 32 non-pad tokens), and even fully flipped mutants keep the parent's length and layout and beat random genomes (0.33 vs 0.17). How much of "evolution is adaptive" is just *shorter genomes pay less token cost*?

**How.** Soup competition of evolved parents against:
- (a) random tokens placed at exactly the parents' non-pad positions (same length and layout);
- (b) random genomes truncated to the parents' lengths (same length, compact layout);
- (c) ordinary 32-token random genomes.

Also rerun with `features.cost` off during the contest, where length gives no energy advantage. *To build:* a `--mode controls` in `soup_competition.py`.

**Reading it.**
- (a) ≈ (b) ≈ 0.33, with cost-off removing most of the gap to (c) → the "half" is token cost.
- (a) > (b) → the pad layout itself matters (positional embeddings).

## 2. Flip dose-response in the soup

**Why.** To find where accumulated damage starts to hurt, and so test whether fidelity's response to heat zones comes from *accumulation* rather than single flips.

```bash
python -m experiments.soup_competition --mode damage --flips 10,20,30 --seeds 1,2,3,4 \
  --over '{"heat": {"base_flip": 0.03}, "fields": {"heat_drift": 0.0}}' \
  --compete-over '{"features": {"heat": false, "hgt": false}}'
```

**Status: done** (`competition.md`). Threshold result: 10 and 20 flips are neutral (0.49), 30 flips (≈ all tokens) cost a third of the territory (0.33), and a random genome loses badly (0.17). Re-run with 8 seeds, and with 22/24/26/28 flips to locate the threshold precisely.

## 3. More seeds for the headline results

All of these used 2 seeds (4–8 for competition). Run 8+ seeds, ideally also at 256×256:

| result | doc | command |
|---|---|---|
| HGT + token cost together give ~2.5× organisms and Shannon ≈ 5 | `m3_features.md` | `python -m experiments.ablation --seeds 1,2,3,4,5,6,7,8` |
| string + ribosome beats direct-weight control on diversity | `m3_features.md` | same (`control` condition) |
| heat zones give local fidelity adaptation (+0.13) | `heat_options.md` | `python -m experiments.heat_options --only baseline,zones,power --seeds 1,...,8` |
| combining options dilutes the zone effect | `heat_options.md` | `--only zones,all_but_stop` |
| evolved lineages take ~80% of the soup from random | `competition.md` | `python -m experiments.soup_competition --seeds 1,...,8` |

## 4. Why are string-mode lineages so much deeper than the control's?

**Why.** In the ablation, full string mode reached lineage depth 10–44, against 3–5 for the direct-weight control, and it isn't explained (`m3_features.md`). It may be a real evolvability difference, or just spores dying younger in the control.

**How.** Log per-slot lifetime and spore survival (whether a spore reaches ≥ N cells) in both modes. Compare the distribution of depth, generation time and survival.

## 5. Junk DNA: sponge, or tolerated damage?

**Why.** With a STOP token and heat, ~60% of each genome becomes untranslated junk and fidelity stops adapting (`heat_options.md`). Is junk an adaptive buffer, or damage that selection doesn't bother to remove?

**How.**
- Soup competition: STOP-evolved parents vs mutants with flips restricted to junk / expressed / the STOP itself. This needs a `--region` option in `soup_competition.py` (`competition.regions()` exists).
- Competition: junk-rich genomes vs the same genomes with junk deleted (padded out), under heat. If junk-rich genomes win under heat but lose without it (junk costs tokens), the junk is protective.

## 6. Performance on the 3060

- `python -m soup.run --config config/tier_s.yaml --bench`, then tier_m and tier_l. Compare with the spec targets (≥ 60 / ≥ 200 / ≥ 60 ticks/s). Tiers M and L likely need the M6 fused kernel.
- Note the `genomes_per_tile` value printed. It decides whether the tiled kernel's shared-memory plan works (~5–12 seen so far).
- Viewer frame rate at each tier (HUD `fps`), and whether the per-frame texture copy is the bottleneck at Tier L.

## 7. Viewer checks on a real display

Tested here only under a virtual display. On a real machine, check:
- drag-to-pan feels right;
- HiDPI scaling (macOS Retina) maps clicks to the right cell;
- all keys work (`K` checkpoint, `H`/`C`/`G` live toggles).

## 8. Checkpoints on CUDA

`tests/test_soup.py::test_checkpoint_resume_is_exact` proves exact resume on CPU. On CUDA, run a world for 1000 ticks, save at 500, resume, and compare. GPU kernels may be non-deterministic (atomics in `scatter_reduce`), so expect *statistical* rather than bit-exact agreement. Document which one holds.
