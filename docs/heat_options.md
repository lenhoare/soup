# Heat options — making heat on code matter

*Follow-up to `m3_features.md`, which found that fidelity never adapted locally (hot vs. cold regions) because heat damage was nearly neutral. Four ideas were each built as an independent switch, off by default. Each was tested alone on a static mid-heat baseline (`base_flip 0.03`, heat field not drifting), then in combination. Setup: 128×128, 8192 ticks, 2 seeds.*

```bash
python -m experiments.heat_options --out runs/heat_options --ticks 8192 --seeds 1,2
```

## The switches

| switch | config | what it does |
|---|---|---|
| **stop** | `ribosome.stop_token: 15` | Translation ignores the first STOP token and everything after it. A flip that creates an early STOP truncates the genome, like a nonsense mutation. Untranslated tokens still cost maintenance energy. Founders never contain STOP. The inspector shows the STOP as a white tile and fades the untranslated tail. |
| **power** | `heat.fid_power: 3` | Damage ∝ `(1 − fidelity)^3` instead of `(1 − fidelity)`. Low- and high-fidelity organisms then differ ~16× in damage instead of ~2.5×. Tested with `base_flip` ×8 so a typical organism (fidelity ≈ 0.65) takes the same damage as baseline. |
| **zones** | `fields.heat_max_wavenumber: 1`, `heat_contrast: 4` | Heat field with few, large, sharp-edged zones (z-scored, then a sigmoid). 68% of cells are clearly hot (> 0.8) or cold (< 0.2), vs 3% before. |
| **direct** | `heat.state_noise: 0.1`, `heat.energy_cost: 0.01` | Heat jostles each cell's hidden state and costs energy, so hot zones are harsh regardless of genome. |

## Results

"Slope" is the hot-quartile minus cold-quartile fidelity, divided by the hot-minus-cold heat gap. It is the local adaptation per unit of heat difference, and it makes zones comparable with the rest (zones roughly double the heat gap by construction). "Junk" is the share of non-pad tokens after the first STOP.

| condition | Δ mean fidelity | hot − cold fidelity | **slope** (seed 1 / 2) | length | expressed | junk | organisms |
|---|---|---|---|---|---|---|---|
| baseline | −0.000 | +0.009 | +0.022 (+0.036 / +0.009) | 32.2 | 32.2 | 0% | 545 |
| **zones** | −0.033 | **+0.126** | **+0.129 (+0.130 / +0.128)** | 23.9 | 23.9 | 0% | 737 |
| **power** | +0.034 | +0.037 | **+0.096 (+0.095 / +0.096)** | 25.1 | 25.1 | 0% | 546 |
| direct | +0.019 | +0.020 | +0.051 (+0.071 / +0.031) | 26.3 | 26.3 | 0% | 695 |
| stop | +0.016 | −0.005 | −0.014 (−0.043 / +0.014) | 27.9 | **11.0** | **61%** | 705 |
| all four | +0.001 | +0.009 | +0.010 (+0.003 / +0.016) | 30.4 | 13.1 | 57% | 608 |
| all but stop | +0.019 | +0.057 | +0.059 (+0.063 / +0.055) | 28.2 | 28.2 | 0% | 596 |
| stop, heat off | −0.016 | +0.004 | +0.011 (−0.007 / +0.029) | 33.6 | 33.0 | 2% | 597 |

## What it says

1. **Heat zones produce clear local adaptation.**
   - Organisms in hot zones carry 0.13 more fidelity than those in cold zones, and both seeds agree to the third decimal.
   - Mean fidelity *falls* overall (−0.033), because cold-zone lineages shed costly fidelity they don't need.
   - So fidelity is now tracking the local environment in both directions.
2. **A steeper fidelity lever works too.** Power³ gives about 4× the baseline slope, also with very tight seeds.
3. **Direct heat effects help a little** (slope 0.05), and they raise diversity (Shannon 4.96, the highest here).
4. **STOP tokens block local adaptation, and produce junk DNA.**
   - Under heat, genomes accumulate STOPs: only 11 of 28 tokens are translated, and **61% is untranslated junk**.
   - Without heat, junk stays at 2%, so heat-made STOPs are what create it.
   - Once most of a genome is junk, most flips land there and do nothing, so fidelity stops mattering.
   - Combining STOP with everything else cancels the zone and power effects (slope 0.010).
5. **Combining the other three helps less than zones alone** (0.059 vs 0.129). Power³ and the direct effects seem to dilute the zone signal; the mechanism isn't clear yet.

### Damage cost with and without STOP

`experiments/damage_cost.py --over '{"heat": {"base_flip": 0.03}, "fields": {"heat_drift": 0.0}, ...}'`, evolved for 4096 ticks.

| evolved with | expressed / non-pad tokens | Δ log size, 1 flip | Δ log size, 3 flips |
|---|---|---|---|
| no STOP | 37 / 37 | −0.06 ± 0.06 | +0.02 ± 0.09 |
| STOP | 21 / 37 | +0.06 ± 0.07 | +0.04 ± 0.10 |

Flips are neutral within noise in both cases, so this **does not** distinguish "junk soaks up damage" from the alternatives. The isolated-growth assay is too insensitive: evolved genomes don't even out-grow random ones in it (3.23–3.36 vs 3.38–3.60), so it isn't measuring the competitive fitness they evolved for. **A head-to-head competition assay is the next tool to build** (mutant vs. parent founders side by side, scored by share of territory). Until then the junk-sponge explanation is a hypothesis.

## Recommendation

- **Turn heat zones on by default** for future heat experiments. They give the effect the spec was looking for, cleanly and reproducibly.
- **Keep power³ as an option.** It's the "make repair matter" lever, and it works on its own.
- **Keep STOP off for fidelity studies.** But STOP + heat → junk DNA is an emergent phenomenon worth its own study: is the junk adaptive buffering, or just tolerated damage?
- These are still 2 seeds on a 128×128 grid. The zones and power results replicate tightly, but more seeds on the GPU would firm them up.
