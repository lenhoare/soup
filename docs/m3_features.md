# M3 — The three features and the ablation grid

*Cloud CPU. Ablation: 128×128 grid, 64 random founders, 8192 ticks, 2 seeds per condition. Means over the last quarter of each run, averaged over seeds. Reproduce:*

```bash
python -m experiments.ablation --out runs/ablation --ticks 8192 --seeds 1,2 --procs 4
python -m experiments.plot_ablation runs/ablation docs/m3_ablation.png
```

## What was built

- **Heat on code** (`World.heat_damage`, every epoch). Each non-pad token flips with `p = base_flip · mean heat of the organism's cells · (1 − fidelity)`. A flip changes the token to any other symbol, pad included, so heat can also delete. Flipped organisms are re-translated at the end of the epoch.
- **Gene transfer** (`World.gene_transfer`). Up to 4096 random border pairs are sampled per epoch. With probability `p_hgt · permeability(recipient)`, a 2–8 token window that starts at a non-pad donor token is copied into the recipient at a random position. `has_hgt` marks the recipient and is inherited by its spores.
- **Token cost** already existed. When switched off, cells now pay a **flat** cost equal to a 32-token genome (see "Confound" below).
- **Direct-weight control** (`mode: direct`). Founders are translated from strings as usual, so both conditions start alike. After that, weights evolve directly:
  - **Spore mutation:** matched to string mode. 34% of spores change; when they do, the relative weight change is 0.144 and trait shifts have sd 0.133 in logit units. All three were measured at α = 6.
  - **Heat:** each "flip" is a Gaussian kick the size of a single-token change.
  - **Gene transfer:** the matching block of parameters is copied from the donor.
- **New metrics:** `hgt_frac`, `fid_heat_corr` (cell-level correlation of fidelity with local heat), `mean_org_size`, plus heat-flip and HGT counters.

## Calibration done along the way

| Issue | Fix |
|---|---|
| **Bug:** founders started with zero energy and all-zero weights until tick 32 | Fixed; M2 α results re-checked and still hold (see `m2_calibration.md`) |
| Heat flipped pads → mostly *inserted* code, mean length 32 → 45 | Heat only hits non-pad tokens |
| `p_hgt 0.05`: ~100% of genomes carried a splice within 1k ticks | `p_hgt 0.005` |
| Attack churn about 1.2% of cells per tick, ragged bodies | `attack_min 0.3` (attacker must have energy > 0.3), which halves conquests |
| `base_flip 2e-4` (spec) gives about 1 flip per 30k ticks | `6e-3`, about 1 flip per ~300 ticks for a typical organism |

### Confound found in the first ablation pass

With "cost off" meaning *zero* cost, every cost-off run saturated the grid: 99% occupancy, 10–24 giant organisms. The token cost is the largest energy sink, so removing it changed the whole energy economy, not just the length pressure. The four cost-off conditions were re-run with a flat cost. Everything below uses those runs.

## Results

| condition | organisms | Shannon | mean depth | mean len | hgt_frac | mean fidelity | corr(fid, heat) |
|---|---|---|---|---|---|---|---|
| baseline | 159 | 2.68 | 3.3 | 31.3 | 0 | 0.680 | −0.01 |
| +heat | 260 | 2.90 | 1.7 | 30.8 | 0 | 0.678 | −0.02 |
| +cost | 369 | 3.03 | 5.2 | 23.0 | 0 | 0.652 | 0.02 |
| +hgt | 177 | 3.01 | 2.2 | **56.0** | 1.0 | 0.651 | −0.02 |
| heat+cost | 221 | 2.85 | 5.7 | 26.5 | 0 | 0.678 | 0.00 |
| heat+hgt | 361 | 3.40 | 2.2 | **54.6** | 1.0 | 0.641 | 0.02 |
| cost+hgt | **844** | **5.06** | 13.8 | 25.8 | 0.97 | 0.587 | 0.00 |
| **full** | **759** | **4.87** | **27.1** | 26.2 | 1.0 | 0.619 | 0.03 |
| **control** (direct weights, all features) | 659 | 3.92 | 3.7 | 32 (fixed) | 1.0 | 0.644 | 0.01 |

Per seed (seed 1 / seed 2), to show how robust each result is:

| condition | organisms | Shannon | mean depth |
|---|---|---|---|
| cost+hgt | 896 / 793 | 5.14 / 4.98 | 14.4 / 13.2 |
| full | 607 / 910 | 4.71 / 5.03 | 10.1 / 44.1 |
| control | 377 / 942 | 3.55 / 4.28 | 2.8 / 4.6 |
| everything else | 108–498 | 2.4–3.6 | 1.4–8.7 |

![Ablation time series](m3_ablation.png)

## What it says (2 seeds — read as leads, not conclusions)

1. **Gene transfer + token cost together are the standout.**
   - Only the two conditions with both (`cost+hgt` and `full`) hold about 800 organisms and Shannon ≈ 5, in both seeds.
   - Every other condition sits at Shannon 2.4–3.6.
   - Neither feature does this alone. HGT without cost bloats genomes to ~55 tokens and gives little diversity. Cost without HGT shortens genomes (23 tokens) with modest diversity.
   - This interaction was not predicted by the spec and is the most interesting thing so far.
2. **Token cost works as designed.** Genomes shrink (31 → 23–26 tokens) and lineages go deeper. Without cost, gene transfer only ever adds material.
3. **Heat on code shows no adaptive signature yet.** Fidelity does not track local heat in any condition (|r| < 0.1, fluctuating sign), and mean fidelity does not rise. Possible reasons:
   - damage is still too rare to matter at this timescale;
   - fidelity's continuous energy cost outweighs its benefit;
   - the heat field drifts faster than lineages can adapt.

   **Next step:** a heat-strength sweep, plus a static heat field.
4. **String + ribosome vs. direct weights (the spec's key question).**
   - Full beats control on diversity in both seeds (Shannon 4.71/5.03 vs 3.55/4.28).
   - Lineages run far deeper in string mode (10–44 vs 3–5 generations).
   - Organism counts overlap between seeds.
   - The depth gap is large and not yet explained. Before claiming it, check whether control spores simply die younger (lifetime and spore-survival metrics).

## Caveats

- Two seeds at 128×128 is small. Seed-to-seed spread is large for some conditions (full's depth 10 vs 44).
- In control, `mean_len` is fixed by construction and `zlib_ratio` is meaningless (genomes are not used after founding).
- `hgt_frac` is an inherited flag, so it saturates even at low transfer rates. Measuring the fraction of *tokens* that came from splices would say more.
- Cloud CPU speed limited the run length. A 3060 should make 4–8 seeds at 256×256 cheap.
