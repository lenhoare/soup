# Competition assays — what does damage really cost, and is evolution adaptive?

*Follow-up to `heat_options.md`, where the isolated-growth damage assay was too blunt to explain anything. All runs use the static mid-heat setting (`base_flip 0.03`, `heat_drift 0`), Tier S config, cloud CPU.*

## 1. Arena assay (`experiments/competition.py`)

The world is split into walled 31×31 arenas (`World.barrier`: nothing is born into a wall, and morphogens stop there). Each arena gets one parent founder and one mutant founder, and each pair runs twice with sides swapped, to cancel position and light luck. Heat and gene transfer are off during the assay. Score: the mutant's share of the arena's live cells after 384 ticks (0.5 = neutral). 128 pairs per test; ± is one standard error.

- **Null control:** a parent vs an identical copy sets the noise floor: 0.49–0.50 ± 0.011–0.016.
- **Positive control:** the parent vs a *random* genome. A real fitness advantage should push the random genome's share well below 0.5.

| parents | random genome's share (+ctrl) | 1 flip | 3 flips | 1 flip, expressed | 1 flip, junk | flip the STOP |
|---|---|---|---|---|---|---|
| **prescreened**: best 128 of 1024 random genomes by isolated growth | **0.366 ± 0.021** | 0.483 ± 0.012 | **0.462 ± 0.013** | | | |
| most abundant after 4096 ticks | 0.527 ± 0.027 | 0.478 ± 0.018 | 0.490 ± 0.021 | | | |
| deepest lineages (10.7 generations, no founders) | 0.513 ± 0.027 | 0.512 ± 0.020 | 0.498 ± 0.022 | | | |
| most abundant, evolved with STOP (21 of 37 tokens expressed) | 0.569 ± 0.029 | 0.496 ± 0.018 | 0.482 ± 0.021 | 0.468 ± 0.020 | 0.481 ± 0.014 | 0.524 ± 0.031 |

What it says:

1. **The assay works.** Prescreened good growers beat random genomes decisively (random's share 0.37, about 6 standard errors from neutral).
2. **Damage does hurt genomes that are actually good, and the cost scales with dose:** −1.7% share for one flip, −3.8% for three (3 standard errors).
3. **The soup's evolved genomes are not better than random ones.** The most abundant genomes, and lineages ten generations deep, both tie with random genomes in the arena. That is why damage looked neutral in every earlier test: there was little to lose.
4. For STOP genomes, flips in expressed tokens cost the most (−3.2%). Junk flips sit at the noise floor, as they must by construction. Nothing clears the noise convincingly with n = 128, so the junk-sponge idea is still unproven.

## 2. In-soup competition (`experiments/soup_competition.py`)

An arena tests colonising empty space. The soup is crowded, hot and swapping genes, and evolved genomes might be adapted to *that*. The classic test, as in Lenski's long-term evolution experiment, is competition in the evolution environment itself. A full 128×128 soup (all features on) is seeded 50/50 with founders from set A and set B at shuffled positions, and scored by the share of live cells descending from A after 2048 ticks. 64 founders per side, 4 seeds.

### Is evolution adaptive?

| A vs B | A's share after 2048 ticks | per seed |
|---|---|---|
| **evolved (deepest 64 lineages, mean depth 40.5) vs random** | **0.796 ± 0.016** | 0.80 0.83 0.76 0.79 |
| prescreened good growers vs random | 0.594 ± 0.051 | 0.52 0.60 0.52 0.74 |
| random vs random (null) | 0.443 ± 0.037 | 0.45 0.47 0.34 0.51 |

**Yes, strongly.** Evolved genomes take about 80% of the soup in every seed, well outside the null's range (0.34–0.51). Genomes chosen for fast growth into empty space have only a modest edge here.

So the soup selects for something other than colonising empty space, plausibly holding territory under crowding, attack and turnover. The arena assay simply measured the wrong kind of fitness. **This corrects §1, point 3:** the evolved genomes *are* better, but only in the environment they evolved in.

### What does heat damage cost in the soup?

A = mutants of the same 64 evolved parents, B = the parents themselves.

| A vs B | A's share | per seed |
|---|---|---|
| 1-flip mutants vs parents (all features on) | 0.516 ± 0.025 | 0.56 0.51 0.45 0.55 |
| 3-flip mutants vs parents (all features on) | 0.471 ± 0.026 | 0.45 0.54 0.49 0.42 |
| copies vs parents, null (all features on) | 0.548 ± 0.031 | 0.59 0.50 0.61 0.49 |
| **3-flip mutants vs parents (heat + HGT off during contest, 8 seeds)** | **0.528 ± 0.016** | 0.55 0.53 0.57 0.44 0.50 0.58 0.53 0.53 |
| **copies vs parents, null (same, 8 seeds)** | **0.486 ± 0.016** | 0.55 0.46 0.54 0.44 0.43 0.52 0.46 0.48 |

**Even in the soup, and even for clearly adapted genomes, a few heat flips cost nothing measurable.** The clean 8-seed test puts 3-flip mutants at +0.04 *above* the null, which is about 2 standard errors, so "no cost" rather than "benefit".

### Dose-response: when does accumulated damage start to hurt?

Same 64 evolved parents (mean depth 40.5, mean non-pad length 27.5 vs 32 for random founders). Heat and HGT off during the contest, 4 seeds. Flips are capped at each genome's non-pad length, so 30 flips ≈ every token changed while keeping the parent's length and pad layout.

| A vs parents | A's share | per seed |
|---|---|---|
| copies (null) | 0.501 ± 0.027 | 0.55 0.46 0.54 0.44 |
| 10-flip mutants | 0.488 ± 0.020 | 0.47 0.48 0.45 0.55 |
| 20-flip mutants | 0.494 ± 0.040 | 0.47 0.43 0.61 0.47 |
| **30-flip mutants** | **0.327 ± 0.018** | 0.32 0.28 0.34 0.36 |
| **random genome** | **0.167 ± 0.019** | 0.11 0.20 0.19 0.17 |

- **There is a threshold, not a gradual cost.** Changing up to 20 of ~28 tokens costs nothing measurable. Changing nearly all of them costs a third of the territory.
- **Part of the advantage isn't in token identity.** Even with nearly every token changed, mutants keep the parent's *length and pad layout*, and they still do far better than a random genome (0.33 vs 0.17). The evolved genomes are about 4.5 tokens shorter than random founders, which saves token-cost energy. Roughly half the evolved advantage could be "be short / this layout", and half lives in the tokens.
- **The token-identity half is remarkably robust:** about one third of the original tokens is enough to keep it. Possible reasons: the ribosome's mean pooling spreads each token's influence thin, evolved genomes are mutationally robust, or both. `tests.md` #1 and #1b separate these.

## Overall conclusions

1. **The soup is adaptive.** Lineages 40 generations deep take about 80% of a mixed soup from random genomes.
2. **That adaptation is robust to small damage.** 1–3 random token flips don't dent it, in arenas or in the soup. The advantage is spread across the genome, and the smooth genotype→phenotype map (~12% weight change per token) means a few flips barely move the phenotype. Evolved genomes may also have become *mutationally robust* ("survival of the flattest"). Testing that means comparing the flip sensitivity of evolved and random genomes.
3. **This is why fidelity is weakly selected.** Individual flips are nearly free. Fidelity still responds to heat zones and to power³ (`heat_options.md`), which suggests the real cost is the *accumulation* of many flips over a lineage's lifetime, not any single flip.
4. **Assays have to match the environment.** Isolated growth and empty arenas measure colonisation ability. That correlates poorly with soup fitness (prescreened growers win the arena 0.63 : 0.37, but the soup only 0.59 : 0.41).

## Next experiments

See `tests.md`. In short:
- **Robustness test:** the flip sensitivity of evolved vs random genomes of similar fitness (survival of the flattest).
- **More seeds on the GPU.** The soup assay's noise floor at 4–8 seeds is about ±0.02–0.03.
