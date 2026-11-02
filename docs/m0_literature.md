# M0 — Literature check

*Done 2026-11-02 using web searches (abstracts, summaries and project pages, not full-text reads). Treat it as a map of what to read, not a finished review.*

## Verdict

**No single prior system found that combines all of:** a discrete, spliceable genome string → a *learned/neural translator* (hypernetwork) → a per-cell neural controller in a spatial, multi-organism world, where the environment damages the string and neighbours exchange string fragments.

Each ingredient on its own **does** have clear prior work, so the honest claim is about the *combination*, plus the specific question in our ablation grid: does a string + neural translator give evolvability that direct weight mutation does not?

## Closest prior work

| Work | What it shares with us | What differs |
|------|------------------------|--------------|
| **Coralai** (Barbieux & Canaan, 2024, [arXiv 2406.09654](https://arxiv.org/abs/2406.09654)) | Ecosystem of embodied NCA organisms on a GPU grid; a per-cell genome-index channel points to that organism's network parameters; local survival/merge/mutation; reports competition, resource cycles, symbiosis. | Genome is a **CPPN** (HyperNEAT-style), not a token string; indirect encoding is graph-based, not a shared learned translator; no environmental damage to the genome; "merging" ≠ fragment-level HGT. **This is the system to compare against most carefully.** |
| **Self-Replicating NCA** (Jain, Reimers, Nichele, 2026, [arXiv 2609.19902](https://arxiv.org/abs/2609.19902); J. Imaging 12(10):496) | Every pixel carries a tiny network (~44 params); founders grow into coexisting/competing species; lineage colouring by genome hash; diversity metrics. | **Direct weight mutation** — this is essentially our *control* condition. Useful as a baseline and for its six diversity metrics. |
| **HyperNCA** (Najarro et al., 2022, [arXiv 2204.11674](https://arxiv.org/abs/2204.11674)) | An NCA *grows* the weights of a policy network: an indirect, developmental genotype → weights map. | Single-agent RL, fitness-driven ES; no world, no strings, no HGT. |
| **Flow-Lenia** (Plantec et al., 2023, [arXiv 2212.07906](https://arxiv.org/abs/2212.07906)) | Update-rule parameters are *localized* and carried by matter, so multiple species with different rules share one world and their parameters **mix** at boundaries. Mass conservation keeps creatures localized. | Continuous parameter mixing, not discrete string splicing; no neural translator. Their note that runs converge to fast-expanding dominators is a warning for our monoculture risk. |
| **Computational Life / BFF** (Agüera y Arcas et al., 2024, [arXiv 2406.19108](https://arxiv.org/abs/2406.19108)) | Soup of 64-byte programs that interact by concatenation; self-replicators emerge spontaneously; 2D grid variant. | Programs are executed directly (no translator, no body); replication is the emergent phenomenon rather than built in. Same genome size as ours by coincidence — a nice point of comparison for compression metrics. |
| **Avida — survival of the flattest** (Wilke et al., *Nature* 2001, [doi:10.1038/35085569](https://doi.org/10.1038/35085569)) | High mutation rates select for mutationally robust (flatter) genotypes over faster replicators. | This is the established prediction for our **heat-on-code** feature: hot regions should favour robustness. We should cite it and test for it rather than claim it as new. |
| **HGT in Avida** (MSU dissertation, [d.lib.msu.edu/etd/6835](https://d.lib.msu.edu/etd/6835/FULL_TEXT/view)) | Horizontal gene transfer in digital organisms. | Only the table of contents was visible — **needs a full read** before we claim anything about HGT novelty. |
| **AutoMap** (Moreno, Banzhaf, Ofria, GECCO 2018, [devolab](https://devolab.org/learning-an-evolvable-genotype-phenotype-mapping/)) | A *learned neural* genotype → phenotype map used to improve evolvability. | Map is trained (autoencoder) and applied to toy problems; ours is frozen-random in Phase 1. Relevant to Phase 2 "evolve/train the translator". |
| **Self-Referential Graph HyperNetworks** ([arXiv 2512.16406](https://arxiv.org/abs/2512.16406)) | Hypernetworks that carry their own variation/inheritance. | Not spatial or ecological. |
| **Review: Self-reproduction & evolution in CA, 25 years after Evoloops** ([arXiv 2402.03961](https://arxiv.org/abs/2402.03961)) | Survey of the field, incl. multi-species CA with mutating per-location parameters. | Read for background and missed references. |

## What this changes in the spec

1. **Weaken "why this might be new"** to the combination claim, and name Coralai and Self-Replicating NCA as the nearest neighbours.
2. **The direct-weight control** (section 8.1) is basically the Self-Replicating NCA setup — reuse its diversity metrics so results are comparable.
3. **Heat on code** has a known expected outcome (survival of the flattest). Our contribution is *how* robustness is achieved (evolved `fidelity` vs. duplicated segments), not *that* it appears.
4. **Monoculture risk** is confirmed by the Lenia-evolution work; keep the niche/heat mitigations as priorities.

## Still to read (full text)

- [ ] Coralai paper in full (their genome and merge operator; their GPU throughput numbers)
- [ ] The MSU HGT-in-Avida dissertation
- [ ] arXiv 2609.19902 diversity metrics, to copy into `metrics.py`
- [ ] Search specifically for "horizontal gene transfer" + "neural cellular automata" and for Tierra-style parasites in spatial NCA worlds
