# Ribosome Soup — Initial Spec (v0.1)

*Working title. An artificial-life system where organisms are grown from short mutable code strings, translated by a shared neural "ribosome", living in a noisy world that damages their code. Sized and designed for a single RTX 3060 (12 GB).*

---

## 1. Idea in one paragraph

Each organism carries a **genome: a short discrete string** (64 tokens from a 16-symbol alphabet). A **shared neural net, the ribosome**, translates that string into the weights of a tiny **cell network**. Every cell of the organism runs that cell network on a 2D torus grid, sensing neighbours, emitting chemical signals, growing, fighting and reproducing. Strings are what evolve: they mutate, get copied, get spliced between neighbours, and get **corrupted by heat**. Each token costs energy to maintain, so evolution is pushed toward short, reusable, error-tolerant code.

The three features we are switching on (all from the design discussion):

| # | Feature | What it does | What pressure it creates |
|---|---------|--------------|--------------------------|
| 1 | **Transplantable genes** | Neighbouring organisms can copy fragments of string into each other (horizontal gene transfer) | Symbiosis, parasites, modular reusable "genes" |
| 2 | **Heat acts on the code** | Thermal noise flips tokens in the string, not just positions | Redundancy, repair, robustness |
| 3 | **Cost per token** | Maintenance energy scales with non-pad genome length | Compression, reuse, hierarchy |

**Why this might be new (revised after M0, see `docs/m0_literature.md`):** each ingredient has prior work. Coralai (2024) runs ecosystems of NCA organisms whose genomes (CPPNs) are indirectly encoded. Self-Replicating NCA (2026) evolves per-pixel networks by direct weight mutation, which is essentially our control. Flow-Lenia mixes localized rule parameters between species. Avida's "survival of the flattest" already predicts that high mutation favours robustness. We found no system that **combines** a discrete, spliceable token genome, a shared neural translator, a spatial multi-organism body, in-world damage to the string, and fragment-level gene transfer. The claim is about the combination, and the key test is the direct-weight control in section 8.1.

---

## 2. Design goals

1. **Runs well on a 3060.** Sizes, precision and kernel design are chosen for 12 GB VRAM and ~360 GB/s bandwidth. No autograd anywhere: the whole system is inference-only.
2. **More than a coloured pixel per network.** Each cell net has a real action space (grow, spore, attack, rest, emit signals, steer direction), and the genome is code, not weights.
3. **Looks beautiful.** Visuals are a first-class deliverable, not an afterthought (section 9).
4. **Scientifically legible.** Every feature can be ablated, and every run logs metrics that tell us whether the idea is doing anything.

Non-goals for v0: open-ended evolution of the ribosome itself, 3D, multi-GPU, gradient-based training of anything.

---

## 3. Architecture

```
  genome string (64 tokens, 16-symbol alphabet, 0 = pad)
        │
        ▼
  ┌───────────────┐   shared, frozen in Phase 1
  │   RIBOSOME    │   2-layer transformer, d=64, ~150k params, fp16
  └───────┬───────┘
          │ translates once per genome (only when dirty)
          ▼
  cell-net weights (~1.2k params) + traits (fidelity, permeability, ...)
          │ cached per genome slot
          ▼
  ┌───────────────────────────────────────────────┐
  │  WORLD: 2D torus grid                         │
  │  every cell: state, morphogens, energy, gid   │
  │  each tick, every cell runs ITS genome's net  │
  └───────────────────────────────────────────────┘
          │
          ▼
  every 32 ticks ("epoch"): heat damage → HGT → spore mutation → re-translate dirty → free dead slots
```

### 3.1 Genome

- 64 tokens, alphabet size 16, stored as `uint8` (4 MB for 65k genomes; bit-packing is possible later and unnecessary now).
- Token `0` is **PAD**. Effective length `L = count(non-pad)`. Pad can appear anywhere, so deletions/insertions are expressible without shifting the array.
- Mutation operators (applied when a spore is created):
  - point substitution (`p_sub ≈ 0.01` per token)
  - small insertion/deletion (`p_indel ≈ 0.003`)
  - segment duplication (`p_dup ≈ 0.002`, copies a 2–8 token window to another position, overwriting pads or tokens)
  - segment deletion (`p_del ≈ 0.002`, pads out a window)

### 3.2 Ribosome (the translator)

- Input: token embeddings (16×64) + learned positional embeddings (64×64).
- 2 transformer layers, 4 heads, d=64, MLP ratio 2. Mean-pool (masking pads) → linear heads.
- Outputs per genome:
  - **cell-net weight delta** (~1.2k values)
  - **traits**: `fidelity`, `permeability`, `aggression`, `spore_threshold` (each squashed to [0,1])
  - a **genome embedding** (8-d), used only for colouring and species analysis.
- **Residual parameterisation:** `W = W_base + α · ΔW(genome)`, where `W_base` is a shared, hand-initialised "reasonable CA" net. This keeps most random strings viable, so the world isn't dead on arrival. `α` is a key hyperparameter (start 0.5).
- **Phase 1: the ribosome is frozen** at a fixed random initialisation (seeded). The "genetic code" is an arbitrary but fixed landscape. Evolution explores string space only. This is the cleanest first experiment.
- Cost: ~6M MACs per genome. Translating all 65k genomes ≈ 0.8 TFLOP ≈ tens of milliseconds, but we only translate **dirty** genomes per epoch (hundreds to a few thousand).

### 3.3 Cell network

Per cell, per tick:

- **Perception:** 12 channels (8 hidden state + 4 morphogens) × 4 fixed depthwise 3×3 filters (identity, Sobel-x, Sobel-y, Laplacian) = 48 values.
- **Extra inputs (4):** own energy, local light, fraction of 8 neighbours sharing my genome id ("kin fraction"), local heat.
- **MLP:** 52 → 16 (tanh) → 20. About 1.2k params (≈2.4 KB in fp16).
- **Outputs (20):**
  - 8 × state delta
  - 4 × morphogen emission
  - 4 × action logits: `grow`, `spore`, `attack`, `rest`
  - 4 × direction logits (N/E/S/W) for the chosen action

### 3.4 World state (per cell)

| Field | Type | Notes |
|-------|------|-------|
| `gid` | uint32 | genome slot id, `0xFFFFFFFF` = empty |
| `state` | 8 × fp16 | hidden state (the "RNN" memory) |
| `morph` | 4 × fp16 | diffusing chemical fields, decay + diffuse each tick |
| `energy` | fp16 | clamped to [0, 1] |

Environment fields (not per genome):
- `light(x,y)`: patchy, slowly drifting noise field. Energy source.
- `heat(x,y)`: smooth, slowly evolving temperature field. Drives code damage. Hot and cold niches.

### 3.5 Actions and energy

- **Photosynthesis:** `+ light · (1 − crowding)` per tick when not attacking.
- **Costs per cell per tick:** basal metabolism + `c_tok · L` (every cell carries and "expresses" a genome copy) + `c_fid · fidelity` (repair is not free).
- **grow:** if energy > threshold and target neighbour is empty, create a cell with the **same gid**, splitting energy. Same gid means the organism grows as one clonal body.
- **spore:** if energy > `spore_threshold`, place a founder cell in the target neighbour with a **new slot** holding a mutated copy of the genome. This is organism-level reproduction.
- **attack:** if target belongs to a different gid, contest by energy × (1 + aggression); winner takes the cell and a share of the loser's energy.
- **rest:** save energy, no action.
- Cells with energy 0 die. A slot whose cell count reaches 0 is freed.

---

## 4. The three features in detail

### 4.1 Horizontal gene transfer (transplantable genes)

Each epoch, sample up to `S = 4096` border pairs (adjacent cells with different gids). For each pair, with probability `p_hgt · permeability(recipient)`, copy a window of `k ∈ [2, 8]` tokens from the donor string into the recipient string at a random position. Mark recipient dirty.

Why strings make this work: a token window means something regardless of where it lands, because the ribosome attends over the whole string. Splicing weights directly would be meaningless.

Optional v0.2 variant: **injection** — a successful `attack` also pushes a fragment into the loser, making viruses/parasites possible.

### 4.2 Heat on the code

Each epoch, for each live slot: `p_flip = base_flip · mean_heat(slot's cells) · (1 − fidelity)`. Each of the 64 tokens flips to a random token with that probability. Mark dirty if anything flipped. Damage hits the whole organism at once (all cells share the slot), mid-life, so the body's behaviour changes under its feet.

`fidelity` is an evolvable trait that reduces damage but costs energy continuously, giving an explicit repair mechanism. Redundancy (duplicated segments) is the implicit one, and we want to see which route evolution prefers.

### 4.3 Cost per token

`c_tok · L` per cell per tick, with `L` the count of non-pad tokens. Calibrate `c_tok` so a 64-token genome costs roughly 25–40% of a typical cell's photosynthetic income; short genomes then have real advantages but are not mandatory. Make `c_tok` a schedule we can ramp up during a run to push compression harder.

---

## 5. Sizes and budget for the RTX 3060 (12 GB)

Hardware assumptions: ~12.7 TFLOPS fp32 peak, ~360 GB/s memory bandwidth, 28 SMs, small L2. All throughput numbers below are **estimates to be replaced by M1 benchmarks**.

### 5.1 Tiers

| Tier | Grid | Cells | Genome slots | Purpose | Target speed |
|------|------|-------|--------------|---------|--------------|
| **S** | 256 × 256 | 65k | 8,192 | dev loop, PyTorch eager | ≥ 60 ticks/s |
| **M** | 512 × 512 | 262k | 32,768 | default experiments, fused kernel | ≥ 200 ticks/s |
| **L** | 1024 × 1024 | 1.05M | 65,536 | showpiece, "1M" runs | ≥ 60 ticks/s (stretch 200) |

### 5.2 Memory (Tier L)

| Item | Size |
|------|------|
| World state, double-buffered (~32 B/cell × 2) | ~64 MB |
| Genome strings (65k × 64 B) | 4 MB |
| Cached cell-net weights (65k × 1.2k × 2 B) | ~160 MB |
| Ribosome (~150k params, fp16) | 0.3 MB |
| Translation batch activations (8k dirty genomes) | ~70 MB |
| Render buffers (RGBA8 + post-process) | ~50 MB |
| **Total** | **< 0.5 GB** |

Memory is nowhere near binding. The headroom means a later upgrade to a bigger cell net (hidden 32, ~4× params) or a bigger alphabet is affordable in VRAM. The real limits are per-tick compute and weight-fetch locality.

### 5.3 Compute per tick (Tier L)

- Perception ≈ 12 ch × 3 filters × 9 taps ≈ 330 MACs/cell
- Cell MLP ≈ 1.2k MACs/cell
- Total ≈ 1.5k MACs ≈ 3 kFLOP/cell → **~3 GFLOP/tick** at 1M cells
- At ~20% utilisation (~2.5 TFLOPS) ≈ 1.2 ms compute; memory traffic ~100–200 MB ≈ 0.3–0.6 ms
- So an ideal fused kernel lands in the low single-digit ms per tick, i.e. hundreds of ticks/s. PyTorch eager will be roughly 10–20× slower, which is fine for Tiers S and M.

### 5.4 The one real kernel risk: weight locality

If every cell fetched its own 2.4 KB of weights from DRAM, that's ~2.4 GB/tick at Tier L (~7 ms, and the dominant cost). Mitigation:

- Process the grid in **16 × 16 tiles**. A tile typically touches only a handful of distinct genomes (organisms are spatially clustered), so load each distinct weight set **once into shared memory** and reuse it across the tile's cells.
- Keep weights fp16. Accumulate in fp32.
- If tiles are too diverse (a very mixed soup), fall back to per-cell loads and accept slower ticks. We log the average distinct-genomes-per-tile to know.

---

## 6. Simulation loop

```
tick (every step):
    perceive → cell net → apply state/morphogen deltas
    resolve actions (grow / spore / attack / rest) with conflict resolution
    diffuse + decay morphogens, update light/heat fields
    apply energy income/costs, kill zero-energy cells

epoch (every 32 ticks):
    heat damage to genome strings
    HGT sampling on border pairs
    translate dirty genomes (batched ribosome pass) → refresh cached weights
    free empty slots (refcount scatter), recycle ids
    log metrics
```

Conflict resolution for two cells targeting the same empty neighbour: random priority from a per-tick hash, so the loop stays parallel and has no atomics-dependent ordering. Perfect determinism is **not** a goal; we use checkpoints for replay instead (section 9.5).

---

## 7. Phased plan

**Phase 1 (this spec):** frozen random ribosome, all three features on, genome strings evolve.

**Phase 2 (later options):**
- *Dialects:* K = 4 ribosomes assigned to regions; migrants face a different genetic code.
- *Self-editing strings:* some tokens act as instructions that rewrite the string (evolving mutation operators).
- *Evolving the translator only:* fix a set of strings and let the ribosome drift.

---

## 8. Experiments and metrics

### 8.1 Ablations (2³ factorial + control)

| Run | Heat on code | Token cost | HGT |
|-----|:---:|:---:|:---:|
| baseline | – | – | – |
| +heat | ✓ | – | – |
| +cost | – | ✓ | – |
| +HGT | – | – | ✓ |
| pairs ×3 | … | … | … |
| **full** | ✓ | ✓ | ✓ |
| **control: direct-weight evolution** | cell-net weights mutated directly, no ribosome/strings | | |

The control answers the key question: does the string + ribosome indirection give anything weight-mutation doesn't?

### 8.2 Metrics (logged per epoch)

- live population (cells, organisms), number of distinct genomes, Shannon diversity
- mean/median effective length `L`, and zlib compression ratio of the genome pool
- HGT events per epoch and fraction of living genomes carrying a spliced fragment
- mean `fidelity`, `permeability`, `aggression`
- cluster size distribution of organisms, mean lifetime, lineage depth
- **phenotype smoothness:** median relative change in cell-net weights for single-token mutants (target roughly 5–15%; much higher means the landscape is too rugged to evolve on)
- average distinct genomes per kernel tile (performance health)

### 8.3 What counts as interesting

- Stable multi-lineage coexistence with visible territory structure
- Genome length that settles well below 64 and compresses well (reuse, repeats)
- Repeated motifs appearing across unrelated lineages (evidence of transferable genes)
- Hot-region lineages with higher fidelity or more redundancy than cold-region ones
- Parasite-like lineages that live off others' spliced material

---

## 9. Visuals ("make it really nice")

### 9.1 Rendering stack

- Simulation on GPU (PyTorch → fused kernel later). A render pass writes an **RGBA8 tensor** on GPU; copy to a GL texture each frame (4 MB at Tier L, trivial).
- Window via `moderngl` + `glfw` (or pygame as a fallback). Post-processing in GLSL: bloom, subtle chromatic aberration, vignette, filmic tonemap.
- Target 60 fps at 1080p with 1 tick/frame at Tier L, or N ticks/frame with rendering decoupled.

### 9.2 Visual language

- **Dark void background** (near-black ink) so tissue glows. Colour work in **OKLCH** for perceptual evenness.
- **Hue = lineage.** Colour comes from the ribosome's 8-d genome embedding via a fixed random projection to (hue, chroma). Similar genomes get similar colours, so speciation shows up as colour gradients and sudden splits are visible events.
- **Lightness = energy.** Rich cells glow, starving cells dim.
- **Interior texture = hidden state.** State channels 0–2 gently modulate hue and lightness inside a body, giving the internal differentiation seen in the reference videos.
- **Membranes.** Where `gid` differs between neighbours, draw a bright rim in a lightened version of the lineage colour (edge detection via a small conv). These are the territory borders.
- **Heat bath shimmer.** Empty cells show faint ember-like speckle with density proportional to local heat: the "thermal bath" made visible, and hot zones read instantly.
- **Event flashes.** Gene transfer produces a gold-white ripple at the splice site (fades over ~30 frames). Heat damage produces a tiny cold-blue sparkle. Spores pop with a short bloom.

### 9.3 View modes (keys 1–6)

1. Lineage (default) 2. Energy 3. Morphogens (4 channels → RGB+alpha) 4. Genome distance from the selected organism 5. Heat/light fields 6. Age/lineage depth

### 9.4 Interaction and HUD

- Mouse zoom/pan (pixel-perfect up to ~40×, with soft interpolation) and a crosshair; click to **inspect an organism**.
- Inspector panel: the genome shown as a **64-tile glyph ribbon** (each token a coloured tile, pads greyed, recently spliced tiles outlined gold), its traits, energy, age, parent slot, and a mini history of its mutations.
- Top-left HUD: zoom, population, ticks/s, fps, diversity, mean genome length.
- Keys: space = pause, `.` = single tick, `[` `]` = tick speed, `h` = toggle heat/cost/HGT live (great for demos), `s` = snapshot.

### 9.5 Capture

- ffmpeg pipe for 1080p/60 H.264 recording, with optional 2× supersampled rendering for 4K.
- Periodic full-state **checkpoints** (cheap: ~100 MB at Tier L). Re-render any interesting moment at high quality from the nearest checkpoint, instead of requiring bit-exact determinism.

---

## 10. Suggested code layout

```
ribosome_soup/
  initial_spec.md
  config/            # YAML presets: tier_s.yaml, tier_m.yaml, tier_l.yaml
  soup/
    genome.py        # strings, mutation, splice, heat
    ribosome.py      # translator net, residual weights, traits
    world.py         # fields, tick loop, actions, energy
    epoch.py         # heat, HGT, translate-dirty, slot GC
    metrics.py       # logging + analysis
    kernels/         # Triton/CUDA fused tick (Phase M6)
  render/
    viewer.py        # moderngl window, view modes, HUD, inspector
    shaders/         # bloom, membrane, tonemap
    capture.py       # ffmpeg + checkpoints
  experiments/       # ablation runners, plots
  tests/
```

Default config block (Tier M):

```yaml
grid: [512, 512]
genome_slots: 32768
genome_len: 64
alphabet: 16
cell_net: {hidden: 16, state_ch: 8, morph_ch: 4}
ribosome: {layers: 2, d_model: 64, heads: 4, alpha: 0.5, frozen: true}
epoch_ticks: 32
mutation: {p_sub: 0.01, p_indel: 0.003, p_dup: 0.002, p_del: 0.002}
heat: {base_flip: 2.0e-4}
hgt: {samples_per_epoch: 4096, p_hgt: 0.05, window: [2, 8]}
energy: {c_tok: 0.0004, c_fid: 0.001, basal: 0.002}
seed: 1
```

(Every constant above is a starting guess to be tuned in M2–M3.)

---

## 11. Milestones

- [x] **M0 — Literature check (½ day).** Done, see `docs/m0_literature.md`. Search for prior work on string-genome → hypernetwork → cellular body systems with in-world code damage and splicing (Tierra, Avida, Lenia/Flow-Lenia, Neural CA, HyperNEAT/HyperNCA, "computational life" self-replicator soups). Update the "why new" claim honestly.
- [x] **M1 — Tier S skeleton (PyTorch).** World, cell net with a hand-set genome, light/heat fields, one benchmark of ticks/s. Done; see README for first numbers.
- [x] **M2 — Evolution.** Spores, mutation, slot allocation/GC, attack. Calibrate for survival and coexistence. Run the viability scan: what fraction of random strings produce a living organism? Done, see `docs/m2_calibration.md`. Note that α = 0.5 (section 3.2) proved far too small; the default is now 6.
- [x] **M3 — Three features.** Heat, token cost, HGT, `fidelity`/`permeability` traits. Run the ablation grid at Tier S. Done, see `docs/m3_features.md`. Changes from this spec: heat only hits non-pad tokens; `base_flip` 6e-3, `p_hgt` 0.005; "cost off" means a flat cost, not zero.
- [x] **M4 — Viewer v1.** Lineage colouring, membranes, heat shimmer, HUD, inspector with genome ribbon. Done, plus parts of M5: bloom/tonemap, event flashes, all six view modes, and headless video via ffmpeg. Checkpoints are still to do.
- [x] **M5 — Polish.** Bloom/tonemap, event flashes, view modes, capture pipeline. Done: the visual items came with M4; checkpoints (exact resume, re-render from any checkpoint) and 2× supersampled video were added here.
- [~] **M6 — Fused kernel.** Tile-based Triton/CUDA tick, hit Tier M target, then Tier L. *In progress:* a Triton kernel groups cells by genome (blocks of 32), so each genome's weights are loaded once per block. This replaces the spatial 16×16 tiles with the same aim (`soup/kernels/cellnet_triton.py`). Verified against eager in the Triton interpreter; GPU benchmark pending (`experiments/bench_kernel.py`). Perception, actions and fields are still separate PyTorch ops; fusing them is the next step if profiling says so.
- [ ] **M7 — Showpiece run.** 1M cells, long run, edited video + write-up of what emerged.

---

## 12. Risks and open questions

| Risk | Mitigation |
|------|------------|
| **Dead world:** frozen random ribosome makes most strings non-viable | Residual weights on a sane base net; viability scan; tune `α`; seed the world with viable strings found by scan |
| **Rugged landscape:** 1-token change → wildly different phenotype | Smooth ribosome init, small `α`, measure phenotype smoothness, add weight decay on ribosome heads at init |
| **Monoculture:** one lineage takes the grid | Light patchiness, heat niches, attack costs, kin-fraction inputs; lower `p_hgt` or raise heat |
| **Heat too high/low:** genomes melt or heat is irrelevant | Calibrate `base_flip` so a median organism takes ~1 flip per few hundred ticks; make it a live-tweakable knob |
| **Kernel locality fails** (very mixed soup) | Measure genomes-per-tile early (M1); fall back to smaller cell net or Tier M |
| **Visual clutter** at high diversity | Chroma/lightness limits in OKLCH, genome-embedding smoothing, inspector-driven highlighting |
| **Nothing interesting happens** | The ablation grid and direct-weight control tell us that too; that's a valid result |

Open questions for the next iteration:
1. Should heat also perturb cell state (not just code) to mimic the original "heat bath" more literally?
2. Is a 16-symbol alphabet right, or does a smaller one (4, DNA-like) make splicing and repeats more visible?
3. Fixed 64-token cap vs. genuinely variable-length genomes (needs a different ribosome input scheme).
4. When we move to Phase 2, which is more interesting first: dialects or self-editing strings?
