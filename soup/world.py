"""The world: a 2D torus of cells, genome slots, and the tick loop.

Grid fields (H, W): gid (int64, -1 = empty), state (S, H, W), morph (M, H, W),
energy (H, W), plus environment fields light and heat.
Slot fields (genome_slots, ...): genome string, cached cell-net weights, traits,
embedding, effective length, bookkeeping.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import cellnet, genome
from .cellnet import ATTACK, GROW, SPORE, CellNetLayout
from .config import Config, resolve_device, resolve_dtype
from .fields import DriftingField
from .ribosome import (AGGRESSION, EMBED_DIM, FIDELITY, PERMEABILITY, SPORE_THRESHOLD, TRAITS,
                       Ribosome)

# direction order matches the net's direction logits: N, E, S, W  (d_row, d_col)
DIRS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])
NEIGH8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def perception_filters() -> torch.Tensor:
    ident = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
    sob_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
    lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) / 16
    return torch.stack([ident, sob_x, sob_x.T, lap])  # (4, 3, 3)


class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = dev = resolve_device(cfg)
        self.dtype = resolve_dtype(cfg)
        self.H, self.W = cfg.grid
        self.gen = torch.Generator(device=dev).manual_seed(cfg.seed)
        self.cpu_gen = torch.Generator().manual_seed(cfg.seed + 1)
        cn = cfg.cell_net
        self.layout = lay = CellNetLayout(cn.state_ch, cn.morph_ch, cn.hidden)
        rc = cfg.ribosome
        self.ribosome = Ribosome(cfg.alphabet, cfg.genome_len, lay, rc.d_model, rc.layers,
                                 rc.heads, rc.mlp_ratio, rc.delta_scale, rc.seed,
                                 rc.frozen, rc.stop_token).to(dev)
        self.w_base = cellnet.base_weights(lay, seed=rc.seed).to(dev)

        S, Lg = cfg.genome_slots, cfg.genome_len
        self.genomes = torch.zeros(S, Lg, dtype=torch.uint8, device=dev)
        self.slot_alive = torch.zeros(S, dtype=torch.bool, device=dev)
        self.dirty = torch.zeros(S, dtype=torch.bool, device=dev)
        self.weights = torch.zeros(S, lay.n_params, dtype=self.dtype, device=dev)
        self.traits = torch.zeros(S, len(TRAITS), device=dev)
        self.embed = torch.zeros(S, EMBED_DIM, device=dev)
        self.glen = torch.zeros(S, device=dev)
        self.expressed = torch.zeros(S, device=dev)   # tokens before the first STOP
        self.parent = torch.full((S,), -1, dtype=torch.long, device=dev)
        self.birth_tick = torch.zeros(S, dtype=torch.long, device=dev)
        self.depth = torch.zeros(S, dtype=torch.long, device=dev)
        # lineage id: unique per founder, inherited by spores (slot ids get recycled)
        self.root = torch.full((S,), -1, dtype=torch.long, device=dev)
        self.next_root = 0
        self.has_hgt = torch.zeros(S, dtype=torch.bool, device=dev)  # carries a spliced fragment
        # control condition (spec 8.1): weights evolve directly, no strings/ribosome
        # after founding. Founders are still translated so both conditions start alike.
        self.direct = cfg.mode == "direct"
        self.trait_logit = torch.zeros(S, len(TRAITS), device=dev)
        # last epoch's events, for analysis and render flashes
        self.last_heat_hit = torch.zeros(S, dtype=torch.bool, device=dev)
        self.last_hgt_pos = torch.zeros(0, dtype=torch.long, device=dev)
        self.last_hgt = (self.last_hgt_pos, self.last_hgt_pos)      # (recipient, donor) slots
        self.last_spore_pos = torch.zeros(0, dtype=torch.long, device=dev)  # this tick
        # per-token "came from a splice" mask (string mode), for the inspector ribbon.
        # Copied to spores as-is, so indels make it approximate.
        self.spliced = torch.zeros(S, Lg, dtype=torch.bool, device=dev)

        H, W = self.H, self.W
        self.gid = torch.full((H, W), -1, dtype=torch.long, device=dev)
        self.state = torch.zeros(cn.state_ch, H, W, device=dev)
        self.morph = torch.zeros(cn.morph_ch, H, W, device=dev)
        self.energy = torch.zeros(H, W, device=dev)
        # optional walls (e.g. competition arenas): nothing can be born into a barrier
        # cell, and morphogens don't cross it
        self.barrier: torch.Tensor | None = None

        fc = cfg.fields
        fgen = torch.Generator().manual_seed(cfg.seed + 2)
        self.light_field = DriftingField(H, W, fc.modes, fc.max_wavenumber, fc.light_drift,
                                         fgen, dev, contrast=fc.light_contrast)
        self.heat_field = DriftingField(H, W, fc.modes, fc.heat_max_wavenumber or fc.max_wavenumber,
                                        fc.heat_drift, fgen, dev, contrast=fc.heat_contrast,
                                        standardize=fc.heat_contrast is not None)
        self.light = self.light_field.at(0)
        self.heat = self.heat_field.at(0)

        C = cn.state_ch + cn.morph_ch
        self.filt = perception_filters().repeat(C, 1, 1).unsqueeze(1).to(dev)  # (4C,1,3,3)
        self.dirs = DIRS.to(dev)
        self.tick_count = 0
        self.kernel = self._pick_kernel()
        self.counters = {"births": 0, "spores": 0, "attacks_won": 0, "deaths": 0,
                         "spores_blocked": 0, "heat_flips": 0, "heat_hits": 0, "hgt": 0}

    # ------------------------------------------------------------------ slots
    def alloc_slots(self, n: int) -> torch.Tensor:
        free = (~self.slot_alive).nonzero().flatten()[:n]
        self.slot_alive[free] = True
        return free

    def translate_dirty(self) -> int:
        idx = self.dirty.nonzero().flatten()
        if idx.numel() == 0:
            return 0
        out = self.ribosome.translate(self.genomes[idx])
        self.weights[idx] = (self.w_base + self.cfg.ribosome.alpha * out["delta"]).to(self.dtype)
        self.traits[idx] = out["traits"]
        self.embed[idx] = out["embed"]
        self.glen[idx] = genome.effective_length(self.genomes[idx]).float()
        self.expressed[idx] = self.ribosome.expressed_length(self.genomes[idx]).float()
        self.dirty[idx] = False
        return idx.numel()

    def seed_founders(self, n: int | None = None, genomes: torch.Tensor | None = None,
                      positions: torch.Tensor | None = None) -> torch.Tensor:
        """Place founder cells, each with its own new slot and lineage, at random empty
        positions (or at the given flat `positions`). Returns the flat positions used."""
        ic = self.cfg.init
        if genomes is None:
            n = ic.founders if n is None else n
            genomes = genome.random_genomes(n, self.cfg.genome_len, self.cfg.alphabet,
                                            ic.genome_len, self.cpu_gen,
                                            exclude=self.cfg.ribosome.stop_token)
        n = genomes.shape[0]
        if positions is not None:
            pos = positions.to(self.device)[:n]
        else:
            empty = (self.gid.view(-1) < 0).nonzero().flatten()
            pos = empty[torch.randperm(empty.numel(), generator=self.gen, device=self.device)[:n]]
        slots = self.alloc_slots(pos.numel())
        pos = pos[: slots.numel()]
        self.genomes[slots] = genomes[: slots.numel()].to(self.device)
        self.dirty[slots] = True
        self.birth_tick[slots] = self.tick_count
        self.parent[slots] = -1
        self.depth[slots] = 0
        self.root[slots] = torch.arange(self.next_root, self.next_root + slots.numel(),
                                        device=self.device)
        self.next_root += slots.numel()
        self.gid.view(-1)[pos] = slots
        self.energy.view(-1)[pos] = ic.energy
        self.has_hgt[slots] = False
        self.spliced[slots] = False
        self.translate_dirty()
        if self.direct:
            self.trait_logit[slots] = torch.logit(self.traits[slots].clamp(1e-4, 1 - 1e-4))
            self.glen[slots] = float(ic.genome_len)   # constant token cost in control
        return pos

    # ------------------------------------------------------------------ tick
    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=self.gen, device=self.device)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        u = self._rand(*logits.shape).clamp_(1e-9, 1 - 1e-9)
        return (logits / self.cfg.dynamics.temperature - torch.log(-torch.log(u))).argmax(-1)

    def _neighbour_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fraction of the 8 neighbours that are kin (same gid) / occupied."""
        kin = torch.zeros(self.H, self.W, device=self.device)
        occ = torch.zeros_like(kin)
        for dy, dx in NEIGH8:
            n = torch.roll(self.gid, (dy, dx), (0, 1))
            occ += n >= 0
            kin += (n == self.gid) & (n >= 0)
        return kin / 8, occ / 8

    def _winners(self, cand: torch.Tensor, target: torch.Tensor, prio: torch.Tensor) -> torch.Tensor:
        """Among candidate cells (indices) pick one per target: highest random priority."""
        if cand.numel() == 0:
            return cand
        best = torch.full((self.H * self.W,), -1, dtype=torch.long, device=self.device)
        best.scatter_reduce_(0, target[cand], prio[cand], reduce="amax")
        return cand[best[target[cand]] == prio[cand]]

    def tick(self) -> None:
        cfg = self.cfg
        if self.tick_count % cfg.fields.update_every == 0:
            self.light = self.light_field.at(self.tick_count)
            self.heat = self.heat_field.at(self.tick_count)

        idx = (self.gid.view(-1) >= 0).nonzero().flatten()
        self.last_spore_pos = idx[:0]
        if idx.numel() > 0:
            self._cells_step(idx)
        self._diffuse_morph()
        self.tick_count += 1
        if self.tick_count % cfg.epoch_ticks == 0:
            self.epoch()

    def _cells_step(self, idx: torch.Tensor) -> None:
        cfg, lay, ec, dc = self.cfg, self.layout, self.cfg.energy, self.cfg.dynamics
        H, W, HW = self.H, self.W, self.H * self.W
        Sc, Mc = lay.state_ch, lay.morph_ch
        N = idx.numel()
        gid_f, en_f = self.gid.view(-1), self.energy.view(-1)
        st_f, mo_f = self.state.view(Sc, HW), self.morph.view(Mc, HW)

        # ---- perceive
        x = torch.cat([self.state, self.morph])[None]
        x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="circular"), self.filt, groups=Sc + Mc)
        percept = x.view(-1, HW)[:, idx].T
        kin, occ = self._neighbour_stats()
        extras = torch.stack([self.energy, self.light, kin, self.heat]).view(4, HW)[:, idx].T
        g = gid_f[idx]
        out = self._cellnet(torch.cat([percept, extras], 1), g)

        # ---- state + morphogen emission
        new_st = st_f[:, idx] * (1 - dc.state_leak) + dc.state_rate * torch.tanh(out[:, lay.out_state]).T
        if cfg.heat.state_noise > 0:   # heat jostles the cell's memory directly
            new_st = new_st + cfg.heat.state_noise * self.heat.view(-1)[idx] * self._randn(*new_st.shape)
        st_f[:, idx] = new_st.clamp_(-1, 1)
        mo_f[:, idx] = (mo_f[:, idx]
                        + dc.emit_rate * F.softplus(out[:, lay.out_morph]).T).clamp_(0, 2)

        # ---- choose actions (from start-of-tick state)
        action = self._sample(out[:, lay.out_act])
        d = self.dirs[self._sample(out[:, lay.out_dir])]
        r, c = idx // W, idx % W
        tgt = ((r + d[:, 0]) % H) * W + (c + d[:, 1]) % W
        tg = gid_f[tgt]
        E = en_f[idx].clone()
        tr = self.traits[g]
        spore_thr = ec.spore_min + ec.spore_span * tr[:, SPORE_THRESHOLD]
        prio = torch.randperm(N, generator=self.gen, device=self.device)
        empty_t = tg < 0
        if self.barrier is not None:
            empty_t = empty_t & ~self.barrier.view(-1)[tgt]

        # grow and spore compete for the same empty targets
        want_birth = (((action == GROW) & (E > ec.grow_threshold))
                      | ((action == SPORE) & (E > spore_thr))) & empty_t
        births = self._winners(want_birth.nonzero().flatten(), tgt, prio)
        grow = births[action[births] == GROW]
        spore = births[action[births] == SPORE]
        want_attack = (action == ATTACK) & ~empty_t & (tg != g) & (E > ec.attack_min)
        attack = self._winners(want_attack.nonzero().flatten(), tgt, prio)

        new_E = en_f.clone()
        attacked = torch.zeros(HW, dtype=torch.bool, device=self.device)

        # grow: same gid, split energy, child inherits parent state
        if grow.numel():
            src, dst = idx[grow], tgt[grow]
            half = (E[grow] - ec.grow_cost).clamp(min=0) / 2
            new_E[src] = half
            new_E[dst] = half
            gid_f[dst] = g[grow]
            st_f[:, dst] = st_f[:, src]
            self.counters["births"] += grow.numel()

        # spore: new slot holding a mutated copy of the genome
        if spore.numel():
            slots = self.alloc_slots(spore.numel())
            self.counters["spores_blocked"] += spore.numel() - slots.numel()
            spore = spore[: slots.numel()]
            if spore.numel():
                parents = g[spore]
                if self.direct:
                    self._direct_mutate(slots, parents)
                else:
                    mc = cfg.mutation
                    self.genomes[slots] = genome.mutate(
                        self.genomes[parents].cpu(), mc.p_sub, mc.p_indel, mc.p_dup, mc.p_del,
                        cfg.alphabet, self.cpu_gen).to(self.device)
                    self.dirty[slots] = True
                self.has_hgt[slots] = self.has_hgt[parents]
                self.spliced[slots] = self.spliced[parents]
                self.parent[slots] = parents
                self.depth[slots] = self.depth[parents] + 1
                self.root[slots] = self.root[parents]
                self.birth_tick[slots] = self.tick_count
                src, dst = idx[spore], tgt[spore]
                new_E[src] = E[spore] - ec.spore_energy - ec.spore_cost
                new_E[dst] = ec.spore_energy
                gid_f[dst] = slots
                st_f[:, dst] = 0
                self.counters["spores"] += spore.numel()

        self.last_spore_pos = tgt[spore]

        # attack: contest energy * (1 + aggression); winner takes cell + share of energy
        if attack.numel():
            src, dst = idx[attack], tgt[attack]
            attacked[src] = True
            new_E[src] = (E[attack] - ec.attack_cost).clamp(min=0)
            def_g = tg[attack]
            sa = E[attack] * (1 + tr[attack, AGGRESSION])
            sd = en_f[dst] * (1 + self.traits[def_g, AGGRESSION])
            won = self._rand(attack.numel()) < sa / (sa + sd + 1e-6)
            if won.any():
                ws, wd = src[won], dst[won]
                new_E[wd] = ec.loot * en_f[wd]
                gid_f[wd] = g[attack[won]]
                st_f[:, wd] = st_f[:, ws]
                self.counters["attacks_won"] += int(won.sum())

        # ---- energy income and costs for every live cell (after actions)
        alive = gid_f >= 0
        live = alive.nonzero().flatten()
        lg = gid_f[live]
        income = ec.photo * self.light.view(-1)[live] * (1 - ec.crowding * occ.view(-1)[live])
        income = income * (~attacked[live])
        cost = ec.basal + ec.c_fid * self.traits[lg, FIDELITY]
        if cfg.heat.energy_cost > 0:   # hot zones are harsh regardless of genome
            cost = cost + cfg.heat.energy_cost * self.heat.view(-1)[live]
        if cfg.features.cost:
            cost = cost + ec.c_tok * self.glen[lg]
        else:
            # same energy budget, no length dependence (M3: a zero cost confounded the
            # ablation by making everyone far richer)
            cost = cost + ec.c_tok * cfg.init.genome_len
        new_E[live] = (new_E[live] + income - cost).clamp(0, 1)

        # ---- death
        dead = alive & (new_E <= 0)
        if dead.any():
            self.counters["deaths"] += int(dead.sum())
            gid_f[dead] = -1
        new_E[gid_f < 0] = 0
        self.energy.copy_(new_E.view(H, W))
        self.state *= (self.gid >= 0)

        if spore.numel():
            self.translate_dirty()   # founders need weights before their first tick

    def _pick_kernel(self) -> str:
        k = self.cfg.kernel
        if k == "auto":
            from .kernels import cellnet_triton
            return "triton" if (self.device.type == "cuda" and cellnet_triton.HAVE_TRITON) else "eager"
        return k

    def _cellnet(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Run every live cell's own genome net. Triton path (M6) groups cells by genome
        so each genome's weights are loaded once per block; eager path gathers per cell."""
        if self.kernel == "triton":
            try:
                from .kernels import cellnet_triton
                return cellnet_triton.forward(self.layout, x, self.weights, g,
                                              self.cfg.kernel_block)
            except Exception as e:  # compile/runtime problem: fall back once, loudly
                import warnings
                warnings.warn(f"triton cell-net kernel failed ({e!r}); using eager")
                self.kernel = "eager"
        return cellnet.forward(self.layout, x, self.weights, g, self.cfg.chunk_cells)

    def _diffuse_morph(self) -> None:
        dc = self.cfg.dynamics
        m = self.morph
        lap = (torch.roll(m, 1, 1) + torch.roll(m, -1, 1) + torch.roll(m, 1, 2)
               + torch.roll(m, -1, 2) - 4 * m)
        self.morph = (m + dc.morph_diffuse * lap) * (1 - dc.morph_decay)
        if self.barrier is not None:
            self.morph = self.morph * ~self.barrier

    # ------------------------------------------------------------------ control
    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=self.gen, device=self.device)

    def _kick_weights(self, slots: torch.Tensor, rel: torch.Tensor) -> None:
        """Direct mode: add Gaussian noise of relative size `rel` (per slot) to weights."""
        w = self.weights[slots].float()
        sig = rel.view(-1, 1) * w.norm(dim=1, keepdim=True) / w.shape[1] ** 0.5
        self.weights[slots] = (w + sig * self._randn(*w.shape)).to(self.dtype)

    def _direct_mutate(self, slots: torch.Tensor, parents: torch.Tensor) -> None:
        dc = self.cfg.direct
        self.weights[slots] = self.weights[parents]
        self.trait_logit[slots] = self.trait_logit[parents]
        mut = slots[self._rand(slots.numel()) < dc.p_mut]
        if mut.numel():
            self._kick_weights(mut, torch.full((mut.numel(),), dc.sigma_rel, device=self.device))
            self.trait_logit[mut] += dc.sigma_trait * self._randn(mut.numel(), len(TRAITS))
        self.traits[slots] = torch.sigmoid(self.trait_logit[slots])
        self.embed[slots] = self.embed[parents]
        self.glen[slots] = self.glen[parents]

    # ------------------------------------------------------------------ epoch
    def epoch(self) -> None:
        """Every `epoch_ticks`: heat damage, HGT, translate dirty, free empty slots."""
        f = self.cfg.features
        if f.heat:
            self.heat_damage()
        if f.hgt:
            self.gene_transfer()
        self.translate_dirty()
        self.free_empty_slots()

    def heat_damage(self) -> int:
        """Each non-pad token of each live genome flips with
        p = base_flip * mean_heat * (1 - fidelity), to any other symbol (pad included, so
        heat can delete). Pads are not hit: flipping them would make heat mostly *insert*
        code (M3 found this inflated genomes). The whole organism is hit at once."""
        S, Lg, A = self.cfg.genome_slots, self.cfg.genome_len, self.cfg.alphabet
        alive = self.gid >= 0
        g = self.gid[alive]
        cnt = torch.bincount(g, minlength=S).float()
        mean_h = torch.bincount(g, weights=self.heat[alive], minlength=S) / cnt.clamp(min=1)
        hc = self.cfg.heat
        p = hc.base_flip * mean_h * (1 - self.traits[:, FIDELITY]) ** hc.fid_power * (cnt > 0)
        if self.direct:
            n = torch.poisson(p * self.glen, generator=self.gen)
            hit = n > 0
            if hit.any():
                s = hit.nonzero().flatten()
                self._kick_weights(s, self.cfg.direct.flip_rel * n[s].sqrt())
            n_flips = int(n.sum())
        else:
            flips = (self._rand(S, Lg) < p[:, None]) & (self.genomes != genome.PAD)
            hit = flips.any(1)
            if hit.any():
                off = torch.randint(1, A, (S, Lg), generator=self.gen, device=self.device,
                                    dtype=torch.uint8)
                self.genomes = torch.where(flips, (self.genomes + off) % A, self.genomes)
                self.dirty |= hit
            n_flips = int(flips.sum())
        self.last_heat_hit = hit
        self.counters["heat_flips"] += n_flips
        self.counters["heat_hits"] += int(hit.sum())
        return n_flips

    def border_pairs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All (recipient gid, donor gid, recipient flat pos) over 4-neighbour borders."""
        rs, ds, ps = [], [], []
        alive = self.gid >= 0
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            n = torch.roll(self.gid, (dy, dx), (0, 1))
            m = alive & (n >= 0) & (n != self.gid)
            rs.append(self.gid[m])
            ds.append(n[m])
            ps.append(m.view(-1).nonzero().flatten())
        return torch.cat(rs), torch.cat(ds), torch.cat(ps)

    def gene_transfer(self) -> int:
        """Spec 4.1: sample border pairs; with p_hgt * permeability(recipient) copy a
        k-token window (starting at a non-pad donor token) into the recipient string."""
        hc, Lg = self.cfg.hgt, self.cfg.genome_len
        rg, dg, pos = self.border_pairs()
        if rg.numel() == 0:
            self.last_hgt_pos = pos
            self.last_hgt = (rg, dg)
            return 0
        pick = torch.randperm(rg.numel(), generator=self.gen, device=self.device)
        pick = pick[: hc.samples_per_epoch]
        rg, dg, pos = rg[pick], dg[pick], pos[pick]
        ok = self._rand(rg.numel()) < hc.p_hgt * self.traits[rg, PERMEABILITY]
        rg, dg, pos = rg[ok], dg[ok], pos[ok]
        n = rg.numel()
        if n:
            kmin, kmax = hc.window
            ks = torch.randint(kmin, kmax + 1, (n,), generator=self.cpu_gen).tolist()
            u = torch.rand(n, 2, generator=self.cpu_gen).tolist()
            if self.direct:
                P = self.layout.n_params
                for r, d, k, (a, _) in zip(rg.tolist(), dg.tolist(), ks, u):
                    blk = max(1, P * k // Lg)
                    o = int(a * (P - blk + 1))
                    self.weights[r, o:o + blk] = self.weights[d, o:o + blk]
            else:
                slots = torch.unique(torch.cat([rg, dg]))
                rows = {s: row for s, row in zip(slots.tolist(), self.genomes[slots].cpu())}
                marks = []
                for r, d, k, (a, b) in zip(rg.tolist(), dg.tolist(), ks, u):
                    donor = rows[d]
                    nz = (donor != genome.PAD).nonzero().flatten()
                    if nz.numel() == 0:
                        continue
                    src = min(int(nz[int(a * nz.numel())]), Lg - k)
                    dst = int(b * (Lg - k + 1))
                    rec = rows[r].clone()
                    rec[dst:dst + k] = donor[src:src + k]
                    rows[r] = rec
                    marks.append((r, dst, k))
                upd = torch.unique(rg)
                self.genomes[upd] = torch.stack([rows[s] for s in upd.tolist()]).to(self.device)
                for r, dst, k in marks:
                    self.spliced[r, dst:dst + k] = True
                self.dirty[upd] = True
            self.has_hgt[rg] = True
            self.counters["hgt"] += n
        self.last_hgt_pos = pos
        self.last_hgt = (rg, dg)
        return n

    def slot_counts(self) -> torch.Tensor:
        g = self.gid[self.gid >= 0]
        return torch.bincount(g, minlength=self.cfg.genome_slots)

    def free_empty_slots(self) -> int:
        empty = self.slot_alive & (self.slot_counts() == 0)
        n = int(empty.sum())
        if n:
            self.slot_alive[empty] = False
            self.genomes[empty] = 0
            self.dirty[empty] = False
            self.has_hgt[empty] = False
            self.spliced[empty] = False
        return n

    def run(self, ticks: int) -> None:
        for _ in range(ticks):
            self.tick()
