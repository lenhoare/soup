import torch

from soup import cellnet, genome
from soup.cellnet import CellNetLayout
from soup.config import load_config
from soup.metrics import genomes_per_tile, phenotype_smoothness, snapshot
from soup.ribosome import Ribosome
from soup.world import World


def small_cfg(**over):
    base = {"grid": [32, 32], "genome_slots": 256, "device": "cpu",
            "init": {"founders": 8, "genome_len": 24}, "epoch_ticks": 8}
    base.update(over)
    return load_config(None, base)


# ---------------------------------------------------------------- genome
def test_random_genomes_and_length():
    g = genome.random_genomes(10, 64, 16, 20, torch.Generator().manual_seed(0))
    assert g.dtype == torch.uint8 and g.shape == (10, 64)
    assert (genome.effective_length(g) == 20).all()
    assert int(g.max()) < 16


def test_mutate_preserves_shape_and_alphabet():
    gen = torch.Generator().manual_seed(0)
    g = genome.random_genomes(200, 64, 16, 40, gen)
    m = genome.mutate(g, 0.05, 0.05, 0.05, 0.05, 16, gen)
    assert m.shape == g.shape and m.dtype == torch.uint8
    assert int(m.max()) < 16
    assert (m != g).any()
    assert (genome.mutate(g, 0, 0, 0, 0, 16, gen) == g).all()


def test_insert_absorbs_into_pad():
    row = torch.tensor([1, 2, 3, 0, 5], dtype=torch.uint8)
    assert genome._insert(row, 1, 9).tolist() == [1, 9, 2, 3, 5]
    full = torch.tensor([1, 2, 3], dtype=torch.uint8)
    assert genome._insert(full, 0, 9).tolist() == [9, 1, 2]


# ---------------------------------------------------------------- cell net
def test_layout_matches_spec():
    lay = CellNetLayout()
    assert (lay.n_in, lay.n_out) == (52, 20)
    assert lay.n_params == 52 * 16 + 16 + 16 * 20 + 20
    w = cellnet.base_weights(lay)
    assert torch.equal(lay.pack(*lay.unpack(w[None]))[0], w)


def test_forward_uses_per_cell_weights():
    lay = CellNetLayout()
    w = torch.stack([cellnet.base_weights(lay, 0), cellnet.base_weights(lay, 1)])
    x = torch.randn(1, lay.n_in).repeat(2, 1)
    out = cellnet.forward(lay, x, w, torch.tensor([0, 1]))
    assert out.shape == (2, 20)
    assert not torch.allclose(out[0], out[1])
    # chunking does not change results
    x = torch.randn(10, lay.n_in)
    g = torch.randint(0, 2, (10,))
    assert torch.allclose(cellnet.forward(lay, x, w, g, chunk=3), cellnet.forward(lay, x, w, g))


# ---------------------------------------------------------------- ribosome
def test_ribosome_is_frozen_seeded_and_bounded():
    lay = CellNetLayout()
    r1, r2 = Ribosome(16, 64, lay, seed=5), Ribosome(16, 64, lay, seed=5)
    assert not any(p.requires_grad for p in r1.parameters())
    g = genome.random_genomes(4, 64, 16, 30, torch.Generator().manual_seed(0))
    o1, o2 = r1.translate(g), r2.translate(g)
    assert torch.equal(o1["delta"], o2["delta"])
    assert o1["delta"].shape == (4, lay.n_params)
    assert ((o1["traits"] >= 0) & (o1["traits"] <= 1)).all()
    assert o1["embed"].shape == (4, 8)
    # an all-pad genome must not produce NaNs
    assert torch.isfinite(r1.translate(torch.zeros(1, 64, dtype=torch.uint8))["delta"]).all()


def test_ribosome_ignores_what_is_under_pads():
    # pads are masked, so the output depends only on non-pad tokens and their positions
    lay = CellNetLayout()
    r = Ribosome(16, 64, lay)
    g = genome.random_genomes(1, 64, 16, 10, torch.Generator().manual_seed(0))
    g2 = g.clone()
    g2[0, 30] = 0  # already pad: no-op
    assert torch.allclose(r.translate(g)["delta"], r.translate(g2)["delta"])


def test_phenotype_smoothness_runs():
    lay = CellNetLayout()
    r = Ribosome(16, 64, lay)
    gen = torch.Generator().manual_seed(0)
    g = genome.random_genomes(32, 64, 16, 32, gen)
    s = phenotype_smoothness(r, cellnet.base_weights(lay), 0.5, g, 16, gen)
    assert 0 < s < 10


# ---------------------------------------------------------------- world
def test_world_invariants_over_ticks():
    w = World(small_cfg())
    w.seed_founders()
    assert int((w.gid >= 0).sum()) == 8
    w.run(40)
    alive = w.gid >= 0
    assert ((w.energy >= 0) & (w.energy <= 1)).all()
    assert (w.energy[~alive] == 0).all()
    assert (w.state[:, ~alive] == 0).all()
    # every occupied cell points at a live, translated slot
    assert w.slot_alive[w.gid[alive]].all()
    assert not w.dirty[w.gid[alive]].any()
    # after an epoch boundary no live slot is empty
    assert (w.slot_counts()[w.slot_alive] > 0).all()
    assert w.tick_count == 40


def test_world_grows():
    w = World(small_cfg())
    w.seed_founders()
    w.run(16)
    assert int((w.gid >= 0).sum()) > 8
    assert w.counters["births"] > 0


def test_spores_create_new_slots_with_parents():
    cfg = small_cfg(energy={"spore_min": 0.0, "spore_span": 0.0})
    w = World(cfg)
    w.seed_founders()
    w.run(24)
    kids = (w.parent >= 0) & w.slot_alive
    assert w.counters["spores"] > 0
    assert kids.any()
    assert (w.depth[kids] >= 1).all()


def test_lineage_root_inherited_by_spores():
    cfg = small_cfg(energy={"spore_min": 0.0, "spore_span": 0.0})
    w = World(cfg)
    w.seed_founders()
    assert w.root[w.slot_alive].tolist() == list(range(8))
    w.run(24)
    live = w.slot_alive.nonzero().flatten()
    assert (w.root[live] >= 0).all() and (w.root[live] < 8).all()
    kids = live[w.parent[live] >= 0]
    # a child's lineage matches its parent's (if the parent slot hasn't been recycled)
    same = w.slot_alive[w.parent[kids]] & (w.depth[w.parent[kids]] == w.depth[kids] - 1)
    assert (w.root[kids][same] == w.root[w.parent[kids]][same]).all()


def test_viability_scan_smoke():
    from experiments.viability import lattice, scan
    assert lattice(64, 64, 32).tolist() == [16 * 64 + 16, 16 * 64 + 48, 48 * 64 + 16, 48 * 64 + 48]
    cfg = small_cfg(grid=[64, 64])
    r = scan(cfg, 4, 16, 32, 2, torch.Generator().manual_seed(0))
    assert 0 <= r["viable_frac"] <= 1


def test_founders_start_with_energy_and_weights():
    w = World(small_cfg())
    pos = w.seed_founders()
    assert torch.allclose(w.energy.view(-1)[pos], torch.tensor(0.5))
    assert (w.weights[w.slot_alive].abs().sum(1) > 0).all()
    assert not w.dirty.any()


def test_heat_damage_flips_tokens_and_marks_dirty():
    w = World(small_cfg(heat={"base_flip": 1.0}))
    w.seed_founders()
    w.heat.fill_(1.0)
    w.traits[:, 0] = 0.0          # no fidelity
    before = w.genomes.clone()
    n = w.heat_damage()
    assert n > 0
    changed = (w.genomes != before).any(1)
    assert torch.equal(changed, w.last_heat_hit)
    assert w.dirty[changed].all()
    # fidelity 1 blocks all damage
    w.traits[:, 0] = 1.0
    assert w.heat_damage() == 0


def test_gene_transfer_copies_donor_window():
    w = World(small_cfg(hgt={"p_hgt": 1.0, "samples_per_epoch": 1, "window": [4, 4]}))
    g = torch.zeros(2, 64, dtype=torch.uint8)
    g[0, :8] = 1
    g[1, :8] = 2
    w.seed_founders(genomes=g, positions=torch.tensor([0, 1]))
    w.traits[:, 1] = 1.0          # fully permeable
    assert w.gene_transfer() == 1
    rec = int(w.has_hgt.nonzero()[0])
    own_tok = int(g[0, 0]) if rec == int(w.gid.view(-1)[0]) else int(g[1, 0])
    donor_tok = 3 - own_tok
    assert int((w.genomes[rec] == donor_tok).sum()) == 4
    assert w.dirty[rec]


def test_direct_mode_runs_without_translation():
    w = World(small_cfg(mode="direct", energy={"spore_min": 0.0, "spore_span": 0.0},
                        heat={"base_flip": 0.5}, hgt={"p_hgt": 1.0}))
    w.seed_founders()
    w.ribosome = None              # must never be needed after founding
    w.run(40)
    assert w.counters["spores"] > 0
    assert w.counters["hgt"] > 0 and w.counters["heat_flips"] > 0
    assert torch.isfinite(w.weights).all()


def test_stop_token_truncates_translation():
    lay = CellNetLayout()
    r = Ribosome(16, 64, lay, stop_token=15)
    g = genome.random_genomes(1, 64, 16, 20, torch.Generator().manual_seed(0), exclude=15)
    assert not (g == 15).any()
    tail_changed = g.clone()
    tail_changed[0, 12] = 15
    other_tail = tail_changed.clone()
    other_tail[0, 13:20] = 3                       # edits after the STOP are silent
    a, b = r.translate(tail_changed)["delta"], r.translate(other_tail)["delta"]
    assert torch.allclose(a, b)
    assert not torch.allclose(a, r.translate(g)["delta"])
    assert int(r.expressed_length(tail_changed)) == 12
    assert int(r.expressed_length(g)) == 20


def test_heat_options_run():
    cfg = small_cfg(ribosome={"stop_token": 15},
                    heat={"base_flip": 2.0, "fid_power": 2.0, "state_noise": 0.2,
                          "energy_cost": 0.005},
                    fields={"heat_max_wavenumber": 1, "heat_contrast": 6.0})
    w = World(cfg)
    w.seed_founders()
    assert not (w.genomes == 15).any()             # founders fully expressed
    w.run(40)
    assert torch.isfinite(w.state).all()
    assert w.counters["heat_flips"] > 0
    live = w.slot_alive
    assert (w.expressed[live] <= w.glen[live]).all()
    # sharp heat zones: most cells are clearly hot or clearly cold
    h = w.heat
    assert float(((h < 0.2) | (h > 0.8)).float().mean()) > 0.4


def test_fid_power_reduces_damage():
    def flips(power):
        w = World(small_cfg(heat={"base_flip": 0.5, "fid_power": power}))
        w.seed_founders()
        w.heat.fill_(1.0)
        w.traits[:, 0] = 0.5
        return sum(w.heat_damage() for _ in range(20))
    assert flips(3.0) < 0.5 * flips(1.0)


def test_slots_freed_when_empty():
    w = World(small_cfg())
    w.seed_founders()
    w.gid.fill_(-1)
    w.energy.zero_()
    assert w.free_empty_slots() == 8
    assert not w.slot_alive.any()


def test_repeat_frac():
    from soup.metrics import repeat_frac
    g = torch.zeros(3, 16, dtype=torch.uint8)
    g[0, :8] = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])        # no repeats
    g[1, :8] = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4])        # 1234 twice
    assert repeat_frac(g)[0] == 0
    assert abs(float(repeat_frac(g)[1]) - 2 / 5) < 1e-6      # 2 of 5 k-mers duplicated
    assert repeat_frac(g)[2] == 0                             # all pad


def test_genomes_per_tile():
    gid = torch.full((32, 32), -1)
    gid[:16, :16] = 3
    gid[0, 0] = 4
    gid[20, 20] = 5
    assert genomes_per_tile(gid, 16) == 1.5  # tile0: {3,4}, tile3: {5}


def test_snapshot_metrics():
    w = World(small_cfg())
    w.seed_founders()
    w.run(8)
    m = snapshot(w)
    assert m["cells"] > 0 and m["organisms"] > 0
    assert 0 < m["zlib_ratio"] <= 1.1
