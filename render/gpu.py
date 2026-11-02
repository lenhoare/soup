"""GPU renderer (moderngl). Works with any context: a glfw window or a headless
EGL context. World state is copied to textures each frame (fine for Tier S/M; a
CUDA-GL interop path would avoid the copy at Tier L)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import moderngl
import numpy as np
import torch

from . import shaders

MODES = {1: "lineage", 2: "energy", 3: "morphogens", 4: "genome distance", 5: "heat / light",
         6: "lineage depth"}
SLOT_W = 256


@dataclass
class Camera:
    cx: float = 0.0
    cy: float = 0.0
    zoom: float = 2.0      # screen px per cell

    def to_cell(self, sx: float, sy: float, vw: int, vh: int) -> tuple[float, float]:
        """Screen px (origin top-left) -> cell coordinates (unwrapped)."""
        return (sx - 0.5 * vw) / self.zoom + self.cx, (sy - 0.5 * vh) / self.zoom + self.cy

    def zoom_at(self, factor: float, sx: float, sy: float, vw: int, vh: int,
                lo: float = 0.25, hi: float = 48.0) -> None:
        wx, wy = self.to_cell(sx, sy, vw, vh)
        self.zoom = min(hi, max(lo, self.zoom * factor))
        self.cx = wx - (sx - 0.5 * vw) / self.zoom
        self.cy = wy - (sy - 0.5 * vh) / self.zoom

    def fit(self, gw: int, gh: int, vw: int, vh: int) -> None:
        self.cx, self.cy = gw / 2, gh / 2
        self.zoom = min(vw / gw, vh / gh)


def fit_colour_projection(world, n: int = 2048, seed: int = 7):
    """Fixed per ribosome: centre + PCA-whiten the 8-d embeddings of random genomes and
    keep the top two components. Hues then spread round the whole circle, and a
    genome's colour never changes as the population around it changes."""
    from soup import genome
    if world.ribosome is None:
        return None
    gen = torch.Generator().manual_seed(seed)
    g = genome.random_genomes(n, world.cfg.genome_len, world.cfg.alphabet,
                              world.cfg.init.genome_len, gen)
    e = world.ribosome.translate(g.to(world.device))["embed"].float().cpu()
    mean = e.mean(0)
    _, s, v = torch.linalg.svd(e - mean, full_matrices=False)
    proj = v[:2].T / (s[:2] / (n - 1) ** 0.5)
    return mean, proj


def slot_colours(embed: torch.Tensor, fit=None, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Genome embedding -> (hue, chroma)."""
    e = embed.float().cpu()
    if fit is not None:
        v = (e - fit[0]) @ fit[1]
    else:
        v = e @ torch.randn(e.shape[1], 2, generator=torch.Generator().manual_seed(seed))
    return torch.atan2(v[:, 1], v[:, 0]), 0.09 + 0.09 * torch.tanh(v.norm(dim=1) / 1.5)


class Renderer:
    def __init__(self, ctx: moderngl.Context, world, bloom: bool = True):
        self.ctx, self.world = ctx, world
        self.H, self.W = world.H, world.W
        S = world.cfg.genome_slots
        self.slot_h = math.ceil(S / SLOT_W)
        tex = ctx.texture
        self.t_ids = tex((self.W, self.H), 2, dtype="i4")
        self.t_cell = tex((self.W, self.H), 4, dtype="f2")
        self.t_env = tex((self.W, self.H), 4, dtype="f2")
        self.t_morph = tex((self.W, self.H), 4, dtype="f2")
        self.t_slots = tex((SLOT_W, self.slot_h), 4, dtype="f4")
        for t in (self.t_ids, self.t_cell, self.t_env, self.t_morph, self.t_slots):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        prog = lambda fs: ctx.program(vertex_shader=shaders.FULLSCREEN_VS, fragment_shader=fs)
        self.p_scene = prog(shaders.SCENE_FS)
        self.p_bright = prog(shaders.BRIGHT_FS)
        self.p_blur = prog(shaders.BLUR_FS)
        self.p_comp = prog(shaders.COMPOSITE_FS)
        self.p_over = prog(shaders.OVERLAY_FS)
        self.vaos = {p: ctx.vertex_array(p, []) for p in
                     (self.p_scene, self.p_bright, self.p_blur, self.p_comp, self.p_over)}
        for name, unit in (("u_ids", 0), ("u_cell", 1), ("u_env", 2), ("u_morph", 3),
                           ("u_slots", 4)):
            self.p_scene[name] = unit
        self.bloom = bloom
        self.size = (0, 0)
        self.t_overlay = None
        # flashes: gold (HGT), blue (heat damage), white (spore); decay per frame
        self.fx = torch.zeros(3, self.H, self.W)
        self.fx_decay = 0.9
        self.selected = -1
        self.dist = torch.zeros(S)
        self._fit = fit_colour_projection(world)
        self._hue, self._chroma = slot_colours(world.embed, self._fit)

    # ------------------------------------------------------------------ events
    def after_tick(self, epoch: bool) -> None:
        """Called after every world tick. Flashes fade in simulation time, so they
        don't pile up when many ticks run per frame (or none are rendered)."""
        w = self.world
        self.fx[:2] *= self.fx_decay          # HGT / heat: ~20 ticks
        self.fx[2] *= 0.6                     # spore pops: a few ticks
        if w.last_spore_pos.numel():
            self.fx[2].view(-1)[w.last_spore_pos.cpu()] = 1.0
        if epoch:
            if w.last_hgt_pos.numel():
                self.fx[0].view(-1)[w.last_hgt_pos.cpu()] = 1.0
            hit = w.last_heat_hit.cpu()
            if hit.any():
                gid = w.gid.cpu()
                cells = (gid >= 0) & hit[gid.clamp(min=0)]
                spark = cells & (torch.rand(gid.shape) < 0.25)
                self.fx[1][spark] = 1.0
            self._hue, self._chroma = slot_colours(w.embed, self._fit)
            if self.selected >= 0:
                self.update_distance()

    def select(self, slot: int) -> None:
        self.selected = slot
        self.update_distance()

    def update_distance(self) -> None:
        w, s = self.world, self.selected
        if s < 0:
            return
        if w.direct:
            d = (w.weights.float() - w.weights[s].float()).norm(dim=1).cpu()
        else:
            d = (w.genomes != w.genomes[s]).float().mean(1).cpu()
        self.dist = d / d[w.slot_alive.cpu()].max().clamp(min=1e-6)

    # ------------------------------------------------------------------ upload
    def upload(self) -> None:
        w = self.world
        gid = w.gid.cpu()
        alive_slots = w.slot_alive.cpu()
        root = torch.where(gid >= 0, w.root.cpu()[gid.clamp(min=0)], torch.full_like(gid, -1))
        ids = torch.stack([gid, root], -1).to(torch.int32)
        self.t_ids.write(ids.numpy().tobytes())
        st = w.state.cpu()
        cell = torch.stack([w.energy.cpu(), st[0], st[1], w.heat.cpu()], -1)
        self.t_cell.write(cell.half().numpy().tobytes())
        env = torch.cat([w.light.cpu()[None], self.fx], 0).permute(1, 2, 0)
        self.t_env.write(env.half().contiguous().numpy().tobytes())
        m = w.morph.cpu()
        m = torch.cat([m, torch.zeros(max(0, 4 - m.shape[0]), self.H, self.W)])[:4]
        self.t_morph.write(m.permute(1, 2, 0).half().contiguous().numpy().tobytes())
        S = w.cfg.genome_slots
        depth = w.depth.cpu().float()
        dmax = depth[alive_slots].max().clamp(min=1) if alive_slots.any() else torch.tensor(1.0)
        slots = torch.zeros(SLOT_W * self.slot_h, 4)
        slots[:S, 0] = self._hue
        slots[:S, 1] = self._chroma
        slots[:S, 2] = depth / dmax
        slots[:S, 3] = self.dist
        self.t_slots.write(slots.numpy().astype(np.float32).tobytes())

    def set_overlay(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self.t_overlay = None
            return
        h, w, _ = rgba.shape
        if self.t_overlay is None or self.t_overlay.size != (w, h):
            if self.t_overlay is not None:
                self.t_overlay.release()
            self.t_overlay = self.ctx.texture((w, h), 4)
        self.t_overlay.write(np.ascontiguousarray(rgba).tobytes())

    # ------------------------------------------------------------------ draw
    def _ensure_targets(self, vw: int, vh: int) -> None:
        if self.size == (vw, vh):
            return
        if self.size != (0, 0):
            for o in (self.f_scene, self.f_a, self.f_b, self.t_scene, self.t_a, self.t_b):
                o.release()
        self.size = (vw, vh)
        self.t_scene = self.ctx.texture((vw, vh), 4, dtype="f2")
        hw, hh = max(1, vw // 2), max(1, vh // 2)
        self.t_a = self.ctx.texture((hw, hh), 4, dtype="f2")
        self.t_b = self.ctx.texture((hw, hh), 4, dtype="f2")
        for t in (self.t_scene, self.t_a, self.t_b):
            t.repeat_x = t.repeat_y = False
        self.f_scene = self.ctx.framebuffer([self.t_scene])
        self.f_a = self.ctx.framebuffer([self.t_a])
        self.f_b = self.ctx.framebuffer([self.t_b])

    def draw(self, target: moderngl.Framebuffer, cam: Camera, mode: int = 1,
             cursor: tuple[float, float] | None = None, t: float = 0.0) -> None:
        vw, vh = target.size
        self._ensure_targets(vw, vh)
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)

        self.f_scene.use()
        ctx.viewport = (0, 0, vw, vh)
        for unit, tex in enumerate((self.t_ids, self.t_cell, self.t_env, self.t_morph,
                                    self.t_slots)):
            tex.use(unit)
        p = self.p_scene
        p["u_view"] = (vw, vh)
        p["u_cam"] = (cam.cx, cam.cy)
        p["u_zoom"] = cam.zoom
        p["u_grid"] = (self.W, self.H)
        p["u_mode"] = mode
        p["u_time"] = t
        p["u_sel"] = self.selected
        p["u_cursor"] = cursor if cursor is not None else (-1.0, -1.0)
        self.vaos[p].render(moderngl.TRIANGLES, vertices=3)

        if self.bloom:
            hw, hh = self.t_a.size
            self.f_a.use()
            ctx.viewport = (0, 0, hw, hh)
            self.t_scene.use(0)
            self.p_bright["u_src"] = 0
            self.p_bright["u_threshold"] = 0.9
            self.vaos[self.p_bright].render(moderngl.TRIANGLES, vertices=3)
            for _ in range(2):
                for src, dst, d in ((self.t_a, self.f_b, (1.0 / hw, 0.0)),
                                    (self.t_b, self.f_a, (0.0, 1.0 / hh))):
                    dst.use()
                    src.use(0)
                    self.p_blur["u_src"] = 0
                    self.p_blur["u_dir"] = d
                    self.vaos[self.p_blur].render(moderngl.TRIANGLES, vertices=3)

        target.use()
        ctx.viewport = (0, 0, vw, vh)
        self.t_scene.use(0)
        self.t_a.use(1)
        c = self.p_comp
        c["u_scene"], c["u_bloom"] = 0, 1
        c["u_bloom_k"] = 0.8 if self.bloom else 0.0
        c["u_ca"] = 0.003
        c["u_vignette"] = 0.4
        c["u_exposure"] = 1.0
        self.vaos[c].render(moderngl.TRIANGLES, vertices=3)

        if self.t_overlay is not None:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self.t_overlay.use(0)
            self.p_over["u_tex"] = 0
            self.vaos[self.p_over].render(moderngl.TRIANGLES, vertices=3)
            ctx.disable(moderngl.BLEND)
