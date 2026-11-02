import numpy as np
import pytest
import torch

from render.gpu import Camera
from render.overlay import PAD_TILE, compose, organism_info, token_colours
from soup.config import load_config
from soup.world import World


def small_cfg(**over):
    base = {"grid": [48, 48], "genome_slots": 512, "device": "cpu",
            "init": {"founders": 12, "genome_len": 24}, "epoch_ticks": 8}
    base.update(over)
    return load_config(None, base)


def gl_ctx():
    moderngl = pytest.importorskip("moderngl")
    try:
        return moderngl.create_standalone_context(backend="egl")
    except Exception as e:  # no EGL / GPU in this environment
        pytest.skip(f"no headless GL: {e}")


def test_camera_zoom_keeps_point_under_cursor():
    cam = Camera(10.0, 20.0, 4.0)
    before = cam.to_cell(300, 200, 800, 600)
    cam.zoom_at(2.0, 300, 200, 800, 600)
    after = cam.to_cell(300, 200, 800, 600)
    assert np.allclose(before, after)
    assert cam.zoom == 8.0


def test_token_colours_distinct_and_pad_grey():
    cols = token_colours(16)
    assert len(cols) == 16 and len(set(cols)) == 16
    assert cols[0] == PAD_TILE


def test_overlay_compose_with_inspector():
    torch.set_grad_enabled(False)
    w = World(small_cfg())
    w.seed_founders()
    w.run(16)
    slot = int(w.gid[w.gid >= 0][0])
    img = compose((800, 600), [("tick", "16")], "hint", organism_info(w, slot, ["x"]), 16)
    assert img.shape == (600, 800, 4)
    assert img[..., 3].max() > 0              # something was drawn
    assert img[300, 300, 3] == 0              # centre stays transparent


def test_session_renders_all_modes_and_selects():
    ctx = gl_ctx()
    from render.viewer import Session
    torch.set_grad_enabled(False)
    size = (320, 240)
    fbo = ctx.simple_framebuffer(size)
    s = Session(small_cfg(), ctx, size)
    s.step(24)
    means = []
    for mode in range(1, 7):
        s.mode = mode
        s.render(fbo, size, (5.5, 5.5), 1.0, force_overlay=True)
        img = np.frombuffer(fbo.read(components=3), np.uint8)
        means.append(img.mean())
    assert all(m > 1 for m in means)
    assert len(set(round(m, 1) for m in means)) > 3     # modes actually differ
    # clicking near a live cell selects its organism
    ys, xs = (s.world.gid >= 0).nonzero(as_tuple=True)
    x, y = float(xs[0]) + 0.5, float(ys[0]) + 0.5
    sx = (x - s.cam.cx) * s.cam.zoom + size[0] / 2
    sy = (y - s.cam.cy) * s.cam.zoom + size[1] / 2
    s.select_at(sx, sy, *size)
    assert s.r.selected == int(s.world.gid[ys[0], xs[0]])
    assert s.history and "born" in s.history[0]


def test_flashes_fade_in_sim_time():
    ctx = gl_ctx()
    from render.viewer import Session
    torch.set_grad_enabled(False)
    s = Session(small_cfg(energy={"spore_min": 0.0, "spore_span": 0.0}), ctx, (64, 64))
    s.step(200)
    assert float(s.r.fx[2].max()) <= 1.0
    assert float(s.r.fx[2].mean()) < 0.2       # spores don't accumulate across ticks
