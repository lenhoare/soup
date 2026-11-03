"""Interactive viewer (glfw + moderngl), and a headless frame/video renderer.

    python -m render.viewer --config config/tier_s.yaml
    python -m render.viewer --headless frames/ --frames 600 --ticks-per-frame 2
    python -m render.viewer --video soup.mp4 --frames 1800 --size 1920x1080

Controls
    mouse wheel  zoom at cursor          drag      pan
    click        inspect organism        Esc       clear selection
    space        pause / resume          .         single tick (when paused)
    [ / ]        slower / faster         1-6       view modes
    H / C / G    toggle heat / token cost / gene transfer live
    F            fit world to window     S         snapshot PNG
    B            toggle bloom            O         toggle HUD
    K            save a checkpoint (checkpoints/); resume or re-render with --load

    python -m render.viewer --load checkpoints/soup_00004096.pt --video hq.mp4 \
        --size 1920x1080 --supersample 2      # re-render a saved moment at quality
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from soup import checkpoint
from soup.config import load_config
from soup.metrics import snapshot
from soup.world import World

from .gpu import MODES, Camera, Renderer
from .overlay import compose, organism_info

HINT = ("space pause · . step · [ ] speed · 1-6 views · H/C/G features · "
        "S snapshot · F fit · click inspect")


class Session:
    """Simulation + renderer state shared by the window and headless paths."""

    def __init__(self, cfg, ctx, size: tuple[int, int], world: World | None = None):
        if world is None:
            world = World(cfg)
            world.seed_founders()
        self.world = world
        self.r = Renderer(ctx, self.world)
        self.cam = Camera()
        self.cam.fit(self.world.W, self.world.H, *size)
        self.mode = 1
        self.ss = 1              # supersampling factor (headless), for the HUD zoom
        self.paused = False
        self.tpf = 1.0           # ticks per frame (may be fractional)
        self._acc = 0.0
        self.show_hud = True
        self.history: list[str] = []
        self.stats: dict = {}
        self.tps = 0.0
        self.fps = 0.0
        self._hud_t = 0.0
        self._tick_times: list[float] = []

    # -------------------------------------------------------------- sim
    def step(self, n: int) -> None:
        w = self.world
        for _ in range(n):
            w.tick()
            epoch = w.tick_count % w.cfg.epoch_ticks == 0
            self.r.after_tick(epoch)
            if epoch:
                self._log_events()
            self._tick_times.append(time.perf_counter())
        self._tick_times = self._tick_times[-60:]
        if len(self._tick_times) > 1:
            dt = self._tick_times[-1] - self._tick_times[0]
            self.tps = (len(self._tick_times) - 1) / dt if dt > 0 else 0.0

    def advance(self) -> None:
        if self.paused:
            return
        self._acc += self.tpf
        n = int(self._acc)
        self._acc -= n
        self.step(n)

    def _log_events(self) -> None:
        s, w = self.r.selected, self.world
        if s < 0:
            return
        t = w.tick_count
        if bool(w.last_heat_hit[s]):
            self.history.append(f"t={t}  heat damage")
        rg, dg = w.last_hgt
        for d in dg[rg == s].tolist():
            self.history.append(f"t={t}  received splice from #{d}")
        for r in rg[dg == s].tolist():
            self.history.append(f"t={t}  donated a splice to #{r}")
        self.history = self.history[-20:]

    # -------------------------------------------------------------- interaction
    def select_at(self, sx: float, sy: float, vw: int, vh: int) -> None:
        """Select the organism under the cursor, or the nearest one within 3 cells."""
        x, y = self.cam.to_cell(sx, sy, vw, vh)
        w = self.world
        cx, cy = int(np.floor(x)), int(np.floor(y))
        best, bd = -1, 1e9
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                g = int(w.gid[(cy + dy) % w.H, (cx + dx) % w.W])
                d = dx * dx + dy * dy
                if g >= 0 and d < bd:
                    best, bd = g, d
        self.select(best)

    def select(self, g: int) -> None:
        w = self.world
        self.r.select(g)
        self.history = []
        if g >= 0:
            p = int(w.parent[g])
            born = int(w.birth_tick[g])
            self.history.append(f"t={born}  born" + (f": spore of #{p}" if p >= 0 else
                                                     " (founder)"))
            if bool(w.has_hgt[g]):
                self.history.append("        carries spliced material")

    def toggle_feature(self, name: str) -> None:
        f = self.world.cfg.features
        f[name] = not f[name]

    # -------------------------------------------------------------- overlay
    def overlay(self, size: tuple[int, int], force: bool = False) -> np.ndarray | None:
        if not self.show_hud:
            return None
        now = time.perf_counter()
        if not force and now - self._hud_t < 0.25:
            return None
        self._hud_t = now
        if force or now - self.stats.get("_t", 0) > 1.0:
            self.stats = snapshot(self.world)
            self.stats["_t"] = now
        st, w = self.stats, self.world
        feats = " ".join(k for k in ("heat", "cost", "hgt") if w.cfg.features[k]) or "none"
        hud = [("tick", f"{w.tick_count:,}"),
               ("cells", f"{st.get('cells', 0):,}"),
               ("organisms", f"{st.get('organisms', 0):,}"),
               ("lineages", f"{st.get('lineages', 0):,}"),
               ("diversity", f"{st.get('shannon', 0):.2f}"),
               ("mean length", f"{st.get('mean_len', 0):.1f}"),
               ("zoom", f"{self.cam.zoom / self.ss:.1f}x"),
               ("ticks/s", f"{self.tps:.1f}" + ("  (paused)" if self.paused else "")),
               ("ticks/frame", f"{self.tpf:g}"),
               ("fps", f"{self.fps:.0f}" if self.fps > 0 else "-"),
               ("view", f"{self.mode} {MODES[self.mode]}"
                + ("  (click an organism)" if self.mode == 4 and self.r.selected < 0 else "")),
               ("features", feats)]
        insp = (organism_info(w, self.r.selected, self.history)
                if self.r.selected >= 0 else None)
        return compose(size, hud, HINT, insp, w.cfg.alphabet)

    def render(self, fbo, size, cursor=None, t: float = 0.0, force_overlay=False) -> None:
        self.r.upload()
        ov = self.overlay(size, force_overlay)
        if ov is not None:
            self.r.set_overlay(ov)
        elif not self.show_hud:
            self.r.set_overlay(None)
        self.r.draw(fbo, self.cam, self.mode, cursor, t)


# ---------------------------------------------------------------------- window
def run_window(cfg, size: tuple[int, int], world: World | None = None) -> None:
    import glfw
    import moderngl

    if not glfw.init():
        raise SystemExit("glfw.init() failed (no display?) - try --headless or --video")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(size[0], size[1], "Ribosome Soup", None, None)
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    ctx = moderngl.create_context()
    fb_size = glfw.get_framebuffer_size(win)
    s = Session(cfg, ctx, fb_size, world)
    scale = fb_size[0] / size[0]          # HiDPI: framebuffer px per window px
    drag = {"down": False, "moved": False, "x": 0.0, "y": 0.0}
    cursor = [None]

    def on_scroll(_w, _dx, dy):
        x, y = glfw.get_cursor_pos(win)
        fw, fh = glfw.get_framebuffer_size(win)
        s.cam.zoom_at(1.15 ** dy, x * scale, y * scale, fw, fh)

    def on_button(_w, button, action, _mods):
        if button != glfw.MOUSE_BUTTON_LEFT:
            return
        x, y = glfw.get_cursor_pos(win)
        if action == glfw.PRESS:
            drag.update(down=True, moved=False, x=x, y=y)
        elif action == glfw.RELEASE:
            if not drag["moved"]:
                fw, fh = glfw.get_framebuffer_size(win)
                s.select_at(x * scale, y * scale, fw, fh)
                s._hud_t = 0
            drag["down"] = False

    def on_move(_w, x, y):
        fw, fh = glfw.get_framebuffer_size(win)
        if drag["down"]:
            dx, dy = x - drag["x"], y - drag["y"]
            if abs(dx) + abs(dy) > 2:
                drag["moved"] = True
            if drag["moved"]:
                s.cam.cx -= dx * scale / s.cam.zoom
                s.cam.cy -= dy * scale / s.cam.zoom
                drag["x"], drag["y"] = x, y
        cursor[0] = s.cam.to_cell(x * scale, y * scale, fw, fh)

    def on_key(_w, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        fw, fh = glfw.get_framebuffer_size(win)
        if key == glfw.KEY_SPACE:
            s.paused = not s.paused
        elif key == glfw.KEY_PERIOD and s.paused:
            s.step(1)
        elif key == glfw.KEY_LEFT_BRACKET:
            s.tpf = max(0.0625, s.tpf / 2)
        elif key == glfw.KEY_RIGHT_BRACKET:
            s.tpf = min(64.0, s.tpf * 2)
        elif glfw.KEY_1 <= key <= glfw.KEY_6:
            s.mode = key - glfw.KEY_0
        elif key == glfw.KEY_H:
            s.toggle_feature("heat")
        elif key == glfw.KEY_C:
            s.toggle_feature("cost")
        elif key == glfw.KEY_G:
            s.toggle_feature("hgt")
        elif key == glfw.KEY_F:
            s.cam.fit(s.world.W, s.world.H, fw, fh)
        elif key == glfw.KEY_B:
            s.r.bloom = not s.r.bloom
        elif key == glfw.KEY_O:
            s.show_hud = not s.show_hud
        elif key == glfw.KEY_ESCAPE:
            s.select(-1)
        elif key == glfw.KEY_K:
            path = checkpoint.save(s.world, f"checkpoints/soup_{s.world.tick_count:08d}.pt")
            print("saved", path)
        elif key == glfw.KEY_S:
            os.makedirs("snapshots", exist_ok=True)
            img = np.frombuffer(ctx.screen.read(components=3), np.uint8).reshape(fh, fw, 3)
            path = f"snapshots/soup_{s.world.tick_count:07d}.png"
            Image.fromarray(img[::-1]).save(path)
            print("saved", path)
        s._hud_t = 0

    glfw.set_scroll_callback(win, on_scroll)
    glfw.set_mouse_button_callback(win, on_button)
    glfw.set_cursor_pos_callback(win, on_move)
    glfw.set_key_callback(win, on_key)

    t0 = last = time.perf_counter()
    while not glfw.window_should_close(win):
        glfw.poll_events()
        s.advance()
        now = time.perf_counter()
        s.fps = 0.9 * s.fps + 0.1 / max(now - last, 1e-6)
        last = now
        s.render(ctx.screen, glfw.get_framebuffer_size(win), cursor[0], now - t0)
        glfw.swap_buffers(win)
    glfw.terminate()


# ---------------------------------------------------------------------- headless
def run_headless(cfg, size, frames: int, tpf: float, out_dir: str | None,
                 video: str | None, fps: int, mode: int, zoom: float | None,
                 warmup: int, hud: bool, world: World | None = None,
                 supersample: int = 1) -> None:
    """Render frames offscreen. With supersample=N the scene is drawn at N x the size
    and box-filtered down (smoother edges/membranes); the HUD is composited at 1x."""
    import moderngl
    try:
        ctx = moderngl.create_standalone_context(backend="egl")     # Linux, no display
    except Exception:
        ctx = moderngl.create_standalone_context()                   # Windows / macOS
    ss = max(1, supersample)
    big = (size[0] * ss, size[1] * ss)
    fbo = ctx.simple_framebuffer(big)
    s = Session(cfg, ctx, size, world)
    s.cam.fit(s.world.W, s.world.H, *big)
    s.mode, s.tpf, s.show_hud, s.ss = mode, tpf, hud and ss == 1, ss
    if zoom:
        s.cam.zoom = zoom * ss
    s.step(warmup)
    proc = None
    if video:
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found")
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-vf", "vflip",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", video],
            stdin=subprocess.PIPE)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for i in range(frames):
        s.advance()
        s.render(fbo, big, None, i / fps, force_overlay=hud and ss == 1)
        raw = fbo.read(components=3)
        if ss > 1:
            img = np.frombuffer(raw, np.uint8).reshape(big[1], big[0], 3).astype(np.float32)
            img = img.reshape(size[1], ss, size[0], ss, 3).mean((1, 3))
            if hud:                         # HUD at 1x, composited after downsampling
                s.show_hud = True
                ov = s.overlay(size, force=True)
                s.show_hud = False
                a = ov[::-1, :, 3:4].astype(np.float32) / 255
                img = img * (1 - a) + ov[::-1, :, :3].astype(np.float32) * a
            raw = np.ascontiguousarray(img.round().astype(np.uint8)).tobytes()
        if proc:
            proc.stdin.write(raw)
        if out_dir:
            img = np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3)[::-1]
            Image.fromarray(img).save(os.path.join(out_dir, f"frame_{i:05d}.png"))
        if i % 50 == 0:
            print(f"frame {i}/{frames}  tick {s.world.tick_count}", flush=True)
    if proc:
        proc.stdin.close()
        proc.wait()
        print("wrote", video)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/tier_s.yaml")
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--device")
    ap.add_argument("--headless", metavar="DIR", help="write PNG frames instead of a window")
    ap.add_argument("--video", metavar="MP4", help="render an H.264 video headlessly")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--ticks-per-frame", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--zoom", type=float)
    ap.add_argument("--warmup", type=int, default=0, help="ticks to run before the first frame")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--load", help="start from a checkpoint (its config is used)")
    ap.add_argument("--supersample", type=int, default=1, help="headless/video: render at Nx")
    a = ap.parse_args(argv)
    over = {}
    if a.seed is not None:
        over["seed"] = a.seed
    if a.device:
        over["device"] = a.device
    torch.set_grad_enabled(False)
    world = None
    if a.load:
        world = checkpoint.load(a.load, device=a.device)
        cfg = world.cfg
        print(f"loaded {a.load} at tick {world.tick_count}")
    else:
        cfg = load_config(a.config, over)
    size = tuple(int(v) for v in a.size.split("x"))
    if a.headless or a.video:
        run_headless(cfg, size, a.frames, a.ticks_per_frame, a.headless, a.video, a.fps,
                     a.mode, a.zoom, a.warmup, not a.no_hud, world, a.supersample)
    else:
        run_window(cfg, size, world)


if __name__ == "__main__":
    main()
