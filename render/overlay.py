"""HUD and inspector drawn with Pillow into an RGBA image, uploaded as a texture.
Redrawn a few times per second, not every frame."""
from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from soup.ribosome import TRAITS

from .snapshot import oklch_to_srgb

INK = (240, 239, 236, 255)
INK2 = (195, 194, 183, 255)
MUTED = (137, 135, 129, 255)
PANEL = (16, 16, 18, 200)
BORDER = (255, 255, 255, 26)
GOLD = (244, 196, 92, 255)
PAD_TILE = (58, 58, 56, 255)

FONT_PATHS = ["/usr/share/fonts/opentype/inter/Inter-Regular.otf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
              "C:/Windows/Fonts/arial.ttf"]
MONO_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/System/Library/Fonts/Menlo.ttc", "C:/Windows/Fonts/consola.ttf"]


@lru_cache(maxsize=None)
def font(size: int, mono: bool = False) -> ImageFont.ImageFont:
    for p in (MONO_PATHS if mono else FONT_PATHS):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=None)
def token_colours(alphabet: int) -> list[tuple[int, int, int, int]]:
    """Token 0 = pad (grey). Tokens 1.. spaced around the OKLCH hue circle; each tile
    also shows its hex glyph, so identity never relies on colour alone."""
    n = alphabet - 1
    h = torch.tensor([2 * math.pi * i / n for i in range(n)])
    L = torch.where(torch.arange(n) % 2 == 0, torch.tensor(0.72), torch.tensor(0.6))
    rgb = (oklch_to_srgb(L, torch.full((n,), 0.13), h) * 255).round().int().tolist()
    return [PAD_TILE] + [(r, g, b, 255) for r, g, b in rgb]


def _panel(d: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    d.rounded_rectangle(box, radius=10, fill=PANEL, outline=BORDER, width=1)


def draw_hud(img: Image.Image, lines: list[tuple[str, str]], hint: str) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    f, fm = font(14), font(14, mono=True)
    lw = max(d.textlength(k, font=f) for k, _ in lines) + 14
    vw = max(d.textlength(v, font=fm) for _, v in lines)
    w, h = int(lw + vw + 28), 14 + 21 * len(lines)
    _panel(d, (12, 12, 12 + w, 12 + h))
    for i, (k, v) in enumerate(lines):
        y = 20 + 21 * i
        d.text((26, y), k, font=f, fill=MUTED)
        d.text((26 + lw, y), v, font=fm, fill=INK)
    fs = font(12)
    tw = d.textlength(hint, font=fs)
    x, y = img.width - tw - 22, img.height - 32
    _panel(d, (x - 8, y - 4, img.width - 12, img.height - 10))
    d.text((x, y), hint, font=fs, fill=INK2)


def draw_inspector(img: Image.Image, info: dict, alphabet: int) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    W = 372
    x0 = img.width - W - 12
    y = 12
    tile, gap, cols = 20, 2, 16
    rows = math.ceil(len(info["genome"]) / cols)
    hist = info.get("history", [])[-6:]
    h = 172 + rows * (tile + gap) + 4 * 24 + 26 + 18 * max(1, len(hist))
    _panel(d, (x0, y, x0 + W, y + h))
    x = x0 + 16
    y += 12
    status = "" if info["alive"] else "  (dead)"
    d.text((x, y), f"Organism #{info['slot']}{status}", font=font(18), fill=INK)
    y += 28
    meta = [("lineage", f"{info['root']}"), ("parent", f"{info['parent']}"),
            ("depth", f"{info['depth']}"), ("age", f"{info['age']} ticks"),
            ("cells", f"{info['cells']}"), ("energy", f"{info['energy']:.2f}"),
            ("length", f"{info['length']} tokens")]
    for i, (k, v) in enumerate(meta):
        cx = x + (i % 2) * 172
        cy = y + (i // 2) * 19
        d.text((cx, cy), k, font=font(13), fill=MUTED)
        d.text((cx + 58, cy), v, font=font(13, mono=True), fill=INK)
    y += 19 * math.ceil(len(meta) / 2) + 10

    d.text((x, y), "genome", font=font(13), fill=INK2)
    leg = "gold outline = spliced in"
    d.text((x0 + W - 16 - d.textlength(leg, font=font(11)), y + 2), leg, font=font(11),
           fill=MUTED)
    y += 20
    cols_rgb = token_colours(alphabet)
    fm = font(11, mono=True)
    for i, tok in enumerate(info["genome"]):
        tx = x + (i % cols) * (tile + gap)
        ty = y + (i // cols) * (tile + gap)
        d.rounded_rectangle((tx, ty, tx + tile, ty + tile), radius=3, fill=cols_rgb[tok])
        if tok:
            ch = f"{tok:x}"
            d.text((tx + tile / 2 - d.textlength(ch, font=fm) / 2, ty + 3), ch, font=fm,
                   fill=(11, 11, 11, 220))
        if info["spliced"][i]:
            d.rounded_rectangle((tx - 1, ty - 1, tx + tile + 1, ty + tile + 1), radius=4,
                                outline=GOLD, width=2)
    y += rows * (tile + gap) + 12

    for name, v in zip(TRAITS, info["traits"]):
        d.text((x, y), name.replace("_", " "), font=font(13), fill=MUTED)
        bx = x + 120
        d.rounded_rectangle((bx, y + 5, bx + 180, y + 13), radius=4, fill=(255, 255, 255, 30))
        d.rounded_rectangle((bx, y + 5, bx + max(8, int(180 * v)), y + 13), radius=4,
                            fill=(122, 167, 236, 255))
        d.text((bx + 190, y), f"{v:.2f}", font=font(13, mono=True), fill=INK)
        y += 24
    y += 4
    d.text((x, y), "history", font=font(13), fill=INK2)
    y += 20
    for line in hist or ["(nothing yet)"]:
        d.text((x, y), line, font=font(12), fill=INK2)
        y += 18


def compose(size: tuple[int, int], hud: list[tuple[str, str]], hint: str,
            inspector: dict | None, alphabet: int) -> np.ndarray:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw_hud(img, hud, hint)
    if inspector is not None:
        draw_inspector(img, inspector, alphabet)
    return np.asarray(img)


def organism_info(world, slot: int, history: list[str]) -> dict:
    w = world
    alive_mask = w.gid == slot
    cells = int(alive_mask.sum())
    return {
        "slot": slot, "alive": cells > 0 and bool(w.slot_alive[slot]),
        "root": int(w.root[slot]), "parent": int(w.parent[slot]),
        "depth": int(w.depth[slot]), "age": w.tick_count - int(w.birth_tick[slot]),
        "cells": cells, "energy": float(w.energy[alive_mask].mean()) if cells else 0.0,
        "length": int(w.glen[slot]), "genome": w.genomes[slot].cpu().tolist(),
        "spliced": w.spliced[slot].cpu().tolist(), "traits": w.traits[slot].cpu().tolist(),
        "history": history,
    }
