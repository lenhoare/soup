"""Still-image renderer (PNG) for headless runs. A first taste of the spec's visual
language: lineage hue from the genome embedding (OKLCH), lightness from energy,
bright membranes between organisms, and heat speckle in empty space.
The interactive moderngl viewer is milestone M4."""
from __future__ import annotations

import math
import struct
import zlib

import numpy as np
import torch


def oklch_to_srgb(L: torch.Tensor, C: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    a, b = C * torch.cos(h), C * torch.sin(h)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    rgb = torch.stack([r, g, bl], -1).clamp(0, 1)
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.pow(1 / 2.4) - 0.055)


@torch.no_grad()
def render(world, seed: int = 7) -> np.ndarray:
    """(H, W, 3) uint8 image of the world."""
    gid, energy, heat = world.gid.cpu(), world.energy.cpu(), world.heat.cpu()
    alive = gid >= 0
    g = gid.clamp(min=0)
    proj = torch.randn(world.embed.shape[1], 2, generator=torch.Generator().manual_seed(seed))
    v = world.embed.cpu() @ proj
    hue = torch.atan2(v[:, 1], v[:, 0])[g]
    chroma = (0.08 + 0.10 * torch.tanh(v.norm(dim=1) / 2))[g]
    light = 0.22 + 0.42 * energy
    # interior texture from hidden state channel 0
    light = light + 0.05 * world.state[0].cpu()
    membrane = torch.zeros_like(alive)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        membrane |= torch.roll(gid, (dy, dx), (0, 1)) != gid
    membrane &= alive
    light = torch.where(membrane, (light + 0.25).clamp(max=0.92), light)
    chroma = torch.where(membrane, chroma * 0.7, chroma)
    # empty space: near-black with ember speckle ~ heat
    rng = torch.Generator().manual_seed(world.tick_count)
    ember = (~alive) & (torch.rand(gid.shape, generator=rng) < 0.04 * heat ** 2)
    L = torch.where(alive, light, torch.where(ember, 0.35 + 0.2 * heat, torch.full_like(light, 0.12)))
    C = torch.where(alive, chroma, torch.where(ember, torch.full_like(light, 0.12),
                                               torch.full_like(light, 0.01)))
    H = torch.where(alive, hue, torch.full_like(light, math.radians(50)))
    rgb = oklch_to_srgb(L, C, H)
    return (rgb * 255 + 0.5).to(torch.uint8).numpy()


def write_png(path: str, rgb: np.ndarray, scale: int = 1) -> None:
    if scale > 1:
        rgb = rgb.repeat(scale, 0).repeat(scale, 1)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
