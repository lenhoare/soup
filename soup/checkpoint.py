"""Full-state checkpoints (spec 9.5): resume a run, or re-render an interesting
moment at high quality, without needing bit-exact replay from tick 0.

A checkpoint holds the config, every world tensor, counters and RNG states, so a
run resumed on the same device continues exactly as the original would have.
The ribosome itself is not stored: it is a pure function of the config's seed.
"""
from __future__ import annotations

import json
import os

import torch

from .config import Config, load_config
from .world import World

# World attributes saved verbatim (tensors)
TENSORS = ["genomes", "slot_alive", "dirty", "weights", "traits", "embed", "glen", "expressed",
           "parent", "birth_tick", "depth", "root", "has_hgt", "trait_logit", "spliced",
           "gid", "state", "morph", "energy", "light", "heat"]
VERSION = 1


def save(world: World, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "version": VERSION,
        "config": json.loads(json.dumps(world.cfg)),   # plain dicts
        "tick": world.tick_count,
        "next_root": world.next_root,
        "counters": dict(world.counters),
        "tensors": {k: getattr(world, k).cpu() for k in TENSORS if k != "weights"},
        # cell-net weights only for live slots (dead slots are never read)
        "live_slots": world.slot_alive.nonzero().flatten().cpu(),
        "live_weights": world.weights[world.slot_alive].cpu(),
        "barrier": None if world.barrier is None else world.barrier.cpu(),
        "rng": {"gen": world.gen.get_state(), "cpu_gen": world.cpu_gen.get_state(),
                "device": str(world.device)},
    }
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)          # never leave a half-written checkpoint
    return path


def load(path: str, device: str | None = None) -> World:
    """Rebuild a World from a checkpoint. `device` overrides the saved config's device
    (RNG state is only restored when the device type matches)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("version") != VERSION:
        raise ValueError(f"unsupported checkpoint version {data.get('version')}")
    over = {"device": device} if device else {}
    cfg: Config = load_config(None, {**data["config"], **over})
    w = World(cfg)
    for k, v in data["tensors"].items():
        cur = getattr(w, k)
        setattr(w, k, v.to(device=w.device, dtype=cur.dtype))
    w.weights[data["live_slots"].to(w.device)] = data["live_weights"].to(w.device, w.weights.dtype)
    w.barrier = None if data["barrier"] is None else data["barrier"].to(w.device)
    w.tick_count = data["tick"]
    w.next_root = data["next_root"]
    w.counters.update(data["counters"])
    if torch.device(data["rng"]["device"]).type == w.device.type:
        w.gen.set_state(data["rng"]["gen"])
    w.cpu_gen.set_state(data["rng"]["cpu_gen"])
    return w


def size_mb(path: str) -> float:
    return os.path.getsize(path) / 2 ** 20
