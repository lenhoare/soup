"""Configuration: defaults deep-merged with a YAML preset and optional overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import yaml

DEFAULTS: dict[str, Any] = {
    "grid": [256, 256],
    "genome_slots": 8192,
    "genome_len": 64,
    "alphabet": 16,
    "seed": 1,
    "device": "auto",          # auto | cpu | cuda
    "dtype": "float32",        # cached cell-net weights; float16 is fine on CUDA
    "chunk_cells": 32768,      # cells per gathered-weight batch (bounds memory)
    "epoch_ticks": 32,
    "cell_net": {"hidden": 16, "state_ch": 8, "morph_ch": 4},
    "ribosome": {
        "layers": 2, "d_model": 64, "heads": 4, "mlp_ratio": 2,
        "alpha": 6.0,          # W = W_base + alpha * dW(genome). M2: below ~4 the genome
                               # barely affects fitness; 6 -> 54% viable, smoothness ~12%
        "delta_scale": 1.0,    # scale of dW relative to a unit-variance init
        "frozen": True,
        "seed": 1234,          # fixes the "genetic code" in Phase 1
    },
    "init": {"founders": 256, "genome_len": 32, "energy": 0.5},
    "mutation": {"p_sub": 0.01, "p_indel": 0.003, "p_dup": 0.002, "p_del": 0.002},
    "heat": {"base_flip": 2.0e-4},
    "hgt": {"samples_per_epoch": 4096, "p_hgt": 0.05, "window": [2, 8]},
    "energy": {
        "photo": 0.12,          # income = photo * light * (1 - crowding * occupied_frac)
        "crowding": 0.9,        # M2: 0.5 saturates the grid (~98% full, no turnover)
        "basal": 0.002,
        "c_tok": 0.0004,
        "c_fid": 0.001,
        "grow_threshold": 0.4,
        "grow_cost": 0.02,
        "spore_cost": 0.05,
        "spore_energy": 0.2,
        "spore_min": 0.1,       # spore threshold = spore_min + spore_span * trait
        "spore_span": 0.4,      # M2: mean energy settles ~0.3, so thresholds must sit below it
        "attack_cost": 0.02,
        "loot": 0.5,
    },
    "dynamics": {
        "state_rate": 0.1, "state_leak": 0.02,
        "emit_rate": 0.05, "morph_diffuse": 0.2, "morph_decay": 0.02,
        "temperature": 1.0,
    },
    "fields": {
        "modes": 6, "max_wavenumber": 3, "update_every": 8,
        "light_drift": 0.002, "heat_drift": 0.001, "light_contrast": 6.0,
    },
    "features": {"heat": True, "cost": True, "hgt": True},
}


class Config(dict):
    """dict with attribute access (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Config({k: Config.wrap(v) for k, v in obj.items()})
        return obj


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    cfg = DEFAULTS
    if path is not None:
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return Config.wrap(cfg)


def resolve_device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg.device)


def resolve_dtype(cfg: Config) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.dtype]
